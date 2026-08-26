"""Continual-learning protocol: A1 -> B(s) -> A2 (Part III + patch Section 4-8).

Runs the three-phase curriculum with a single persistent optimizer (weights and
optimizer state are never reset between phases), measures forward transfer and
interference at the phase boundaries, and extracts clean-data hidden representations
on the schedule from patch Section 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .config import CONSTANTS, RunConfig
from .environment import Environment
from .init_scale import apply_init_scale
from .models import build_model, ToyRNN


# Phase plan: (name, similarity-source, task_id).
# A1 and A2 use the reference task T(0); B uses T(s).
_TASK0 = 0
_TASK1 = 1


@dataclass
class RunResult:
    config: dict
    learning_curves: Dict[str, List[float]] = field(default_factory=dict)
    behavioral: Dict[str, float] = field(default_factory=dict)
    extractions: List[dict] = field(default_factory=list)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32)


def _linear_probe_mse(hidden: np.ndarray, targets: np.ndarray,
                      coef: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """OLS linear probe with bias; return (MSE, coefficients)."""
    n = hidden.shape[0]
    design = np.hstack([hidden, np.ones((n, 1))])
    if coef is None:
        coef, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    pred = design @ coef
    mse = float(np.mean((pred - targets) ** 2))
    return mse, coef


@torch.no_grad()
def diagnostic_probes(model: ToyRNN, env: Environment, cfg: RunConfig) -> dict[str, float]:
    """Post-A1 probes that disambiguate routing shortcuts from rule transfer."""
    probes: dict[str, float] = {
        "forward_transfer_Ts_task0": evaluate_clean(
            model, env, cfg.similarity, _TASK0, object_set="B"
        )["loss"],
    }
    if model.n_modules <= 1:
        return probes

    rec_a = evaluate_clean(model, env, 0.0, _TASK0, object_set="A")
    rec_b = evaluate_clean(model, env, cfg.similarity, _TASK0, object_set="B")
    y_a = env.targets(0.0, "A")
    y_b = env.targets(cfg.similarity, "B")

    for m in range(model.n_modules):
        sl = model.module_slice(m)
        h_a = rec_a["hidden"][:, sl]
        h_b = rec_b["hidden"][:, sl]
        rule_mse, coef = _linear_probe_mse(h_a, y_a)
        forward_mse, _ = _linear_probe_mse(h_b, y_b, coef=coef)
        probes[f"probe_module{m}_rule_mse"] = rule_mse
        probes[f"probe_module{m}_forward_mse"] = forward_mse

    return probes


@torch.no_grad()
def evaluate_clean(model: ToyRNN, env: Environment, similarity: float, task_id: int,
                   object_set: str = "A") -> dict:
    """Evaluate MSE on noise-free observations for task ``T(similarity)``.

    ``object_set`` selects task A's objects (``"A"``) or task B's objects (``"B"``);
    they differ only in the ``novel`` stimulus regime. Returns the scalar loss plus
    per-object loss, hidden representations, predictions and targets (all numpy).
    """
    model.eval()
    x = _to_tensor(env.clean_observation(object_set))   # (N, d_x)
    y = _to_tensor(env.targets(similarity, object_set))  # (N, d_y)
    out, hidden = model(x, task_id=task_id)
    per_obj = ((out - y) ** 2).sum(dim=1)            # (N,)
    loss = per_obj.mean().item()
    return {
        "loss": float(loss),
        "per_object_loss": per_obj.cpu().numpy(),
        "hidden": hidden.cpu().numpy(),
        "preds": out.cpu().numpy(),
        "targets": y.cpu().numpy(),
    }


def _extract_record(model, env, cfg: RunConfig, phase: str, task_id: int,
                    epoch: int, global_epoch: int) -> dict:
    """Build one representation-extraction record at the current weights."""
    # Hidden reps under the phase's own task gate (matters for task_routed) and
    # the phase's own object set (matters for the novel stimulus regime).
    phase_sim = cfg.similarity if phase == "B" else 0.0
    phase_objs = "B" if phase == "B" else "A"
    rec = evaluate_clean(model, env, phase_sim, task_id, object_set=phase_objs)
    # Clean losses on both tasks (each on its own object set) for trajectory plots.
    loss_t0 = evaluate_clean(model, env, 0.0, _TASK0, object_set="A")["loss"]
    loss_ts = evaluate_clean(model, env, cfg.similarity, _TASK1, object_set="B")["loss"]
    return {
        "arch": cfg.arch,
        "off_module_policy": cfg.off_module_policy,
        "hidden_size": cfg.hidden_size,
        "gamma": cfg.gamma,
        "similarity": cfg.similarity,
        "seed": cfg.seed,
        "phase": phase,
        "task_id": task_id,
        "epoch": epoch,
        "global_epoch": global_epoch,
        "hidden": rec["hidden"],
        "preds": rec["preds"],
        "targets": rec["targets"],
        "clean_loss_phase": rec["loss"],
        "clean_loss_T0": loss_t0,
        "clean_loss_Ts": loss_ts,
    }


def _run_phase(model, env, optimizer, loss_fn, cfg, phase, similarity, task_id,
               global_epoch_start, extractions) -> tuple[list[float], int]:
    """Train one phase; return (per-epoch mean losses, next global epoch index)."""
    c = CONSTANTS
    phase_offset = {"A1": 1, "B": 2, "A2": 3}[phase]
    object_set = "B" if phase == "B" else "A"
    rng = np.random.default_rng((cfg.seed + 1) * 100003 + phase_offset)
    curve: list[float] = []

    # Extraction at the start of the phase (before any update in this phase).
    extractions.append(
        _extract_record(model, env, cfg, phase, task_id, epoch=0,
                        global_epoch=global_epoch_start)
    )

    n_epochs = cfg.epochs_per_phase
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = env.epoch_order(rng)
        x_all = env.noisy_observation(rng, object_set=object_set)  # (N, d_x), fresh noise
        y_all = env.targets(similarity, object_set=object_set)     # (N, d_y)
        epoch_losses = []
        for idx in order:
            x = _to_tensor(x_all[idx : idx + 1])     # (1, d_x)
            y = _to_tensor(y_all[idx : idx + 1])     # (1, d_y)
            optimizer.zero_grad()
            out, _ = model(x, task_id=task_id)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        curve.append(float(np.mean(epoch_losses)))

        global_epoch = global_epoch_start + epoch
        is_final = epoch == n_epochs
        if epoch % c.extract_every == 0 or is_final:
            extractions.append(
                _extract_record(model, env, cfg, phase, task_id, epoch=epoch,
                                global_epoch=global_epoch)
            )

    return curve, global_epoch_start + n_epochs


def run_experiment(cfg: RunConfig) -> RunResult:
    """Execute a full A1 -> B(s) -> A2 run for one configuration."""
    c = CONSTANTS
    env = Environment.from_seed(
        cfg.seed, stimulus_regime=cfg.stimulus_regime, angle_mode=cfg.angle_mode
    )
    model = build_model(
        cfg.arch, cfg.hidden_size, cfg.off_module_policy,
        seed=cfg.seed, comms_bandwidth=cfg.effective_comms_bandwidth,
    )
    apply_init_scale(model, cfg.gamma, scope=cfg.init_scope)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=c.lr, momentum=c.momentum, weight_decay=c.weight_decay
    )
    loss_fn = nn.MSELoss()

    result = RunResult(config=cfg.to_dict())
    extractions: list[dict] = []

    # ---- Phase A1 (T(0), task 0) ------------------------------------------
    curve_a1, g = _run_phase(model, env, optimizer, loss_fn, cfg, "A1", 0.0, _TASK0,
                             global_epoch_start=0, extractions=extractions)
    result.learning_curves["A1"] = curve_a1

    # Forward transfer: clean loss on T(s) (task B's objects) BEFORE any B update.
    result.behavioral["forward_transfer_Ts"] = evaluate_clean(
        model, env, cfg.similarity, _TASK1, object_set="B")["loss"]
    result.behavioral["a1_end_T0"] = evaluate_clean(model, env, 0.0, _TASK0, object_set="A")["loss"]
    result.behavioral.update(diagnostic_probes(model, env, cfg))

    # ---- Phase B (T(s), task 1) -------------------------------------------
    curve_b, g = _run_phase(model, env, optimizer, loss_fn, cfg, "B", cfg.similarity, _TASK1,
                            global_epoch_start=g, extractions=extractions)
    result.learning_curves["B"] = curve_b

    # Interference: clean loss on T(0) (task A's objects) BEFORE any A2 update.
    result.behavioral["interference_T0"] = evaluate_clean(model, env, 0.0, _TASK0, object_set="A")["loss"]
    result.behavioral["b_end_Ts"] = evaluate_clean(
        model, env, cfg.similarity, _TASK1, object_set="B")["loss"]

    # ---- Phase A2 (T(0), task 0) ------------------------------------------
    curve_a2, g = _run_phase(model, env, optimizer, loss_fn, cfg, "A2", 0.0, _TASK0,
                             global_epoch_start=g, extractions=extractions)
    result.learning_curves["A2"] = curve_a2
    result.behavioral["a2_end_T0"] = evaluate_clean(model, env, 0.0, _TASK0, object_set="A")["loss"]

    result.extractions = extractions
    return result
