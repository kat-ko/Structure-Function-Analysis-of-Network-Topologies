"""Diagnostic helpers for nb09 task-design experiments (angle layout vs rule learning).

Supports comparing ``angle_mode`` variants (random vs even) without touching the
overnight run cache.  Metrics focus on *verifiable* signals of whether the network
infers a global ``+s`` rule vs memorises per-object mappings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import lstsq

from .config import ANGLE_EVEN, ANGLE_RANDOM, CONSTANTS, RunConfig, STIMULUS_NOVEL
from .environment import Environment
from .init_scale import apply_init_scale
from .models import build_model, ToyRNN
from .training import evaluate_clean

_TASK0 = 0
_TASK1 = 1
Protocol = Literal["a1", "full"]


@dataclass
class DiagnosticMetrics:
    """Scalar outcomes for one diagnostic run."""

    angle_mode: str
    seed: int
    arch: str
    gamma: float
    similarity: float
    protocol: str
    min_sep_A_deg: float
    min_sep_B_deg: float
    # After requested training:
    a1_end_T0: float = float("nan")
    forward_transfer_Ts: float = float("nan")
    interference_T0: float = float("nan")
    b_end_Ts: float = float("nan")
    a2_end_T0: float = float("nan")
    # Hidden linear angle probe (degrees, mean circular error):
    hidden_err_A_deg: float = float("nan")
    hidden_err_B_deg: float = float("nan")
    # Oracle (linear x -> cos/sin theta) forward transfer MSE on B:
    oracle_forward_mse: float = float("nan")
    oracle_a_mse: float = float("nan")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnosticMetrics":
        fields = cls.__dataclass_fields__
        return cls(**{k: d[k] for k in fields if k in d})


# ------------------------------------------------------------------ run cache
def diagnostic_run_id(cfg: RunConfig, angle_mode: str, protocol: Protocol) -> str:
    """Filesystem-safe id for one nb09 diagnostic run (distinct from ``RunConfig.run_id``)."""
    return f"{cfg.run_id()}_ang{angle_mode}_p{protocol}"


def diagnostic_run_dir(out_root: str | Path, run_id: str) -> Path:
    p = Path(out_root) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_diagnostic_complete(out_root: str | Path, run_id: str) -> bool:
    """True when a prior run wrote ``metrics.json``."""
    return (Path(out_root) / run_id / "metrics.json").is_file()


def save_diagnostic(
    metrics: DiagnosticMetrics,
    out_root: str | Path,
    cfg: RunConfig,
    run_id: str | None = None,
) -> Path:
    """Persist scalar metrics + config under ``<out_root>/<run_id>/metrics.json``."""
    rid = run_id or diagnostic_run_id(cfg, metrics.angle_mode, metrics.protocol)  # type: ignore[arg-type]
    d = diagnostic_run_dir(out_root, rid)
    payload = {
        "run_id": rid,
        "config": cfg.to_dict(),
        "metrics": metrics.to_dict(),
    }
    with open(d / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    return d


def load_diagnostic(out_root: str | Path, run_id: str) -> DiagnosticMetrics:
    """Load a cached nb09 diagnostic run."""
    path = Path(out_root) / run_id / "metrics.json"
    with open(path) as f:
        payload = json.load(f)
    return DiagnosticMetrics.from_dict(payload["metrics"])


def load_diagnostic_meta(out_root: str | Path, run_id: str) -> dict:
    """Load full cached payload (config + metrics)."""
    path = Path(out_root) / run_id / "metrics.json"
    with open(path) as f:
        return json.load(f)


def build_env(
    seed: int,
    angle_mode: str = ANGLE_RANDOM,
    stimulus_regime: str = STIMULUS_NOVEL,
) -> Environment:
    return Environment.from_seed(
        seed, stimulus_regime=stimulus_regime, angle_mode=angle_mode
    )


def circular_error_deg(pred_angles: np.ndarray, true_angles: np.ndarray) -> float:
    diff = np.arctan2(
        np.sin(pred_angles - true_angles),
        np.cos(pred_angles - true_angles),
    )
    return float(np.degrees(np.abs(diff).mean()))


def hidden_angle_error(hidden: np.ndarray, theta: np.ndarray) -> float:
    """Mean circular error for the best linear readout hidden -> (cos theta, sin theta)."""
    coef, _, _, _ = lstsq(hidden, np.column_stack([np.cos(theta), np.sin(theta)]), rcond=None)
    pred = hidden @ coef
    pred_ang = np.arctan2(pred[:, 1], pred[:, 0])
    return circular_error_deg(pred_ang, theta)


def mse_cos_sin(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def oracle_linear_forward(
    env: Environment,
    similarity: float,
    *,
    ridge: float = 1e-6,
) -> tuple[float, float]:
    """Train linear map ``x_A -> y_A``; evaluate MSE on ``y_B`` with rule ``+s``.

  Returns ``(mse_A, mse_forward_B)``.
    """
    x_a = env.clean_observation("A")
    x_b = env.clean_observation("B")
    y_a = env.targets(0.0, "A")
    y_b = env.targets(similarity, "B")
  # Ridge via normal equations
    d = x_a.shape[1]
    xtx = x_a.T @ x_a + ridge * np.eye(d)
    coef = np.linalg.solve(xtx, x_a.T @ y_a)
    pred_a = x_a @ coef
    pred_b = x_b @ coef
    return mse_cos_sin(pred_a, y_a), mse_cos_sin(pred_b, y_b)


def _run_phase(
    model: ToyRNN,
    env: Environment,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    cfg: RunConfig,
    phase: str,
    similarity: float,
    task_id: int,
    rng: np.random.Generator,
) -> None:
    c = CONSTANTS
    object_set = "B" if phase == "B" else "A"
    for _ in range(cfg.epochs_per_phase):
        model.train()
        order = env.epoch_order(rng)
        x_all = env.noisy_observation(rng, object_set=object_set)
        y_all = env.targets(similarity, object_set=object_set)
        for idx in order:
            x = torch.as_tensor(x_all[idx : idx + 1], dtype=torch.float32)
            y = torch.as_tensor(y_all[idx : idx + 1], dtype=torch.float32)
            optimizer.zero_grad()
            out, _ = model(x, task_id=task_id)
            loss_fn(out, y).backward()
            optimizer.step()


def run_diagnostic(
    cfg: RunConfig,
    angle_mode: str,
    protocol: Protocol = "full",
) -> DiagnosticMetrics:
    """Train one diagnostic configuration and return scalar metrics."""
    env = build_env(cfg.seed, angle_mode=angle_mode, stimulus_regime=cfg.stimulus_regime)
    model = build_model(
        cfg.arch, cfg.hidden_size, cfg.off_module_policy,
        seed=cfg.seed, comms_bandwidth=getattr(cfg, "comms_bandwidth", 0.0),
    )
    apply_init_scale(model, cfg.gamma, scope=cfg.init_scope)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONSTANTS.lr,
        momentum=CONSTANTS.momentum,
        weight_decay=CONSTANTS.weight_decay,
    )
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(cfg.seed + 17_001)

    oa, ob = oracle_linear_forward(env, cfg.similarity)
    m = DiagnosticMetrics(
        angle_mode=angle_mode,
        seed=cfg.seed,
        arch=cfg.arch,
        gamma=cfg.gamma,
        similarity=cfg.similarity,
        protocol=protocol,
        min_sep_A_deg=env.min_pairwise_sep_deg("A"),
        min_sep_B_deg=env.min_pairwise_sep_deg("B"),
        oracle_a_mse=oa,
        oracle_forward_mse=ob,
    )

    _run_phase(model, env, optimizer, loss_fn, cfg, "A1", 0.0, _TASK0, rng)
    m.a1_end_T0 = evaluate_clean(model, env, 0.0, _TASK0, object_set="A")["loss"]
    m.forward_transfer_Ts = evaluate_clean(
        model, env, cfg.similarity, _TASK1, object_set="B"
    )["loss"]
    h_a = evaluate_clean(model, env, 0.0, _TASK0, object_set="A")["hidden"]
    m.hidden_err_A_deg = hidden_angle_error(h_a, env.base_angles("A"))

    if protocol == "a1":
        return m

    _run_phase(model, env, optimizer, loss_fn, cfg, "B", cfg.similarity, _TASK1, rng)
    m.interference_T0 = evaluate_clean(model, env, 0.0, _TASK0, object_set="A")["loss"]
    m.b_end_Ts = evaluate_clean(
        model, env, cfg.similarity, _TASK1, object_set="B"
    )["loss"]

    _run_phase(model, env, optimizer, loss_fn, cfg, "A2", 0.0, _TASK0, rng)
    m.a2_end_T0 = evaluate_clean(model, env, 0.0, _TASK0, object_set="A")["loss"]
    h_b = evaluate_clean(model, env, cfg.similarity, _TASK1, object_set="B")["hidden"]
    m.hidden_err_B_deg = hidden_angle_error(
        h_b, env.base_angles("B") + cfg.similarity
    )
    return m


def run_grid(
    configs: List[RunConfig],
    angle_modes: List[str],
    protocol: Protocol = "full",
    *,
    out_root: str | Path | None = None,
    force: bool = False,
    show_progress: bool = True,
) -> List[DiagnosticMetrics]:
    """Run a Cartesian grid of configs × angle_modes.

    When ``out_root`` is set, each run is written to
    ``<out_root>/<diagnostic_run_id>/metrics.json`` and skipped on re-run if
    complete (same resumable pattern as ``scripts/overnight_grid.py``).
    """
    from tqdm.auto import tqdm

    rows: List[DiagnosticMetrics] = []
    pairs = [(cfg, am) for cfg in configs for am in angle_modes]
    iterator = tqdm(pairs, desc=f"nb09 {protocol}") if show_progress else pairs
    n_skip = 0
    for cfg, am in iterator:
        rid = diagnostic_run_id(cfg, am, protocol)
        if out_root is not None and not force and is_diagnostic_complete(out_root, rid):
            rows.append(load_diagnostic(out_root, rid))
            n_skip += 1
            continue
        m = run_diagnostic(cfg, am, protocol=protocol)
        if out_root is not None:
            save_diagnostic(m, out_root, cfg, run_id=rid)
        rows.append(m)
    if out_root is not None and show_progress and n_skip:
        print(f"  loaded {n_skip}/{len(pairs)} from cache ({out_root})")
    return rows


def metrics_to_frame(rows: List[DiagnosticMetrics]):
    import pandas as pd

    return pd.DataFrame([r.to_dict() for r in rows])


def score_angle_mode_comparison(df, metric: str = "forward_transfer_Ts") -> dict:
    """Summarise random vs even for decision support."""
    import pandas as pd

    sub = df.groupby("angle_mode")[metric].agg(["mean", "std", "median"]).astype(float)
    if ANGLE_RANDOM in sub.index and ANGLE_EVEN in sub.index:
        delta = float(sub.loc[ANGLE_EVEN, "mean"] - sub.loc[ANGLE_RANDOM, "mean"])
    else:
        delta = float("nan")
    # Lower forward transfer loss is better (rule generalises from A to B).
    favor_even = delta < 0
    return {
        "metric": metric,
        "random_mean": float(sub.loc[ANGLE_RANDOM, "mean"]) if ANGLE_RANDOM in sub.index else float("nan"),
        "even_mean": float(sub.loc[ANGLE_EVEN, "mean"]) if ANGLE_EVEN in sub.index else float("nan"),
        "even_minus_random": delta,
        "favor_even_if_negative": favor_even,
        "summary_table": sub,
    }
