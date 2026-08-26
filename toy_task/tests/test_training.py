"""Training protocol: phase persistence, extraction schedule, task-routed freezing."""

import copy

import numpy as np
import torch

from toy_task.config import (
    RunConfig,
    CONSTANTS,
    ARCH_DENSE,
    ARCH_MODULAR_SHARED,
    ARCH_MODULAR_TASK_ROUTED,
    OFF_FREEZE,
    OFF_READOUT_COORD,
)
from toy_task.environment import Environment
from toy_task.models import build_model
from toy_task.init_scale import apply_init_scale
from toy_task.training import run_experiment, evaluate_clean


def test_run_smoke_dense():
    cfg = RunConfig(arch=ARCH_DENSE, hidden_size=24, gamma=1.0, similarity=0.0,
                    seed=0, epochs_per_phase=5)
    res = run_experiment(cfg)
    assert set(res.learning_curves) == {"A1", "B", "A2"}
    assert len(res.learning_curves["A1"]) == 5
    # A1 should reduce loss from first to last epoch on T(0).
    assert res.learning_curves["A1"][-1] < res.learning_curves["A1"][0]
    assert "forward_transfer_Ts" in res.behavioral
    assert "forward_transfer_Ts_task0" in res.behavioral
    assert "interference_T0" in res.behavioral


def test_diagnostic_probes_modular():
    cfg = RunConfig(arch=ARCH_MODULAR_SHARED, hidden_size=24, gamma=1.0, similarity=0.5,
                    seed=3, epochs_per_phase=5)
    res = run_experiment(cfg)
    assert "probe_module0_rule_mse" in res.behavioral
    assert "probe_module0_forward_mse" in res.behavioral
    assert "probe_module1_forward_mse" in res.behavioral


def test_extraction_schedule_count():
    """Per phase: start + every 10 + final. With 20 epochs: epochs {0,10,20} = 3 each."""
    cfg = RunConfig(arch=ARCH_DENSE, hidden_size=12, gamma=1.0, similarity=0.5,
                    seed=1, epochs_per_phase=20)
    res = run_experiment(cfg)
    per_phase = {}
    for e in res.extractions:
        per_phase.setdefault(e["phase"], []).append(e["epoch"])
    for phase in ("A1", "B", "A2"):
        assert per_phase[phase] == [0, 10, 20]
    # Hidden matrices are (N, H).
    assert res.extractions[0]["hidden"].shape == (CONSTANTS.n_objects, 12)


def test_run_is_reproducible():
    cfg = RunConfig(arch=ARCH_DENSE, hidden_size=24, gamma=1.0, similarity=0.3,
                    seed=2, epochs_per_phase=5)
    r1 = run_experiment(cfg)
    r2 = run_experiment(cfg)
    assert np.allclose(r1.learning_curves["A1"], r2.learning_curves["A1"])
    assert np.isclose(r1.behavioral["interference_T0"], r2.behavioral["interference_T0"])


def test_task_routed_freeze_protects_module0_during_B():
    """In freeze mode module 0 must be bit-identical before/after Phase B."""
    cfg = RunConfig(arch=ARCH_MODULAR_TASK_ROUTED, hidden_size=24, gamma=1.0,
                    similarity=1.0, seed=0, off_module_policy=OFF_FREEZE,
                    epochs_per_phase=10)
    # Reproduce the run's phase structure manually to snapshot module 0 weights.
    env = Environment.from_seed(cfg.seed)
    model = build_model(cfg.arch, cfg.hidden_size, cfg.off_module_policy, seed=cfg.seed)
    apply_init_scale(model, cfg.gamma)
    opt = torch.optim.SGD(model.parameters(), lr=CONSTANTS.lr)
    loss_fn = torch.nn.MSELoss()

    def train_phase(sim, task_id):
        rng = np.random.default_rng(99)
        for _ in range(5):
            x_all = env.noisy_observation(rng)
            y_all = env.targets(sim)
            for idx in env.epoch_order(rng):
                opt.zero_grad()
                out, _ = model(torch.as_tensor(x_all[idx:idx+1], dtype=torch.float32), task_id=task_id)
                loss = loss_fn(out, torch.as_tensor(y_all[idx:idx+1], dtype=torch.float32))
                loss.backward()
                opt.step()

    train_phase(0.0, 0)  # A1
    w0_before = model.weight_ih[model.module_slice(0)].detach().clone()
    train_phase(cfg.similarity, 1)  # B
    w0_after = model.weight_ih[model.module_slice(0)].detach().clone()
    assert torch.allclose(w0_before, w0_after), "module 0 must be frozen during B (freeze)"


def test_task_routed_readout_coord_changes_both_in_B():
    cfg = RunConfig(arch=ARCH_MODULAR_TASK_ROUTED, hidden_size=24, gamma=1.0,
                    similarity=1.0, seed=0, off_module_policy=OFF_READOUT_COORD,
                    epochs_per_phase=10)
    env = Environment.from_seed(cfg.seed)
    model = build_model(cfg.arch, cfg.hidden_size, cfg.off_module_policy, seed=cfg.seed)
    opt = torch.optim.SGD(model.parameters(), lr=CONSTANTS.lr)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(7)

    # Train A1 a bit so module 0 is engaged, then snapshot before B.
    for _ in range(5):
        x_all = env.noisy_observation(rng); y_all = env.targets(0.0)
        for idx in env.epoch_order(rng):
            opt.zero_grad()
            out, _ = model(torch.as_tensor(x_all[idx:idx+1], dtype=torch.float32), task_id=0)
            loss_fn(out, torch.as_tensor(y_all[idx:idx+1], dtype=torch.float32)).backward()
            opt.step()
    w0_before = model.weight_ih[model.module_slice(0)].detach().clone()
    head0_before = model.readout[0].parametrizations.weight.original.detach().clone()

    # Phase B uses task head 1; module 0 should still update (gradient through both).
    for _ in range(5):
        x_all = env.noisy_observation(rng); y_all = env.targets(cfg.similarity)
        for idx in env.epoch_order(rng):
            opt.zero_grad()
            out, _ = model(torch.as_tensor(x_all[idx:idx+1], dtype=torch.float32), task_id=1)
            loss_fn(out, torch.as_tensor(y_all[idx:idx+1], dtype=torch.float32)).backward()
            opt.step()
    w0_after = model.weight_ih[model.module_slice(0)].detach().clone()
    head0_after = model.readout[0].parametrizations.weight.original.detach().clone()

    assert not torch.allclose(w0_before, w0_after), "module 0 should adapt during B (readout_coord)"
    assert torch.allclose(head0_before, head0_after), "unused head 0 must be frozen during B"
