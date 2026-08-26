"""Model shape contracts and router correctness for the a1b2-aligned RNN."""

import torch

from toy_task.config import (
    CONSTANTS,
    ARCH_DENSE,
    ARCH_MODULAR_SHARED,
    ARCH_MODULAR_FEATURE_ROUTED,
    ARCH_MODULAR_TASK_ROUTED,
    OFF_FREEZE,
    OFF_READOUT_COORD,
)
from toy_task.models import build_model


H = 24
PER = H // CONSTANTS.n_modules


def _x(batch=5):
    return torch.randn(batch, CONSTANTS.d_obs)


def test_shape_contract_all_archs():
    for arch in (ARCH_DENSE, ARCH_MODULAR_SHARED, ARCH_MODULAR_FEATURE_ROUTED):
        m = build_model(arch, H, seed=0)
        out, hid = m(_x(), task_id=0)
        assert out.shape == (5, CONSTANTS.d_out)
        assert hid.shape == (5, H)
    for off in (OFF_FREEZE, OFF_READOUT_COORD):
        m = build_model(ARCH_MODULAR_TASK_ROUTED, H, off, seed=0)
        out, hid = m(_x(), task_id=1)
        assert out.shape == (5, CONSTANTS.d_out)
        assert hid.shape == (5, H)


def test_recurrent_weights_are_bias_free():
    for arch in (ARCH_DENSE, ARCH_MODULAR_SHARED):
        m = build_model(arch, H, seed=0)
        param_names = dict(m.core.named_parameters())
        assert not any("bias" in n for n in param_names)


def test_no_self_connections_in_recurrent_weight():
    """a1b2 parity: effective recurrent weight has a zero diagonal (1 - I per block)."""
    for arch in (ARCH_DENSE, ARCH_MODULAR_SHARED):
        m = build_model(arch, H, seed=0)
        whh = m.core.weight_hh_l0.detach()
        assert torch.allclose(torch.diag(whh), torch.zeros(H))


def test_no_inter_module_recurrence():
    """Cross-module recurrent blocks are masked to zero (no communication)."""
    m = build_model(ARCH_MODULAR_SHARED, H, seed=0)
    whh = m.core.weight_hh_l0.detach()
    s0, s1 = m.module_slice(0), m.module_slice(1)
    assert torch.allclose(whh[s0, s1], torch.zeros(PER, PER))
    assert torch.allclose(whh[s1, s0], torch.zeros(PER, PER))


def test_feature_routed_input_mask_restricts_to_slice():
    """Each module's effective input weights are nonzero only within its feature slice."""
    m = build_model(ARCH_MODULAR_FEATURE_ROUTED, H, seed=0)
    wih = m.core.weight_ih_l0.detach()  # (H, d_obs * n_modules)
    d = CONSTANTS.d_obs
    step = d // CONSTANTS.n_modules
    for mod in range(CONSTANTS.n_modules):
        rows = m.module_slice(mod)
        block = wih[rows, mod * d:(mod + 1) * d]  # this module's input copy
        keep = block[:, mod * step:(mod + 1) * step]
        # Everything outside the module's feature slice must be zero.
        total = block.abs().sum()
        kept = keep.abs().sum()
        assert torch.allclose(total, kept)


def test_task_routed_freeze_off_module_zero_hidden():
    m = build_model(ARCH_MODULAR_TASK_ROUTED, H, OFF_FREEZE, seed=0)
    _, hid0 = m(_x(), task_id=0)
    assert torch.allclose(hid0[:, PER:], torch.zeros_like(hid0[:, PER:]))
    _, hid1 = m(_x(), task_id=1)
    assert torch.allclose(hid1[:, :PER], torch.zeros_like(hid1[:, :PER]))


def test_task_routed_freeze_off_module_zero_gradient():
    m = build_model(ARCH_MODULAR_TASK_ROUTED, H, OFF_FREEZE, seed=0)
    out, _ = m(_x(), task_id=0)
    (out ** 2).mean().backward()
    g = m.weight_ih.grad
    assert g[m.module_slice(0)].abs().sum() > 0      # active module
    assert g[m.module_slice(1)].abs().sum() == 0     # frozen module


def test_task_routed_readout_coord_gradient_through_both():
    m = build_model(ARCH_MODULAR_TASK_ROUTED, H, OFF_READOUT_COORD, seed=0)
    out, _ = m(_x(), task_id=1)
    (out ** 2).mean().backward()
    g = m.weight_ih.grad
    assert g[m.module_slice(0)].abs().sum() > 0
    assert g[m.module_slice(1)].abs().sum() > 0


def test_readout_coord_has_one_head_per_task():
    m = build_model(ARCH_MODULAR_TASK_ROUTED, H, OFF_READOUT_COORD, seed=0)
    assert len(m.readout) == CONSTANTS.n_modules


def test_feature_routed_task_agnostic():
    """Feature-routed output does not depend on task_id."""
    m = build_model(ARCH_MODULAR_FEATURE_ROUTED, H, seed=0)
    x = _x()
    out0, _ = m(x, task_id=0)
    out1, _ = m(x, task_id=1)
    assert torch.allclose(out0, out1)


def test_readout_sums_per_module_outputs():
    """a1b2 readout: output equals the sum of the two per-module readout blocks."""
    m = build_model(ARCH_MODULAR_SHARED, H, seed=0)
    x = _x()
    out, hidden = m(x, task_id=0)
    full = m.readout(hidden)  # (batch, d_out * n_modules)
    manual = full.view(x.shape[0], CONSTANTS.n_modules, CONSTANTS.d_out).sum(dim=1)
    assert torch.allclose(out, manual, atol=1e-6)


def test_bandwidth_zero_matches_no_comms_pathway():
    m0 = build_model(ARCH_MODULAR_SHARED, H, seed=1, comms_bandwidth=0.0)
    assert m0.comms is None
    x = _x(3)
    out0, hid0 = m0(x, task_id=0)
    m1 = build_model(ARCH_MODULAR_SHARED, H, seed=1, comms_bandwidth=0.0)
    out1, hid1 = m1(x, task_id=0)
    assert torch.allclose(out0, out1)
    assert torch.allclose(hid0, hid1)


def test_bandwidth_enables_cross_module_recurrence():
    m = build_model(ARCH_MODULAR_SHARED, H, seed=0, comms_bandwidth=1.0)
    assert m.comms is not None
    whh = m.comms.weight_hh_l0.detach()
    s0, s1 = m.module_slice(0), m.module_slice(1)
    assert whh[s0, s1].abs().sum() > 0
    assert whh[s1, s0].abs().sum() > 0
    assert torch.allclose(whh[s0, s0], torch.zeros(PER, PER))


def test_bandwidth_gradient_flows_across_modules():
    m = build_model(ARCH_MODULAR_SHARED, H, seed=0, comms_bandwidth=0.5)
    out, _ = m(_x(), task_id=0)
    (out ** 2).mean().backward()
    g = m.comms.parametrizations.weight_hh_l0.original.grad
    s0, s1 = m.module_slice(0), m.module_slice(1)
    assert g[s0, s1].abs().sum() > 0
    assert g[s1, s0].abs().sum() > 0
