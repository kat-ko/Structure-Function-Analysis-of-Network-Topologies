"""init_scale gamma scopes: all / no_readout / input_only."""

import torch

from toy_task.config import (
    ARCH_MODULAR_SHARED,
    INIT_SCOPE_ALL,
    INIT_SCOPE_NO_READOUT,
    INIT_SCOPE_INPUT_ONLY,
)
from toy_task.models import build_model
from toy_task.init_scale import apply_init_scale


H = 24
GAMMA = 2.0


def _snapshot(m):
    return {
        "wih": m.core.parametrizations.weight_ih_l0.original.detach().clone(),
        "whh": m.core.parametrizations.weight_hh_l0.original.detach().clone(),
        "rw": m.readout.parametrizations.weight.original.detach().clone(),
        "rb": m.readout.bias.detach().clone(),
    }


def _ratio(after, before):
    mask = before.abs() > 1e-8
    return (after[mask] / before[mask])


def test_scope_all_scales_everything():
    m = build_model(ARCH_MODULAR_SHARED, H, seed=0)
    b = _snapshot(m)
    apply_init_scale(m, GAMMA, scope=INIT_SCOPE_ALL)
    a = _snapshot(m)
    for k in ("wih", "whh", "rw", "rb"):
        assert torch.allclose(_ratio(a[k], b[k]), torch.full_like(_ratio(a[k], b[k]), GAMMA))


def test_scope_no_readout_leaves_readout():
    m = build_model(ARCH_MODULAR_SHARED, H, seed=0)
    b = _snapshot(m)
    apply_init_scale(m, GAMMA, scope=INIT_SCOPE_NO_READOUT)
    a = _snapshot(m)
    assert torch.allclose(_ratio(a["wih"], b["wih"]), torch.full_like(_ratio(a["wih"], b["wih"]), GAMMA))
    assert torch.allclose(_ratio(a["whh"], b["whh"]), torch.full_like(_ratio(a["whh"], b["whh"]), GAMMA))
    assert torch.allclose(a["rw"], b["rw"])
    assert torch.allclose(a["rb"], b["rb"])


def test_scope_input_only_scales_only_input():
    m = build_model(ARCH_MODULAR_SHARED, H, seed=0)
    b = _snapshot(m)
    apply_init_scale(m, GAMMA, scope=INIT_SCOPE_INPUT_ONLY)
    a = _snapshot(m)
    assert torch.allclose(_ratio(a["wih"], b["wih"]), torch.full_like(_ratio(a["wih"], b["wih"]), GAMMA))
    assert torch.allclose(a["whh"], b["whh"])
    assert torch.allclose(a["rw"], b["rw"])
    assert torch.allclose(a["rb"], b["rb"])


def test_gamma_one_is_noop():
    m = build_model(ARCH_MODULAR_SHARED, H, seed=0)
    b = _snapshot(m)
    apply_init_scale(m, 1.0, scope=INIT_SCOPE_ALL)
    a = _snapshot(m)
    for k in ("wih", "whh", "rw", "rb"):
        assert torch.allclose(a[k], b[k])
