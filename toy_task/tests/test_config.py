"""RunConfig validation and run_id cache-key construction."""

import pytest

from toy_task.config import (
    RunConfig,
    ARCH_DENSE,
    ARCH_MODULAR_SHARED,
    ANGLE_EVEN,
    ANGLE_RANDOM,
    INIT_SCOPE_NO_READOUT,
    STIMULUS_SHARED,
    STIMULUS_NOVEL,
)


def _cfg(**kw):
    base = dict(arch=ARCH_DENSE, hidden_size=24, gamma=1.0, similarity=0.0, seed=0)
    base.update(kw)
    return RunConfig(**base)


def test_default_regime_is_shared():
    assert _cfg().stimulus_regime == STIMULUS_SHARED


def test_invalid_regime_rejected():
    with pytest.raises(ValueError):
        _cfg(stimulus_regime="bogus")


def test_run_id_encodes_regime():
    rid = _cfg(stimulus_regime=STIMULUS_NOVEL).run_id()
    assert "rnovel" in rid
    assert "rshared" not in rid
    assert _cfg(stimulus_regime=STIMULUS_SHARED).run_id().count("rshared") == 1


def test_run_id_regime_distinguishes_caches():
    shared = _cfg(stimulus_regime=STIMULUS_SHARED).run_id()
    novel = _cfg(stimulus_regime=STIMULUS_NOVEL).run_id()
    assert shared != novel


def test_run_id_still_encodes_scope_and_epochs():
    rid = _cfg(init_scope=INIT_SCOPE_NO_READOUT, epochs_per_phase=250).run_id()
    assert "scno_readout" in rid
    assert "e250" in rid


def test_run_id_even_includes_ang_token():
    rid = _cfg(angle_mode=ANGLE_EVEN).run_id()
    assert "angeven" in rid


def test_run_id_random_omits_ang_token_for_legacy_compat():
    rid = _cfg(angle_mode=ANGLE_RANDOM).run_id()
    assert "ang" not in rid


def test_run_id_even_distinguishes_from_random():
    assert _cfg(angle_mode=ANGLE_RANDOM).run_id() != _cfg(angle_mode=ANGLE_EVEN).run_id()


def test_comms_bandwidth_in_run_id():
    assert "bw0.0" in _cfg(comms_bandwidth=0.0).run_id()
    rid = _cfg(arch=ARCH_MODULAR_SHARED, comms_bandwidth=0.5).run_id()
    assert "bw0.5" in rid


def test_dense_ignores_bandwidth_in_model():
    cfg = _cfg(comms_bandwidth=0.75)
    assert cfg.effective_comms_bandwidth == 0.0
