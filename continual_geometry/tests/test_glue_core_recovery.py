"""Fast §B.5 ground-truth recovery checks for `src/glue/core.py` (`02` §1).

Reduced parameters so the suite stays quick; the full B.5 sweep
(P=2, M=200, N=1000, n_t=200, 3 seeds) is `scripts/run_glue_core_recovery.py`
and its output is `results/glue_core_recovery.json`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.glue import core
from src.manifolds import generator

P, N, M, N_T = 2, 300, 80, 80
D_FIXED, R_FIXED = 4, 1.0


def _measure(*, D=D_FIXED, R=R_FIXED, rho_C=0.0, seed=0):
    rng = np.random.default_rng(seed)
    arr = generator.make_arrangement(P, N, D, R, M, rng, rho_C=rho_C)
    return core.glue_measures(
        core.from_arrangement(arr.points), np.random.default_rng(seed + 100), n_t=N_T
    )


def _is_increasing(xs, slack=0.0):
    return all(b >= a - slack for a, b in zip(xs, xs[1:]))


def test_D_eff_tracks_ground_truth_dimension():
    Ds = [2, 4, 8]
    res = [_measure(D=D) for D in Ds]
    assert _is_increasing([r.D_eff for r in res])
    assert all(abs(r.D_eff - D) / D < 0.35 for r, D in zip(res, Ds))
    assert not _is_increasing([r.alpha for r in res])  # capacity falls with D


def test_R_eff_tracks_ground_truth_radius_without_moving_D_eff():
    Rs = [0.8, 1.4, 2.0]
    res = [_measure(R=R) for R in Rs]
    assert _is_increasing([r.R_eff for r in res])
    assert all(abs(r.R_eff - R) / R < 0.35 for r, R in zip(res, Rs))
    D_effs = [r.D_eff for r in res]
    assert max(D_effs) - min(D_effs) < 0.35 * np.mean(D_effs)  # axes stay separable


def test_rho_c_glue_tracks_generated_center_correlation():
    rhos = [0.0, 0.4, 0.8]
    res = [_measure(rho_C=r) for r in rhos]
    assert _is_increasing([r.rho_c_glue for r in res])
    assert res[-1].rho_c_glue == pytest.approx(0.8, abs=0.2)


def test_psi_eff_in_unit_interval():
    for r in (_measure(), _measure(D=8), _measure(R=2.0)):
        assert 0.0 < r.Psi_eff <= 1.0
        assert r.identity_residual < 1e-8
