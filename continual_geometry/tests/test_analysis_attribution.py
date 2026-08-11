"""Tests for the three-factor attribution and the center-collapse conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import attribution as A

ROOT = Path(__file__).resolve().parents[1]


def _point(D=4.0, R=1.0, Psi=0.8, rho=0.1, rho_s=0.1, label=""):
    alpha = Psi * (1.0 + R**-2) / D
    return A.GeometryPoint(alpha=alpha, D_eff=D, R_eff=R, Psi_eff=Psi,
                           rho_c_glue=rho, rho_c_signed=rho_s, label=label)


def test_identity_closes_exactly():
    a = A.attribute(_point(D=4.0, R=1.0, Psi=0.8), _point(D=5.5, R=1.4, Psi=0.6))
    assert abs(a.residual) < 1e-12
    assert a.dlog_alpha == pytest.approx(sum(a.terms.values()), abs=1e-12)


def test_inconsistent_alpha_is_rejected():
    """The residual's job: catch two points that did not come from one estimator."""
    before, after = _point(), _point(D=5.0)
    broken = A.GeometryPoint(**{**after.__dict__, "alpha": after.alpha * 1.5})
    with pytest.raises(ValueError, match="identity residual"):
        A.attribute(before, broken)
    loose = A.attribute(before, broken, strict=False)
    assert abs(loose.residual) > 0.1


@pytest.mark.parametrize(
    "kw, factor, sign",
    [
        (dict(D=8.0), "dimension", -1),   # dimension up -> capacity down
        (dict(D=2.0), "dimension", +1),
        (dict(Psi=0.4), "utility", -1),
        (dict(R=2.0), "radius", -1),      # radius up -> (1+R^-2) down
        (dict(R=0.5), "radius", +1),
    ],
)
def test_single_factor_moves_only_its_own_term_with_the_right_sign(kw, factor, sign):
    a = A.attribute(_point(), _point(**kw))
    assert a.dominant == factor
    assert np.sign(a.terms[factor]) == sign
    assert np.sign(a.dlog_alpha) == sign
    for other in set(A.FACTORS) - {factor}:
        assert a.terms[other] == pytest.approx(0.0, abs=1e-12)
    assert a.shares[factor] == pytest.approx(1.0)


def test_cancellation_is_flagged_when_factors_oppose():
    """Two large opposing terms and a near-zero net: shares must not be read alone."""
    a = A.attribute(_point(D=4.0, Psi=0.8), _point(D=8.0, Psi=1.6))
    assert abs(a.dlog_alpha) < 1e-12
    assert a.meta["cancellation"] > 0.99
    assert min(abs(v) for v in (a.terms["utility"], a.terms["dimension"])) > 0.5


def test_rho_calibration_matches_the_recovery_sweep_it_was_fitted_on():
    """Pin the hardcoded constants to their source, so a refit cannot drift silently."""
    d = json.loads((ROOT / "results" / "glue_core_recovery.json").read_text())
    sweep = d["sweeps"]["center_correlation"]["n_t=200,policy=all"]
    rho = np.array([p["rho_c_glue"]["mean"] for p in sweep])
    R = np.array([p["R_eff"]["mean"] for p in sweep])

    slope, intercept = np.polyfit(-np.log1p(-rho), np.log(R), 1)
    assert slope == pytest.approx(A.RHO_R_EXPONENT, rel=1e-3)
    assert intercept == pytest.approx(A.RHO_R_INTERCEPT, rel=1e-2, abs=1e-4)

    pred = np.log([A.radius_from_rho(r) for r in rho])
    r2 = 1 - np.sum((np.log(R) - pred) ** 2) / np.sum((np.log(R) - np.log(R).mean()) ** 2)
    assert r2 == pytest.approx(A.RHO_R_R2, abs=1e-3)

    # D_eff flat across the sweep is why the radius channel is the one that absorbs
    # center correlation; if this ever fails the conversion is attributing to the
    # wrong channel.
    D = np.array([p["D_eff"]["mean"] for p in sweep])
    assert (D.max() - D.min()) / D.mean() < 0.01


def test_center_collapse_explains_all_of_a_pure_rho_driven_radius_change():
    """A radius change generated *by* the calibration must be fully attributed."""
    rho0, rho1 = 0.1, 0.6
    before = _point(R=A.radius_from_rho(rho0), rho=rho0)
    after = _point(R=A.radius_from_rho(rho1), rho=rho1)
    frac = A.attribute(before, after).center["attributable_fraction"]
    assert frac == pytest.approx(1.0, abs=1e-9)


def test_center_collapse_explains_none_of_a_radius_change_at_fixed_rho():
    a = A.attribute(_point(R=1.0, rho=0.3), _point(R=1.6, rho=0.3))
    assert a.center["attributable_fraction"] is None
    assert a.center["reason"] == "rho_c unchanged — no conversion to report"
    assert a.center["dlog_R_eff_observed"] > 0


def test_unresolvable_radius_change_is_not_used_as_a_denominator():
    """An unbounded ratio from a sub-noise-floor radius change is suppressed."""
    tiny = 0.4 * A.MIN_DLOG_R
    c = A.attribute(_point(R=1.0, rho=0.2), _point(R=float(np.exp(tiny)), rho=0.5)).center
    assert c["attributable_fraction"] is None
    assert c["reason"] == "radius change below noise floor"
    # just above the floor it is reported again, and it is large — which is the point:
    # the number is meaningful there and meaningless below
    big = 1.2 * A.MIN_DLOG_R
    c2 = A.attribute(_point(R=1.0, rho=0.2), _point(R=float(np.exp(big)), rho=0.5)).center
    assert c2["attributable_fraction"] > 1.0


def test_center_collapse_is_partial_when_radius_outruns_rho():
    """The realistic case: centers move, radius moves more."""
    rho0, rho1 = 0.2, 0.4
    implied = A.radius_from_rho(rho1) / A.radius_from_rho(rho0)
    before = _point(R=1.0, rho=rho0)
    after = _point(R=1.0 * implied * 2.0, rho=rho1)
    frac = A.attribute(before, after).center["attributable_fraction"]
    assert 0.0 < frac < 1.0
    assert A.attribute(before, after).center["same_direction"]


def test_opposing_center_and_radius_is_detectable():
    """rho_c falls while R_eff rises — not a center-collapse story at all."""
    c = A.attribute(_point(R=1.0, rho=0.6), _point(R=1.4, rho=0.2)).center
    assert c["same_direction"] is False
    assert c["attributable_fraction"] < 0


def test_pooling_preserves_the_identity_and_recomputes_shares():
    rows = [A.attribute(_point(), _point(D=4.0 + 0.3 * k, R=1.0 + 0.1 * k))
            for k in range(1, 6)]
    t = A.attribution_table(rows)
    assert t["dlog_alpha"] == pytest.approx(sum(t["terms"].values()), abs=1e-12)
    assert sum(t["shares"].values()) == pytest.approx(1.0)
    assert t["n"] == 5 and t["max_residual"] < 1e-12
    # a mean of ratios is not the ratio of means; shares come from the pooled terms
    naive = float(np.mean([r.shares["dimension"] for r in rows]))
    assert t["shares"]["dimension"] == pytest.approx(
        abs(t["terms"]["dimension"]) / sum(abs(v) for v in t["terms"].values()))
    assert naive != pytest.approx(t["shares"]["dimension"], abs=1e-9)


def test_from_result_accepts_dicts_and_objects():
    p = _point(D=5.0, R=1.2, Psi=0.7, rho=0.3, rho_s=-0.3)
    d = {k: getattr(p, k) for k in A.GEOMETRY}
    assert A.GeometryPoint.from_result(d, "x").D_eff == 5.0
    assert A.GeometryPoint.from_result(p, "x").rho_c_signed == -0.3
    assert A.GeometryPoint.from_result(d).alpha_from_identity == pytest.approx(p.alpha)
