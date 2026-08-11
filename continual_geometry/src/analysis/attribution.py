"""Attribution of forgetting to the three capacity factors (`00` §8).

Two things happen here.

**The decomposition.** `α = Ψ_eff · (1 + R_eff⁻²) / D_eff` is exact, so the change in
retained capacity between two boundaries splits additively in log space with **zero
residual**:

    Δ log α = Δ log Ψ_eff + Δ log(1 + R_eff⁻²) − Δ log D_eff

No cross-terms, no leftover. The residual is computed anyway and asserted small,
because it is the cheapest available check that the two geometry measurements being
differenced came from the same estimator configuration — a nonzero residual means
something upstream is inconsistent, not that the identity failed.

**The center-collapse share of the radius channel.** `ρ_c` sits *outside* the
identity, yet Wakhloo, Sussman & Chung (PRL 2023) show center correlation is
effectively equivalent to shrunk center separation — which is to say it arrives in
`R_eff`. So a radius contribution is not automatically an anisotropy story: some of
it is centers falling together. `00` §8 already requires reporting `ρ_c` alongside
`R_eff`; this makes the joint report quantitative by asking what fraction of an
observed `Δ log R_eff` the observed `Δ ρ_c` already accounts for.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

FACTORS = ("utility", "radius", "dimension")
GEOMETRY = ("alpha", "D_eff", "R_eff", "Psi_eff", "rho_c_glue", "rho_c_signed")

# Measured on the `00` §6.1 center-correlation sweep of the B.5 recovery suite
# (`results/glue_core_recovery.json`, `center_correlation`, n_t=200, policy=all;
# P=2, M=200, N=1000, D=4, R=1). Ground-truth ρ_C 0 → 0.8 drives R_eff 1.02 → 1.82
# while D_eff stays flat to 0.38%, so on synthetic manifolds the radius absorbs
# center correlation essentially alone, exactly as the duality says.
#
#     log R_eff = RHO_R_INTERCEPT + RHO_R_EXPONENT · (−log(1 − ρ_c_glue))
#
# i.e. R_eff ∝ (1 − ρ_c)^(−0.355), which fits the five sweep points at R² = 0.99986.
# Refitted and pinned by `tests/test_analysis_attribution.py`.
RHO_R_EXPONENT = 0.35484
RHO_R_INTERCEPT = 0.012613
RHO_R_R2 = 0.99986
RHO_R_CONVENTION = "rho_c_glue"
RHO_R_PROVENANCE = "glue_core_recovery.json:center_correlation/n_t=200,policy=all"

# `R_eff` Monte-Carlo noise floor, CV 0.50% (`results/cost_model.json`), in log units.
# Radius changes below this are unresolvable, so they cannot be a ratio's denominator.
MIN_DLOG_R = float(np.log1p(0.0050))


@dataclass(frozen=True)
class GeometryPoint:
    """One `(module, boundary, past-task)` geometry measurement.

    `alpha` is **retained** capacity `α(·; y_j)` for the past task being attributed,
    not generic capacity — the identity holds for whichever ensemble `Y` was used,
    but a decomposition of generic capacity answers a different question.
    """

    alpha: float
    D_eff: float
    R_eff: float
    Psi_eff: float
    rho_c_glue: float
    rho_c_signed: float
    label: str = ""

    @classmethod
    def from_result(cls, r, label: str = "") -> "GeometryPoint":
        get = r.get if isinstance(r, dict) else lambda k: getattr(r, k)
        return cls(*(float(get(k)) for k in GEOMETRY), label=label)

    @property
    def alpha_from_identity(self) -> float:
        return self.Psi_eff * (1.0 + self.R_eff**-2) / self.D_eff


@dataclass(frozen=True)
class Attribution:
    """Signed log-space contributions. Positive = pushed capacity up."""

    dlog_alpha: float
    terms: dict[str, float]
    shares: dict[str, float]
    residual: float
    dominant: str
    center: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def attribute(
    before: GeometryPoint,
    after: GeometryPoint,
    *,
    tol: float = 1e-8,
    strict: bool = True,
) -> Attribution:
    """Decompose `Δ log α` from `before` to `after` into the three factors.

    Sign convention: each term is the amount that factor **pushed capacity up**, so
    the dimension term carries the minus sign internally and the three terms sum to
    `Δ log α` as reported. Forgetting is `dlog_alpha < 0`, and the factor
    responsible is the most negative term.

    `shares` are `|term| / Σ|term|` — a fraction of *total motion*, not of the net
    change, because factors routinely oppose each other and shares of a small net
    difference of large opposing terms exceed 1 and mislead. Read them with the
    signs, and read `cancellation` in `meta` first: when it is high the factors are
    fighting and "which factor explains the change" is the wrong question.
    """
    d = lambda f, g: float(np.log(g) - np.log(f))  # noqa: E731
    terms = {
        "utility": d(before.Psi_eff, after.Psi_eff),
        "radius": d(1.0 + before.R_eff**-2, 1.0 + after.R_eff**-2),
        "dimension": -d(before.D_eff, after.D_eff),
    }
    dlog_alpha = d(before.alpha, after.alpha)
    residual = dlog_alpha - sum(terms.values())

    if strict and abs(residual) > max(tol, 1e-6 * abs(dlog_alpha)):
        raise ValueError(
            f"identity residual {residual:.3e} exceeds tolerance — the three-factor "
            f"identity is exact, so this means the two points were measured under "
            f"different estimator settings, or `alpha` is not the same ensemble's "
            f"capacity as the factors ({before.label!r} -> {after.label!r})"
        )

    total = sum(abs(v) for v in terms.values())
    shares = {k: (abs(v) / total if total > 0 else 0.0) for k, v in terms.items()}
    dominant = max(terms, key=lambda k: abs(terms[k])) if total > 0 else "none"

    return Attribution(
        dlog_alpha=dlog_alpha,
        terms=terms,
        shares=shares,
        residual=residual,
        dominant=dominant,
        center=center_collapse_share(before, after),
        meta={
            "cancellation": float(1.0 - abs(dlog_alpha) / total) if total > 0 else 0.0,
            "before": before.label,
            "after": after.label,
        },
    )


def radius_from_rho(rho: float) -> float:
    """`R_eff` predicted by center correlation alone, per the measured calibration."""
    rho = float(np.clip(rho, 0.0, 0.995))
    return float(np.exp(RHO_R_INTERCEPT + RHO_R_EXPONENT * -np.log1p(-rho)))


def center_collapse_share(
    before: GeometryPoint, after: GeometryPoint, *, min_dlog_R: float = MIN_DLOG_R
) -> dict:
    """How much of the observed `Δ log R_eff` does the observed `Δ ρ_c` account for?

    Uses the synthetic-manifold calibration above to convert the measured change in
    center correlation into the radius change it would have produced on its own, and
    reports that as a fraction of the radius change actually observed:

        attributable = Δ log R_eff(ρ_c) / Δ log R_eff observed

    Read it as follows. Near **1**, the radius channel *is* the center channel and
    reporting them as separate mechanisms would double-count. Near **0** with `ρ_c`
    moving, centers moved and the radius did not follow, so the radius change is
    anisotropy rather than center collapse. **Above 1** means centers collapsed more
    than the radius reflects, so something else offset it.

    Three caveats, all of which keep this a reported attribution rather than a
    correction. The calibration is fitted on synthetic manifolds where `ρ_C` is the
    *only* thing varying, so it is an upper bound on how much a representation's
    `ρ_c` movement can explain — in a representation, radius and centers move
    together. It is fitted on `rho_c_glue` and is not valid for the signed
    convention, whose sign carries information the absolute calibration cannot see.
    And it is monotone in `ρ_c` only, so it says nothing when `ρ_c` is unchanged;
    `attributable` is then reported as `None` rather than 0.

    **The denominator is floored at the `R_eff` noise floor.** A ratio whose
    denominator is a radius change smaller than the estimator can resolve is
    unbounded — the first smoke grid produced fractions of 42 and −134 this way, from
    `Δ log R_eff` of order 1e-3. Below the floor the fraction is `None` with
    `reason: "radius change below noise floor"`, which is the honest statement: the
    radius did not measurably move, so there is nothing to attribute.
    """
    dlog_R = float(np.log(after.R_eff) - np.log(before.R_eff))
    drho = after.rho_c_glue - before.rho_c_glue
    predicted = float(np.log(radius_from_rho(after.rho_c_glue))
                      - np.log(radius_from_rho(before.rho_c_glue)))

    out = {
        "dlog_R_eff_observed": dlog_R,
        "dlog_R_eff_predicted_from_rho_c": predicted,
        "d_rho_c_glue": float(drho),
        "d_rho_c_signed": float(after.rho_c_signed - before.rho_c_signed),
        "calibration_convention": RHO_R_CONVENTION,
    }
    out["min_dlog_R"] = float(min_dlog_R)
    if abs(drho) < 1e-12:
        out["attributable_fraction"] = None
        out["reason"] = "rho_c unchanged — no conversion to report"
    elif abs(dlog_R) < min_dlog_R:
        out["attributable_fraction"] = None
        out["reason"] = "radius change below noise floor"
    else:
        out["attributable_fraction"] = predicted / dlog_R
        out["same_direction"] = bool(np.sign(predicted) == np.sign(dlog_R))
    return out


def attribution_table(rows: list[Attribution]) -> dict:
    """Pool a set of attributions — per stream condition, γ arm, or module.

    Averages in log space, which is where the identity is additive; the mean of the
    three term-means therefore still equals the mean `Δ log α` exactly. `shares` are
    recomputed from the pooled terms rather than averaged, since a mean of ratios is
    not the ratio of means.
    """
    if not rows:
        raise ValueError("no attributions to pool")
    terms = {k: float(np.mean([r.terms[k] for r in rows])) for k in FACTORS}
    dlog_alpha = float(np.mean([r.dlog_alpha for r in rows]))
    total = sum(abs(v) for v in terms.values())
    frac = [r.center["attributable_fraction"] for r in rows
            if r.center.get("attributable_fraction") is not None]
    return {
        "n": len(rows),
        "dlog_alpha": dlog_alpha,
        "dlog_alpha_sem": float(np.std([r.dlog_alpha for r in rows], ddof=1)
                                / np.sqrt(len(rows))) if len(rows) > 1 else 0.0,
        "terms": terms,
        "term_sem": {
            k: float(np.std([r.terms[k] for r in rows], ddof=1) / np.sqrt(len(rows)))
            if len(rows) > 1 else 0.0 for k in FACTORS
        },
        "shares": {k: (abs(v) / total if total else 0.0) for k, v in terms.items()},
        "dominant": max(terms, key=lambda k: abs(terms[k])) if total else "none",
        "max_residual": float(max(abs(r.residual) for r in rows)),
        "center_attributable_median": float(np.median(frac)) if frac else None,
        "center_attributable_n": len(frac),
    }
