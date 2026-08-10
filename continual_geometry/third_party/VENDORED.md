# Vendored estimators

Pinned third-party capacity/geometry code. Vendored (not pip-installed) so we can
read it, verify it against `docs/00-math-spec.md`, and patch it for float64 /
explicit rcond / RNG isolation. See `docs/03-references.md` §D.

`.git` histories were stripped after cloning; the pinned SHA below is the
provenance of record. Re-clone from the URL at that SHA to reproduce.

**Fetched:** 2026-07-31 (both repos), default branch, `git clone` then SHA pinned.

## Corrections policy — do NOT edit `third_party/` in place

`third_party/` is kept **byte-identical to upstream** so the SHA↔directory
correspondence stays meaningful. All corrections (RNG-Generator injection to remove
global `np.random.seed`, float64 enforcement, explicit pseudo-inverse `rcond` with
logged rank) live in a thin **`src/glue/adapters/`** layer that wraps these
functions — see `src/glue/adapters/README.md`. If an in-place edit is ever
unavoidable, record it as a `.patch` under **`third_party/patches/`** and reference
it here; never silently mutate vendored source.

---

## Estimator version strings (for the results key)

Every geometry result records which estimator produced it as `{repo}@{sha}`.
The `docs/02-validation-suite.md` §9 pooling firewall applies **pairwise across
all three** — numbers from different estimators must never be pooled.

| Version string | Role |
|---|---|
| `replicaMFT@d56eda1d86a71f4f4601ef1acc3254e20259a80c` | Primary near-term estimator: {α, R, D, ρ_c} |
| `correlated_capacity@de8dac79760e8eb552c036604503efcb87e1e6b6` | Ground-truth α_sim + correlated-capacity (Wakhloo duality). **Not GLUE.** |
| `GLUE@<pending>` | Full GLUE incl. ρ_a, ψ — pending early-access (form in §GLUE below) |

---

## replicaMFT — `schung039/neural_manifolds_replicaMFT`

- **URL:** https://github.com/schung039/neural_manifolds_replicaMFT
- **Fetched:** 2026-07-31
- **Pinned SHA:** `d56eda1d86a71f4f4601ef1acc3254e20259a80c`
- **Entry point:** `mftma/manifold_analysis_correlation.py::manifold_analysis_corr(XtotT, kappa, n_t, t_vecs=None, n_reps=10)`
- **Returns (verified by reading source):** `(a_Mfull_vec, R_M_vec, D_M_vec, res_coeff0, KK)`
  = capacity α, radius R, dimension D, center-correlation ρ_c, and K.
- **Provides:** {α, R, D, ρ_c}. **Missing:** ρ_a, ψ (the open dependency — GLUE only).
- **Use at κ = 0** (`docs/00-math-spec.md` §6; estimators are not interchangeable at κ ≠ 0).
- **Patch-needs before use:** no explicit `rcond` on pseudo-inverses and no float64
  enforcement observed — apply per `AGENTS.md` §4 and log effective rank.

## correlated_capacity — `awakhloo/correlated_capacity`

- **URL:** https://github.com/awakhloo/correlated_capacity  (= Zenodo v1.0.0, doi 10.5281/zenodo.7844169)
- **Fetched:** 2026-07-31
- **Pinned SHA:** `de8dac79760e8eb552c036604503efcb87e1e6b6`
- **Reproduces:** Wakhloo, Sussman & Chung, PRL 131, 027301 (2023).
- **Provides:**
  - `capacity/manifold_simcap_analysis.py::manifold_simcap_analysis(XtotT, n_rep, seed)`
    — **general point-cloud simulation capacity α_sim** (SVM separability via cvxopt QP
    at κ=0, bisection on feature dim). Input is a sequence of `(N, P_i)` arrays.
  - `capacity/sphere_sim_capacity.py::sphere_simcap(sphere_axes, ...)` — sphere-parameterized
    α_sim (analytic axes + internal surface sampling; iterative constraint augmentation).
  - `capacity/replica_correlations.py` — correlated replica mean-field capacity (the duality).
  - `capacity/mean_field_cap.py`, `low_rank_capacity.py` — base MFT and low-rank approximation.
- **Amendment 3 finding (verified by reading source):** `manifold_simcap_analysis` handles
  **general point clouds**, not only spheres. Therefore α_sim **can** serve as the
  `docs/02-validation-suite.md` §2a gate (α_sim vs α_mf) at project parameters.
  `sphere_sim_capacity` is the spherical special case, sufficient for the §1
  ground-truth-recovery tests but not required for §2a.
- **NOT a GLUE source:** it does not emit ρ_a or ψ. Distinct estimator; keep the §9
  pooling firewall.
- **Patch-needs before use:** `manifold_simcap_analysis.py:173` and
  `sphere_sim_capacity.py:194` call global `np.random.seed(...)` — **violates
  `AGENTS.md` §4** (explicit `numpy.random.Generator`, no global seeding). Wrap for
  RNG isolation before use. Enforce float64 / explicit rcond as above.

---

## GLUE — full estimator (ρ_a, ψ), pending

- **Code is NOT public.** `chung-neuroai-lab/feature-learning-geometry` (the ICML 2025
  figure-repro repo) *calls* GLUE; it does not contain Algorithm 1, ρ_a, or ψ (verified
  by reading its README + file tree).
- **Early-access request form:**
  https://docs.google.com/forms/d/e/1FAIpQLSc_IHUkc2zlJv0DIhSL_tiyD7Ty4nCeFdW0U7s-hCVWchefBg/viewform
- Until granted: ρ_a and ψ remain the open dependency; do not fabricate them.
