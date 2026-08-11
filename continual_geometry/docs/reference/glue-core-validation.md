# Reference sheet — validation of our GLUE core implementation

```
Source:       Own implementation, src/glue/core.py (estimator `glue_core`).
              Definitions from Chou, Kirsanov, Yang & Chung, ICLR 2026, App. B.3
              (via docs/reference/glue-decomposition.md) and Chou, Le, Wang &
              Chung, ICML 2025, Algorithm 2 + App. B.5 (recovery protocol).
Version:      glue_core @ commit of 2026-08-11
Derived from: measurements in results/*.json produced by scripts/run_*.py.
              Every number below is measured, none transcribed from a paper.
Transcribed by: agent (Opus 5), 2026-08-11
Verified by (human): ____________          # blank until checked
Status: agent-measured, reproducible from the scripts named per section.
        Code may not cite this sheet until "Verified by (human)" is filled.
```

This sheet is the methods paragraph for the paper and the answer to the reviewer
question *"you implemented the estimator yourself — why should we believe it?"*

**The honest framing.** GLUE's reference implementation is not public. We
implemented the three-factor decomposition from the authors' published
definitions and validated it two ways: against an **exact analytic limit** the
estimator must reproduce (point manifolds, `α = 2` at `κ = 0`), and against the
authors' **own recovery protocol** (App. B.5) plus an independent simulation
ground truth (`α_sim`). Code is released. Section 5 states a measured limitation
we did not have to disclose.

Why the implementation exists at all: `replicaMFT`, the public estimator, is
label-invariant by construction and cannot express capacity for a specified
dichotomy — see `third_party/VENDORED.md`.

---

## 0. What the implementation does

Anchor points come from the **joint** capacity QP over all `P` manifolds at
`κ = 0`, `min ½‖v − t‖²  s.t.  Gv ≤ 0` with rows of `G` equal to `y_μ z^μ_i`.
This is not a stylistic choice. In §B.3 `S_y = diag(y)·S`, and `diag(y)` cancels
in all three quadratic forms (`diag(y)² = I`, `pinv(D A D) = D A† D` for
orthogonal diagonal `D`), so `a`, `b`, `c` are invariant to `y` *given* `S`.
All label dependence must therefore enter through anchors solved for jointly; a
per-manifold construction reproduces exactly the label-invariance that makes
`replicaMFT` unusable here.

The QP's KKT conditions make its duals a non-negative least-squares problem,
`λ = argmin_{λ≥0} ‖Gᵀλ − t‖²`, so no vendored code sits in the estimation path.

A manifold whose dual mass is zero contributes a **zero row** (its constraint is
slack). Section 1 is what pins that convention.

---

## 1. Exact analytic limit — point manifolds

For `P` single-point manifolds in general position, capacity at `κ = 0` is
`α = 2`. This is the strongest available check because it is exact, and it
discriminates the zero-row convention: with zero rows, about half the manifolds
are active for random `t`, `E[a] = P/2`, and `α = P/E[a] = 2`.

| quantity | expected | measured |
|---|---|---|
| `α` (P = 8, N = 200, n_t = 400) | 2 | **2.00 ± 0.12** |
| active fraction | 0.5 | **0.50 ± 0.05** |

`tests/test_glue_core.py::test_point_manifold_capacity_is_two`.

Also verified in the same suite: `diag(y)` cancels numerically as the algebra
says; anchors *do* change with `y` (so the joint QP is label-sensitive); the
identity `α = Ψ_eff(1 + R_eff⁻²)/D_eff` closes to `< 1e-8` relative, which is a
transcription check on the file since it is exact by construction.

---

## 2. Authors' recovery protocol — ICML 2025 App. B.5

`P = 2`, `M = 200`, `N = 1000`, `n_t = 200`, 3 seeds, `κ = 0`. Three
**independent** sweeps: `D` and `R` are each swept with correlation at zero, then
`D = 4, R = 1` are held fixed while `ρ_C` sweeps.
`scripts/run_glue_core_recovery.py` → `results/glue_core_recovery.json`.

**Dimension** (`D → D_eff`, at `R = 1`, `ρ_C = 0`):

| ground truth `D` | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| `D_eff` | 2.25 | 4.07 | 5.61 | 6.98 | 8.17 |

**Radius** (`R → R_eff`, at `D = 4`, `ρ_C = 0`):

| ground truth `R` | 0.8 | 1.0 | 1.4 | 1.7 | 2.0 |
|---|---|---|---|---|---|
| `R_eff` | 0.87 | 1.03 | 1.34 | 1.56 | 1.77 |

**Center correlation** (`ρ_C → rho_c_glue`, at `D = 4`, `R = 1`):

| ground truth `ρ_C` | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|---|
| `rho_c_glue` | 0.04 | 0.22 | 0.41 | 0.61 | 0.81 |

All three are monotone, `α` falls monotonically along each, and `Ψ_eff` stays in
`[0.85, 0.91] ⊂ [0,1]`.

**Cross-axis separability.** `D_eff` sits at 4.07–4.08 across the *entire* `R`
sweep and the *entire* `ρ_C` sweep. The axes are separately identifiable, which is
what the §8 attribution needs.

**Where correlation goes — the P5/F4 question, answered with a number.** Sweeping
`ρ_C` from 0 to 0.8 moves `R_eff` 1.02 → 1.82 and leaves `D_eff` at 4.07 → 4.08.
Center correlation is absorbed **entirely by the radius**, as the Wakhloo duality
predicts, and not at all by dimension. `00` §8 must keep reporting `ρ_c` beside
`R`; it must not attribute an `R` change to geometry alone.

---

## 3. Independent capacity ground truth — `α_sim`, and the `α_mf` cross-check

`α_sim` is simulation capacity (SVM separability at `κ = 0`, bisection on the
random-projection dimension, `α = P/N_c`), computed through
`src/glue/adapters/simcap.py`. `α_mf` is `replicaMFT`, which is generic-only and
therefore appears at `β = 0` only, reduced with the harmonic mean.

Identical point clouds, `M = 60`, `D = 4`, `R = 1`, `n_t = 200`, `n_rep = 10`,
3 seeds. `scripts/run_gate_2a.py` → `results/gate_2a.json`.

| `P` | `N` | `α_sim` | `α_core` | `α_mf` | core vs sim | mf vs sim |
|---|---|---|---|---|---|---|
| 8 | 300 | 0.483 ± 0.052 | 0.440 ± 0.006 | 0.388 ± 0.003 | 8.9% | 19.8% |
| 8 | 600 | 0.449 ± 0.046 | 0.439 ± 0.013 | 0.387 ± 0.003 | 2.2% | 13.9% |
| 8 | 1200 | 0.436 ± 0.043 | 0.430 ± 0.016 | 0.387 ± 0.001 | 1.5% | 11.3% |
| **16** | **300** | 0.420 ± 0.017 | 0.434 ± 0.002 | 0.413 ± 0.001 | **3.4%** | **1.6%** |
| 16 | 600 | 0.440 ± 0.009 | 0.440 ± 0.004 | 0.414 ± 0.003 | 0.0% | 6.0% |
| 16 | 1200 | 0.460 ± 0.013 | 0.437 ± 0.006 | 0.413 ± 0.001 | 4.9% | 10.1% |
| 32 | 300 | 0.437 ± 0.014 | 0.440 ± 0.005 | 0.423 ± 0.002 | 0.7% | 3.1% |
| 32 | 600 | 0.451 ± 0.007 | 0.433 ± 0.001 | 0.421 ± 0.001 | 4.0% | 6.6% |
| 32 | 1200 | 0.440 ± 0.019 | 0.439 ± 0.004 | 0.422 ± 0.001 | 0.1% | 3.9% |

`α_core` agrees with `α_sim` to **2.9% on average and ≤ 5% everywhere except
`P = 8, N = 300`**, with no systematic sign.

Two observations that matter for how the residual should be read:

- **`α_core` is the more stable of the two.** Across `N` at fixed geometry it
  varies by under 2% (0.434 / 0.440 / 0.437 at `P = 16`) with a seed SD around
  0.004, while `α_sim` drifts monotonically upward with ambient `N`
  (0.420 / 0.440 / 0.460) with a seed SD 3–10× larger. The generated geometry is
  `N`-independent, so that drift is a property of the simulation estimator — its
  bisection range and random-projection statistics both scale with `N`. Part of
  the residual at large `N` is therefore `α_sim`'s bias, not `α_core`'s error. We
  do not claim which is right; we report both.
- **`α_mf` is systematically low**, by 1.6–19.8%, and worst at `P = 8` where the
  mean-field limit is furthest away. It improves with `N` at `P = 8`
  (19.8 → 13.9 → 11.3%), the expected direction.

**This settles the `P = 16` question (`01` Phase 0).** At the project design point
`P = 16, N = 300`, `α_core` is within 3.4% of `α_sim` and `α_mf` within 1.6% — the
best mean-field agreement anywhere in the table, and inside `α_sim`'s own seed
scatter. The pre-specified escalation (raise `N`, then accept-and-report, then
raise `P` to 32) is **not triggered**. `P = 16, N = 300` stands, and the
five-factor redesign is off the table.

---

## 4. `center_policy` — settled by measurement

`s⁰_μ = E[s^μ(y,t)]` has to decide what to do with samples where manifold `μ` was
inactive. The active fraction is ~0.5, so this is half the data, and §1 does not
discriminate: `α` depends only on `S`, whereas `R_eff` and both `ρ_c` conventions
normalize by `‖s⁰‖`. Both policies were run over the full B.5 protocol.

Mean absolute relative recovery error, by swept axis:

| policy | `D → D_eff` | `R → R_eff` | `ρ_C → rho_c_glue` |
|---|---|---|---|
| `"all"` (literal §B.3 reading) | 10.3% | 7.0% | **3.7%** |
| `"active"` (condition on activity) | **9.6%** | **6.4%** | 8.6% |

Stable under `n_t = 1000` (10.7 / 7.0 / 4.6 vs 9.8 / 5.8 / 9.2).

**Decision: `"all"`, on evidence.** `"active"` is very slightly better on `D` and
`R` — under a percentage point, comparable to seed noise — while `"all"` is better
on `ρ_C` by more than a factor of two, consistently at both `n_t`. The `ρ_C` axis
is also the most direct test of the three, since the generated correlation is
exactly known. `"active"` systematically over-estimates `ρ_c` (0.043 / 0.239 /
0.434 against a truth of 0 / 0.2 / 0.4), which is precisely the measure H1d turns
on.

Conclusions about `D` and `R` are robust to the choice; conclusions about `ρ_c`
are not. The policy is recorded in every result key either way.

---

## 5. Measured limitation — scale compression, and its mechanism

`D_eff` and `R_eff` over-report small ground-truth values and under-report large
ones, crossing over near `D ≈ 4`, `R ≈ 1`. At `D = 10` the estimate is 18% low; at
`R = 2.0`, 11% low. We chased this to a mechanism rather than reporting it as an
unexplained caveat. `scripts/run_scale_compression.py` →
`results/scale_compression.json`.

**Not Monte-Carlo error over `(y, t)`.** Raising `n_t` from 200 to 1000 moves
`D_eff` at `D = 10` from 8.17 to 8.19 and `R_eff` at `R = 2` from 1.773 to 1.785.
The gap does not close.

**Not an artifact of B.5's small `P`.** Mean absolute recovery error on `D`:
13.1% at `P = 2`, 14.4% at `P = 8`, 16.5% at `P = 16` (`M = 100`). Nearly flat.

**It is finite sampling of each manifold.** `D_eff/D` at `P = 2`, `N = 1000`:

| `M` | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|
| `D = 2` | 1.112 | 1.118 | 1.123 | 1.127 | 1.130 |
| `D = 6` | 0.846 | 0.898 | 0.934 | 0.955 | 0.969 |
| `D = 10` | 0.671 | 0.750 | 0.817 | 0.865 | 0.897 |

The deficit at `D = 10` falls as roughly `M^-0.45` and is converging to 1. Anchors
are extreme points of a finite sample, and a finite sample under-represents the
extent of a high-dimensional manifold; the higher `D`, the more points are needed
to reach the same relative coverage. The small over-report at `D = 2` is a
separate, `M`-independent effect of about 12%.

**What this means for the paper.**

1. The bias is a **monotone increasing** function of the ground truth, so rank
   ordering and the sign of every effect are preserved.
2. It is a function of `(D, M)`. `M` is held fixed across all conditions, so the
   distortion is a fixed transform, not a between-condition artifact.
3. It **compresses** the dynamic range, so estimated `Δ log D_eff` is biased
   *toward zero*. In the §8 attribution the dimension channel is therefore
   **conservative**: a real contribution can be understated, not manufactured.
4. It must nonetheless be stated, because our streams change representational
   dimension: a bias correlated with a manipulated variable is a confound, not a
   nuisance. Any absolute `D_eff` value should be read as a lower bound above
   `D ≈ 4`.

Reaching a 5% deficit at `D = 10` would need `M ≈ 4000`, which is not affordable
at Phase-1 scale — hence a stated caveat rather than a fix.

---

## 6. Reproducing everything here

```
scripts/run_glue_core_recovery.py   -> results/glue_core_recovery.json   (§2, §4)
scripts/run_gate_2a.py              -> results/gate_2a.json              (§3)
scripts/run_scale_compression.py    -> results/scale_compression.json    (§5)
scripts/run_psi_eff_diagnostic.py   -> results/psi_eff_diagnostic.json   (02 §2c)
scripts/run_cost_model.py           -> results/cost_model.json
pytest tests/test_glue_core.py tests/test_glue_core_recovery.py          (§1)
```

Estimator string `glue_core@<sha>`; the `02` §9 pooling firewall applies between
it, `α_sim`, and `α_mf` pairwise.
