# 02 — Validation Suite

**A component is not done when it runs. It is done when its test here passes.**

Geometry estimators fail silently: they return well-formatted, plausible numbers
that are wrong. Every test below exists because a specific silent failure is
possible.

Tests are ordered by dependency. **§1 and §2 are hard gates** — do not build on an
unvalidated capacity estimator.

---

## 1. Manifold generator — ground-truth recovery

**Gate. Must pass before any GLUE work.**

Source: Chou et al. (ICML 2025) Appendix B.5, Figures 9–12. Their stated settings:

```
N = 1000 (ambient), P = 2, M = 200
D_ground from 2 to 10
R_ground from 0.8 to 2.0
ρ_c, ρ_a, ψ from 0 to 0.8
```

| Test | Assertion |
|---|---|
| `test_dimension_recovery` | `D_mf` increases monotonically with `D_ground` over 2→10 |
| `test_radius_recovery` | `R_mf` increases monotonically with `R_ground` over 0.8→2.0 |
| `test_center_alignment_recovery` | `ρ_c` (estimated) increases monotonically with `ρ_C` (generator) over 0→0.8 |
| `test_axis_alignment_recovery` | `ρ_a` (estimated) increases monotonically with `ρ_A` (generator) over 0→0.8 |
| `test_capacity_direction_D` | capacity **decreases** as `D_ground` increases |
| `test_capacity_direction_R` | capacity **decreases** as `R_ground` increases |
| `test_capacity_direction_rho_c` | capacity **decreases** as center correlation increases |
| `test_capacity_direction_rho_a` | capacity **increases** as axis correlation increases |
| `test_unit_normalization` | pre-scaled manifold points have unit norm before `R` scaling |

The last four are sign checks and are the cheapest way to catch a transcription
error in Step 4 of the algorithm. Note `ρ_a` and `ρ_c` move capacity in
**opposite** directions.

If monotonicity fails, the estimator is wrong. Do not proceed, do not "adjust
tolerances."

---

## 2. Capacity fidelity — one gate (mean-field validity)

The three-factor decomposition `α = Ψ_eff·(1 + R_eff⁻²)/D_eff` is an **exact
identity** (`00` §6.2, ICLR 2026 §B.3), so there is no approximation-fidelity gap
to test — the former **§2b is deleted**. Two capacity quantities remain, with one
real error source between them:

| Quantity | Source | Error it exposes |
|---|---|---|
| `α_sim` | simulation, bisection on N (ground truth) | — |
| `α_mf` | replica mean-field estimate (§6.1) | **does mean-field theory hold at our P, M, N, and tilt β?** |

**α_sim source (verified):** `third_party/correlated_capacity`
`capacity/manifold_simcap_analysis.py::manifold_simcap_analysis(XtotT, n_rep, seed)`
computes **general point-cloud** simulation capacity (SVM separability at κ=0,
bisection on feature dimension) — confirmed to accept arbitrary `(N, P_i)` point
clouds, so it is valid at project parameters. `α_sim`, `α_mf`, and any GLUE number
are **different estimators**: the §9 pooling firewall applies pairwise
(`{repo}@{sha}` in the result key). Patch `manifold_simcap_analysis` for RNG
isolation first — it calls global `np.random.seed` (see `third_party/VENDORED.md`).

### §2a — Mean-field validity (the sole capacity gate)

**Gate.**

```
test_mean_field_validity:
    at project parameters (d=150, P=16, M=150, N=300, D and R in range),
    over the tilt axis  β ∈ {0, 0.5, 1, 2, 4, ∞}   (β=∞ ≡ retained, y fixed at y_j)
    compare  α_sim  (manifold_simcap_analysis, ground truth)
    against  α_mf   (replicaMFT manifold_analysis_corr, §6.1)
    record mean and max relative error per β -> results/mft_validity.json
```

The **β axis is load-bearing, not cosmetic.** `β = 0` validates generic capacity;
**`β = ∞` validates retained capacity, which is what the §8 money figure runs on.**
The usable β range (where `α_sim ≈ α_mf`) is the headline output of this gate — it
decides whether retained/tilted capacity run on public code (wide range) or Figure
3 waits on GLUE (narrow). This is also where `docs/reference/glue-refinements.md`'s
A2 concern — mean-field accuracy under a degenerate, fixed-label ensemble — is
**empirically settled** (that concern and 04 §C4 are compatible: C4 is about the
estimator's *definition*, A2 about mean-field *accuracy*; this sweep measures the
latter).

If `α_sim` vs `α_mf` disagree substantially at a given β, mean-field theory does
not hold there and **no downstream capacity claim at that β is safe** — this is a
stop, not a caption change. See also the P×N sweep in `01-experiments.md` Phase 0
(`test_mean_field_validity_at_project_P`), which decides whether `(P, N)` are large
enough *before* anything is built.

**Conventions the comparison must fix (verified against vendored code):**

- **Margin.** Both sides at **κ = 0**. `α_mf` (replicaMFT) is called with `kappa=0`;
  `α_sim` (`manifold_simcap_analysis` → `check_data_separability_general`) hard-codes
  `kappa = 0`. If either side were run at κ ≠ 0 the gate compares different
  quantities and reports a mean-field failure that isn't one — assert κ=0 on both.
- **What is held fixed.** Both estimators receive the **identical** per-manifold
  point clouds (same `P`, `M`, same ambient `N`). `α_sim` finds critical capacity by
  **bisecting the random-projection dimension at fixed `(P, M)`** (`α_sim = P / N_c`);
  it does **not** vary `P`. `α_mf` is the replica estimate on the same clouds.
  Compare the two capacity ratios directly.
- **RNG.** `manifold_simcap_analysis` calls global `np.random.seed` — must be run
  through the `src/glue/adapters/` RNG-injection wrapper (see `third_party/VENDORED.md`),
  not called directly, or the noise floor is not reproducible.

*(§2b — approximation fidelity — deleted 2026-08-10: the decomposition is an exact
identity, so there is no `(1+R⁻²)/D` approximation to test. See `00` §6.2 and 04
§C1. `results/approximation_fidelity.json` is no longer produced.)*

### §2c — Ψ_eff identity check (GLUE-dependent, per 05 §3.3)

Not a mean-field gate — a **transcription check** on the adapter. The adapter
derives `Ψ_eff = α · D_eff / (1 + R_eff⁻²)` from `replicaMFT` outputs (`00` §6.1),
which is exact *by construction* and therefore cannot catch an error. Once the
anchor points are exposed (GLUE access, or the vendored-QP change gated on Kati —
`05` §2.2), compute `Ψ_eff` **independently** as `E[c]/E[a]` from §6.1 `a/b/c` and
assert it equals the identity-derived value to float64 tolerance.

```
test_psi_eff_identity:   # requires a/b/c exposure; skipped until then
    assert  E[c]/E[a]  ==  alpha * D_eff / (1 + R_eff**-2)   (float64 tol)
```

Until it runs, `Ψ_eff` carries `derived_via_identity=True` and `§8` attribution
may fall back to two factors (`00` §8).

---

## 3. Parameterization orthogonality

**Gate for the whole (a, γ) factorial design.**

```
test_hidden_init_identical_across_gamma:
    for gamma in [0.01, 0.1, 1.0, 10.0]:
        build model with same seed
        assert W_hidden is bitwise identical across all gamma
```

```
test_base_width_equivalence:
    assert  muP(gamma_0=1, N=64)  ≡  NTP(N=64)
    numerically, to float64 tolerance, on forward pass and first gradient step
```

The second test resolves the **UNCERTAIN** base-width normalization constant
(spec §4.2). It must be derived, not guessed; write the derivation to
`docs/reference/parameterization-derivation.md` when it passes.

```
test_lr_scales_with_gamma:
    assert per-module lr == lr0 * gamma_m**2 * N        (quadratic mode)
    assert corrected mode uses gamma**(2/L) for gamma > 1
    assert the same lr applies to BOTH W_m and u_m
```

```
test_output_zero_at_init:
    assert f(x; theta_0) == 0 exactly, for all gamma
```

---

## 4. Alignment knob (Gate 2 — open empirical question)

```
test_alignment_preserves_norm_and_rank:
    for a in linspace(0, 1, 11):
        assert ||W(a)||_F == ||W(0)||_F        (float64 tolerance)
        assert rank(W(a)) == rank(W(0))
```

```
test_grassmann_endpoints:
    assert subspace(Y(0)) == subspace(V_k)
    assert subspace(Y(1)) == subspace(U_C)
    assert principal angles are monotone in a
```

```
test_alignment_moves_initial_capacity:      # GATE 2 — may fail
    sweep a in linspace(0, 1, 7)
    assert alpha_generic at init is monotone in a
    assert range(alpha_generic) >= 2 * identifiability_noise_floor
```

**Gate 2 is expected to be the most likely failure in the project.** If
`test_alignment_moves_initial_capacity` fails, do not attempt to fix the geodesic.
Switch `wealth_knob` to `"input_dim"` and record the loss of orthogonality with γ
as a known limitation.

---

## 5. Stream generator

```
test_dichotomies_balanced:          all sampled dichotomies sum to zero
test_hamming_distances_even:        all pairwise distances between balanced dichotomies are even
test_readout_similarity_target:     s_r(t, t-1) matches config within one Hamming step
test_sign_symmetry:                 s_r(y, y) == s_r(y, -y) == 1.0
test_feature_similarity_target:     realized center correlation matches rho_C target
test_probe_never_trained:           y* does not appear in any training task, any stream
test_similarity_matrices_recorded:  full T×T S_f and S_r written for every stream
test_stream_reproducible:           same stream_id + seed -> identical stream
```

---

## 6. Noise floors

**Required before any effect is interpreted. Not optional, not deferrable.**

For every measure in {`α_generic`, `α_retained`, `Ψ_eff`, `R_eff`, `D_eff`,
`rho_c_glue`, `rho_c_signed`, `ρ_a`, `ψ`, `PR`, `CKA`, `CCGP`, `shattering`,
rotation, `ΔPR`, reuse}:

```
identifiability_floor:
    15 seeds, SAME condition, SAME timepoint
    -> spread of the measure across seeds

trajectory_floor:
    same seed, same config, ONLY data order varies
    -> spread of the measure
```

Then:

```
minimum_detectable_effect(measure) = f(identifiability_floor, n_seeds_planned)
```

Write all of it to `results/noise_floors.json`.

**Rules that follow:**
- Any reported effect must exceed **both** floors.
- Any measure whose MDE exceeds its expected effect is **demoted from
  confirmatory to exploratory** in `01-experiments.md` §5 *before* thresholds are
  written.
- `R_eff` and `ψ` are expected to be the noisiest. Plan for them failing.

---

## 7. Training loop

```
test_no_weight_decay:               optimizer has weight_decay == 0
test_no_normalization_layers:       model contains no BatchNorm/LayerNorm/RMSNorm
test_optimizer_is_sgd:              no Adam/RMSProp/momentum in core configs
test_activation_is_relu:            no tanh/sigmoid/GELU
test_no_gradient_clipping:          no clip_grad_* call in the core path
test_single_readout:                exactly one readout; no task-ID input reaches the model
test_matched_loss_stopping:         under stopping="matched_loss", all conditions terminate within tolerance of target_loss
```

These are guardrail tests. They exist because these are exactly the changes an
agent is most likely to make as an "improvement."

---

## 8. Manipulation checks (runtime assertions, not unit tests)

Logged every boundary; violations raise a **warning in the results record**, not
an exception — a run that loses its manipulation mid-stream is still data.

| Check | Threshold |
|---|---|
| `‖ΔW_m‖/‖W_m‖` separation across γ | ≥ 1 order of magnitude; warn if it collapses below 3× |
| per-module gradient-norm share | warn if any module < 5% for 3 consecutive boundaries (gradient starvation) |
| per-module output-variance share | warn if any module < 5% (module effectively dead) |
| capacity-at-init vs `a` | warn if non-monotone |
| pseudo-inverse effective rank | log always; warn on rank deficiency > 20% |

---

## 9. Reproducibility

```
test_rng_isolation:
    changing rng_data_order must NOT change the initialization
    changing rng_init_shape must NOT change the manifold arrangement
    changing rng_stream must NOT change either
```

```
test_paired_init:
    for fixed seed, W_tilde is identical across all configs in that seed
    conditions differ only by the scalars under test
```

```
test_shared_streams:
    for fixed stream_id, the dichotomy sequence is identical across configs
```

```
test_artifact_completeness:
    every result file records: resolved config, git SHA, estimator version,
    full RNG seed state, noise-floor version
```

**Pooling rule:** geometry numbers computed under different estimator versions
must **never** be pooled. The estimator version is part of the result key.

---

## 10. Definition of done (per component)

1. Its test section above passes
2. Its noise floor is **measured**, not assumed
3. Its cost at target parameters is recorded in `results/cost_model.json`

"It runs without erroring" is not done.
