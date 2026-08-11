# 01 — Experiment Registry, Phases, and Pre-Registration

**Project rule: decisions are pre-registered before compute is committed.**
Adding an experiment means adding it here first, with its outcome thresholds,
*then* running it. An agent asked to run something not registered here should ask
whether it should be registered.

---

## 1. Configuration schema

Every run is fully determined by one config. No hidden defaults in function
bodies. Fields:

```yaml
# --- identity ---
name: str
phase: int
registered: bool              # must be true to run

# --- manifolds ---
d: 150
P: 16
M: 150
D: int                        # intrinsic dimension
R: float                      # radius
rho_A: 0.0
psi_generator: 0.0
factor_structure: "labeling_only"     # do not change

# --- stream ---
T: 16
feature_similarity: float     # s_f(t, t-1)
readout_similarity: float     # s_r(t, t-1)
dichotomy_family: "random_balanced" | "single_factor" | "xor" | "mixed"
probe_dichotomy: true
stream_id: int                # shared across configs — blocking factor

# --- architecture ---
n_modules: 1 | 2
N: 300
gamma: [float, ...]           # per module
alignment: [float, ...]       # per module
wealth_knob: "alignment" | "input_dim"
frozen_modules: [bool, ...]
lr_scaling: "quadratic" | "corrected"
readout_init: "zero"          # do not change

# --- training ---
optimizer: "sgd"              # do not change
lr0: float
epochs_per_task: int
stopping: "matched_loss" | "fixed_steps"
target_loss: float            # if matched_loss
joint_training: false         # true = i.i.d. control

# --- evaluation ---
tier2_interval: 4
n_t: 200
kappa: 0.0                    # do not change
estimation_mode: "pairwise" | "full_P"   # pairwise = lab standard (2 manifolds/QP); full_P = all P jointly. Compared in Phase 0
capacity_modes: ["raw", "gaussianized"]  # gaussianized = Wakhloo 2023 preprocessing (00 §6.3)
rho_convention: "both"       # MANDATORY — both always computed; no path may select glue_abs or signed_normalized alone. rho_c_glue = reporting/comparability; rho_c_signed = H1d instrument (03 §E.4)
beta: 0.0                    # tilt for tilted_capacity: 0 = generic, ∞ = retained (00 §7)
ccgp_enabled: true           # per-family CCGP (00 §10)

# --- reproducibility ---
seed: int
```

**Seeding.** Four independent generators derived from `seed`, never shared:
`rng_arrangement`, `rng_init_shape`, `rng_stream`, `rng_data_order`.
The paired-initialization design requires holding some fixed while varying others.

**Paired initialization.** One shape matrix `W̃` drawn per `seed` from
`rng_init_shape` and reused across **all** configs at that seed, scaled and
rotated per condition. Conditions differ only by the scalars under test.

**Shared streams.** Stream realizations generated once from `rng_stream` keyed by
`stream_id`, reused across configs. Stream identity is a blocking factor in the
statistical model, not residual noise.

---

## 2. Architecture configurations

All at matched total width `2N` and matched parameter count.

| ID | Modules | γ | a | Purpose / rules out |
|---|---|---|---|---|
| `C1` | 1 (width 2N) | swept | swept | dense baseline; recovers γ₀\* |
| `C2` | 2 | `[γ, γ]` | `[a, a]` | **primary control** — isolates modularity per se |
| `C3` | 2 | `[γ_lo, γ_hi]` | `[a_hi, a_lo]` | **treatment** — wealthy-lazy + poor-rich |
| `C4` | 2 | `[γ_hi, γ_lo]` | `[a_hi, a_lo]` | anti-diagonal — does the *specific* pairing matter? |
| `C5` | 2 | `[γ, —]` | `[a, a]` | module B frozen — lazy dynamics vs. static random features |
| `C6` | any of above | — | — | **`joint_training: true`** — the sequentiality control |

`C6` is the headline control. Required pattern: heterogeneity helps sequentially
and **not** jointly, on identical data. If it helps jointly, the effect is
ensembling.

`C4` is easy to forget and important: if the anti-diagonal works as well as the
diagonal, the story about *which* geometry pairs with *which* update rule is
wrong, and the finding is generic diversity.

---

## 3. Stream conditions

### 3.1 Primary — Hiratani 2×2

| ID | `feature_similarity` | `readout_similarity` | Prediction |
|---|---|---|---|
| `S-HH` | high (0.9) | high (0.9) | benign |
| `S-HL` | high (0.9) | low (0.1) | **catastrophic corner** — largest rotation/overwrite, largest ρ_c increase |
| `S-LH` | low (0.1) | high (0.9) | relatively benign |
| `S-LL` | low (0.1) | low (0.1) | intermediate |

### 3.2 Secondary

| ID | Description |
|---|---|
| `S-fixed-r` | `s_f = 1.0` (arrangement frozen), `s_r ∈ {0.1, 0.5, 0.9}` — isolated readout axis |
| `S-blocked` / `S-shuffled` | matched marginals, run-ordered vs shuffled — **deferred to full paper** |

### 3.3 Naturalistic confirmation

Split-CIFAR100, 10 tasks × 10 classes (`P = 10` per task).
**Not** Split-CIFAR10 or Split-MNIST: at `P = 2` there is only one balanced
dichotomy (up to sign), so the factorial/dichotomy design collapses and the
generic/retained distinction has nothing to range over. (This is a
*task-structure* argument, not an estimability one — the lab estimates GLUE
pairwise at `P = 2` routinely; see `PROJECT.md` §2 P1 and 04 §C6.)

---

## 4. Phases

### Phase 0 — Manipulation validation (~2 days)

| Check | Pass criterion | Failure action |
|---|---|---|
| Capacity-at-init tracks `a` | monotone, range ≥ 2× noise floor | fall back to `wealth_knob: input_dim` |
| Capacity-at-init flat in `γ` | variation within noise floor | parameterization bug — **stop** |
| `‖ΔW_m‖/‖W_m‖` separates across γ | ≥ 1 order of magnitude | parameterization bug — **stop** |
| **Estimation-mode comparison** (spec §13) | `pairwise` vs `full_P` agree within noise floor on {α, R_eff, D_eff, ρ_c} | if they diverge, report both — `pairwise` is comparable-to-published (lab standard), `full_P` is primary |
| **Time-reparameterization test** (spec §12) | residual after warping **exceeds** noise floor | if trajectories coincide, **H3 is dead** — restrict contrasts to γ≪1 vs γ~1, proceed with H1/H2 only |

#### Phase 0 precondition — mean-field validity at project `P` (decide **before** building)

Capacity theory is a thermodynamic-limit result in **both** `P` and `N`. `P = 16`
was chosen for GLUE estimability (vs Split-CIFAR10's `P = 2`) and four binary
factors — "better than 2" is **not** "large enough". The `02-validation-suite.md`
§1 recovery tests run at `P = 2` and say nothing about capacity accuracy at
`P = 16`. `α_sim` (now vendored, general point-cloud) makes this checkable cheaply.
Sweep the two limit knobs jointly:

```
test_mean_field_validity_at_project_P:
    sweep P in {8, 16, 32} x N in {300, 600, 1200}   (M fixed)
    compare alpha_sim vs alpha_mf   at each (P, N)   # generic (beta=0) only
    compare alpha_sim vs alpha_core at each (P, N)   # generic + retained
    record relative error surfaces -> results/mft_validity.json
```

**Fix the first lever correctly.** `N` (module width) is nearly free — no design
commitments attached; `P` is expensive — it is tied to the factorial design
(4 factors → 16, 5 → 32), and changing it changes the dichotomy families, CCGP
structure, single-factor/XOR counts, and every similarity calculation.

**RESOLVED 2026-08-11 — `P = 16, N = 300` stands; escalation not triggered.**
`results/gate_2a.json` (`scripts/run_gate_2a.py`, `P × N`, 3 seeds, M=60, n_t=200):
at the design point `α_core` is within **3.4%** of `α_sim` and `α_mf` within
**1.6%** — the best mean-field agreement anywhere in the sweep, and inside
`α_sim`'s own seed scatter (±0.017). `α_core` varies < 2% across `N` at fixed
geometry while `α_sim` itself drifts upward with ambient `N`, so the residual is
at least partly the simulation estimator's own bias. **The five-factor `P = 32`
redesign is off the table**; the ladder below is retained as the record of what
would have happened. See `docs/reference/glue-core-validation.md` §3.

The pre-specified order, had §2a failed at `(P=16, N=300)`:

1. **Raise `N`** (300 → 600 → 1200). Costs compute and nothing else. Prefer this.
2. If capacity accuracy is still marginal but `N` cannot go higher, **accept and
   report capacity as approximate** (feeds the two-factor attribution fallback,
   `00` §8 — a caption caveat, since the decomposition itself is exact).
3. **Raise `P` (→ 5 factors, `P = 32`) only as a last resort.** Design cost:
   dichotomy families become 5 single-factor + 10 XOR, `f: [P] → {0,1}^5`, and every
   `P`-dependent number in `00`/`01` shifts.

This is a scientific call for the human; the sweep produces the evidence. Note it
runs on `replicaMFT` + `correlated_capacity` + our own `src/glue/core.py` alone —
**no GLUE dependency**.

**`α_mf` is generic-only (verified 2026-08-11, `00` §6.1).** `replicaMFT`'s
`manifold_analysis_corr` takes no `y`; the replica derivation integrates the
dichotomy average out analytically, so it is **label-invariant by construction**
and can never yield retained or tilted capacity. It is a cross-check on generic
`α`, nothing more. Retained/tilted capacity and the whole three-factor
decomposition route through `α_sim` and `α_core` (`src/glue/core.py`,
estimator string `glue_core@<sha>`, subject to the same §9 pooling firewall).

**α aggregation is pinned to the harmonic mean** `1/mean(1/α_i)` wherever a
per-manifold `α` vector is reduced (`α = P/N_crit`; critical dimensions add).
Arithmetic means of `α_i` are a bug, not a convention choice.

### Phase 1 — H1, H2, H6 (homogeneous only)

Grid: `γ` sweep × `a` sweep × Hiratani 2×2 streams, `C1`/`C2`, 8 seeds, 20 streams.

Outputs: forgetting-attribution table (exact three-factor, log-space);
generic/retained crossing; probe-dichotomy
validation of generic capacity; KTA measured alongside; **Wakhloo/Slatton four
statistics** (dimensionality, total correlation, signal–signal factorization,
signal–noise factorization) measured alongside as the sharpened Gate-3 competitor
to generic capacity. Measure them regardless of the RQ2-reframe decision; their
exact definitions are pending human transcription to
`docs/reference/optimal-coding-statistics.md` and must not be wired into the Gate-3
comparison until verified.

**This phase alone is a complete NeurReps abstract.** No heterogeneity claims
required. Do not gate submission on Phase 2.

Kill criteria:
- GLUE attributions do not separate by regime → fall back to the rotation/expansion decomposition (spec §9), which is far more robust
- Generic capacity does not track the probe metric → report as a negative finding about label-agnostic geometry measures

### Phase 2 — H3, H5 (heterogeneous)

All of `C1`–`C6`, sequential and joint, paired inits, shared streams.

Kill criteria:
- Pair mean-γ predicts performance (H5) → the effect is a reparameterization
- Joint training shows the same effect → the effect is ensembling

### Phase 3 — H4 (regime half-life)

`T = 40` at three similarity levels. Track per-module regime over the stream via
the §11 manipulation checks. Measure the half-life of the initial regime
difference.

Confound to address: loss of plasticity may be an artifact of *abrupt*
environment change and largely mitigated under gradual change. Include a
gradual-change arm (`s_f`, `s_r` drifting smoothly) as a control.

### Phase 4 — Stream variance and predictability

Deferred to the full paper. Registered here so it is not silently re-scoped in.

---

## 5. Pre-registration table

Fill thresholds **after** the noise-floor measurement (`02-validation-suite.md` §6),
not before. A threshold below a measure's minimum detectable effect is not a
prediction.

| ID | Hypothesis | Measure | Direction | Threshold | Disconfirming observation |
|---|---|---|---|---|---|
| H1a | Rich forgetting is radius/utility-accounted | share of `Δ log α` from `Δ log(1+R_eff⁻²)` and `Δ log Ψ_eff` | rich > lazy | _TBD_ | shares equal within noise, or lazy > rich |
| H1b | Lazy forgetting is ρ_c-accounted | `Δ rho_c_glue` (covariate; `rho_c_signed` for direction) | lazy > rich | _TBD_ | reversed or null |
| H1c | Forgetting ≡ ultra-rich OOD signature | sign pattern over (R, ψ, D) | matches Chou Fig 7c | qualitative match on all 3 | any sign mismatch |
| H1d | Progressive center-decorrelation (Menghi) | **`rho_c_signed` trajectory** (NOT `rho_c_glue` — the absolute value erases the sign and makes H1d untestable; see `03` §E.4) | decreasing over stream, faster in rich | _TBD_ | flat or increasing |
| H2a | Generic capacity falls with γ | `α_generic` vs `γ` | monotone decreasing | _TBD_ | non-monotone or flat |
| H2b | Retained capacity rises with γ | `α_retained` vs `γ` | monotone increasing | _TBD_ | non-monotone or flat |
| H2c | Crossing predicts γ\* | argmax(`α_generic × α_retained`) | = argmin(average error) | within 1 grid step | separated by >1 grid step |
| H2d | **Generic capacity is label-valid** | corr(`α_generic`, probe metric) | positive | _TBD_ | null or negative → reportable negative result |
| H3a | Division of labour | `α_generic`(wealthy-lazy) > `α_generic`(poor-rich) | as stated | _TBD_ | equal or reversed |
| H3b | Reliance shifts with similarity | output-variance share vs `s_r(t,t−1)` | rich share ↑ as `s_r` ↓ | _TBD_ | flat |
| H4 | Regime erodes | `‖ΔW_m‖` separation vs stream position | decreasing; half-life shorter at high `s_f` | _TBD_ | separation persists to T=40 |
| H5 | γ_eff insufficiency | residual after regressing outcome on pair-mean γ | non-zero | _TBD_ | pair-mean γ explains everything |
| H6 | Catastrophic corner is geometrically worst | rotation, `Δρ_c` across S-HH/HL/LH/LL | max at `S-HL` | _TBD_ | max elsewhere |

---

## 6. Statistical model

```
outcome ~ alignment * gamma * feature_similarity * readout_similarity
          + (1 | seed) + (1 | stream_id)
```

- Paired differences per seed (enabled by shared `W̃`), not difference of means
- Effect sizes with intervals; **no p-values at small seed counts**
- Long-range similarity `s_f(t,j)`, `s_r(t,j)` enter drift regressions as covariates

**Budget rule:** for stream-level hypotheses, more streams beats more seeds.
~10 seeds × 20 streams over 40 seeds × 5 streams.

---

## 7. Figure targets

The abstract is 4 pages ≈ 4 figures. Every registered experiment should map to one.

| Fig | Content | Depends on |
|---|---|---|
| 1 | Setup + `a`/`γ` orthogonality demonstration | Phase 0 |
| 2 | **Forgetting attribution**, lazy vs rich, with Chou Fig 7c overlay | Phase 1 |
| 3 | Generic vs retained capacity vs γ, with probe metric overlaid | Phase 1 |
| 4a | Heterogeneous vs homogeneous on the front + γ_eff control | Phase 2 |
| 4b | *(fallback if Phase 2 fails)* regime half-life | Phase 3 |

Both versions of Figure 4 are coherent papers. Do not treat 4b as a failure state.
