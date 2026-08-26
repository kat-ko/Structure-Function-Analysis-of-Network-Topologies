# toy_task — Implementation Plan

A synthetic continual-learning benchmark for studying how **structural priors**
(dense vs. modular RNN) and **learning regime** (rich vs. lazy, via init scale γ)
shape **representational organization** as **task similarity** `s` varies, under a
fixed `T(0) → T(s) → T(0)` (A1 → B → A2) curriculum.

This plan is the single source of truth for the build. It consolidates
`description/part_1..5.md` and applies `description/benchmark_specification_patch.md`,
which is **authoritative wherever it overlaps the parts**. It also pins the
"follows the previous protocol" references to concrete implementations already in
this repository so the toy_task results are methodologically comparable.

---

## 0. Causal question and scope

```
Architecture × Learning Regime × Task Similarity
        ↓
Representational Organization        (primary scientific object)
        ↓
Transfer / Interference              (behavioral consequences)
```

The benchmark is deliberately minimal. The hidden representation — not final task
loss — is the primary object of study. Everything except the task mapping `T(s)`
is held fixed within a seed. `toy_task` stays **standalone**: it does not import
`a1b2_modular`, `dynamics_of_specialization`, `transfer-interference`, or `gecco`.
Where a formula is borrowed, it is **re-implemented locally** (the reference is
cited for parity, not imported).

---

## 1. Frozen benchmark constants (patch §1, §3, §6)

These are **fixed** for the baseline and are NOT swept:

| Symbol | Meaning | Value |
|--------|---------|-------|
| `N` | number of persistent objects | `8` |
| `d_z` | latent dimensionality | `4` |
| `d_x` | observation dimensionality | `8` |
| `d_y` | output dimensionality | `2` |
| `n_modules` | modules in the modular network | `2` |
| inter-module comms | recurrent coupling between modules | **none** (omitted entirely; see §3.2) |
| `seq_len` | repeated-input timesteps per presentation | `3` |
| `epochs_per_phase` | fixed epoch budget per phase | `100` |
| `sigma_train` | observation noise std (training) | `0.05` |
| `sigma_eval` | observation noise std (evaluation) | `0.0` |
| `lr` | SGD learning rate | `0.01` |
| `momentum` | SGD momentum | `0.0` |
| `weight_decay` | SGD weight decay | `0.0` |
| `batch_size` | minibatch size | `1` (one update per object presentation) |

**Swept (experimental) variables:**

- **Architecture** ∈ {`dense`, `modular_shared`, `modular_feature_routed`,
  `modular_task_routed`} (see §3). `modular_task_routed` has an `off_module_policy`
  ∈ {`freeze`, `readout_coord`}; both are run by default (5 effective configs).
- **Hidden size** `H` ∈ total-unit grid (default `{12, 24, 50, 100}`; see §3.5).
  *(Promoted from frozen to swept per project decision; overrides the patch §6
  fixed `H=50`, which becomes one point on the grid.)*
- **Init scale** `γ` ∈ rich→lazy ladder (default `{0.001, 0.01, 0.1, 1.0, 2.0}`; see §6)
- **Task similarity** `s` ∈ `{0, π/12, π/6, π/4, π/3, π/2, π}` (patch §3)
- **Seed** ∈ default `range(10)` (each seed = one independent environment)

Per-phase update accounting (patch §4): 8 updates/epoch × 100 epochs = 800
updates/phase → **2400 updates per A1→B→A2 run**.

> **No inter-module communication.** Unlike `a1b2_modular` (whose `Community`
> carries a sparse `comms` pathway controlled by `sparsity`), toy_task omits the
> communication pathway altogether from the outset. The modular network is two
> independent recurrent cores with a shared readout — equivalent to
> `a1b2_modular`'s `sparsity=0` / no-comms case, hard-coded for simplicity and
> zero cross-project dependency.

---

## 2. Environment (Part II; patch §2, §3) — `toy_task/environment.py`

A seed fully determines one environment. All RNG flows from a single
`numpy.random.Generator(seed)` plus a paired `torch.Generator` for model init, so
runs are bit-reproducible.

### 2.1 Latent identities
`z_i ∼ N(0, I_{d_z})`, generated once per seed → `Z` shape `(N, d_z) = (8, 4)`.

### 2.2 Observation function (patch §2 supersedes Part II §6)
```
x = tanh(W z + b) + ε
W ∈ ℝ^{8×4},  W_ij ∼ N(0, 1/d_z) = N(0, 0.25)
b = 0   (zero vector, fixed)
ε ∼ N(0, σ² I_{d_x})
```
`W` is drawn once per seed and never trained. Provide:
- `clean_observation(Z) -> (N, d_x)`  uses σ = 0 (used for all evaluation/representation extraction).
- `noisy_observation(Z, sigma_train) -> (N, d_x)`  fresh ε per call (training).

> Note: this resolves the Part II §6 vs patch §2 conflict in favor of the patch
> (explicit `1/d_z` weight variance, `b=0`, noise folded into the observation).

### 2.3 Targets and task family (patch §3)
```
θ_i = 2π (i-1) / N          → 0,45,...,315°   (i = 1..N)
T(0):  y_i = (cos θ_i, sin θ_i)
T(s):  y_i = (cos(θ_i + s), sin(θ_i + s))
```
`targets(s) -> (N, d_y)`. The rotation is rigid: object-to-object geometry is
invariant; only the absolute mapping rotates by `s`.

### 2.4 Sampling (Part III §5–6; patch §4)
One epoch = one randomized permutation of the 8 objects, one fresh noisy
observation each, one SGD update each. New permutation every epoch. Balanced
exposure; no curriculum.

### 2.5 `Environment` dataclass
Bundles: `seed, Z (8×4), W (8×4), b, sigma_train`; methods `clean_observation()`,
`noisy_observation()`, `targets(s)`, `epoch_batches(rng)`. Serializable to
`config + .npz` for reproducibility (Part IV §17, Part V §7).

### Stage-1 validation (Part V §2, no NN)
- Scatter/PCA of `Z`, of clean `X`, and of `targets(s)` for each `s`.
- Assert `Z`, `W`, `b`, clean `X` are identical across phases and across `s`.
- Assert only targets change with `s`; check `‖y_i(s)‖ = 1` and that the angular
  gap between consecutive objects stays `2π/N` for every `s`.

---

## 3. Models & input feeding (Part III §11–15; patch §6) — `toy_task/models.py`

### 3.0 How a single presentation is fed to the network

One **object presentation** = one observation vector `x ∈ ℝ^{d_x}` (`d_x = 8`),
noisy in training (σ=0.05) and clean in evaluation (σ=0). Feeding rules, identical
for every architecture (Part III §14–15; patch §6):

1. **Hidden state reset to zero** at the start of every presentation. Recurrence
   operates *within* a presentation, never across objects; continual learning
   happens through weight updates, not carried-over hidden state.
2. **Temporal unrolling:** the *same* `x` is fed for `seq_len = 3` consecutive
   timesteps (`x → x → x`).
3. **Readout at the final timestep only.** MSE loss is computed on the step-3
   hidden state's readout vs. the `(cosθ, sinθ)` target.

```
x (d_x)  ──repeat──►  [t1: x] ─► [t2: x] ─► [t3: x] ─► h_3 ─► Linear ─► ŷ (d_y)
                       (hidden reset to 0 before t1; loss only at t3)
```

The network **never** receives latent variables, object indices, or task/phase
identifiers (Part III §9). Task identity must be inferred purely from supervision.

What differs between architectures is **only how the `d_x` features are wired into
the recurrent core(s)** — captured by an `input_mask` exactly as in
`a1b2_modular`'s `Community` (block-diagonal `state_mask` when modules are
separated, all-ones when input is shared). The four architectures below are the
four wirings. The two routed variants (§3.3, §3.4) realize the **feature-routed
vs. task-routed** distinction discussed below.

### 3.1 `dense` — single network (baseline topology)
One Elman `nn.RNNCell(d_x → H)` (tanh) + shared `Linear(H, d_y)`. The full
8-dim observation feeds the single recurrent population. Rolled `seq_len` times
with an explicit loop so reset and final-step readout are unambiguous.

### 3.2 `modular_shared` — modular, shared input (modularity baseline)
Two independent Elman cells, `H/2` units each, **no inter-module connections**
(hidden states never mix; no comms pathway). **Each module receives the full
`d_x = 8` observation** (its own copy), every phase, every trial — both modules
always active. Hidden = concatenation `(batch, H)`; single **shared**
`Linear(H, d_y)` readout. This is the `a1b2_modular` analog of
`input_routing="shared"`, `common_input=False`, `sparsity=0`.

Purpose: isolates the effect of *splitting the recurrent population into two
non-communicating blocks* while every module still sees the same input. The
control that separates a pure connectivity-partition effect from any routing
effect.

### 3.3 `modular_feature_routed` — modular, **feature-routed** input
Two independent `H/2`-unit cells + shared readout as §3.2, but the `d_x = 8`
observation is **split by feature dimension** across modules: with `n_modules = 2`
and a `d_x / n_modules = 4` split, **module 0 receives `x[0:4]`, module 1 receives
`x[4:8]`** — every phase, every trial, both modules always active. The split is
fixed across the whole experiment and **does not depend on task/phase**. Realized
by a block-diagonal `input_mask` (per-module `input_size = d_x/n_modules = 4`).

Properties:
- **Task-agnostic**: identical wiring in A1, B, A2; the network is *not* told which
  task is active (respects Part III §9 "no task identifiers").
- Both modules contribute to every prediction in every phase; both train in every
  phase.
- Each module builds its representation from a **different sensory subspace**.

Purpose: a fixed structural prior on the *input*. Compared with `modular_shared`
(same recurrent partition, full input to both), it isolates the effect of
**dividing the sensory features** between modules.

> This is the scheme `a1b2_modular` effectively realized under the name
> `task_routed` — its router keyed on the per-trial `feature_idx`/`feature_probe`
> (a *feature* selector), routing whole inputs by feature identity rather than by
> task. toy_task makes it an explicit, separately named architecture.

### 3.4 `modular_task_routed` — modular, **task-routed** input (task-specialist modules)
Two independent `H/2`-unit cells + shared readout, but routing is **by task**:
each module is dedicated to one task in the curriculum. There are exactly two
distinct tasks across `T(0) → T(s) → T(0)` — namely `T(0)` and `T(s)` — so:
- **Phases A1 and A2 (task `T(0)`):** full `x` → **module 0**; module 1 receives
  zeros.
- **Phase B (task `T(s)`):** full `x` → **module 1**; module 0 receives zeros.

A per-trial `task_id ∈ {0,1}` (known from the phase) drives the routing — the
direct analog of a1b2's `feature_probe`, but here it genuinely encodes **task**.
**Task-conditioned (oracle):** the gate *injects task identity*, a deliberate,
documented exception to Part III §9 and the whole point of the contrast (the "give
each task its own module" prior; cf. progressive nets / task-specific columns).

What happens to the **off-task module** is a real design choice with different
gradient flow and scientific meaning. We implement it as an `off_module_policy`
flag with two values (both run by default):

**(a) `freeze` — input-gated hard separation (separation ceiling).**
Off module gets zero input; with `bias=False` and zero initial hidden it stays at
`tanh(0)=0` for all timesteps. A **shared** readout `Linear(H, d_y)` sums both
modules, but the off module contributes 0 and therefore receives **zero gradient
everywhere — including its readout columns**. (Note: with bias-free cells,
"feed empty input" and "freeze" are mathematically identical; there is no signal
to flow back.)
- Module 0 is **physically untouched during B** → A1's `T(0)` representation is
  protected; **interference ≈ 0 by construction, ~independent of `s`**. An honest
  upper bound on structural separation, not a learning phenomenon.
- A1/A2 reps live in module 0; B reps in module 1 → cross-phase comparisons
  (§5.3) read **module-aware** (A1→A2 drift is within module 0; A1→B is across
  modules).

**(b) `readout_coord` — readout-coordinated (gradient flows through both modules).**
Both modules **always receive the full `x`** and always run (no input gate). Task
identity enters at the **readout** instead: two heads `W₀, W₁`, each a
`Linear(H, d_y)` reading from **both** modules; trial uses head `task_id`.
- During B, head `W₁` is active and reads both modules, so **gradient flows back
  into module 0 too** — both modules keep adapting every phase. The unused head
  `W₀` is frozen during B, so A1's input→output *mapping* is **preserved in the
  readout** while the *representations* are free to drift.
- Coordination genuinely happens: the readout must learn to select/combine modules
  per task, and modules can specialize under that pressure.
- **Interference is a graded phenomenon**: at A2 onset the preserved `W₀` no longer
  matches the drifted reps; magnitude should scale with `s` and with rich/lazy γ —
  exactly the dynamics the benchmark is meant to expose.

Purpose: `freeze` is the separation ceiling; `readout_coord` is the scientifically
richer task-conditioned model where gradient-through-unused-module and readout
coordination live. Together they bracket the spectrum and, contrasted with
`modular_feature_routed`, isolate **task routing** from **feature routing**.

> Routing requires `d_x % n_modules == 0` (8 % 2 = 0 ✓) for feature routing; task
> routing requires `n_tasks ≤ n_modules` (2 ≤ 2 ✓). `freeze` reuses the
> `input_mask` machinery (task-gated full input); `readout_coord` leaves the input
> ungated and instead carries a per-task readout head.

### 3.5 Architecture matching (Part III §13) and parameter accounting
Matched across all four: **total hidden units `H`**, tanh activation, optimizer,
readout shape `H→d_y`, init procedure, init-scale application, `seq_len`, reset
policy, `bias=False` in recurrent cells. **Only the input wiring + recurrent
connectivity differ.** Input-weight parameter counts (intrinsic, intended):

| Arch | per-module input dim | recurrent params | input params |
|------|----------------------|------------------|--------------|
| `dense` | 8 → H | `H·H` (full) | `d_x·H` |
| `modular_shared` | 8 → H/2 (×2) | `2·(H/2)²` (block-diag) | `d_x·H` |
| `modular_feature_routed` | 4 → H/2 (×2) | `2·(H/2)²` | `(d_x/2)·H` |
| `modular_task_routed` (`freeze`) | 8 → H/2 (×2) | `2·(H/2)²` | `d_x·H` |
| `modular_task_routed` (`readout_coord`) | 8 → H/2 (×2) | `2·(H/2)²` | `d_x·H` |

Recurrent capacity is matched across all modular variants; dense has the only full
(non-block-diagonal) recurrence. Readout capacity is matched except
`modular_task_routed (readout_coord)`, which carries **two** heads (`2·H·d_y`)
instead of one — intrinsic to having a per-task readout. These differences are
recorded per run.

### 3.6 Hidden-size variation (project decision; Part III §13)
`H` is a **swept variable** (overriding patch §6's fixed 50). Default grid
`H ∈ {12, 24, 50, 100}`; per-module size for the modular nets is `H/2 ∈
{6, 12, 25, 50}` (echoing the `a1b2_modular` size ladder). Constraints: `H` even,
and `H/2` integer. For each `H`, all four architectures use the **same total
`H`** so dense-vs-modular comparisons are capacity-matched at every grid point.
This directly tests whether modular vs. dense effects depend on network width /
per-module capacity. (`H` is config-overridable; a single-value grid recovers the
original fixed-50 benchmark.)

### 3.7 Init-scale γ (Part III §18; patch §6) — `toy_task/init_scale.py`
After standard init, multiply trainable **recurrent** parameters in place by `γ`
(`apply_init_scale(model, gamma)`), mirroring
`a1b2_modular/a1b2/models/rnn_init.py` (`scope="global"` analog, restricted to the
recurrent core; readout left at standard init). γ is an experimental lever; the
rich/lazy label is assigned post hoc from dimensionality dynamics, not from γ
itself.

### Stage-2 / Stage-5 validation (Part V §2)
- Train each model on `T(0)` only; verify convergence to low MSE, stable reps,
  reproducible curves under a fixed seed.
- Before any continual-learning comparison, confirm `dense`, `modular_shared`,
  `modular_feature_routed`, and `modular_task_routed` reach comparable `T(0)` loss
  at each `H` (Stage-5 gate). Note: `modular_task_routed` on `T(0)` uses **only
  module 0** (module 1 idle), so it is effectively a single `H/2` module on `T(0)`;
  expect it to match a half-width net there, with its second module only engaged
  in Phase B.

---

## 4. Training & continual protocol (Part III; patch §4–5, §7) — `toy_task/training.py`

### 4.1 Optimizer
`torch.optim.SGD(lr=0.01, momentum=0.0, weight_decay=0.0)`, batch size 1, no LR
schedule, no gradient clipping. **One optimizer instance created once and reused
across all three phases** — optimizer state and weights are never reset between
phases (patch §5, Part III §3–4).

### 4.2 Phase loop
```
build env(seed); build model; apply γ; build optimizer
run_phase(A1: s=0,  task_id=0)   → 100 epochs
[measure forward transfer onto B BEFORE any B update]
run_phase(B:  s=s,  task_id=1)   → 100 epochs
[measure interference onto A BEFORE any A2 update]
run_phase(A2: s=0,  task_id=0)   → 100 epochs
```
`run_phase` per epoch: permute objects → noisy obs → per-object forward/backward/
step → record mean epoch loss (Part IV §4, patch §7).

**`task_id` and routing.** Each phase carries a `task_id` (0 for `T(0)` phases A1/
A2, 1 for `T(s)` phase B). It is consumed **only** by `modular_task_routed`:
under `freeze` it selects the active input-gated module; under `readout_coord` it
selects the active readout head (both modules always run). `dense`,
`modular_shared`, and `modular_feature_routed` ignore it entirely (task-agnostic
wiring). The `forward(x, task_id=...)` signature is uniform across architectures so
the training loop is identical. The same `task_id` is used at representation-
extraction time so extracted hidden states / outputs correspond to the correct
module gate (`freeze`) or readout head (`readout_coord`).

### 4.3 Loss
MSE on the 2-D output vs. `(cos, sin)` target (Part III §16).

### 4.4 Behavioral metrics (Part IV §4; patch §7)
- **Learning curves**: mean training loss per epoch, per phase.
- **Forward transfer**: clean-data MSE on `T(s)` evaluated the instant A1 ends,
  before any B gradient step.
- **Interference**: clean-data MSE on `T(0)` evaluated the instant B ends, before
  any A2 gradient step.
- Also log clean `T(0)` and `T(s)` loss at every extraction point for trajectories.

### Stage-3 validation (Part V §2)
Confirm weights/optimizer persist across switches, stimuli unchanged, targets
switch correctly; transfer↑ with similarity (small `s`), interference sensible.

---

## 5. Representation extraction & analysis (Part IV; patch §8)

### 5.1 Extraction schedule (patch §8) — within `training.py`
Extract at: before training, every 10 epochs within each phase, immediately
before and after every phase transition, and the final epoch of each phase.
Extraction always uses **clean** observations (σ=0), one row per object → `R` is
`(N, H) = (8, 50)`.

### 5.2 Per-extraction record (Part IV §17; patch §8; Part V §7)
Store: `phase, epoch, global_step, s, gamma, arch, seed`, plus arrays
`Z, X_clean, hidden R (8×50), output preds (8×2), targets (8×2)`, and the scalar
clean losses. Persist as per-run `.npz`/`.parquet` + a `config.json`. No reliance
on implicit defaults (Part V §7).

### 5.3 Analyses — `toy_task/analysis.py` (formulas reused for parity)
Operate on `R` matrices (`8×50`) and across phases A1/B/A2.

- **PCA geometry & trajectories** (Part IV §7): 2-D/3-D projections, explained
  variance, principal axes; for trajectory plots fit one shared PCA on stacked
  `[R_A1; R_B; R_A2]` (mirrors `prepare_pca_shared_three_phase`).
- **Effective dimensionality** (Part IV §8): participation ratio
  `PR = (Σλ_i)² / Σλ_i²` on the covariance eigenvalues of centered `R`
  (mirrors `compute_participation_ratio`); also components-to-variance-threshold
  (0.9, 0.99).
- **Principal angles** (Part IV §9): between top-2 PCA subspaces of pairs
  {A1,B}, {A1,A2}, {B,A2} via SVD of `components_A · components_Bᵀ`, in degrees
  (mirrors `compute_principal_angle`).
- **RSA** (Part IV §10): per-phase `8×8` representational similarity matrices
  (cosine and Pearson correlation), compared across phases/architectures.
- **CKA (optional)** (Part IV §11): linear CKA between phase `R` matrices
  (mirrors `correlations.CKA.linear_CKA`).
- **Drift** (Part IV §12, §13): per-object Euclidean + cosine distance of `h_i`
  across A1→B→A2 (mean L2 mirrors `compute_hidden_drift`), reported globally and
  per object (does each object return to its A1 representation after A2?).

### Stage-4 validation (Part V §2)
Verify object ordering is consistent across extractions, reps are reproducible
under fixed seed, and initial PCA trajectories look sensible.

---

## 6. Learning regimes — γ ladder (Part III §18; Part V Stage 6)

Default γ grid `{0.001, 0.01, 0.1, 1.0, 2.0}` — the same init-scale ladder used
in `a1b2_modular` (`PRIMARY_GRID_STORAGE_POLICY.md`, paper figures) to span lazy
(small γ) through rich (large γ) for this architectural setup. Labels are
assigned post hoc from dimensionality dynamics, not from γ itself. **Gate (Stage
6):** confirm effective dimensionality (participation ratio at end of A1) varies
systematically with γ before launching the full sweep. Grid is editable in config
without touching code.

---

## 7. Experiment driver & outputs — `scripts/run_experiment.py`

CLI over the full grid `arch × H × γ × s × seed` (Part III §19, §21):
```
for seed:
  build environment(seed)                 # shared across all configs
  for arch_config in [dense,
                      modular_shared,
                      modular_feature_routed,
                      modular_task_routed(off=freeze),
                      modular_task_routed(off=readout_coord)]:
    for H in H_grid:
      for gamma in γ_grid:
        for s in s_grid:
          run A1→B→A2, extract reps, dump per-run artifacts
```
Flags: `--arch`, `--off-module-policy`, `--hidden-sizes`, `--gammas`,
`--similarities`, `--seeds`, `--epochs-per-phase`, `--out`, `--print-config`. All
configs share identical seeds/environments so comparisons are paired (Part IV §16).
Outputs land under `data/runs/<run_id>/` with `run_id` encoding
`arch[_off]_Hxx_gammaG_sIDX_seedK` (analogous to a1b2's `build_run_id`).

Default grid size: `5 configs × 4 H × 5 γ × 7 s × 10 seeds = 7000` runs ×
2400 updates — cheap by design (Part V §8); subset via flags for pilots (e.g. drop
one `off_module_policy`, or one routing scheme).

A second script `scripts/make_figures.py` (+ `notebooks/`) aggregates runs into
the Part IV §18 figure set: learning curves, transfer, interference, PCA
trajectories, RSA matrices, principal-angle plots, effective dimensionality, drift.
Primary contrasts, each across the `H` grid and the `γ × s` plane, with mean ±
std / CI over seeds (Part IV §16):
- `dense` vs `modular_shared` → isolates **recurrent partitioning** (same input).
- `modular_shared` vs `modular_feature_routed` → isolates **feature routing**
  (sensory split, task-agnostic).
- `modular_feature_routed` vs `modular_task_routed` → isolates **task routing**
  vs feature routing.
- `modular_task_routed(freeze)` vs `modular_task_routed(readout_coord)` → isolates
  **hard separation** vs **readout coordination / gradient through both modules**.
- `modular_task_routed(freeze)` is the **separation upper bound** (interference ≈ 0
  by construction); `readout_coord` should show graded, `s`- and γ-dependent
  interference.

---

## 8. Package layout

```
toy_task/
├── IMPLEMENTATION_PLAN.md        ← this file
├── README.md
├── pyproject.toml                ← deps: numpy, torch, matplotlib (+ scikit-learn, pandas)
├── description/                  ← spec (parts 1–5 + patch)
├── toy_task/
│   ├── __init__.py
│   ├── config.py                 ← frozen constants + grids (dataclasses)
│   ├── environment.py            ← latents, observation fn, targets, sampling
│   ├── models.py                 ← Dense + Modular (shared / feature-routed / task-routed)
│   ├── init_scale.py             ← apply_init_scale(model, γ)
│   ├── training.py               ← phase loop, transfer/interference, extraction
│   ├── analysis.py               ← PCA, PR, principal angles, RSA, CKA, drift
│   └── storage.py                ← run_id, save/load configs + artifacts
├── scripts/
│   ├── run_experiment.py
│   └── make_figures.py
├── notebooks/                    ← exploratory figures
├── data/                         ← runs/ (gitignored at repo root via data/*)
└── figures/
```

`pyproject.toml` gains `scikit-learn` (PCA) and `pandas` (aggregation) on top of
the current `numpy, torch, matplotlib`.

---

## 9. Build order (Part V staged; each stage validated before the next)

| Stage | Deliverable | Validation gate |
|-------|-------------|-----------------|
| 1 | `config.py`, `environment.py` | env invariances; only targets change with `s`; |z|,|x|,|y| visuals |
| 2 | `models.DenseRNN`, `init_scale`, single-phase train | converges on `T(0)`; reproducible curves |
| 3 | `training.py` A1→B→A2 + transfer/interference | weights/opt persist; transfer↑ for small `s` |
| 4 | `analysis.py` + extraction schedule | reps reproducible; consistent object order; PCA sane |
| 5 | `modular_shared`, `modular_feature_routed`, `modular_task_routed` (`freeze` + `readout_coord`) + `H` sweep | all configs ≈ on `T(0)` at each `H`; routers verified (feature split fixed & task-agnostic; task `freeze`: module 0 unchanged across B; task `readout_coord`: gradient reaches both modules in B, off-head frozen in B) |
| 6 | γ ladder + `run_experiment.py`/`make_figures.py` | effective dim varies with γ; full grid runs |

Tests (lightweight, under `toy_task/tests/`): env determinism per seed, target
geometry, shape contracts of all configs, **router correctness** —
- feature-routed: each module's input weights only touch its feature slice; wiring
  identical across phases;
- task-routed `freeze`: off module receives zeros and gets zero gradient in its
  off-phase; module 0 params bit-identical before/after Phase B; readout columns
  for the off module unchanged in the off-phase;
- task-routed `readout_coord`: both modules' recurrent params **change** during B
  (nonzero gradient through the unused module); the off-task readout head is
  bit-identical before/after B;

optimizer-state persistence across phases, and analysis formulas on tiny synthetic
inputs.

---

## 10. Resolved decisions & defaults (would otherwise be ambiguous)

1. **Patch wins** over Parts I–V on every overlap (observation `+ε` & `W` scale,
   lr, seq_len=3, similarity grid, update budget) — **except hidden size**, which
   is promoted to a swept variable per project decision (§3.5).
2. **γ scope** = recurrent core only, `global`-style multiply (per a1b2
   `rnn_init.py`), readout untouched. (Open to "all params" if requested.)
3. **No inter-module communication at all** — the comms pathway is omitted from
   the start (not an ablation). Modular = two independent recurrent cores + shared
   readout (`a1b2` `sparsity=0` analog).
4. **Four architectures** (5 effective configs): `dense`; `modular_shared` (both
   modules see full `x`); `modular_feature_routed` (input feature dims split,
   task-agnostic, both always active); `modular_task_routed` (task-gated) with
   `off_module_policy ∈ {freeze, readout_coord}`. **Both routing schemes**
   (feature vs task) are implemented (the distinction conflated in `a1b2_modular`),
   and the task-routed off-module behavior is a flag: `freeze` = input-gated hard
   separation (interference ≈ 0 by construction; with bias-free cells "empty input"
   ≡ "freeze"); `readout_coord` = both modules always on with per-task readout
   heads, so **gradient flows through the unused module** and coordination happens
   at the readout. `modular_task_routed` is the only architecture that uses task
   identity (documented exception to Part III §9).
5. **Hidden size `H` is swept** (default `{12,24,50,100}`, modular per-module
   `H/2`), all archs capacity-matched at each `H` (§3.5).
6. **Seeds** default `range(10)`; **γ grid** default `{0.001,0.01,0.1,1.0,2.0}`
   (a1b2 primary-grid ladder) — both config-overridable.
7. **Representation** = final-timestep hidden state on **clean** inputs, `N×H`.
8. **Analysis formulas re-implemented locally** (no cross-project imports) but
   numerically matched to the cited a1b2 functions for comparability.
9. **Storage**: per-run `.npz` + `config.json`; `data/` is gitignored at repo
   root (`data/*`), so artifacts stay local.

### Open questions for confirmation (non-blocking; sensible defaults chosen)
- `H` grid `{12,24,50,100}`, γ grid `{0.001,0.01,0.1,1.0,2.0}`, seed count 10 — proceed?
- Feature-routing split: contiguous halves of `x` (default) vs. a fixed random
  per-seed permutation of feature indices before splitting.
- `modular_task_routed`: both off-module policies (`freeze`, `readout_coord`) are
  run by default (grid 5×4×5×7×10 = 7000). OK, or restrict to one to shrink the
  grid? (`freeze` = separation ceiling; `readout_coord` = the richer, `s`/γ-
  dependent dynamics — recommended primary.)
- Whether γ should scale the readout too (default: no).
- Whether to include the optional CKA in the default figure set (default: yes,
  it's cheap at `N×H`).

---

*Scope guard:* multi-output (Holton-style), non-rigid task deformations,
structured latent manifolds, inter-module communication, and hierarchical/
multi-layer architectures are **explicitly deferred** to ablations (Part V §6) and
are out of scope for this baseline build. (Both feature-routed and task-routed
modularity are **promoted into the baseline** as `modular_feature_routed` and
`modular_task_routed`; inter-module communication is dropped entirely rather than
deferred.)
