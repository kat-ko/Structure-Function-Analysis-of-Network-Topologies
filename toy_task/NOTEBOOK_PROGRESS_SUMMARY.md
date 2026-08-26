# toy_task — Notebook progress summary

**Purpose:** Record what has been *manipulated*, what has been *measured*, and what the
notebooks *report so far* — without causal interpretation or paper claims. Use this as a
map when revisiting results or planning the next experiment.

**Last updated:** 2026-07-13 (pilot Phase A grid ~65% complete).

---

## 1. Experimental protocol (constant across notebooks)

All neural-network notebooks use the same continual-learning curriculum unless noted:

| Element | Description |
|---------|-------------|
| **Phases** | A1 (task 0, similarity 0) → B (task 1, similarity *s*) → A2 (task 0 again) |
| **Optimizer** | Single persistent SGD; weights never reset between phases |
| **Task** | 8 objects, 2D angular readout; similarity *s* rotates B's target rule |
| **Metrics** | Angular error (°), interference = err(A2-start) − err(A1-end), switch cost / forward transfer = err(B-start) − err(A1-end), geometry (PR, #PCs 99%, principal angles), rule-shift (`pull_fraction`) |

Reference slice used in most digests: **H=50**, **init_scope=all**, **Same/Near/Far** similarities, **3 seeds** (overnight grid).

---

## 2. Factors manipulated over time

Chronological order of *changes to the benchmark or grids* (not notebook order):

| Factor | Levels explored | When introduced | Notebooks |
|--------|-----------------|-----------------|-----------|
| **Architecture** | `dense`, `modular_shared`, `modular_feature_routed`, `modular_task_routed` (+ `freeze` / `readout_coord`) | From start | nb01–nb11 |
| **Init scale γ** | 0.001 … 2.0 (rich → lazy) | From start | all |
| **Task similarity *s*** | 7 values, 0 … π | From start | all |
| **Hidden size H** | 12, 24, 50, 100 | Overnight grid | nb07, nb08, nb10 |
| **Training length** | 100 → 150 → 300 → 4000 → **8000** epochs/phase | Progressive | nb01–nb05 vs nb07+ |
| **Init scope** | `all` vs `no_readout` | Overnight ablation block | nb07 |
| **Stimulus regime** | **`shared`** (B reuses A's objects) vs **`novel`** (B has new objects) | Patch / overnight | nb07, nb08, nb10 |
| **Angle layout** | **`random`** (`atan2(z)`, legacy) vs **`even`** (45° ring, patch-faithful) | nb09 diagnostic → default | nb09, nb10; nb08 uses legacy random only |
| **Inter-module bandwidth** | `comms_bandwidth` 0, 0.25, 0.5, 0.75, 1.0 on dense + mod-shared | Phase A (Jul 2026) | nb11 (pilot in progress) |
| **Diagnostic probes** | `forward_transfer_Ts_task0`, per-module linear probes | Phase A code | nb11 (when mod-shared runs exist) |

**Not yet implemented in any grid:** summer/winter seasons, reuse-matched novel control, feature-routing bandwidth sweep.

---

## 3. Notebook map

| NB | Data source | Main manipulation / focus | Status |
|----|-------------|---------------------------|--------|
| **01** | Ad-hoc runs | Smoke test, small arch comparison, intro analyses | Exploratory |
| **02** | Scaled runs | γ scaling behaviour | Exploratory |
| **03** | Shorter grid | Full-grid figures at ~150 ep | Superseded by longer training |
| **04** | Per-config | Individual run deep-dives | Supporting |
| **05** | Long training (300–4000 ep) | Convergence probe; lazy γ needs more epochs | Motivated 8000 ep overnight |
| **06** | Alignment checks | a1b2 mask / RNN parity | Implementation validation |
| **07** | `runs_overnight/` (4,650 runs, 8000 ep, **random** angles, shared+novel) | Full factorial; regime contrast | **Primary legacy analysis** |
| **08** | Same cache, filter **novel** only | Novel slice, **random** angles | Same phenomena as nb07 novel arm |
| **09** | `runs_nb09/` (54 runs) | **Angle diagnostic**: random vs even on novel | Drove even-angle default |
| **10** | `runs_overnight/` `_angeven_` subset (4,650 runs) | **Novel + even angles** (post-nb09) | **Current novel-regime reference** |
| **11** | `runs_bandwidth_phase_a_pilot/` (180 planned, 500 ep) | **Shared + even + bandwidth** sweep | **In progress** (~116/180 at last check) |

---

## 4. Observations reported in notebooks (descriptive only)

Numbers below are **digest tables** from executed notebooks (means over seeds/similarities unless noted). They describe what was measured, not why.

### 4.1 Learning regime (γ) → effective dimensionality

Reported consistently in nb07, nb10 (and qualitatively in nb05):

| γ | Typical #PCs (99% var) at A1-end |
|---|----------------------------------|
| 0.001 (rich) | ~2.3–3.3 |
| 2.0 (lazy) | ~4.0–4.7 |

Architecture has only minor effect on this split; the γ ladder separates representational complexity more clearly than arch does in most slices.

### 4.2 Architecture × interference (reference: H=50, scope=all)

**Shared stimuli, random angles (nb07):**

| Architecture | Interference rich (°) | Interference lazy (°) | lazy − rich |
|--------------|----------------------:|------------------------:|------------:|
| Single (dense) | 58.8 | 59.1 | +0.3 |
| Mod-shared | 58.9 | 59.3 | +0.4 |
| Mod-feature | 58.4 | 59.0 | +0.6 |
| Mod-task (coord) | 8.4 | 9.2 | +0.8 |
| Mod-task (freeze) | 4.3 | 3.9 | −0.4 |

**Novel stimuli, random angles (nb07):**

| Architecture | Interference rich (°) | Interference lazy (°) | lazy − rich |
|--------------|----------------------:|------------------------:|------------:|
| Single | 71.5 | 66.6 | −4.9 |
| Mod-shared | 72.5 | 69.0 | −3.5 |
| Mod-feature | 64.0 | 69.2 | +5.2 |
| Mod-task (coord) | 29.6 | 18.6 | −11.0 |
| Mod-task (freeze) | 4.7 | 3.1 | −1.6 |

**Novel stimuli, even angles (nb10):**

| Architecture | Interference rich (°) | Interference lazy (°) | lazy − rich |
|--------------|----------------------:|------------------------:|------------:|
| Single | 59.2 | 56.3 | −2.9 |
| Mod-shared | 60.4 | 59.3 | −1.1 |
| Mod-feature | 61.8 | 59.3 | −2.5 |
| Mod-task (coord) | 11.3 | 8.4 | −2.9 |
| Mod-task (freeze) | 2.4 | 3.5 | +1.1 |

**Pattern (descriptive):** `dense`, `modular_shared`, and `modular_feature_routed` cluster together (~56–72° depending on regime/angles). `modular_task_routed` + `freeze` stays near ~2–5°. `readout_coord` is intermediate (~8–30°) but shares confounds with task routing (see §4.4).

### 4.3 Stimulus regime effect (novel − shared interference, nb07)

Largest positive shifts (more interference in novel) for dense/mod-shared/mod-feature at rich γ, e.g.:

- Mod-shared: +13.6° (γ=0.001), +9.7° (γ=2.0)
- Single: +12.7° (γ=0.001), +7.4° (γ=2.0)
- Mod-task (freeze): near 0 (−0.8 to +0.5°)

Novel regime raises interference for the “fully shared” architectures; task-routed freeze is largely unchanged.

### 4.4 Forward transfer to novel B objects (err at B-start, °)

**Random angles (nb07, novel):** dense / mod-shared / mod-feature ≈ **59–63°** (poor). Mod-task (coord) **78–96°**; freeze **82–91°**.

**Even angles (nb10, novel):** dense / mod-shared / mod-feature ≈ **63–65°** (still poor; modest change vs random). Mod-task (coord) **85–99°**; freeze **90–95°**.

Forward transfer on the standard pipeline metric (`task_id=1` at B-start) does **not** separate dense from mod-shared in any completed notebook.

### 4.5 Angle-layout diagnostic (nb09)

| Quantity | Random | Even |
|----------|-------:|-----:|
| Oracle linear forward MSE | 25.35 | 0.25 |
| RNN forward MSE after A1 | 0.67 | 0.40 |
| Interference MSE (Near, dense, full protocol) | 0.56 | 0.35 |

nb09 also reports mean correlation between min pairwise angle separation and forward transfer on random layouts: **r ≈ 0.39**.

**Decision recorded in nb09:** adopt even-spaced angles as default before further grid investment.

### 4.6 Rule-shift / Holton orthogonalisation (selected)

**Rule-pull (`pull_fraction`, mid-range s, nb07 shared):** dense / mod-shared / mod-feature ≈ **0.76–1.0** (predictions drift toward B's rule). Mod-task (freeze/coord) ≈ **0**.

**Task subspace angle @ post-B (`pang_tasks_postB`, nb10 novel even):** freeze ≈ **90°** (orthogonal by construction); dense/mod-shared/mod-feature ≈ **3–9°** (low).

### 4.7 Phase A bandwidth pilot (nb11, partial)

- Grid: dense vs mod-shared × 5 bandwidths × 3 γ × 3 sims × 2 seeds; **shared**, **even**, H=50, **500 epochs/phase**.
- At partial execution (~42 dense runs): mean A1-end error **≈ 3.2°** (training not fully converged vs 8000 ep runs).
- Pre-registration evaluator returned **Outcome D (inconclusive)** — expected with incomplete grid and no mod-shared rows yet.
- Probe figures (`fig4`) not generated until mod-shared runs with probe fields are present.

---

## 5. Methodological changes driven by notebook results

These are *documented responses*, not scientific conclusions:

| Observation in notebooks | Code / grid change |
|--------------------------|-------------------|
| Short training left lazy γ under-saturated (nb05) | Overnight grid at **8000** epochs/phase |
| Random angles hurt learnability (nb09) | Default **`angle_mode=even`**; `_angeven_` run_id token |
| Discrete archs lack dynamic range (nb07/10) | Phase A: scalar **`comms_bandwidth`** on mod-shared |
| `forward_transfer_Ts` confounded for task-routed (noted in plan) | Probes: **`forward_transfer_Ts_task0`**, module linear decodes |
| Legacy vs even caches must not mix | `run_id` encodes `angeven`, `bw*`, `rshared`/`rnovel` |

---

## 6. What has *not* been established yet

Stated neutrally — open questions the current notebooks do not answer:

1. Whether **inter-module bandwidth** creates a graded interference/geometry axis (nb11 pilot incomplete; 500 ep only).
2. Whether **even angles** restore **modularity × γ** effects on novel transfer (nb10: archs still collapse).
3. Whether **seasons** or **reuse-matched novel** change conclusions (not run).
4. Whether **feature-routed** input separation is a separate lever from recurrent bandwidth (not swept in Phase A).
5. Causal story for **γ** (optimization dynamics vs dimensionality) — correlated measures, not disambiguated.

---

## 7. Data assets (quick reference)

| Path | Contents |
|------|----------|
| `data/runs_overnight/` | 4,650 legacy (random angles) + 4,650 `_angeven_` (even) |
| `data/overnight_summary.pkl` | nb07 cached table (legacy ids) |
| `data/novel_even_summary.pkl` | nb10 cached table |
| `data/runs_nb09/`, `data/nb09_diagnostics.pkl` | Angle diagnostic |
| `data/runs_bandwidth_phase_a_pilot/` | Phase A pilot (in progress) |
| `figures/nb07` … `figures/nb11` | Saved figures per notebook |

---

## 8. Suggested reading order for new collaborators

1. **nb07** — full factorial baseline (both regimes, random angles).
2. **nb09** — why angle layout changed.
3. **nb10** — current novel-regime numbers with even angles.
4. **nb11** — ongoing bandwidth axis (re-run when pilot completes).
5. **PHASE_A_BANDWIDTH_PLAN.md** — pre-registered outcomes for Phase A only.

---

*This document will be updated when the Phase A pilot finishes and nb11 is re-executed on the full 180-run cache.*
