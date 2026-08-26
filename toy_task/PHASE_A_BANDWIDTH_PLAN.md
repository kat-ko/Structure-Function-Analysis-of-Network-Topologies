# Phase A: Communication Bandwidth — Integration Plan

Status: **approved direction** (refinement of prior architecture-first sequencing, not a pivot).  
Goal: restore a **continuous architecture axis** before any further task manipulations (seasons, reuse-matched novel).

---

## 0. Problem statement (confirmed diagnosis)

The five discrete `ARCHITECTURES` are **not a manipulable axis**:

| Config | Role in current grid | Interference (shared, nb07, γ rich/lazy) |
|--------|----------------------|------------------------------------------|
| `dense` | Full sharing baseline | ~58–59° / ~59° |
| `modular_shared` | “Modular” with full input to both modules | ~59° / ~59° (≈ dense) |
| `modular_feature_routed` | Input split by feature dim | ~58° / ~59° (≈ dense) |
| `modular_task_routed` + `freeze` | Hard separation ceiling | ~4° / ~4° (by construction) |
| `modular_task_routed` + `readout_coord` | Intended middle ground | ~8–30° (metric confounds on forward transfer) |

**Interpretation:** Flat `mod-shared` / `mod-feature` vs `dense` is **no dynamic range**, not evidence that modularity is irrelevant.  
**Sequencing:** Fix architecture sensitivity → then task structure (seasons H1/H3/H4, reuse-matched novel).

---

## 1. What this plan changes vs leaves alone

### In scope (Phase A + probes)

- Restore **a1b2-style inter-module recurrent coupling** as scalar `comms_bandwidth`.
- Base on **`modular_shared` input wiring** (both modules see full `x`) to isolate **recurrent mixing**.
- Add **diagnostic probes** (per-module decode, frozen-A forward transfer) — fixes apply to all future runs.
- Small grid (~180 runs): shared + even angles only.
- Pre-registered outcome table before interpreting results.

### Explicitly out of scope (Phase A)

- Seasons / dual-output (H1/H3/H4) — **Phase B**, only if Phase A shows slope.
- Reuse-matched novel condition — **Phase B**.
- Full 4,650-run overnight replication.
- **Feature-routing axis** (`modular_feature_routed`) — not swept in Phase A; a flat bandwidth result **does not** rule it out (see §8).

### Still open (not resolved by Phase A)

- **γ as optimization dynamics vs dimensionality** — bandwidth sweep does not disambiguate.
- **Whether graded separation requires an explicit loss term** vs emerges from architecture alone.

---

## 2. Pre-registered outcomes (commit before running)

Evaluate Phase A on **reference slice**: `stimulus_regime=shared`, `angle_mode=even`, `H=50`, `init_scope=all`, `epochs_per_phase=8000` (or 2000 for pilot — pick one and stick to it).

Primary metrics: `interf_deg`, `pang_A1B`, `npc99_A1`, `transfer_deg`.  
Secondary: new probes (`probe_module0_rule_mse`, `forward_transfer_Ts_task0`).

| Outcome | Criterion | Next step |
|---------|-----------|-----------|
| **A — Gradient exists** | At least one primary metric **monotonic in `comms_bandwidth`** (Spearman \|ρ\| ≥ 0.6 across {0, .25, .5, .75, 1}, same sign across ≥2 γ or ≥2 of 3 sims) | Phase B: seasons / reuse-matched on **2–3 bandwidth values** that showed sensitivity |
| **B — Flat** | No monotonic trend; dense ≈ bandwidth=0 ≈ all intermediate | **Do not** conclude “modularity doesn’t matter.” Conclude **recurrent coupling is not the lever** under mod-shared wiring. Plan **feature-routing sweep** or input-separation axis as Phase A′ |
| **C — Binary switch** | Only extremes differ (0 ≈ dense, 1 ≈ freeze-like); mid bandwidths ≈ one extreme | Paper claim becomes **threshold / regime switch**, not graded modularity benefit |
| **D — Inconclusive** | High variance across seeds or non-monotonic with overlapping CIs | Increase seeds to 3 or epochs; do not proceed to seasons until resolved |

**Anti-pattern to avoid:** Bandwidth-null → “modularity doesn’t help” (same trap as flat mod-shared vs dense).

---

## 3. Work packages (implementation order)

### WP1 — `comms_bandwidth` in the model (blocking)

**Reference:** `a1b2_modular/a1b2/models/community.py` — `core` + `comms` pathways, `comms_mask(sparsity, ...)`, shared weights, summed hidden states.

**Design choice for toy_task (recommended):**

Two-pathway layout (faithful to a1b2):

```
h_t = core_path(x_t, h_{t-1}) + comms_path(x_t, h_{t-1})
```

- **`core` mask:** block-diagonal recurrent `(1 - I)` per module (current `_recurrent_mask`).
- **`comms` mask:** off-diagonal blocks only; each entry scaled by `comms_bandwidth ∈ [0, 1]`.
- **Deterministic mask:** For reproducibility across seeds, fix inter-module connectivity pattern at init (e.g. fixed random pattern per `(seed, bandwidth)` or fully dense off-diagonal scaled by bandwidth). Document choice in `run_id` if pattern is seed-dependent.

**Simpler alternative (if dual-RNN is too heavy):**

Single `nn.RNN` with combined recurrent mask:

```
W_hh_eff = W_hh * (core_mask + comms_bandwidth * comms_mask_offdiag)
```

Prefer dual-pathway if parity with a1b2 analyses matters.

**Files:**

| File | Change |
|------|--------|
| `toy_task/config.py` | `COMMS_BANDWIDTH_GRID`, add `comms_bandwidth: float = 0.0` to `RunConfig`; encode in `run_id()` as `bw{bandwidth}` (e.g. `bw0`, `bw0.5`) |
| `toy_task/models.py` | Implement comms pathway; `build_model(..., comms_bandwidth=0)`; only for `arch in {modular_shared, modular_comms}` or extend `modular_shared` with bandwidth param |
| `toy_task/__init__.py` | Export new constants if any |
| `tests/test_models.py` | `bandwidth=0` matches current masks; `bandwidth=1` has non-zero cross-module `weight_hh`; gradient flows across modules when bandwidth > 0 |

**Backward compatibility:**

- `comms_bandwidth=0` + `modular_shared` must be **bit-identical** to current `modular_shared` (regression test).
- `dense` ignores bandwidth (always 0).

**Architecture naming:**

- Option A: Keep `arch=modular_shared`, add `comms_bandwidth` field (cleanest for grid).
- Option B: New `arch=modular_comms` alias — avoid unless needed for clarity.

Recommend **Option A**.

---

### WP2 — Diagnostic probes (parallel, before bandwidth sweep)

Fix known metric confounds; useful for all architectures, especially `modular_task_routed`.

| Probe | Definition | When computed |
|-------|------------|---------------|
| `forward_transfer_Ts` (existing) | B objects, `task_id=1`, before B training | Keep |
| **`forward_transfer_Ts_task0`** (new) | B objects, **`task_id=0`** / head 0, before B training | A1-end weights; tests rule in A pathway |
| **`probe_module{m}_rule_mse`** (new) | Linear map from module-`m` hidden (A1-end, A objects) → A targets; evaluate on B objects with +s | Per-module representational adequacy |
| **`probe_module{m}_forward_mse`** (new) | Same linear probe fit on A1-end (A, T0), test on B (T(s)) | Cheap oracle-style generalization per module |

**Implementation sketch:**

- `training.py`: after A1, call new `diagnostic_probes(model, env, cfg)` → dict of floats in `result.behavioral`.
- Use `model.module_slice(m)` + `sklearn.linear_model.Ridge` or closed-form linear regression (2D output).
- `analysis.py` / notebooks: add columns to `_row()` builder.

**Files:** `training.py`, `storage.py` (schema in `behavioral` JSON), `tests/test_training.py` (smoke), update nb07/08/10 loaders when re-running.

**Priority:** Complete WP2 **before** interpreting any new `modular_task_routed` forward-transfer numbers.

---

### WP3 — Phase A grid runner

**Script:** `scripts/bandwidth_phase_a_grid.py` (new; mirror `overnight_grid.py` / `even_novel_smoke_grid.py`).

**Grid (target ~180 runs):**

| Factor | Levels | Count |
|--------|--------|-------|
| `arch` | `dense`, `modular_shared` | 2 |
| `comms_bandwidth` | `0, 0.25, 0.5, 0.75, 1.0` | 5 |
| `gamma` | `0.001, 0.1, 2.0` (rich, mid, lazy) | 3 |
| `similarity_index` | Same(0), Near(2), Far(6) | 3 |
| `seed` | `0, 1` | 2 |
| `stimulus_regime` | `shared` | 1 |
| `angle_mode` | `even` | 1 |
| `hidden_size` | `50` | 1 |
| `init_scope` | `all` | 1 |
| `epochs_per_phase` | `8000` (or `2000` pilot) | 1 |

**Total:** 2 × 5 × 3 × 3 × 2 = **180** configs.

**Output:** `data/runs_bandwidth_phase_a/` (separate from `runs_overnight`).

**Execution:** Resumable pool; `nice -n 19`; ~64 workers optional (180 runs ≈ few hours at 8000 ep).

**Dry-run:** `python scripts/bandwidth_phase_a_grid.py --dry-run`

---

### WP4 — Analysis notebook

**Script:** `scripts/gen_nb11.py` → `notebooks/11_bandwidth_phase_a.ipynb`

Sections:

1. Load + cache `bandwidth_phase_a_summary.pkl`
2. **Primary:** interference vs bandwidth × γ (lines per arch)
3. **Geometry:** `pang_A1B`, `npc99_A1` vs bandwidth
4. **Probes:** module-0 decode vs full-pipeline forward transfer (demonstrate confound fix on task_routed if included in spot-check)
5. **Pre-registration evaluator:** automated Spearman + outcome class A/B/C/D
6. **Explicit disclaimer:** flat bandwidth ≠ “modularity irrelevant”; feature-routing axis untested

Optional spot-check: re-run 6 `modular_task_routed` configs with new probes only (no full grid).

---

### WP5 — Paper framing (Phase C, no compute)

Update WIP text **now**:

**Lead (robust):**

- γ sets **effective dimensionality** (rich ~2–3 PCs vs lazy ~4; stable across regimes/angles).
- Learning regime shapes representational complexity independent of task Holton-faithfulness.

**Conditional (pending Phase A):**

- Inter-module **coupling bandwidth** gates interference / subspace reorientation under shared identities.

**Demote until data support:**

- “Modularity protects abstract rule transfer in Holton-faithful novel regime.”

**Methods caveats to add:**

- Shared regime allows object-reuse shortcuts (reuse-matched condition deferred).
- Task-routed forward transfer requires `task0` probe for interpretability.

---

## 4. `comms_bandwidth` semantics (spec for colleague)

| `comms_bandwidth` | Behavior |
|-------------------|----------|
| `0` | Current toy_task: block-diagonal recurrence only; no cross-module hidden mixing |
| `1` | Full off-diagonal recurrent coupling (a1b2 `sparsity=1` analog between modules) |
| `(0,1)` | Scaled comms mask; linear interpolation unless binary mask chosen |

**Not the same as:**

- `modular_feature_routed` (input routing axis)
- `modular_task_routed` (task gating axis)
- `mod-task(freeze)` (input zeroing = hard separation, not bandwidth)

**Mapping to old configs:**

| Old config | Approximate bandwidth analog |
|------------|------------------------------|
| `dense` | N/A (single module) |
| `modular_shared` @ bw=0 | Current mod-shared |
| `modular_task_routed` freeze | bw→∞ + input gating (not captured by bandwidth alone) |

---

## 5. Decision tree after Phase A

```
Phase A complete
    │
    ├─ Outcome A (gradient) ──► Pick 2–3 bandwidth values (e.g. 0, 0.5, 1)
    │                           Phase B1: seasons H1/H3/H4 pilot (small grid)
    │                           Phase B2: reuse-matched novel (small grid)
    │
    ├─ Outcome B (flat) ──────► Phase A′: feature-routing or input-separation sweep
    │                           (do NOT jump to seasons as explanation)
    │
    ├─ Outcome C (binary) ────► Reframe paper: threshold phenomenon
    │                           Still run seasons at bw∈{0,1} only
    │
    └─ Outcome D ─────────────► More seeds / epochs; hold Phase B
```

---

## 6. Testing checklist

- [ ] `pytest tests/` — bandwidth=0 regression for mod-shared
- [ ] `bandwidth_phase_a_grid.py --dry-run` → 180 ids
- [ ] 2-epoch smoke: one config per bandwidth level saves valid `config.json` + probes
- [ ] nb11 pre-registration cell classifies synthetic monotone / flat fixtures correctly

---

## 7. Immediate todos (ordered)

1. **WP2 probes** — `forward_transfer_Ts_task0` + per-module linear probes in `training.py`
2. **WP1 model** — `comms_bandwidth` in `models.py` + `RunConfig` + tests
3. **WP3 runner** — `bandwidth_phase_a_grid.py`; pilot with `epochs=500` then full `8000`
4. **WP4 nb11** — analysis + pre-registration evaluator
5. **WP5** — one-paragraph framing update in WIP / README pointer to this plan
6. **Evaluate** against §2 table → decide Phase B

---

## 8. Writing-up guardrails (for Phase A results section)

When reporting Phase A, always include:

1. **What was isolated:** recurrent inter-module coupling with mod-shared input.
2. **What was not tested:** feature-routing, task-routing, seasons, novel reuse-matched.
3. **What flat null means:** no evidence that *this knob* matters — not that modularity is irrelevant.
4. **Probe interpretation:** distinguish representational adequacy (module decode) from pipeline metrics (task_id routing).

---

## 9. Reference files

| Path | Role |
|------|------|
| `a1b2_modular/a1b2/models/community.py` | `comms_mask`, core+comms forward |
| `toy_task/toy_task/models.py` | Current single-pathway implementation |
| `toy_task/IMPLEMENTATION_PLAN.md` §10.3 | Original “no comms” decision |
| `toy_task/data/runs_overnight/` | Legacy grids (do not mutate) |
| `toy_task/notebooks/10_even_novel_analysis.ipynb` | Even-angle novel baseline for contrast |
