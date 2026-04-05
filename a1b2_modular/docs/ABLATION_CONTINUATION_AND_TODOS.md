# Ablation work: how to continue (continuation guide)

This file is a **practical continuation guide** for humans and agents. It is **not** the same document as the canonical implementation checklist.

---

## Two kinds of content (do not mix them up)

| Kind | Where it lives | What it is |
|------|----------------|------------|
| **Canonical implementation plan** | [`PLAN_rnn_depth_and_celltype_ablations.md`](PLAN_rnn_depth_and_celltype_ablations.md) | **Phase 1–6**: ordered technical phases (plumb depth, verify hiddens, GRU, optional `build_run_id`, LSTM, docs). Use that file for *what the codebase should support* and for phase-level acceptance wording. |
| **Operational follow-up TODOs** | **Section “Operational follow-up TODO list” below** | **Checklist items** `[ ]` / `[x]` for *remaining validation runs*, *readout ablation rollout*, and *optional follow-ups*. These are **not** renumbered as “Phase” steps. |

**Rule:** When working in Ask mode or handing off to an agent, point to **this file** for “what to do next” and **checkbox state**. Point to **`PLAN_rnn_depth_and_celltype_ablations.md`** for “what Phase 3 means technically” or to sync acceptance criteria there after work completes.

---

## Current implementation snapshot (high level)

Already in the repo (see canonical plan for file paths):

- Training passes **`n_layers`**, **`dropout`**, **`cell_type`** from `experiments.json` into `TwoModuleRNNWrapper` / `Community`.
- **Depth / dropout expansion** and **GRU pilot** condition names live in `a1b2/models/experiments.json`; **`build_run_id`** includes **`cell_type`** (`RNN` vs `GRU`) but does **not** encode `n_layers` or `dropout` (unique **`name`** required for depth/dropout).
- Docs: [`PRIMARY_GRID_RUN_INVENTORY.md`](../PRIMARY_GRID_RUN_INVENTORY.md), verification snippets in the canonical plan.

---

## Workstreams (narrative order — not checkboxes)

These are **recommended sequencing** only. They do not replace the checkbox list below.

### Workstream A — Close validation gaps (empirical)

1. Run **short training smokes** (or full grid on cluster) for: one **depth** condition, one **dropout** condition, both **GRU** pilots.
2. On finished `.npz` files, confirm **`hiddens`** are **2D** per phase with width **`n_modules × dim_hidden`** (same convention as 1-layer parents).
3. Run **one** existing analysis path (geometry / PCA / principal angles) on a **depth** run and confirm no shape or API surprises.
4. Optionally re-check **one legacy** RNN condition **without** `n_layers` / `dropout` / `cell_type` for regression.

### Workstream B — Readout coordination ablation (`common_readout`)

**Goal:** Compare **shared readout** (`common_readout=true`, default in the primary grid) to **per-module readout** (`common_readout=false`) to reduce **readout-mediated coordination** during learning. Interpretation notes: [`ARCHITECTURAL_ABLATION_NOTES.md`](../ARCHITECTURAL_ABLATION_NOTES.md) (section *Readout separation*).

**Design principles:**

- Focus on **`two_module_rnn`** (single-module has one module; readout coupling is not the same comparison).
- Add **matched pairs**: same routing, sparsity, `dim_hidden`, `nb_steps`, `init_scale`, etc.; only flip `common_readout` and use a **new unique `name`**. **Convention (no_comms pilots):** suffix **`_pr`** on the anchor name (e.g. `..._init0.001` → `..._init0.001_pr`). Older conditions may use **`_sep_readout`** in the stem instead; both map to `common_readout: false` and **`sep_pr`** in `build_run_id`.
- `build_run_id` already distinguishes readout: **`cr`** vs **`pr`** in the folder suffix.
- Start **small** (e.g. no_comms + task_routed at one init; then shared input at one init as recommended in architectural notes), then expand the grid if results are stable.

**Implementation steps (when you execute this workstream):**

1. Pick anchor condition(s) from `experiments.json`.
2. Clone each to `common_readout: false`, new `name`, identical other keys.
3. Run simulations; compare behavior and representations **within** pair (`cr` vs `pr`).
4. Update inventory / paper notes if these rows become part of the primary comparison story.

### Workstream C — Optional engineering follow-ups

- **Phase 4 (canonical plan):** extend `build_run_id` for `n_layers` and/or `dropout` with golden tests (backward compatible).
- **Phase 5 (canonical plan):** LSTM only after GRU + depth + validation are stable.
- **Phase 6 (canonical plan):** short subsection in `research_overview_no_comms_primary_grid.md` (or dedicated architecture doc) for depth / cell-type / readout axes.

---

## Operational follow-up TODO list

Use **`[ ]` / `[x]` only in this section** for tracking. Do **not** treat item numbers here as “Phase” numbers from the canonical plan.

### Validation (maps to open items in canonical plan §1.4, §2.3, §3.3)

- [ ] Forward / construct: run `python scripts/validate_ablation_continuation.py --forward` from `a1b2_modular` (requires PyTorch). Without torch, run the same script **without** `--forward` for JSON + `build_run_id` only.
- [ ] Training smoke: `scripts/02_run_simulations.py` completes for at least one **depth-expanded**, one **dropout**, and both **GRU** pilot names (or equivalent cluster jobs).
- [ ] Load `.npz`: confirm **`hiddens`** 2D and expected width for **nl1 vs nl2/nl3** (and optionally dropout) vs parent.
- [ ] Run **one** geometry / PCA / angles workflow on a **depth** run without errors.
- [ ] GRU: confirm **no mask/runtime errors** on full forward+train for both GRU pilots.
- [ ] Regression: one **legacy** RNN condition without new keys still behaves as before.

### After validation — sync canonical checklist

- [ ] In [`PLAN_rnn_depth_and_celltype_ablations.md`](PLAN_rnn_depth_and_celltype_ablations.md), tick **§1.4**, **§2.3**, **§3.3** items that you have actually completed (leave unchecked until runs + analysis checks are done).

### Readout ablation (new axis — not in original depth/cell-type phases)

- [x] Document naming convention for **per-module readout** twins (`_pr` for no_comms pilots; legacy `_sep_readout` elsewhere) in this file and `PRIMARY_GRID_RUN_INVENTORY.md`.
- [x] Add **first matched pair(s)** to `experiments.json`: `two_module_rnn_25_no_comms_nb2_init0.001_pr`, `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_pr`.
- [x] Verify **`run_id`** contains **`sep_pr`** vs **`sep_cr`** for twins (automated: `python scripts/validate_ablation_continuation.py`).
- [ ] Run simulations for new `pr` conditions (cluster).
- [ ] Compare metrics / representations **cr vs pr** for the same anchor (notebook or script path noted here when known).

### Optional

- [ ] Canonical **Phase 4**: `build_run_id` hardening for `n_layers` / `dropout`.
- [ ] Canonical **Phase 5**: LSTM support.
- [ ] Canonical **Phase 6**: reader-facing overview paragraph for all ablation axes.

---

## Command pointers

**Bundled config check (recommended first):**

```bash
cd a1b2_modular
python scripts/validate_ablation_continuation.py
python scripts/validate_ablation_continuation.py --forward   # if PyTorch installed
```

**Readout pilot training (after validation passes):**

```bash
python scripts/02_run_simulations.py two_module_rnn_25_no_comms_nb2_init0.001_pr --base-folder .
python scripts/02_run_simulations.py two_module_rnn_25_task_routed_no_comms_nb2_init0.001_pr --base-folder .
```

Further snippets (depth batch, GRU) live in **`PLAN_rnn_depth_and_celltype_ablations.md`** under *Verification commands* and *Depth + dropout expansion batch* / *GRU Phase 3 pilot*. Run from **`a1b2_modular`** unless your layout requires `--base-folder`.

---

*When this checklist grows stale, update the boxes above and optionally add dates in commit messages rather than duplicating phase text from the canonical plan.*
