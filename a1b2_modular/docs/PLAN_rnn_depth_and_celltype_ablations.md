# Implementation plan: RNN depth and recurrent cell-type ablations

This document is the **canonical checklist** for agents implementing modular/single-module RNN ablations in `a1b2_modular`. Work through items **in order** unless a step is explicitly parallelizable. After each phase, run the verification commands in §6.

**Handoff / remaining operational TODOs** (validation runs, readout ablation rollout) live in a **separate** file so “Phase 1–6” is not confused with day-to-day checkboxes: [`ABLATION_CONTINUATION_AND_TODOS.md`](ABLATION_CONTINUATION_AND_TODOS.md).

---

## Goals

1. **Depth:** Support `n_layers > 1` (and optional `dropout > 0` between stacked layers) for `two_module_rnn` and `single_module_rnn`, driven from `experiments.json`, with unique run folders and reproducible analysis.
2. **Cell type:** Support **`GRU`** via configuration only (already partially wired). Optionally add **`LSTM`** with correct masking and forward pass.
3. **No change** to the A1–B–A2 schedule, loss logic, or data pipeline unless a bug is discovered during testing.

**Out of scope for this plan:** New paper figures, full primary-grid re-run inventory, communication (non-zero sparsity) conditions.

---

## Current state (audit summary)

| Component | Depth (`n_layers`) | `dropout` | `cell_type` from condition |
|-----------|--------------------|-----------|----------------------------|
| `Community` | Supported in `nn.RNN` / `nn.GRU` via `num_layers=` | Constructor accepts `dropout` | `cell_types_dict` in `community.py` |
| `TwoModuleRNNWrapper` | Constructor args `n_layers`, `dropout` | Passed into config | Passed into config |
| `simulation.run_simulation` | **Passed** from condition (default `1`) | **Passed** (default `0.0`) | **Passed** (`condition.get("cell_type", "RNN")`) |

**Status:** `n_layers` / `dropout` are wired in `simulation.py`; new depth conditions use a **unique `name`** (e.g. `_nl2`, `_nl3`) so `build_run_id` folders stay distinct.

**Reference files:**

- `a1b2/training/simulation.py` — `TwoModuleRNNWrapper(...)` construction
- `a1b2/models/two_module_rnn.py` — wrapper + `build_community_rnn_config`
- `a1b2/models/community.py` — stacked RNN, masks, `cell_types_dict`, `forward`
- `a1b2/utils/run_config.py` — `build_run_id` (folder uniqueness)
- `a1b2/models/rnn_init.py` — `apply_init_scale` (`weight_ih*` for `input_only`)

---

## Phase 1 — Plumb depth and dropout into training

### 1.1 Code changes

1. In **`a1b2/training/simulation.py`**, when constructing `TwoModuleRNNWrapper`, add:

   - `n_layers=condition.get("n_layers", 1)`
   - `dropout=condition.get("dropout", 0.0)`

   Use defaults so existing conditions without these keys behave as today.

2. **Optional but recommended:** If `n_layers < 1`, raise a clear `ValueError`.

3. **Document** in a short comment at the call site that `dim_hidden` passed to `runSchedule` remains `n_modules * hidden_size` (per-layer width convention used elsewhere); stacked layers do not change that scalar used for tensor allocation in `runSchedule` — do **not** multiply by `n_layers` here unless you change `schedule.py` array sizing (not required for standard PyTorch `nn.RNN`/`nn.GRU` interface).

### 1.2 Run ID / folder collision

- Today, **`build_run_id` does not encode `n_layers` or `dropout`**. Collisions are avoided if every distinct setup has a **distinct `name`** in `experiments.json`.

**Dropout rows:** Use a **unique `name`** that encodes dropout in the string (e.g. `_nl2_drop0.1`), not only the JSON field. Inter-layer `dropout` in PyTorch RNN/GRU applies **between stacked layers** when `n_layers > 1`; for `n_layers == 1` it has no effect.

**Agent rule:** Every new ablation condition MUST use a **unique `name`** (e.g. suffix `_nl2`, `_nl3`, `_drop01`).

**Follow-up (Phase 4, optional):** Extend `build_run_id` to append `nl{n_layers}` and `do{dropout}` (with sane formatting) when `n_layers != 1` or `dropout != 0`, so future refactors cannot accidentally reuse names.

### 1.3 Example `experiments.json` entries (after Phase 1)

Add **minimal** test conditions, e.g.:

- Clone an existing `two_module_rnn_25_task_routed_no_comms_nb2_init0.001` → add `"n_layers": 2`, new name.
- Clone an existing `single_module_rnn_50_nb2_init0.001` → add `"n_layers": 2`, new name.

Do not expand the full primary grid until smoke tests pass.

### 1.4 Acceptance criteria (Phase 1)

- [ ] `python -c` import + construct network from a parsed condition with `n_layers=2` matches parameter groups (no shape errors).
- [ ] One short simulation run (or single participant) completes and writes `.npz` with expected keys (`scripts/02_run_simulations.py <condition>` or single-participant `run_simulation`; requires PyTorch).
- [x] Existing conditions **without** `n_layers` / `dropout` still run unchanged (regression; defaults remain `1` and `0.0`).

---

## Phase 2 — Verify hidden-state logging and geometry

### 2.1 What gets stored

Training stores `hiddens` from whatever `TwoModuleRNNWrapper.forward` returns as `hid`. For multi-layer RNNs, PyTorch final state has a **layer** dimension; the wrapper indexes `all_states` and **uses the last layer** (`final_states[-1, ...]`) — documented in `two_module_rnn.py`.

### 2.2 Analysis impact

- **PCA / `n_pcs_*` / principal angles** in notebooks and `transfer_interference`-style loaders assume finite 2D hiddens per trial. If shape changes unexpectedly, fix **wrapper output** (not every notebook).

### 2.3 Acceptance criteria (Phase 2)

- [ ] Load one `.npz` from `n_layers=1` vs `n_layers=2` run: `hiddens` shape is consistent across phases and matches downstream expectations (wrapper is implemented to return 2D **last-layer** `hid` of width `n_modules * dim_hidden`; confirm on saved arrays after training).
- [ ] Run one existing geometry cell (or small script) on `n_layers=2` without errors.

---

## Phase 3 — GRU ablation (configuration-first)

### 3.1 Code

- **Likely none** beyond Phase 1 if `cell_type` is already passed (verify in `simulation.py`).
- Confirm `apply_init_scale` still applies to GRU `weight_ih_*` / `weight_hh_*` as expected for `input_only` scope.

### 3.2 Config

- Duplicate a known-good RNN condition; set `"cell_type": "GRU"`, new `name`, same `init_scale` / routing / sparsity.

### 3.3 Acceptance criteria (Phase 3)

- [x] Pilot rows in `experiments.json` with unique `_gru` names; `build_run_id` differs from RNN parent (includes `GRU` token).
- [ ] Training completes for at least one `two_module_rnn` and one `single_module_rnn` GRU condition (`scripts/02_run_simulations.py` with PyTorch).
- [ ] No mask-related runtime errors from `Community` (`gru=True` branch in `comms_mask` / `state_mask`) — confirm during training forward.

**Pilot condition names:** `single_module_rnn_50_nb2_init0.001_gru`, `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_gru`.

---

## Phase 4 (optional) — Harden `build_run_id`

Add optional tokens when non-default:

- `n_layers` when `!= 1`
- `dropout` when `> 0`
- Keep backward compatibility: if omitted from condition, behave as today.

**Acceptance:** Same condition dict → same run_id as before for all existing JSON entries (golden-file test: hash of `build_run_id` for 3–5 representative conditions before/after).

---

## Phase 5 — LSTM support (only after Phases 1–3 stable)

### 5.1 Why separate phase

`Community` gate masking uses a **GRU-specific 3× repeat** on masks (`repeat_interleave(3, 0)`). **LSTM** uses **four** linear maps per direction (`i, f, g, o`). Blindly adding `nn.LSTM` to `cell_types_dict` **will break** masks or silently mis-apply them.

### 5.2 Implementation sketch

1. Add `"LSTM": nn.LSTM` to `cell_types_dict`.
2. Generalize mask helpers:

   - Replace `gru: bool` with something like `cell_family: "rnn" | "gru" | "lstm"` or derive `n_gate_chunks` ∈ {1, 3, 4} from `cell_type`.
   - `state_mask` / `comms_mask`: stack/repeat masks to match **flattened gate dimension** of `weight_ih` / `weight_hh` for that cell (match PyTorch parameter layout).
3. **`Community.forward`:** `nn.LSTM` returns `(output, (h_n, c_n))`. RNN/GRU return `(output, h_n)`. Branch on cell type:

   - Combine core/comms contributions consistently with current design (`core_out[0] + comms_out[0]`, etc.).
   - Expose **hidden** for logging: typically **`h_n` of last layer** (not cell state `c_n`) unless the project explicitly wants cell state (unlikely for PCA on “activations”).

4. **Parametrization loops** (`Masked_weight` on `weight_hh` / `weight_ih`): verify **parameter names** for multi-layer LSTM (`weight_ih_l0`, …) still match string checks (`n[-1]` layer index).

### 5.3 Acceptance criteria (Phase 5)

- [ ] Unit-style test: forward pass `Community` with `n_modules=2`, `n_layers=1` and `n_layers=2`, `cell_type="LSTM"`, small batch, no NaNs.
- [ ] One full short simulation `.npz` written.
- [ ] Compare **parameter count** / mask nonzero pattern to a manual sanity check (optional).

---

## Phase 6 — Documentation and inventory

1. Update **`PRIMARY_GRID_RUN_INVENTORY.md`** (or sibling doc): document that `n_layers`, `dropout`, `cell_type` are supported knobs and list **naming convention** for new conditions.
2. Add a **short subsection** to `docs/research_overview_no_comms_primary_grid.md` (or separate `docs/ARCHITECTURE_ablations.md`) describing depth/cell-type axes for readers.

---

## Suggested implementation order (for agents)

| Order | Task | Depends on |
|-------|------|------------|
| A | Phase 1 — plumb `n_layers`, `dropout` | — |
| B | Phase 2 — validate `hiddens` / geometry | A |
| C | Phase 3 — GRU JSON + smoke runs | A |
| D | Phase 4 — optional `build_run_id` | A (optional anytime after A) |
| E | Phase 5 — LSTM | B, C recommended |
| F | Phase 6 — docs | A–E as completed |

---

## Verification commands (quick)

From `a1b2_modular` (adjust condition names to those added in JSON):

```bash
# JSON valid
python3 -c "import json; json.load(open('a1b2/models/experiments.json'))"

# build_run_id resolves
python3 -c "
import json
from a1b2.utils.run_config import build_run_id
s=json.load(open('a1b2/models/experiments.json'))
# Replace with your new condition name:
c=next(x for x in s['conditions'] if x['name']=='YOUR_CONDITION_NAME')
print(build_run_id(c))
"

# Short simulation (if 02_run_simulations supports your condition name)
python3 scripts/02_run_simulations.py YOUR_CONDITION_NAME
```

### Depth + dropout expansion batch (wave 2)

After adding the 16 depth clones and 4 `dropout: 0.1` pilots, from `a1b2_modular`:

```bash
# Unique condition names (entire file)
python3 -c "
import json
from collections import Counter
s=json.load(open('a1b2/models/experiments.json'))
names=[c['name'] for c in s['conditions']]
assert not [n for n,k in Counter(names).items() if k>1]
print('ok', len(names))
"

# build_run_id distinct for all newly added depth/dropout conditions
python3 -c "
import json
from a1b2.utils.run_config import build_run_id
NEW = [
  'single_module_rnn_50_nb2_nl2','single_module_rnn_50_nb2_nl3',
  'single_module_rnn_50_nb2_init0.1_nl2','single_module_rnn_50_nb2_init0.1_nl3',
  'single_module_rnn_50_nb2_init0.01_nl2','single_module_rnn_50_nb2_init0.01_nl3',
  'single_module_rnn_50_nb2_init2_nl2','single_module_rnn_50_nb2_init2_nl3',
  'two_module_rnn_25_task_routed_no_comms_nb2_nl2','two_module_rnn_25_task_routed_no_comms_nb2_nl3',
  'two_module_rnn_25_task_routed_no_comms_nb2_init0.1_nl2','two_module_rnn_25_task_routed_no_comms_nb2_init0.1_nl3',
  'two_module_rnn_25_task_routed_no_comms_nb2_init0.01_nl2','two_module_rnn_25_task_routed_no_comms_nb2_init0.01_nl3',
  'two_module_rnn_25_task_routed_no_comms_nb2_init2_nl2','two_module_rnn_25_task_routed_no_comms_nb2_init2_nl3',
  'single_module_rnn_50_nb2_init0.001_nl2_drop0.1','single_module_rnn_50_nb2_init0.001_nl3_drop0.1',
  'two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2_drop0.1','two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl3_drop0.1',
]
s=json.load(open('a1b2/models/experiments.json'))
by={c['name']:c for c in s['conditions']}
rids=[build_run_id(by[n]) for n in NEW]
assert len(rids)==len(set(rids)), sorted(rids)
for n in NEW:
    print(n, '->', build_run_id(by[n]))
"

# Forward smoke (requires torch): example task_routed + dropout
python3 -c "
import json, torch
from a1b2.models.two_module_rnn import TwoModuleRNNWrapper
s=json.load(open('a1b2/models/experiments.json'))
name='two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2_drop0.1'
c=next(x for x in s['conditions'] if x['name']==name)
net=TwoModuleRNNWrapper(
  input_size=12, output_size=4, hidden_size=c['dim_hidden'],
  n_modules=c.get('n_modules', 2 if c['arch']=='two_module_rnn' else 1),
  n_layers=int(c.get('n_layers',1)), dropout=float(c.get('dropout',0.0)),
  sparsity=c.get('sparsity',1.0), common_input=c.get('common_input',False),
  common_readout=c.get('common_readout',True), input_routing=c.get('input_routing','shared'),
)
x=torch.randn(4,12)
out,hid=net(x, feature_probe=torch.tensor([0,1,0,1]))
print(out.shape, hid.shape)
"

# Simulation smokes (pick any three)
python3 scripts/02_run_simulations.py single_module_rnn_50_nb2_init0.01_nl2 --base-folder .
python3 scripts/02_run_simulations.py two_module_rnn_25_task_routed_no_comms_nb2_init2_nl3 --base-folder .
python3 scripts/02_run_simulations.py two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2_drop0.1 --base-folder .
```

### GRU Phase 3 pilot

```bash
# build_run_id (from a1b2_modular)
python3 -c "
import json
from a1b2.utils.run_config import build_run_id
s=json.load(open('a1b2/models/experiments.json'))
by={c['name']:c for c in s['conditions']}
for n in ['single_module_rnn_50_nb2_init0.001_gru','two_module_rnn_25_task_routed_no_comms_nb2_init0.001_gru']:
    print(n, '->', build_run_id(by[n]))
"

# Forward smoke (requires torch); pass cell_type from JSON
python3 -c "
import json, torch
from a1b2.models.two_module_rnn import TwoModuleRNNWrapper
s=json.load(open('a1b2/models/experiments.json'))
name='two_module_rnn_25_task_routed_no_comms_nb2_init0.001_gru'
c=next(x for x in s['conditions'] if x['name']==name)
net=TwoModuleRNNWrapper(
  input_size=12, output_size=4, hidden_size=c['dim_hidden'],
  n_modules=c.get('n_modules', 2 if c['arch']=='two_module_rnn' else 1),
  n_layers=int(c.get('n_layers',1)), dropout=float(c.get('dropout',0.0)),
  sparsity=c.get('sparsity',1.0), common_input=c.get('common_input',False),
  common_readout=c.get('common_readout',True), cell_type=c.get('cell_type','RNN'),
  input_routing=c.get('input_routing','shared'),
)
x=torch.randn(4,12)
out,hid=net(x, feature_probe=torch.tensor([0,1,0,1]))
print(out.shape, hid.shape)
"

python3 scripts/02_run_simulations.py single_module_rnn_50_nb2_init0.001_gru --base-folder .
python3 scripts/02_run_simulations.py two_module_rnn_25_task_routed_no_comms_nb2_init0.001_gru --base-folder .
```

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Silent run folder overwrite | Unique `name` per condition; Phase 4 hardening |
| Wrong hidden tensor for PCA | Phase 2 explicit shape check; document last-layer convention |
| LSTM mask shape bugs | Phase 5 isolated test + small `batch_size` forward |
| Training instability with depth / GRU | Keep `init_scale` grid; may need LR sweep (out of scope unless requested) |

---

*Last updated: plan for agent execution. Edit this file when phases complete or scope changes.*
