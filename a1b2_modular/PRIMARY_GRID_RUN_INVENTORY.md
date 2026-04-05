# Primary grid run inventory

*Last refreshed: 2026-04-04 14:56 UTC*

All `experiments.json` conditions matching the paper comparison grid (`nb_steps=2`, `common_input=False`, `common_readout=True`, sparsity ∈ {no_comms, 0.5, 1.0}, init ∈ {0.001, 0.01, 0.1, 1, 2}, excluding `init_scope=input_only`).

- **Folder:** `data/simulations/<run_id>/` (folder name equals `run_id`).
- **Von Mises companion (name-matched):** `state_<participant_id>.pt` for each `sim_<participant_id>.npz` in the same folder.
- **Single-module baseline:** `single_module` rows use a capacity-matched hidden size for that `dim_h` column (e.g. **100** hidden units when `dim_h=50`, comparable to two modules × 50). Condition names use `single_module_rnn_<hidden>_nb2…`.
- **Total rows:** 144

### RNN depth and dropout (no_comms focal lines)

Stacked vanilla RNN depth (`n_layers` 2 or 3) is expanded across the same **init** grid as the one-layer parents for:

- **Single-module baseline:** `single_module_rnn_50_nb2` (init 1.0, no `init_scale` key), `init0.01`, `init0.1`, `init2` — each has `_nl2` / `_nl3` siblings (plus existing `init0.001_nl2` / `_nl3`).
- **Task-routed modular no_comms:** `two_module_rnn_25_task_routed_no_comms_nb2` (no `init_scale`), `init0.01`, `init0.1`, `init2` — same `_nl2` / `_nl3` pattern (plus existing `init0.001_nl2` / `_nl3`).

**Naming:** `_nl2`, `_nl3` on the parent name; **`build_run_id` does not encode `n_layers` or `dropout`**, so every condition must keep a **unique `name`**.

**Inter-layer dropout (pilot):** `"dropout": 0.1` on stacked nets only (`n_layers` 2 or 3). PyTorch applies this **between** stacked layers; with `n_layers == 1` it has no effect. Pilot names (init `0.001` only): `single_module_rnn_50_nb2_init0.001_nl2_drop0.1`, `..._nl3_drop0.1`, `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2_drop0.1`, `..._nl3_drop0.1`.

### GRU cell-type ablation (Phase 3 pilot)

`"cell_type": "GRU"` with vanilla `nn.GRU` in `Community`; training reads the key from each condition. **`build_run_id` includes the cell token** (`GRU` vs `RNN`), but names still use a `_gru` suffix for clarity.

- `single_module_rnn_50_nb2_init0.001_gru` — single-module baseline, init `0.001`
- `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_gru` — task-routed no_comms, init `0.001`

### Per-module readout pilot (`common_readout=false`)

**Naming:** suffix **`_pr`** on the **shared-readout** (`common_readout=true`) anchor; `build_run_id` uses token **`sep_pr`** instead of **`sep_cr`**.

Pilot pairs at `init_scale=0.001`, `dim_hidden=25`, no_comms (`sparsity=0`):

| Anchor (`cr`) | Per-module readout twin (`pr`) |
| --- | --- |
| `two_module_rnn_25_no_comms_nb2_init0.001` (shared input) | `two_module_rnn_25_no_comms_nb2_init0.001_pr` |
| `two_module_rnn_25_task_routed_no_comms_nb2_init0.001` | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_pr` |

Older **`_sep_readout`** conditions (e.g. full comms) remain in `experiments.json`; these `_pr` rows are the **no_comms** matched controls for the focal grid.

**Verification commands** (from `a1b2_modular`): see [docs/PLAN_rnn_depth_and_celltype_ablations.md](docs/PLAN_rnn_depth_and_celltype_ablations.md) section *Verification commands (quick)*; for a bundled config check use `python scripts/validate_ablation_continuation.py` (add `--forward` if PyTorch is installed).

| dim_h | routing | sparsity | init | condition | run_id (= folder) | exists | npz | state matched | VM OK |
| ---: | --- | --- | --- | --- | --- | :---: | ---: | --- | :---: |
| 6 | shared | 0.5 | 0.001 | two_module_rnn_6_sp05_nb2_init0.001 | `two_module_rnn_6_sp05_nb2_init0.001_nb2_shared_sp0.5_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 6 | shared | 0.5 | 0.01 | two_module_rnn_6_sp05_nb2_init0.01 | `two_module_rnn_6_sp05_nb2_init0.01_nb2_shared_sp0.5_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 0.5 | 0.1 | two_module_rnn_6_sp05_nb2_init0.1 | `two_module_rnn_6_sp05_nb2_init0.1_nb2_shared_sp0.5_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 0.5 | 1.0 | two_module_rnn_6_sp05_nb2 | `two_module_rnn_6_sp05_nb2_nb2_shared_sp0.5_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 0.5 | 2.0 | two_module_rnn_6_sp05_nb2_init2 | `two_module_rnn_6_sp05_nb2_init2_nb2_shared_sp0.5_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 6 | shared | 1.0 | 0.001 | two_module_rnn_6_nb2_init0.001 | `two_module_rnn_6_nb2_init0.001_nb2_shared_sp1.0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 1.0 | 0.01 | two_module_rnn_6_nb2_init0.01 | `two_module_rnn_6_nb2_init0.01_nb2_shared_sp1.0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 1.0 | 0.1 | two_module_rnn_6_nb2_init0.1 | `two_module_rnn_6_nb2_init0.1_nb2_shared_sp1.0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 1.0 | 1.0 | two_module_rnn_6_nb2 | `two_module_rnn_6_nb2_nb2_shared_sp1.0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 1.0 | 2.0 | two_module_rnn_6_nb2_init2 | `two_module_rnn_6_nb2_init2_nb2_shared_sp1.0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 6 | shared | no_comms | 0.001 | two_module_rnn_6_no_comms_nb2_init0.001 | `two_module_rnn_6_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 6 | shared | no_comms | 0.01 | two_module_rnn_6_no_comms_nb2_init0.01 | `two_module_rnn_6_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | shared | no_comms | 0.1 | two_module_rnn_6_no_comms_nb2_init0.1 | `two_module_rnn_6_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | shared | no_comms | 1.0 | two_module_rnn_6_no_comms_nb2 | `two_module_rnn_6_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | shared | no_comms | 2.0 | two_module_rnn_6_no_comms_nb2_init2 | `two_module_rnn_6_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 6 | single_module | 1.0 | 0.001 | single_module_rnn_12_nb2_init0.001 | `single_module_rnn_12_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 6 | single_module | 1.0 | 0.01 | single_module_rnn_12_nb2_init0.01 | `single_module_rnn_12_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | single_module | 1.0 | 0.1 | single_module_rnn_12_nb2_init0.1 | `single_module_rnn_12_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | single_module | 1.0 | 1.0 | single_module_rnn_12_nb2 | `single_module_rnn_12_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | single_module | 1.0 | 2.0 | single_module_rnn_12_nb2_init2 | `single_module_rnn_12_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 0.5 | 0.001 | two_module_rnn_6_task_routed_sp05_nb2_init0.001 | `two_module_rnn_6_task_routed_sp05_nb2_init0.001_nb2_task_routed_sp0.5_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 6 | task_routed | 0.5 | 0.01 | two_module_rnn_6_task_routed_sp05_nb2_init0.01 | `two_module_rnn_6_task_routed_sp05_nb2_init0.01_nb2_task_routed_sp0.5_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 0.5 | 0.1 | two_module_rnn_6_task_routed_sp05_nb2_init0.1 | `two_module_rnn_6_task_routed_sp05_nb2_init0.1_nb2_task_routed_sp0.5_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 0.5 | 1.0 | two_module_rnn_6_task_routed_sp05_nb2 | `two_module_rnn_6_task_routed_sp05_nb2_nb2_task_routed_sp0.5_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 0.5 | 2.0 | two_module_rnn_6_task_routed_sp05_nb2_init2 | `two_module_rnn_6_task_routed_sp05_nb2_init2_nb2_task_routed_sp0.5_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 6 | task_routed | 1.0 | 0.001 | two_module_rnn_6_task_routed_nb2_init0.001 | `two_module_rnn_6_task_routed_nb2_init0.001_nb2_task_routed_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 1.0 | 0.01 | two_module_rnn_6_task_routed_nb2_init0.01 | `two_module_rnn_6_task_routed_nb2_init0.01_nb2_task_routed_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 1.0 | 0.1 | two_module_rnn_6_task_routed_nb2_init0.1 | `two_module_rnn_6_task_routed_nb2_init0.1_nb2_task_routed_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 1.0 | 1.0 | two_module_rnn_6_task_routed_nb2 | `two_module_rnn_6_task_routed_nb2_nb2_task_routed_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 1.0 | 2.0 | two_module_rnn_6_task_routed_nb2_init2 | `two_module_rnn_6_task_routed_nb2_init2_nb2_task_routed_sp1.0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | no_comms | 0.001 | two_module_rnn_6_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | no_comms | 0.01 | two_module_rnn_6_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | no_comms | 0.1 | two_module_rnn_6_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | no_comms | 1.0 | two_module_rnn_6_task_routed_no_comms_nb2 | `two_module_rnn_6_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | no_comms | 2.0 | two_module_rnn_6_task_routed_no_comms_nb2_init2 | `two_module_rnn_6_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 12 | shared | 0.5 | 0.001 | two_module_rnn_12_sp05_nb2_init0.001 | `two_module_rnn_12_sp05_nb2_init0.001_nb2_shared_sp0.5_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 12 | shared | 0.5 | 0.01 | two_module_rnn_12_sp05_nb2_init0.01 | `two_module_rnn_12_sp05_nb2_init0.01_nb2_shared_sp0.5_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 12 | shared | 0.5 | 0.1 | two_module_rnn_12_sp05_nb2_init0.1 | `two_module_rnn_12_sp05_nb2_init0.1_nb2_shared_sp0.5_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 12 | shared | 0.5 | 1.0 | two_module_rnn_12_sp05_nb2 | `two_module_rnn_12_sp05_nb2_nb2_shared_sp0.5_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 12 | shared | 0.5 | 2.0 | two_module_rnn_12_sp05_nb2_init2 | `two_module_rnn_12_sp05_nb2_init2_nb2_shared_sp0.5_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 12 | shared | 1.0 | 0.001 | two_module_rnn_12_nb2_init0.001 | `two_module_rnn_12_nb2_init0.001_nb2_shared_sp1.0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 12 | shared | 1.0 | 0.01 | two_module_rnn_12_nb2_init0.01 | `two_module_rnn_12_nb2_init0.01_nb2_shared_sp1.0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 12 | shared | 1.0 | 0.1 | two_module_rnn_12_nb2_init0.1 | `two_module_rnn_12_nb2_init0.1_nb2_shared_sp1.0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 12 | shared | 1.0 | 1.0 | two_module_rnn_12_nb2 | `two_module_rnn_12_nb2_nb2_shared_sp1.0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 12 | shared | 1.0 | 2.0 | two_module_rnn_12_nb2_init2 | `two_module_rnn_12_nb2_init2_nb2_shared_sp1.0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 12 | shared | no_comms | 0.001 | two_module_rnn_12_no_comms_nb2_init0.001 | `two_module_rnn_12_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 12 | shared | no_comms | 0.01 | two_module_rnn_12_no_comms_nb2_init0.01 | `two_module_rnn_12_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 12 | shared | no_comms | 0.1 | two_module_rnn_12_no_comms_nb2_init0.1 | `two_module_rnn_12_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 12 | shared | no_comms | 1.0 | two_module_rnn_12_no_comms_nb2 | `two_module_rnn_12_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 12 | shared | no_comms | 2.0 | two_module_rnn_12_no_comms_nb2_init2 | `two_module_rnn_12_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 12 | single_module | 1.0 | 0.001 | single_module_rnn_25_nb2_init0.001 | `single_module_rnn_25_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 12 | single_module | 1.0 | 0.01 | single_module_rnn_25_nb2_init0.01 | `single_module_rnn_25_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 12 | single_module | 1.0 | 0.1 | single_module_rnn_25_nb2_init0.1 | `single_module_rnn_25_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 219 | 219/219 | Yes |
| 12 | single_module | 1.0 | 1.0 | single_module_rnn_25_nb2 | `single_module_rnn_25_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 12 | single_module | 1.0 | 2.0 | single_module_rnn_25_nb2_init2 | `single_module_rnn_25_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | 0.5 | 0.001 | two_module_rnn_12_task_routed_sp05_nb2_init0.001 | `two_module_rnn_12_task_routed_sp05_nb2_init0.001_nb2_task_routed_sp0.5_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 12 | task_routed | 0.5 | 0.01 | two_module_rnn_12_task_routed_sp05_nb2_init0.01 | `two_module_rnn_12_task_routed_sp05_nb2_init0.01_nb2_task_routed_sp0.5_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | 0.5 | 0.1 | two_module_rnn_12_task_routed_sp05_nb2_init0.1 | `two_module_rnn_12_task_routed_sp05_nb2_init0.1_nb2_task_routed_sp0.5_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | 0.5 | 1.0 | two_module_rnn_12_task_routed_sp05_nb2 | `two_module_rnn_12_task_routed_sp05_nb2_nb2_task_routed_sp0.5_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | 0.5 | 2.0 | two_module_rnn_12_task_routed_sp05_nb2_init2 | `two_module_rnn_12_task_routed_sp05_nb2_init2_nb2_task_routed_sp0.5_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 12 | task_routed | 1.0 | 0.001 | two_module_rnn_12_task_routed_nb2_init0.001 | `two_module_rnn_12_task_routed_nb2_init0.001_nb2_task_routed_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | 1.0 | 0.01 | two_module_rnn_12_task_routed_nb2_init0.01 | `two_module_rnn_12_task_routed_nb2_init0.01_nb2_task_routed_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | 1.0 | 0.1 | two_module_rnn_12_task_routed_nb2_init0.1 | `two_module_rnn_12_task_routed_nb2_init0.1_nb2_task_routed_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | 1.0 | 1.0 | two_module_rnn_12_task_routed_nb2 | `two_module_rnn_12_task_routed_nb2_nb2_task_routed_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | 1.0 | 2.0 | two_module_rnn_12_task_routed_nb2_init2 | `two_module_rnn_12_task_routed_nb2_init2_nb2_task_routed_sp1.0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | no_comms | 0.001 | two_module_rnn_12_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | no_comms | 0.01 | two_module_rnn_12_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | no_comms | 0.1 | two_module_rnn_12_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | no_comms | 1.0 | two_module_rnn_12_task_routed_no_comms_nb2 | `two_module_rnn_12_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | no_comms | 2.0 | two_module_rnn_12_task_routed_no_comms_nb2_init2 | `two_module_rnn_12_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 25 | shared | 0.5 | 0.001 | two_module_rnn_25_sp05_nb2_init0.001 | `two_module_rnn_25_sp05_nb2_init0.001_nb2_shared_sp0.5_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | shared | 0.5 | 0.01 | two_module_rnn_25_sp05_nb2_init0.01 | `two_module_rnn_25_sp05_nb2_init0.01_nb2_shared_sp0.5_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | shared | 0.5 | 0.1 | two_module_rnn_25_sp05_nb2_init0.1 | `two_module_rnn_25_sp05_nb2_init0.1_nb2_shared_sp0.5_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | shared | 0.5 | 1.0 | two_module_rnn_25_sp05_nb2 | `two_module_rnn_25_sp05_nb2_nb2_shared_sp0.5_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 25 | shared | 0.5 | 2.0 | two_module_rnn_25_sp05_nb2_init2 | `two_module_rnn_25_sp05_nb2_init2_nb2_shared_sp0.5_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 25 | shared | 1.0 | 0.001 | two_module_rnn_25_nb2_init0.001 | `two_module_rnn_25_nb2_init0.001_nb2_shared_sp1.0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | shared | 1.0 | 0.01 | two_module_rnn_25_nb2_init0.01 | `two_module_rnn_25_nb2_init0.01_nb2_shared_sp1.0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | shared | 1.0 | 0.1 | two_module_rnn_25_nb2_init0.1 | `two_module_rnn_25_nb2_init0.1_nb2_shared_sp1.0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | shared | 1.0 | 1.0 | two_module_rnn_25_nb2 | `two_module_rnn_25_nb2_nb2_shared_sp1.0_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 25 | shared | 1.0 | 2.0 | two_module_rnn_25_nb2_init2 | `two_module_rnn_25_nb2_init2_nb2_shared_sp1.0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 25 | shared | no_comms | 0.001 | two_module_rnn_25_no_comms_nb2_init0.001 | `two_module_rnn_25_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | shared | no_comms | 0.01 | two_module_rnn_25_no_comms_nb2_init0.01 | `two_module_rnn_25_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | shared | no_comms | 0.1 | two_module_rnn_25_no_comms_nb2_init0.1 | `two_module_rnn_25_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | shared | no_comms | 1.0 | two_module_rnn_25_no_comms_nb2 | `two_module_rnn_25_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 25 | shared | no_comms | 2.0 | two_module_rnn_25_no_comms_nb2_init2 | `two_module_rnn_25_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 0.001 | single_module_rnn_50_nb2_init0.001 | `single_module_rnn_50_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 0.001 | single_module_rnn_50_nb2_init0.001_nl2 | `single_module_rnn_50_nb2_init0.001_nl2_nb2_shared_sp1_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 25 | single_module | 1.0 | 0.001 | single_module_rnn_50_nb2_init0.001_nl3 | `single_module_rnn_50_nb2_init0.001_nl3_nb2_shared_sp1_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 25 | single_module | 1.0 | 0.01 | single_module_rnn_50_nb2_init0.01 | `single_module_rnn_50_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 0.1 | single_module_rnn_50_nb2_init0.1 | `single_module_rnn_50_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 1.0 | single_module_rnn_50_nb2 | `single_module_rnn_50_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 2.0 | single_module_rnn_50_nb2_init2 | `single_module_rnn_50_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | 0.5 | 0.001 | two_module_rnn_25_task_routed_sp05_nb2_init0.001 | `two_module_rnn_25_task_routed_sp05_nb2_init0.001_nb2_task_routed_sp0.5_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | 0.5 | 0.01 | two_module_rnn_25_task_routed_sp05_nb2_init0.01 | `two_module_rnn_25_task_routed_sp05_nb2_init0.01_nb2_task_routed_sp0.5_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | 0.5 | 0.1 | two_module_rnn_25_task_routed_sp05_nb2_init0.1 | `two_module_rnn_25_task_routed_sp05_nb2_init0.1_nb2_task_routed_sp0.5_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | 0.5 | 1.0 | two_module_rnn_25_task_routed_sp05_nb2 | `two_module_rnn_25_task_routed_sp05_nb2_nb2_task_routed_sp0.5_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 25 | task_routed | 0.5 | 2.0 | two_module_rnn_25_task_routed_sp05_nb2_init2 | `two_module_rnn_25_task_routed_sp05_nb2_init2_nb2_task_routed_sp0.5_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | 1.0 | 0.001 | two_module_rnn_25_task_routed_nb2_init0.001 | `two_module_rnn_25_task_routed_nb2_init0.001_nb2_task_routed_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | 1.0 | 0.01 | two_module_rnn_25_task_routed_nb2_init0.01 | `two_module_rnn_25_task_routed_nb2_init0.01_nb2_task_routed_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | 1.0 | 0.1 | two_module_rnn_25_task_routed_nb2_init0.1 | `two_module_rnn_25_task_routed_nb2_init0.1_nb2_task_routed_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | 1.0 | 1.0 | two_module_rnn_25_task_routed_nb2 | `two_module_rnn_25_task_routed_nb2_nb2_task_routed_sp1_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 25 | task_routed | 1.0 | 2.0 | two_module_rnn_25_task_routed_nb2_init2 | `two_module_rnn_25_task_routed_nb2_init2_nb2_task_routed_sp1.0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.001 | two_module_rnn_25_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.001 | two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 25 | task_routed | no_comms | 0.001 | two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl3 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl3_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 25 | task_routed | no_comms | 0.01 | two_module_rnn_25_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.1 | two_module_rnn_25_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 1.0 | two_module_rnn_25_task_routed_no_comms_nb2 | `two_module_rnn_25_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 25 | task_routed | no_comms | 2.0 | two_module_rnn_25_task_routed_no_comms_nb2_init2 | `two_module_rnn_25_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 50 | shared | 0.5 | 0.001 | two_module_rnn_50_sp05_nb2_init0.001 | `two_module_rnn_50_sp05_nb2_init0.001_nb2_shared_sp0.5_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 50 | shared | 0.5 | 0.01 | two_module_rnn_50_sp05_nb2_init0.01 | `two_module_rnn_50_sp05_nb2_init0.01_nb2_shared_sp0.5_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 50 | shared | 0.5 | 0.1 | two_module_rnn_50_sp05_nb2_init0.1 | `two_module_rnn_50_sp05_nb2_init0.1_nb2_shared_sp0.5_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 50 | shared | 0.5 | 1.0 | two_module_rnn_50_sp05_nb2 | `two_module_rnn_50_sp05_nb2_nb2_shared_sp0.5_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 50 | shared | 0.5 | 2.0 | two_module_rnn_50_sp05_nb2_init2 | `two_module_rnn_50_sp05_nb2_init2_nb2_shared_sp0.5_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 50 | shared | 1.0 | 0.001 | two_module_rnn_50_nb2_init0.001 | `two_module_rnn_50_nb2_init0.001_nb2_shared_sp1.0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 50 | shared | 1.0 | 0.01 | two_module_rnn_50_nb2_init0.01 | `two_module_rnn_50_nb2_init0.01_nb2_shared_sp1.0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 50 | shared | 1.0 | 0.1 | two_module_rnn_50_nb2_init0.1 | `two_module_rnn_50_nb2_init0.1_nb2_shared_sp1.0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 50 | shared | 1.0 | 1.0 | two_module_rnn_50_nb2 | `two_module_rnn_50_nb2_nb2_shared_sp1.0_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 50 | shared | 1.0 | 2.0 | two_module_rnn_50_nb2_init2 | `two_module_rnn_50_nb2_init2_nb2_shared_sp1.0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 50 | shared | no_comms | 0.001 | two_module_rnn_50_no_comms_nb2_init0.001 | `two_module_rnn_50_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 50 | shared | no_comms | 0.01 | two_module_rnn_50_no_comms_nb2_init0.01 | `two_module_rnn_50_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 50 | shared | no_comms | 0.1 | two_module_rnn_50_no_comms_nb2_init0.1 | `two_module_rnn_50_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 50 | shared | no_comms | 1.0 | two_module_rnn_50_no_comms_nb2 | `two_module_rnn_50_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 50 | shared | no_comms | 2.0 | two_module_rnn_50_no_comms_nb2_init2 | `two_module_rnn_50_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 50 | single_module | 1.0 | 0.001 | single_module_rnn_100_nb2_init0.001 | `single_module_rnn_100_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 50 | single_module | 1.0 | 0.01 | single_module_rnn_100_nb2_init0.01 | `single_module_rnn_100_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | No | 0 | — | N/A |
| 50 | single_module | 1.0 | 0.1 | single_module_rnn_100_nb2_init0.1 | `single_module_rnn_100_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | No | 0 | — | N/A |
| 50 | single_module | 1.0 | 1.0 | single_module_rnn_100_nb2 | `single_module_rnn_100_nb2_nb2_shared_sp1_sep_cr_RNN` | No | 0 | — | N/A |
| 50 | single_module | 1.0 | 2.0 | single_module_rnn_100_nb2_init2 | `single_module_rnn_100_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 50 | task_routed | 0.5 | 0.001 | two_module_rnn_50_task_routed_sp05_nb2_init0.001 | `two_module_rnn_50_task_routed_sp05_nb2_init0.001_nb2_task_routed_sp0.5_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | 0.5 | 0.01 | two_module_rnn_50_task_routed_sp05_nb2_init0.01 | `two_module_rnn_50_task_routed_sp05_nb2_init0.01_nb2_task_routed_sp0.5_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | 0.5 | 0.1 | two_module_rnn_50_task_routed_sp05_nb2_init0.1 | `two_module_rnn_50_task_routed_sp05_nb2_init0.1_nb2_task_routed_sp0.5_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | 0.5 | 1.0 | two_module_rnn_50_task_routed_sp05_nb2 | `two_module_rnn_50_task_routed_sp05_nb2_nb2_task_routed_sp0.5_sep_cr_RNN` | Yes | 154 | 0/154 | Partial |
| 50 | task_routed | 0.5 | 2.0 | two_module_rnn_50_task_routed_sp05_nb2_init2 | `two_module_rnn_50_task_routed_sp05_nb2_init2_nb2_task_routed_sp0.5_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 50 | task_routed | 1.0 | 0.001 | two_module_rnn_50_task_routed_nb2_init0.001 | `two_module_rnn_50_task_routed_nb2_init0.001_nb2_task_routed_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | 1.0 | 0.01 | two_module_rnn_50_task_routed_nb2_init0.01 | `two_module_rnn_50_task_routed_nb2_init0.01_nb2_task_routed_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | 1.0 | 0.1 | two_module_rnn_50_task_routed_nb2_init0.1 | `two_module_rnn_50_task_routed_nb2_init0.1_nb2_task_routed_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | 1.0 | 1.0 | two_module_rnn_50_task_routed_nb2 | `two_module_rnn_50_task_routed_nb2_nb2_task_routed_sp1_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 50 | task_routed | 1.0 | 2.0 | two_module_rnn_50_task_routed_nb2_init2 | `two_module_rnn_50_task_routed_nb2_init2_nb2_task_routed_sp1.0_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 50 | task_routed | no_comms | 0.001 | two_module_rnn_50_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | no_comms | 0.01 | two_module_rnn_50_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | no_comms | 0.1 | two_module_rnn_50_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | no_comms | 1.0 | two_module_rnn_50_task_routed_no_comms_nb2 | `two_module_rnn_50_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 50 | task_routed | no_comms | 2.0 | two_module_rnn_50_task_routed_no_comms_nb2_init2 | `two_module_rnn_50_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
