# Primary grid run inventory

Auto-generated: all `experiments.json` conditions matching the paper comparison grid (`nb_steps=2`, `common_input=False`, `common_readout=True`, sparsity ∈ {no_comms, 0.5, 1.0}, init ∈ {0.001, 0.01, 0.1, 1, 2}, excluding `init_scope=input_only`).

- **Folder:** `data/simulations/<run_id>/` (folder name equals `run_id`).
- **Von Mises companion (name-matched):** `state_<participant_id>.pt` for each `sim_<participant_id>.npz` in the same folder.
- **Total rows:** 135

### RNN depth ablation (`n_layers` 2 or 3)

Four pilot conditions extend the **no-comms / task_routed@25** and **single-module@50** baselines (`init_scale=0.001`) with stacked vanilla RNN layers. Each uses a **unique `name`** suffix (`_nl2`, `_nl3`) because `build_run_id` does not encode `n_layers`. Training passes `n_layers` and optional `dropout` from JSON into `TwoModuleRNNWrapper`; stored `hiddens` remain the **last-layer** state with width `n_modules * dim_hidden` (same as the one-layer parent).

| Baseline parent | Depth condition | run_id (= folder) |
| --- | --- | --- |
| `two_module_rnn_25_task_routed_no_comms_nb2_init0.001` | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2` | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2_nb2_task_routed_sp0_sep_cr_RNN_init0.001` |
| same | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl3` | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl3_nb2_task_routed_sp0_sep_cr_RNN_init0.001` |
| `single_module_rnn_50_nb2_init0.001` | `single_module_rnn_50_nb2_init0.001_nl2` | `single_module_rnn_50_nb2_init0.001_nl2_nb2_shared_sp1_sep_cr_RNN_init0.001` |
| same | `single_module_rnn_50_nb2_init0.001_nl3` | `single_module_rnn_50_nb2_init0.001_nl3_nb2_shared_sp1_sep_cr_RNN_init0.001` |

| dim_h | routing | sparsity | init | condition | run_id (= folder) | exists | npz | state matched | VM OK |
| ---: | --- | --- | --- | --- | --- | :---: | ---: | --- | :---: |
| 6 | shared | 0.5 | 0.001 | two_module_rnn_6_sp05_nb2_init0.001 | `two_module_rnn_6_sp05_nb2_init0.001_nb2_shared_sp0.5_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 6 | shared | 0.5 | 0.01 | two_module_rnn_6_sp05_nb2_init0.01 | `two_module_rnn_6_sp05_nb2_init0.01_nb2_shared_sp0.5_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 0.5 | 0.1 | two_module_rnn_6_sp05_nb2_init0.1 | `two_module_rnn_6_sp05_nb2_init0.1_nb2_shared_sp0.5_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 0.5 | 1.0 | two_module_rnn_6_sp05_nb2 | `two_module_rnn_6_sp05_nb2_nb2_shared_sp0.5_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 0.5 | 2.0 | two_module_rnn_6_sp05_nb2_init2 | `two_module_rnn_6_sp05_nb2_init2_nb2_shared_sp0.5_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 6 | shared | 1.0 | 0.001 | two_module_rnn_6_nb2_init0.001 | `two_module_rnn_6_nb2_init0.001_nb2_shared_sp1.0_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 6 | shared | 1.0 | 0.01 | two_module_rnn_6_nb2_init0.01 | `two_module_rnn_6_nb2_init0.01_nb2_shared_sp1.0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 1.0 | 0.1 | two_module_rnn_6_nb2_init0.1 | `two_module_rnn_6_nb2_init0.1_nb2_shared_sp1.0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 1.0 | 1.0 | two_module_rnn_6_nb2 | `two_module_rnn_6_nb2_nb2_shared_sp1.0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | shared | 1.0 | 2.0 | two_module_rnn_6_nb2_init2 | `two_module_rnn_6_nb2_init2_nb2_shared_sp1.0_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 6 | shared | no_comms | 0.001 | two_module_rnn_6_no_comms_nb2_init0.001 | `two_module_rnn_6_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 6 | shared | no_comms | 0.01 | two_module_rnn_6_no_comms_nb2_init0.01 | `two_module_rnn_6_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | shared | no_comms | 0.1 | two_module_rnn_6_no_comms_nb2_init0.1 | `two_module_rnn_6_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | shared | no_comms | 1.0 | two_module_rnn_6_no_comms_nb2 | `two_module_rnn_6_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | shared | no_comms | 2.0 | two_module_rnn_6_no_comms_nb2_init2 | `two_module_rnn_6_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 6 | single_module | 1.0 | 0.001 | single_module_rnn_12_nb2_init0.001 | `single_module_rnn_12_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 6 | single_module | 1.0 | 0.01 | single_module_rnn_12_nb2_init0.01 | `single_module_rnn_12_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 199 | 199/199 | Yes |
| 6 | single_module | 1.0 | 0.1 | single_module_rnn_12_nb2_init0.1 | `single_module_rnn_12_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 202 | 202/202 | Yes |
| 6 | single_module | 1.0 | 1.0 | single_module_rnn_12_nb2 | `single_module_rnn_12_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | single_module | 1.0 | 2.0 | single_module_rnn_12_nb2_init2 | `single_module_rnn_12_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 0.5 | 0.001 | two_module_rnn_6_task_routed_sp05_nb2_init0.001 | `two_module_rnn_6_task_routed_sp05_nb2_init0.001_nb2_task_routed_sp0.5_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 6 | task_routed | 0.5 | 0.01 | two_module_rnn_6_task_routed_sp05_nb2_init0.01 | `two_module_rnn_6_task_routed_sp05_nb2_init0.01_nb2_task_routed_sp0.5_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 0.5 | 0.1 | two_module_rnn_6_task_routed_sp05_nb2_init0.1 | `two_module_rnn_6_task_routed_sp05_nb2_init0.1_nb2_task_routed_sp0.5_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 0.5 | 1.0 | two_module_rnn_6_task_routed_sp05_nb2 | `two_module_rnn_6_task_routed_sp05_nb2_nb2_task_routed_sp0.5_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 0.5 | 2.0 | two_module_rnn_6_task_routed_sp05_nb2_init2 | `two_module_rnn_6_task_routed_sp05_nb2_init2_nb2_task_routed_sp0.5_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 6 | task_routed | 1.0 | 0.001 | two_module_rnn_6_task_routed_nb2_init0.001 | `two_module_rnn_6_task_routed_nb2_init0.001_nb2_task_routed_sp1_sep_cr_RNN_init0.001` | Yes | 181 | 181/181 | Yes |
| 6 | task_routed | 1.0 | 0.01 | two_module_rnn_6_task_routed_nb2_init0.01 | `two_module_rnn_6_task_routed_nb2_init0.01_nb2_task_routed_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 1.0 | 0.1 | two_module_rnn_6_task_routed_nb2_init0.1 | `two_module_rnn_6_task_routed_nb2_init0.1_nb2_task_routed_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 1.0 | 1.0 | two_module_rnn_6_task_routed_nb2 | `two_module_rnn_6_task_routed_nb2_nb2_task_routed_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | 1.0 | 2.0 | two_module_rnn_6_task_routed_nb2_init2 | `two_module_rnn_6_task_routed_nb2_init2_nb2_task_routed_sp1.0_sep_cr_RNN_init2` | Yes | 180 | 180/180 | Yes |
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
| 12 | shared | 1.0 | 0.001 | two_module_rnn_12_nb2_init0.001 | `two_module_rnn_12_nb2_init0.001_nb2_shared_sp1.0_sep_cr_RNN_init0.001` | Yes | 188 | 188/188 | Yes |
| 12 | shared | 1.0 | 0.01 | two_module_rnn_12_nb2_init0.01 | `two_module_rnn_12_nb2_init0.01_nb2_shared_sp1.0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 12 | shared | 1.0 | 0.1 | two_module_rnn_12_nb2_init0.1 | `two_module_rnn_12_nb2_init0.1_nb2_shared_sp1.0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 12 | shared | 1.0 | 1.0 | two_module_rnn_12_nb2 | `two_module_rnn_12_nb2_nb2_shared_sp1.0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 12 | shared | 1.0 | 2.0 | two_module_rnn_12_nb2_init2 | `two_module_rnn_12_nb2_init2_nb2_shared_sp1.0_sep_cr_RNN_init2` | Yes | 189 | 189/189 | Yes |
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
| 12 | task_routed | 1.0 | 0.001 | two_module_rnn_12_task_routed_nb2_init0.001 | `two_module_rnn_12_task_routed_nb2_init0.001_nb2_task_routed_sp1_sep_cr_RNN_init0.001` | Yes | 189 | 189/189 | Yes |
| 12 | task_routed | 1.0 | 0.01 | two_module_rnn_12_task_routed_nb2_init0.01 | `two_module_rnn_12_task_routed_nb2_init0.01_nb2_task_routed_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | 1.0 | 0.1 | two_module_rnn_12_task_routed_nb2_init0.1 | `two_module_rnn_12_task_routed_nb2_init0.1_nb2_task_routed_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | 1.0 | 1.0 | two_module_rnn_12_task_routed_nb2 | `two_module_rnn_12_task_routed_nb2_nb2_task_routed_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | 1.0 | 2.0 | two_module_rnn_12_task_routed_nb2_init2 | `two_module_rnn_12_task_routed_nb2_init2_nb2_task_routed_sp1.0_sep_cr_RNN_init2` | Yes | 188 | 188/188 | Yes |
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
| 50 | shared | 1.0 | 2.0 | two_module_rnn_50_nb2_init2 | `two_module_rnn_50_nb2_init2_nb2_shared_sp1.0_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 50 | shared | no_comms | 0.001 | two_module_rnn_50_no_comms_nb2_init0.001 | `two_module_rnn_50_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 50 | shared | no_comms | 0.01 | two_module_rnn_50_no_comms_nb2_init0.01 | `two_module_rnn_50_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 50 | shared | no_comms | 0.1 | two_module_rnn_50_no_comms_nb2_init0.1 | `two_module_rnn_50_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | No | 0 | — | N/A |
| 50 | shared | no_comms | 1.0 | two_module_rnn_50_no_comms_nb2 | `two_module_rnn_50_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 50 | shared | no_comms | 2.0 | two_module_rnn_50_no_comms_nb2_init2 | `two_module_rnn_50_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | No | 0 | — | N/A |
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
| 50 | task_routed | no_comms | 0.001 | two_module_rnn_50_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 50 | task_routed | no_comms | 0.01 | two_module_rnn_50_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | no_comms | 0.1 | two_module_rnn_50_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | no_comms | 1.0 | two_module_rnn_50_task_routed_no_comms_nb2 | `two_module_rnn_50_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 0/305 | Partial |
| 50 | task_routed | no_comms | 2.0 | two_module_rnn_50_task_routed_no_comms_nb2_init2 | `two_module_rnn_50_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | No | 0 | — | N/A |
