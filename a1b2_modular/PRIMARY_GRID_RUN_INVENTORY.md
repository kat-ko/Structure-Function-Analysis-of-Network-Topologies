# Primary grid run inventory

*Last refreshed: 2026-04-08 12:45 UTC*

All `experiments.json` conditions matching the primary comparison grid (`nb_steps=2`, `common_input=False`, `common_readout=True`, init ∈ {0.001, 0.01, 0.1, 1, 2}); for `two_module_rnn`, the primary grid uses **no_comms only** (`sparsity=0`).

- **Folder:** `data/simulations/<run_id>/` (folder name equals `run_id`).
- **Von Mises companion (name-matched):** `state_<participant_id>.pt` for each `sim_<participant_id>.npz` in the same folder.
- **Single-module baseline:** `single_module` rows use a capacity-matched hidden size for that `dim_h` column (e.g. **100** hidden units when `dim_h=50`, comparable to two modules × 50). Condition names use `single_module_rnn_<hidden>_nb2…`.
- **Total rows:** 86

| dim_h | routing | sparsity | init | condition | run_id (= folder) | exists | npz | state matched | VM OK |
| ---: | --- | --- | --- | --- | --- | :---: | ---: | --- | :---: |
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
| 6 | task_routed | no_comms | 0.001 | two_module_rnn_6_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | no_comms | 0.01 | two_module_rnn_6_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | no_comms | 0.1 | two_module_rnn_6_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | no_comms | 1.0 | two_module_rnn_6_task_routed_no_comms_nb2 | `two_module_rnn_6_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 6 | task_routed | no_comms | 2.0 | two_module_rnn_6_task_routed_no_comms_nb2_init2 | `two_module_rnn_6_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
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
| 12 | task_routed | no_comms | 0.001 | two_module_rnn_12_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | no_comms | 0.01 | two_module_rnn_12_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | no_comms | 0.1 | two_module_rnn_12_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | no_comms | 1.0 | two_module_rnn_12_task_routed_no_comms_nb2 | `two_module_rnn_12_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 12 | task_routed | no_comms | 2.0 | two_module_rnn_12_task_routed_no_comms_nb2_init2 | `two_module_rnn_12_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 25 | shared | no_comms | 0.001 | two_module_rnn_25_no_comms_nb2_init0.001 | `two_module_rnn_25_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | shared | no_comms | 0.01 | two_module_rnn_25_no_comms_nb2_init0.01 | `two_module_rnn_25_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | shared | no_comms | 0.1 | two_module_rnn_25_no_comms_nb2_init0.1 | `two_module_rnn_25_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | shared | no_comms | 1.0 | two_module_rnn_25_no_comms_nb2 | `two_module_rnn_25_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 25 | shared | no_comms | 2.0 | two_module_rnn_25_no_comms_nb2_init2 | `two_module_rnn_25_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 0.001 | single_module_rnn_50_nb2_init0.001_gru | `single_module_rnn_50_nb2_init0.001_gru_nb2_shared_sp1_sep_cr_GRU_init0.001` | No | 0 | — | N/A |
| 25 | single_module | 1.0 | 0.001 | single_module_rnn_50_nb2_init0.001 | `single_module_rnn_50_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 0.001 | single_module_rnn_50_nb2_init0.001_nl2_drop0.1 | `single_module_rnn_50_nb2_init0.001_nl2_drop0.1_nb2_shared_sp1_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 25 | single_module | 1.0 | 0.001 | single_module_rnn_50_nb2_init0.001_nl2 | `single_module_rnn_50_nb2_init0.001_nl2_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 0.001 | single_module_rnn_50_nb2_init0.001_nl3_drop0.1 | `single_module_rnn_50_nb2_init0.001_nl3_drop0.1_nb2_shared_sp1_sep_cr_RNN_init0.001` | No | 0 | — | N/A |
| 25 | single_module | 1.0 | 0.001 | single_module_rnn_50_nb2_init0.001_nl3 | `single_module_rnn_50_nb2_init0.001_nl3_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 0.01 | single_module_rnn_50_nb2_init0.01 | `single_module_rnn_50_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 0.01 | single_module_rnn_50_nb2_init0.01_nl2 | `single_module_rnn_50_nb2_init0.01_nl2_nb2_shared_sp1_sep_cr_RNN_init0.01` | No | 0 | — | N/A |
| 25 | single_module | 1.0 | 0.01 | single_module_rnn_50_nb2_init0.01_nl3 | `single_module_rnn_50_nb2_init0.01_nl3_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 35 | 35/35 | Yes |
| 25 | single_module | 1.0 | 0.1 | single_module_rnn_50_nb2_init0.1 | `single_module_rnn_50_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 0.1 | single_module_rnn_50_nb2_init0.1_nl2 | `single_module_rnn_50_nb2_init0.1_nl2_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 41 | 41/41 | Yes |
| 25 | single_module | 1.0 | 0.1 | single_module_rnn_50_nb2_init0.1_nl3 | `single_module_rnn_50_nb2_init0.1_nl3_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 35 | 35/35 | Yes |
| 25 | single_module | 1.0 | 1.0 | single_module_rnn_50_nb2 | `single_module_rnn_50_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 1.0 | single_module_rnn_50_nb2_nl2 | `single_module_rnn_50_nb2_nl2_nb2_shared_sp1_sep_cr_RNN` | Yes | 41 | 41/41 | Yes |
| 25 | single_module | 1.0 | 1.0 | single_module_rnn_50_nb2_nl3 | `single_module_rnn_50_nb2_nl3_nb2_shared_sp1_sep_cr_RNN` | Yes | 35 | 35/35 | Yes |
| 25 | single_module | 1.0 | 2.0 | single_module_rnn_50_nb2_init2 | `single_module_rnn_50_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 25 | single_module | 1.0 | 2.0 | single_module_rnn_50_nb2_init2_nl2 | `single_module_rnn_50_nb2_init2_nl2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 41 | 41/41 | Yes |
| 25 | single_module | 1.0 | 2.0 | single_module_rnn_50_nb2_init2_nl3 | `single_module_rnn_50_nb2_init2_nl3_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 40 | 40/40 | Yes |
| 25 | task_routed | no_comms | 0.001 | two_module_rnn_25_task_routed_no_comms_nb2_init0.001_gru | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_gru_nb2_task_routed_sp0_sep_cr_GRU_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.001 | two_module_rnn_25_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.001 | two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2_drop0.1 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2_drop0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.001 | two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl2_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.001 | two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl3_drop0.1 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl3_drop0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.001 | two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl3 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nl3_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.01 | two_module_rnn_25_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.01 | two_module_rnn_25_task_routed_no_comms_nb2_init0.01_nl2 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.01_nl2_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.01 | two_module_rnn_25_task_routed_no_comms_nb2_init0.01_nl3 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.01_nl3_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.1 | two_module_rnn_25_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.1 | two_module_rnn_25_task_routed_no_comms_nb2_init0.1_nl2 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.1_nl2_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 0.1 | two_module_rnn_25_task_routed_no_comms_nb2_init0.1_nl3 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.1_nl3_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 1.0 | two_module_rnn_25_task_routed_no_comms_nb2 | `two_module_rnn_25_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 1.0 | two_module_rnn_25_task_routed_no_comms_nb2_nl2 | `two_module_rnn_25_task_routed_no_comms_nb2_nl2_nb2_task_routed_sp0_sep_cr_RNN` | No | 0 | — | N/A |
| 25 | task_routed | no_comms | 1.0 | two_module_rnn_25_task_routed_no_comms_nb2_nl3 | `two_module_rnn_25_task_routed_no_comms_nb2_nl3_nb2_task_routed_sp0_sep_cr_RNN` | No | 0 | — | N/A |
| 25 | task_routed | no_comms | 2.0 | two_module_rnn_25_task_routed_no_comms_nb2_init2 | `two_module_rnn_25_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 25 | task_routed | no_comms | 2.0 | two_module_rnn_25_task_routed_no_comms_nb2_init2_nl2 | `two_module_rnn_25_task_routed_no_comms_nb2_init2_nl2_nb2_task_routed_sp0_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 25 | task_routed | no_comms | 2.0 | two_module_rnn_25_task_routed_no_comms_nb2_init2_nl3 | `two_module_rnn_25_task_routed_no_comms_nb2_init2_nl3_nb2_task_routed_sp0_sep_cr_RNN_init2` | No | 0 | — | N/A |
| 50 | shared | no_comms | 0.001 | two_module_rnn_50_no_comms_nb2_init0.001 | `two_module_rnn_50_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 50 | shared | no_comms | 0.01 | two_module_rnn_50_no_comms_nb2_init0.01 | `two_module_rnn_50_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 50 | shared | no_comms | 0.1 | two_module_rnn_50_no_comms_nb2_init0.1 | `two_module_rnn_50_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 50 | shared | no_comms | 1.0 | two_module_rnn_50_no_comms_nb2 | `two_module_rnn_50_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 50 | shared | no_comms | 2.0 | two_module_rnn_50_no_comms_nb2_init2 | `two_module_rnn_50_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 50 | single_module | 1.0 | 0.001 | single_module_rnn_100_nb2_init0.001 | `single_module_rnn_100_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 50 | single_module | 1.0 | 0.01 | single_module_rnn_100_nb2_init0.01 | `single_module_rnn_100_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | No | 0 | — | N/A |
| 50 | single_module | 1.0 | 0.1 | single_module_rnn_100_nb2_init0.1 | `single_module_rnn_100_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | No | 0 | — | N/A |
| 50 | single_module | 1.0 | 1.0 | single_module_rnn_100_nb2 | `single_module_rnn_100_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 50 | single_module | 1.0 | 2.0 | single_module_rnn_100_nb2_init2 | `single_module_rnn_100_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | no_comms | 0.001 | two_module_rnn_50_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | no_comms | 0.01 | two_module_rnn_50_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | no_comms | 0.1 | two_module_rnn_50_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | no_comms | 1.0 | two_module_rnn_50_task_routed_no_comms_nb2 | `two_module_rnn_50_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes |
| 50 | task_routed | no_comms | 2.0 | two_module_rnn_50_task_routed_no_comms_nb2_init2 | `two_module_rnn_50_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes |
