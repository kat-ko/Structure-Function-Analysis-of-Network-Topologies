# Primary grid run inventory

*Last refreshed: 2026-04-09 00:19 UTC*

All `experiments.json` conditions matching the **standard primary grid** (`is_primary_grid_condition` in `a1b2.utils.sim_storage`): `nb_steps=2`, `common_input=False`, `common_readout=True`, `cell_type=RNN`, `n_layers=1`, `dropout=0`, init ∈ {0.001, 0.01, 0.1, 1, 2}; for `two_module_rnn`, **no_comms only** (`sparsity=0`).

For a dedicated tracker with the same rows, see [`STANDARD_PRIMARY_GRID_RUN_INVENTORY.md`](STANDARD_PRIMARY_GRID_RUN_INVENTORY.md).

- **Folder:** `data/simulations/<run_id>/` (folder name equals `run_id`).
- **State checkpoints:** `state_<participant_id>.pt` for each `sim_<participant_id>.npz` inside the run folder.
- **Von Mises fits:** `data/simulations/<run_id>_vonmises_fits.csv` (sibling of the run folder; from `scripts/03_fit_vonmises.py simulations --sim-name <run_id>`).
- **Single-module baseline:** `single_module` rows use a capacity-matched hidden size for that `dim_h` column (e.g. **100** hidden units when `dim_h=50`, comparable to two modules × 50). Condition names use `single_module_rnn_<hidden>_nb2…`.
- **Total rows:** 60

| dim_h | routing | sparsity | init | condition | run_id (= folder) | exists | npz | state matched | state OK | VM OK |
| ---: | --- | --- | --- | --- | --- | :---: | ---: | --- | :---: | :---: |
| 6 | shared | no_comms | 0.001 | two_module_rnn_6_no_comms_nb2_init0.001 | `two_module_rnn_6_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | shared | no_comms | 0.01 | two_module_rnn_6_no_comms_nb2_init0.01 | `two_module_rnn_6_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | No |
| 6 | shared | no_comms | 0.1 | two_module_rnn_6_no_comms_nb2_init0.1 | `two_module_rnn_6_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | shared | no_comms | 1.0 | two_module_rnn_6_no_comms_nb2 | `two_module_rnn_6_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | shared | no_comms | 2.0 | two_module_rnn_6_no_comms_nb2_init2 | `two_module_rnn_6_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | single_module | 1.0 | 0.001 | single_module_rnn_12_nb2_init0.001 | `single_module_rnn_12_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | single_module | 1.0 | 0.01 | single_module_rnn_12_nb2_init0.01 | `single_module_rnn_12_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | single_module | 1.0 | 0.1 | single_module_rnn_12_nb2_init0.1 | `single_module_rnn_12_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | No |
| 6 | single_module | 1.0 | 1.0 | single_module_rnn_12_nb2 | `single_module_rnn_12_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | single_module | 1.0 | 2.0 | single_module_rnn_12_nb2_init2 | `single_module_rnn_12_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | task_routed | no_comms | 0.001 | two_module_rnn_6_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | No |
| 6 | task_routed | no_comms | 0.01 | two_module_rnn_6_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | No |
| 6 | task_routed | no_comms | 0.1 | two_module_rnn_6_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | No |
| 6 | task_routed | no_comms | 1.0 | two_module_rnn_6_task_routed_no_comms_nb2 | `two_module_rnn_6_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | No |
| 6 | task_routed | no_comms | 2.0 | two_module_rnn_6_task_routed_no_comms_nb2_init2 | `two_module_rnn_6_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | No |
| 12 | shared | no_comms | 0.001 | two_module_rnn_12_no_comms_nb2_init0.001 | `two_module_rnn_12_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | shared | no_comms | 0.01 | two_module_rnn_12_no_comms_nb2_init0.01 | `two_module_rnn_12_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | No |
| 12 | shared | no_comms | 0.1 | two_module_rnn_12_no_comms_nb2_init0.1 | `two_module_rnn_12_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | No |
| 12 | shared | no_comms | 1.0 | two_module_rnn_12_no_comms_nb2 | `two_module_rnn_12_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | No |
| 12 | shared | no_comms | 2.0 | two_module_rnn_12_no_comms_nb2_init2 | `two_module_rnn_12_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | No |
| 12 | single_module | 1.0 | 0.001 | single_module_rnn_25_nb2_init0.001 | `single_module_rnn_25_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | single_module | 1.0 | 0.01 | single_module_rnn_25_nb2_init0.01 | `single_module_rnn_25_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | No |
| 12 | single_module | 1.0 | 0.1 | single_module_rnn_25_nb2_init0.1 | `single_module_rnn_25_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 219 | 219/219 | Yes | No |
| 12 | single_module | 1.0 | 1.0 | single_module_rnn_25_nb2 | `single_module_rnn_25_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 0/305 | Partial | Partial |
| 12 | single_module | 1.0 | 2.0 | single_module_rnn_25_nb2_init2 | `single_module_rnn_25_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | task_routed | no_comms | 0.001 | two_module_rnn_12_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | No |
| 12 | task_routed | no_comms | 0.01 | two_module_rnn_12_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | No |
| 12 | task_routed | no_comms | 0.1 | two_module_rnn_12_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | No |
| 12 | task_routed | no_comms | 1.0 | two_module_rnn_12_task_routed_no_comms_nb2 | `two_module_rnn_12_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | No |
| 12 | task_routed | no_comms | 2.0 | two_module_rnn_12_task_routed_no_comms_nb2_init2 | `two_module_rnn_12_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | No |
| 25 | shared | no_comms | 0.001 | two_module_rnn_25_no_comms_nb2_init0.001 | `two_module_rnn_25_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | shared | no_comms | 0.01 | two_module_rnn_25_no_comms_nb2_init0.01 | `two_module_rnn_25_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | shared | no_comms | 0.1 | two_module_rnn_25_no_comms_nb2_init0.1 | `two_module_rnn_25_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | shared | no_comms | 1.0 | two_module_rnn_25_no_comms_nb2 | `two_module_rnn_25_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | shared | no_comms | 2.0 | two_module_rnn_25_no_comms_nb2_init2 | `two_module_rnn_25_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | single_module | 1.0 | 0.001 | single_module_rnn_50_nb2_init0.001 | `single_module_rnn_50_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | single_module | 1.0 | 0.01 | single_module_rnn_50_nb2_init0.01 | `single_module_rnn_50_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | single_module | 1.0 | 0.1 | single_module_rnn_50_nb2_init0.1 | `single_module_rnn_50_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | single_module | 1.0 | 1.0 | single_module_rnn_50_nb2 | `single_module_rnn_50_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | single_module | 1.0 | 2.0 | single_module_rnn_50_nb2_init2 | `single_module_rnn_50_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | task_routed | no_comms | 0.001 | two_module_rnn_25_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | task_routed | no_comms | 0.01 | two_module_rnn_25_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | task_routed | no_comms | 0.1 | two_module_rnn_25_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_25_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | task_routed | no_comms | 1.0 | two_module_rnn_25_task_routed_no_comms_nb2 | `two_module_rnn_25_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 25 | task_routed | no_comms | 2.0 | two_module_rnn_25_task_routed_no_comms_nb2_init2 | `two_module_rnn_25_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | shared | no_comms | 0.001 | two_module_rnn_50_no_comms_nb2_init0.001 | `two_module_rnn_50_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | No |
| 50 | shared | no_comms | 0.01 | two_module_rnn_50_no_comms_nb2_init0.01 | `two_module_rnn_50_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | No |
| 50 | shared | no_comms | 0.1 | two_module_rnn_50_no_comms_nb2_init0.1 | `two_module_rnn_50_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | No |
| 50 | shared | no_comms | 1.0 | two_module_rnn_50_no_comms_nb2 | `two_module_rnn_50_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | shared | no_comms | 2.0 | two_module_rnn_50_no_comms_nb2_init2 | `two_module_rnn_50_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | No |
| 50 | single_module | 1.0 | 0.001 | single_module_rnn_100_nb2_init0.001 | `single_module_rnn_100_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | No |
| 50 | single_module | 1.0 | 0.01 | single_module_rnn_100_nb2_init0.01 | `single_module_rnn_100_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 277 | 277/277 | Yes | No |
| 50 | single_module | 1.0 | 0.1 | single_module_rnn_100_nb2_init0.1 | `single_module_rnn_100_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 304 | 304/304 | Yes | No |
| 50 | single_module | 1.0 | 1.0 | single_module_rnn_100_nb2 | `single_module_rnn_100_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | No |
| 50 | single_module | 1.0 | 2.0 | single_module_rnn_100_nb2_init2 | `single_module_rnn_100_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | No |
| 50 | task_routed | no_comms | 0.001 | two_module_rnn_50_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | No |
| 50 | task_routed | no_comms | 0.01 | two_module_rnn_50_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | No |
| 50 | task_routed | no_comms | 0.1 | two_module_rnn_50_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | No |
| 50 | task_routed | no_comms | 1.0 | two_module_rnn_50_task_routed_no_comms_nb2 | `two_module_rnn_50_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | task_routed | no_comms | 2.0 | two_module_rnn_50_task_routed_no_comms_nb2_init2 | `two_module_rnn_50_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | No |
