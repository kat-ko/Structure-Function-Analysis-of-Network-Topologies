# Standard primary grid run inventory

*Last refreshed: 2026-04-27 13:13 UTC*

This file tracks **only** the **standard primary grid** as defined by `a1b2.utils.sim_storage.is_primary_grid_condition`: `nb_steps=2`, `common_input=False`, `common_readout=True`, `cell_type=RNN`, `n_layers=1`, `dropout=0`, init ∈ {0.001, 0.01, 0.1, 1, 2}; `two_module_rnn` with **no_comms** (`sparsity=0`); capacity-matched `single_module_rnn`.

- **Folder:** `data/simulations/<run_id>/` only (ablations live under `data/simulations/primary_grid_ablations/`).
- **State checkpoints:** `state_<participant_id>.pt` for each `sim_<participant_id>.npz` inside the run folder.
- **Von Mises fits:** `data/simulations/<run_id>_vonmises_fits.csv` (sibling of the run folder; from `scripts/03_fit_vonmises.py simulations --sim-name <run_id>`).
- **Regenerate:** `python3 scripts/regenerate_standard_primary_grid_inventory.py`
- **Total rows:** 60

| dim_h | routing | sparsity | init | condition | run_id (= folder) | exists | npz | state matched | state OK | VM OK |
| ---: | --- | --- | --- | --- | --- | :---: | ---: | --- | :---: | :---: |
| 6 | shared | no_comms | 0.001 | two_module_rnn_6_no_comms_nb2_init0.001 | `two_module_rnn_6_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | shared | no_comms | 0.01 | two_module_rnn_6_no_comms_nb2_init0.01 | `two_module_rnn_6_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | shared | no_comms | 0.1 | two_module_rnn_6_no_comms_nb2_init0.1 | `two_module_rnn_6_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | shared | no_comms | 1.0 | two_module_rnn_6_no_comms_nb2 | `two_module_rnn_6_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | shared | no_comms | 2.0 | two_module_rnn_6_no_comms_nb2_init2 | `two_module_rnn_6_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | single_module | 1.0 | 0.001 | single_module_rnn_12_nb2_init0.001 | `single_module_rnn_12_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | single_module | 1.0 | 0.01 | single_module_rnn_12_nb2_init0.01 | `single_module_rnn_12_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | single_module | 1.0 | 0.1 | single_module_rnn_12_nb2_init0.1 | `single_module_rnn_12_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | single_module | 1.0 | 1.0 | single_module_rnn_12_nb2 | `single_module_rnn_12_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | single_module | 1.0 | 2.0 | single_module_rnn_12_nb2_init2 | `single_module_rnn_12_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | task_routed | no_comms | 0.001 | two_module_rnn_6_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | task_routed | no_comms | 0.01 | two_module_rnn_6_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | task_routed | no_comms | 0.1 | two_module_rnn_6_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_6_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | task_routed | no_comms | 1.0 | two_module_rnn_6_task_routed_no_comms_nb2 | `two_module_rnn_6_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 6 | task_routed | no_comms | 2.0 | two_module_rnn_6_task_routed_no_comms_nb2_init2 | `two_module_rnn_6_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | shared | no_comms | 0.001 | two_module_rnn_12_no_comms_nb2_init0.001 | `two_module_rnn_12_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | shared | no_comms | 0.01 | two_module_rnn_12_no_comms_nb2_init0.01 | `two_module_rnn_12_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | shared | no_comms | 0.1 | two_module_rnn_12_no_comms_nb2_init0.1 | `two_module_rnn_12_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | shared | no_comms | 1.0 | two_module_rnn_12_no_comms_nb2 | `two_module_rnn_12_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | shared | no_comms | 2.0 | two_module_rnn_12_no_comms_nb2_init2 | `two_module_rnn_12_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | single_module | 1.0 | 0.001 | single_module_rnn_25_nb2_init0.001 | `single_module_rnn_25_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | single_module | 1.0 | 0.01 | single_module_rnn_25_nb2_init0.01 | `single_module_rnn_25_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | single_module | 1.0 | 0.1 | single_module_rnn_25_nb2_init0.1 | `single_module_rnn_25_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | single_module | 1.0 | 1.0 | single_module_rnn_25_nb2 | `single_module_rnn_25_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 0/305 | Partial | Partial |
| 12 | single_module | 1.0 | 2.0 | single_module_rnn_25_nb2_init2 | `single_module_rnn_25_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | task_routed | no_comms | 0.001 | two_module_rnn_12_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | task_routed | no_comms | 0.01 | two_module_rnn_12_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | task_routed | no_comms | 0.1 | two_module_rnn_12_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_12_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | task_routed | no_comms | 1.0 | two_module_rnn_12_task_routed_no_comms_nb2 | `two_module_rnn_12_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 12 | task_routed | no_comms | 2.0 | two_module_rnn_12_task_routed_no_comms_nb2_init2 | `two_module_rnn_12_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
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
| 50 | shared | no_comms | 0.001 | two_module_rnn_50_no_comms_nb2_init0.001 | `two_module_rnn_50_no_comms_nb2_init0.001_nb2_shared_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | shared | no_comms | 0.01 | two_module_rnn_50_no_comms_nb2_init0.01 | `two_module_rnn_50_no_comms_nb2_init0.01_nb2_shared_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | shared | no_comms | 0.1 | two_module_rnn_50_no_comms_nb2_init0.1 | `two_module_rnn_50_no_comms_nb2_init0.1_nb2_shared_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | shared | no_comms | 1.0 | two_module_rnn_50_no_comms_nb2 | `two_module_rnn_50_no_comms_nb2_nb2_shared_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | shared | no_comms | 2.0 | two_module_rnn_50_no_comms_nb2_init2 | `two_module_rnn_50_no_comms_nb2_init2_nb2_shared_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | single_module | 1.0 | 0.001 | single_module_rnn_100_nb2_init0.001 | `single_module_rnn_100_nb2_init0.001_nb2_shared_sp1_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | single_module | 1.0 | 0.01 | single_module_rnn_100_nb2_init0.01 | `single_module_rnn_100_nb2_init0.01_nb2_shared_sp1_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | single_module | 1.0 | 0.1 | single_module_rnn_100_nb2_init0.1 | `single_module_rnn_100_nb2_init0.1_nb2_shared_sp1_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | single_module | 1.0 | 1.0 | single_module_rnn_100_nb2 | `single_module_rnn_100_nb2_nb2_shared_sp1_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | single_module | 1.0 | 2.0 | single_module_rnn_100_nb2_init2 | `single_module_rnn_100_nb2_init2_nb2_shared_sp1_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | task_routed | no_comms | 0.001 | two_module_rnn_50_task_routed_no_comms_nb2_init0.001 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.001_nb2_task_routed_sp0_sep_cr_RNN_init0.001` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | task_routed | no_comms | 0.01 | two_module_rnn_50_task_routed_no_comms_nb2_init0.01 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.01_nb2_task_routed_sp0_sep_cr_RNN_init0.01` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | task_routed | no_comms | 0.1 | two_module_rnn_50_task_routed_no_comms_nb2_init0.1 | `two_module_rnn_50_task_routed_no_comms_nb2_init0.1_nb2_task_routed_sp0_sep_cr_RNN_init0.1` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | task_routed | no_comms | 1.0 | two_module_rnn_50_task_routed_no_comms_nb2 | `two_module_rnn_50_task_routed_no_comms_nb2_nb2_task_routed_sp0_sep_cr_RNN` | Yes | 305 | 305/305 | Yes | Partial |
| 50 | task_routed | no_comms | 2.0 | two_module_rnn_50_task_routed_no_comms_nb2_init2 | `two_module_rnn_50_task_routed_no_comms_nb2_init2_nb2_task_routed_sp0_sep_cr_RNN_init2` | Yes | 305 | 305/305 | Yes | Partial |
