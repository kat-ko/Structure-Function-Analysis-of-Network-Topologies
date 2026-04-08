# a1b2_modular

Unified codebase for the **A1–B–A2** transfer-interference task with both the original **FFN** and the **two-module RNN** (dynspec-style), plus analyses from both pipelines.

## Setup

From the project root (`a1b2_modular/`):

```bash
pip install -e .
```

Requirements: Python ≥3.8, numpy, pandas, scipy, matplotlib, seaborn, torch, tqdm, scikit-learn.

## Pipeline (scripts)

1. **Preprocess participant data**  
   `python scripts/01_preprocess_data.py`  
   Reads from `data/participants/raw/`, writes `data/participants/trial_df.csv`.

2. **Run simulations**  
   `python scripts/02_run_simulations.py <condition>`  
   Conditions are defined in `a1b2/models/experiments.json`. Each condition has `arch`: `"ffn"`, `"single_module_rnn"`, or `"two_module_rnn"`.  
   Examples: `rich_50`, `two_module_rnn_50`, `single_module_rnn_50`.  
   Results are saved under a policy-based root:
   - **Primary grid:** `data/simulations/<run_id>/`
   - **Ablations:** `data/simulations/primary_grid_ablations/<run_id>/`
   Routing is automatic with `--storage-mode auto` (default), and can be overridden with `--storage-mode primary|ablation`.
   Use `--print-output-path` to preview destination without training.

   **Sparsity 0.1 and 0.9 (25-dim nb2):** To run the additional sparsity levels 0.1 and 0.9 (shared and task_routed input):
   ```bash
   python scripts/02_run_simulations.py two_module_rnn_25_sp01_nb2
   python scripts/02_run_simulations.py two_module_rnn_25_task_routed_sp01_nb2
   python scripts/02_run_simulations.py two_module_rnn_25_sp09_nb2
   python scripts/02_run_simulations.py two_module_rnn_25_task_routed_sp09_nb2
   ```
   From repo root with explicit base folder: `python a1b2_modular/scripts/02_run_simulations.py two_module_rnn_25_sp01_nb2 --base-folder a1b2_modular`

   Optional: geometry run (single participant, same/near/far B)  
   `python scripts/02_run_simulations.py <condition> --geometry [--participant ID]`

3. **Fit von Mises**  
   `python scripts/03_fit_vonmises.py participants`  
   `python scripts/03_fit_vonmises.py simulations --sim-name <run_id>`  
   Ablation root example: `python scripts/03_fit_vonmises.py simulations --sim-name <run_id> --simulations-subdir simulations/primary_grid_ablations`

## Documentation (ablations and handoff)

- **Canonical checklist** (depth, dropout, GRU, LSTM, `build_run_id`): `docs/PLAN_rnn_depth_and_celltype_ablations.md`
- **Continuation + operational TODOs** (validation, readout ablation, optional phases — checklist separate from plan “Phase” numbering): `docs/ABLATION_CONTINUATION_AND_TODOS.md`
- **Storage policy + migration SOP** (primary grid vs ablations): `docs/PRIMARY_GRID_STORAGE_POLICY.md`

**Config validation (ablation handoff):** From `a1b2_modular/`, run `python scripts/validate_ablation_continuation.py` (JSON uniqueness, `build_run_id` for depth/dropout/GRU/readout pilots). Add `--forward` if PyTorch is installed for forward smokes.

## Analyses

- **Transfer / interference / von Mises**: same methodology as transfer-interference (figures 2–4, `a1b2.analysis.transfer_interference`, `stats`, `participant`).
- **Dynspec-style**: retraining, correlations, ablations, Experiment sweeps (`a1b2.training.community`, `a1b2.analysis.retraining`, `correlations`, `experiment`).

Notebooks in `notebooks/`: `figure2_transfer_interference.ipynb`, `figure3_anns.ipynb`, `figure4_individual_differences.ipynb`, `modular_analyses.ipynb`.

**Tests / MVP analysis:** `tests/nb_no_comms_geometry_init_size_similarity_mvp.ipynb` — exports `data/derived/no_comms_mvp_*_long.csv`, heatmaps (init × `grid_h` × similarity) for **task_routed**, **shared**, and **single_module** (capacity-matched baselines), optional MixedLM via `pip install -e ".[notebook-stats]"`.

## Experimental factors (RNN studies)

The primary RNN comparison is **no-module (single-module) vs two-module** architecture. Varying factors that affect **run_id** (and thus the results folder) include:

- **Task similarity**: same / near / far B (geometry run).
- **Architecture**: `single_module_rnn` (n_modules=1) vs `two_module_rnn` (n_modules=2).
- **Input routing**: `shared` (both modules get same input) vs `task_routed` (A → module 1, B → module 2 by feature_probe).
- **nb_steps**: sequence length (1 = single step; >1 = temporal input and optional trajectory logging). Each RNN condition has an optional **\_nb2** variant (e.g. `two_module_rnn_50_nb2`) with `nb_steps=2`; run folder names then include `nb2` so they stay distinct from nb_steps=1 runs.
- **Communication**: sparsity, common_readout, cell_type (RNN/GRU). **Per-module readout** conditions use `common_readout: false`; matched **no_comms** pilots are named with suffix **`_pr`** on the shared-readout anchor (e.g. `..._init0.001` → `..._init0.001_pr`); `build_run_id` uses **`sep_pr`** instead of **`sep_cr`**. **Input separation**: `common_input` (default **false**). With `common_input=false`, each module's first layer only receives its own input chunk (true modular input separation). **common_input=true** is for **ablation only** (to validate main results); use conditions such as `two_module_rnn_50_ablation_common_input` or `two_module_rnn_50_task_routed_ablation_common_input`.
- **Init scale**: optional `init_scale` for RNN weights (experimental lever; regime labels are not assigned by init—see Rich/lazy below).

**Run names:** New RNN runs use a **full run_id** (e.g. `two_module_rnn_50_nb1_shared_sp1_sep_cr_RNN`) so they do not overwrite legacy result folders. The run_id includes **sep** (separate input per module, `common_input=false`) or **ci** (common input, `common_input=true`, ablation).

**Analysis factors:** **Input scenario** = `input_routing`: `"shared"` means both modules receive the same input; `"task_routed"` means task A→module 1, task B→module 2 (different inputs per module). **Communication** = `sparsity` (inter-module connection strength, 0–1) and optionally `common_readout` (shared vs per-module readout). Example conditions for communication × input-scenario runs: `two_module_rnn_50`, `two_module_rnn_50_low_sparse`, `two_module_rnn_50_task_routed`, `two_module_rnn_50_task_routed_low_sparse`, `two_module_rnn_50_sp05`, `two_module_rnn_50_task_routed_sp05`, `two_module_rnn_50_sep_readout`, and no_comms readout pilots `two_module_rnn_25_no_comms_nb2_init0.001_pr`, `two_module_rnn_25_task_routed_no_comms_nb2_init0.001_pr`.

What is held fixed (e.g. n_epochs, learning_rate, batch_size) is in `experiments.json` and the script.

## Module use (operational)

For RNN runs, **module use** is defined from logged activity: per-module hidden magnitudes (L2 norm per module over time or post-phase), and optionally comms/core ratio if exposed. When `rnn_extra` is set, the pipeline logs concatenated `hiddens` and **per-module** arrays (`hiddens_per_module`, `hiddens_post_phase_*_per_module`), and when `nb_steps > 1` also **trajectory** (`hiddens_post_phase_*_trajectory`), so analyses can use shared or per-module data.

## Rich/lazy (empirical)

We do **not** assign "rich" or "lazy" by initialization scale. Init scale (and optional recurrent rank) are **experimental knobs**. Regime is determined **post hoc** from measured behaviour: e.g. parameter-change norm (init → convergence), representation alignment (hidden activations before vs after training), and optionally tangent-kernel alignment (see e.g. arXiv:2310.08513). Existing keys `hiddens_pre_training` and `hiddens_post_phase_*` (and optional init/final state_dict) support these analyses.

## Data layout

- `data/participants/`: `raw/`, `trial_df.csv`, `human_vonmises_fits.csv` (after 01 and 03).
- `data/simulations/<run_id>/`: primary-grid simulations.
- `data/simulations/primary_grid_ablations/<run_id>/`: ablation simulations.

## Package layout (`a1b2`)

- `a1b2.data`: basic_funcs, preprocessing, temporal (optional RNN data).
- `a1b2.models`: ffn, community, two_module_rnn, rnn_init, vonmises, experiments.json.
- `a1b2.training`: schedule, simulation, community (dynspec train/test).
- `a1b2.analysis`: transfer_interference, participant, stats, retraining, correlations, experiment.
- `a1b2.decision`: get_decision, max_decision, etc. (for Community readouts).
- `a1b2.utils`: figure_settings, figure_utils, plotting, run_config (build_run_id).
