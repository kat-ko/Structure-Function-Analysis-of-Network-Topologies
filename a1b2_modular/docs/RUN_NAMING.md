# Simulation run folders and NPZ schema

## Run folder layout

- **Path**: `data/simulations/<run_id>/`
- **Per participant**: `sim_<participant_id>.npz`, optional `state_<participant_id>.pt` (RNN)
- **Config**: `settings.json` (condition, training params, schema metadata)

## `run_id` (`a1b2.utils.run_config.build_run_id`)

- Derived deterministically from the **condition** dict in `experiments.json`.
- **Legacy** conditions keep the same folder name as before (e.g. `two_module_rnn_50` when only classic keys are set).
- **Schema v2** (during-training core/comms capture): if `sim_schema_version >= 2` or `log_during_training: true`, a suffix is appended, e.g. `__s2`, so new runs **never overwrite** older bundles for the same base condition.
- Optional **`run_id_suffix`**: extra segment after the schema token for A/B comparisons (e.g. `__s2_ablation`).

## `settings.json` fields (extended)

| Field | Meaning |
|--------|---------|
| `sim_schema_version` | `1` = legacy NPZ keys only; `2` = may include `during_*` arrays |
| `log_during_training` | Whether the run logged per-trial core/comms during training |
| `during_log_post_step` | Default `true`: log **after** `optimizer.step()` via a second forward (post-update). Set `false` in the condition for **pre-update** core/comms from the training forward only (faster; one fewer forward per step when logging). |
| `dataloader_num_workers` | DataLoader workers (default `0`; set at top level of `experiments.json` if desired) |

## NPZ schema v2 (additive)

Legacy arrays are unchanged. When `log_during_training` is enabled for a two-module RNN:

| Key | Shape | Description |
|-----|--------|-------------|
| `during_core_per_module` | `(n_phase, n_trials, n_modules, hidden_size)` | Core branch state (see timing below) |
| `during_comms_per_module` | same | Comms branch state |
| `during_core_l2` | `(n_phase, n_trials, n_modules)` | L2 norm per module |
| `during_comms_l2` | same | L2 norm per module |
| `during_comms_m0_over_m1_l2` | `(n_phase, n_trials)` | `during_comms_l2[...,0] / (during_comms_l2[...,1] + ε)` (two-module only); cheap proxy for cross-module comms balance |

**Timing (methods)**:

- **`during_log_post_step: true` (default)**: a second forward in `torch.no_grad()` runs **after** the optimizer step when an update occurred → logged core/comms reflect **post-update** weights.
- **`during_log_post_step: false`**: the training forward uses `return_core_comms=True`; tensors are copied with `.detach()` **before** `backward` → **pre-update** activations (aligned with the loss forward). Saves roughly one community forward per logged step.

If no optimizer step ran, the post-step forward still reflects current weights. Trial order matches `hiddens_per_module`.

**Functional specialization over time**: Use `hiddens_per_module` (and labels/probes) in notebooks or `03_functional_specialization.py` on sliding windows; `during_*` adds core/comms decomposition for comms-focused hypotheses.

## Analysis helpers (`a1b2.analysis.during_training`)

Reusable slices for common hypotheses (task_routed: module 0 = A, module 1 = B):

- `phase_B_probe1_comms_ratio(...)` — B phase, `feature_probe==1` trials: distribution of `comms_m0_over_m1_l2` (cross-module comms proxy during B training).
- `phase_A2_by_probe(...)` — A2 split by probe 0 vs 1 for B→A style comparisons.
- `summarize_during_npz(d)` — quick QC dict from a loaded `npz` mapping.

## Notebooks

Any notebook that loads `sim_*.npz` can branch on `"during_core_per_module" in data`. Particularly relevant:

- `notebooks/nb_task_routed_comms_minimal_pair.ipynb`
- `notebooks/nb_task_routed_within_phase_by_similarity.ipynb`
- `notebooks/nb_task_routed_within_phase_signal_test.ipynb`

## Running schema v2 simulations

Use a condition that sets `log_during_training: true`, e.g. **`two_module_rnn_50_task_routed_trainlog_s2`**:

```bash
python scripts/02_run_simulations.py two_module_rnn_50_task_routed_trainlog_s2
```

Outputs go to a **new** `run_id` folder (includes `__s2`), separate from runs without that flag.
