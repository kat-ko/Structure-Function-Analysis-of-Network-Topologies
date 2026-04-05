# Upstream projects: transfer-interference & dynspec

This file is the **canonical map** of what lives in the sibling repositories under this monorepo. Use it when you change **learning algorithms** or **task setup** in **gecco** and need to decide what to **import**, **copy**, or **replace**.

## GECCO LBA track (in this folder)

- **Narrative outline** for a 2-page late-breaking abstract: `LBA_NARRATIVE.md`.
- **Runnable one-figure experiment** (NSGA-II + Holton `trial_df` subset + cyclic A/B schedule): `python -m gecco.experiments.run_lba_figure` (see `README.md`).
- **Data helpers** are **vendored** from the Holton pipeline in `gecco/data/holton_trials.py` (no dependency on `a1b2_modular`).

Paths below are relative to the repository root: `Structure-Function-Analysis-of-Network-Topologies/`.

---

## 1. transfer-interference/

### What it is

Code and data for **Holton et al.** (*Nature Human Behaviour*, 2026): [**Humans and neural networks show similar patterns of transfer and interference during continual learning**](https://doi.org/10.1038/s41562-025-02318-y). Humans and **twinned two-layer linear ANNs** perform an **A1 → B → A2** continual-learning protocol on a **ring** (summer/winter “seasons”, six stimuli per task, rule-based angular structure).

### What gecco is most likely to reuse

| Piece | Location | Role |
|--------|-----------|------|
| Participant preprocessing | `transfer-interference/scripts/01_preprocess_data.py` | Raw CSVs → `trial_df.csv`, exclusions, regressors |
| Preprocessing logic | `transfer-interference/src/analysis/preprocessing.py` | Load batches, computed columns, exclusions |
| ANN-facing trial table | `transfer-interference/src/analysis/ann.py` | `load_participant_data`, `setup_task_parameters`, `load_ann_data` (`.npz` by condition), schedule helpers |
| Linear FFN model | `transfer-interference/src/models/neural_network.py` | `simpleLinearNet`, dataset, training utilities, **MSE** on Cartesian ring targets, participant-matched schedules |
| Von Mises / interference | `transfer-interference/src/models/vonmises.py`, `scripts/03_fit_vonmises.py` | Mixture fits for “rule A vs rule B” at retest; human + simulation pipelines |
| Human metrics | `transfer-interference/src/analysis/participant.py`, `stats.py` | Transfer, lumpers/splitters, behavioural summaries |
| Experiment registry | `transfer-interference/src/models/ann_experiments.json` | Named conditions (e.g. `rich_50`, `lazy_50`) for **FFN** γ / hidden size |

### Learning algorithm (as shipped)

- **Optimizer:** AdamW (via `init_model`-style pattern in neural_network flow; see script wiring).
- **Loss:** **MSE** on **cos/sin** outputs for the **probed** season only; trial-wise updates mirroring human feedback (including **no gradient** on withheld winters / retest winters).
- **Architecture:** Two-layer **linear** network, one-hot inputs (12-d for two tasks × six stimuli), hidden + 4-D output head.
- **Continual aspect:** **Same weights** across A1, B, and A2; no explicit task labels in the UI—task change is **inferred** from new stimuli.

### Task setup (as shipped)

- **Phases:** Train A1 → train B → retest A (A2) with **asymmetric feedback** on retest (summer feedback, winter not).
- **Conditions:** **Same / Near / Far** rule between task A and task B (angle relationship).
- **Twinned ANNs:** One network per participant, **identical trial order** and targets.

### Likely **replace or fork** in gecco

- Anything that **fixes** the task to plants/ring/six stimuli (you may keep the **data schema** but change generators or features).
- The **linear FFN** itself if you move to RNNs or different heads—while keeping `ann.py`-style **batch structure** if you still fit von Mises the same way.
- Training **hyperparameters** and **number of replay epochs** vs human trial counts.

### Likely **keep as library imports**

- `preprocessing` → `trial_df` pipeline if human data format unchanged.
- `vonmises` + fit script **interface** if interference metric stays mixture-based.
- `ann.load_participant_data` if `trial_df` columns remain compatible.

---

## 2. dynamics_of_specialization/ (package **dynspec**)

### What it is

Reference implementation for **Béna & Goodman** — journal version: [**Dynamics of specialization in neural modules under resource constraints**](https://doi.org/10.1038/s41467-024-55188-9) (see also [arXiv:2106.02626](https://arxiv.org/abs/2106.02626)). **Modular RNNs** (“**Community**” model): **core** recurrence + **comms** pathways with **masked** `nn.RNN`/`nn.GRU` weights, optional **binary** comms with surrogate gradients.

### What gecco is most likely to reuse

| Piece | Location | Role |
|--------|-----------|------|
| **Community** architecture | `dynamics_of_specialization/dynspec/models.py` | Core + comms RNNs, masks, `Readout`, `init_model`, `Masked_weight` parametrization |
| Implementation details | `dynamics_of_specialization/DYNSPEC_RNN_IMPLEMENTATION.md` | Shapes `(T,B,·)`, mask layout, forward `core + comms`, readout |
| Training loop (dynspec-native) | `dynamics_of_specialization/dynspec/training.py` | `train_community`, `process_data`, `get_decision`, `get_loss` (**cross-entropy** typical) |
| Decisions | `dynamics_of_specialization/dynspec/decision.py` | Temporal (last/mean/sum) + module (max/sum/random) aggregation |
| Sequences from static inputs | `dynamics_of_specialization/dynspec/data_process.py` | `temporal_data`, `process_data` → `(T, B, D)` |
| Sweeps / hashing | `dynamics_of_specialization/dynspec/experiment.py` | `Experiment`, `copy_and_change_config`, grids |
| Synthetic tasks (paper) | `dynamics_of_specialization/dynspec/tasks.py`, `datasets.py` | Parity/MNIST/EMNIST-style **global** task—**not** the Holton ring task |
| Surrogate spikes | `dynamics_of_specialization/dynspec/surrogate.py` | `super_spike` for binary comms |

### Learning algorithm (as shipped)

- **Optimizer:** AdamW + optional `ExponentialLR` (`init_model` in `models.py`).
- **Loss:** Often **cross-entropy** after `get_decision`, on classification-style targets from `tasks.py`.
- **Forward:** Combined hidden state = **core output + comms output**; readout on trajectory.

### Task setup (as shipped)

- **Environment:** Controlled **digit/letter** stimuli and **parity**-style **global** tasks—not A1-B-A2 Holton.
- **Data:** `datasets.py` (MNIST/EMNIST, custom splits); `process_data` expands to RNN time steps.

### Likely **reuse verbatim or via `pip install -e`**

- `Community` **config dict**, **mask** construction, **`init_model`** if you keep their training API.
- `data_process.process_data` if your gecco inputs are still “static pattern repeated/noised over T steps”.

### Likely **adapt in gecco (adapter layer)**

- **Inputs:** Holton uses **12-D one-hot + ring targets**; dynspec expects **`input_size * n_modules`** and its own readout layout. You need a **thin wrapper** (similar in spirit to `a1b2_modular`’s `TwoModuleRNNWrapper`) that:
  - builds `(T, B, D)` from your task batches,
  - maps **readout** to **4-D Cartesian** (or your new target format),
  - applies **MSE + probe masking** (Holton) instead of CE (dynspec default), unless you intentionally switch.
- **`get_decision` / `get_task_target`:** replace or bypass if your loss is **not** multi-class CE on dynspec task objects.

### Relationship to **a1b2_modular** (in this monorepo)

`a1b2_modular` **vendors** a Community-compatible stack under `a1b2_modular/a1b2/models/` (e.g. `community.py`, `surrogate.py`) and wraps it for **A1-B-A2**. It does **not** import the `dynspec` package by name in `pyproject.toml`; it is a **forked copy** kept in sync manually. For gecco you can either:

- **Import `dynspec`** from `dynamics_of_specialization/` (editable install), or  
- **Vendor** the same way as `a1b2_modular`, or  
- **Import `a1b2`** as a dependency (if you publish/install it)—usually heavier than needed.

Document your choice in this file’s **Decision log** (below).

---

## 3. Quick comparison (reuse decisions)

| Concern | transfer-interference | dynspec |
|---------|------------------------|---------|
| **Task** | Holton A1-B-A2, ring, human twinned | Modular RNN on synthetic vision / parity |
| **Default loss** | MSE on selected outputs | CE after decision |
| **Modularity** | Single hidden FFN | Core + comms + masks |
| **Time** | Implicit (one step per trial in FFN) | Explicit RNN `(T,B,·)` |
| **Best “import” for gecco** | Data + metrics + FFN baseline | `Community` + masks + optional `process_data` |

---

## 4. Suggested gecco layout (when you add code)

- **`gecco/models/wrappers.py`** — Community (or dynspec) ↔ your task tensors and loss.
- **`gecco/training/schedule.py`** — A1-B-A2 + update masking (copy **ideas** from transfer-interference / a1b2_modular).
- **`gecco/config/experiments.json`** — Named runs; **do not** overload `ann_experiments.json` unless you want one global file.

---

## 5. Decision log (edit as you go)

Use this table to record **what gecco actually uses** so it stays explicit over time.

| Date | Choice | Notes |
|------|--------|-------|
| _YYYY-MM-DD_ | e.g. dynspec via editable install vs vendored `community.py` | |
| _YYYY-MM-DD_ | e.g. transfer-interference: import `src` vs copied `preprocessing` | |
| _YYYY-MM-DD_ | Task schema: compatible with `trial_df` or new | |

---

## 6. References

- Holton et al. (2026). *Nat Hum Behav*. https://doi.org/10.1038/s41562-025-02318-y  
- Béna & Goodman (2025). *Nat Commun*. https://doi.org/10.1038/s41467-024-55188-9  
- dynspec RNN write-up: `dynamics_of_specialization/DYNSPEC_RNN_IMPLEMENTATION.md`  
- Original transfer-interference README: `transfer-interference/README.md` (OSF / GitHub pointers)
