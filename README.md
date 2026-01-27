## Structural Biases Shape Transfer-Interference Tradeoffs under Task Similarity in Continual Learning

This repository contains the anonymized code accompanying the GECCO submission **“Structural Biases Shape Transfer-Interference Tradeoffs under Task Similarity in Continual Learning”**.

### Abstract

Continual learning requires systems to balance reuse of previously acquired representational subspaces against protection of these subspaces from disruptive updates, a stability-plasticity tradeoff. Prior work in artificial and biological systems shows that both learning dynamics and architectural constraints shape representational geometry, yet their interaction remains poorly understood. In particular, it is unclear whether architectural biases, under identical learning dynamics, can systematically bias representational overlap during sequential task learning in ways that facilitate transfer while mitigating interference. Using a seasons-style continual learning task with parametrically defined task similarity, we analyze how representational subspaces align in feedforward architectures with task-specific modules, controlled inter-module communication, and a shared readout. By probing forward passes and training dynamics under identical optimization regimes, we characterize how communication bandwidth, input embedding regime (lazy to rich), and latent task structure bias representational alignment. Our results suggest that architectural communication constraints bias the space of admissible representational geometries, shaping reuse and separation even under identical optimization dynamics. The findings provide insight into how structural biases can complement learning dynamics in managing the transfer-interference tradeoff in continual learning, and motivate future work on evolutionary and dynamic specialization as adaptive responses to task similarity.

---

## Architectures: Two-Module vs No-Module Baseline

- **Two-module architecture (`TwoModuleMLP`)**  
  - Two parallel input modules (A and B) with task-based routing for the three-phase A1–B–A2 continual-learning schedule.  
  - Optional **inter-module communication** with bandwidth presets (`none`, `low`, `high`) controlling bidirectional linear connections between module-specific hidden states.  
  - A **shared linear readout** combines both modules’ contributions, allowing us to study how structural communication constraints bias representational overlap and transfer/interference.

- **No-modules baseline (`simpleLinearNet`)**  
  - A single fully shared hidden layer with the same total hidden dimensionality as the two-module network.  
  - No task-based routing and no explicit communication pathway: all tasks update the same representational subspace under identical optimization dynamics.  
  - Serves as the control architecture for assessing how explicit modular structure and communication shape geometry and behavioral transfer/interference.

Both architectures are trained on identical participant schedules and optimization settings; architectural configuration (two-module vs baseline, communication bandwidth, initialization regime) is specified in `transfer-interference/src/models/ann_experiments.json`.

---

## Installation (Anonymized Code Snapshot)

From the repository root:

```bash
conda create -n transfer-interference python=3.10
conda activate transfer-interference
pip install -r requirements.txt
```

This installs the scientific Python stack (PyTorch, NumPy, pandas, etc.) required by the `transfer-interference` experiments. For system-specific PyTorch builds, you may optionally follow the instructions at `https://pytorch.org/get-started/locally`.

---

## Running the Main Experiments

All commands below are run from the `transfer-interference` subdirectory.

1. **Preprocess human data and apply exclusion criteria**

   ```bash
   cd transfer-interference
   python scripts/01_preprocess_data.py
   ```

   This creates the processed trial-level dataframe used for fitting and simulation in `data/participants/trial_df.csv`.

2. **Run simulations for a given architecture/configuration**

   Configurations (including two-module vs no-modules baseline, communication bandwidth, and initialization regime) are defined in `src/models/ann_experiments.json` via the `conditions` list.

   ```bash
   # Example: run simulations for condition "rich_50"
   python scripts/02_run_simulations.py rich_50
   ```

   Simulation outputs (behavior and representational states across A1–B–A2) are written under `data/simulations/<condition_name>/`.

3. **Generate representational-geometry summaries**

   To reproduce the representational geometry analyses across all configured conditions:

   ```bash
   python scripts/generate_all_geometry_results.py
   ```

   or for a restricted set of configurations:

   ```bash
   python scripts/generate_all_geometry_results.py --configs rich_50 lazy_50
   ```

   This produces `geom_results_<condition>.npz` files in `data/simulations/`, which are consumed by the accompanying analysis notebooks in `transfer-interference/notebooks*`.

---

## Anonymization Notice

This code snapshot and README are intended for **double-blind review**. Identifying information (e.g., author names and affiliations) has been deliberately omitted; please keep any additional documentation or comments likewise anonymized.