# Structure–Function Analysis of Network Topologies

This repository contains code and analyses for **modular neural networks**, **transfer and interference** in continual learning, and the **A1–B–A2** paradigm. It brings together three related subprojects.

## Repository structure

| Subproject | Description |
|------------|-------------|
| **[a1b2_modular/](a1b2_modular/)** | Unified A1–B–A2 transfer–interference task with **feedforward** and **two-module RNN** (dynspec-style). Same schedules as transfer-interference; supports shared/task-routed input, sparse communication, and modular input separation (`common_input=false`). See [a1b2_modular/README.md](a1b2_modular/README.md). |
| **[transfer-interference/](transfer-interference/)** | Human and ANN transfer/interference (Holton et al.). Scripts and notebooks for figures 2–4, FFN simulations, and two-module RNN comparisons. See [transfer-interference/README.md](transfer-interference/README.md). |
| **[dynamics_of_specialization/](dynamics_of_specialization/)** | **dynspec**: modular RNNs under resource constraints (Béna et al., [arXiv:2106.02626](https://arxiv.org/abs/2106.02626)). Implements the Community model (core + comms, masked weights). Example: `modular_networks.ipynb`. |

There is no single install at the repo root. Install and run from each subproject as needed.

## Quick start

- **A1–B–A2 with FFN or two-module RNN (recommended entry point)**  
  From `a1b2_modular/`: `pip install -e .` then `python scripts/02_run_simulations.py <condition>`. Conditions in `a1b2/models/experiments.json` (e.g. `rich_50`, `two_module_rnn_50`, `two_module_rnn_50_task_routed_low_sparse`).

- **Transfer/interference (Holton-style)**  
  From `transfer-interference/`: see [transfer-interference/README.md](transfer-interference/README.md) for setup and scripts.

- **Dynspec (modular RNN only)**  
  From `dynamics_of_specialization/`: `pip install -e .` (creates the `dynspec` package). Install PyTorch for your system from [pytorch.org](https://pytorch.org/get-started/locally/). Main example: `modular_networks.ipynb`.

## Requirements

- Python ≥3.8 (a1b2_modular), 3.10 suggested for dynspec
- Common deps: numpy, pandas, scipy, matplotlib, seaborn, torch, tqdm; a1b2 adds scikit-learn

Each subproject lists its own dependencies in `pyproject.toml` or `setup.py`.
