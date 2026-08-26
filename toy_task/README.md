# toy_task

Standalone synthetic continual-learning benchmark for the structure–function /
modular-network line of work. It studies how **structural priors** (dense vs.
modular RNNs) and **learning regime** (rich/lazy via init scale γ) shape
**representational organization** as **task similarity** `s` varies, under a fixed
`T(0) → T(s) → T(0)` (A1 → B → A2) curriculum.

This subproject is **independent** of `a1b2_modular`,
`dynamics_of_specialization`, `transfer-interference`, and `gecco` — no
cross-imports. See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the full
specification and [`description/`](description/) for the source spec.

## Install

From this directory:

```bash
pip install -e .
```

Dependencies: numpy, torch, matplotlib, scikit-learn, pandas (pytest for tests).

## Quick start

```bash
# Tiny end-to-end smoke run
python -m toy_task

# Visualize the environment (Stage-1 gate, no neural network)
python scripts/visualize_task.py --seed 0 --out figures/task_overview.png

# A small pilot sweep (all 5 configs)
python scripts/run_experiment.py --out data/runs \
    --hidden-sizes 24 50 --gammas 0.1 1.0 \
    --similarity-indices 0 3 5 --seeds 0 1

# Aggregate runs into behavioral + dimensionality figures
python scripts/make_figures.py --runs data/runs --out figures
```

The full default grid is `5 configs × 4 H × 5 γ × 7 s × 10 seeds = 7000` runs
(`python scripts/run_experiment.py --out data/runs`).

## Architectures

| Name | Input wiring | Notes |
|------|--------------|-------|
| `dense` | full `x` → one population | baseline topology |
| `modular_shared` | full `x` → both modules | modularity baseline (no comms) |
| `modular_feature_routed` | `x` split by feature dim | task-agnostic sensory routing |
| `modular_task_routed` (`freeze`) | full `x` gated to the task's module | hard separation ceiling |
| `modular_task_routed` (`readout_coord`) | full `x` → both; per-task readout head | gradient through both modules |

All modular variants use two non-communicating Elman cells (`H/2` units each) and
a shared readout (except `readout_coord`, which has one head per task). Recurrent
cells are bias-free; the hidden state resets each presentation; inputs are repeated
for `seq_len = 3` timesteps with loss at the final step.

## Package layout

| Path | Purpose |
|------|---------|
| `toy_task/config.py` | frozen constants + experimental grids |
| `toy_task/environment.py` | latents, observation function, targets, sampling |
| `toy_task/models.py` | dense + 3 modular wirings |
| `toy_task/init_scale.py` | `apply_init_scale(model, γ)` |
| `toy_task/training.py` | A1→B→A2 loop, transfer/interference, extraction |
| `toy_task/analysis.py` | PCA, participation ratio, principal angles, RSA, CKA, drift |
| `toy_task/storage.py` | per-run `config.json` + `extractions.npz` |
| `scripts/` | experiment driver, figures, task visualization |
| `tests/` | environment, model/router, training, analysis checks |

## Tests

```bash
python -m pytest tests/ -q
```
