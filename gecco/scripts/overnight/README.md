# Overnight GECCO / LBA runs (GPUs 4–7)

Eight **sequential** jobs in four scripts (two per GPU). Each script should run in a **separate terminal** after locking one GPU with `gpu_block.sh`.

## Why these runs

| GPU | Run A | Run B | Rationale |
|-----|--------|--------|-----------|
| **4** | seed 0 | seed 1 | **Replication** — see if Near/Far Pareto ordering is stable. |
| **5** | seed 2 | seed 3 | More seeds for the same “production” hyperparameters. |
| **6** | `init-scale 0.001` | `init-scale 2.0` | **Rich vs lazy** scale on **evolved** dual-path nets (matches your a1b2 narrative). |
| **7** | `near far same` | long `ABABABAB` + more trials | **Same** condition baseline + **harder** continual stress (more A/B repeats). |

Shared **overnight** preset (in `_common.sh`): pop **44**, gen **24**, train-steps **180**, **3** participants/condition, `--plot-all-population` (faint cloud).

Logs + timestamped run folder: `gecco/runs/overnight/YYYYMMDD_HHMM/`.  
Figures: `gecco/figures/overnight/*.png`.

## Commands (four terminals)

From the **monorepo root** (where `gpu_block.sh` lives):

```bash
# Terminal 1
source gpu_block.sh 4
bash gecco/scripts/overnight/gpu4_two_runs.sh

# Terminal 2
source gpu_block.sh 5
bash gecco/scripts/overnight/gpu5_two_runs.sh

# Terminal 3
source gpu_block.sh 6
bash gecco/scripts/overnight/gpu6_two_runs.sh

# Terminal 4
source gpu_block.sh 7
bash gecco/scripts/overnight/gpu7_two_runs.sh
```

Install once per env:

```bash
cd gecco && pip install -e .
```

## Overrides

```bash
export TRIAL_DF=/abs/path/to/trial_df.csv
export RUN_DIR=/abs/path/custom_run_dir   # optional; default is timestamped under gecco/runs/overnight/
```

## If two jobs per GPU must run in parallel

Not configured here (VRAM). Options: (1) halve `population` / `train-steps` per job, or (2) run **only one** script per GPU and move two seeds to daytime.

## After morning

- Compare `figures/overnight/gpu4_*.png` … `gpu5_*.png` for **seed stability**.  
- Compare `gpu6_*.png` for **init-scale** effect on Pareto vs stars.  
- `gpu7_near_far_same.png`: check **Same** vs **Near/Far** geometry.  
- See `EXPECTED_OUTCOMES_AND_FIGURES.md` for how to read the plots.
