# gecco — GECCO late-breaking experiment track

Minimal, **standalone** code for a small multi-objective evolution experiment on **Holton-style** continual data ([Holton et al., 2025](https://doi.org/10.1038/s41562-025-02318-y)), without importing **`a1b2_modular`**.

- **Upstream reference:** see `REFERENCE_UPSTREAM_PROJECTS.md`.
- **Abstract narrative (problem / prior / gap / contribution):** `LBA_NARRATIVE.md`.

## Install

From this directory:

```bash
pip install -e .
```

## Data

Point to a `trial_df.csv` (same schema as `transfer-interference` after `01_preprocess_data.py`). Example:

```bash
export GECCO_TRIAL_DF=/path/to/transfer-interference/data/participants/trial_df.csv
```

Or pass `--trial-df` to the script below.

## One-command LBA figure

```bash
python -m gecco.experiments.run_lba_figure \
  --trial-df "${GECCO_TRIAL_DF}" \
  --out figures/lba_pareto.png \
  --generations 18 --population 32 --train-steps 120
```

Defaults use a **tiny** participant subset and **fast** training so you can iterate on a laptop; increase for cleaner fronts.

**Note (LBA speed trade-off):** the default `DualRouteRNN` uses **two parallel `Linear → tanh` encoders** with hard input masks (not a recurrent core over trial index). That keeps evolution cheap while preserving the **routing genome** story. Swapping in `nn.RNNCell` per module is the natural next step for a full “modular RNN” claim.

**Notebook:** `notebooks/lba_experiments.ipynb` — run GA + save **Pareto** plot, **MSE histogram**, and a **CSV** of Pareto genomes (set kernel cwd to `gecco/` or use `sys.path` as in the first code cell).

**Interpreting results:** `EXPECTED_OUTCOMES_AND_FIGURES.md` — hypothesis-linked success criteria and what a convincing figure would look like.

**Overnight GPU batches:** `scripts/overnight/README.md` — four scripts (GPUs 4–7), two sequential jobs each (seeds, init-scale sweep, Same condition, longer A/B pattern); uses repo-root `gpu_block.sh`.

**Preliminary title / abstract / results checklist:** `PRELIMINARY_TITLE_ABSTRACT_AND_GAPS.md`.
