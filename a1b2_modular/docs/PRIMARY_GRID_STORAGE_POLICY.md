# Primary Grid Storage Policy

This policy separates **standard primary-grid** results from **ablations** while preserving existing primary-grid folders in place.

## Storage roots

- **Primary grid (must stay in place):** `data/simulations/<run_id>/`
- **Ablations:** `data/simulations/primary_grid_ablations/<run_id>/`

## Primary-grid condition contract

A condition is primary only if all rules are satisfied:

- `nb_steps == 2`
- `common_input == false`
- `common_readout == true`
- `cell_type == "RNN"` (GRU/LSTM are ablations)
- `n_layers == 1` (`nl2`/`nl3` are ablations)
- `dropout == 0.0` (dropout pilots are ablations)
- `init_scale in {0.001, 0.01, 0.1, 1.0, 2.0}` (missing `init_scale` is treated as `1.0`)

Two-module primary:
- `arch == "two_module_rnn"`
- `dim_hidden in {6, 12, 25, 50}`
- `sparsity == 0` (no_comms)

Single-module primary:
- `arch == "single_module_rnn"`
- `dim_hidden in {12, 25, 50, 100}`

Everything else is ablation.

## Standard primary grid inventory (tracked file)

Regenerate the markdown tracker (only standard-primary rows, scanned under `data/simulations/`):

```bash
python3 scripts/regenerate_standard_primary_grid_inventory.py
```

Output: `STANDARD_PRIMARY_GRID_RUN_INVENTORY.md` (project root of `a1b2_modular`).

The legacy-named file `PRIMARY_GRID_RUN_INVENTORY.md` uses the **same row set**; keep both in sync by running either script after config changes.

## Run commands (future-safe)

From `a1b2_modular`:

```bash
python3 scripts/02_run_simulations.py <condition_name> --base-folder .
```

Default `--storage-mode auto` routes output by policy. Optional overrides:

```bash
python3 scripts/02_run_simulations.py <condition_name> --base-folder . --storage-mode primary
python3 scripts/02_run_simulations.py <condition_name> --base-folder . --storage-mode ablation
python3 scripts/02_run_simulations.py <condition_name> --base-folder . --print-output-path
```

## Migration SOP (safety-first)

1. **Audit only (default dry-run)**

```bash
python3 scripts/audit_simulation_layout.py --base-folder .
```

2. **Review generated reports** under:
- `data/simulations/primary_grid_ablations/reports/`

3. **Dry-run migration preview**

```bash
python3 scripts/audit_simulation_layout.py --base-folder . --migrate
```

4. **Apply migration** (ablation-only moves)

```bash
python3 scripts/audit_simulation_layout.py --base-folder . --migrate --apply
```

5. **Post-check audit**

```bash
python3 scripts/audit_simulation_layout.py --base-folder .
```

## Rollback

Use the migration log:

```bash
python3 scripts/audit_simulation_layout.py --reverse-from-log data/simulations/primary_grid_ablations/migration_log.jsonl
python3 scripts/audit_simulation_layout.py --reverse-from-log data/simulations/primary_grid_ablations/migration_log.jsonl --apply
```

Rollback follows log entries in reverse order.
