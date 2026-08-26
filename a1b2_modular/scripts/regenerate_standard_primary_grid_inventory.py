#!/usr/bin/env python3
"""
Regenerate STANDARD_PRIMARY_GRID_RUN_INVENTORY.md: same row set as the storage-policy
standard primary grid (`is_primary_grid_condition` in a1b2.utils.sim_storage), scanned
only under data/simulations/ (not primary_grid_ablations).

Run from a1b2_modular:
  python3 scripts/regenerate_standard_primary_grid_inventory.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from primary_grid_inventory_core import collect_standard_primary_rows, esc


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent


def render_markdown(rows: list[tuple], ts: str) -> str:
    lines = [
        "# Standard primary grid run inventory",
        "",
        f"*Last refreshed: {ts}*",
        "",
        "This file tracks **only** the **standard primary grid** as defined by "
        "`a1b2.utils.sim_storage.is_primary_grid_condition`: "
        "`nb_steps=2`, `common_input=False`, `common_readout=True`, `cell_type=RNN`, "
        "`n_layers=1`, `dropout=0`, init ∈ {0.001, 0.01, 0.1, 1, 2}; "
        "`two_module_rnn` with **no_comms** (`sparsity=0`); capacity-matched `single_module_rnn`.",
        "",
        "- **Folder:** `data/simulations/<run_id>/` only (ablations live under "
        "`data/simulations/primary_grid_ablations/`).",
        "- **State checkpoints:** `state_<participant_id>.pt` for each `sim_<participant_id>.npz` "
        "inside the run folder.",
        "- **Von Mises fits:** `data/simulations/<run_id>_vonmises_fits.csv` (sibling of the run folder; "
        "from `scripts/03_fit_vonmises.py simulations --sim-name <run_id>`).",
        "- **Regenerate:** `python3 scripts/regenerate_standard_primary_grid_inventory.py`",
        f"- **Total rows:** {len(rows)}",
        "",
        "| dim_h | routing | sparsity | init | condition | run_id (= folder) | exists | npz | state matched | state OK | VM OK |",
        "| ---: | --- | --- | --- | --- | --- | :---: | ---: | --- | :---: | :---: |",
    ]
    for dim_h, routing, sp, ini, name, rid, ex, n, sm, state_ok, vm_ok in rows:
        fe = "Yes" if ex else "No"
        lines.append(
            f"| {dim_h} | {esc(routing)} | {esc(sp)} | {ini} | {esc(name)} | `{esc(rid)}` | {fe} | {n} | {sm} | {state_ok} | {vm_ok} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate standard primary grid inventory markdown.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output markdown path (default: a1b2_modular/STANDARD_PRIMARY_GRID_RUN_INVENTORY.md)",
    )
    args = parser.parse_args()

    root = _project_root()
    sys.path.insert(0, str(root))
    from a1b2.utils.run_config import build_run_id

    out = args.output if args.output is not None else root / "STANDARD_PRIMARY_GRID_RUN_INVENTORY.md"
    config_path = root / "a1b2" / "models" / "experiments.json"
    sim_folder = root / "data" / "simulations"

    settings = json.loads(config_path.read_text(encoding="utf-8"))
    rows = collect_standard_primary_rows(settings, sim_folder, build_run_id)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out.write_text(render_markdown(rows, ts), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
