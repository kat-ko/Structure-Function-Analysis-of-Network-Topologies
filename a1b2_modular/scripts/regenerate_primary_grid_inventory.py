#!/usr/bin/env python3
"""
Regenerate PRIMARY_GRID_RUN_INVENTORY.md: scan data/simulations for each
standard-primary-grid condition (experiments.json), count sim_*.npz and whether
name-matched state_<id>.pt exists for each (von Mises pipeline companion).

Row selection matches `is_primary_grid_condition` (same as STANDARD_PRIMARY_GRID_RUN_INVENTORY.md).

Run from a1b2_modular:
  python3 scripts/regenerate_primary_grid_inventory.py
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
        "# Primary grid run inventory",
        "",
        f"*Last refreshed: {ts}*",
        "",
        "All `experiments.json` conditions matching the **standard primary grid** "
        "(`is_primary_grid_condition` in `a1b2.utils.sim_storage`): "
        "`nb_steps=2`, `common_input=False`, `common_readout=True`, `cell_type=RNN`, "
        "`n_layers=1`, `dropout=0`, init ∈ {0.001, 0.01, 0.1, 1, 2}; "
        "for `two_module_rnn`, **no_comms only** (`sparsity=0`).",
        "",
        "For a dedicated tracker with the same rows, see "
        "[`STANDARD_PRIMARY_GRID_RUN_INVENTORY.md`](STANDARD_PRIMARY_GRID_RUN_INVENTORY.md).",
        "",
        "- **Folder:** `data/simulations/<run_id>/` (folder name equals `run_id`).",
        "- **State checkpoints:** `state_<participant_id>.pt` for each `sim_<participant_id>.npz` "
        "inside the run folder.",
        "- **Von Mises fits:** `data/simulations/<run_id>_vonmises_fits.csv` (sibling of the run folder; "
        "from `scripts/03_fit_vonmises.py simulations --sim-name <run_id>`).",
        "- **Single-module baseline:** `single_module` rows use a capacity-matched hidden size "
        "for that `dim_h` column (e.g. **100** hidden units when `dim_h=50`, comparable to "
        "two modules × 50). Condition names use `single_module_rnn_<hidden>_nb2…`.",
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
    parser = argparse.ArgumentParser(description="Regenerate primary grid run inventory markdown.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output markdown path (default: a1b2_modular/PRIMARY_GRID_RUN_INVENTORY.md)",
    )
    args = parser.parse_args()

    root = _project_root()
    sys.path.insert(0, str(root))
    from a1b2.utils.run_config import build_run_id

    out = args.output if args.output is not None else root / "PRIMARY_GRID_RUN_INVENTORY.md"
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
