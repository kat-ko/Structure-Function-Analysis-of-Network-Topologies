#!/usr/bin/env python3
"""
Regenerate PRIMARY_GRID_RUN_INVENTORY.md: scan data/simulations for each
paper primary-grid condition (experiments.json), count sim_*.npz and whether
name-matched state_<id>.pt exists for each (von Mises pipeline companion).

Run from a1b2_modular:
  python3 scripts/regenerate_primary_grid_inventory.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent


def sparsity_label(c: dict) -> str:
    sp = c.get("sparsity", 1.0)
    if sp == 0 or "no_comms" in c.get("name", ""):
        return "no_comms"
    sf = float(sp)
    return "1.0" if abs(sf - 1.0) < 1e-9 else str(sf)


def init_scale(c: dict) -> float:
    from a1b2.utils.sim_storage import normalized_init_scale

    return normalized_init_scale(c)


def row_for_condition(
    c: dict,
    routing_label: str,
    dim_h: int,
    *,
    build_run_id,
    sim_folder: Path,
) -> tuple:
    rid = build_run_id(c)
    folder = sim_folder / rid
    exists = folder.is_dir()
    npz_files = list(folder.glob("sim_*.npz")) if exists else []
    n = len(npz_files)
    matched = 0
    for p in npz_files:
        stem = p.stem
        pid = stem[len("sim_") :] if stem.startswith("sim_") else stem
        if (folder / f"state_{pid}.pt").is_file():
            matched += 1
    if not exists:
        sm, ok = "—", "N/A"
    elif n == 0:
        sm, ok = "0", "N/A"
    elif matched == n:
        sm, ok = f"{matched}/{n}", "Yes"
    else:
        sm, ok = f"{matched}/{n}", "Partial"
    return (dim_h, routing_label, sparsity_label(c), init_scale(c), c.get("name", ""), rid, exists, n, sm, ok)


def collect_rows(settings: dict, sim_folder: Path, build_run_id) -> list[tuple]:
    from a1b2.utils.sim_storage import is_primary_grid_condition

    size_grid = [6, 12, 25, 50]
    baseline_single_hidden = {6: 12, 12: 25, 25: 50, 50: 100}

    rows: list[tuple] = []
    for h in size_grid:
        for routing in ("task_routed", "shared"):
            for c in settings["conditions"]:
                if c.get("arch") != "two_module_rnn":
                    continue
                if c.get("dim_hidden") != h:
                    continue
                if c.get("nb_steps", 1) != 2:
                    continue
                if c.get("common_readout", True) is not True:
                    continue
                if c.get("common_input", False) is not False:
                    continue
                if c.get("input_routing", "shared") != routing:
                    continue
                if c.get("init_scope") == "input_only":
                    continue
                if not is_primary_grid_condition(c):
                    continue
                rows.append(
                    row_for_condition(c, routing, h, build_run_id=build_run_id, sim_folder=sim_folder)
                )

        sh = baseline_single_hidden[h]
        for c in settings["conditions"]:
            if c.get("arch") != "single_module_rnn":
                continue
            if c.get("dim_hidden") != sh:
                continue
            if c.get("n_modules", 1) != 1:
                continue
            if c.get("nb_steps", 1) != 2:
                continue
            if c.get("common_readout", True) is not True:
                continue
            if c.get("common_input", False) is not False:
                continue
            if abs(float(c.get("sparsity", 1.0)) - 1.0) > 1e-9:
                continue
            if not is_primary_grid_condition(c):
                continue
            rows.append(
                row_for_condition(c, "single_module", h, build_run_id=build_run_id, sim_folder=sim_folder)
            )

    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[5]))
    return rows


def esc(s: str) -> str:
    return str(s).replace("|", "\\|")


def render_markdown(rows: list[tuple], ts: str) -> str:
    lines = [
        "# Primary grid run inventory",
        "",
        f"*Last refreshed: {ts}*",
        "",
        "All `experiments.json` conditions matching the primary comparison grid "
        "(`nb_steps=2`, `common_input=False`, `common_readout=True`, init ∈ {0.001, 0.01, 0.1, 1, 2}); "
        "for `two_module_rnn`, the primary grid uses **no_comms only** (`sparsity=0`).",
        "",
        "- **Folder:** `data/simulations/<run_id>/` (folder name equals `run_id`).",
        "- **Von Mises companion (name-matched):** `state_<participant_id>.pt` for each "
        "`sim_<participant_id>.npz` in the same folder.",
        "- **Single-module baseline:** `single_module` rows use a capacity-matched hidden size "
        "for that `dim_h` column (e.g. **100** hidden units when `dim_h=50`, comparable to "
        "two modules × 50). Condition names use `single_module_rnn_<hidden>_nb2…`.",
        f"- **Total rows:** {len(rows)}",
        "",
        "| dim_h | routing | sparsity | init | condition | run_id (= folder) | exists | npz | state matched | VM OK |",
        "| ---: | --- | --- | --- | --- | --- | :---: | ---: | --- | :---: |",
    ]
    for dim_h, routing, sp, ini, name, rid, ex, n, sm, ok in rows:
        fe = "Yes" if ex else "No"
        lines.append(
            f"| {dim_h} | {esc(routing)} | {esc(sp)} | {ini} | {esc(name)} | `{esc(rid)}` | {fe} | {n} | {sm} | {ok} |"
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
    rows = collect_rows(settings, sim_folder, build_run_id)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = render_markdown(rows, ts)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
