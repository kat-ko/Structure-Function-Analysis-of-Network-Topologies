#!/usr/bin/env python3
"""
Audit and migrate simulation folders between primary and ablation roots.

Default behavior is dry-run audit only.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from a1b2.utils.run_config import build_run_id
from a1b2.utils.sim_storage import (
    ABLATION_SIM_SUBDIR,
    PRIMARY_SIM_SUBDIR,
    is_primary_grid_condition,
)


@dataclass
class Row:
    condition_name: str
    run_id: str
    classification: str
    expected_root: str
    in_primary: bool
    in_ablation: bool
    action: str
    reason: str


def _load_conditions(config_path: Path) -> list[dict]:
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    return settings["conditions"]


def _rows_for_conditions(conditions: list[dict], primary_root: Path, ablation_root: Path) -> list[Row]:
    rows: list[Row] = []
    for c in conditions:
        name = c.get("name", "unknown")
        run_id = build_run_id(c)
        is_primary = is_primary_grid_condition(c)
        cls = "primary" if is_primary else "ablation"
        in_primary = (primary_root / run_id).is_dir()
        in_ablation = (ablation_root / run_id).is_dir()

        if in_primary and in_ablation:
            action = "conflict"
            reason = "run_id exists in both roots"
        elif cls == "primary":
            action = "keep_primary"
            reason = "primary condition must remain in primary root"
        elif in_primary:
            action = "move_to_ablation"
            reason = "ablation currently in primary root"
        elif in_ablation:
            action = "keep_ablation"
            reason = "already in ablation root"
        else:
            action = "missing"
            reason = "no folder found in either root"

        rows.append(
            Row(
                condition_name=name,
                run_id=run_id,
                classification=cls,
                expected_root=PRIMARY_SIM_SUBDIR if is_primary else ABLATION_SIM_SUBDIR,
                in_primary=in_primary,
                in_ablation=in_ablation,
                action=action,
                reason=reason,
            )
        )
    return rows


def _write_reports(rows: list[Row], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"simulation_layout_audit_{ts}.csv"
    md_path = out_dir / f"simulation_layout_audit_{ts}.md"

    fields = [
        "condition_name",
        "run_id",
        "classification",
        "expected_root",
        "in_primary",
        "in_ablation",
        "action",
        "reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.action] = counts.get(r.action, 0) + 1

    lines = [
        "# Simulation layout audit",
        "",
        f"- Total conditions: {len(rows)}",
        f"- move_to_ablation: {counts.get('move_to_ablation', 0)}",
        f"- keep_primary: {counts.get('keep_primary', 0)}",
        f"- keep_ablation: {counts.get('keep_ablation', 0)}",
        f"- missing: {counts.get('missing', 0)}",
        f"- conflict: {counts.get('conflict', 0)}",
        "",
        "| condition_name | run_id | classification | in_primary | in_ablation | action | reason |",
        "| --- | --- | --- | :---: | :---: | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.condition_name} | `{r.run_id}` | {r.classification} | "
            f"{'Yes' if r.in_primary else 'No'} | {'Yes' if r.in_ablation else 'No'} | {r.action} | {r.reason} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _run_migration(
    rows: list[Row],
    *,
    primary_root: Path,
    ablation_root: Path,
    apply: bool,
    allow_overwrite: bool,
    log_path: Path,
) -> tuple[int, int]:
    conflicts = [r for r in rows if r.action == "conflict"]
    if conflicts:
        raise RuntimeError("Conflicts detected: same run_id exists in both roots. Resolve before migration.")

    primary_violations = [r for r in rows if r.classification == "primary" and r.action == "move_to_ablation"]
    if primary_violations:
        raise RuntimeError("Safety guard: primary conditions flagged for move; aborting.")

    to_move = [r for r in rows if r.action == "move_to_ablation"]
    moved = 0
    skipped = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for r in to_move:
        src = primary_root / r.run_id
        dst = ablation_root / r.run_id
        if not src.is_dir():
            skipped += 1
            continue
        if dst.exists() and not allow_overwrite:
            raise RuntimeError(f"Destination exists (use --allow-overwrite): {dst}")
        if dst.exists() and allow_overwrite:
            if apply:
                shutil.rmtree(dst)
            else:
                skipped += 1
                continue

        record = {
            "time": int(time.time()),
            "run_id": r.run_id,
            "condition_name": r.condition_name,
            "src": str(src),
            "dst": str(dst),
            "applied": bool(apply),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if apply:
            ablation_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        moved += 1

    return moved, skipped


def _reverse_from_log(log_file: Path, apply: bool, allow_overwrite: bool) -> tuple[int, int]:
    if not log_file.is_file():
        raise FileNotFoundError(f"Missing log file: {log_file}")
    records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    reversed_count = 0
    skipped = 0
    for rec in reversed(records):
        src = Path(rec["dst"])
        dst = Path(rec["src"])
        if not src.is_dir():
            skipped += 1
            continue
        if dst.exists() and not allow_overwrite:
            raise RuntimeError(f"Reverse destination exists (use --allow-overwrite): {dst}")
        if dst.exists() and allow_overwrite:
            if apply:
                shutil.rmtree(dst)
            else:
                skipped += 1
                continue
        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        reversed_count += 1
    return reversed_count, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-folder", type=Path, default=_root, help="a1b2_modular root")
    parser.add_argument("--apply", action="store_true", help="Apply moves (default: dry-run)")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow replacing destination folder")
    parser.add_argument("--migrate", action="store_true", help="Migrate ablation runs from primary to ablation root")
    parser.add_argument("--reverse-from-log", type=Path, default=None, help="Reverse moves from JSONL log")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Report output directory (default: data/simulations/primary_grid_ablations/reports)",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Migration log path (default: data/simulations/primary_grid_ablations/migration_log.jsonl)",
    )
    args = parser.parse_args()

    root = args.base_folder.resolve()
    data_root = root / "data"
    primary_root = data_root / PRIMARY_SIM_SUBDIR
    ablation_root = data_root / ABLATION_SIM_SUBDIR
    config_path = root / "a1b2" / "models" / "experiments.json"
    report_dir = args.report_dir or (ablation_root / "reports")
    log_path = args.log_path or (ablation_root / "migration_log.jsonl")

    if args.reverse_from_log is not None:
        reversed_count, skipped = _reverse_from_log(
            args.reverse_from_log, apply=args.apply, allow_overwrite=args.allow_overwrite
        )
        print(f"{'Applied' if args.apply else 'Dry-run'} reverse: {reversed_count}, skipped: {skipped}")
        return 0

    conditions = _load_conditions(config_path)
    rows = _rows_for_conditions(conditions, primary_root=primary_root, ablation_root=ablation_root)
    csv_path, md_path = _write_reports(rows, report_dir)
    print(f"Audit written: {csv_path}")
    print(f"Audit written: {md_path}")

    if args.migrate:
        moved, skipped = _run_migration(
            rows,
            primary_root=primary_root,
            ablation_root=ablation_root,
            apply=args.apply,
            allow_overwrite=args.allow_overwrite,
            log_path=log_path,
        )
        print(f"{'Applied' if args.apply else 'Dry-run'} migration entries: {moved}, skipped: {skipped}")
        print(f"Migration log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

