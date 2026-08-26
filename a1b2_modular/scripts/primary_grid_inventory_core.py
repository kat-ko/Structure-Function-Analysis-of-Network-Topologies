"""
Shared helpers for primary / standard-primary run inventory markdown generation.
"""
from __future__ import annotations

from pathlib import Path


def sparsity_label(c: dict) -> str:
    sp = c.get("sparsity", 1.0)
    if sp == 0 or "no_comms" in c.get("name", ""):
        return "no_comms"
    sf = float(sp)
    return "1.0" if abs(sf - 1.0) < 1e-9 else str(sf)


def init_scale(c: dict) -> float:
    from a1b2.utils.sim_storage import normalized_init_scale

    return normalized_init_scale(c)


def _vonmises_fits_status(sim_folder: Path, run_id: str, n_npz: int) -> str:
    """
    Von Mises fits from scripts/03_fit_vonmises.py are written next to the run folder:
    ``<sim_folder>/<run_id>_vonmises_fits.csv`` (not inside <run_id>/).
    """
    if not n_npz:
        return "N/A"
    csv_path = sim_folder / f"{run_id}_vonmises_fits.csv"
    if not csv_path.is_file():
        return "No"
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            n_lines = sum(1 for _ in f)
        n_rows = max(0, n_lines - 1)  # header
        if n_rows >= n_npz:
            return "Yes"
        if n_rows > 0:
            return "Partial"
        return "No"
    except OSError:
        return "No"


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
        sm, state_ok = "—", "N/A"
    elif n == 0:
        sm, state_ok = "0", "N/A"
    elif matched == n:
        sm, state_ok = f"{matched}/{n}", "Yes"
    else:
        sm, state_ok = f"{matched}/{n}", "Partial"
    vm_ok = _vonmises_fits_status(sim_folder, rid, n)
    return (
        dim_h,
        routing_label,
        sparsity_label(c),
        init_scale(c),
        c.get("name", ""),
        rid,
        exists,
        n,
        sm,
        state_ok,
        vm_ok,
    )


def collect_standard_primary_rows(settings: dict, sim_folder: Path, build_run_id) -> list[tuple]:
    """Rows for conditions matching `is_primary_grid_condition` (storage policy standard grid)."""
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
