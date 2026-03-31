"""
Paper notebooks: single-module baseline selection and sparsity broadcast.

Two-module per-module hidden h → total width ~2h; single baseline uses dim_hidden ≈ 2h.
Size12 notebook uses 25 as nearest existing width to 24.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, List

import numpy as np
import pandas as pd

# two_module per-module dim_hidden -> single_module dim_hidden
BASELINE_SINGLE_HIDDEN: dict[int, int] = {
    6: 12,
    12: 25,
    25: 50,
    50: 100,
}

ROUTING_HUE_ORDER: List[str] = ["task_routed", "shared", "single_module"]

# tab10 blue, orange; third = green for single_module
ROUTING_PALETTE: List[str] = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def baseline_single_hidden(two_module_per_module_h: int) -> int:
    if two_module_per_module_h not in BASELINE_SINGLE_HIDDEN:
        raise KeyError(
            f"No single-module baseline mapped for two_module dim_hidden={two_module_per_module_h!r}; "
            f"expected one of {sorted(BASELINE_SINGLE_HIDDEN)}"
        )
    return BASELINE_SINGLE_HIDDEN[two_module_per_module_h]


def select_single_baseline_df(
    settings: dict[str, Any],
    sim_folder: Path,
    *,
    two_module_dim_hidden: int,
    paper_sparsities: Iterable[str],
    paper_init_scales: Iterable[float],
    sparsity_order: list[str],
    build_run_id: Callable[[dict], str],
    sparsity_label_from_condition: Callable[[dict], str],
    init_scale_from_condition: Callable[[dict], float],
    target_nb_steps: int = 2,
    target_common_readout: bool = True,
    target_common_input: bool = False,
) -> pd.DataFrame:
    """Rows for single_module_rnn, full comms (sparsity 1), matching paper init scales."""
    dim = baseline_single_hidden(two_module_dim_hidden)
    paper_sparsities = set(paper_sparsities)
    paper_init_scales = set(float(x) for x in paper_init_scales)

    conds: list[dict[str, Any]] = []
    for c in settings["conditions"]:
        if c.get("arch") != "single_module_rnn":
            continue
        if c.get("dim_hidden") != dim:
            continue
        if c.get("n_modules", 1) != 1:
            continue
        if c.get("nb_steps", 1) != target_nb_steps:
            continue
        if c.get("common_readout", True) is not target_common_readout:
            continue
        if c.get("common_input", False) is not target_common_input:
            continue
        sp = float(c.get("sparsity", 1.0))
        if not np.isclose(sp, 1.0):
            continue

        sp_label = sparsity_label_from_condition(c)
        if sp_label not in paper_sparsities:
            continue
        init_scale = init_scale_from_condition(c)
        if init_scale not in paper_init_scales:
            continue

        run_id = build_run_id(c)
        path = sim_folder / run_id
        conds.append(
            {
                **c,
                "run_id": run_id,
                "path": str(path),
                "path_exists": path.exists(),
                "sparsity_label": sp_label,
                "init_scale": init_scale,
            }
        )

    df = pd.DataFrame(conds)
    if df.empty:
        return df
    df["sparsity_label"] = pd.Categorical(df["sparsity_label"], categories=sparsity_order, ordered=True)
    return df.sort_values(["sparsity_label", "init_scale", "name"]).reset_index(drop=True)


def broadcast_baseline_sparsity(df: pd.DataFrame | None, sparsity_order: list[str]) -> pd.DataFrame:
    """Repeat each baseline row for every sparsity facet (same numeric values)."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    parts: list[pd.DataFrame] = []
    for sp in sparsity_order:
        chunk = df.copy()
        chunk["sparsity_label"] = sp
        parts.append(chunk)
    out = pd.concat(parts, ignore_index=True)
    out["sparsity_label"] = pd.Categorical(out["sparsity_label"], categories=sparsity_order, ordered=True)
    return out
