"""
Simulation storage policy and primary-grid classification helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

PRIMARY_GRID_INITS = {0.001, 0.01, 0.1, 1.0, 2.0}
PRIMARY_GRID_TWO_MODULE_HIDDEN = {6, 12, 25, 50}
PRIMARY_GRID_SINGLE_MODULE_HIDDEN = {12, 25, 50, 100}

PRIMARY_SIM_SUBDIR = "simulations"
ABLATION_SIM_SUBDIR = "simulations/primary_grid_ablations"


def normalized_init_scale(condition: dict[str, Any]) -> float:
    """Return init_scale, treating missing as 1.0."""
    return float(condition.get("init_scale", 1.0))


def is_primary_grid_condition(condition: dict[str, Any]) -> bool:
    """
    Classification contract for primary-grid storage.

    Primary grid is intentionally strict:
    - nb_steps=2, common_input=False, common_readout=True
    - cell_type=RNN, n_layers=1, dropout=0.0
    - two_module_rnn: dim_hidden in {6,12,25,50}, sparsity==0 (no_comms)
    - single_module_rnn: dim_hidden in {12,25,50,100}
    - init_scale in {0.001, 0.01, 0.1, 1.0, 2.0} (missing => 1.0)
    """
    arch = condition.get("arch")
    if arch not in ("two_module_rnn", "single_module_rnn"):
        return False

    if int(condition.get("nb_steps", 1)) != 2:
        return False
    if bool(condition.get("common_input", False)) is not False:
        return False
    if bool(condition.get("common_readout", True)) is not True:
        return False
    if condition.get("cell_type", "RNN") != "RNN":
        return False
    if int(condition.get("n_layers", 1)) != 1:
        return False
    if float(condition.get("dropout", 0.0)) != 0.0:
        return False

    init_scale = normalized_init_scale(condition)
    if init_scale not in PRIMARY_GRID_INITS:
        return False

    dim_hidden = int(condition.get("dim_hidden", -1))
    if arch == "two_module_rnn":
        if dim_hidden not in PRIMARY_GRID_TWO_MODULE_HIDDEN:
            return False
        return abs(float(condition.get("sparsity", 1.0))) < 1e-9

    # single_module_rnn
    if dim_hidden not in PRIMARY_GRID_SINGLE_MODULE_HIDDEN:
        return False
    return True


def resolve_sim_root(condition: dict[str, Any], data_root: Path, mode: str = "auto") -> Path:
    """
    Return target root for condition outputs.

    mode:
      - auto: classifier-based (primary vs ablation)
      - primary: force primary root
      - ablation: force ablation root
    """
    mode = mode.lower()
    if mode not in {"auto", "primary", "ablation"}:
        raise ValueError(f"Invalid storage mode: {mode}")

    if mode == "primary":
        return data_root / PRIMARY_SIM_SUBDIR
    if mode == "ablation":
        return data_root / ABLATION_SIM_SUBDIR

    if is_primary_grid_condition(condition):
        return data_root / PRIMARY_SIM_SUBDIR
    return data_root / ABLATION_SIM_SUBDIR

