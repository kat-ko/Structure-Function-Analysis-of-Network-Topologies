"""Toy task: standalone modular structure-function experiments.

A synthetic continual-learning benchmark (A1 -> B(s) -> A2) for studying how
structural priors (dense vs. modular RNNs) and learning regime (rich/lazy via init
scale) shape representational organization as task similarity varies.
"""

__version__ = "0.1.0"

from .config import (
    CONSTANTS,
    RunConfig,
    ARCHITECTURES,
    OFF_MODULE_POLICIES,
    INIT_SCOPE_ALL,
    INIT_SCOPE_NO_READOUT,
    INIT_SCOPE_INPUT_ONLY,
    INIT_SCOPES,
    STIMULUS_SHARED,
    STIMULUS_NOVEL,
    STIMULUS_REGIMES,
    ANGLE_RANDOM,
    ANGLE_EVEN,
    ANGLE_MODES,
    SIMILARITY_GRID,
    PAPER_SIM_IDX,
    PAPER_SIM_LABELS,
    SIM_IDX_SAME,
    SIM_IDX_NEAR,
    SIM_IDX_FAR,
    FULL_SIM_IDX,
    QUICK_GAMMAS,
    GAMMA_GRID,
    HIDDEN_SIZE_GRID,
    SEEDS,
    COMMS_BANDWIDTH_GRID,
)
from .environment import Environment
from .models import ToyRNN, build_model
from .init_scale import apply_init_scale
from .training import run_experiment, evaluate_clean, RunResult

__all__ = [
    "CONSTANTS",
    "RunConfig",
    "ARCHITECTURES",
    "OFF_MODULE_POLICIES",
    "INIT_SCOPE_ALL",
    "INIT_SCOPE_NO_READOUT",
    "INIT_SCOPE_INPUT_ONLY",
    "INIT_SCOPES",
    "STIMULUS_SHARED",
    "STIMULUS_NOVEL",
    "STIMULUS_REGIMES",
    "ANGLE_RANDOM",
    "ANGLE_EVEN",
    "ANGLE_MODES",
    "SIMILARITY_GRID",
    "PAPER_SIM_IDX",
    "PAPER_SIM_LABELS",
    "SIM_IDX_SAME",
    "SIM_IDX_NEAR",
    "SIM_IDX_FAR",
    "FULL_SIM_IDX",
    "QUICK_GAMMAS",
    "GAMMA_GRID",
    "HIDDEN_SIZE_GRID",
    "SEEDS",
    "COMMS_BANDWIDTH_GRID",
    "Environment",
    "ToyRNN",
    "build_model",
    "apply_init_scale",
    "run_experiment",
    "evaluate_clean",
    "RunResult",
]
