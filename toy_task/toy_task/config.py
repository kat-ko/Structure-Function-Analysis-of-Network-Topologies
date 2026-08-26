"""Frozen benchmark constants and experimental grids for toy_task.

These mirror Parts I-V of ``description/`` with the patch
(``benchmark_specification_patch.md``) applied. Hidden size ``H`` is promoted
from a frozen constant to a swept variable per project decision (see
``IMPLEMENTATION_PLAN.md`` Section 3.6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List


# --- Architecture identifiers -------------------------------------------------
ARCH_DENSE = "dense"
ARCH_MODULAR_SHARED = "modular_shared"
ARCH_MODULAR_FEATURE_ROUTED = "modular_feature_routed"
ARCH_MODULAR_TASK_ROUTED = "modular_task_routed"

ARCHITECTURES = (
    ARCH_DENSE,
    ARCH_MODULAR_SHARED,
    ARCH_MODULAR_FEATURE_ROUTED,
    ARCH_MODULAR_TASK_ROUTED,
)

# Off-task module behavior for ``modular_task_routed`` (Section 3.4).
OFF_FREEZE = "freeze"
OFF_READOUT_COORD = "readout_coord"
OFF_MODULE_POLICIES = (OFF_FREEZE, OFF_READOUT_COORD)

# Init-scale (gamma) application scope. Mirrors a1b2_modular's rnn_init scopes.
#   "all"        : scale every trainable parameter, including the readout (a1b2 "global").
#   "no_readout" : scale recurrent + input weights, leave the readout at standard init.
#   "input_only" : scale only the input-to-hidden weights (a1b2 "input_only").
INIT_SCOPE_ALL = "all"
INIT_SCOPE_NO_READOUT = "no_readout"
INIT_SCOPE_INPUT_ONLY = "input_only"
INIT_SCOPES = (INIT_SCOPE_ALL, INIT_SCOPE_NO_READOUT, INIT_SCOPE_INPUT_ONLY)

# Stimulus regime for task B (Holton-style rule reuse vs output remapping).
#   "shared" : task B reuses task A's objects; B target = rule s applied to the
#              same stimuli (pure output remapping; A and B share inputs).
#   "novel"  : task B draws a fresh, independent object set; the *rule* s is the
#              only shared structure, so success requires transferring the rule
#              to unseen stimuli (the Holton-faithful regime).
# In BOTH regimes the per-object reference angle is a function of the input
# (atan2 of the first two latent coordinates), so an abstract input->angle rule
# exists and can generalize. This differs from the legacy index-based angle;
# the run_id always carries a regime token so new runs never alias old caches.
STIMULUS_SHARED = "shared"
STIMULUS_NOVEL = "novel"
STIMULUS_REGIMES = (STIMULUS_SHARED, STIMULUS_NOVEL)

# Per-object reference-angle layout (``Environment.from_seed`` / ``RunConfig``).
#   "even"   : patch-faithful ring ``theta_i = 2*pi*(i-1)/N`` (benchmark default).
#              ``run_id`` includes ``angeven`` so caches stay disjoint from legacy runs.
#   "random" : legacy ``theta_i = atan2(z_{i,1}, z_{i,0})`` with ``z ~ N(0, I)``.
#              ``run_id`` omits any ``ang*`` token (backward-compatible with overnight cache).
ANGLE_RANDOM = "random"
ANGLE_EVEN = "even"
ANGLE_MODES = (ANGLE_RANDOM, ANGLE_EVEN)


# --- Default experimental grids ----------------------------------------------
# Task-similarity grid (patch Section 3): 0, pi/12, pi/6, pi/4, pi/3, pi/2, pi.
SIMILARITY_GRID: List[float] = [
    0.0,
    math.pi / 12,
    math.pi / 6,
    math.pi / 4,
    math.pi / 3,
    math.pi / 2,
    math.pi,
]

# Paper-comparable Same / Near / Far (matches a1b2_modular: s = 0, pi/6, pi).
SIM_IDX_SAME = 0
SIM_IDX_NEAR = 2
SIM_IDX_FAR = 6
PAPER_SIM_IDX: List[int] = [SIM_IDX_SAME, SIM_IDX_NEAR, SIM_IDX_FAR]
PAPER_SIM_LABELS = {SIM_IDX_SAME: "Same", SIM_IDX_NEAR: "Near", SIM_IDX_FAR: "Far"}

# Full benchmark similarity sweep (all 7 grid points) vs paper triple for QUICK / categorical plots.
FULL_SIM_IDX: List[int] = list(range(len(SIMILARITY_GRID)))

# QUICK smoke: rich + lazy extremes only (matches notebook convention).
QUICK_GAMMAS: List[float] = [0.001, 2.0]

# Init-scale (rich->lazy) ladder, matching a1b2_modular primary grid.
GAMMA_GRID: List[float] = [0.001, 0.01, 0.1, 1.0, 2.0]

# Total hidden-unit grid (modular per-module size is H/2).
HIDDEN_SIZE_GRID: List[int] = [12, 24, 50, 100]

# Independent environments.
SEEDS: List[int] = list(range(10))

# Inter-module recurrent coupling (a1b2 ``comms`` pathway). 0 = block-diagonal only.
COMMS_BANDWIDTH_GRID: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass(frozen=True)
class BenchmarkConstants:
    """Constants frozen across all experiments (patch Section 1)."""

    n_objects: int = 8          # N
    d_latent: int = 4           # d_z
    d_obs: int = 8              # d_x
    d_out: int = 2              # d_y
    n_modules: int = 2
    seq_len: int = 3            # repeated-input timesteps per presentation
    epochs_per_phase: int = 100
    sigma_train: float = 0.05
    sigma_eval: float = 0.0
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    batch_size: int = 1
    extract_every: int = 10     # extract representations every N epochs (patch Section 8)


CONSTANTS = BenchmarkConstants()


@dataclass
class RunConfig:
    """Full specification of one A1->B->A2 run.

    One run is a single point on the ``arch x H x gamma x s x seed`` grid.
    """

    arch: str
    hidden_size: int
    gamma: float
    similarity: float
    seed: int
    off_module_policy: str = OFF_FREEZE
    epochs_per_phase: int = CONSTANTS.epochs_per_phase
    # Index of similarity in SIMILARITY_GRID (for compact run ids); -1 if custom.
    similarity_index: int = -1
    # Scope of the gamma init-scaling (see INIT_SCOPES). Default scales everything.
    init_scope: str = INIT_SCOPE_ALL
    # Stimulus regime for task B (see STIMULUS_REGIMES). Default reuses A's objects.
    stimulus_regime: str = STIMULUS_SHARED
    # Object ring layout (see ANGLE_MODES). Default is patch-faithful even spacing.
    angle_mode: str = ANGLE_EVEN
    # Scalar inter-module recurrent coupling on modular architectures (a1b2 comms).
    comms_bandwidth: float = 0.0

    def __post_init__(self) -> None:
        if self.arch not in ARCHITECTURES:
            raise ValueError(f"Unknown arch {self.arch!r}; expected one of {ARCHITECTURES}")
        if self.off_module_policy not in OFF_MODULE_POLICIES:
            raise ValueError(
                f"Unknown off_module_policy {self.off_module_policy!r}; "
                f"expected one of {OFF_MODULE_POLICIES}"
            )
        if self.init_scope not in INIT_SCOPES:
            raise ValueError(
                f"Unknown init_scope {self.init_scope!r}; expected one of {INIT_SCOPES}"
            )
        if self.stimulus_regime not in STIMULUS_REGIMES:
            raise ValueError(
                f"Unknown stimulus_regime {self.stimulus_regime!r}; "
                f"expected one of {STIMULUS_REGIMES}"
            )
        if self.angle_mode not in ANGLE_MODES:
            raise ValueError(
                f"Unknown angle_mode {self.angle_mode!r}; expected one of {ANGLE_MODES}"
            )
        if not (0.0 <= self.comms_bandwidth <= 1.0):
            raise ValueError(
                f"comms_bandwidth must be in [0, 1]; got {self.comms_bandwidth}"
            )
        if self.hidden_size % CONSTANTS.n_modules != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by "
                f"n_modules {CONSTANTS.n_modules}"
            )

    @property
    def effective_comms_bandwidth(self) -> float:
        """Bandwidth applied by the model (dense always uses 0)."""
        return 0.0 if self.arch == ARCH_DENSE else self.comms_bandwidth

    @property
    def is_modular(self) -> bool:
        return self.arch != ARCH_DENSE

    @property
    def is_task_routed(self) -> bool:
        return self.arch == ARCH_MODULAR_TASK_ROUTED

    @property
    def per_module_size(self) -> int:
        return self.hidden_size // CONSTANTS.n_modules

    def run_id(self) -> str:
        """Compact, filesystem-safe identifier (Section 7).

        Includes ``epochs_per_phase`` (so runs with different training durations
        never collide), ``init_scope`` (so gamma-scaling variants are distinct),
        and ``stimulus_regime`` (so shared vs novel task-B object sets never mix).
        ``angle_mode=even`` adds an ``angeven`` token so even-spaced runs never
        alias legacy random-angle caches (which omit any ``ang*`` token).
        """
        s_idx = self.similarity_index if self.similarity_index >= 0 else 0
        parts = [self.arch]
        if self.is_task_routed:
            parts.append(self.off_module_policy)
        parts += [
            f"H{self.hidden_size}",
            f"g{self.gamma}",
            f"s{s_idx}",
            f"sc{self.init_scope}",
            f"r{self.stimulus_regime}",
        ]
        if self.angle_mode == ANGLE_EVEN:
            parts.append(f"ang{self.angle_mode}")
        parts.append(f"bw{self.comms_bandwidth}")
        parts += [
            f"e{self.epochs_per_phase}",
            f"seed{self.seed}",
        ]
        return "_".join(parts)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_id"] = self.run_id()
        return d
