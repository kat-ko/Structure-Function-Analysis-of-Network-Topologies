"""Experiment driver: sweep ``arch x H x gamma x s x seed`` (Section 7).

Examples
--------
Full default grid (7000 runs)::

    python scripts/run_experiment.py --out data/runs

A small pilot::

    python scripts/run_experiment.py --out data/pilot \
        --hidden-sizes 24 50 --gammas 0.1 1.0 \
        --similarity-indices 0 3 5 --seeds 0 1
"""

from __future__ import annotations

import argparse
import itertools
import time

from toy_task.config import (
    RunConfig,
    ARCHITECTURES,
    ARCH_MODULAR_TASK_ROUTED,
    OFF_MODULE_POLICIES,
    SIMILARITY_GRID,
    GAMMA_GRID,
    HIDDEN_SIZE_GRID,
    SEEDS,
    ANGLE_EVEN,
    ANGLE_MODES,
    STIMULUS_REGIMES,
    STIMULUS_NOVEL,
)
from toy_task.environment import Environment
from toy_task.training import run_experiment
from toy_task.storage import save_result


def _arch_configs(archs, off_policies):
    """Yield (arch, off_module_policy) effective configs."""
    for arch in archs:
        if arch == ARCH_MODULAR_TASK_ROUTED:
            for off in off_policies:
                yield arch, off
        else:
            yield arch, "freeze"  # off policy unused for non-task-routed


def build_configs(args) -> list[RunConfig]:
    sims = (
        [(i, SIMILARITY_GRID[i]) for i in args.similarity_indices]
        if args.similarity_indices
        else list(enumerate(SIMILARITY_GRID))
    )
    configs = []
    for seed in args.seeds:
        for arch, off in _arch_configs(args.arch, args.off_module_policy):
            for H in args.hidden_sizes:
                for gamma in args.gammas:
                    for s_idx, s in sims:
                        configs.append(
                            RunConfig(
                                arch=arch,
                                hidden_size=H,
                                gamma=gamma,
                                similarity=s,
                                seed=seed,
                                off_module_policy=off,
                                epochs_per_phase=args.epochs_per_phase,
                                similarity_index=s_idx,
                                stimulus_regime=args.stimulus_regime,
                                angle_mode=args.angle_mode,
                            )
                        )
    return configs


def main() -> None:
    p = argparse.ArgumentParser(description="toy_task experiment driver")
    p.add_argument("--out", default="data/runs", help="output root directory")
    p.add_argument("--arch", nargs="+", default=list(ARCHITECTURES), choices=list(ARCHITECTURES))
    p.add_argument("--off-module-policy", nargs="+", default=list(OFF_MODULE_POLICIES),
                   choices=list(OFF_MODULE_POLICIES))
    p.add_argument("--hidden-sizes", nargs="+", type=int, default=list(HIDDEN_SIZE_GRID))
    p.add_argument("--gammas", nargs="+", type=float, default=list(GAMMA_GRID))
    p.add_argument("--similarity-indices", nargs="+", type=int, default=None,
                   help=f"indices into the {len(SIMILARITY_GRID)}-point similarity grid")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--epochs-per-phase", type=int, default=100)
    p.add_argument("--stimulus-regime", default=STIMULUS_NOVEL, choices=list(STIMULUS_REGIMES))
    p.add_argument("--angle-mode", default=ANGLE_EVEN, choices=list(ANGLE_MODES))
    p.add_argument("--print-config", action="store_true", help="list configs and exit")
    p.add_argument("--save-env", action="store_true", help="store environment arrays per run")
    args = p.parse_args()

    configs = build_configs(args)
    print(f"Planned runs: {len(configs)}")
    if args.print_config:
        for cfg in configs:
            print(" ", cfg.run_id())
        return

    t0 = time.time()
    for i, cfg in enumerate(configs, 1):
        res = run_experiment(cfg)
        env = (
            Environment.from_seed(
                cfg.seed,
                stimulus_regime=cfg.stimulus_regime,
                angle_mode=cfg.angle_mode,
            )
            if args.save_env
            else None
        )
        save_result(res, args.out, env=env)
        if i % 10 == 0 or i == len(configs):
            dt = time.time() - t0
            print(f"  [{i}/{len(configs)}] {cfg.run_id()}  ({dt:.1f}s elapsed)")
    print(f"Done: {len(configs)} runs in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
