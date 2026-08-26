"""QUICK smoke grid: novel + even angles (post-nb09 default).

Mirrors the nb08 paper slice at reduced scale for go/no-go before a full
overnight re-run with ``angle_mode=even``.

Example::

    python scripts/even_novel_smoke_grid.py --out data/runs_even_smoke --epochs 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from toy_task.config import (
    RunConfig,
    ARCHITECTURES,
    ARCH_MODULAR_TASK_ROUTED,
    OFF_MODULE_POLICIES,
    OFF_FREEZE,
    SIMILARITY_GRID,
    QUICK_GAMMAS,
    ANGLE_EVEN,
    STIMULUS_NOVEL,
    PAPER_SIM_IDX,
    INIT_SCOPE_ALL,
)
from toy_task.training import run_experiment
from toy_task.storage import save_result


def _arch_pairs():
    for arch in ARCHITECTURES:
        if arch == ARCH_MODULAR_TASK_ROUTED:
            for off in OFF_MODULE_POLICIES:
                yield arch, off
        else:
            yield arch, OFF_FREEZE


def build_smoke_configs(seeds, epochs, hidden_size=50) -> list[RunConfig]:
    cfgs: dict[str, RunConfig] = {}
    for seed in seeds:
        for arch, off in _arch_pairs():
            for gamma in QUICK_GAMMAS:
                for s_idx in PAPER_SIM_IDX:
                    cfg = RunConfig(
                        arch=arch,
                        hidden_size=hidden_size,
                        gamma=gamma,
                        similarity=SIMILARITY_GRID[s_idx],
                        seed=seed,
                        off_module_policy=off,
                        epochs_per_phase=epochs,
                        similarity_index=s_idx,
                        init_scope=INIT_SCOPE_ALL,
                        stimulus_regime=STIMULUS_NOVEL,
                        angle_mode=ANGLE_EVEN,
                    )
                    cfgs[cfg.run_id()] = cfg
    return list(cfgs.values())


def _complete(out: Path, run_id: str) -> bool:
    d = out / run_id
    return (d / "config.json").is_file() and (d / "extractions.npz").is_file()


def main() -> None:
    p = argparse.ArgumentParser(description="even-angle novel smoke grid")
    p.add_argument("--out", default="data/runs_even_smoke")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--hidden-size", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    cfgs = build_smoke_configs(args.seeds, args.epochs, args.hidden_size)
    todo = [c for c in cfgs if not _complete(out, c.run_id())]
    print(f"configs={len(cfgs)}  todo={len(todo)}  out={out}  epochs={args.epochs}")
    if args.dry_run:
        for c in cfgs[:5]:
            print(" ", c.run_id())
        return

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, cfg in enumerate(todo, 1):
        res = run_experiment(cfg)
        save_result(res, out)
        if i % 5 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {cfg.run_id()}  ({time.time()-t0:.0f}s)")
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
