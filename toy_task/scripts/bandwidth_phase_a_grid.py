"""Phase A grid: comms_bandwidth sweep on dense vs modular_shared.

Target ~180 runs (shared + even angles). Resumable; skips completed run dirs.

Example::

    python scripts/bandwidth_phase_a_grid.py --dry-run
    python scripts/bandwidth_phase_a_grid.py --epochs 500 --out data/runs_bandwidth_phase_a_pilot
    python scripts/bandwidth_phase_a_grid.py --epochs 8000 --workers 32
"""

from __future__ import annotations

# Pin BLAS/OpenMP to one thread before torch import (one core per worker).
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from toy_task.config import (
    RunConfig,
    ARCH_DENSE,
    ARCH_MODULAR_SHARED,
    SIMILARITY_GRID,
    PAPER_SIM_IDX,
    COMMS_BANDWIDTH_GRID,
    ANGLE_EVEN,
    STIMULUS_SHARED,
    INIT_SCOPE_ALL,
    OFF_FREEZE,
)
from toy_task.storage import save_result
from toy_task.training import run_experiment


PHASE_A_ARCHS = (ARCH_DENSE, ARCH_MODULAR_SHARED)
PHASE_A_GAMMAS = (0.001, 0.1, 2.0)
PHASE_A_SEEDS = (0, 1)
DEFAULT_HIDDEN = 50


def build_phase_a_configs(
    seeds: tuple[int, ...] = PHASE_A_SEEDS,
    gammas: tuple[float, ...] = PHASE_A_GAMMAS,
    bandwidths: tuple[float, ...] = tuple(COMMS_BANDWIDTH_GRID),
    epochs: int = 8000,
    hidden_size: int = DEFAULT_HIDDEN,
) -> list[RunConfig]:
    cfgs: dict[str, RunConfig] = {}
    for seed in seeds:
        for arch in PHASE_A_ARCHS:
            for bw in bandwidths:
                for gamma in gammas:
                    for s_idx in PAPER_SIM_IDX:
                        cfg = RunConfig(
                            arch=arch,
                            hidden_size=hidden_size,
                            gamma=gamma,
                            similarity=SIMILARITY_GRID[s_idx],
                            seed=seed,
                            off_module_policy=OFF_FREEZE,
                            epochs_per_phase=epochs,
                            similarity_index=s_idx,
                            init_scope=INIT_SCOPE_ALL,
                            stimulus_regime=STIMULUS_SHARED,
                            angle_mode=ANGLE_EVEN,
                            comms_bandwidth=bw,
                        )
                        cfgs[cfg.run_id()] = cfg
    return list(cfgs.values())


def _complete(out: Path, run_id: str) -> bool:
    d = out / run_id
    return (d / "config.json").is_file() and (d / "extractions.npz").is_file()


def _run_one(cfg_dict: dict, out_str: str) -> str:
    cfg = RunConfig(**{k: v for k, v in cfg_dict.items() if k != "run_id"})
    out = Path(out_str)
    res = run_experiment(cfg)
    save_result(res, out)
    return cfg.run_id()


def main() -> None:
    p = argparse.ArgumentParser(description="Phase A bandwidth grid")
    p.add_argument("--out", default="data/runs_bandwidth_phase_a")
    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    cfgs = build_phase_a_configs(epochs=args.epochs, hidden_size=args.hidden_size)
    todo = [c for c in cfgs if not _complete(out, c.run_id())]
    print(f"configs={len(cfgs)}  todo={len(todo)}  out={out}  epochs={args.epochs}")
    if args.dry_run:
        for c in cfgs[:8]:
            print(" ", c.run_id())
        print(" ...")
        return

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.workers <= 1:
        for i, cfg in enumerate(todo, 1):
            res = run_experiment(cfg)
            save_result(res, out)
            if i % 5 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {cfg.run_id()}  ({time.time()-t0:.0f}s)")
    else:
        cfg_dicts = [c.to_dict() for c in todo]
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run_one, d, str(out)): d["run_id"] for d in cfg_dicts
            }
            for fut in as_completed(futures):
                rid = fut.result()
                done += 1
                if done % 5 == 0 or done == len(todo):
                    print(f"  [{done}/{len(todo)}] {rid}  ({time.time()-t0:.0f}s)")
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
