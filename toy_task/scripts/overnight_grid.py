"""Overnight experiment grid: parallel, resumable sweep for the WIP paper.

Two blocks (see ``build_configs``):

  Block 1 - main factorial (init_scope=all):
      arch(5) x gamma(5) x similarity(7) x regime(2) x hidden(4) x seed(3) = 4200
  Block 2 - readout-scaling ablation (init_scope=no_readout):
      arch(5) x gamma(5) x similarity{Same,Near,Far}(3) x regime(2) x H=50 x seed(3) = 450

Runs are single-threaded each and dispatched across a process pool. Already-completed
run directories are skipped, so the job is safe to re-launch after an interruption.

Examples
--------
Dry run (count + first ids)::

    python scripts/overnight_grid.py --dry-run

Full overnight job::

    python scripts/overnight_grid.py --out data/runs_overnight --workers 64 --epochs 8000
"""

from __future__ import annotations

# Pin BLAS/OpenMP to one thread *before* importing torch so each pooled worker
# stays single-threaded and we avoid fork+OpenMP oversubscription.
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import torch

from toy_task.config import (
    RunConfig,
    ARCHITECTURES,
    ARCH_MODULAR_TASK_ROUTED,
    OFF_MODULE_POLICIES,
    OFF_FREEZE,
    SIMILARITY_GRID,
    GAMMA_GRID,
    INIT_SCOPE_ALL,
    INIT_SCOPE_NO_READOUT,
    STIMULUS_SHARED,
    STIMULUS_NOVEL,
    ANGLE_EVEN,
    ANGLE_MODES,
    SIM_IDX_SAME,
    SIM_IDX_NEAR,
    SIM_IDX_FAR,
)
from toy_task.training import run_experiment
from toy_task.storage import save_result


# --------------------------------------------------------------------------- grid
HIDDEN_FULL = [12, 24, 50, 100]
HIDDEN_REF = 50
REGIMES = [STIMULUS_SHARED, STIMULUS_NOVEL]
PAPER_SIM = [SIM_IDX_SAME, SIM_IDX_NEAR, SIM_IDX_FAR]


def _arch_configs():
    """Yield effective (arch, off_module_policy) pairs."""
    for arch in ARCHITECTURES:
        if arch == ARCH_MODULAR_TASK_ROUTED:
            for off in OFF_MODULE_POLICIES:
                yield arch, off
        else:
            yield arch, OFF_FREEZE  # unused for non-task-routed


def build_configs(seeds, epochs, angle_mode: str = ANGLE_EVEN) -> list[RunConfig]:
    cfgs: dict[str, RunConfig] = {}

    def add(arch, off, H, gamma, s_idx, regime, scope, seed):
        cfg = RunConfig(
            arch=arch,
            hidden_size=H,
            gamma=gamma,
            similarity=SIMILARITY_GRID[s_idx],
            seed=seed,
            off_module_policy=off,
            epochs_per_phase=epochs,
            similarity_index=s_idx,
            init_scope=scope,
            stimulus_regime=regime,
            angle_mode=angle_mode,
        )
        cfgs.setdefault(cfg.run_id(), cfg)

    # Block 1: main factorial (init_scope = all).
    for seed in seeds:
        for arch, off in _arch_configs():
            for H in HIDDEN_FULL:
                for gamma in GAMMA_GRID:
                    for s_idx in range(len(SIMILARITY_GRID)):
                        for regime in REGIMES:
                            add(arch, off, H, gamma, s_idx, regime, INIT_SCOPE_ALL, seed)

    # Block 2: readout-scaling ablation (init_scope = no_readout), reference H, paper sims.
    for seed in seeds:
        for arch, off in _arch_configs():
            for gamma in GAMMA_GRID:
                for s_idx in PAPER_SIM:
                    for regime in REGIMES:
                        add(arch, off, HIDDEN_REF, gamma, s_idx, regime, INIT_SCOPE_NO_READOUT, seed)

    return list(cfgs.values())


# ----------------------------------------------------------------------- workers
def _is_complete(out_root: str, run_id: str) -> bool:
    d = Path(out_root) / run_id
    return (d / "config.json").is_file() and (d / "extractions.npz").is_file()


def _run_one(args: tuple[RunConfig, str]) -> tuple[str, str, float]:
    cfg, out_root = args
    rid = cfg.run_id()
    if _is_complete(out_root, rid):
        return rid, "skip", 0.0
    t0 = time.time()
    try:
        torch.set_num_threads(1)
        res = run_experiment(cfg)
        save_result(res, out_root)
        return rid, "ok", time.time() - t0
    except Exception as exc:  # keep the sweep alive; report at the end
        return rid, f"ERROR: {type(exc).__name__}: {exc}", time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser(description="overnight WIP grid")
    p.add_argument("--out", default="data/runs_overnight")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8000, help="epochs_per_phase")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--angle-mode", default=ANGLE_EVEN, choices=list(ANGLE_MODES),
                   help="object ring layout (even=patch default; random=legacy)")
    p.add_argument("--dry-run", action="store_true", help="print plan and exit")
    p.add_argument("--limit", type=int, default=0, help="cap #configs (smoke test)")
    args = p.parse_args()

    cfgs = build_configs(args.seeds, args.epochs, angle_mode=args.angle_mode)
    if args.limit:
        cfgs = cfgs[: args.limit]

    out = args.out
    todo = [c for c in cfgs if not _is_complete(out, c.run_id())]
    done = len(cfgs) - len(todo)
    print(f"Total configs: {len(cfgs)}  |  already complete: {done}  |  to run: {len(todo)}")
    print(f"Output: {out}  |  workers: {args.workers}  |  epochs/phase: {args.epochs}"
          f"  |  angle_mode: {args.angle_mode}")
    if args.dry_run:
        for c in cfgs[:20]:
            print("  ", c.run_id())
        if len(cfgs) > 20:
            print(f"  ... (+{len(cfgs) - 20} more)")
        return
    if not todo:
        print("Nothing to do.")
        return

    Path(out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    errors: list[str] = []
    work = [(c, out) for c in todo]
    with Pool(processes=args.workers) as pool:
        for i, (rid, status, dt) in enumerate(pool.imap_unordered(_run_one, work, chunksize=1), 1):
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                errors.append(f"{rid}  {status}")
            if i % 25 == 0 or i == len(work):
                el = time.time() - t0
                rate = i / el if el > 0 else 0.0
                eta = (len(work) - i) / rate if rate > 0 else 0.0
                print(f"  [{i}/{len(work)}] ok={n_ok} err={n_err}  "
                      f"{el/60:.1f}min elapsed, ETA {eta/60:.1f}min  (last {dt:.1f}s)")

    print(f"Done: {n_ok} ok, {n_skip} skipped, {n_err} errors in {(time.time()-t0)/60:.1f} min -> {out}")
    if errors:
        print("Errors:")
        for e in errors[:50]:
            print("  ", e)


if __name__ == "__main__":
    main()
