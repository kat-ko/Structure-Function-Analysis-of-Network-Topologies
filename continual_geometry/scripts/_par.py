"""Process-pool helper for the measurement scripts.

BLAS threads are pinned to 1 in the workers: the per-sample linear algebra is
small (P×P Grams, one NNLS), so multi-threaded BLAS only adds contention once we
are already running one process per core.

**Worker count is capped at 128, measured** (`results/scaling.json`). Throughput does
not increase with cores past that and then *falls*: 4.08 eval/s at 32 workers, 5.17 at
128, and **3.75 at 254 — worse than 32**. Two reasons. `nproc` reports 256 on this
2×64-core EPYC 7763 only because of SMT, so there are 128 physical cores; and the
anchor QP has `P·M = 2400` variables, whose Gram matrix at 46 MB exceeds the 32 MB L3,
so every worker streams it from DRAM and the workload is memory-bandwidth-bound.
Taking `nproc − 2` was silently running the grid at 73% of its own peak throughput.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

MAX_WORKERS = 128

# `spawn`, not the Linux default `fork`. A forked worker inherits the parent's
# already-imported modules, so a pool launched after an edit runs the *old* code while
# the repository holds the new — which is how an 8-arm pilot came to time a superseded
# solver. `spawn` re-imports from disk in every worker. It costs a second of startup
# per worker against arms that run for tens of minutes.
START_METHOD = "spawn"

_PIN = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def pin_threads() -> None:
    """Call before importing numpy in a worker entry point."""
    os.environ.update(_PIN)


def n_workers(cap: int | None = None) -> int:
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    n = min(max(1, (n or 1) - 2), MAX_WORKERS)
    return min(n, cap) if cap else n


def pmap(fn, jobs, *, cap: int | None = None, chatty: bool = True):
    """Run `fn` over `jobs` in a process pool, preserving order."""
    pin_threads()
    workers = min(n_workers(cap), max(1, len(jobs)))
    if chatty:
        print(f"  [{workers} workers x {len(jobs)} jobs]", flush=True)
    if workers == 1:
        return [fn(j) for j in jobs]
    ctx = mp.get_context(START_METHOD)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        return list(ex.map(fn, jobs))
