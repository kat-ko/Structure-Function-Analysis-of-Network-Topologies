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

import os
from concurrent.futures import ProcessPoolExecutor

MAX_WORKERS = 128

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
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs))
