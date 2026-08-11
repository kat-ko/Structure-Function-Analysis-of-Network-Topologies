"""Process-pool helper for the measurement scripts.

BLAS threads are pinned to 1 in the workers: the per-sample linear algebra is
small (P×P Grams, one NNLS), so multi-threaded BLAS only adds contention once we
are already running one process per core.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

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
    n = max(1, (n or 1) - 2)
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
