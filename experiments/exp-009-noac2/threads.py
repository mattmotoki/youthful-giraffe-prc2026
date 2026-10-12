"""Thread budget for the `tail` lane: 6 runnable threads (docs/COMPUTE.md, compute allocation).

Import this FIRST, before numpy/pandas/lightgbm, so the BLAS pools are capped at
import time rather than after they are already built. `THREADS` is the number to
pass to every `n_jobs` / `num_threads` / `thread_count`.
"""
import os

THREADS = 6
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))


def gate(limit: float = 140.0) -> None:
    """Refuse to start a fit when the box is over the launch gate."""
    with open("/proc/loadavg") as f:
        one = float(f.read().split()[0])
    if one > limit:
        raise SystemExit(f"load {one:.0f} is over the {limit:.0f} launch gate; holding")
