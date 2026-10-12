#!/usr/bin/env python3
"""Check the vectorised cluster bootstrap against a brute-force implementation.

`paired()` avoids materialising the resampled row index by pre-summing squared
error per cluster with `np.add.reduceat` and re-weighting by cluster size. That
is easy to get subtly wrong, and it carries every interval in RESULTS section
6.2 and 6.3, so it is checked against the obvious slow version on the same RNG
stream. Sol review 2026-09-05, blocking finding 2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import paired  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(7)
    n, k = 4000, 60
    cluster = rng.integers(0, k, size=n)
    y = rng.normal(1000, 300, n)
    a = y + rng.normal(0, 100, n)
    b = y + rng.normal(0, 90, n)

    ra, rb, lo, hi = paired(y, a, b, cluster, np.random.default_rng(0), draws=4000)

    groups = {c: np.flatnonzero(cluster == c) for c in range(k)}
    g = np.random.default_rng(0)
    diffs = []
    for _ in range(4000):
        pick = g.integers(0, k, size=k)
        idx = np.concatenate([groups[c] for c in pick])
        diffs.append(np.sqrt(np.mean((y[idx] - b[idx]) ** 2))
                     - np.sqrt(np.mean((y[idx] - a[idx]) ** 2)))
    diffs = np.asarray(diffs)
    blo, bhi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    checks = [
        ("rmse(a)", np.isclose(ra, np.sqrt(np.mean((y - a) ** 2)))),
        ("rmse(b)", np.isclose(rb, np.sqrt(np.mean((y - b) ** 2)))),
        ("interval low", np.isclose(lo, blo, atol=1e-9)),
        ("interval high", np.isclose(hi, bhi, atol=1e-9)),
    ]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"vectorised [{lo:.6f}, {hi:.6f}]  brute force [{blo:.6f}, {bhi:.6f}]")
    if not all(ok for _, ok in checks):
        sys.exit(1)
    print("cluster bootstrap matches brute force")


if __name__ == "__main__":
    main()
