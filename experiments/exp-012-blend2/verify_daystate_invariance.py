"""Is the winter member's day-state column the same whether the frame is masked?

Adopted from exp-018's `verify_extras.py` (2026-09-06) rather than merely cited.
The winter member hands `day_state_step` a ranking-MASKED validation frame in the
fold path and per-ranking-month frames in the submission path, and this lane had
been treating mask-invariance as a docstring promise — the same class of
assumption exp-018 stopped trusting in three of their own modules.

Reading the module says the FIT cannot matter: `DayState.transform` uses
`reference_` only for the `_z` columns and joins `_deice(frame)` unconditionally.
That leaves the FRAME, which reading cannot settle. This measures it.

Exits non-zero on any difference, so it is a test rather than a note.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W7 = ROOT / "experiments" / "exp-007-winter" / "code"
for p in (str(ROOT), str(W7 / "pipeline"), str(W7)):
    if p not in sys.path:
        sys.path.insert(0, p)
import prc.data as D  # noqa: E402
import day_state_step as DS  # noqa: E402


def main(month: str = "2025-03") -> int:
    raw = D.load_month(month)
    masked, _ = D.mask_like_ranking(raw)
    blanked = sum(int(raw[c].notna().sum() - masked[c].notna().sum())
                  for c in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt") if c in raw.columns)
    print(f"{month}: the mask blanks {blanked:,} label cells")

    bad = 0
    for name in ("deice", "core", "all"):
        cols = DS.SETS[name]
        a = DS._one(raw, DS.DayState().fit(raw), cols)
        b = DS._one(masked, DS.DayState().fit(raw), cols)
        if len(a) != len(b):
            print(f"  {name:6s} ROW COUNT DIFFERS {len(a)} vs {len(b)}"); bad += 1; continue
        diff = []
        for c in cols:
            x, y = a[c].to_numpy("float64"), b[c].to_numpy("float64")
            same = np.array_equal(x, y, equal_nan=True)
            if not same:
                d = np.abs(np.nan_to_num(x) - np.nan_to_num(y))
                diff.append((c, int((d > 0).sum()), float(d.max())))
        if diff:
            bad += len(diff)
            for c, n, mx in diff[:6]:
                print(f"  {name:6s} DIFFERS {c}: {n:,} rows, max {mx:.6g}")
        else:
            print(f"  {name:6s} {len(cols):2d}/{len(cols):2d} columns bit-identical")
    print("\nMASK-INVARIANT" if not bad else f"\n{bad} PROBLEM(S) — a column depends on the mask")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2] or [])))
