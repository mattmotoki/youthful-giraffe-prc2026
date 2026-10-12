"""Does clipping the day-state block help where exp-007 measured the harm?

The whole-fold comparison says clipping changes nothing (july_loo 307.61 against
307.62, january 354.11 against 354.10). But exp-007's +72.1 s was never a
whole-fold number: it was measured on **the most disrupted 5 % of airport-days**,
chosen label-free by `ds_span_std_z`. A 7,564-row slice moving 72 s is about
0.9 s on a 153,706-row fold, which is inside the noise of everything else.

So the honest comparison is on that slice, with exp-007's own definition
imported rather than re-implemented. The two predictions being compared come
from ONE fitted model and differ only in whether the day-state inputs were
allowed outside the range the trees were fitted on, so this is paired and
carries no seed or round-count noise.
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
import bootstrap_ds as BD  # noqa: E402


def rmse(y, p, m=None):
    if m is not None:
        y, p = y[m], p[m]
    return float(np.sqrt(((y - p) ** 2).mean())) if len(y) else float("nan")


def main(a_name="D1w-all", b_name="D1w-all-clip", folds=("july_loo", "january")):
    """Compare two arms on the disrupted slice.

    `D1w-all` against `D1w-all-clip` is PAIRED - one fitted model, two
    predictions - so its differences carry no seed or round-count noise. Any
    other pair, such as the block against the deice-only arm, is UNPAIRED: the
    two are separate fits with separately drawn round counts, and on this arm a
    re-draw alone has been measured at several seconds. Read an unpaired
    comparison as a level difference, not as an effect.
    """
    paired = b_name == f"{a_name}-clip"
    print(f"comparison: {a_name} vs {b_name}   "
          f"({'PAIRED, one model' if paired else 'UNPAIRED, separate fits'})")
    print(f"{'fold':>9} {'slice':>26} {'rows':>7} {a_name[:9]:>9} {b_name[:9]:>9} {'delta':>8}")
    for f in folds:
        a = pd.read_parquet(HERE / "oof" / f"{a_name}-{f}.parquet").sort_values("MVT_ID_mvt")
        b = pd.read_parquet(HERE / "oof" / f"{b_name}-{f}.parquet").sort_values("MVT_ID_mvt")
        assert (a["MVT_ID_mvt"].to_numpy() == b["MVT_ID_mvt"].to_numpy()).all()
        y = a["y_true"].to_numpy("float64")
        pa, pb = a["y_pred"].to_numpy("float64"), b["y_pred"].to_numpy("float64")
        ids = a["MVT_ID_mvt"].to_numpy()
        dz = BD.disrupted_days(f, ids)
        moved = np.abs(pa - pb) > 1e-9
        slices = [("whole fold", None), ("most disrupted 5 % of days", dz),
                  ("rest of the fold", ~dz)]
        if paired:
            slices += [("rows the clip moved", moved), ("disrupted AND moved", dz & moved)]
        for lab, m in slices:
            n = len(y) if m is None else int(m.sum())
            if n == 0:
                continue
            ra, rb = rmse(y, pa, m), rmse(y, pb, m)
            print(f"{f:>9} {lab:>26} {n:7d} {ra:9.2f} {rb:9.2f} {rb-ra:+8.2f}")
        if paired and moved.any():
            print(f"{'':>9} {'clip moved':>26} {int(moved.sum()):7d} rows, "
                  f"mean |delta| {np.abs(pa-pb)[moved].mean():.1f} s, "
                  f"max {np.abs(pa-pb).max():.1f} s")
        print()


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["D1w-all", "D1w-all-clip"]))
