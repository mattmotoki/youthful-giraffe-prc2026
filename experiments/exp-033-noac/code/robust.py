"""Step 3f: is the one surviving arm (the 2-4h correction) stable?

Per fold, per k, and against two things it must not depend on: the shrinkage
constant and the placeholder tolerance.  A gain that lives in one fold, or that
only exists at k = 5, is a selection artefact wearing an interval.

The NULL is not here.  Sol round 1, S6: the first version of this module carried
a placebo that drew a random subset of the same per-band size, which preserves
neither the geometry of a nearest-neighbour match nor its day structure, and
which moved the scoring population as well as the fit set.  `placebo2.py`
replaces it with a query permutation at a fixed scoring population.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
import direction as DIR  # noqa: E402
from describe import LAB  # noqa: E402
from direction import apply_head, fit_head, paired_rmse_diff, rmse  # noqa: E402

OOF = ROOT / "experiments" / "exp-026-rebuild" / "oof" / "stack5-{fold}.parquet"


def arms(a, mm, bands):
    """Rebuild the fold arms for a given matched mask, return the frame."""
    a = a.assign(_m=mm)
    keep = []
    for fold in ("july_loo", "december", "january"):
        vm = D.FOLDS[fold].valid_month
        oof = pd.read_parquet(str(OOF).format(fold=fold))[["MVT_ID_mvt", "y_pred"]]
        va = a[a.month == vm].merge(oof, on="MVT_ID_mvt", how="left")
        fit = a[a.month.isin(D.FOLDS[fold].train_months)]
        ha, hm = fit_head(fit), fit_head(fit[fit._m])
        c = apply_head(hm, va) - apply_head(ha, va)
        va["arm"] = va.y_pred + np.where(va.band.isin(bands), c, 0.0)
        va["fold"] = fold
        keep.append(va)
    return pd.concat(keep, ignore_index=True)


def report(v, tag):
    s = np.flatnonzero(v._m.to_numpy())
    y, base, arm, day = (v.y_true.to_numpy()[s], v.y_pred.to_numpy()[s],
                         v.arm.to_numpy()[s], v.day.to_numpy()[s])
    lo, hi = paired_rmse_diff(y, base, arm, day)
    g = rmse(y, base) - rmse(y, arm)
    print(f"  {tag:34s} n={len(s):4d} gain {g:+7.2f} [{lo:+7.2f}, {hi:+7.2f}]")
    return g


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    BANDS = ["2-4h"]

    print("=== the 2-4h correction, per fold (matched scope) ===")
    v = arms(a, DIR.matched_mask(a, b), BANDS)
    for fold in ("july_loo", "december", "january"):
        s = v[(v.fold == fold) & v._m]
        print(f"  {fold:10s} n={len(s):3d} shipped {rmse(s.y_true, s.y_pred):8.1f} "
              f"arm {rmse(s.y_true, s.arm):8.1f} gain {rmse(s.y_true, s.y_pred) - rmse(s.y_true, s.arm):+8.2f}")

    print("\n=== k sensitivity of the matched set ===")
    for k in (1, 3, 5, 10, 20):
        report(arms(a, DIR.matched_mask(a, b, k=k), BANDS), f"k={k}")

    print("\n=== space sensitivity ===")
    for sp in ("rowlevel", "full"):
        report(arms(a, DIR.matched_mask(a, b, space=sp), BANDS), f"space={sp}")

    print("\n=== shrinkage constant m ===")
    m0 = DIR.M_SHRINK
    for m in (0.0, 2.0, 5.0, 10.0, 25.0):
        DIR.M_SHRINK = m
        report(arms(a, DIR.matched_mask(a, b), BANDS), f"m={m}")
    DIR.M_SHRINK = m0

    print("\n=== placeholder tolerance ===")
    for tol in (120.0, 300.0, 600.0, 1200.0):
        aa = a.copy()
        aa["ph"] = (np.abs(aa.mvt_sched - aa.y_true) < tol).astype(float)
        report(arms(aa, DIR.matched_mask(aa, b), BANDS), f"tol={tol:.0f}s")

    print("\n(the placebo null lives in placebo2.py: this module only varies the knobs)")


if __name__ == "__main__":
    main()
