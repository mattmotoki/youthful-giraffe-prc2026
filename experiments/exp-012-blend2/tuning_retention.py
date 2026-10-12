"""How much of a MEMBER improvement does the blend keep?

This prices any tuning assignment. Seed bagging was measured at 7 % retention
(RESULTS), but that is variance reduction, which the blend already performs --
a tuning gain is a genuinely better model and should carry further. So it is
measured rather than assumed, using the largest member improvement this lane
has: CatBoost's round budget, `C1-d8` at 425-600 rounds against `C2-d8p` at
5,325, worth 15.45 s to the member on july_loo.

Both versions are swapped into the same rotated blend, everything else held.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parents[1]), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import blend as BL  # noqa: E402
import prc.blend as B  # noqa: E402

MEMBERS = {
    "T-tailv2": "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v2.parquet",
    "C2-cat":   "experiments/exp-008-diversity/oof/{arm}-{fold}.parquet",
    "L2-hub":   "experiments/exp-012-blend2/pinned/L2-hub--L2-hub-{fold}.parquet",
    "D1-winter": "experiments/exp-007-winter/oof/d1-ds-s7-{fold}.parquet",
}

def run(arm, mode):
    files = {k: v.replace("{arm}", arm) for k, v in MEMBERS.items()}
    data, names = BL.load(files)
    T, _ = BL.rotate(data, names, mode)
    per = {r.fold: r.rmse for r in T.itertuples()}
    s = BL.singles(data, names)
    cat = {r.fold: r.rmse for r in s.itertuples() if r.member == "C2-cat"}
    return per, cat

for mode in ("nnls", "flat"):
    print(f"\n=== blend mode {mode}")
    print(f"{'CatBoost arm':>12} {'member july_loo':>16} {'member board':>13} "
          f"{'blend board':>12}")
    out = {}
    for arm in ("C1-d8", "C2-d8p"):
        per, cat = run(arm, mode)
        out[arm] = (BL.board_estimate(cat), BL.board_estimate(per))
        print(f"{arm:>12} {cat['july_loo']:16.2f} {out[arm][0]:13.2f} {out[arm][1]:12.3f}")
    dm = out["C1-d8"][0] - out["C2-d8p"][0]
    db = out["C1-d8"][1] - out["C2-d8p"][1]
    print(f"  member gain {dm:.2f} s  ->  blend gain {db:.3f} s   "
          f"retention {100*db/dm:.0f} %")
