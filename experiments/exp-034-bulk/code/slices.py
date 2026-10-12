"""`COMMON.md`'s four slices for the bulk MEMBER, on each fold.

Sol round 1, blocking 5: `assignments/COMMON.md` requires RMSE overall, on
`y <= 3600`, on `y > 3600` and on no-NM rows, for every fold, in every RESULTS
table.  The member table had the bulk slice only.

The member is undefined off the NM join, so the no-NM slice is **N/A by
construction** and is reported as such with its row count rather than filled.
Every other slice is computed on the rows where the member is defined, and the
number of rows each slice loses to that is printed beside it.  The mask is
`review1.no_nm_mask`, exp-028's corrected one ("every `_flt` column null"), the
same mask `score.py` uses for the candidate arms.

    python slices.py   ->  analysis/member-slices.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
sys.path.insert(0, str(ROOT))
import review1 as R1  # noqa: E402
import candidate as C  # noqa: E402
import frames as F  # noqa: E402
import prc.data as D  # noqa: E402


def main() -> None:
    rows = []
    for fold in C.FOLDS:
        m = F.fold_frame(fold, C.NAMES5 + (C.K2, C.BULK))
        y = m["y_true"].to_numpy("float64")
        ids = m["MVT_ID_mvt"].to_numpy("float64")
        v = m[C.BULK].to_numpy("float64")
        ok = np.isfinite(v)
        clean = ~pd.Series(ids).isin(D.excluded_ids(fold)).to_numpy()
        nonm = R1.no_nm_mask(fold, ids)
        noac = m["regime"].to_numpy() == "noac"
        base = clean & ok
        def r(mask):
            mm = base & mask
            if not mm.any():
                return None, 0, int((clean & mask).sum())
            return (round(float(np.sqrt(np.mean((y[mm] - v[mm]) ** 2))), 4), int(mm.sum()),
                    int((clean & mask).sum()))
        for name, mask in (("overall", np.ones(len(y), bool)),
                           ("le3600", y <= 3600.0),
                           ("gt3600", y > 3600.0),
                           ("no_nm", nonm),
                           ("noac", noac),
                           ("bulk (le3600, NM, not noac)", (y <= 3600.0) & ~noac)):
            val, n_used, n_clean = r(mask)
            rows.append({"fold": fold, "slice": name,
                         "rmse": "N/A" if val is None else val,
                         "n_scored": n_used, "n_in_slice": n_clean,
                         "n_undefined_in_slice": n_clean - n_used})
    d = pd.DataFrame(rows)
    d.to_csv(EXP / "analysis" / "member-slices.csv", index=False)
    print(d.to_string(index=False))
    print("\n-> analysis/member-slices.csv")
    print("N/A = the member is undefined on every row of the slice (no NM join, "
          "hence no nm_taxi, hence no prediction).")


if __name__ == "__main__":
    main()
