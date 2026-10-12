"""COMMON.md's four required slices for every variant this experiment reports.

`COMMON.md` line 98 asks every RESULTS table for RMSE overall, on rows with the
target at or below 3600 s, on rows above it, and on no-NM rows, per fold.  The
tables in RESULTS SS 2, 6 and 7 report the rotation's clean RMSE and the board
instead, because that is the quantity the gate is written in.  This file is the
one place the four slices are given, for every variant those tables name, so the
requirement is met once rather than repeated in four shapes (Sol round 1, Spec 7).

The no-NM mask is `COMMON.md`'s own: **every** `_flt` column null, read from the
raw month, not `FLIGHT_ID_mvt` alone (Sol round 1, Standards 4).  Both are
printed so the difference is visible: it is 3 to 6 rows per fold.

    python slices.py            # -> analysis/slices.csv

Nothing is refitted: every number is arithmetic on the saved rotation vectors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import panel as P  # noqa: E402
import review1 as R1  # noqa: E402
import where as W  # noqa: E402
sys.path.insert(0, str(P.ROOT))
import prc.data as D  # noqa: E402

VARIANTS = {"stack-5": None,
            "all-but-noac": ("nm_bulk", "nm_late", "nonm"),
            "bulk+late": ("nm_bulk", "nm_late"),
            "bulk+nonm": ("nm_bulk", "nonm"),
            "bulk": ("nm_bulk",)}


def main() -> None:
    rows = []
    for tag in ("w300", "w600"):
        W.TAG = tag
        for f in P.FOLDS:
            d = W.load(f)
            y = d["y_true"].to_numpy("float64")
            a = d["y_pred_stack5"].to_numpy("float64")
            b = d["y_pred_depx"].to_numpy("float64")
            ids = d["MVT_ID_mvt"].to_numpy("float64")
            keep = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
            strict = R1.no_nm_mask(f, d["MVT_ID_mvt"].to_numpy())
            for name, regs in VARIANTS.items():
                p = a if regs is None else np.where(np.isin(d["regime"].to_numpy(), regs), b, a)
                e = ((p - y) ** 2)[keep]
                yy, ss = y[keep], strict[keep]
                bulk = yy <= 3600.0
                r = lambda m: round(float(np.sqrt(e[m].mean())), 4) if m.any() else float("nan")
                rows.append(dict(tag=tag, fold=f, variant=name,
                                 rmse=round(D.rmse_both(y, p, ids, f)["rmse_clean"], 4),
                                 le3600=r(bulk), gt3600=r(~bulk), no_nm=r(ss),
                                 no_nm_flight_id_only=r(d["nm_missing"].to_numpy()
                                                        .astype(bool)[keep]),
                                 n=int(keep.sum()), n_no_nm=int(ss.sum())))
    T = pd.DataFrame(rows)
    T.to_csv(P.EXP / "analysis" / "slices.csv", index=False)
    for tag in ("w300", "w600"):
        print(f"\n=== {tag}, clean rows, COMMON.md's four slices ===")
        print(T[T.tag == tag].drop(columns=["tag"]).to_string(index=False))


if __name__ == "__main__":
    main()
