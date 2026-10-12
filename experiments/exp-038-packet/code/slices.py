"""COMMON.md's four slices for the two Q01 finalists' bounded-amplitude blends.

The ledger's `july tail>3600` and `july no-NM` columns, on the same clean
population the point estimate uses, with exp-028's corrected no-NM mask
(`review1.no_nm_mask`: every `_flt` column null, not `FLIGHT_ID_mvt` alone).

    python slices.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
sys.path.insert(0, str(ROOT))
import align as A  # noqa: E402
import review1 as R1  # noqa: E402
import prc.data as D  # noqa: E402

FINALISTS = ("stack-5", "k2-w300")


def main() -> None:
    T = pd.read_csv(EXP / "analysis" / "q01-table.csv").set_index("candidate")
    cols = json.loads((EXP / "packet" / "columns.json").read_text())["d_columns"]
    inv = {v: k for k, v in cols.items()}
    rows = []
    for name in FINALISTS:
        for f in A.FOLDS:
            d0 = pd.read_parquet(EXP / "packet" / f"rows-{f}.parquet")
            ids = d0["MVT_ID_mvt"].to_numpy("float64")
            y = d0["y_true"].to_numpy("float64")
            s = float(T.loc[name, f"amp_loo_{f}"])
            p10 = d0["p_v10"].to_numpy("float64")
            p = p10 + s * d0[inv[name]].to_numpy("float64")
            ok = d0["clean"].to_numpy()
            strict = R1.no_nm_mask(f, ids)
            e = lambda v, m: (float(np.sqrt(np.mean((y[m] - v[m]) ** 2)))
                              if m.any() else float("nan"))
            for lab, m in (("all", ok), ("le3600", ok & (y <= 3600.0)),
                           ("gt3600", ok & (y > 3600.0)), ("no_nm", ok & strict),
                           ("noac", ok & (d0["regime"].to_numpy() == "noac")),
                           ("no_noac", ok & (d0["regime"].to_numpy() != "noac"))):
                rows.append({"candidate": name, "fold": f, "amplitude": round(s, 4),
                             "slice": lab, "n": int(m.sum()),
                             "rmse_v10": round(e(p10, m), 4), "rmse_blend": round(e(p, m), 4),
                             "gain": round(e(p10, m) - e(p, m), 4)})
    S = pd.DataFrame(rows)
    S.to_csv(EXP / "analysis" / "q01-amplitude-slices.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.pivot_table(index=["candidate", "slice"], columns="fold",
                        values=["rmse_blend", "gain"]).round(4).to_string())


if __name__ == "__main__":
    main()
