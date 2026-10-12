"""`COMMON.md`'s four-way RMSE table for every arm this experiment saved.

*"Every RESULTS table must show RMSE overall, RMSE on rows with target at or
below 3600 s, RMSE on rows above, and RMSE on no-NM rows, for each fold."*
The arms are read from the committed out-of-fold parquets, so this file scores
the same vectors the record carries and refits nothing.

    python slices.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import base as B  # noqa: E402

EXP = B.EXP
FOLDS = B.FOLDS
D = B.D
#: EVERY measured arm, not a selection (Sol round 1, blocking finding 10): each
#: screen now writes an out-of-fold parquet per arm and this file scores all of
#: them on the four slices the contract names.
def arms() -> tuple[str, ...]:
    """EVERY measured arm, not a selection (Sol round 1, blocking finding 10):
    each screen writes an out-of-fold parquet per arm and this file scores all of
    them on the four slices the contract names."""
    return tuple(sorted(q.stem[len("exp-041-"):]
                        for q in (EXP / "oof").glob("exp-041-*.parquet")))


def main() -> None:
    ARMS = arms()
    pan = B.V.load()
    out = {"note": "RMSE overall, on y <= 3600, on y > 3600 and on no-NM rows, "
                   "per fold; v10 first, then each saved arm, on the same rows",
           "folds": {}}
    for f in FOLDS:
        m = pan[f]
        y = m["y_true"].to_numpy("float64")
        ids = m["MVT_ID_mvt"].to_numpy("float64")
        nonm = m["nm_missing"].to_numpy().astype(bool) | m["noac"].to_numpy().astype(bool)
        sl = {"all": np.ones(len(y), bool), "le3600": y <= 3600.0,
              "gt3600": y > 3600.0, "no_nm": nonm}
        rec = {"counts": {k: int(v.sum()) for k, v in sl.items()}}
        vecs = {"v10": m["p10"].to_numpy("float64")}
        for a in ARMS:
            p = EXP / "oof" / f"exp-041-{a}.parquet"
            t = pd.read_parquet(p)
            t = t[t["fold"] == f]
            t = t.set_index(t["MVT_ID_mvt"].to_numpy("float64")).reindex(ids)
            if not np.allclose(t["y_true"].to_numpy("float64"), y):
                raise SystemExit(f"{a}/{f}: the saved labels are not the panel's")
            vecs[a] = t["y_pred"].to_numpy("float64")
        for name, p in vecs.items():
            rec[name] = {k: round(float(np.sqrt(((y[v] - p[v]) ** 2).mean())), 4)
                         if v.sum() else None for k, v in sl.items()}
        out["folds"][f] = rec
    p = EXP / "analysis" / "slices.json"
    p.write_text(json.dumps(out, indent=2))
    for f in FOLDS:
        r = out["folds"][f]
        print(f"[{f}] n {r['counts']}")
        for name in ["v10"] + list(ARMS):
            print(f"    {name:30s} all {r[name]['all']:9.4f}  <=3600 "
                  f"{r[name]['le3600']:8.4f}  >3600 {r[name]['gt3600']:9.4f}  "
                  f"no-NM {r[name]['no_nm']:9.4f}")
    print(f"  written: {p}")


if __name__ == "__main__":
    main()
