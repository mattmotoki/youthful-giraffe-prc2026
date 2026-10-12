"""Cross-checks against another experiment's independently built files.

`frames.py` builds its own noac frames and its own labels.  Both must agree with
`exp-033-noac`, which built the same population and extracted the same labels
through a different code path (`exp-033-noac/code/build.py` and
`join_labels.py`).  Two things are checked and both are exact:

  the population   the `MVT_ID_mvt` sets, 1,507 and 388
  the labels       `y_true` and `ph` per row

The label check also stands in for the row-order assertion inside `frames.py`:
`_one` resets the index, so a misaligned label join would move every label, and
this comparison would not survive it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REF = EXP.parents[0] / "exp-033-noac" / "work"


def main() -> None:
    rows = []
    for f, n in (("noac_2025", 1507), ("noac_2026", 388)):
        mine = pd.read_parquet(EXP / "work" / f"{f}.parquet")
        ref = pd.read_parquet(REF / f"{f}.parquet")
        same = set(mine.MVT_ID_mvt) == set(ref.MVT_ID_mvt)
        rows.append({"check": f"{f} population", "n_mine": len(mine), "n_ref": len(ref),
                     "n_expected": n, "result": bool(same and len(mine) == n)})
        assert same and len(mine) == n, f

    mine = pd.read_parquet(EXP / "work" / "noac_2025.parquet",
                           columns=["MVT_ID_mvt", "y_true", "ph"])
    ref = pd.read_parquet(REF / "labels_2025.parquet")[["MVT_ID_mvt", "y_true", "ph"]]
    j = mine.merge(ref.rename(columns={"y_true": "y_ref", "ph": "ph_ref"}),
                   on="MVT_ID_mvt", how="left", validate="one_to_one")
    dy = float(np.abs(j.y_true - j.y_ref).max())
    dph = int((j.ph != j.ph_ref).sum())
    rows.append({"check": "2025 labels against exp-033's own extraction",
                 "n_mine": len(j), "n_ref": len(ref), "n_expected": 1507,
                 "max_abs_y_diff": dy, "ph_disagreements": dph,
                 "result": bool(dy == 0.0 and dph == 0)})
    assert dy == 0.0 and dph == 0

    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "checks.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
