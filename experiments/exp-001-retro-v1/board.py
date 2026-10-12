"""The board estimate: blend the two leave-one-out folds by the ranking's month shares.

The submission trains on all twelve 2025 months, so July 2026 is predicted with
July and August 2025 in training and January 2026 with January and February. The
forward `july` fold has no same-season data at all and is therefore pessimistic
for the ranking month; `july_loo` (train the other eleven months, validate July
2025) is the matched analogue, and `january` already is one by construction.

So the pair (july_loo, january) is the fold estimate of the board score, weighted
by how many ranking departures fall in each month. `december` stays useful as a
third reading but is not part of this number: no ranking month is December.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import prc.data as D  # noqa: E402

PAIR = {"july_loo": "2026-07", "january": "2026-01"}


def shares() -> dict:
    r = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    r = r[r.PHASE_mvt == "DEP"]
    n = r.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").value_counts()
    return (n / n.sum()).to_dict()


def estimate(name: str, w: dict | None = None) -> dict:
    """Weighted-RMSE board estimate for a run, from its july_loo and january folds."""
    w = w or shares()
    out, sse, tot = {}, 0.0, 0.0
    for fold, month in PAIR.items():
        p = HERE / "oof" / f"{name}-{fold}.parquet"
        if not p.exists():
            return {"run": name, "board_estimate": float("nan"), "missing": fold}
        o = pd.read_parquet(p)
        rmse = float(np.sqrt(((o.y_true - o.y_pred) ** 2).mean()))
        out[fold] = round(rmse, 2)
        sse += w[month] * rmse ** 2
        tot += w[month]
    out["run"] = name
    out["board_estimate"] = round(float(np.sqrt(sse / tot)), 2)
    return out


if __name__ == "__main__":
    w = shares()
    print("ranking month shares:", {k: round(v, 4) for k, v in w.items()})
    rows = [estimate(n, w) for n in sys.argv[1:]]
    print(pd.DataFrame(rows).to_string(index=False))
