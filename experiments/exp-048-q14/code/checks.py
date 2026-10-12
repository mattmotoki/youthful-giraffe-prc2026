"""The invariants the reading rests on, asserted rather than asserted about.

Run last.  Every line either prints a measured invariant or raises.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "code"))
from arms import FIT_MONTHS, READ_MONTHS, CAPACITY  # noqa: E402


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "arms_2025.parquet")
    s = pd.read_parquet(EXP / "work" / "arms_388.parquet")
    cfg = json.loads((EXP / "analysis" / "arms.json").read_text())

    assert set(FIT_MONTHS) | set(READ_MONTHS) == set(a.month.unique())
    assert not set(FIT_MONTHS) & set(READ_MONTHS)
    ft, rd = a[a.month.isin(FIT_MONTHS)], a[a.month.isin(READ_MONTHS)]
    assert not set(ft.MVT_ID_mvt) & set(rd.MVT_ID_mvt)
    print(f"fit rows {len(ft)} / reading rows {len(rd)} / total {len(a)}; no row in both")

    d = rd.day.dt.strftime("%Y-%m-%d")
    assert (rd.groupby(d).month.nunique() == 1).all()
    assert (rd.day.dt.strftime("%Y-%m") == rd.month).all()
    print(f"reading days {d.nunique()}, each inside exactly one month; "
          f"rows per day min {d.value_counts().min()} median "
          f"{d.value_counts().median():.0f} max {d.value_counts().max()}")

    for k in ("P", "W"):
        m = float(np.sqrt(np.mean(s[f"d_{k}"] ** 2)))
        assert abs(m - CAPACITY) < 0.5, (k, m)
        assert m < 500.0
        print(f"arm {k}: 388 shift {m:.2f} s RMS, budget 500 s, "
              f"moved rows {(s[f'd_{k}'] != 0).sum()} of 388")

    assert (a.base >= 60.0 - 1e-9).all()
    for k in ("P", "W"):
        assert ((a.base + a[f"d_{k}"]) >= 60.0 - 1e-9).all()
        assert ((s.v7 + s[f"d_{k}"]) >= 60.0 - 1e-9).all()
    print("every base and every served candidate value is at or above the 60 s floor")

    for k in ("P", "W"):
        al = cfg["arms"][k]["alpha"]
        assert abs(sum(al) - 1.0) < 1e-9 and min(al) >= 0.0
    L = cfg["arms"]["W"]["L_fit_per_month"]
    spread = max(L.values()) - min(L.values())
    assert spread < 0.005 * abs(np.mean(list(L.values()))), spread
    print(f"the minimax equalises the three fit months to within {spread:.0f} s^2 "
          f"of each other (mean {np.mean(list(L.values())):,.0f})")
    assert cfg["repair_triggered"] is False
    print("registered repair trigger evaluated and NOT fired "
          f"(alpha* = {[round(x, 3) for x in cfg['arms']['W']['alpha']]})")


if __name__ == "__main__":
    main()
