"""What the error on the five envelope days actually is, by regime.

Two taxonomies, because they answer different questions: `prc.blend`'s
four-regime label is what the served stack switches on, and exp-007's three-way
split (`NM bulk` / `NM tail` / `no-NM`) is the one the per-day tables in exp-007
and exp-014 are written in.
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
sys.path.insert(0, str(ROOT))
import base_moved as BM  # noqa: E402
import envelope as E  # noqa: E402

BASES = {"candidate 1": "cand1", "blend-5": "y_pred_anchor", "stack-5": "y_pred"}


def main() -> None:
    R = pd.read_parquet(EXP / "analysis" / "fold_rows.parquet")
    c1 = BM.candidate1()
    R = R.merge(pd.concat([c1[f] for f in E.FOLDS], ignore_index=True),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
    g = R[R["flagged"].to_numpy()]
    rows = []
    for base, col in BASES.items():
        for key, lab in (("breg", "prc.blend"), ("regime", "exp-007")):
            tot = float((((g[col] - g["y_true"]) ** 2)).sum())
            for reg, s in g.groupby(key, observed=True):
                e = s[col].to_numpy("float64") - s["y_true"].to_numpy("float64")
                rows.append(dict(base=base, taxonomy=lab, regime=reg, n=len(s),
                                 row_share=len(s) / len(g),
                                 bias=float(e.mean()),
                                 rmse=float(np.sqrt((e ** 2).mean())),
                                 sse_share=float((e ** 2).sum() / tot)))
    T = pd.DataFrame(rows)
    T.to_csv(EXP / "analysis" / "regime_split.csv", index=False)
    for lab in ("prc.blend", "exp-007"):
        print(f"\n=== the five envelope days by {lab} regime ===")
        print(T[T["taxonomy"] == lab].round(3).to_string(index=False))

    print("\n=== per day, stack-5, exp-007's split ===")
    rows = []
    for (f, a, d), s in g.groupby(["fold", "apt", "day"], observed=True):
        rec = {"fold": f, "airport": a, "day": str(pd.Timestamp(d).date())}
        for reg, q in s.groupby("regime", observed=True):
            e = q["y_pred"].to_numpy("float64") - q["y_true"].to_numpy("float64")
            rec[f"{reg} n"] = len(q)
            rec[f"{reg} bias"] = round(float(e.mean()), 1)
        rows.append(rec)
    P = pd.DataFrame(rows)
    P.to_csv(EXP / "analysis" / "regime_by_day.csv", index=False)
    print(P.to_string(index=False))


if __name__ == "__main__":
    main()
