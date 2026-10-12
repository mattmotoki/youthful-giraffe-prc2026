"""Board estimate on all rows, and CLEAN, for every candidate that has the folds.

CLEAN excludes the rows the census marks **unreached**: block time on an earlier
calendar day, an abnormal label, and every timing input ordinary. Four such rows
exist in the twelve 2025 months and **two of them are in the january fold**;
`july_loo` and `december` contain none, so the clean correction acts on the
january half of the board estimate alone.

This is not a cosmetic adjustment. `exp-007 rollover` measured those two rows at
55.3 % of the whole january fold's squared error. A number dominated by two rows
of a class whose Poisson mean in the ranking file is well under one is a number
about 2025's accidents, not about our skill on 2026.

No fits. Reads committed OOF only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

OUT = EXP / "analysis"
E16 = ROOT / "experiments" / "exp-016-stack" / "oof"
FOLDS = ("july_loo", "january")          # the two that make the board estimate
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}

#: (label, how to get y_true and y_pred for a fold). Every entry reads committed
#: OOF; nothing is refitted.
CANDIDATES = {
    "stack-1 (exp-016, block member)":  ("e16", "exp-016-oof-{fold}.parquet", "y_pred"),
    "stack-2 (exp-016, block member)":  ("e16", "exp-016-oof-{fold}.parquet", "y_pred_a2_keep_noac"),
    "blend-2 rule (block member)":      ("e16", "exp-016-oof-{fold}.parquet", "y_pred_blend2"),
    "flat4 (block member)":             ("e16", "exp-016-oof-{fold}.parquet", "y_pred_flat4"),
    "stack-3 (exp-016, deice member)":  ("e16", "exp-016-oof-blend3-{fold}.parquet", "y_pred"),
    "blend-3 rule (deice member)":      ("e16", "exp-016-oof-blend3-{fold}.parquet", "y_pred_blend2"),
    "flat4 (deice member)":             ("e16", "exp-016-oof-blend3-{fold}.parquet", "y_pred_flat4"),
    "T-tail v3 (best single model)":    ("tail", "exp-009-noac2-oof-v3.parquet", "y_pred"),
    "T-tail v2 (single model)":         ("tail", "exp-009-noac2-oof-v2.parquet", "y_pred"),
}


def load(kind: str, tmpl: str, col: str, fold: str) -> pd.DataFrame:
    if kind == "e16":
        d = pd.read_parquet(E16 / tmpl.format(fold=fold))
    else:
        d = pd.read_parquet(ROOT / "experiments" / "exp-009-noac2" / "oof" / tmpl)
        d = d[d["fold"] == fold]
    return pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy("float64"),
                         "y_true": d["y_true"].to_numpy("float64"),
                         "y_pred": d[col].to_numpy("float64")})


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, "float64") - np.asarray(p, "float64")) ** 2)))


def board(per_fold: dict) -> float:
    return float(np.sqrt(sum(SHARES[f] * per_fold[f] ** 2 for f in SHARES)))


def main() -> None:
    cen = pd.read_csv(OUT / "rollover_census.csv")
    un = cen[~cen["reachable"].astype(bool)]
    excluded = {m: set(un[un["month"] == m]["MVT_ID_mvt"].astype("float64"))
                for m in un["month"].unique()}
    fold_month = {f: D.FOLDS[f].valid_month for f in FOLDS}
    print("unreached rows per fold validation month:")
    for f in FOLDS:
        ids = excluded.get(fold_month[f], set())
        print(f"  {f:9s} ({fold_month[f]}): {len(ids)} -> {sorted(int(i) for i in ids)}")

    rows = []
    for name, (kind, tmpl, col) in CANDIDATES.items():
        rec = {"candidate": name}
        ok = True
        per_all, per_clean = {}, {}
        for f in FOLDS:
            try:
                d = load(kind, tmpl, col, f)
            except (FileNotFoundError, KeyError):
                ok = False
                break
            ids = excluded.get(fold_month[f], set())
            keep = ~d["MVT_ID_mvt"].isin(ids).to_numpy()
            per_all[f] = rmse(d["y_true"], d["y_pred"])
            per_clean[f] = rmse(d.loc[keep, "y_true"], d.loc[keep, "y_pred"])
            rec[f"{f}_all"] = per_all[f]
            rec[f"{f}_clean"] = per_clean[f]
            rec[f"{f}_n_excluded"] = int((~keep).sum())
        if not ok:
            continue
        rec["board_all"] = board(per_all)
        rec["board_clean"] = board(per_clean)
        rec["delta"] = rec["board_clean"] - rec["board_all"]
        rows.append(rec)

    T = pd.DataFrame(rows).sort_values("board_clean")
    T.to_csv(OUT / "clean_board.csv", index=False)
    print("\n=== board estimate, all rows and CLEAN (0.5571 july_loo / 0.4429 january) ===")
    cols = ["candidate", "july_loo_all", "july_loo_clean", "january_all", "january_clean",
            "board_all", "board_clean", "delta"]
    print(T[cols].round(3).to_string(index=False))

    print("\n=== what the two january rows are worth ===")
    for r in T.itertuples():
        print(f"  {r.candidate:34s} january {r.january_all:7.2f} -> {r.january_clean:7.2f} "
              f"({r.january_all - r.january_clean:6.2f} s)   board {r.board_all:7.2f} -> "
              f"{r.board_clean:7.2f}")

    # the ordering question: does cleaning change who is best?
    a = T.sort_values("board_all")["candidate"].tolist()
    b = T.sort_values("board_clean")["candidate"].tolist()
    print(f"\nordering by board_all   : {a[0]}  then  {a[1]}")
    print(f"ordering by board_clean : {b[0]}  then  {b[1]}")
    print(f"same leader: {a[0] == b[0]};  identical ordering: {a == b}")


if __name__ == "__main__":
    main()
