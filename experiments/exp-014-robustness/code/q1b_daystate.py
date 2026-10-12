"""Q1b: the day-state block on the ranking months -- candidate 2's added exposure.

`D1-winter` carries 0.25 of candidate 2 outside `nm_late` and 0.240 inside it,
and `winter_arm.py` builds it with `ds_columns="all"`: the full 25-column block,
including the nine unbounded `_z` columns. exp-007 finding 13 measured that
block at **+72.1 s on january's disrupted days** and named the mechanism -- a
`_z` column beyond its own training maximum pushes a near-calibrated model past
the truth. This script asks how much of that mechanism January 2026 presents.

Per airport, because the `_z` columns are standardised inside the airport
(`DayState.fit` takes the median and spread per `apt`), which is the range exp-007
compared against.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames14 as F  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.data as D  # noqa: E402
from prc.day_state_features import COLUMNS, DayState, Z_STATS  # noqa: E402

OUT = F.EXP / "analysis"
CACHE = F.ROOT / "artifacts" / "exp-014" / "frames"


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    ptr, prk = CACHE / "ds-train.parquet", CACHE / "ds-ranking.parquet"
    if ptr.exists() and prk.exists():
        return pd.read_parquet(ptr), pd.read_parquet(prk)
    raw = {m: D.load_month(m) for m in D.MONTHS}
    ds = DayState().fit(pd.concat(list(raw.values()), ignore_index=True))

    def one(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
        f = ds.transform(frame)
        dep = frame[frame["PHASE_mvt"].eq("DEP")]
        out = f.reindex(dep.index)[list(COLUMNS)].copy()
        out.insert(0, "MVT_ID_mvt", dep["MVT_ID_mvt"].to_numpy())
        out.insert(1, "apt", dep["ADEP_mvt"].astype(str).to_numpy())
        out.insert(2, "day", dep["MVT_TIME_UTC_mvt"].dt.floor("D").to_numpy())
        out.insert(3, "month_str", tag)
        return out.reset_index(drop=True)

    tr = pd.concat([one(raw[m], m) for m in D.MONTHS], ignore_index=True)
    rk = D.load_ranking()
    rkf = pd.concat([one(rk[rk["month"] == m], m) for m in sorted(rk["month"].unique())],
                    ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(ptr, index=False)
    rkf.to_parquet(prk, index=False)
    return tr, rkf


def main() -> None:
    tr, rk = build()
    zcols = [f"{c}_z" for c in Z_STATS]
    print(f"day-state: train {tr.shape} ranking {rk.shape}, {len(zcols)} `_z` columns",
          flush=True)

    # per-airport training range, which is the range exp-007's mechanism is about
    lim = tr.groupby("apt", observed=True)[zcols].agg(["min", "max"])
    rows, per_row_over = [], np.zeros(len(rk), dtype=int)
    for c in zcols:
        lo = rk["apt"].map(lim[(c, "min")]).to_numpy("float64")
        hi = rk["apt"].map(lim[(c, "max")]).to_numpy("float64")
        v = rk[c].to_numpy("float64")
        over = np.isfinite(v) & (v > hi)
        under = np.isfinite(v) & (v < lo)
        out = over | under
        per_row_over += over.astype(int)
        for m in ("2026-01", "2026-07"):
            s = (rk["month_str"] == m).to_numpy()
            n = int(s.sum())
            rows.append(dict(column=c, month=m, n=n, n_over_max=int((over & s).sum()),
                             n_under_min=int((under & s).sum()),
                             share_out=float((out & s).sum()) / n,
                             max_exceedance=float(np.nanmax((v - hi)[over & s])) if (over & s).any() else 0.0,
                             worst_ratio=float(np.nanmax((v / np.where(hi > 0, hi, np.nan))[over & s]))
                             if (over & s).any() else np.nan))
    Z = pd.DataFrame(rows)
    Z.to_csv(OUT / "q1b_daystate_z_exposure.csv", index=False)
    print("\n=== `_z` columns past their airport's TRAINING range on the ranking months ===")
    print(Z[Z.n_over_max + Z.n_under_min > 0]
          .sort_values("share_out", ascending=False).round(3).to_string(index=False), flush=True)

    rk = rk.assign(n_z_over=per_row_over)
    rk["exposed"] = rk["n_z_over"] > 0
    by = rk.groupby(["month_str", "apt"], observed=True).agg(
        n=("exposed", "size"), n_exposed=("exposed", "sum"),
        max_z_over=("n_z_over", "max")).reset_index()
    by["share"] = by["n_exposed"] / by["n"]
    by.to_csv(OUT / "q1b_daystate_by_airport.csv", index=False)
    print("\n=== rows with at least one `_z` column above its airport's training maximum ===")
    print(by[by.n_exposed > 0].sort_values("n_exposed", ascending=False).round(3)
          .to_string(index=False), flush=True)

    # exp-007's own counting rule: how many of the nine, per exposed airport-day
    day = rk[rk["exposed"]].groupby(["month_str", "apt", "day"], observed=True).agg(
        n=("n_z_over", "size"), z_over=("n_z_over", "max")).reset_index()
    day = day.sort_values(["z_over", "n"], ascending=False)
    day.to_csv(OUT / "q1b_daystate_exposed_days.csv", index=False)
    print(f"\n=== {len(day)} exposed airport-days, exp-007's count of nine ===")
    print(day.head(30).to_string(index=False), flush=True)

    # what candidates 1 and 2 predict on the exposed rows
    c1 = F.candidate("submissions/candidates/exp-012-blend-1.parquet")
    c2 = F.candidate("submissions/candidates/exp-012-blend-2.parquet")
    ids = rk["MVT_ID_mvt"].to_numpy("float64")
    rk["c1"] = c1.reindex(ids).to_numpy("float64")
    rk["c2"] = c2.reindex(ids).to_numpy("float64")
    rk["d21"] = rk["c2"] - rk["c1"]
    cmp_rows = []
    for m in ("2026-01", "2026-07"):
        s = rk["month_str"] == m
        for label, sub in ((f"{m} all", rk[s]),
                           (f"{m} exposed rows", rk[s & rk["exposed"]]),
                           (f"{m} rows with 3+ `_z` over", rk[s & (rk["n_z_over"] >= 3)])):
            if not len(sub):
                continue
            cmp_rows.append(dict(slice=label, n=len(sub), c1_mean=float(sub["c1"].mean()),
                                 c2_mean=float(sub["c2"].mean()),
                                 d21_mean=float(sub["d21"].mean()),
                                 d21_p90=float(sub["d21"].quantile(0.9)),
                                 d21_max=float(sub["d21"].max()),
                                 d21_min=float(sub["d21"].min())))
    C = pd.DataFrame(cmp_rows)
    C.to_csv(OUT / "q1b_candidate2_vs_1_on_exposure.csv", index=False)
    print("\n=== candidate 2 minus candidate 1, on the day-state exposure ===")
    print(C.round(1).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
