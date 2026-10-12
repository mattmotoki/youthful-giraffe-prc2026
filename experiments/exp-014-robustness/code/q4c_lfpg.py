"""Q4c: LFPG's retired runways -- is the candidate's stand-to-runway response sane?

CDG lost 27R (9.0 % of its 2025 departures) and 09L (4.8 %); 27L absorbed them
(11.2 -> 29.5 % in January 2026, 20.4 % in July). Every level is still seen, so
the categorical audit finds nothing -- the exposure is in the PAIRS: a stand that
never departed from 27L in 2025 does so in 2026, and stand-to-runway distance is
the physics of this task.

Two label-free checks, both on the shipped candidates:

1. Coverage. How many 2026 LFPG rows carry a (stand, runway) pair that 2025
   never contains, and what does the model predict there against the same
   stand's 2025 median and the same runway's 2025 median?
2. Monotone sanity. Rank LFPG's runways by their 2025 median taxi time. For each
   stand with at least two runways in 2026, does the candidate's per-runway mean
   prediction preserve that ranking? A model using runway distance coherently
   keeps the order; one leaning on the stand alone produces a flat or scrambled
   response.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames14 as F  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.data as D  # noqa: E402

OUT = F.EXP / "analysis"
APT = "LFPG"


def main() -> None:
    tr = D.load_months(list(D.MONTHS),
                       columns=["PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt",
                                "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
    tr = tr[(tr["PHASE_mvt"] == "DEP") & (tr["ADEP_mvt"].astype(str) == APT)
            & tr["TAXITIME_SEC_mvt"].notna()].copy()
    tr["stand"] = tr["STAND_mvt"].astype(str)
    tr["rwy"] = tr["RUNWAY_mvt"].astype(str)
    rk = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt",
                                 "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
    rk = rk[(rk["PHASE_mvt"] == "DEP") & (rk["ADEP_mvt"].astype(str) == APT)].copy()
    rk["stand"] = rk["STAND_mvt"].astype(str)
    rk["rwy"] = rk["RUNWAY_mvt"].astype(str)
    rk["month"] = rk["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    for name, path in (("c1", "submissions/candidates/exp-012-blend-1.parquet"),
                       ("c2", "submissions/candidates/exp-012-blend-2.parquet")):
        rk[name] = F.candidate(path).reindex(rk["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
    rk["obs"] = (rk["MVT_TIME_UTC_mvt"] - rk["AOBT_3_flt"]).dt.total_seconds()

    share = pd.concat([
        (tr["rwy"].value_counts(normalize=True) * 100).rename("2025"),
        (rk[rk.month == "2026-01"]["rwy"].value_counts(normalize=True) * 100).rename("2026-01"),
        (rk[rk.month == "2026-07"]["rwy"].value_counts(normalize=True) * 100).rename("2026-07"),
        tr.groupby("rwy")["TAXITIME_SEC_mvt"].median().rename("median_taxi_2025"),
        tr.groupby("rwy").size().rename("n_2025"),
    ], axis=1).fillna(0).sort_values("2025", ascending=False)
    share.to_csv(OUT / "q4c_lfpg_runway_share.csv")
    print(f"=== {APT} runway share (% of the airport's departures) and 2025 median taxi ===")
    print(share.round(1).to_string(), flush=True)

    # ---- 1. unseen (stand, runway) pairs ---------------------------------
    seen = set(zip(tr["stand"], tr["rwy"]))
    rk["pair_unseen"] = [(s, r) not in seen for s, r in zip(rk["stand"], rk["rwy"])]
    st_med = tr.groupby("stand")["TAXITIME_SEC_mvt"].median()
    rw_med = tr.groupby("rwy")["TAXITIME_SEC_mvt"].median()
    rk["stand_med25"] = rk["stand"].map(st_med)
    rk["rwy_med25"] = rk["rwy"].map(rw_med)
    rows = []
    for m in ("2026-01", "2026-07"):
        s = rk["month"] == m
        for label, sub in ((f"{m} all {APT}", rk[s]),
                           (f"{m} unseen (stand, runway) pair", rk[s & rk["pair_unseen"]]),
                           (f"{m} seen pair", rk[s & ~rk["pair_unseen"]])):
            if not len(sub):
                continue
            rows.append(dict(slice=label, n=len(sub),
                             c1=float(sub["c1"].mean()), c2=float(sub["c2"].mean()),
                             obs=float(sub["obs"].mean()),
                             stand_med25=float(sub["stand_med25"].mean()),
                             rwy_med25=float(sub["rwy_med25"].mean()),
                             c1_minus_stand=float((sub["c1"] - sub["stand_med25"]).mean())))
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "q4c_lfpg_pairs.csv", index=False)
    print(f"\n=== {APT}: (stand, runway) pairs 2025 never contains ===")
    print(P.round(1).to_string(index=False), flush=True)
    top = (rk[rk["pair_unseen"]].groupby(["month", "stand", "rwy"], observed=True)
           .agg(n=("c1", "size"), c1=("c1", "mean"), obs=("obs", "mean")).reset_index()
           .sort_values("n", ascending=False))
    top.to_csv(OUT / "q4c_lfpg_unseen_pairs.csv", index=False)
    print(top.head(12).round(1).to_string(index=False), flush=True)

    # ---- 2. monotone sanity ------------------------------------------------
    order25 = rw_med.dropna().sort_values()
    keep = [r for r in order25.index if int(tr["rwy"].value_counts().get(r, 0)) >= 200]
    order25 = order25.loc[keep]
    res = []
    for m in ("2026-01", "2026-07"):
        s = rk[(rk["month"] == m) & rk["rwy"].isin(order25.index)]
        for stand, g in s.groupby("stand", observed=True):
            per = g.groupby("rwy", observed=True)["c1"].agg(["size", "mean"])
            per = per[per["size"] >= 20]
            if len(per) < 3:
                continue
            rho, _ = spearmanr(order25.loc[per.index].to_numpy(), per["mean"].to_numpy())
            res.append(dict(month=m, stand=stand, n_runways=len(per), n=int(per["size"].sum()),
                            rho=float(rho),
                            spread=float(per["mean"].max() - per["mean"].min())))
    M = pd.DataFrame(res)
    M.to_csv(OUT / "q4c_lfpg_monotone.csv", index=False)
    print(f"\n=== monotone sanity: does the candidate keep 2025's runway ordering per stand? ===")
    print(f"{len(M)} (month, stand) cells with 3+ runways and 20+ rows each")
    if len(M):
        print(M.groupby("month").agg(cells=("rho", "size"), rho_mean=("rho", "mean"),
                                     rho_median=("rho", "median"),
                                     n_negative=("rho", lambda s: int((s < 0).sum())),
                                     spread_mean=("spread", "mean")).round(2).to_string())
        print("\nworst cells:")
        print(M.sort_values("rho").head(10).round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
