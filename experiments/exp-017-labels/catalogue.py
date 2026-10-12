"""exp-017 (a-d): catalogue the 2025 training labels that are artefacts.

An L2 tree fitted on a target of 86,400 s spends leaves reaching it and drags
the neighbouring rows with it. The scored population is never touched -- every
validation and ranking row is predicted and scored as always; only what the
model is *fitted on* changes.

Classes, in the order the desk named them:

  R-undetectable  block dated a day early while AOBT_3 looks normal, so nothing
                  observable marks the row -- exp-002 finding 13 measured 2 of
                  these a year and showed no detector exists
  R-detectable    the same defect where AOBT_3 carries the shift too, so the row
                  can be corrected rather than dropped (subtract 86,400)
  negative        taxi at or below zero
  impossible      take-off before off-block by the movement's own columns, or a
                  duplicate (MVT_ID, or the same flight twice at one stand-minute)
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exp-009-noac2"))
from threads import THREADS          # caps BLAS at import; docs/COMPUTE.md compute allocation
import numpy as np, pandas as pd
import prc.data as D

AIR = ("LTFM", "EHAM", "LFPG", "EDDF", "EGLL", "LEMD", "LEBL", "EDDM", "LIRF", "LSZH")
DAY = 86400.0
S = lambda a, b: (a - b).dt.total_seconds()


def load_all() -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "FLIGHT_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt",
            "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt", "LOBT_flt"]
    d = pd.concat([D.load_month(m, columns=cols).assign(month=m) for m in D.MONTHS],
                  ignore_index=True)
    d = d[(d.PHASE_mvt == "DEP") & d.ADEP_mvt.isin(AIR)].copy()
    d["y"] = d.TAXITIME_SEC_mvt.astype(float)
    d["ms"] = S(d.MVT_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
    d["mvt_aobt"] = S(d.MVT_TIME_UTC_mvt, d.AOBT_3_flt)
    d["noac"] = d.AIRCRAFT_TYPE_mvt.isna()
    return d


def classify(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    # a rollover: the target is a day plus an ordinary taxi
    roll = (d.y - DAY).abs() < 3600
    # detectable when the NM off-block ALSO sits about a day before the take-off,
    # i.e. the shift is visible in a column the ranking file carries
    aobt_shift = (d.mvt_aobt - DAY).abs() < 3600
    out["R_detectable"] = roll & aobt_shift
    out["R_undetectable"] = roll & ~aobt_shift
    out["negative"] = d.y <= 0
    out["impossible"] = d.MVT_TIME_UTC_mvt < d.BLOCK_TIME_UTC_mvt
    out["dup_mvt_id"] = d.MVT_ID_mvt.duplicated(keep=False)
    key = (d.ADEP_mvt.astype(str) + "|" + d.STAND_mvt.astype(str) + "|"
           + d.FLIGHT_mvt.astype(str) + "|" + d.MVT_TIME_UTC_mvt.dt.floor("min").astype(str))
    out["dup_flight_stand_minute"] = key.duplicated(keep=False)
    out["any"] = out.any(axis=1)
    return out


if __name__ == "__main__":
    d = load_all()
    c = classify(d)
    print(f"=== {len(d)} training departures, 2025, ten ranking airports ===\n")
    tot = pd.DataFrame({"rows": c.sum(), "share_%": (c.mean() * 100).round(4)})
    print(tot.to_string())
    print(f"\nrows in ANY class: {int(c['any'].sum())} ({c['any'].mean()*100:.4f} %)")
    print("\n=== squared-error weight these rows carry in an L2 fit ===")
    med = float(d.y.median())
    sse = ((d.y - med) ** 2)
    for k in c.columns.drop("any"):
        if c[k].sum():
            print(f"  {k:26s} n={int(c[k].sum()):5d}  share of total SSE about the median: "
                  f"{sse[c[k]].sum()/sse.sum()*100:7.3f} %")
    print("\n=== per month ===")
    print(pd.crosstab(d.month, c['any']).to_string())
    print("\n=== the rollover rows in detail ===")
    r = d[c.R_detectable | c.R_undetectable].copy()
    r["detectable"] = c.R_detectable[r.index]
    print(r[["month", "ADEP_mvt", "FLIGHT_mvt", "y", "ms", "mvt_aobt", "noac", "detectable"]]
          .sort_values("y", ascending=False).round(0).to_string(index=False))
    d.assign(**{f"cls_{k}": c[k] for k in c.columns}).to_parquet(
        D.ROOT / "artifacts" / "exp-017-labels-catalogue.parquet")
    print(f"\nwrote artifacts/exp-017-labels-catalogue.parquet")
