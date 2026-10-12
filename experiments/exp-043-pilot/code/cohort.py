"""exp-043-pilot: the week's EHAM departures, from the July 2025 training file.

The truth side of the pilot.  Nothing here is masked: this is a TRAINING week, so
`BLOCK_TIME_UTC_mvt` is present and is the thing the surface observation has to
reconstruct.  Writes `analysis/cohort.json` (the census the report quotes) and
`work/cohort.parquet` (the rows the matcher joins).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
DAYS = pd.date_range("2025-07-07", "2025-07-13", tz="UTC")
FLT = ["LOBT_flt", "CALLSIGN_flt", "ADEP_flt", "IOBT_flt", "AIRCRAFT_TYPE_flt",
       "AIRCRAFT_OPERATOR_flt", "EOBT_1_flt", "AOBT_3_flt"]


def main() -> None:
    d = pd.read_parquet(ROOT / "data/raw/training_2025-07-01_2025-08-01.parquet")
    dep = d[(d["PHASE_mvt"] == "DEP") & (d["ADEP_mvt"] == "EHAM")].copy()
    # the day of a departure is the day of its OFF-BLOCK, which is the event the
    # surface trace has to see; a trace file is keyed by the UTC day it covers.
    dep["day"] = dep["BLOCK_TIME_UTC_mvt"].dt.floor("D")
    wk = dep[dep["day"].isin(DAYS)].copy()
    wk["has_nm"] = wk["AOBT_3_flt"].notna()
    wk["y"] = wk["TAXITIME_SEC_mvt"].astype("float64")
    wk["b_minus_aobt"] = (wk["BLOCK_TIME_UTC_mvt"] - wk["AOBT_3_flt"]).dt.total_seconds()

    (EXP / "work").mkdir(exist_ok=True)
    keep = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "day",
            "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
            "has_nm", "y", "b_minus_aobt", *FLT]
    wk[keep].to_parquet(EXP / "work/cohort.parquet", index=False)

    nm = wk[wk["has_nm"]]
    res = {
        "week": ["2025-07-07", "2025-07-13"], "airport": "EHAM",
        "departures": int(len(wk)),
        "by_day": {str(k.date()): int(v) for k, v in wk.groupby("day").size().items()},
        "no_nm": int((~wk["has_nm"]).sum()),
        "callsign_present": int(wk["CALLSIGN_flt"].notna().sum()),
        "callsign_present_share": round(float(wk["CALLSIGN_flt"].notna().mean()), 4),
        "target": {"median": float(wk["y"].median()), "mean": float(wk["y"].mean()),
                   "std": float(wk["y"].std()),
                   "p01": float(wk["y"].quantile(.01)), "p99": float(wk["y"].quantile(.99))},
        "incumbent_b_minus_aobt": {
            "rows": int(len(nm)), "median": float(nm["b_minus_aobt"].median()),
            "rmse_about_median": float(np.sqrt(
                ((nm["b_minus_aobt"] - nm["b_minus_aobt"].median()) ** 2).mean())),
            "p01": float(nm["b_minus_aobt"].quantile(.01)),
            "p99": float(nm["b_minus_aobt"].quantile(.99))},
        "distinct_callsigns": int(wk["CALLSIGN_flt"].nunique()),
        "join_key_collisions": int(
            (wk.dropna(subset=["CALLSIGN_flt"])
               .groupby(["day", "CALLSIGN_flt"])["MVT_ID_mvt"].transform("size") > 1).sum()),
    }
    (EXP / "analysis/cohort.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
