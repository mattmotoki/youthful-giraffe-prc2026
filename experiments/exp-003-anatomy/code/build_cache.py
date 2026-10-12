"""Cache a slim all-2025 + ranking movement frame with derived offsets.

Writes artifacts/exp-003-anatomy/mvt_all.parquet (all phases, all months incl.
the two 2026 ranking months) so later steps do not re-read 13 parquet files.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import prc.data as d

OUT = d.ROOT / "artifacts" / "exp-003-anatomy"
OUT.mkdir(parents=True, exist_ok=True)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    s = lambda a, b: (df[a] - df[b]).dt.total_seconds()
    df["nm_taxi"] = s("MVT_TIME_UTC_mvt", "AOBT_3_flt")
    df["blk_aobt"] = s("BLOCK_TIME_UTC_mvt", "AOBT_3_flt")
    df["blk_lobt"] = s("BLOCK_TIME_UTC_mvt", "LOBT_flt")
    df["blk_iobt"] = s("BLOCK_TIME_UTC_mvt", "IOBT_flt")
    df["blk_eobt1"] = s("BLOCK_TIME_UTC_mvt", "EOBT_1_flt")
    df["blk_sched"] = s("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
    df["aobt_lobt"] = s("AOBT_3_flt", "LOBT_flt")
    df["aobt_sched"] = s("AOBT_3_flt", "SCHED_TIME_UTC_mvt")
    df["lobt_iobt"] = s("LOBT_flt", "IOBT_flt")
    df["lobt_eobt1"] = s("LOBT_flt", "EOBT_1_flt")
    df["lobt_sched"] = s("LOBT_flt", "SCHED_TIME_UTC_mvt")
    df["mvt_sched"] = s("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
    t = df["MVT_TIME_UTC_mvt"]
    df["hour"] = t.dt.hour.astype("int8")
    df["dow"] = t.dt.dayofweek.astype("int8")
    df["doy"] = t.dt.dayofyear.astype("int16")
    df["date"] = t.dt.floor("D")
    df["has_nm"] = df["AOBT_3_flt"].notna()
    return df


def main():
    parts = []
    for m in d.MONTHS:
        df = d.load_month(m)
        parts.append(derive(df))
        print(m, len(df), flush=True)
    rk = d.load_ranking()
    rk["is_ranking"] = True
    parts.append(derive(rk))
    print("ranking", len(rk), flush=True)
    allm = pd.concat(parts, ignore_index=True)
    allm["is_ranking"] = allm["is_ranking"].fillna(False).astype(bool)
    for c in ("ADEP_mvt", "ADES_mvt", "PHASE_mvt", "STAND_mvt", "RUNWAY_mvt",
              "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
              "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt",
              "FLIGHT_RULE_flt", "month"):
        allm[c] = allm[c].astype("category")
    p = OUT / "mvt_all.parquet"
    allm.to_parquet(p, index=False)
    print(p, len(allm), d.sha256(p))


if __name__ == "__main__":
    main()
