"""Per-month feature frames for the exp-003 bulk model.

Every feature here is computable from a ranking-style masked month: nothing
reads a departure's own BLOCK_TIME_UTC_mvt or TAXITIME_SEC_mvt. `build_month`
is therefore run on the raw month frame and is bit-identical on the masked one
(asserted in `check_mask_invariance`).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import prc.data as d
import prc.stand_features as sf

ART = d.ROOT / "artifacts" / "exp-003-anatomy" / "feat"
ART.mkdir(parents=True, exist_ok=True)

CAT = ["apt", "stand_raw", "stand_base", "stand_pfx", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
       "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt",
       "ADES_mvt", "ADES_FILED_flt", "CALLSIGN_PFX"]
BASE_NUM = ["nm_taxi", "aobt_lobt", "aobt_sched", "aobt_eobt1", "lobt_eobt1", "lobt_sched",
            "mvt_sched", "hour", "minute_of_day", "dow", "doy", "sched_hour", "sched_minofday",
            "lobt_changed", "n_dep_stand_month", "n_arr_stand_month",
            "mvt_lobt", "mvt_eobt1", "mvt_iobt", "aobt_iobt", "noac", "flight_len",
            "noac_day_apt", "has_nm_f", "type_mvt_eq_flt", "same_flight_arr", "airborne"]
CSN = ["cs_prev_arr_gap", "cs_prev_arr_seen", "cs_arr_count", "cs_prev_arr_taxi",
       "cs_prev_arr_apt_same", "fid_arr_earlier", "fid_arr_gap"]
NBR = ["nbr_apt_30m", "nbr_apt_prev10_nmtaxi", "nbr_rwy_prev10_nmtaxi", "nbr_apt_prev10_mvtsched",
       "nbr_apt_next10_nmtaxi", "nbr_stand_prev5_nmtaxi"]
OCC = ["taxi_ub_pm", "taxi_lb_nm", "taxi_ub_pa", "taxi_lb_na", "off_lb_pm", "off_lb_pa",
       "off_ub_na", "off_ub_nm", "arr_taxi_pa", "arr_taxi_na", "arr_turnaround_pa",
       "arr_td_gap_pa", "same_type_pa", "same_type_na", "dep_prev_aobt_gap",
       "dep_prev_mvt_gap", "dep_next_aobt_gap", "dep_next_mvt_gap"]


def build_month(m: pd.DataFrame) -> pd.DataFrame:
    df = sf.add_stand_keys(m)
    df["apt_pfx"] = df["apt"] + "|" + df["stand_pfx"]
    occ = sf.occupancy_features(df, key="key_raw")
    occb = sf.occupancy_features(df, key="key_base").add_suffix("_b")
    is_dep = df["PHASE_mvt"].astype("string") == "DEP"
    dep = df.loc[is_dep].copy()

    secs = lambda a, b: (dep[a] - dep[b]).dt.total_seconds()
    dep["nm_taxi"] = secs("MVT_TIME_UTC_mvt", "AOBT_3_flt")
    dep["aobt_lobt"] = secs("AOBT_3_flt", "LOBT_flt")
    dep["aobt_sched"] = secs("AOBT_3_flt", "SCHED_TIME_UTC_mvt")
    dep["aobt_eobt1"] = secs("AOBT_3_flt", "EOBT_1_flt")
    dep["lobt_eobt1"] = secs("LOBT_flt", "EOBT_1_flt")
    dep["lobt_sched"] = secs("LOBT_flt", "SCHED_TIME_UTC_mvt")
    dep["mvt_sched"] = secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
    dep["mvt_lobt"] = secs("MVT_TIME_UTC_mvt", "LOBT_flt")
    dep["mvt_eobt1"] = secs("MVT_TIME_UTC_mvt", "EOBT_1_flt")
    dep["mvt_iobt"] = secs("MVT_TIME_UTC_mvt", "IOBT_flt")
    dep["aobt_iobt"] = secs("AOBT_3_flt", "IOBT_flt")
    dep["lobt_changed"] = (dep["lobt_eobt1"].abs() > 0).astype("float64")
    dep["noac"] = dep["AIRCRAFT_TYPE_mvt"].isna().astype("float64")
    dep["flight_len"] = dep["FLIGHT_mvt"].astype("string").str.len().astype("float64")
    dep["has_nm_f"] = dep["AOBT_3_flt"].notna().astype("float64")
    dep["type_mvt_eq_flt"] = (dep["AIRCRAFT_TYPE_mvt"].astype("string")
                              == dep["AIRCRAFT_TYPE_flt"].astype("string")).astype("float64")
    day = dep["MVT_TIME_UTC_mvt"].dt.floor("D")
    nd_noac = dep.assign(_d=day).groupby(["apt", "_d"], observed=True)["noac"].transform("sum")
    dep["noac_day_apt"] = nd_noac.astype("float32")
    t = dep["MVT_TIME_UTC_mvt"]
    dep["hour"] = t.dt.hour.astype("float32")
    dep["minute_of_day"] = (t.dt.hour * 60 + t.dt.minute).astype("float32")
    dep["dow"] = t.dt.dayofweek.astype("float32")
    dep["doy"] = t.dt.dayofyear.astype("float32")
    s = dep["SCHED_TIME_UTC_mvt"]
    dep["sched_hour"] = s.dt.hour.astype("float32")
    dep["sched_minofday"] = (s.dt.hour * 60 + s.dt.minute).astype("float32")
    dep["CALLSIGN_PFX"] = dep["CALLSIGN_flt"].astype("string").str[:3]
    nd = dep.groupby("key_raw", observed=True)["MVT_ID_mvt"].transform("size")
    dep["n_dep_stand_month"] = nd.astype("float32")
    na = df.loc[~is_dep].groupby("key_raw", observed=True)["MVT_ID_mvt"].size()
    dep["n_arr_stand_month"] = dep["key_raw"].map(na).astype("float32")

    # same NM flight appearing as an arrival elsewhere in the frame
    arr_ids = df.loc[~is_dep, ["FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt"]].dropna(subset=["FLIGHT_ID_mvt"])
    arr_td = arr_ids.groupby("FLIGHT_ID_mvt")["MVT_TIME_UTC_mvt"].min()
    a_td = dep["FLIGHT_ID_mvt"].map(arr_td)
    dep["same_flight_arr"] = a_td.notna().astype("float64")
    dep["airborne"] = (a_td - dep["MVT_TIME_UTC_mvt"]).dt.total_seconds()
    # the assignment asks specifically for an arrival *earlier* in the month
    # frame; the arrival of the same FLIGHT_ID is normally the later
    # destination record, so this is a separate, almost-empty signal.
    dep["fid_arr_earlier"] = ((a_td.notna()) & (a_td < dep["MVT_TIME_UTC_mvt"])).astype("float64")
    dep["fid_arr_gap"] = np.where(dep["fid_arr_earlier"].to_numpy() > 0,
                                  (dep["MVT_TIME_UTC_mvt"] - a_td).dt.total_seconds(), np.nan)

    # Q3: the same CALLSIGN_flt appearing as an arrival earlier in the month frame.
    # (The same FLIGHT_ID_mvt link is `same_flight_arr` above; a callsign match is
    # a different object -- 0.8 % of matched pairs are the same flight leg, the
    # rest are the same flight number on an earlier day.)
    arr_cs = df.loc[~is_dep, ["CALLSIGN_flt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "apt"]].dropna(
        subset=["CALLSIGN_flt", "MVT_TIME_UTC_mvt"]).copy()
    arr_cs["CALLSIGN_flt"] = arr_cs["CALLSIGN_flt"].astype(str)
    arr_cs = arr_cs.sort_values("MVT_TIME_UTC_mvt", kind="mergesort")
    left = dep[["CALLSIGN_flt", "MVT_TIME_UTC_mvt"]].copy()
    left["CALLSIGN_flt"] = left["CALLSIGN_flt"].astype(str)
    left = left.reset_index().rename(columns={"index": "_i"}).sort_values(
        "MVT_TIME_UTC_mvt", kind="mergesort")
    # keep the arrival's own touchdown time under a second name: merge_asof
    # returns the LEFT frame's value in the join column, so joining on
    # MVT_TIME_UTC_mvt and then differencing it gives zero on every row.
    arr_cs["_arr_t"] = arr_cs["MVT_TIME_UTC_mvt"]
    mm = pd.merge_asof(left, arr_cs.rename(columns={"TAXITIME_SEC_mvt": "_at", "apt": "_aapt"}),
                       on="MVT_TIME_UTC_mvt", by="CALLSIGN_flt", direction="backward",
                       allow_exact_matches=False, suffixes=("", "_r")).set_index("_i")
    prev_t = mm["_arr_t"].reindex(dep.index)
    dep["cs_prev_arr_gap"] = (dep["MVT_TIME_UTC_mvt"] - prev_t).dt.total_seconds()
    dep["cs_prev_arr_seen"] = prev_t.notna().astype("float64")
    dep["cs_prev_arr_taxi"] = mm["_at"].reindex(dep.index).astype("float64")
    dep["cs_prev_arr_apt_same"] = (mm["_aapt"].reindex(dep.index).astype(str)
                                   == dep["apt"].astype(str)).astype("float64")
    cs_n = arr_cs.groupby("CALLSIGN_flt").size()
    dep["cs_arr_count"] = dep["CALLSIGN_flt"].astype(str).map(cs_n).astype("float64")

    # neighbouring departures at the same airport / runway / stand (take-off order)
    dep = dep.sort_values("MVT_TIME_UTC_mvt", kind="mergesort")
    def _roll(by, col, k, name, shift=1):
        g = dep.groupby(by, observed=True)[col]
        dep[name] = (g.shift(shift).rolling(k, min_periods=1).mean().to_numpy()
                     if shift > 0 else
                     g.transform(lambda s: s[::-1].shift(1).rolling(k, min_periods=1).mean()[::-1]).to_numpy())
    dep["nbr_apt_prev10_nmtaxi"] = dep.groupby("apt", observed=True)["nm_taxi"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()).astype("float32")
    dep["nbr_apt_next10_nmtaxi"] = dep.groupby("apt", observed=True)["nm_taxi"].transform(
        lambda s: s[::-1].shift(1).rolling(10, min_periods=1).mean()[::-1]).astype("float32")
    dep["nbr_rwy_prev10_nmtaxi"] = dep.groupby(["apt", "RUNWAY_mvt"], observed=True)["nm_taxi"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()).astype("float32")
    dep["nbr_apt_prev10_mvtsched"] = dep.groupby("apt", observed=True)["mvt_sched"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()).astype("float32")
    dep["nbr_stand_prev5_nmtaxi"] = dep.groupby("key_raw", observed=True)["nm_taxi"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()).astype("float32")
    # seconds, whatever unit the parquet stores (these files are datetime64[us],
    # so `.astype("int64") // 10**9` silently gave kiloseconds and turned the
    # +/-30-minute window into +/-21 days; caught 2026-09-05 by the Q4 numeric
    # shift table, where the "30-minute" count had a mean of 16,421).
    ts = dep["MVT_TIME_UTC_mvt"].dt.as_unit("s").astype("int64")
    cnt = []
    for a, idx in dep.groupby("apt", observed=True).indices.items():
        v = ts.to_numpy()[idx]
        cnt.append(pd.Series(np.searchsorted(v, v + 1800, "right") - np.searchsorted(v, v - 1800, "left") - 1,
                             index=dep.index[idx]))
    dep["nbr_apt_30m"] = pd.concat(cnt).reindex(dep.index).astype("float32")
    dep = dep.sort_index()

    keep = ["MVT_ID_mvt", "key_raw", "key_base", "apt_pfx", "month",
            "MVT_TIME_UTC_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    out = pd.concat([dep[keep + CAT + BASE_NUM + NBR + CSN],
                     occ.reindex(dep.index), occb.reindex(dep.index)], axis=1)
    out["has_nm"] = dep["AOBT_3_flt"].notna().to_numpy()
    return out.reset_index(drop=True)


def build_all():
    for m in list(d.MONTHS):
        p = ART / f"{m}.parquet"
        if p.exists():
            print("skip", m, flush=True); continue
        f = build_month(d.load_month(m))
        f.to_parquet(p, index=False)
        print("built", m, f.shape, flush=True)
    p = ART / "ranking.parquet"
    if not p.exists():
        # The ranking file holds January and July 2026 as two isolated months
        # (`prc/data.py`). Build each separately: counting and neighbour
        # features must not join across the six-month gap, or the ranking rows
        # get features from a distribution the training rows never see (before
        # this split, 98.2 % of ranking rows had a stand count roughly twice
        # its within-month value and 2,154 rows had a neighbour gap of up to
        # 204 days). Sol review, 2026-09-05.
        rk = d.load_ranking()
        parts = [build_month(g) for _, g in rk.groupby(rk["month"], observed=True)]
        f = pd.concat(parts, ignore_index=True)
        f.to_parquet(p, index=False)
        print("built ranking", f.shape, "from", len(parts), "monthly frames", flush=True)


def check_mask_invariance(month="2025-07"):
    m = d.load_month(month)
    a = build_month(m)
    masked, _ = d.mask_like_ranking(m)
    b = build_month(masked)
    cols = [c for c in a.columns if c not in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")]
    same = all(a[c].astype(str).equals(b[c].astype(str)) for c in cols)
    print("mask-invariant:", same)
    if not same:
        for c in cols:
            if not a[c].astype(str).equals(b[c].astype(str)):
                print("  differs:", c)
    return same


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_mask_invariance()
    else:
        build_all()
