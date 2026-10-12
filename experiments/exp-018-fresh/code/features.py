"""exp-018 fresh lineage: month-local feature construction.

Contract (docs/input-rule.md): a month's features are computed from that
month's frame alone, using only columns present in the ranking file. Departure
BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt are never read here -- not even in a
training month -- so a training row and a ranking row see the same feature
definition. Arrival rows keep every column, including their in-block time and
taxi-in, and are used freely.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

SEC = 1e9  # ns per second when viewing datetime64 as int64


def _secs(a: pd.Series, b: pd.Series) -> np.ndarray:
    return (a - b).dt.total_seconds().to_numpy(dtype="float64")


_INAT = np.int64(-9223372036854775808)


def _epoch(s: pd.Series) -> np.ndarray:
    """UTC seconds as float, NaN where NaT."""
    iv = s.to_numpy(dtype="datetime64[ns]").view("int64")
    out = np.where(iv == _INAT, np.nan, iv / 1e9)
    return out.astype("float64")


def _count_in_window(event_t: np.ndarray, query_t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """#events with query+lo <= event <= query+hi. event_t must be sorted, finite."""
    a = np.searchsorted(event_t, query_t + lo, side="left")
    b = np.searchsorted(event_t, query_t + hi, side="right")
    return (b - a).astype("float32")


def _grouped_window_counts(gkey: np.ndarray, event_t: np.ndarray, qkey: np.ndarray,
                           query_t: np.ndarray, windows: list[tuple[float, float]]) -> list[np.ndarray]:
    """Per-group window counts. Groups are integer codes over a shared space."""
    out = [np.zeros(len(query_t), dtype="float32") for _ in windows]
    ng = int(max(gkey.max() if len(gkey) else 0, qkey.max() if len(qkey) else 0)) + 2
    order = np.lexsort((event_t, gkey))
    gk, et = gkey[order], event_t[order]
    starts = np.searchsorted(gk, np.arange(ng), side="left")
    qorder = np.argsort(qkey, kind="stable")
    qk, qt = qkey[qorder], query_t[qorder]
    qstarts = np.searchsorted(qk, np.arange(ng), side="left")
    for g in range(ng - 1):
        s, e = starts[g], starts[g + 1]
        qs, qe = qstarts[g], qstarts[g + 1]
        if qe <= qs or e <= s:
            continue
        ev = et[s:e]
        for wi, (lo, hi) in enumerate(windows):
            out[wi][qorder[qs:qe]] = _count_in_window(ev, qt[qs:qe], lo, hi)
    return out


def _prev_next(event_t: np.ndarray, event_v: np.ndarray, gkey_e: np.ndarray,
               query_t: np.ndarray, gkey_q: np.ndarray, strict: bool = False):
    """For each query: (dt to previous event, value of previous event,
    dt to next event, value of next event) within the same group."""
    n = len(query_t)
    pdt = np.full(n, np.nan); pv = np.full(n, np.nan)
    ndt = np.full(n, np.nan); nv = np.full(n, np.nan)
    order = np.lexsort((event_t, gkey_e))
    gk, et, evv = gkey_e[order], event_t[order], event_v[order]
    ng = int(max(gk.max() if len(gk) else 0, gkey_q.max() if n else 0)) + 2
    starts = np.searchsorted(gk, np.arange(ng), side="left")
    qorder = np.argsort(gkey_q, kind="stable")
    qk, qt = gkey_q[qorder], query_t[qorder]
    qstarts = np.searchsorted(qk, np.arange(ng), side="left")
    for g in range(ng - 1):
        s, e = starts[g], starts[g + 1]
        qs, qe = qstarts[g], qstarts[g + 1]
        if qe <= qs:
            continue
        idx = qorder[qs:qe]
        if e <= s:
            continue
        t = qt[qs:qe]
        jl = np.searchsorted(et[s:e], t, side="left")
        jr = np.searchsorted(et[s:e], t, side="right")
        jn0 = jr if strict else jl
        has_p = jl > 0
        jp = np.clip(jl - 1, 0, e - s - 1)
        pdt[idx] = np.where(has_p, t - et[s:e][jp], np.nan)
        pv[idx] = np.where(has_p, evv[s:e][jp], np.nan)
        has_n = jn0 < (e - s)
        jn = np.clip(jn0, 0, e - s - 1)
        ndt[idx] = np.where(has_n, et[s:e][jn] - t, np.nan)
        nv[idx] = np.where(has_n, evv[s:e][jn], np.nan)
    return pdt, pv, ndt, nv


def _rolling_prev_mean(event_t: np.ndarray, event_v: np.ndarray, gkey_e: np.ndarray,
                       query_t: np.ndarray, gkey_q: np.ndarray, k: int) -> np.ndarray:
    """Mean of the last k event values strictly before the query, per group."""
    n = len(query_t)
    out = np.full(n, np.nan)
    order = np.lexsort((event_t, gkey_e))
    gk, et, evv = gkey_e[order], event_t[order], event_v[order]
    ng = int(max(gk.max() if len(gk) else 0, gkey_q.max() if n else 0)) + 2
    starts = np.searchsorted(gk, np.arange(ng), side="left")
    qorder = np.argsort(gkey_q, kind="stable")
    qk, qt = gkey_q[qorder], query_t[qorder]
    qstarts = np.searchsorted(qk, np.arange(ng), side="left")
    for g in range(ng - 1):
        s, e = starts[g], starts[g + 1]
        qs, qe = qstarts[g], qstarts[g + 1]
        if qe <= qs or e <= s:
            continue
        v = evv[s:e]
        cs = np.concatenate([[0.0], np.nancumsum(v)])
        cnt = np.concatenate([[0], np.cumsum(~np.isnan(v))])
        j = np.searchsorted(et[s:e], qt[qs:qe], side="left")
        lo = np.maximum(j - k, 0)
        num = cs[j] - cs[lo]
        den = cnt[j] - cnt[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[qorder[qs:qe]] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return out


def build(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature frame for the DEP rows of one month, indexed like `frame`'s DEP rows."""
    f = frame
    apt = np.where(f["PHASE_mvt"].to_numpy() == "DEP", f["ADEP_mvt"], f["ADES_mvt"])
    f = f.assign(_apt=apt)
    dep = f[f["PHASE_mvt"] == "DEP"]
    arr = f[f["PHASE_mvt"] == "ARR"]
    X = pd.DataFrame(index=dep.index)

    mvt = _epoch(dep["MVT_TIME_UTC_mvt"])                     # take-off, always known
    X["r_aobt"] = mvt - _epoch(dep["AOBT_3_flt"])             # NM proxy for the target
    X["ms_sched"] = mvt - _epoch(dep["SCHED_TIME_UTC_mvt"])
    X["m_lobt"] = mvt - _epoch(dep["LOBT_flt"])
    X["m_iobt"] = mvt - _epoch(dep["IOBT_flt"])
    X["m_eobt"] = mvt - _epoch(dep["EOBT_1_flt"])
    X["aobt_sched"] = _epoch(dep["AOBT_3_flt"]) - _epoch(dep["SCHED_TIME_UTC_mvt"])
    X["aobt_eobt"] = _epoch(dep["AOBT_3_flt"]) - _epoch(dep["EOBT_1_flt"])
    X["sched_lobt"] = _epoch(dep["SCHED_TIME_UTC_mvt"]) - _epoch(dep["LOBT_flt"])
    X["eobt_lobt"] = _epoch(dep["EOBT_1_flt"]) - _epoch(dep["LOBT_flt"])
    X["plan_flight_time"] = _epoch(dep["ARVT_1_flt"]) - _epoch(dep["EOBT_1_flt"])
    X["act_flight_time"] = _epoch(dep["ARVT_3_flt"]) - mvt
    X["arvt3_arvt1"] = _epoch(dep["ARVT_3_flt"]) - _epoch(dep["ARVT_1_flt"])

    t = pd.DatetimeIndex(dep["MVT_TIME_UTC_mvt"])
    X["hour"] = t.hour.to_numpy(dtype="float32") + t.minute.to_numpy() / 60.0
    X["dow"] = t.dayofweek.to_numpy(dtype="float32")
    X["dom"] = t.day.to_numpy(dtype="float32")
    X["doy"] = t.dayofyear.to_numpy(dtype="float32")
    X["sin_h"] = np.sin(2 * np.pi * X["hour"] / 24.0)
    X["cos_h"] = np.cos(2 * np.pi * X["hour"] / 24.0)

    # ------------------------------------------------------------ congestion
    dep_t = mvt
    aobt = _epoch(dep["AOBT_3_flt"])
    arr_land = _epoch(arr["MVT_TIME_UTC_mvt"])
    arr_blk = _epoch(arr["BLOCK_TIME_UTC_mvt"])
    arr_taxi = arr["TAXITIME_SEC_mvt"].to_numpy(dtype="float64")

    apt_cat = pd.Categorical(f["_apt"])
    apt_dep = apt_cat.codes[(f["PHASE_mvt"] == "DEP").to_numpy()].astype("int64")
    apt_arr = apt_cat.codes[(f["PHASE_mvt"] == "ARR").to_numpy()].astype("int64")
    rwy_cat = pd.Categorical(f["_apt"].astype(str) + "|" + f["RUNWAY_mvt"].astype(str))
    rwy_dep = rwy_cat.codes[(f["PHASE_mvt"] == "DEP").to_numpy()].astype("int64")
    rwy_arr = rwy_cat.codes[(f["PHASE_mvt"] == "ARR").to_numpy()].astype("int64")
    stand_cat = pd.Categorical(f["_apt"].astype(str) + "|" + f["STAND_mvt"].astype(str))
    stand_dep = stand_cat.codes[(f["PHASE_mvt"] == "DEP").to_numpy()].astype("int64")
    stand_arr = stand_cat.codes[(f["PHASE_mvt"] == "ARR").to_numpy()].astype("int64")

    wins = [(-900.0, 0.0), (-1800.0, 0.0), (0.0, 900.0), (-300.0, 300.0)]
    c = _grouped_window_counts(apt_dep, dep_t, apt_dep, dep_t, wins)
    for name, v in zip(["apt_to_15m_before", "apt_to_30m_before", "apt_to_15m_after", "apt_to_5m_around"], c):
        X[name] = v
    c = _grouped_window_counts(rwy_dep, dep_t, rwy_dep, dep_t, wins)
    for name, v in zip(["rwy_to_15m_before", "rwy_to_30m_before", "rwy_to_15m_after", "rwy_to_5m_around"], c):
        X[name] = v
    ok = np.isfinite(arr_land)
    c = _grouped_window_counts(apt_arr[ok], arr_land[ok], apt_dep, dep_t, [(-900.0, 0.0), (0.0, 900.0), (-1800.0, 0.0)])
    for name, v in zip(["apt_land_15m_before", "apt_land_15m_after", "apt_land_30m_before"], c):
        X[name] = v
    ok = np.isfinite(arr_blk)
    c = _grouped_window_counts(apt_arr[ok], arr_blk[ok], apt_dep, dep_t, [(-1800.0, 0.0), (0.0, 1800.0)])
    for name, v in zip(["apt_inblk_30m_before", "apt_inblk_30m_after"], c):
        X[name] = v

    # queue proxy: departures off-block (NM) at t but not yet airborne
    finite = np.isfinite(aobt)
    pushed = _grouped_window_counts(apt_dep[finite], aobt[finite], apt_dep, dep_t,
                                    [(-7200.0, 0.0), (-1800.0, 0.0), (-600.0, 0.0)])
    gone = _grouped_window_counts(apt_dep[finite], dep_t[finite], apt_dep, dep_t, [(-7200.0, 0.0)])
    X["apt_pushed_30m"] = pushed[1]
    X["apt_pushed_10m"] = pushed[2]
    X["apt_queue"] = pushed[0] - gone[0]          # off-block and still on the surface
    pushed_r = _grouped_window_counts(rwy_dep[finite], aobt[finite], rwy_dep, dep_t, [(-1800.0, 0.0), (-7200.0, 0.0)])
    gone_r = _grouped_window_counts(rwy_dep[finite], dep_t[finite], rwy_dep, dep_t, [(-7200.0, 0.0)])
    X["rwy_pushed_30m"] = pushed_r[0]
    X["rwy_queue"] = pushed_r[1] - gone_r[0]

    # gaps on the runway
    pdt, _, ndt, _ = _prev_next(dep_t, dep_t, rwy_dep, dep_t, rwy_dep, strict=True)
    X["rwy_gap_prev"] = pdt
    X["rwy_gap_next"] = ndt
    pdt, _, ndt, _ = _prev_next(dep_t, dep_t, apt_dep, dep_t, apt_dep, strict=True)
    X["apt_gap_prev"] = pdt
    X["apt_gap_next"] = ndt

    # live surface state from arrivals (fully observed in the ranking file)
    ok = np.isfinite(arr_blk) & np.isfinite(arr_taxi)
    X["arr_taxiin_last10"] = _rolling_prev_mean(arr_blk[ok], arr_taxi[ok], apt_arr[ok], dep_t, apt_dep, 10)
    X["arr_taxiin_last30"] = _rolling_prev_mean(arr_blk[ok], arr_taxi[ok], apt_arr[ok], dep_t, apt_dep, 30)
    apt_med = pd.Series(arr_taxi[ok]).groupby(apt_arr[ok]).median()
    base = apt_med.reindex(apt_dep).to_numpy()
    X["arr_taxiin_ratio"] = X["arr_taxiin_last30"].to_numpy() / np.where(base > 0, base, np.nan)

    # ---- realised surface speed: other departures' MVT - AOBT (legal in ranking)
    r_all = mvt - aobt
    okr = np.isfinite(r_all)
    X["r_prev10_apt"] = _rolling_prev_mean(dep_t[okr], r_all[okr], apt_dep[okr], dep_t, apt_dep, 10)
    X["r_prev30_apt"] = _rolling_prev_mean(dep_t[okr], r_all[okr], apt_dep[okr], dep_t, apt_dep, 30)
    X["r_prev10_rwy"] = _rolling_prev_mean(dep_t[okr], r_all[okr], rwy_dep[okr], dep_t, rwy_dep, 10)
    X["r_prev5_stand"] = _rolling_prev_mean(dep_t[okr], r_all[okr], stand_dep[okr], dep_t, stand_dep, 5)
    r_apt_med = pd.Series(r_all[okr]).groupby(apt_dep[okr]).median()
    rb = r_apt_med.reindex(apt_dep).to_numpy()
    X["r_prev30_ratio"] = X["r_prev30_apt"].to_numpy() / np.where(rb > 0, rb, np.nan)
    r_rwy_med = pd.Series(r_all[okr]).groupby(rwy_dep[okr]).median()
    X["r_rwy_median"] = r_rwy_med.reindex(rwy_dep).to_numpy()

    # ---- day-level disruption, from arrivals and from other departures
    day = np.floor(dep_t / 86400.0)
    day_a = np.floor(arr_land / 86400.0)
    dkey = pd.Series(apt_dep.astype("int64") * 100000 + day.astype("int64"))
    okt = np.isfinite(arr_taxi) & np.isfinite(day_a)
    dkey_a = pd.Series(apt_arr[okt].astype("int64") * 100000 + day_a[okt].astype("int64"))
    X["day_arr_taxiin"] = dkey.map(pd.Series(arr_taxi[okt]).groupby(dkey_a).mean()).to_numpy()
    dkey_d = pd.Series(apt_dep[okr].astype("int64") * 100000 + day[okr].astype("int64"))
    X["day_r_mean"] = dkey.map(pd.Series(r_all[okr]).groupby(dkey_d).mean()).to_numpy()
    X["day_n_dep"] = dkey.map(dkey.value_counts()).to_numpy()

    # ---- arrival punctuality nearby: landing minus the filed arrival time
    arr_delay = arr_land - _epoch(arr["ARVT_1_flt"])
    oka = np.isfinite(arr_delay)
    X["arr_delay_last20"] = _rolling_prev_mean(arr_land[oka], arr_delay[oka], apt_arr[oka], dep_t, apt_dep, 20)

    # ---- scheduled demand around the take-off (capacity pressure)
    sch = _epoch(dep["SCHED_TIME_UTC_mvt"])
    oks = np.isfinite(sch)
    c = _grouped_window_counts(apt_dep[oks], sch[oks], apt_dep, dep_t, [(-1800.0, 1800.0), (-3600.0, 0.0)])
    X["apt_sched_1h_around"] = c[0]
    X["apt_sched_1h_before"] = c[1]
    X["apt_dep_1h_before"] = _grouped_window_counts(apt_dep, dep_t, apt_dep, dep_t, [(-3600.0, 0.0)])[0]
    X["demand_over_served"] = X["apt_sched_1h_before"] / np.maximum(X["apt_dep_1h_before"], 1.0)

    # ------------------------------------------------------- stand occupancy
    ok = np.isfinite(arr_blk)
    pdt, pv, ndt, nv = _prev_next(arr_blk[ok], arr_land[ok], stand_arr[ok], dep_t, stand_dep)
    X["stand_prev_arr_inblock_dt"] = pdt          # hard-ish upper bound on the taxi time
    X["stand_next_arr_inblock_dt"] = ndt
    pdt2, _, ndt2, _ = _prev_next(dep_t, dep_t, stand_dep, dep_t, stand_dep, strict=True)
    X["stand_prev_dep_dt"] = pdt2
    X["stand_next_dep_dt"] = ndt2
    X["stand_bound_minus_ms"] = X["stand_prev_arr_inblock_dt"] - X["ms_sched"]
    X["stand_bound_over_ms"] = X["stand_prev_arr_inblock_dt"] / X["ms_sched"].replace(0, np.nan)
    X["stand_bound_minus_r"] = X["stand_prev_arr_inblock_dt"] - X["r_aobt"]

    # ---- stand geometry measured from arrivals in the same month (legal, and
    # it works for a stand that never appears in a training month)
    okt = np.isfinite(arr_taxi)
    st_med = pd.Series(arr_taxi[okt]).groupby(stand_arr[okt]).median()
    st_cnt = pd.Series(arr_taxi[okt]).groupby(stand_arr[okt]).size()
    X["stand_taxiin_med"] = st_med.reindex(stand_dep).to_numpy()
    X["stand_taxiin_n"] = st_cnt.reindex(stand_dep).fillna(0).to_numpy()
    apt_med_t = pd.Series(arr_taxi[okt]).groupby(apt_arr[okt]).median()
    X["stand_taxiin_rel"] = X["stand_taxiin_med"].to_numpy() / apt_med_t.reindex(apt_dep).to_numpy()
    rw_med = pd.Series(arr_taxi[okt]).groupby(rwy_arr[okt]).median()
    X["rwy_taxiin_med"] = rw_med.reindex(rwy_dep).to_numpy()

    # ---- the turnaround: who was on this stand before us
    ok = np.isfinite(arr_blk)
    oper_arr = pd.factorize(f["AIRCRAFT_OPERATOR_flt"].astype(str))[0]
    type_arr = pd.factorize(f["AIRCRAFT_TYPE_mvt"].astype(str))[0]
    is_arr = (f["PHASE_mvt"] == "ARR").to_numpy()
    is_dep = ~is_arr
    _, prev_oper, _, _ = _prev_next(arr_blk[ok], oper_arr[is_arr][ok].astype("float64"),
                                    stand_arr[ok], dep_t, stand_dep)
    _, prev_type, _, _ = _prev_next(arr_blk[ok], type_arr[is_arr][ok].astype("float64"),
                                    stand_arr[ok], dep_t, stand_dep)
    _, prev_taxiin, _, _ = _prev_next(arr_blk[ok], arr_taxi[ok], stand_arr[ok], dep_t, stand_dep)
    X["prev_arr_same_oper"] = (prev_oper == oper_arr[is_dep]).astype("float32")
    X["prev_arr_same_type"] = (prev_type == type_arr[is_dep]).astype("float32")
    X["prev_arr_taxiin"] = prev_taxiin

    # ---- runway configuration in force
    rc = _grouped_window_counts(rwy_dep, dep_t, rwy_dep, dep_t, [(-3600.0, 0.0)])[0]
    ac = _grouped_window_counts(apt_dep, dep_t, apt_dep, dep_t, [(-3600.0, 0.0)])[0]
    X["rwy_share_1h"] = rc / np.maximum(ac, 1.0)
    arr_rc = _grouped_window_counts(rwy_arr[np.isfinite(arr_land)], arr_land[np.isfinite(arr_land)],
                                    rwy_dep, dep_t, [(-3600.0, 0.0)])[0]
    X["rwy_arrivals_1h"] = arr_rc                # is our departure runway also landing traffic

    # --------------------------------------------------------------- flags
    X["no_nm"] = dep["AOBT_3_flt"].isna().to_numpy().astype("float32")
    X["ghost"] = dep["AIRCRAFT_TYPE_mvt"].isna().to_numpy().astype("float32")
    X["n_stand_month"] = pd.Series(stand_dep).map(pd.Series(stand_dep).value_counts()).to_numpy()

    # ------------------------------------------- K2: the transient-window block
    # Off by default, so the frozen fold4-final path is byte-identical without
    # the flag and the Kaggle acceptance fold needs no GPU to re-baseline.
    # See window_features.py for why this is a complement and not a family.
    if os.environ.get("PRC_WINDOW") == "1":
        import window_features as _wf
        W = _wf.build(dep_t, apt_dep, r_all, rwy_dep, arr_land, apt_arr, arr_taxi)
        for c in W.columns:
            X[c] = W[c].to_numpy()

    # ---------------------------------------------------------- categoricals
    cats = {
        "c_apt": dep["ADEP_mvt"],
        "c_rwy": dep["ADEP_mvt"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str),
        "c_stand": dep["ADEP_mvt"].astype(str) + "|" + dep["STAND_mvt"].astype(str),
        "c_standpfx": dep["ADEP_mvt"].astype(str) + "|" + dep["STAND_mvt"].astype(str).str.replace(r"\d+$", "", regex=True).str[:3]
                      + dep["STAND_mvt"].astype(str).str.extract(r"^(\D*\d)", expand=False).fillna(""),
        "c_actype": dep["AIRCRAFT_TYPE_mvt"],
        "c_wtc": dep["WK_TBL_CAT_flt"],
        "c_seg": dep["MARKET_SEGMENT_flt"],
        "c_ftype": dep["FLIGHT_TYPE_flt"],
        "c_frule": dep["FLIGHT_RULE_mvt"],
        "c_oper": dep["AIRCRAFT_OPERATOR_flt"].astype(str).str[:12],
        "c_ades": dep["ADES_mvt"],
        "c_adescty": dep["ADES_mvt"].astype(str).str[:2],
        "c_csgn": dep["FLIGHT_mvt"].astype(str).str[:3],
    }
    for k, v in cats.items():
        X[k] = v.astype(str).to_numpy()
    return X


CAT_COLS = ["c_apt", "c_rwy", "c_stand", "c_standpfx", "c_actype", "c_wtc", "c_seg",
            "c_ftype", "c_frule", "c_oper", "c_ades", "c_adescty", "c_csgn"]
