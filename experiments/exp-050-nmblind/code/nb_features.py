"""exp-050-nmblind: the two feature blocks, built from one masked month frame.

**BLIND block** -- read only through `BLIND_WHITELIST`.  `blind_features()` first cuts
the frame down to those columns, so no other column can reach it by construction, and
`prove_blind()` checks the same thing by mutation.  The whitelist is the movement table
minus every field the assignment excludes:

    kept      MVT_ID_mvt (row key only)  PHASE_mvt (DEP/ARR split only)  ADEP_mvt  ADES_mvt
              FLIGHT_RULE_mvt  MVT_TIME_UTC_mvt  SCHED_TIME_UTC_mvt  AIRCRAFT_TYPE_mvt
              RUNWAY_mvt  STAND_mvt  BLOCK_TIME_UTC_mvt  TAXITIME_SEC_mvt
    excluded  FLIGHT_mvt (assignment)  FLIGHT_ID_mvt (the NM join key: its null IS the
              no-NM flag)  every *_flt column (NM, including AIRCRAFT_OPERATOR_flt)

`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are read on ARRIVAL rows only; the frame is
masked like the ranking file first and `_assert_masked` refuses a departure value.

**AWARE block** -- the blind block plus `NM_FEATURES`, which read the NM fields and their
NM-derived context.  Same rows, same learner downstream: the two experts differ only by
this block.

Every context feature reads other rows of the SAME masked month frame, at the same
airport (departures: `ADEP_mvt`; arrivals: `ADES_mvt`).

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-050-nmblind/code/nb_features.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.blend as PB                                                # noqa: E402
import prc.data as D                                                  # noqa: E402

BLIND_WHITELIST = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_RULE_mvt",
                   "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt",
                   "RUNWAY_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
NM_INPUTS = ["FLIGHT_ID_mvt", "FLIGHT_mvt"] + [
    "LOBT_flt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "EOBT_1_flt",
    "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]

BLIND_CATS = ["apt", "ac_type", "ades", "flight_rule", "apt_rwy", "apt_stand"]
BLIND_NUM = ["mvt_sched", "hour", "sched_hour", "dow",
             "dep_apt_pre900", "dep_apt_pre1800", "dep_apt_post900", "dep_apt_post1800",
             "dep_rwy_pre900", "dep_rwy_post900", "gap_prev_rwy", "gap_next_rwy",
             "n_to_since_sched", "sched_apt_pre1800", "sched_apt_post1800",
             "arr_apt_pre1800", "arr_apt_post1800", "arr_inblk_pre1800",
             "arr_taxiin_mean3600", "st_prev_arr", "st_prev_arr_same_type",
             "st_next_arr", "st_prev_dep"]
BLIND_FEATURES = BLIND_CATS + BLIND_NUM
NM_CATS = ["nm_operator", "nm_market", "nm_ftype", "nm_wtc"]
NM_NUM = ["to_aobt", "to_lobt", "to_iobt", "to_eobt1", "to_arvt1", "to_arvt3",
          "aobt_lobt", "aobt_sched", "nm_ades_filed_differs", "nm_taxiing_at_aobt",
          "nm_to_since_aobt", "brk_prev_span", "brk_next_span"]
NM_FEATURES = NM_CATS + NM_NUM
AWARE_FEATURES = BLIND_FEATURES + NM_FEATURES
CATS = BLIND_CATS + NM_CATS
WORK = EXP / "work"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def ep(s: pd.Series) -> np.ndarray:
    return (s - EPOCH).dt.total_seconds().to_numpy("float64")


def _assert_masked(frame: pd.DataFrame) -> None:
    dep = frame["PHASE_mvt"].to_numpy() == "DEP"
    for c in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"):
        if c in frame.columns and frame.loc[dep, c].notna().any():
            raise SystemExit(f"{c} carries a departure value: the frame is not masked")


def _groups(key: np.ndarray):
    codes, uniq = pd.factorize(key)
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    for idx in np.split(order, bounds):
        if len(idx):
            yield idx


def _count_window(q: np.ndarray, qkey: np.ndarray, s: np.ndarray, skey: np.ndarray,
                  lo_off: float, hi_off: float) -> np.ndarray:
    """Per query: #source rows with the same key and time in (q+lo_off, q+hi_off]."""
    out = np.zeros(len(q))
    skey_codes, suniq = pd.factorize(skey)
    src = pd.Series(np.arange(len(s))).groupby(skey_codes).indices
    lut = {k: i for i, k in enumerate(suniq)}
    for idx in _groups(qkey):
        k = lut.get(qkey[idx[0]])
        if k is None:
            continue
        st = np.sort(s[src[k]])
        st = st[np.isfinite(st)]
        qq = q[idx]
        out[idx] = (np.searchsorted(st, qq + hi_off, "right")
                    - np.searchsorted(st, qq + lo_off, "right"))
    out[~np.isfinite(q)] = np.nan
    return out


def _prev_next_same(t: np.ndarray, key: np.ndarray, valid: np.ndarray):
    """Time to the previous / next other row with the same key, in stable time order."""
    prev_g, next_g = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for idx in _groups(key):
        idx = idx[valid[idx] & np.isfinite(t[idx])]
        if len(idx) < 2:
            continue
        o = idx[np.argsort(t[idx], kind="stable")]
        tt = t[o]
        prev_g[o[1:]] = tt[1:] - tt[:-1]
        next_g[o[:-1]] = tt[1:] - tt[:-1]
    return prev_g, next_g


def blind_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The BLIND block for every departure row of one masked month frame."""
    f = frame[BLIND_WHITELIST]            # structural whitelist: nothing else is readable
    _assert_masked(f)
    ph = f["PHASE_mvt"].to_numpy()
    dep = f[ph == "DEP"]
    arr = f[ph == "ARR"]
    t = ep(dep["MVT_TIME_UTC_mvt"])
    sch = ep(dep["SCHED_TIME_UTC_mvt"])
    apt = dep["ADEP_mvt"].astype(str).to_numpy()
    rwy = np.char.add(np.char.add(apt.astype("U8"), ":"),
                      dep["RUNWAY_mvt"].astype(str).to_numpy().astype("U16"))
    stand_raw = dep["STAND_mvt"]
    stand = np.char.add(np.char.add(apt.astype("U8"), ":"),
                        stand_raw.astype(str).to_numpy().astype("U32"))
    X = pd.DataFrame(index=dep.index)
    X["MVT_ID_mvt"] = dep["MVT_ID_mvt"].to_numpy("float64")
    X["apt"] = apt
    X["ac_type"] = dep["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()
    X["ades"] = dep["ADES_mvt"].astype(str).to_numpy()
    X["flight_rule"] = dep["FLIGHT_RULE_mvt"].astype(str).to_numpy()
    X["apt_rwy"] = rwy
    X["apt_stand"] = stand
    X["mvt_sched"] = t - sch
    mv = dep["MVT_TIME_UTC_mvt"]
    sc = dep["SCHED_TIME_UTC_mvt"]
    X["hour"] = (mv.dt.hour + mv.dt.minute / 60.0).to_numpy("float64")
    X["sched_hour"] = (sc.dt.hour + sc.dt.minute / 60.0).to_numpy("float64")
    X["dow"] = mv.dt.dayofweek.to_numpy("float64")
    # other departures' take-off at the airport and on the runway (self excluded: -1 on pre)
    X["dep_apt_pre900"] = _count_window(t, apt, t, apt, -900, 0) - 1
    X["dep_apt_pre1800"] = _count_window(t, apt, t, apt, -1800, 0) - 1
    X["dep_apt_post900"] = _count_window(t, apt, t, apt, 0, 900)
    X["dep_apt_post1800"] = _count_window(t, apt, t, apt, 0, 1800)
    X["dep_rwy_pre900"] = _count_window(t, rwy, t, rwy, -900, 0) - 1
    X["dep_rwy_post900"] = _count_window(t, rwy, t, rwy, 0, 900)
    pg, ng = _prev_next_same(t, rwy, np.ones(len(t), bool))
    X["gap_prev_rwy"], X["gap_next_rwy"] = pg, ng
    # departures that took off between this row's schedule and its own take-off
    lo = np.minimum(sch, t)
    since = np.zeros(len(t))
    for idx in _groups(apt):
        st = np.sort(t[idx][np.isfinite(t[idx])])
        since[idx] = (np.searchsorted(st, t[idx], "left")
                      - np.searchsorted(st, lo[idx], "right"))
    since[~np.isfinite(lo)] = np.nan
    X["n_to_since_sched"] = np.maximum(since, 0)
    X["sched_apt_pre1800"] = _count_window(t, apt, sch, apt, -1800, 0)
    X["sched_apt_post1800"] = _count_window(t, apt, sch, apt, 0, 1800)
    # arrivals at the same airport: landing, in-block and taxi-in (all legal on ARR rows)
    a_apt = arr["ADES_mvt"].astype(str).to_numpy()
    a_land = ep(arr["MVT_TIME_UTC_mvt"])
    a_blk = ep(arr["BLOCK_TIME_UTC_mvt"])
    a_taxi = np.clip(arr["TAXITIME_SEC_mvt"].to_numpy("float64"), 0, 1800)
    X["arr_apt_pre1800"] = _count_window(t, apt, a_land, a_apt, -1800, 0)
    X["arr_apt_post1800"] = _count_window(t, apt, a_land, a_apt, 0, 1800)
    X["arr_inblk_pre1800"] = _count_window(t, apt, a_blk, a_apt, -1800, 0)
    mean_ti = np.full(len(t), np.nan)
    a_codes, a_uniq = pd.factorize(a_apt)
    a_lut = {k: i for i, k in enumerate(a_uniq)}
    for idx in _groups(apt):
        k = a_lut.get(apt[idx[0]])
        if k is None:
            continue
        sel = np.flatnonzero((a_codes == k) & np.isfinite(a_land) & np.isfinite(a_taxi))
        o = sel[np.argsort(a_land[sel], kind="stable")]
        st, sv = a_land[o], a_taxi[o]
        cs = np.concatenate([[0.0], np.cumsum(sv)])
        hi = np.searchsorted(st, t[idx], "right")
        lo_ = np.searchsorted(st, t[idx] - 3600, "right")
        n = hi - lo_
        mean_ti[idx] = np.where(n > 0, (cs[hi] - cs[lo_]) / np.maximum(n, 1), np.nan)
    X["arr_taxiin_mean3600"] = mean_ti
    # the stand: the last arrival in-block at or before take-off, the next one after, and
    # the previous other departure from the same stand
    a_stand = np.char.add(np.char.add(a_apt.astype("U8"), ":"),
                          arr["STAND_mvt"].astype(str).to_numpy().astype("U32"))
    a_type = arr["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()
    own_type = X["ac_type"].to_numpy()
    sp, sn = np.full(len(t), np.nan), np.full(len(t), np.nan)
    same = np.full(len(t), np.nan)
    s_codes, s_uniq = pd.factorize(a_stand)
    s_lut = {k: i for i, k in enumerate(s_uniq)}
    s_rows = pd.Series(np.arange(len(a_stand))).groupby(s_codes).indices
    for idx in _groups(stand):
        k = s_lut.get(stand[idx[0]])
        if k is None:
            continue
        sel = s_rows[k]
        sel = sel[np.isfinite(a_blk[sel])]
        if not len(sel):
            continue
        o = sel[np.argsort(a_blk[sel], kind="stable")]
        st = a_blk[o]
        pos = np.searchsorted(st, t[idx], "right")
        okp, okn = pos > 0, pos < len(st)
        pi = np.clip(pos - 1, 0, len(st) - 1)
        ni = np.clip(pos, 0, len(st) - 1)
        sp[idx] = np.where(okp, t[idx] - st[pi], np.nan)
        sn[idx] = np.where(okn, st[ni] - t[idx], np.nan)
        same[idx] = np.where(okp, (a_type[o][pi] == own_type[idx]).astype(float), np.nan)
    bad_stand = stand_raw.isna().to_numpy()
    sp[bad_stand], sn[bad_stand], same[bad_stand] = np.nan, np.nan, np.nan
    X["st_prev_arr"], X["st_next_arr"], X["st_prev_arr_same_type"] = sp, sn, same
    pgs, _ = _prev_next_same(t, stand, ~bad_stand)
    X["st_prev_dep"] = pgs
    return X[["MVT_ID_mvt"] + BLIND_FEATURES]


def nm_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The NM block for every departure row: the NM fields and their NM context."""
    _assert_masked(frame)
    dep = frame[frame["PHASE_mvt"].to_numpy() == "DEP"]
    t = ep(dep["MVT_TIME_UTC_mvt"])
    sch = ep(dep["SCHED_TIME_UTC_mvt"])
    aobt, lobt = ep(dep["AOBT_3_flt"]), ep(dep["LOBT_flt"])
    apt = dep["ADEP_mvt"].astype(str).to_numpy()
    rwy = np.char.add(np.char.add(apt.astype("U8"), ":"),
                      dep["RUNWAY_mvt"].astype(str).to_numpy().astype("U16"))
    X = pd.DataFrame(index=dep.index)
    X["MVT_ID_mvt"] = dep["MVT_ID_mvt"].to_numpy("float64")
    X["nm_operator"] = dep["AIRCRAFT_OPERATOR_flt"].astype(str).to_numpy()
    X["nm_market"] = dep["MARKET_SEGMENT_flt"].astype(str).to_numpy()
    X["nm_ftype"] = dep["FLIGHT_TYPE_flt"].astype(str).to_numpy()
    X["nm_wtc"] = dep["WK_TBL_CAT_flt"].astype(str).to_numpy()
    X["to_aobt"] = t - aobt
    X["to_lobt"] = t - lobt
    X["to_iobt"] = t - ep(dep["IOBT_flt"])
    X["to_eobt1"] = t - ep(dep["EOBT_1_flt"])
    X["to_arvt1"] = t - ep(dep["ARVT_1_flt"])
    X["to_arvt3"] = t - ep(dep["ARVT_3_flt"])
    X["aobt_lobt"] = aobt - lobt
    X["aobt_sched"] = aobt - sch
    fl = dep["ADES_FILED_flt"].astype(str).to_numpy()
    de = dep["ADES_flt"].astype(str).to_numpy()
    X["nm_ades_filed_differs"] = np.where(dep["ADES_FILED_flt"].isna().to_numpy(), np.nan,
                                          (fl != de).astype(float))
    # NM-view queue at pushback: others off-block before this AOBT and not yet airborne
    taxiing = np.full(len(t), np.nan)
    since = np.full(len(t), np.nan)
    for idx in _groups(apt):
        both = idx[np.isfinite(aobt[idx]) & np.isfinite(t[idx])]
        a_s = np.sort(aobt[both])            # off-block and take-off over ONE set of rows,
        tb_s = np.sort(t[both])              # so the difference counts rows still taxiing
        t_s = np.sort(t[idx][np.isfinite(t[idx])])
        q = aobt[idx]
        n_off = np.searchsorted(a_s, q, "left")
        n_air = np.searchsorted(tb_s, q, "right")
        taxiing[idx] = np.maximum(n_off - n_air, 0)
        since[idx] = np.maximum(np.searchsorted(t_s, t[idx], "left")
                                - np.searchsorted(t_s, q, "right"), 0)
    ok = np.isfinite(aobt)
    taxiing[~ok], since[~ok] = np.nan, np.nan
    X["nm_taxiing_at_aobt"], X["nm_to_since_aobt"] = taxiing, since
    # the previous / next same-runway departure's own take-off minus AOBT
    span = t - aobt
    bp, bn = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for idx in _groups(rwy):
        o = idx[np.argsort(t[idx], kind="stable")]
        sv = span[o]
        bp[o[1:]] = sv[:-1]
        bn[o[:-1]] = sv[1:]
    X["brk_prev_span"], X["brk_next_span"] = bp, bn
    return X[["MVT_ID_mvt"] + NM_FEATURES]


def month_frame(month: str) -> tuple[pd.DataFrame, pd.Series]:
    """The masked month frame and its hidden departure labels."""
    raw = D.load_month(month)
    masked, y = D.mask_like_ranking(raw)
    return masked, y


def build_month(month: str) -> pd.DataFrame:
    """Both blocks, the label, the regime and the label hygiene flag, per departure."""
    masked, y = month_frame(month)
    B = blind_features(masked)
    N = nm_features(masked)
    if not np.array_equal(B.MVT_ID_mvt.to_numpy(), N.MVT_ID_mvt.to_numpy()):
        raise SystemExit(f"{month}: the two blocks are not in one row order")
    out = pd.concat([B, N.drop(columns="MVT_ID_mvt")], axis=1)
    dep = masked[masked["PHASE_mvt"] == "DEP"]
    out["y"] = y.loc[dep.index].to_numpy("float64")
    out["regime"] = PB.regimes_from_raw(dep)
    out["keep_reachable"] = D.keep_reachable(dep).to_numpy(bool)
    out["month"] = month
    return out.reset_index(drop=True)


def feat_path(month: str) -> Path:
    return WORK / f"feat-{month}.parquet"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    rec = {"blind_whitelist": BLIND_WHITELIST, "nm_inputs_excluded_from_blind": NM_INPUTS,
           "blind_features": BLIND_FEATURES, "nm_features": NM_FEATURES,
           "code_sha256": sha_file(Path(__file__)), "months": {}}
    for m in D.MONTHS:
        F = build_month(m)
        p = feat_path(m)
        F.to_parquet(p, index=False)
        rec["months"][m] = {"rows": int(len(F)),
                            "regime_rows": {k: int(v) for k, v in
                                            F.regime.value_counts().items()},
                            "month_file_sha256": sha_file(D.month_file(m)),
                            "parquet": str(p.relative_to(ROOT)),
                            "parquet_sha256": sha_file(p)}
        print(m, rec["months"][m]["rows"], f"({time.time() - t0:.0f}s)", flush=True)
    (EXP / "analysis/features.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
    print("written analysis/features.json", f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
