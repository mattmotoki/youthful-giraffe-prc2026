"""Stand-sequence and apron context features (label-free), for all 2025 departures and the 2026 ranking.

Each departure: the last arrivals at the same stand before take-off (in-block gaps, same type flags),
the previous departure from the same stand, apron (stand prefix) and runway counts around the NM off-block
and the take-off. Departure block times are never read.

    .venv/bin/python w2/feat2.py   -> work/w2/feat2_2025.parquet, work/w2/feat2_2026.parquet
"""
from __future__ import annotations

import glob
from pathlib import Path

import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "iso"))
import finalmode as FM  # noqa: E402
RAW = ROOT / "input-root/data/raw"
ID = "MVT_ID_mvt"
T0 = pd.Timestamp("2025-01-01", tz="UTC")
COLS = [ID, "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "AOBT_3_flt", "LOBT_flt"]


def sec(s):
    return (s - T0).dt.total_seconds().to_numpy()


def count_in(tsorted, lo, hi):
    return np.searchsorted(tsorted, hi, "right") - np.searchsorted(tsorted, lo, "left")


def build(f: pd.DataFrame) -> pd.DataFrame:
    f = f[COLS].copy()
    dep = f[f.PHASE_mvt.eq("DEP")].copy()
    arr = f[f.PHASE_mvt.eq("ARR")].copy()
    dep["ap"] = dep.ADEP_mvt.astype(str); arr["ap"] = arr.ADES_mvt.astype(str)
    dep["st"] = dep.ap + "|" + dep.STAND_mvt.astype(str); arr["st"] = arr.ap + "|" + arr.STAND_mvt.astype(str)
    dep["tt_"] = sec(dep.MVT_TIME_UTC_mvt); dep["oo_"] = sec(dep.AOBT_3_flt); dep["ss_"] = sec(dep.SCHED_TIME_UTC_mvt)
    arr["IB"] = sec(arr.BLOCK_TIME_UTC_mvt); arr["AT"] = sec(arr.MVT_TIME_UTC_mvt)
    out = pd.DataFrame({ID: dep[ID].to_numpy("int64")}, index=dep.index)
    n = len(dep)
    # --- stand sequence: last two arrivals at the stand with in-block before take-off
    a = arr[arr.STAND_mvt.notna() & np.isfinite(arr.IB)][["st", "IB", "AIRCRAFT_TYPE_mvt", "SCHED_TIME_UTC_mvt"]]
    a = a.sort_values(["st", "IB"])
    res = {k: np.full(n, np.nan) for k in ("last_arr_gap", "last_arr_same", "prev_arr_gap", "prev_arr_same",
                                            "last_arr_sched_gap", "prev_dep_gap", "next_dep_gap")}
    dst = dep.st.to_numpy(); dT = dep["tt_"].to_numpy(); dS = dep["ss_"].to_numpy(); dty = dep.AIRCRAFT_TYPE_mvt.astype(str).to_numpy()
    ag = {k: v for k, v in a.groupby("st")}
    dg = {k: v for k, v in dep[dep.STAND_mvt.notna()][["st", "tt_"]].sort_values("tt_").groupby("st")}
    pos = pd.Series(np.arange(n), index=dep.index)
    for s, idx in dep[dep.STAND_mvt.notna()].groupby("st").groups.items():
        rows = pos.loc[idx].to_numpy()
        t = dT[rows]
        if s in ag:
            g = ag[s]; ib = g.IB.to_numpy(); ty = g.AIRCRAFT_TYPE_mvt.astype(str).to_numpy()
            sch = (g.SCHED_TIME_UTC_mvt - T0).dt.total_seconds().to_numpy()
            k = np.searchsorted(ib, t, "left") - 1
            ok = k >= 0
            res["last_arr_gap"][rows[ok]] = t[ok] - ib[k[ok]]
            res["last_arr_same"][rows[ok]] = (ty[k[ok]] == dty[rows[ok]]).astype(float)
            res["last_arr_sched_gap"][rows[ok]] = dS[rows[ok]] - sch[k[ok]]
            ok2 = k >= 1
            res["prev_arr_gap"][rows[ok2]] = t[ok2] - ib[k[ok2] - 1]
            res["prev_arr_same"][rows[ok2]] = (ty[k[ok2] - 1] == dty[rows[ok2]]).astype(float)
        gd = dg[s]["tt_"].to_numpy()
        j = np.searchsorted(gd, t, "left")
        okp = j >= 1
        res["prev_dep_gap"][rows[okp]] = t[okp] - gd[j[okp] - 1]
        jn = np.searchsorted(gd, t, "right")
        okn = jn < len(gd)
        res["next_dep_gap"][rows[okn]] = gd[jn[okn]] - t[okn]
    for k, v in res.items():
        out[k] = v
    # --- apron (stand prefix) and runway counts
    dep["apron"] = dep.ap + "|" + dep.STAND_mvt.astype(str).str[:1]
    dep["rw"] = dep.ap + "|" + dep.RUNWAY_mvt.astype(str)
    arr["rw"] = arr.ap + "|" + arr.RUNWAY_mvt.astype(str)
    for name in ("apron_dep_T_1800b", "apron_O_600", "apron_O_1800b", "rw_dep_T_900b", "rw_arr_T_600", "ap_O_600",
                 "ap_O_1800b", "ap_arr_T_900b"):
        out[name] = np.nan
    O = dep["oo_"].to_numpy()
    for key, grpcol in (("apron", "apron"), ("rw", "rw"), ("ap", "ap")):
        for g, idx in dep.groupby(grpcol).groups.items():
            rows = pos.loc[idx].to_numpy()
            tt = np.sort(dT[rows]); oo = np.sort(O[rows][np.isfinite(O[rows])])
            t = dT[rows]; o = O[rows]
            if key == "apron":
                out.iloc[rows, out.columns.get_loc("apron_dep_T_1800b")] = count_in(tt, t - 1800, t)
                with np.errstate(invalid="ignore"):
                    out.iloc[rows, out.columns.get_loc("apron_O_600")] = np.where(np.isfinite(o), count_in(oo, o - 600, o + 600), np.nan)
                    out.iloc[rows, out.columns.get_loc("apron_O_1800b")] = np.where(np.isfinite(o), count_in(oo, o - 1800, o), np.nan)
            elif key == "rw":
                out.iloc[rows, out.columns.get_loc("rw_dep_T_900b")] = count_in(tt, t - 900, t)
                at = np.sort(arr.loc[arr.rw.eq(g), "AT"].to_numpy())
                out.iloc[rows, out.columns.get_loc("rw_arr_T_600")] = count_in(at, t - 600, t + 600)
            else:
                with np.errstate(invalid="ignore"):
                    out.iloc[rows, out.columns.get_loc("ap_O_600")] = np.where(np.isfinite(o), count_in(oo, o - 600, o + 600), np.nan)
                    out.iloc[rows, out.columns.get_loc("ap_O_1800b")] = np.where(np.isfinite(o), count_in(oo, o - 1800, o), np.nan)
                at = np.sort(arr.loc[arr.ap.eq(g), "AT"].to_numpy())
                out.iloc[rows, out.columns.get_loc("ap_arr_T_900b")] = count_in(at, t - 900, t)
    # departures that pushed (NM) before me but took off after me (overtaken) and the reverse, airport level
    out["T_minus_O_rank"] = np.nan
    return out.reset_index(drop=True)


def main():
    (ROOT / "work/w2").mkdir(parents=True, exist_ok=True)
    parts = [pd.read_parquet(p, columns=COLS) for p in sorted(glob.glob(str(RAW / "training_*.parquet")))]
    f = pd.concat(parts, ignore_index=True)
    f.loc[f.PHASE_mvt.eq("DEP"), "BLOCK_TIME_UTC_mvt"] = pd.NaT
    b = build(f)
    b.to_parquet(ROOT / "work/w2/feat2_2025.parquet", index=False); print("2025", b.shape, flush=True)
    r = pd.read_parquet(FM.ranking_path(RAW), columns=COLS)
    r.loc[r.PHASE_mvt.eq("DEP"), "BLOCK_TIME_UTC_mvt"] = pd.NaT
    b = build(r)
    b.to_parquet(ROOT / "work/w2/feat2_2026.parquet", index=False); print("2026", b.shape)


if __name__ == "__main__":
    main()
