"""Inputs of the airport-context sequence member s1 (label-free except the training target).

One movement table over all 2025 files and the 2026 ranking file (departure block times blanked):
numeric token features, categorical codes, the airport-time order, and for every departure the indices of
the 12 movements before and after it at the same airport within 40 minutes. Center features: the
iso/features.py and w2/feat2.py columns of the departure.

    python3 w2/seq/prepare.py [--nb 12 --win 2400 --out work/seq]      -> <out>/*.npy, <out>/meta.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "input-root/data/raw"
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "iso"))
import finalmode as FM  # noqa: E402
OUT = ROOT / "work/seq"
ID = "MVT_ID_mvt"
T0 = pd.Timestamp("2025-01-01", tz="UTC")
NB = 12          # movements on each side
WIN = 2400.0     # seconds
TOKCAT = ["rw", "apron", "wake", "phase"]
CENTERCAT = ["airport", "runway", "stand", "actype", "wake", "segment", "operator", "ftype", "stand_prefix",
             "ades_region", "callsign_prefix"]


def secs(a, b):
    return (a - b).dt.total_seconds().to_numpy()


def movements() -> pd.DataFrame:
    parts = []
    rank = str(FM.ranking_path(RAW))
    for p in sorted(glob.glob(str(RAW / "training_*.parquet"))) + [rank]:
        f = pd.read_parquet(p)
        f["src"] = "2026" if p == rank else Path(p).stem
        parts.append(f)
    f = pd.concat(parts, ignore_index=True)
    dep = f.PHASE_mvt.eq("DEP").to_numpy()
    y = f.TAXITIME_SEC_mvt.to_numpy(float).copy(); y[~dep] = np.nan
    f.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    f["y"] = y
    f["ap"] = np.where(dep, f.ADEP_mvt.astype(str), f.ADES_mvt.astype(str))
    f["t"] = secs(f.MVT_TIME_UTC_mvt, T0)
    return f


def main():
    global NB, WIN, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=NB); ap.add_argument("--win", type=float, default=WIN)
    ap.add_argument("--stride", type=int, default=0, help="add --sparse tokens each side at every stride-th movement beyond --nb")
    ap.add_argument("--sparse", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(); NB, WIN, OUT = a.nb, a.win, ROOT / a.out if not a.out.is_absolute() else a.out
    OUT.mkdir(parents=True, exist_ok=True)
    f = movements()
    dep = f.PHASE_mvt.eq("DEP").to_numpy()
    m = f.MVT_TIME_UTC_mvt
    num = pd.DataFrame(index=f.index)
    num["is_dep"] = dep.astype(float)
    num["nm"] = np.where(dep, secs(m, f.AOBT_3_flt), np.nan) / 600
    num["aobt_lobt"] = secs(f.AOBT_3_flt, f.LOBT_flt) / 600
    num["mvt_sched"] = secs(m, f.SCHED_TIME_UTC_mvt) / 1800
    num["mvt_eobt"] = secs(m, f.EOBT_1_flt) / 1800
    num["lobt_iobt"] = secs(f.LOBT_flt, f.IOBT_flt) / 1800
    num["taxi_in"] = np.where(~dep, secs(f.BLOCK_TIME_UTC_mvt, m), np.nan) / 600
    num["arvt_change"] = secs(f.ARVT_3_flt, f.ARVT_1_flt) / 1800
    num["joined"] = f.LOBT_flt.notna().to_numpy().astype(float)
    num["typed"] = f.AIRCRAFT_TYPE_mvt.notna().to_numpy().astype(float)
    num["vfr"] = f.FLIGHT_RULE_mvt.eq("V").to_numpy().astype(float)
    for c in list(num.columns):
        if c in ("is_dep", "joined", "typed", "vfr"):
            continue
        v = num[c].to_numpy(float)
        num[c + "_ok"] = np.isfinite(v).astype(float)
        num[c] = np.clip(np.nan_to_num(v, nan=0.0), -20, 20)
    cat = pd.DataFrame(index=f.index)
    cat["rw"] = f.ap + "|" + f.RUNWAY_mvt.astype(str)
    cat["apron"] = f.ap + "|" + f.STAND_mvt.astype(str).str[:1]
    cat["wake"] = f.WK_TBL_CAT_flt.astype(str)
    cat["phase"] = f.PHASE_mvt.astype(str)
    codes = {}
    for c in TOKCAT:
        codes[c], uniq = pd.factorize(cat[c], sort=True)
        codes[c] = codes[c] + 1
    # neighbour index per departure: same airport, sorted by time
    order = np.lexsort((f.t.to_numpy(), f.ap.to_numpy()))
    ap_sorted = f.ap.to_numpy()[order]; t_sorted = f.t.to_numpy()[order]
    pos = np.empty(len(f), dtype=np.int64); pos[order] = np.arange(len(f))
    dep_idx = np.flatnonzero(dep)
    P = pos[dep_idx]
    offs = list(range(-NB, NB + 1))
    if a.stride and a.sparse:
        far = [NB + a.stride * i for i in range(1, a.sparse + 1)]
        offs = [-o for o in far[::-1]] + offs + far
    nbr = np.full((len(dep_idx), len(offs)), -1, dtype=np.int64)
    for j, off in enumerate(offs):
        q = P + off
        ok = (q >= 0) & (q < len(f))
        qq = np.clip(q, 0, len(f) - 1)
        ok &= ap_sorted[qq] == ap_sorted[P]
        ok &= np.abs(t_sorted[qq] - t_sorted[P]) <= WIN
        nbr[ok, j] = order[qq[ok]]
    dt = np.where(nbr >= 0, (f.t.to_numpy()[np.maximum(nbr, 0)] - f.t.to_numpy()[dep_idx][:, None]) / 600, 0.0)
    same_rw = np.where(nbr >= 0, codes["rw"][np.maximum(nbr, 0)] == codes["rw"][dep_idx][:, None], False)
    # center features
    import sys
    sys.path.insert(0, str(ROOT / "iso"))
    import features as F
    cen = []
    for p in sorted(glob.glob(str(ROOT / "work/feat/months/*.parquet"))):
        cen.append(pd.read_parquet(p).drop(columns=["y", "mvt_month", "file"]))
    cen.append(pd.read_parquet(ROOT / "work/feat/ranking.parquet"))
    cen = pd.concat(cen, ignore_index=True).drop_duplicates(ID)
    f2 = pd.concat([pd.read_parquet(ROOT / "work/w2/feat2_2025.parquet"),
                    pd.read_parquet(ROOT / "work/w2/feat2_2026.parquet")], ignore_index=True).drop_duplicates(ID)
    cen = cen.merge(f2.drop(columns=["T_minus_O_rank"]), on=ID, how="left")
    ids = f[ID].to_numpy("int64")[dep_idx]
    cen = cen.set_index(ID).reindex(ids)
    ccodes = np.stack([pd.factorize(cen[c].astype(str), sort=True)[0] + 1 for c in CENTERCAT], 1).astype(np.int32)
    cnum = cen.drop(columns=CENTERCAT).select_dtypes("number")
    cnames = list(cnum.columns)
    cnum = cnum.to_numpy(float)
    okc = np.isfinite(cnum)
    mu = np.nanmedian(cnum, 0); sd = np.nanpercentile(cnum, 90, 0) - np.nanpercentile(cnum, 10, 0); sd[sd <= 0] = 1
    cnum = np.clip(np.nan_to_num((cnum - mu) / sd, nan=0.0), -10, 10)
    cnum = np.concatenate([cnum, okc.astype(float)], 1).astype(np.float32)
    meta = {"src": f.src.to_numpy()[dep_idx].tolist(), }
    np.save(OUT / "tok_num.npy", num.to_numpy(np.float32))
    np.save(OUT / "tok_cat.npy", np.stack([codes[c] for c in TOKCAT], 1).astype(np.int32))
    np.save(OUT / "nbr.npy", nbr.astype(np.int32))
    np.save(OUT / "dt.npy", dt.astype(np.float32))
    np.save(OUT / "same_rw.npy", same_rw)
    np.save(OUT / "dep_idx.npy", dep_idx.astype(np.int32))
    np.save(OUT / "ids.npy", ids)
    np.save(OUT / "y.npy", f.y.to_numpy(float)[dep_idx].astype(np.float32))
    nm_raw = np.where(dep, secs(m, f.AOBT_3_flt), np.nan)[dep_idx]
    np.save(OUT / "nm_raw.npy", nm_raw.astype(np.float32))
    np.save(OUT / "month.npy", m.dt.month.to_numpy()[dep_idx].astype(np.int8))
    np.save(OUT / "dom.npy", m.dt.day.to_numpy()[dep_idx].astype(np.int8))
    np.save(OUT / "is2026.npy", (f.src.to_numpy()[dep_idx] == "2026"))
    np.save(OUT / "srcfile.npy", pd.factorize(f.src.to_numpy()[dep_idx], sort=True)[0].astype(np.int8))
    np.save(OUT / "c_num.npy", cnum); np.save(OUT / "c_cat.npy", ccodes)
    json.dump({"tok_num": list(num.columns), "tok_cat": TOKCAT, "tok_cat_sizes": [int(codes[c].max()) + 1 for c in TOKCAT],
               "c_num": cnames, "c_cat": CENTERCAT, "c_cat_sizes": [int(ccodes[:, j].max()) + 1 for j in range(len(CENTERCAT))],
               "srcfiles": sorted(set(f.src)), "n_dep": int(len(dep_idx)), "NB": NB, "WIN": WIN},
              open(OUT / "meta.json", "w"), indent=1)
    print("done", len(dep_idx), num.shape, cnum.shape)


if __name__ == "__main__":
    main()
