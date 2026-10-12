"""Label-free row and context features from the official files, aligned to the chain rows.

Fold rows: the January and July 2025 training files, with every departure block time and
taxi time blank before any context is built (as in the ranking file). Ranking rows: the
ranking file. Output: work/feat/{fold,ranking}.parquet in the row order of work/chain.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import finalmode as FM

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "input-root/data/raw"
ID = "MVT_ID_mvt"


def secs(a, b):
    return (a - b).dt.total_seconds()


def load_split(split: str) -> pd.DataFrame:
    if split.startswith("training_"):
        f = pd.read_parquet(RAW / f"{split}.parquet")
    elif split == "ranking":
        f = pd.read_parquet(FM.ranking_path(RAW))
    else:
        f = pd.concat([pd.read_parquet(RAW / "training_2025-01-01_2025-02-01.parquet"),
                       pd.read_parquet(RAW / "training_2025-07-01_2025-08-01.parquet")], ignore_index=True)
    f = f.copy()
    dep = f.PHASE_mvt.eq("DEP")
    f.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    f.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    return f


def window_stats(t: np.ndarray, v: np.ndarray, half: float):
    """For sorted times t: count and mean of finite v in [t-half, t+half] (row itself included)."""
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    fin = np.isfinite(v)
    cs = np.concatenate([[0.0], np.cumsum(np.where(fin, v, 0.0))])
    cn = np.concatenate([[0], np.cumsum(fin)])
    n = cn[hi] - cn[lo]
    s = cs[hi] - cs[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return (hi - lo).astype(float), mean


def build(split: str) -> pd.DataFrame:
    f = load_split(split)
    arr = f[f.PHASE_mvt.eq("ARR")]
    dep = f[f.PHASE_mvt.eq("DEP")].copy()
    m = dep.MVT_TIME_UTC_mvt
    out = pd.DataFrame({ID: dep[ID].to_numpy("int64")})
    out["airport"] = dep.ADEP_mvt.to_numpy()
    out["hour"] = (m.dt.hour + m.dt.minute / 60).to_numpy()
    out["dow"] = m.dt.dayofweek.to_numpy()
    out["dom"] = m.dt.day.to_numpy()
    out["nm_taxi"] = secs(m, dep.AOBT_3_flt).to_numpy()
    out["aobt_lobt"] = secs(dep.AOBT_3_flt, dep.LOBT_flt).to_numpy()
    out["mvt_sched"] = secs(m, dep.SCHED_TIME_UTC_mvt).to_numpy()
    out["mvt_eobt"] = secs(m, dep.EOBT_1_flt).to_numpy()
    out["mvt_lobt"] = secs(m, dep.LOBT_flt).to_numpy()
    out["lobt_iobt"] = secs(dep.LOBT_flt, dep.IOBT_flt).to_numpy()
    out["sched_lobt"] = secs(dep.SCHED_TIME_UTC_mvt, dep.LOBT_flt).to_numpy()
    out["arvt_change"] = secs(dep.ARVT_3_flt, dep.ARVT_1_flt).to_numpy()
    out["joined"] = dep.LOBT_flt.notna().to_numpy().astype(float)
    out["vfr"] = dep.FLIGHT_RULE_mvt.eq("V").to_numpy().astype(float)
    for c, name in (("RUNWAY_mvt", "runway"), ("STAND_mvt", "stand"), ("AIRCRAFT_TYPE_mvt", "actype"),
                    ("WK_TBL_CAT_flt", "wake"), ("MARKET_SEGMENT_flt", "segment"),
                    ("AIRCRAFT_OPERATOR_flt", "operator"), ("FLIGHT_TYPE_flt", "ftype")):
        out[name] = (dep.ADEP_mvt.astype(str) + "|" + dep[c].astype(str)).to_numpy() if c in ("RUNWAY_mvt", "STAND_mvt") else dep[c].astype(str).to_numpy()
    out["stand_prefix"] = (dep.ADEP_mvt.astype(str) + "|" + dep.STAND_mvt.astype(str).str[:1]).to_numpy()
    out["ades_region"] = dep.ADES_mvt.astype(str).str[:2].to_numpy()
    out["callsign_prefix"] = dep.FLIGHT_mvt.astype(str).str[:3].to_numpy()

    # context windows on take-off time, per airport and per airport-runway
    tsec = (m - pd.Timestamp("2025-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    nm = out["nm_taxi"].to_numpy()
    for key, cols in (("ap", ["ADEP_mvt"]), ("rw", ["ADEP_mvt", "RUNWAY_mvt"])):
        grp = dep[cols].astype(str).agg("|".join, axis=1).to_numpy()
        for half in (300, 900, 1800):
            cnt = np.full(len(dep), np.nan); mean = np.full(len(dep), np.nan)
            for g in np.unique(grp):
                idx = np.flatnonzero(grp == g)
                order = idx[np.argsort(tsec[idx], kind="stable")]
                c, mu = window_stats(tsec[order], nm[order], half)
                cnt[order] = c; mean[order] = mu
            out[f"{key}_n{half}"] = cnt
            out[f"{key}_nm{half}"] = mean
    # arrivals around take-off at the same airport
    at = (arr.MVT_TIME_UTC_mvt - pd.Timestamp("2025-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    aap = arr.ADES_mvt.astype(str).to_numpy()
    dap = dep.ADEP_mvt.astype(str).to_numpy()
    for half in (600, 1800):
        cnt = np.full(len(dep), np.nan)
        for g in np.unique(dap):
            ta = np.sort(at[aap == g]); idx = np.flatnonzero(dap == g)
            cnt[idx] = np.searchsorted(ta, tsec[idx] + half, "right") - np.searchsorted(ta, tsec[idx] - half, "left")
        out[f"arr_n{half}"] = cnt
    # airport-day state from NM taxi times (label-free)
    day = m.dt.strftime("%m-%d").to_numpy()
    key = pd.Series(dap + "|" + day)
    s = pd.Series(nm)
    g = s.groupby(key.to_numpy())
    out["day_nm_mean"] = key.map(g.mean()).to_numpy()
    out["day_nm_p90"] = key.map(g.quantile(0.9)).to_numpy()
    out["day_nm_long"] = key.map((s >= 3600).groupby(key.to_numpy()).mean()).to_numpy()
    out["day_n"] = key.map(key.value_counts()).to_numpy().astype(float)
    out["day_join"] = key.map(pd.Series(out["joined"]).groupby(key.to_numpy()).mean()).to_numpy()
    # same flight number on the previous / next UTC day at the same airport
    fl = dep.FLIGHT_mvt.astype(str).to_numpy()
    dkey = pd.Series(dap + "|" + fl)
    dd = m.dt.normalize()
    first = pd.DataFrame({"k": dkey, "d": dd.to_numpy(), "t": tsec})
    prev = first.copy(); prev["d"] = prev["d"] + pd.Timedelta(days=1)
    prev = prev.drop_duplicates(["k", "d"], keep="last").set_index(["k", "d"]).t
    idx = pd.MultiIndex.from_arrays([first.k, first.d])
    out["prev_day_same_flight_gap"] = tsec - prev.reindex(idx).to_numpy()
    # arrival-side NM data of the same flight (callsign, origin, destination), for rows without a join
    a2 = arr[["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt", "EOBT_1_flt"]]
    a2 = a2[a2.FLIGHT_mvt.notna()].rename(columns={"MVT_TIME_UTC_mvt": "arr_mvt"})
    q = pd.DataFrame({"row": np.arange(len(dep)), "FLIGHT_mvt": dep.FLIGHT_mvt.to_numpy(), "ADEP_mvt": dep.ADEP_mvt.to_numpy(),
                      "ADES_mvt": dep.ADES_mvt.to_numpy(), "mvt": m.to_numpy()})
    q = q[dep.LOBT_flt.isna().to_numpy() & q.FLIGHT_mvt.notna().to_numpy()]
    mm = q.merge(a2, on=["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"], how="inner")
    mm["fl"] = (mm.arr_mvt - mm.mvt).dt.total_seconds()
    mm = mm[(mm.fl > 0) & (mm.fl < 16 * 3600)].sort_values("fl").drop_duplicates("row")
    for name, col in (("arr_nm_taxi", "AOBT_3_flt"), ("arr_mvt_lobt", "LOBT_flt"), ("arr_mvt_eobt", "EOBT_1_flt")):
        v = np.full(len(dep), np.nan)
        v[mm.row.to_numpy()] = (mm.mvt - mm[col]).dt.total_seconds().to_numpy()
        out[name] = v
    out["arr_flight_time"] = np.nan
    out.loc[mm.row.to_numpy(), "arr_flight_time"] = mm.fl.to_numpy()
    if "TAXITIME_SEC_mvt" in dep and split.startswith("training_"):
        pass
    return out


def months():
    """Feature tables of every 2025 month file (context within the file), with the label."""
    import glob
    (ROOT / "work/feat/months").mkdir(parents=True, exist_ok=True)
    for path in sorted(glob.glob(str(RAW / "training_*.parquet"))):
        split = Path(path).stem
        feat = build(split)
        lab = pd.read_parquet(path, columns=[ID, "PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
        lab = lab[lab.PHASE_mvt.eq("DEP")].set_index(ID)
        feat["y"] = lab.TAXITIME_SEC_mvt.reindex(feat[ID].to_numpy()).to_numpy(float)
        feat["mvt_month"] = lab.MVT_TIME_UTC_mvt.reindex(feat[ID].to_numpy()).dt.month.to_numpy()
        feat["file"] = split
        feat.to_parquet(ROOT / f"work/feat/months/{split}.parquet", index=False)
        print(split, feat.shape, flush=True)


def chain_order(split: str) -> np.ndarray:
    """Row order of work/chain/<split>. The ranking chain is in template order (release/v73_stage.py checks it),
    so the ranking split can be built before the chain exists; when the chain exists, its order must agree."""
    path = ROOT / f"work/chain/{split}.parquet"
    if split != "ranking":
        return pd.read_parquet(path, columns=[ID])[ID].to_numpy()
    tmpl = pd.read_parquet(FM.submitting_path(RAW), columns=[ID])[ID].to_numpy("int64")
    assert len(tmpl) == FM.n_rows()
    if path.exists():
        assert np.array_equal(pd.read_parquet(path, columns=[ID])[ID].to_numpy("int64"), tmpl), "chain order is not the template order"
    return tmpl


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "months":
        return months()
    (ROOT / "work/feat").mkdir(parents=True, exist_ok=True)
    splits = ("ranking",) if len(sys.argv) > 1 and sys.argv[1] == "ranking" else ("fold", "ranking")
    for split in splits:
        chain = pd.DataFrame({ID: chain_order(split)})
        feat = build(split)
        feat = feat.drop_duplicates(ID).set_index(ID).reindex(chain[ID].to_numpy()).reset_index()
        assert feat["airport"].notna().all(), split
        feat.to_parquet(ROOT / f"work/feat/{split}.parquet", index=False)
        print(split, feat.shape)


if __name__ == "__main__":
    main()
