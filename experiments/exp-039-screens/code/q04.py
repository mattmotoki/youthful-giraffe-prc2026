"""Q04 -- do arrivals reveal the departure errors v10 still makes?

The queue's first test.  Ordinary joined rows and the v10 serving mask are frozen
first; `h0(W)` carries the current arrival, runway, clock and window summaries
AND the v10 prediction; `h1(W, z)` is the same learner at the same capacity with
ONE thing added: the arrival innovation.

The innovation is `taxi_in - g(arrival inputs)`, where `g` is a nuisance model
fitted on the OTHER 2025 months only -- never the month the correction is scored
on -- from airport, runway, stand pier, wake category, hour and day of week.  No
in-block clock and no departure label enters `g`, so `z` is the part of the
arrival stream the current summaries do not already carry.

Three arms at identical capacity, splits and seed:
    h0    current summaries + v10
    h1    h0 + the innovation of the departure's own airport x 10-minute cell
    h1c   h0 + a TIME-MATCHED innovation: the same airport and time-of-day bin
          seven days away.  A specificity check, not a randomisation test.

Readouts: the coverage the queue predeclares (>= 6 departures and >= 6 arrivals
per cell, 200 supported cells over 20 dates before any airport-specific claim),
the split-half sensor agreement of `z`, the self-excluded cross-products, and
`2A - B` for each arm against v10 on held-out day blocks.

    python q04.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import v10 as V  # noqa: E402
import rowfeat as R  # noqa: E402
import donor as Dn  # noqa: E402
import packet as PK  # noqa: E402

ROOT = V.ROOT
FOLDS = V.FOLDS
D = V.D

# ---- predeclared -------------------------------------------------------------
BIN = 600
MIN_DEP = 6
MIN_ARR = 6
SUPPORT_CELLS = 200        # before any airport-specific claim
SUPPORT_DATES = 20
BLOCKS = 5
SEED = 39
BOOT = 2000
TREES, DEPTH, LR = 64, 2, 0.05
SHRINK = 50.0              # nuisance arrival model, empirical-Bayes constant
HOLD_AIRPORTS = ("EHAM", "EDDF")
ELIGIBLE_REGIME = "nm_bulk"
TIME_MATCH_DAYS = 7
ARR_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MVT_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]


def arrivals(month: str) -> pd.DataFrame:
    """Arrival rows of one month.  `TAXITIME_SEC_mvt` on an ARR row is the taxi-IN
    time and the ranking mask does not blank it, so it is an input."""
    raw = D.load_month(month, columns=ARR_COLS)
    a = raw[raw["PHASE_mvt"] == "ARR"].reset_index(drop=True)
    del raw
    out = pd.DataFrame({
        "mid": a["MVT_ID_mvt"].to_numpy("float64"),
        "apt": a["ADES_mvt"].astype("string").fillna("<na>").to_numpy(),
        "rwy": a["RUNWAY_mvt"].astype("string").fillna("<na>").to_numpy(),
        "pier": a["STAND_mvt"].astype("string").fillna("").str.slice(0, 2).to_numpy(),
        "wtc": a["WK_TBL_CAT_flt"].astype("string").fillna("<na>").to_numpy(),
        "t": Dn.epoch(a["MVT_TIME_UTC_mvt"]),
        "sched": Dn.epoch(a["SCHED_TIME_UTC_mvt"]),
        "taxi_in": a["TAXITIME_SEC_mvt"].to_numpy("float64")})
    out["hour"] = np.floor(np.mod(out["t"], 86400) / 3600)
    out["dow"] = np.floor(out["t"] / 86400) % 7
    return out[np.isfinite(out["taxi_in"])].reset_index(drop=True)


#: The shipped current feature block, built by `experiments/exp-018-fresh/code/
#: features.py` and `window_features.py` from one month's frame alone.  Its own
#: contract (that file's docstring) is this experiment's contract: departure
#: `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are never read, arrival rows keep
#: every column.  Sol round 2, blocking finding b1: the first `h0` was 19
#: hand-built columns, which is matched LEARNER capacity but not the queue's
#: "all current arrival, runway, clock and window summaries".  `h0` now carries
#: the shipped block itself.
W_CACHE = ROOT / "artifacts" / "exp-018-fresh" / "cache"
W_DROP = ("y", "month", "MVT_ID_mvt")


def shipped_W(month: str) -> pd.DataFrame:
    """The shipped current feature block for one month, keyed by movement id."""
    f = W_CACHE / f"{month}.parquet"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} is missing; build it with experiments/exp-018-fresh/code/features.py")
    w = pd.read_parquet(f)
    for c in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "TAXITIME_SEC", "y_true"):
        if c in w.columns:
            raise ValueError(f"{c} is present in the shipped feature block")
    w = w.set_index(w["MVT_ID_mvt"].to_numpy("float64"))
    keep = [c for c in w.columns if c not in W_DROP
            and pd.api.types.is_numeric_dtype(w[c])]
    return w[keep].astype("float64")


def dep_inputs(month: str) -> pd.DataFrame:
    """The departure-side current inputs `h0` carries, from the MASKED frame."""
    dep = R.departures(month)
    out = pd.DataFrame({
        "mid": dep["MVT_ID_mvt"].to_numpy("float64"),
        "rwy": dep["RUNWAY_mvt"].astype("string").fillna("<na>").to_numpy(),
        "pier": dep["STAND_mvt"].astype("string").fillna("").str.slice(0, 2).to_numpy(),
        "typ": dep["AIRCRAFT_TYPE_mvt"].astype("string").fillna("<na>").to_numpy(),
        "wtc": dep["WK_TBL_CAT_flt"].astype("string").fillna("<na>").to_numpy(),
        "seg": dep["MARKET_SEGMENT_flt"].astype("string").fillna("<na>").to_numpy(),
        "t_sched": (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]
                    ).dt.total_seconds().to_numpy("float64"),
        "t_eobt": (dep["MVT_TIME_UTC_mvt"] - dep["EOBT_1_flt"]
                   ).dt.total_seconds().to_numpy("float64")})
    return out


class Nuisance:
    """Shrunk group-mean taxi-in model with backoff, fitted on months only."""

    #: Backoff order, deepest first.  Every field is an arrival-side INPUT; no
    #: in-block clock and no departure label appears here.  Sol round 1, blocking
    #: finding 5: the first version's docstring named wake category and day of
    #: week and the levels used neither.  They are in the levels now.
    LEVELS = (("apt", "rwy", "pier", "wtc"), ("apt", "rwy", "pier"),
              ("apt", "rwy", "hour"), ("apt", "rwy"), ("apt", "hour"), ("apt",))

    def fit(self, months: list[str]) -> "Nuisance":
        tot = {lv: None for lv in self.LEVELS}
        self.g = 0.0
        n = 0
        for mth in months:
            a = arrivals(mth)
            self.g += float(a["taxi_in"].sum())
            n += len(a)
            for lv in self.LEVELS:
                s = a.groupby(list(lv))["taxi_in"].agg(["sum", "size"])
                tot[lv] = s if tot[lv] is None else tot[lv].add(s, fill_value=0)
            del a
        self.g /= max(n, 1)
        self.tab = {}
        for lv in self.LEVELS:
            t = tot[lv]
            self.tab[lv] = (t["sum"] / t["size"]).where(t["size"] >= 30)
            self.tab[lv].name = "m"
        self.n_train = n
        return self

    def predict(self, a: pd.DataFrame) -> np.ndarray:
        out = np.full(len(a), np.nan)
        for lv in self.LEVELS:
            idx = pd.MultiIndex.from_arrays([a[c].to_numpy() for c in lv]) \
                if len(lv) > 1 else pd.Index(a[lv[0]].to_numpy())
            v = self.tab[lv].reindex(idx).to_numpy()
            out = np.where(np.isnan(out), v, out)
        return np.where(np.isnan(out), self.g, out)


def cell_stats(a: pd.DataFrame, inn: np.ndarray) -> pd.DataFrame:
    """Per airport x 10-minute cell: counts, the raw summary and the innovation."""
    b = np.floor(a["t"].to_numpy() / BIN).astype("int64")
    h = (a["mid"].to_numpy("int64") % 2).astype(int)     # fixed hash split-half
    df = pd.DataFrame({"apt": a["apt"].to_numpy(), "bin": b, "inn": inn,
                       "taxi_in": a["taxi_in"].to_numpy(),
                       "late": a["t"].to_numpy() - a["sched"].to_numpy(), "half": h})
    g = df.groupby(["apt", "bin"])
    out = g.agg(n_arr=("inn", "size"), z=("inn", "mean"),
                arr_taxi_mean=("taxi_in", "mean"), arr_late_mean=("late", "mean"))
    half = df.groupby(["apt", "bin", "half"])["inn"].agg(["mean", "size"]).unstack("half")
    out["z_a"] = half[("mean", 0)]
    out["z_b"] = half[("mean", 1)]
    out["n_a"] = half[("size", 0)]
    out["n_b"] = half[("size", 1)]
    return out


def wide_window(cs: pd.DataFrame, bins: int = 4) -> pd.DataFrame:
    """Rolling sums over EXPLICIT consecutive time bins, empty bins included.

    Sol round 1, blocking finding 6: the first version rolled over four POPULATED
    cells, which is not a fixed-width window.  Here each airport's bins are
    reindexed onto a complete grid first, so four bins is always 40 minutes.
    """
    out = []
    for apt, d in cs.groupby(level=0):
        b = d.index.get_level_values(1)
        grid = pd.RangeIndex(int(b.min()), int(b.max()) + 1)
        g = d.reset_index(level=0, drop=True).reindex(grid).fillna(
            {"n_arr": 0.0, "z": 0.0, "arr_taxi_mean": 0.0})
        zs = (g["z"] * g["n_arr"]).rolling(bins, min_periods=1).sum()
        ts = (g["arr_taxi_mean"] * g["n_arr"]).rolling(bins, min_periods=1).sum()
        na = g["n_arr"].rolling(bins, min_periods=1).sum()
        w = pd.DataFrame({"z_wide": zs / na.replace(0, np.nan),
                          "taxi_wide": ts / na.replace(0, np.nan),
                          "n_arr_wide": na})
        w.index = pd.MultiIndex.from_product([[apt], grid], names=["apt", "bin"])
        out.append(w)
    return pd.concat(out)


def blocks_of(days: np.ndarray) -> np.ndarray:
    u = np.unique(days)
    idx = {d: b for b, chunk in enumerate(np.array_split(u, BLOCKS)) for d in chunk}
    return np.array([idx[d] for d in days])


def boot_days(v: np.ndarray, days: np.ndarray, rng):
    codes, uniq = pd.factorize(days)
    k = len(uniq)
    num = np.bincount(codes, weights=v, minlength=k)
    cnt = np.bincount(codes, minlength=k).astype("float64")
    draws = rng.integers(0, k, size=(BOOT, k))
    b = num[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(num.sum() / cnt.sum()), float(lo), float(hi)


def run_fold(fold: str, panel: pd.DataFrame, rng, preds_out: dict,
             preds_matched: dict) -> dict:
    month = R.month_of(fold)
    others = [m for m in D.MONTHS if m != month]
    nu = Nuisance().fit(others)
    a = arrivals(month)
    inn = a["taxi_in"].to_numpy() - nu.predict(a)
    cs = cell_stats(a, inn)

    W4 = wide_window(cs)
    apt = panel["airport"].astype(str).to_numpy()
    t = panel["t"].to_numpy("float64")
    b = np.floor(t / BIN).astype("int64")
    key = pd.MultiIndex.from_arrays([apt, b], names=["apt", "bin"])
    elig = ((panel["regime"].to_numpy() == ELIGIBLE_REGIME)
            & ~np.isin(apt, HOLD_AIRPORTS))
    # Sol round 1, blocking finding 5: the departure count is over the FROZEN
    # eligible cohort, not over every regime.
    ek = pd.MultiIndex.from_arrays([apt[elig], b[elig]], names=["apt", "bin"])
    dep_cnt = pd.Series(1, index=ek).groupby(level=[0, 1]).size()
    n_dep = dep_cnt.reindex(key).fillna(0).to_numpy()
    C = cs.reindex(key)
    n_arr = C["n_arr"].fillna(0).to_numpy()
    z = C["z"].to_numpy()
    zprev = W4["z_wide"].reindex(key).to_numpy()

    # The time-matched control shifts the WHOLE innovation block seven days and
    # demands the same support there, so it can carry no current information
    # (Sol round 1, blocking finding 6).
    shift = TIME_MATCH_DAYS * 86400 // BIN
    key_tm = pd.MultiIndex.from_arrays([apt, b + shift], names=["apt", "bin"])
    Ctm = cs.reindex(key_tm)
    n_arr_tm = Ctm["n_arr"].fillna(0).to_numpy()
    n_dep_tm = dep_cnt.reindex(key_tm).fillna(0).to_numpy()
    # Sol round 2, blocking finding b2: both shifted columns are gated on the
    # SHIFTED cell's own declared support, and the shift is +7 days (LATER).
    tm_sup = (n_arr_tm >= MIN_ARR) & (n_dep_tm >= MIN_DEP)
    z_tm = np.where(tm_sup, Ctm["z"].to_numpy(), np.nan)
    zprev_tm = np.where(tm_sup, W4["z_wide"].reindex(key_tm).to_numpy(), np.nan)
    supported = elig & (n_dep >= MIN_DEP) & (n_arr >= MIN_ARR)
    days = V.day(t)
    blk = blocks_of(days)
    r = panel["r"].to_numpy("float64")
    y = panel["y_true"].to_numpy("float64")
    ids = panel["MVT_ID_mvt"].to_numpy("float64")

    out = {"fold": fold, "month": month,
           "nuisance_train_arrivals": int(nu.n_train),
           "nuisance_months": len(others),
           "n_eligible": int(elig.sum()), "n_supported": int(supported.sum()),
           "supported_pct_of_eligible": round(100 * float(supported.sum()
                                                          / max(elig.sum(), 1)), 2)}
    # the census counts only cells the screen can score: eligible departures,
    # held airports excluded
    sup_cells = cs[(cs["n_arr"] >= MIN_ARR)
                   & ~cs.index.get_level_values(0).isin(HOLD_AIRPORTS)]
    both = sup_cells.join(dep_cnt.rename("n_dep"), how="inner")
    both = both[both["n_dep"] >= MIN_DEP]
    out["supported_cells"] = int(len(both))
    out["supported_dates"] = int(pd.Series(
        np.floor(both.index.get_level_values(1) * BIN / 86400)).nunique())
    out["support_rule_met"] = bool(out["supported_cells"] >= SUPPORT_CELLS
                                   and out["supported_dates"] >= SUPPORT_DATES)
    out["cells_per_airport"] = both.groupby(level=0).size().to_dict()

    # ---- split-half sensor agreement ---------------------------------------
    h = cs[(cs["n_a"] >= 3) & (cs["n_b"] >= 3)]
    out["split_half"] = {
        "n_cells": int(len(h)),
        "corr": round(float(np.corrcoef(h["z_a"], h["z_b"])[0, 1]), 4) if len(h) > 30
        else None,
        "note": "each half is centred by the OFF-MONTH nuisance model, never by "
                "its own stream mean"}

    # ---- cross-products -----------------------------------------------------
    m = supported & np.isfinite(z)
    pt, lo, hi = boot_days(r[m] * z[m], days[m], rng)
    out["cross_product_r_z"] = {"n": int(m.sum()), "mean": round(pt, 2),
                                "ci": [round(lo, 2), round(hi, 2)],
                                "corr": round(float(np.corrcoef(r[m], z[m])[0, 1]), 4)}
    # self-excluded departure cell mean, for context
    code, uniq = pd.factorize(pd.MultiIndex.from_arrays([apt[m], b[m]]))
    s = np.bincount(code, weights=r[m], minlength=len(uniq))
    c = np.bincount(code, minlength=len(uniq)).astype("float64")
    rbar = (s[code] - r[m]) / np.maximum(c[code] - 1, 1)
    pt2, lo2, hi2 = boot_days(r[m] * rbar, days[m], rng)
    out["cross_product_r_rbar_self_excluded"] = {
        "mean": round(pt2, 2), "ci": [round(lo2, 2), round(hi2, 2)]}
    # The repair the queue names before more capacity: reliability shrinkage and
    # a restriction to the cells where the sensor is best measured.
    out["reliability_repair"] = {}
    out["n_arr_distribution"] = {
        f"p{q}": float(np.percentile(n_arr[elig], q)) for q in (50, 75, 90, 99)}
    for k in (8, 10, 12, 15):
        mm = supported & np.isfinite(z) & (n_arr >= k)
        if mm.sum() < 200:
            continue
        p3, l3, h3 = boot_days(r[mm] * z[mm], days[mm], rng)
        out["reliability_repair"][f"n_arr_ge_{k}"] = {
            "n": int(mm.sum()), "mean_r_z": round(p3, 2),
            "ci": [round(l3, 2), round(h3, 2)],
            "corr": round(float(np.corrcoef(r[mm], z[mm])[0, 1]), 4)}

    # ---- h0 / h1 / h1c ------------------------------------------------------
    di = dep_inputs(month).set_index("mid").reindex(
        panel["MVT_ID_mvt"].to_numpy("float64"))
    cat = lambda c: pd.factorize(di[c].to_numpy())[0].astype("float64")
    Wsh = shipped_W(month).reindex(panel["MVT_ID_mvt"].to_numpy("float64"))
    out["W_columns"] = int(Wsh.shape[1])
    out["W_cache_sha256"] = PK.sha_file(W_CACHE / f"{month}.parquet")
    out["W_missing_rows"] = int(Wsh.isna().all(axis=1).sum())
    base = np.column_stack([
        panel["p10"].to_numpy("float64"), panel["d_takeoff_aobt"].to_numpy("float64"),
        panel["d_aobt_lobt"].to_numpy("float64"),
        np.nan_to_num(di["t_sched"].to_numpy(), nan=-1e5),
        np.nan_to_num(di["t_eobt"].to_numpy(), nan=-1e5),
        np.mod(t, 86400) / 3600.0, np.floor(t / 86400) % 7,
        pd.factorize(apt)[0].astype("float64"),
        cat("rwy"), cat("pier"), cat("typ"), cat("wtc"), cat("seg"),
        n_dep.astype("float64"), n_arr,
        np.nan_to_num(W4["n_arr_wide"].reindex(key).to_numpy(), nan=0.0),
        np.nan_to_num(C["arr_taxi_mean"].to_numpy(), nan=-1.0),
        np.nan_to_num(C["arr_late_mean"].to_numpy(), nan=-1e5),
        np.nan_to_num(W4["taxi_wide"].reindex(key).to_numpy(), nan=-1.0),
        np.nan_to_num(Wsh.to_numpy("float64"), nan=-1e5,
                      posinf=1e9, neginf=-1e9)])
    z_sh = np.nan_to_num(z, nan=0.0) * (n_arr / (n_arr + 10.0))
    zp_sh = np.nan_to_num(zprev, nan=0.0) * (n_arr / (n_arr + 10.0))
    arms = {"h0": base,
            "h1": np.column_stack([base, np.nan_to_num(z, nan=0.0),
                                   np.nan_to_num(zprev, nan=0.0)]),
            "h1c": np.column_stack([base, np.nan_to_num(z_tm, nan=0.0),
                                    np.nan_to_num(zprev_tm, nan=0.0)]),
            "h1_shrunk": np.column_stack([base, z_sh, zp_sh])}
    supported_both = supported & tm_sup
    out["n_supported_matched"] = int(supported_both.sum())
    out["matched_pct_of_supported"] = round(
        100 * float(supported_both.sum() / max(supported.sum(), 1)), 2)
    out["time_match_shift_days"] = TIME_MATCH_DAYS
    out["time_match_direction"] = "later"

    def run_arms(si: np.ndarray, tag: str, keep_preds: bool) -> dict:
        """Fit and score every arm on ONE row support, so the arms are comparable."""
        res = {}
        for name, X in arms.items():
            hpred = np.zeros(len(panel))
            for bb in range(BLOCKS):
                fit = si[blk[si] != bb]
                sc = si[blk[si] == bb]
                if not len(fit) or not len(sc):
                    continue          # a day block with no row on this support
                g = GradientBoostingRegressor(n_estimators=TREES, max_depth=DEPTH,
                                              learning_rate=LR, random_state=SEED)
                g.fit(X[fit], r[fit])
                hpred[sc] = g.predict(X[sc])
            p_new = panel["p10"].to_numpy("float64") + hpred
            v = 2 * r[si] * hpred[si] - hpred[si] ** 2
            pt, lo, hi = boot_days(v, days[si], rng)
            A = float((r[si] * hpred[si]).mean()); Bq = float((hpred[si] ** 2).mean())
            ex = {"all_v10": float(D.rmse(y, panel["p10"])),
                  "all_new": float(D.rmse(y, p_new)),
                  "clean_v10": float(D.rmse_clean(y, panel["p10"], ids, fold)),
                  "clean_new": float(D.rmse_clean(y, p_new, ids, fold))}
            if keep_preds:
                preds_out.setdefault(name, {})[fold] = p_new
                PK.dates("q04", name, fold, days, y,
                         panel["p10"].to_numpy("float64"), p_new)
                if name in ("h0", "h1"):
                    PK.rows("q04", name, fold, ids, days, y,
                            panel["p10"].to_numpy("float64"), p_new)
            else:
                preds_matched.setdefault(name, {})[fold] = p_new
            res[name] = {
                "n_features": int(X.shape[1]), "n_scored": int(len(si)),
                "rmse_exact": ex,
                "rmse_all_v10": round(ex["all_v10"], 4),
                "rmse_all_new": round(ex["all_new"], 4),
                "multiday": PK.boot_blocks(v, days[si], rng),
                "A_mean_r_h": round(A, 3), "B_mean_h2": round(Bq, 3),
                "mse_gain_s2": round(pt, 3), "ci": [round(lo, 3), round(hi, 3)],
                "unrestricted_scale": round(A / Bq, 4) if Bq > 0 else None,
                "rmse_clean_v10": round(ex["clean_v10"], 4),
                "rmse_clean_new": round(ex["clean_new"], 4),
                "gain_clean": round(ex["clean_v10"] - ex["clean_new"], 4)}
        for k in arms:
            if k != "h0":
                res[k]["increment_over_h0_s2"] = round(
                    res[k]["mse_gain_s2"] - res["h0"]["mse_gain_s2"], 3)
        return res

    #: `arms` is scored on the CURRENT declared support (the screen's primary
    #: comparison, h1 against h0) and again on the INTERSECTION of the current and
    #: the 7-day-later declared support, which is the only support on which the
    #: placebo `h1c` carries a real value on every row it is scored on.
    out["arms"] = run_arms(np.flatnonzero(supported), "current", True)
    out["arms_matched"] = run_arms(np.flatnonzero(supported_both), "matched", False)
    return out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    pan = V.load()
    recs, preds_out, preds_matched = {}, {}, {}
    for f in FOLDS:
        recs[f] = run_fold(f, pan[f], rng, preds_out, preds_matched)
        c = recs[f]
        print(f"[{f}] supported {c['n_supported']:,} of {c['n_eligible']:,} eligible "
              f"({c['supported_pct_of_eligible']} %); {c['supported_cells']:,} cells "
              f"over {c['supported_dates']} dates, rule met {c['support_rule_met']}",
              flush=True)
        if c["reliability_repair"]:
            print("    repair (reliability): " + "; ".join(
                f"{k} n={v['n']:,} r*z {v['mean_r_z']:+.1f} {v['ci']}"
                for k, v in c["reliability_repair"].items()))
        print(f"    split-half corr of z {c['split_half']['corr']} on "
              f"{c['split_half']['n_cells']:,} cells;  mean r*z "
              f"{c['cross_product_r_z']['mean']} "
              f"{c['cross_product_r_z']['ci']} (corr "
              f"{c['cross_product_r_z']['corr']});  mean r*rbar(self-excluded) "
              f"{c['cross_product_r_rbar_self_excluded']['mean']} "
              f"{c['cross_product_r_rbar_self_excluded']['ci']}")
        for a, v in c["arms"].items():
            print(f"    {a:4s} {v['n_features']:2d} feats  2A-B {v['mse_gain_s2']:+8.2f} "
                  f"[{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] s^2  clean "
                  f"{v['gain_clean']:+.4f} s"
                  + (f"  increment over h0 {v['increment_over_h0_s2']:+.2f}"
                     if "increment_over_h0_s2" in v else ""), flush=True)
    def boards_of(block: str) -> dict:
        b = {}
        for a in ("h0", "h1", "h1c", "h1_shrunk"):
            ex = lambda k: {f: recs[f][block][a]["rmse_exact"][k] for f in FOLDS}
            cl_v10, cl_new = V.board(ex("clean_v10")), V.board(ex("clean_new"))
            ar_v10, ar_new = V.board(ex("all_v10")), V.board(ex("all_new"))
            b[a] = {"new": round(cl_new, 4), "gain_board": round(cl_v10 - cl_new, 4),
                    "all_row": round(ar_new, 4),
                    "gain_all_row": round(ar_v10 - ar_new, 4)}
        return b

    board = boards_of("arms")
    board_matched = boards_of("arms_matched")
    out = {"screen": "Q04",
           "predeclared": {"bin_s": BIN, "min_dep": MIN_DEP, "min_arr": MIN_ARR,
                           "support_cells": SUPPORT_CELLS,
                           "support_dates": SUPPORT_DATES, "blocks": BLOCKS,
                           "trees": TREES, "depth": DEPTH, "lr": LR, "seed": SEED,
                           "time_match_days": TIME_MATCH_DAYS,
                           "eligible_regime": ELIGIBLE_REGIME,
                           "hold_airports": list(HOLD_AIRPORTS)},
           "primary": PK.primary(pan, preds_out["h1"],
                                 "headline arm: h1, h0 plus the arrival innovation"),
           "primary_h0": PK.primary(pan, preds_out["h0"], "the base arm h0"),
           "primary_matched": PK.primary(
               pan, preds_matched["h1"],
               "matched support (current AND 7-day-later): h1"),
           "primary_matched_h1c": PK.primary(
               pan, preds_matched["h1c"],
               "matched support (current AND 7-day-later): the placebo h1c"),
           "folds": recs, "board": board, "board_matched": board_matched,
           "wall_s": round(time.time() - t0, 1)}
    p = EXP / "analysis" / "q04.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    PK.write("q04", out, pan)
    pr = out["primary"]
    print(f"\n  PRIMARY (all-row january/july board), h1: "
          f"{pr['board_all_row']['v10']:.4f} -> {pr['board_all_row']['arm']:.4f} "
          f"({pr['board_all_row']['gain']:+.4f} s); h0 "
          f"{out['primary_h0']['board_all_row']['arm']:.4f} "
          f"({out['primary_h0']['board_all_row']['gain']:+.4f} s)")
    print("  matched-support arms (current AND 7-day-later declared support):")
    for f in FOLDS:
        c = recs[f]
        print(f"    [{f}] {c['n_supported_matched']:,} of {c['n_supported']:,} rows "
              f"({c['matched_pct_of_supported']} %); "
              + "; ".join(f"{a} {v['mse_gain_s2']:+.1f}"
                          for a, v in c["arms_matched"].items()))
    print("  secondary clean board against v10's 257.0248:")
    for a, v in board.items():
        print(f"    {a:10s} clean {v['new']:.4f} ({v['gain_board']:+.4f} s)  "
              f"all-row {v['all_row']:.4f} ({v['gain_all_row']:+.4f} s)")
    print(f"  written: {p}  ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
