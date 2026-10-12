"""exp-013 ceiling instruments for the no-NM non-noac segment.

The assignment asks for an oracle that gives the segment its true
`BLOCK − SCHED`. That instrument is degenerate and this script shows why, then
replaces it with one that answers the question it was asked for.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import prc.data as D
from segment import segment_mask, rmse, FOLDS

# Threads: docs/COMPUTE.md rule compute allocation allocates this lane 6 of the desk's 40.

N_JOBS = 6  # desk compute rule compute allocation (docs/COMPUTE.md): congestion is allocated 6 threads
CATS = ["apt", "apt_stand", "apt_rwy", "ac_type", "ades", "flight_rule"]


def features(month: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Legal features for every departure of a month, plus y and the segment mask.

    Only columns that survive on a no-NM row are used: no `_flt` column appears.
    """
    m = D.load_month(month)
    dep = m[m["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    mvt, sch = dep["MVT_TIME_UTC_mvt"], dep["SCHED_TIME_UTC_mvt"]
    apt = dep["ADEP_mvt"].astype(str)
    X = pd.DataFrame({
        "mvt_sched": (mvt - sch).dt.total_seconds().to_numpy(),
        "hour": mvt.dt.hour.to_numpy() + mvt.dt.minute.to_numpy() / 60.0,
        "sched_hour": sch.dt.hour.to_numpy() + sch.dt.minute.to_numpy() / 60.0,
        "dow": mvt.dt.dayofweek.to_numpy(),
        "flight_len": dep["FLIGHT_mvt"].astype(str).str.len().to_numpy(),
        "apt": apt.to_numpy(),
        "apt_stand": (apt + ":" + dep["STAND_mvt"].astype(str)).to_numpy(),
        "apt_rwy": (apt + ":" + dep["RUNWAY_mvt"].astype(str)).to_numpy(),
        "ac_type": dep["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy(),
        "ades": dep["ADES_mvt"].astype(str).to_numpy(),
        "flight_rule": dep["FLIGHT_RULE_mvt"].astype(str).to_numpy(),
    })
    for c in CATS:
        X[c] = X[c].astype("category")
    y = dep[D.TARGET].to_numpy(float)
    block_sched = (dep["BLOCK_TIME_UTC_mvt"] - sch).dt.total_seconds().to_numpy()
    return X, y, segment_mask(dep), block_sched


def fit_predict(Xtr, ytr, Xva, rounds=600):
    import lightgbm as lgb
    cats = [c for c in CATS if c in Xtr.columns]
    for c in cats:  # one category space across train and valid
        a, b = Xtr[c].astype(str), Xva[c].astype(str)
        u = pd.Index(pd.unique(np.concatenate([a.to_numpy(), b.to_numpy()])))
        Xtr[c] = pd.Categorical(a, categories=u)
        Xva[c] = pd.Categorical(b, categories=u)
    mdl = lgb.LGBMRegressor(n_estimators=rounds, learning_rate=0.05, num_leaves=63,
                            min_child_samples=40, subsample=0.9, subsample_freq=1,
                            colsample_bytree=0.9, n_jobs=N_JOBS, random_state=7, verbose=-1)
    mdl.fit(Xtr, ytr, categorical_feature=cats)
    return mdl.predict(Xva)


def run_fold(fold: str, out: dict, segment_only: bool = False):
    t0 = time.time()
    f = D.FOLDS[fold]
    tr = [features(m) for m in f.train_months]
    Xtr = pd.concat([t[0] for t in tr], ignore_index=True)
    ytr = np.concatenate([t[1] for t in tr])
    segtr = np.concatenate([t[2] for t in tr])
    bstr = np.concatenate([t[3] for t in tr])
    Xva, yva, segva, bsva = features(f.valid_month)

    res = {}
    # (a) trained on every departure, scored on the segment -- the legal model
    p = fit_predict(Xtr.copy(), ytr, Xva.copy())
    res["legal_all_rows"] = rmse(yva[segva], p[segva])
    # (b) trained on the segment's own population only. Answered on july
    # (753.0 against 584.9), so it is skipped elsewhere to stay inside the
    # 6-thread allocation.
    if segment_only:
        p2 = fit_predict(Xtr[segtr].copy(), ytr[segtr], Xva.copy())
        res["legal_segment_only"] = rmse(yva[segva], p2[segva])
    # (c) the assignment's oracle: add the true BLOCK - SCHED
    Xtr_o, Xva_o = Xtr.copy(), Xva.copy()
    Xtr_o["block_sched"] = bstr
    Xva_o["block_sched"] = bsva
    p3 = fit_predict(Xtr_o, ytr, Xva_o)
    res["oracle_block_sched"] = rmse(yva[segva], p3[segva])
    # (d) the same oracle used as arithmetic rather than as a feature
    res["oracle_arithmetic"] = rmse(yva[segva], (Xva["mvt_sched"].to_numpy() - bsva)[segva])
    # (e) predicting the training segment mean
    res["segment_mean"] = rmse(yva[segva], np.full(segva.sum(), ytr[segtr].mean()))
    res["n"] = int(segva.sum())
    res["secs"] = round(time.time() - t0)
    out[fold] = res
    print(f"[{fold}] " + "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in res.items()), flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    out = {}
    if (HERE / "oracle_bound.json").exists():
        out = json.loads((HERE / "oracle_bound.json").read_text())
    for fold in (args or list(FOLDS)):
        run_fold(fold, out, segment_only="--segment-only" in sys.argv)
        (HERE / "oracle_bound.json").write_text(json.dumps(out, indent=2))
