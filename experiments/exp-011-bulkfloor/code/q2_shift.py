"""Q2: label-free importance weighting against the validation month's feature mix.

A classifier separates the validation month's rows from the training rows on the
*model's own features*; the density ratio `w = p/(1-p) * n_tr/n_va` reweights the
training rows to look like the validation month; the plain-tree baseline is
refitted with those weights and scored against the unweighted fit with a
day-clustered bootstrap on the paired difference.

`doy` is excluded from the classifier: it is a pure time index, so it separates
the two frames perfectly and the density ratio degenerates. Everything else the
model uses is kept, including the traffic-volume counts, because those are part
of the shift rather than an artefact of it. The AUC with `doy` is reported too,
to show the difference.
"""
import argparse, json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
sys.path.insert(0, "experiments/exp-011-bulkfloor/code")
import lightgbm as lgb
import prc.data as d
from features import ART, CAT, BASE_NUM, OCC, NBR

FEATURES = CAT + BASE_NUM + OCC + NBR
CLF_DROP = ["doy"]
CLF_FEATURES = [c for c in FEATURES if c not in CLF_DROP]
MODEL_PARAMS = dict(objective="regression", metric="l2", learning_rate=0.05, num_leaves=127,
                    min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7,
                    max_cat_threshold=64, cat_smooth=20.0)
CLF_PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7,
                  max_cat_threshold=64, cat_smooth=20.0)
ROUNDS, CLF_ROUNDS = 1200, 300
CLIP = (0.05, 20.0)
OUT = d.ROOT / "experiments/exp-011-bulkfloor"


def frame(m):
    f = pd.read_parquet(ART / f"{m}.parquet")
    f["y"] = f.TAXITIME_SEC_mvt.astype(float)
    f["off"] = (f.BLOCK_TIME_UTC_mvt - f.AOBT_3_flt).dt.total_seconds()
    return f


def prep(df, feats):
    X = df[feats].copy()
    for c in feats:
        if c in CAT:
            X[c] = X[c].astype("category")
    return X


def fit_ratio(tr, va, feats, threads):
    """p(validation | x) by LightGBM, then the clipped density ratio on tr."""
    X = pd.concat([prep(tr, feats), prep(va, feats)], ignore_index=True)
    for c in feats:
        if c in CAT:
            X[c] = X[c].astype("category")
    lab = np.r_[np.zeros(len(tr)), np.ones(len(va))]
    p = dict(CLF_PARAMS); p["num_threads"] = threads
    m = lgb.train(p, lgb.Dataset(X, label=lab), num_boost_round=CLF_ROUNDS)
    ph = m.predict(X)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(lab, ph))
    p_tr = np.clip(ph[:len(tr)], 1e-6, 1 - 1e-6)
    w = (p_tr / (1 - p_tr)) * (len(tr) / len(va))
    w = np.clip(w, *CLIP)
    return w / w.mean(), auc, m, ph


def boot_paired(y, pa, pb, cluster, n_boot=400, seed=7):
    rng = np.random.default_rng(seed)
    ea, eb = (y - pa) ** 2, (y - pb) ** 2
    g = pd.factorize(cluster)[0]
    order = np.argsort(g, kind="stable")
    ga, gb, gs = ea[order], eb[order], g[order]
    bounds = np.searchsorted(gs, np.arange(gs.max() + 2))
    sizes = np.diff(bounds)
    ca, cb = np.add.reduceat(ga, bounds[:-1]), np.add.reduceat(gb, bounds[:-1])
    G = len(sizes)
    k = rng.integers(0, G, size=(n_boot, G))
    nn = sizes[k].sum(axis=1)
    diffs = np.sqrt(ca[k].sum(axis=1) / nn) - np.sqrt(cb[k].sum(axis=1) / nn)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run(fold, threads):
    F = d.FOLDS[fold]
    tr = pd.concat([frame(m) for m in F.train_months], ignore_index=True)
    va = frame(F.valid_month)
    mtr = (tr.has_nm & tr.y.le(3600) & tr.off.notna() & tr.noac.ne(1)).to_numpy()
    TR = tr[mtr]
    bulk = (va.has_nm & va.y.le(3600) & va.noac.ne(1)).to_numpy()
    t0 = time.time()
    w, auc, clf, ph = fit_ratio(TR, va[bulk], CLF_FEATURES, threads)
    _, auc_doy, _, _ = fit_ratio(TR, va[bulk], FEATURES, threads)
    print(f"[{fold}] classifier AUC {auc:.4f} (with doy {auc_doy:.4f}), "
          f"weights p1 {np.percentile(w,1):.3f} p50 {np.percentile(w,50):.3f} "
          f"p99 {np.percentile(w,99):.3f} clipped {(w>=CLIP[1]/w.mean()).mean()*100:.2f} % "
          f"({round(time.time()-t0)}s)", flush=True)
    Xtr, Xva = prep(TR, FEATURES), prep(va, FEATURES)
    lab = TR["off"].to_numpy()
    preds = {}
    for name, weight in (("unweighted", None), ("weighted", w)):
        t0 = time.time()
        p = dict(MODEL_PARAMS); p["num_threads"] = threads
        m = lgb.train(p, lgb.Dataset(Xtr, label=lab, weight=weight), num_boost_round=ROUNDS)
        preds[name] = np.maximum(va["nm_taxi"].to_numpy() - m.predict(Xva), 0.0)
        print(f"    {name}: bulk {d.rmse(va['y'].to_numpy()[bulk], preds[name][bulk]):.2f} "
              f"({round(time.time()-t0)}s)", flush=True)
    y = va["y"].to_numpy()
    cl = (va["apt"].astype(str) + "|" + va["MVT_TIME_UTC_mvt"].dt.floor("D").astype(str)).to_numpy()
    lo, hi = boot_paired(y[bulk], preds["weighted"][bulk], preds["unweighted"][bulk], cl[bulk])
    res = dict(name="shift", fold=fold, auc=auc, auc_with_doy=auc_doy,
               n_train=int(len(TR)), n_valid=int(bulk.sum()),
               w_p01=float(np.percentile(w, 1)), w_p50=float(np.percentile(w, 50)),
               w_p99=float(np.percentile(w, 99)), w_max=float(w.max()),
               ess=float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
               bulk_unweighted=d.rmse(y[bulk], preds["unweighted"][bulk]),
               bulk_weighted=d.rmse(y[bulk], preds["weighted"][bulk]),
               diff_lo=lo, diff_hi=hi)
    print("[shift] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in res.items()}), flush=True)
    with open(OUT / "climb.log", "a") as fh:
        fh.write(json.dumps(res) + "\n")
    pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "fold": fold, "y_true": y,
                  "y_pred": preds["weighted"], "y_pred_unweighted": preds["unweighted"],
                  "bulk": bulk}).to_parquet(OUT / "oof" / f"shift_{fold}.parquet", index=False)
    return res


def ranking(threads):
    """The same classifier against each 2026 ranking month."""
    tr = pd.concat([frame(m) for m in d.MONTHS], ignore_index=True)
    TR = tr[(tr.has_nm & tr.y.le(3600) & tr.off.notna() & tr.noac.ne(1)).to_numpy()]
    rk = pd.read_parquet(ART / "ranking.parquet")
    rk["month"] = rk["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    rows = []
    for m, g in rk.groupby("month"):
        g = g[g.has_nm & g.noac.ne(1)]
        w, auc, _, _ = fit_ratio(TR, g, CLF_FEATURES, threads)
        rows.append(dict(name="shift_ranking", month=m, n_valid=int(len(g)), auc=auc,
                         w_p01=float(np.percentile(w, 1)), w_p50=float(np.percentile(w, 50)),
                         w_p99=float(np.percentile(w, 99)), w_max=float(w.max()),
                         ess=float(w.sum() ** 2 / (w ** 2).sum() / len(w))))
        print("[ranking] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in rows[-1].items()}), flush=True)
        with open(OUT / "climb.log", "a") as fh:
            fh.write(json.dumps(rows[-1]) + "\n")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="july,july_loo,december,january")
    ap.add_argument("--threads", type=int, default=6)   # desk cap compute allocation, 2026-09-05
    ap.add_argument("--ranking", action="store_true")
    a = ap.parse_args()
    if a.ranking:
        ranking(a.threads)
    else:
        for f in a.folds.split(","):
            run(f, a.threads)
