"""Review round 1, B2 and B3: the two repairs the first pass coded but did not run.

1. **Cross-fitted propensities.** `fit_ratio` scored the classifier on its own
   training matrix, so both the AUC and the weights were in-sample. Here the
   propensity is produced by 5-fold cross-fitting, which is what an honest
   density ratio needs.
2. **Reduced features.** The full feature set contains `doy` and three
   month-level count aggregates that act as clocks; `q2b_auc.py` showed the
   genuine separability is 0.85-0.90 once they are removed. That ratio is less
   extreme (effective sample size 43-53 % instead of 32-40 %) and is the version
   that deserves the retrain before the lane is closed.
"""
import argparse, json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
sys.path.insert(0, "experiments/exp-011-bulkfloor/code")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import prc.data as d
from features import CAT
from q2_shift import (frame, prep, FEATURES, CLF_PARAMS, MODEL_PARAMS, CLF_ROUNDS,
                      ROUNDS, CLIP, OUT, boot_paired)
from q2b_auc import MONTHY

REDUCED = [c for c in FEATURES if c not in MONTHY]


def crossfit_ratio(tr, va, feats, threads, folds=5, seed=7):
    X = pd.concat([prep(tr, feats), prep(va, feats)], ignore_index=True)
    for c in feats:
        if c in CAT:
            X[c] = X[c].astype("category")
    lab = np.r_[np.zeros(len(tr)), np.ones(len(va))]
    rng = np.random.default_rng(seed)
    part = rng.integers(0, folds, size=len(X))
    ph = np.zeros(len(X))
    p = dict(CLF_PARAMS); p["num_threads"] = threads
    for k in range(folds):
        m = lgb.train(p, lgb.Dataset(X[part != k], label=lab[part != k]),
                      num_boost_round=CLF_ROUNDS)
        ph[part == k] = m.predict(X[part == k])
    auc = float(roc_auc_score(lab, ph))
    p_tr = np.clip(ph[:len(tr)], 1e-6, 1 - 1e-6)
    raw = (p_tr / (1 - p_tr)) * (len(tr) / len(va))
    clipped_lo = float((raw <= CLIP[0]).mean())
    clipped_hi = float((raw >= CLIP[1]).mean())
    w = np.clip(raw, *CLIP)
    return w / w.mean(), auc, dict(clip_lo=clipped_lo, clip_hi=clipped_hi,
                                   raw_mean=float(raw.mean()),
                                   raw_p01=float(np.percentile(raw, 1)),
                                   raw_p99=float(np.percentile(raw, 99)))


def run(fold, threads, arms):
    F = d.FOLDS[fold]
    tr = pd.concat([frame(m) for m in F.train_months], ignore_index=True)
    va = frame(F.valid_month)
    TR = tr[(tr.has_nm & tr.y.le(3600) & tr.off.notna() & tr.noac.ne(1)).to_numpy()]
    bulk = (va.has_nm & va.y.le(3600) & va.noac.ne(1)).to_numpy()
    Xtr, Xva = prep(TR, FEATURES), prep(va, FEATURES)
    lab = TR["off"].to_numpy()
    y = va["y"].to_numpy()
    cl = (va["apt"].astype(str) + "|" + va["MVT_TIME_UTC_mvt"].dt.floor("D").astype(str)).to_numpy()
    base = pd.read_parquet(OUT / "oof" / f"shift_{fold}.parquet")
    base = base.set_index("MVT_ID_mvt").reindex(va.MVT_ID_mvt.to_numpy())
    unw = base["y_pred_unweighted"].to_numpy()
    for arm, feats in arms.items():
        t0 = time.time()
        w, auc, diag = crossfit_ratio(TR, va[bulk], feats, threads)
        print(f"[{fold}/{arm}] cross-fitted AUC {auc:.4f}  raw mean {diag['raw_mean']:.3f} "
              f"p01 {diag['raw_p01']:.4f} p99 {diag['raw_p99']:.2f}  on the low clip "
              f"{100*diag['clip_lo']:.2f} %  on the high clip {100*diag['clip_hi']:.3f} %  "
              f"ESS {100*w.sum()**2/(w**2).sum()/len(w):.1f} % ({round(time.time()-t0)}s)", flush=True)
        p = dict(MODEL_PARAMS); p["num_threads"] = threads
        m = lgb.train(p, lgb.Dataset(Xtr, label=lab, weight=w), num_boost_round=ROUNDS)
        pred = np.maximum(va["nm_taxi"].to_numpy() - m.predict(Xva), 0.0)
        lo, hi = boot_paired(y[bulk], pred[bulk], unw[bulk], cl[bulk])
        res = dict(name="shift_repair", fold=fold, arm=arm, auc_crossfit=auc,
                   n_feat=len(feats), **diag,
                   ess=float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
                   bulk_unweighted=d.rmse(y[bulk], unw[bulk]),
                   bulk_weighted=d.rmse(y[bulk], pred[bulk]),
                   diff_lo=lo, diff_hi=hi)
        print("[repair] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in res.items()}), flush=True)
        with open(OUT / "climb.log", "a") as fh:
            fh.write(json.dumps(res) + "\n")
        pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "fold": fold, "y_true": y,
                      "y_pred": pred, "bulk": bulk}).to_parquet(
            OUT / "oof" / f"repair_{arm}_{fold}.parquet", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="july_loo,december")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--arms", default="")
    a = ap.parse_args()
    NODOY = [c for c in FEATURES if c != "doy"]
    ALL_ARMS = {"crossfit_full": FEATURES, "crossfit_reduced": REDUCED,
                "crossfit_nodoy": NODOY}
    arms = {k: v for k, v in ALL_ARMS.items() if not a.arms or k in a.arms.split(",")}
    for f in a.folds.split(","):
        run(f, a.threads, arms)
