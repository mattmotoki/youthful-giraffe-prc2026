"""Q1(c): seed-ensemble variance decomposition on the bulk.

With K seeds of the same configuration, f_k, and their mean f_bar_K:

    MSE_single - MSE_K = sigma_m^2 (1 - 1/K)          =>  sigma_m^2 identified
    MSE_inf = MSE_K - sigma_m^2 / K                    =  irreducible + bias^2

`sigma_m` is pure estimation variance, removable by averaging more seeds.
`MSE_inf` is the best this model family reaches; the gap between it and the
near-duplicate bound of Q1(a) is the bias term.
"""
import argparse, json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
sys.path.insert(0, "experiments/exp-011-bulkfloor/code")
import lightgbm as lgb
import prc.data as d
from features import ART, CAT, BASE_NUM, OCC, NBR

FEATURES = CAT + BASE_NUM + OCC + NBR
PARAMS = dict(objective="regression", metric="l2", learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, max_cat_threshold=64,
              cat_smooth=20.0)
ROUNDS = 600   # halved from the 1200 of exp-003 T_best: the workstation was
               # at load 250-400 and a 1200-round fit was taking over 27 min.
               # Fewer rounds means *more* seed variance, so sigma_m below is a
               # conservative (upper) estimate of the deployed model's.


def frame(m):
    return pd.read_parquet(ART / f"{m}.parquet")


def prep(df):
    X = df[FEATURES].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    return X


def run(fold, seeds, threads, rounds=ROUNDS):
    F = d.FOLDS[fold]
    tr = pd.concat([frame(m) for m in F.train_months], ignore_index=True)
    va = frame(F.valid_month)
    for f in (tr, va):
        f["off"] = (f.BLOCK_TIME_UTC_mvt - f.AOBT_3_flt).dt.total_seconds()
        f["y"] = f.TAXITIME_SEC_mvt.astype(float)
    mtr = tr.has_nm & tr.y.le(3600) & tr.off.notna() & tr.noac.ne(1)
    Xtr, Xva = prep(tr[mtr]), prep(va)
    preds = []
    for s in seeds:
        t0 = time.time()
        p = dict(PARAMS); p["seed"] = s; p["num_threads"] = threads
        m = lgb.train(p, lgb.Dataset(Xtr, label=tr.loc[mtr, "off"]), num_boost_round=rounds)
        preds.append(np.maximum(va["nm_taxi"].to_numpy() - m.predict(Xva), 0.0))
        print(f"  {fold} seed {s}: {round(time.time()-t0)}s", flush=True)
    P = np.column_stack(preds)
    y = va["y"].to_numpy()
    bulk = (va.has_nm & va.y.le(3600) & va.noac.ne(1)).to_numpy()
    out = pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "fold": fold,
                        "y_true": y, "y_pred": P.mean(axis=1),
                        "seed_sd": P.std(axis=1, ddof=0), "bulk": bulk})
    for i, s in enumerate(seeds):
        out[f"p_s{s}"] = P[:, i]
    o = d.ROOT / "experiments/exp-011-bulkfloor/oof" / f"seeds_{fold}.parquet"
    out.to_parquet(o, index=False)
    K = len(seeds)
    yb, Pb = y[bulk], P[bulk]
    mse_single = float(np.mean([(np.mean((yb - Pb[:, i]) ** 2)) for i in range(K)]))
    mse_K = float(np.mean((yb - Pb.mean(axis=1)) ** 2))
    sigma_m2 = (mse_single - mse_K) * K / (K - 1)
    mse_inf = mse_K - sigma_m2 / K
    res = dict(fold=fold, K=K, rounds=rounds, n_bulk=int(bulk.sum()),
               rmse_single=np.sqrt(mse_single), rmse_K=np.sqrt(mse_K),
               rmse_inf=np.sqrt(max(mse_inf, 0.0)), sigma_m=np.sqrt(max(sigma_m2, 0.0)))
    print("[seeds] " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                   for k, v in res.items()}), flush=True)
    with open(d.ROOT / "experiments/exp-011-bulkfloor/climb.log", "a") as fh:
        fh.write(json.dumps({"name": "seeds", **res}) + "\n")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="july_loo,december")
    ap.add_argument("--seeds", default="7,17,42,101,2029")
    ap.add_argument("--threads", type=int, default=6)   # desk cap compute allocation, 2026-09-05
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    a = ap.parse_args()
    for fold in a.folds.split(","):
        run(fold, [int(s) for s in a.seeds.split(",")], a.threads, a.rounds)
