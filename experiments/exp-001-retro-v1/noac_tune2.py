"""Second screen for the noac head: a placeholder-probability mixture against
the anchored regression, plus a rounds sweep around the first screen's optimum."""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import lightgbm as lgb
import prc.data as D
from run import CatEncoder, frame_for_months, valid_frame
from features import CAT_FEATURES

FOLD = sys.argv[1] if len(sys.argv) > 1 else "july"
fold = D.FOLDS[FOLD]
Xtr, ytr, _ = frame_for_months(fold.train_months, "v2")
Xva, yva = valid_frame(fold, "v2")
mtr = Xtr["noac"].to_numpy().astype(bool)
mva = Xva["noac"].to_numpy().astype(bool) & yva.notna().to_numpy()
Xt, yt = Xtr[mtr], ytr.to_numpy()[mtr]
Xv, yv = Xva[mva], yva.to_numpy()[mva]
bt = Xt["d_takeoff_sched"].to_numpy("float64")
bv = Xv["d_takeoff_sched"].to_numpy("float64")
r = lambda p: float(np.sqrt(np.mean((yv - np.maximum(p, 60.0)) ** 2)))
print(f"{FOLD}: train {mtr.sum()} valid {mva.sum()} noac rows", flush=True)

# a placeholder is a row whose block time is the schedule; the label uses the
# training months' block time, which is not masked there.
ph_t = np.abs(yt - bt) < 60.0
print(f"placeholder share: train {ph_t.mean():.3f}", flush=True)
taxi_t = yt[~ph_t]
print(f"non-placeholder noac taxi: mean {taxi_t.mean():.0f} median {np.median(taxi_t):.0f}", flush=True)

print("\n-- rules --")
print(f"  always take-off-minus-SCHED         : {r(bv):8.0f}")
for thr in (1800, 3600, 5400, 7200, 10800):
    p = np.where(bv > thr, bv, taxi_t.mean())
    print(f"  ds>{thr:5d} -> ds, else train taxi mean : {r(p):8.0f}")

rows = []
CATS = CAT_FEATURES
for leaves, mdl, lr, rounds in itertools.product([15, 31, 63], [3, 5, 10], [0.03, 0.05], [30, 60, 100, 150]):
    prm = dict(objective="regression", metric="rmse", learning_rate=lr, num_leaves=leaves,
               min_data_in_leaf=mdl, feature_fraction=0.8, bagging_fraction=1.0, lambda_l2=1.0,
               max_cat_threshold=16, cat_smooth=5.0, min_data_per_group=5, num_threads=6,
               verbosity=-1, seed=7)
    enc = CatEncoder(CATS).fit(Xt)
    m = lgb.train(prm, lgb.Dataset(enc.transform(Xt), label=yt, init_score=bt, categorical_feature=CATS),
                  num_boost_round=rounds)
    rows.append(dict(kind="anchored-reg", leaves=leaves, min_data=mdl, lr=lr, rounds=rounds,
                     rmse=r(m.predict(enc.transform(Xv)) + bv)))

    prm_c = dict(prm, objective="binary", metric="binary_logloss")
    mc = lgb.train(prm_c, lgb.Dataset(enc.transform(Xt), label=ph_t.astype(int), categorical_feature=CATS),
                   num_boost_round=rounds)
    q = mc.predict(enc.transform(Xv))
    for taxi in ("mean", "median"):
        tv = taxi_t.mean() if taxi == "mean" else np.median(taxi_t)
        rows.append(dict(kind=f"mixture-{taxi}", leaves=leaves, min_data=mdl, lr=lr, rounds=rounds,
                         rmse=r(q * bv + (1 - q) * tv)))
    # mixture with a regressed non-placeholder taxi time
    mr = lgb.train(prm, lgb.Dataset(enc.transform(Xt[~ph_t]), label=taxi_t, categorical_feature=CATS),
                   num_boost_round=rounds)
    rows.append(dict(kind="mixture-reg", leaves=leaves, min_data=mdl, lr=lr, rounds=rounds,
                     rmse=r(q * bv + (1 - q) * mr.predict(enc.transform(Xv)))))

t = pd.DataFrame(rows).sort_values("rmse")
pd.set_option("display.width", 200)
print("\n-- models --"); print(t.head(15).to_string(index=False))
print("\nbest per kind:"); print(t.groupby("kind").rmse.min().sort_values().round(0).to_string())
