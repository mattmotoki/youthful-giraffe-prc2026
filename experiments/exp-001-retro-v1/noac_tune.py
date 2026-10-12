"""Screen head configurations for the placeholder-block-time (noac) rows alone.

The head trains on a few hundred rows, so it is screened on its own instead of
paying a full model fit per configuration.
"""
from __future__ import annotations
import itertools, sys, time
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
print(f"{FOLD}: {mtr.sum()} training noac rows, {mva.sum()} validation noac rows", flush=True)

bt = Xt["d_takeoff_sched"].to_numpy("float64")
bv = Xv["d_takeoff_sched"].to_numpy("float64")
r = lambda p: float(np.sqrt(np.mean((yv - p) ** 2)))
print(f"  baseline take-off-minus-SCHED  : {r(bv):8.0f}")
print(f"  baseline train mean            : {r(np.full(len(yv), yt.mean())):8.0f}")
print(f"  baseline train median          : {r(np.full(len(yv), np.median(yt))):8.0f}")

rows = []
for leaves, mdl, lr, rounds, anchored in itertools.product([4, 8, 15, 31], [5, 10, 20], [0.03, 0.08], [100, 300, 800], [True, False]):
    p = dict(objective="regression", metric="rmse", learning_rate=lr, num_leaves=leaves,
             min_data_in_leaf=mdl, feature_fraction=0.8, bagging_fraction=1.0, lambda_l2=1.0,
             max_cat_threshold=16, cat_smooth=5.0, min_data_per_group=5, num_threads=6,
             verbosity=-1, seed=7)
    enc = CatEncoder(CAT_FEATURES).fit(Xt)
    d = lgb.Dataset(enc.transform(Xt), label=yt, init_score=bt if anchored else None,
                    categorical_feature=CAT_FEATURES)
    m = lgb.train(p, d, num_boost_round=rounds)
    pred = m.predict(enc.transform(Xv)) + (bv if anchored else 0.0)
    rows.append(dict(leaves=leaves, min_data=mdl, lr=lr, rounds=rounds, anchored=anchored,
                     rmse=r(pred), rmse_floor=r(np.maximum(pred, 60.0))))
t = pd.DataFrame(rows).sort_values("rmse_floor")
pd.set_option("display.width", 200)
print(t.head(20).to_string(index=False))
print("\nworst 5:"); print(t.tail(5).to_string(index=False))
