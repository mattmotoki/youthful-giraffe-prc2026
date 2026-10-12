"""Leave-one-month-out check of the noac sub-model over all twelve 2025 months."""
import numpy as np, pandas as pd, prc.data as D
from noac_model import fit_predict, PH_TOL
ART = D.ROOT/"artifacts"/"exp-002-tail"
df = pd.concat([pd.read_parquet(ART/f"signals_{m}.parquet") for m in D.MONTHS], ignore_index=True)
na = df[(df.noac == 1) & df.y.notna()].reset_index(drop=True)
print(f"noac rows: {len(na)}  P(placeholder)={(np.abs(na.mvt_sched-na.y)<PH_TOL).mean():.3f}")
preds, ys, mons = {}, [], []
for m in D.MONTHS:
    te = na[na.month == m]; tr = na[na.month != m]
    if len(te) == 0: continue
    o = fit_predict(tr, te)
    for k, v in o.items():
        if k.startswith("_"): continue
        preds.setdefault(k, []).append(v)
    ys.append(te.y.to_numpy(float)); mons.append(np.repeat(m, len(te)))
y = np.concatenate(ys); mon = np.concatenate(mons)
rows = []
NR, NN = 344841, 388
for k, v in preds.items():
    p = np.concatenate(v); se = ((y-p)**2)
    rows.append(dict(predictor=k, rmse=np.sqrt(se.mean()),
                     rmse_le3600=np.sqrt(se[y<=3600].mean()), rmse_gt3600=np.sqrt(se[y>3600].mean()),
                     ranking_contrib=np.sqrt(NN*se.mean()/NR),
                     worst_month=max(np.sqrt(se[mon==m].mean()) for m in np.unique(mon))))
r = pd.DataFrame(rows).sort_values("rmse")
print(r.round(1).to_string(index=False))
r.to_csv(D.ROOT/"experiments"/"exp-002-tail"/"work"/"noac_lomo.csv", index=False)
best = r.iloc[0].predictor
p = np.concatenate(preds[best])
pd.DataFrame({"month": mon, "y": y, "pred": p}).to_parquet(D.ROOT/"experiments"/"exp-002-tail"/"work"/"noac_lomo_pred.parquet")
print("\nper-month RMSE of", best)
print(pd.DataFrame({"m": mon, "se": (y-p)**2}).groupby("m").agg(n=("se","size"), rmse=("se", lambda s: np.sqrt(s.mean()))).round(0).to_string())
