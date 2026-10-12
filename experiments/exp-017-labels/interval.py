"""Day-clustered paired bootstrap for arm A1 against the control.

Rows within a calendar day are not independent -- weather, stand closures and
the departure bank are day-level -- so an unclustered interval overstates
significance. Clustering is by calendar day of take-off. Seeds are averaged per
row first, so the interval is about the recipe and not about the draw.
"""
import numpy as np, pandas as pd, prc.data as D

P = pd.read_parquet(D.ROOT / "experiments/exp-017-labels/work/interval_preds.parquet")
VAL = {"july": "2025-07", "january": "2025-01", "july_loo": "2025-07"}
B, RNG = 2000, np.random.default_rng(7)
rows = []
for fold, mo in VAL.items():
    f = P[P.fold == fold]
    piv = (f.groupby(["arm", "MVT_ID_mvt"])
             .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")).reset_index())
    a = piv[piv.arm == "A1 unreached"].set_index("MVT_ID_mvt")
    c = piv[piv.arm == "0 control"].set_index("MVT_ID_mvt")
    idx = a.index.intersection(c.index)
    a, c = a.loc[idx], c.loc[idx]
    dep = D.load_month(mo).query("PHASE_mvt == 'DEP'")
    day = (dep.set_index("MVT_ID_mvt")["MVT_TIME_UTC_mvt"].dt.floor("D")
              .reindex(idx).to_numpy())
    y = a.y_true.to_numpy(float)
    ea = (y - a.y_pred.to_numpy(float)) ** 2
    ec = (y - c.y_pred.to_numpy(float)) ** 2
    excl = D.excluded_ids(fold)
    clean = ~pd.Index(idx).isin(excl)
    days, inv = np.unique(day, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(days))]
    for label, m in (("rmse", np.ones(len(y), bool)), ("rmse_clean", clean)):
        obs = np.sqrt(ea[m].mean()) - np.sqrt(ec[m].mean())
        boot = np.empty(B)
        for b in range(B):
            pick = np.concatenate([groups[i] for i in RNG.integers(0, len(groups), len(groups))])
            pk = pick[m[pick]]
            boot[b] = np.sqrt(ea[pk].mean()) - np.sqrt(ec[pk].mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(fold=fold, metric=label, n=int(m.sum()), days=len(days),
                         delta=round(obs, 3), lo=round(lo, 3), hi=round(hi, 3),
                         excludes_zero=bool(hi < 0 or lo > 0)))
        print(f"{fold:9s} {label:11s} n={int(m.sum()):7,} days={len(days):3d}  "
              f"A1 - control = {obs:+7.3f}  95% [{lo:+7.3f}, {hi:+7.3f}]  "
              f"{'CLEAR OF ZERO' if (hi<0 or lo>0) else 'spans zero'}", flush=True)
pd.DataFrame(rows).to_csv(D.ROOT / "experiments/exp-017-labels/work/interval.csv", index=False)
