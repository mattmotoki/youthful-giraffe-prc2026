"""Which observable separates a PLACEHOLDER from an ORDINARY taxi in the 1-8 h
MVT-SCHED band? That band is 88.4 % of noac SSE and 20.0 % of the july_loo fold,
and the delivered head's six worst rows are all ordinary taxis it called
placeholders.

If nothing separates them, the hedge is already optimal and the lane closes with
a measured ceiling. That is a real outcome, so it is stated either way.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import prc.data as D

t = pd.read_parquet(D.ROOT / "artifacts/exp-009-noac2/noac_train.parquet")
ms, y = t.mvt_sched.astype(float), t.y.astype(float)
band = (ms >= 3600) & (ms < 8 * 3600) & y.notna()
ph = (y - ms).abs() <= 300                     # the block time was the schedule
roll = (y - 86400).abs() < 3600
ordinary = ~ph & ~roll
b = t[band].copy()
b["cls"] = np.where(ph[band], "placeholder", np.where(roll[band], "rollover", "ordinary"))
print(f"1-8 h band: {len(b)} rows of {int(y.notna().sum())} labelled noac rows")
print(b.cls.value_counts().to_string())
sub = b[b.cls.isin(("placeholder", "ordinary"))].copy()
lab = (sub.cls == "placeholder").to_numpy().astype(int)
print(f"\nplaceholder vs ordinary in band: n={len(sub)}, P(placeholder)={lab.mean():.3f}\n")

num = [c for c in sub.columns if c not in ("cls", "apt", "stand", "runway",
                                           "flight_prefix", "op_prefix", "month",
                                           "MVT_ID_mvt", "y", "noac")
       and pd.api.types.is_numeric_dtype(sub[c])]
rows = []
for c in num:
    v = sub[c].astype(float)
    m = v.notna().to_numpy()
    if m.sum() < 50 or v[m].nunique() < 2:
        continue
    try:
        auc = roc_auc_score(lab[m], v[m].to_numpy())
    except ValueError:
        continue
    rows.append(dict(observable=c, n=int(m.sum()), auc=round(max(auc, 1 - auc), 4),
                     dir="+" if auc >= 0.5 else "-",
                     ph_med=round(float(v[m][lab[m] == 1].median()), 1),
                     ord_med=round(float(v[m][lab[m] == 0].median()), 1)))
r = pd.DataFrame(rows).sort_values("auc", ascending=False)
print("single-observable separation (AUC, folded so 0.5 = useless):")
print(r.head(18).to_string(index=False))
print("\n... weakest:")
print(r.tail(5).to_string(index=False))

print("\ncategoricals, placeholder rate by level (>=20 rows):")
for c in ("apt", "op_prefix", "flight_prefix"):
    g = sub.groupby(c).agg(n=("cls", "size"), p=("cls", lambda s: (s == "placeholder").mean()))
    g = g[g.n >= 20].sort_values("p")
    if len(g):
        print(f"  -- {c}: {len(g)} levels, p from {g.p.min():.3f} to {g.p.max():.3f}")
        print(g.head(3).round(3).to_string(), "\n  ...")
        print(g.tail(3).round(3).to_string())
