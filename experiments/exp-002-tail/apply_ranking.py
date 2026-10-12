"""Fit the noac sub-model on all twelve 2025 months and predict the ranking rows.

Writes `experiments/exp-002-tail/noac_ranking_predictions.parquet`
(MVT_ID_mvt, month, mvt_sched, y_pred) so any other experiment can splice these
388 predictions in over its own bulk prediction.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import prc.data as D
from prc.tail_features import tail_signals
sys.path.insert(0, str(D.ROOT / "experiments" / "exp-002-tail"))
from noac_model import NoacPredictor

ART = D.ROOT / "artifacts" / "exp-002-tail"
EXP = D.ROOT / "experiments" / "exp-002-tail"

tr = pd.concat([pd.read_parquet(ART / f"signals_{m}.parquet") for m in D.MONTHS], ignore_index=True)
model = NoacPredictor.fit(tr)

rk = D.load_ranking()
out = []
for mo, g in rk.groupby("month"):
    s = tail_signals(g)
    s["MVT_ID_mvt"] = g["MVT_ID_mvt"].reindex(s.index)
    s["month"] = mo
    na = s[s.noac == 1].copy()
    na["y_pred"] = np.clip(model.predict(na), 60.0, None)
    out.append(na[["MVT_ID_mvt", "month", "mvt_sched", "y_pred"]])
res = pd.concat(out, ignore_index=True)
res.to_parquet(EXP / "noac_ranking_predictions.parquet", index=False)
print(f"{len(res)} noac ranking rows")
print(res.groupby("month").agg(n=("y_pred", "size"), pred_mean=("y_pred", "mean"),
                               pred_med=("y_pred", "median"), pred_max=("y_pred", "max")).round(0).to_string())
print("predictions at the 60 s clip:", int((res.y_pred <= 60.0).sum()))
print("sha256", D.sha256(EXP / "noac_ranking_predictions.parquet"))
