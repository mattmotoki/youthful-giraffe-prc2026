"""Measure prc.tail_model.TailModel against my own predictions, per regime.

The desk's instruction is to compare per regime rather than swap wholesale, so
this scores both models on the same rows of the same folds and reports which
wins where, plus the best fixed blend.  It does not retrain my model: it reads
the saved out-of-fold predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import prc.data as D  # noqa: E402
from prc.tail_model import TailModel  # noqa: E402

MINE = sys.argv[1] if len(sys.argv) > 1 else "c6-noac"
FOLDS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["july", "december", "january"]
pd.set_option("display.width", 250)

rows = []
for fold in FOLDS:
    oof_path = HERE / "oof" / f"{MINE}-{fold}.parquet"
    if not oof_path.exists():
        print(f"[{fold}] no {oof_path.name}, skipped", flush=True)
        continue
    train, masked, y_valid = D.fold_frames(fold)
    tm = TailModel().fit(train)
    pred = tm.predict(masked)
    theirs = pred.set_index("MVT_ID_mvt")

    save = HERE / "oof" / f"tailmodel-{fold}.parquet"
    theirs.reset_index()[["MVT_ID_mvt", "regime", "y_pred", "p_placeholder"]].to_parquet(save, index=False)
    print(f"[{fold}] tail-model predictions saved to {save.name}", flush=True)

    mine = pd.read_parquet(oof_path).set_index("MVT_ID_mvt")
    j = mine.join(theirs[["regime", "y_pred", "p_placeholder"]].rename(
        columns={"y_pred": "y_them", "p_placeholder": "p"}), how="inner")
    j = j[j.y_true.notna() & j.y_them.notna()]
    print(f"[{fold}] {len(j)} rows joined, regimes {j.regime.value_counts().to_dict()}", flush=True)

    for reg, g in list(j.groupby("regime")) + [("ALL", j)]:
        e_me = float(np.sqrt(((g.y_true - g.y_pred) ** 2).mean()))
        e_th = float(np.sqrt(((g.y_true - g.y_them) ** 2).mean()))
        ws = np.linspace(0, 1, 21)
        errs = [float(np.sqrt(((g.y_true - (w * g.y_them + (1 - w) * g.y_pred)) ** 2).mean())) for w in ws]
        k = int(np.argmin(errs))
        rows.append(dict(fold=fold, regime=reg, n=len(g), mine=e_me, theirs=e_th,
                         best_w_theirs=ws[k], blend=errs[k],
                         # what the whole fold would score if this regime took the better of the two
                         gain_if_better=e_me - min(e_me, e_th)))

t = pd.DataFrame(rows)
print("\n== RMSE by regime: my model, the tail model, and the best fixed blend ==")
print(t.round(1).to_string(index=False))

print("\n== whole-fold RMSE if each regime keeps its winner ==")
for fold in t.fold.unique():
    sub = t[(t.fold == fold) & (t.regime != "ALL")]
    n = sub.n.sum()
    sse_mine = (sub.n * sub.mine ** 2).sum()
    sse_best = (sub.n * np.minimum(sub.mine, sub.theirs) ** 2).sum()
    sse_blend = (sub.n * sub.blend ** 2).sum()
    print(f"  {fold:9s} mine {np.sqrt(sse_mine/n):7.1f} | per-regime winner {np.sqrt(sse_best/n):7.1f} "
          f"| per-regime blend {np.sqrt(sse_blend/n):7.1f}")
