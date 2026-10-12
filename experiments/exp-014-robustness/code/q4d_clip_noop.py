"""Q4d: is a predict-time numeric clip capable of changing anything?

q4_fixes measured the clip at **exactly 0.000 s of prediction change** on 2,947
affected july_loo rows. That is not a small effect, it is a structural one: a
gradient-boosted tree splits on thresholds drawn from the training data, so a
value above the training maximum takes the same path through every tree as the
maximum itself. Clipping to the training range is the identity map for a plain
tree ensemble.

The exception is a LINEAR-LEAF tree, where the leaf value is an affine function
of the features and an out-of-range value is extrapolated by that function.
`D1-winter` is `linear_tree=True`; `prc.tail_model` and `L2-hub` are not.

This script measures both on the rows the exposure actually lives on -- the
1,422 January EHAM rows where `apt_day_p90_mvt_aobt` is past its twelve-month
training maximum -- and needs no labels, because the question is whether the
clip moves the PREDICTION.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames14 as F  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.data as D  # noqa: E402
from prc.tail_model import MAIN_PARAMS  # noqa: E402

OUT = F.EXP / "analysis"
THREADS = 12
CLIP_COLS = ("apt_day_p90_mvt_aobt", "apt_day_med_mvt_aobt", "nonm_day_count",
             "apt_hour_med_mvt_aobt", "mvt_aobt_vs_day_med", "apt_day_long_frac")


def main() -> None:
    import lightgbm as lgb

    t0 = time.time()
    tr = F.tail_training()
    rk = F.tail_ranking()
    lab = []
    for m in D.MONTHS:
        d = D.load_month(m, columns=["MVT_ID_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt"])
        d = d[d["PHASE_mvt"] == "DEP"]
        lab.append(pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy("float64"),
                                 "y": d["TAXITIME_SEC_mvt"].to_numpy("float64")}))
    tr = tr.merge(pd.concat(lab, ignore_index=True), on="MVT_ID_mvt", how="inner")
    tr = tr[np.isfinite(tr["y"].to_numpy("float64"))]
    feats = [c for c in rk.columns if c not in ("MVT_ID_mvt", "month_str")]
    cats = [c for c in feats if tr[c].dtype == object or pd.api.types.is_string_dtype(tr[c])]
    for c in cats:
        vocab = {v: i for i, v in enumerate(pd.unique(tr[c].astype("object").dropna()))}
        tr[c] = tr[c].astype("object").map(vocab).fillna(-1).astype("int32")
        rk[c] = rk[c].astype("object").map(vocab).fillna(-1).astype("int32")
    Xtr, ytr = tr[feats], tr["y"].to_numpy("float64")
    print(f"[q4d] train {Xtr.shape} ranking {rk.shape} in {time.time()-t0:.0f}s", flush=True)

    caps = {c: float(np.nanmax(Xtr[c].to_numpy("float64"))) for c in CLIP_COLS if c in feats}
    Xrk = rk[feats].copy()
    Xcl = Xrk.copy()
    touched = np.zeros(len(Xrk), bool)
    for c, cap in caps.items():
        v = Xcl[c].to_numpy("float64")
        m = np.isfinite(v) & (v > cap)
        touched |= m
        Xcl.loc[m, c] = cap
    print(f"[q4d] clip caps {caps}", flush=True)
    print(f"[q4d] clip touches {int(touched.sum())} of {len(Xrk)} ranking rows", flush=True)

    rows = []
    for label, extra in (("plain trees (prc.tail_model, L2-hub, C2-cat)", {}),
                         ("linear leaves (D1-winter)", dict(linear_tree=True, linear_lambda=1.0))):
        p = dict(MAIN_PARAMS, num_threads=THREADS, **extra)
        p.pop("n_estimators", None)
        ds = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats)
        mdl = lgb.train(p, ds, num_boost_round=int(MAIN_PARAMS["n_estimators"]))
        a = mdl.predict(Xrk)
        b = mdl.predict(Xcl)
        d = b - a
        t = touched
        rows.append(dict(learner=label, n_touched=int(t.sum()),
                         n_changed=int((np.abs(d) > 1e-9).sum()),
                         max_abs_change=float(np.abs(d).max()),
                         mean_abs_change_on_touched=float(np.abs(d[t]).mean()) if t.any() else 0.0,
                         mean_change_on_touched=float(d[t].mean()) if t.any() else 0.0,
                         pred_raw_on_touched=float(a[t].mean()) if t.any() else np.nan,
                         pred_clip_on_touched=float(b[t].mean()) if t.any() else np.nan))
        print(f"[q4d] {label}: {rows[-1]}  ({time.time()-t0:.0f}s)", flush=True)

    T = pd.DataFrame(rows)
    T.to_csv(OUT / "q4d_clip_noop.csv", index=False)
    print("\n=== can a predict-time clip move the prediction at all? ===")
    print(T.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
