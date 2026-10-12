"""Q4e: the linear-leaf half of the clip question, sized to the question.

q4d measured the plain-tree half at full scale: `prc.tail_model.MAIN_PARAMS` on
2.1 M rows, the clip touching 1,422 ranking rows, 0 predictions changed. Its
`linear_tree=True` arm was still fitting after an hour on a box at load 470 and
was stopped, because the question it answers is **qualitative** — can a
predict-time clip move a linear-leaf model at all? — and a qualitative question
does not need a 2.1 M-row fit.

So: the same two learners, same frame, same clip, on a 400,000-row sample and
150 rounds. If the plain arm reproduces 0.000 at this size the instrument is
sound, and whatever the linear arm gives is then the answer.
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
THREADS = 8
N_SAMPLE = 400_000
ROUNDS = 150
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
    caps_src = tr           # caps come from ALL twelve months, not the sample
    feats = [c for c in rk.columns if c not in ("MVT_ID_mvt", "month_str")]
    cats = [c for c in feats if tr[c].dtype == object or pd.api.types.is_string_dtype(tr[c])]
    caps = {c: float(np.nanmax(caps_src[c].to_numpy("float64"))) for c in CLIP_COLS if c in feats}
    tr = tr.sample(n=min(N_SAMPLE, len(tr)), random_state=7)
    for c in cats:
        vocab = {v: i for i, v in enumerate(pd.unique(tr[c].astype("object").dropna()))}
        tr[c] = tr[c].astype("object").map(vocab).fillna(-1).astype("int32")
        rk[c] = rk[c].astype("object").map(vocab).fillna(-1).astype("int32")
    Xtr, ytr = tr[feats], tr["y"].to_numpy("float64")
    Xrk = rk[feats].copy()
    Xcl = Xrk.copy()
    touched = np.zeros(len(Xrk), bool)
    for c, cap in caps.items():
        v = Xcl[c].to_numpy("float64")
        m = np.isfinite(v) & (v > cap)
        touched |= m
        Xcl.loc[m, c] = cap
    print(f"[q4e] train {Xtr.shape} (sample of the twelve months) rounds {ROUNDS}; "
          f"clip touches {int(touched.sum())} ranking rows; {time.time()-t0:.0f}s", flush=True)

    rows = []
    for label, extra in (("plain trees", {}),
                         ("linear leaves", dict(linear_tree=True, linear_lambda=1.0))):
        p = dict(MAIN_PARAMS, num_threads=THREADS, **extra)
        p.pop("n_estimators", None)
        mdl = lgb.train(p, lgb.Dataset(Xtr, label=ytr, categorical_feature=cats),
                        num_boost_round=ROUNDS)
        a, b = mdl.predict(Xrk), mdl.predict(Xcl)
        d = b - a
        rows.append(dict(learner=label, rounds=ROUNDS, train_rows=len(Xtr),
                         n_touched=int(touched.sum()),
                         n_changed=int((np.abs(d) > 1e-9).sum()),
                         max_abs_change=float(np.abs(d).max()),
                         mean_abs_change_on_touched=float(np.abs(d[touched]).mean()),
                         mean_change_on_touched=float(d[touched].mean()),
                         pred_raw_on_touched=float(a[touched].mean()),
                         pred_clip_on_touched=float(b[touched].mean()),
                         seconds=round(time.time() - t0)))
        print(f"[q4e] {label}: {rows[-1]}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "q4e_linear_leaf.csv", index=False)
    print("\n=== can a predict-time clip move the prediction? ===")
    print(T.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
