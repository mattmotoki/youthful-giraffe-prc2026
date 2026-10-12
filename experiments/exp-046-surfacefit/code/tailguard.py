"""A label-free guard for the rows where the observation cannot explain the label.

`analysis/lirf-july.json` found the mechanism behind every july tail failure in this block.
Two facts, both on the fold side:

* **LIRF's july tail is SERVED where every other airport's is dark.**  226 of LIRF's 390
  july tail rows carry an observation and can move, against 12 of EGLL's 173 and 4 of
  EDDF's 13 -- because LIRF's july clock share is 0.853, the highest in the fold.
* **In the tail the observation carries no signal, anywhere.**  `corr(obs_taxi, y)` falls
  from 0.599 in LIRF's july bulk and 0.723 in every other airport's to **0.018** and
  **0.013** in the two tails.  A tail row is an aircraft that WAITED, not one that taxied
  slowly: LIRF's july tail shows a 467 s ground run against a 5,048 s label, a gap of
  4,005 s the ground run cannot see (714 s in the bulk).

So the arm applies a bulk-fitted correction on rows whose input is uninformative, and a
bigger model does it more confidently -- which is the capacity gradient.

The tail itself is a label (`y > 3600`) and cannot be a serving rule.  But the SHAPE that
identifies those rows is label-free and is on every ranking row too: **the base predicts a
long taxi while the observation shows a short run**.  This prices `pbase - obs_taxi` as a
guard -- hold the update above a threshold learned on the FITTING months only -- on the
shipped arm and on the 255-leaf setting the tail clause stopped, with no refit.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/tailguard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parent.parent

import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402

ARMS = ["c_lgb_shipped@leaves_127x300_lr050", "c_lgb_shipped@leaves_255x300_lr050"]
#: climbed until it turns over or fails a clause, then stopped -- the same
#: discipline the desk set for the leaf climb (2026-09-08 18:40 UTC)
QS = [95.0, 99.0, 99.5, 99.8, 99.9, 99.95]
OUT = EXP / "analysis/tail-guard.json"


def main() -> None:
    rng = np.random.default_rng(A.SEED)
    pan = A.panel_with_block(BA.load())
    out = {"what": __doc__.strip().splitlines()[0],
           "guard": "hold the update where pbase - obs_taxi exceeds a fitting-month quantile",
           "label_free": ("the threshold is a quantile of pbase - obs_taxi over the "
                          "FITTING months' with-clock rows; `y` never enters it"),
           "no_refit": "the saved update vector with a mask", "arms": {}}

    def gap(f: str) -> np.ndarray:
        p = pan[f]
        return (p.pbase.to_numpy("float64")
                - pd.to_numeric(p.obs_taxi, errors="coerce").to_numpy("float64"))

    def cut(f: str, q: float) -> float:
        """The quantile over the two FITTING folds, never the scored one."""
        v = np.concatenate([gap(g)[pan[g].usable.to_numpy(bool)]
                            for g in A.FOLDS if g != f])
        return float(np.nanpercentile(v, q))

    for arm in ARMS:
        z = S.npz_path(arm)
        if not z.exists():
            continue
        full = {f: np.load(z)[f].astype("float64") for f in A.FOLDS}
        rec = {}
        for q in [None] + QS:
            kept, zeroed, cuts = {}, {}, {}
            for f in A.FOLDS:
                d = full[f].copy()
                if q is None:
                    m = np.zeros(len(d), bool)
                else:
                    c = cut(f, q)
                    cuts[f] = round(c, 1)
                    g = gap(f)
                    m = np.isfinite(g) & (g > c)
                zeroed[f] = int((d[m] != 0).sum())
                d[m] = 0.0
                kept[f] = d
            pr = BA.price(pan, kept, rng)
            tc = {f: A.tail_clauses(pan[f], kept[f])["registered_clause_y_over_3600s"]
                  for f in A.FOLDS}
            name = "none" if q is None else f"gap_p{q:g}"
            rec[name] = {
                "board_all_row": pr["board_all_row"], "board_clean": pr["board_clean"],
                "board_december": pr.get("board_december"),
                "every_fold_interval_clear": pr["every_fold_interval_clear"],
                "cut_s_by_fold": cuts, "rows_returned_to_the_base_by_fold": zeroed,
                "tail_clauses": {f: {k: tc[f][k] for k in
                                     ("rmse_gain_s", "passes_the_clause", "rows",
                                      "rows_moved")} for f in A.FOLDS}}
            g_ = pr["board_all_row"]["gain_s"]
            ci = pr["board_all_row"]["interval"]["ci95"]
            t = {f: round(tc[f]["rmse_gain_s"], 3) for f in A.FOLDS}
            ok = all(tc[f]["passes_the_clause"] for f in A.FOLDS)
            print(f"{arm.split('@')[-1]:22s} {name:10s} all-row {g_:+8.4f} "
                  f"[{ci[0]:+7.3f},{ci[1]:+7.3f}] clean {pr['board_clean']['gain_s']:+8.4f} "
                  f"held {zeroed} tail {t} {'ALL PASS' if ok else 'TAIL FAILS'}", flush=True)
            out["arms"][arm] = rec
            OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # what the guard would hold on the ranking side, at the same quantiles
    dep, _ = S.ranking_block()
    bs = S.BASE_SETS["v12"]
    base = pd.read_parquet(bs["price"]).set_index("MVT_ID_mvt")[S.D.TARGET]
    pbase = dep.MVT_ID_mvt.map(base).to_numpy("float64")
    served = dep.usable.to_numpy(bool) & ~dep.noac.to_numpy(bool)
    g = pbase - pd.to_numeric(dep.obs_taxi, errors="coerce").to_numpy("float64")
    ap = dep.airport.to_numpy().astype(str)
    rk = {}
    for q in QS:
        v = np.concatenate([gap(f)[pan[f].usable.to_numpy(bool)] for f in A.FOLDS])
        c = float(np.nanpercentile(v, q))
        m = served & np.isfinite(g) & (g > c)
        rk[f"gap_p{q:g}"] = {
            "cut_s": round(c, 1), "rows_returned_to_the_base": int(m.sum()),
            "share_of_the_served_class": round(float(m.sum() / served.sum()), 4),
            "by_airport": {str(a): int(((ap == a) & m).sum())
                           for a in sorted(set(ap[m]))}}
    out["ranking_side"] = rk
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print(json.dumps(rk, indent=1)[:900])
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
