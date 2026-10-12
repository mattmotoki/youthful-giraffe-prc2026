"""RESULTS section 47.68: lever C on a YEAR-CROSSING cell key, stages 0-2, no fit.

Registered at 47.67 before this ran.  Two keys (airport x hour x dow, airport x hour) and two
readings per key:

  (a) WITHIN-FOLD  -- the cell constant from the same month's other rows, self-excluded.
                      Comparable to 47.66's +0.2400 and NOT board-relevant on its own.
  (b) CROSS-MONTH  -- the constant estimated ENTIRELY from the other two fold months and
                      applied to the held-out month.  The board's geometry; the headline.

(b) needs no self-exclusion: every donor is in a different month from every scored row.  Both
are priced with the record's own statistic, `2 mean(r d) - mean(d d)`, day-clustered.

    .venv/bin/python experiments/exp-046-surfacefit/code/leverc2.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import base as BA                                                       # noqa: E402
import arms as A                                                        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "queue_base", ROOT / "experiments/exp-047-queue/code/base.py")
QB = importlib.util.module_from_spec(_spec)
sys.modules["queue_base"] = QB
_spec.loader.exec_module(QB)

ARM = EXP / "work/d-seasonal-bag10-255-free.npz"
OUT = EXP / "analysis/leverc-yearcrossing.json"
SEED, BODY, FLOOR_S = 47, 3600.0, 0.1
CJ = {"july_loo": 0.0009318464575330257, "january": 0.00074096, "december": 0.0}
KEYS = {"apt_hour_dow": ("hour", "dow"), "apt_hour": ("hour",)}


def build(pan: dict, arm) -> dict:
    d = {}
    for f in BA.FOLDS:
        p = pan[f]
        y = p.y_true.to_numpy("float64")
        a = arm[f].astype("float64")
        cls = (a != 0) & (y <= BODY)
        t = pd.to_datetime(p.t.to_numpy("float64"), unit="s", utc=True)
        d[f] = {"r": (p.rbase.to_numpy("float64") - a)[cls],
                "apt": p.airport.astype(str).to_numpy()[cls],
                "hour": t.hour.to_numpy()[cls], "dow": t.dayofweek.to_numpy()[cls],
                "day": p.day.to_numpy()[cls],
                "class_rows": int(cls.sum()), "fold_rows": int(len(p))}
    return d


def keyed(dd: dict, parts: tuple) -> np.ndarray:
    cols = [pd.Series(dd["apt"])] + [pd.Series(dd[k]).astype(str) for k in parts]
    return cols[0].str.cat(cols[1:], sep="|").to_numpy()


def main() -> None:
    pan = BA.load()
    arm = np.load(ARM)
    D = build(pan, arm)
    rng = np.random.default_rng(SEED)
    out = {"what": __doc__.strip().splitlines()[0], "seed": SEED, "keys": {}}

    for kname, parts in KEYS.items():
        rec = {"by_fold": {}}
        for f in BA.FOLDS:
            dd = D[f]
            key = keyed(dd, parts)
            r = dd["r"]
            code, uniq = pd.factorize(key)
            code = code.astype('int64')
            ncode = int(len(uniq))
            cnt = np.bincount(code, minlength=ncode)
            per_row = cnt[code]

            # ---- stage 1: the shared component, queue's estimator ---------------------
            sh = QB.shared(r, code, ncode, dd["day"], rng)
            mean_r2 = float(np.mean(r ** 2))
            tau2 = float(sh["value"]) if sh["value"] is not None else 0.0
            s = tau2 / mean_r2
            sig2 = max(mean_r2 - tau2, 1e-9)

            # ---- (a) WITHIN-FOLD: self-excluded cell mean, shrunk --------------------
            tot = np.bincount(code, weights=r, minlength=ncode)
            loo = np.where(per_row > 1, (tot[code] - r) / np.maximum(per_row - 1, 1), 0.0)
            lam_a = np.where(per_row > 1,
                             tau2 / (tau2 + sig2 / np.maximum(per_row - 1, 1)), 0.0)
            d_a = lam_a * loo

            # ---- (b) CROSS-MONTH: donors are the OTHER TWO fold months ---------------
            don_sum, don_n = {}, {}
            for g in BA.FOLDS:
                if g == f:
                    continue
                kg = keyed(D[g], parts)
                sg = pd.Series(D[g]["r"]).groupby(kg).sum()
                ng = pd.Series(D[g]["r"]).groupby(kg).size()
                for k2, v in sg.items():
                    don_sum[k2] = don_sum.get(k2, 0.0) + float(v)
                for k2, v in ng.items():
                    don_n[k2] = don_n.get(k2, 0) + int(v)
            nd = np.array([don_n.get(k2, 0) for k2 in key], "float64")
            mu = np.array([don_sum.get(k2, 0.0) / don_n[k2] if don_n.get(k2, 0) > 0 else 0.0
                           for k2 in key], "float64")
            lam_b = np.where(nd > 0, tau2 / (tau2 + sig2 / np.maximum(nd, 1)), 0.0)
            d_b = lam_b * mu

            # ---- kappa: do the cell constants agree between months? -----------------
            own = np.where(per_row > 1, (tot[code] - r) / np.maximum(per_row - 1, 1), np.nan)
            ok = np.isfinite(own) & (nd > 0)
            kappa = (float(np.corrcoef(own[ok], mu[ok])[0, 1]) if ok.sum() > 10 else None)

            price = {}
            for lab, dv in (("within_fold", d_a), ("cross_month", d_b)):
                pr = QB.paired(r, dv, dd["day"], rng)
                pr["board_s_this_fold"] = (CJ[f] * pr["mse_gain_s2"]
                                           * dd["class_rows"] / dd["fold_rows"])
                pr["ci95_board_s_this_fold"] = [
                    CJ[f] * v * dd["class_rows"] / dd["fold_rows"] for v in pr["ci95"]]
                price[lab] = pr

            # ---- the two MANDATORY controls (47.65.5), on the within-fold read ------
            # 1. target placebo: residuals permuted within airport x day, key untouched.
            # 2. shuffled cell: rows reassigned to random cells of the SAME size profile.
            ctl = {}
            g = np.random.default_rng(SEED + 1)
            perm = np.empty_like(r)
            akey = pd.Series(dd["apt"]).str.cat(pd.Series(dd["day"]), sep="|").to_numpy()
            for _, idx in pd.Series(np.arange(len(r))).groupby(akey):
                iv = idx.to_numpy()
                perm[iv] = r[g.permutation(iv)]
            shuf = g.permutation(code)              # same cell-size profile by construction
            for lab, (rr2, cc) in (("target_placebo", (perm, code)),
                                   ("shuffled_cell", (r, shuf))):
                tt = np.bincount(cc, weights=rr2, minlength=ncode)
                pc = np.bincount(cc, minlength=ncode)[cc]
                lo2 = np.where(pc > 1, (tt[cc] - rr2) / np.maximum(pc - 1, 1), 0.0)
                lm = np.where(pc > 1, tau2 / (tau2 + sig2 / np.maximum(pc - 1, 1)), 0.0)
                pr = QB.paired(rr2, lm * lo2, dd["day"], rng)
                pr["board_s_this_fold"] = (CJ[f] * pr["mse_gain_s2"]
                                           * dd["class_rows"] / dd["fold_rows"])
                ctl[lab] = pr

            rec["by_fold"][f] = {
                "controls": ctl,
                "class_rows": dd["class_rows"], "fold_rows": dd["fold_rows"],
                "cells": int(ncode),
                "row_weighted_n": float(per_row.mean()),
                "row_weighted_n_donor": float(nd.mean()),
                "rows_with_no_donor_cell": int((nd == 0).sum()),
                "shared_component_s2": sh, "mean_r2_s2": mean_r2, "reachable_share": s,
                "mean_lambda_star_within": float(lam_a.mean()),
                "mean_lambda_star_cross": float(lam_b.mean()),
                "kappa_cross_month_cell_means": kappa,
                "price": price}

        for lab in ("within_fold", "cross_month"):
            pt = sum(rec["by_fold"][f]["price"][lab]["board_s_this_fold"] for f in BA.FOLDS)
            lo = sum(rec["by_fold"][f]["price"][lab]["ci95_board_s_this_fold"][0]
                     for f in BA.FOLDS)
            hi = sum(rec["by_fold"][f]["price"][lab]["ci95_board_s_this_fold"][1]
                     for f in BA.FOLDS)
            rec[f"board_s_{lab}"] = {"point": pt, "lo": lo, "hi": hi}
        rec["CROSS_MONTH_clears_the_floor_at_its_bottom"] = bool(
            rec["board_s_cross_month"]["lo"] >= FLOOR_S)
        out["keys"][kname] = rec

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    print(f"written {OUT.relative_to(ROOT)}\n")
    for kname in KEYS:
        rec = out["keys"][kname]
        print(f"===== {kname}")
        for f in BA.FOLDS:
            b = rec["by_fold"][f]
            k = b["kappa_cross_month_cell_means"]
            print(f"  {f:10} cells {b['cells']:5,}  n {b['row_weighted_n']:7.1f}  "
                  f"n_donor {b['row_weighted_n_donor']:8.1f}  s {b['reachable_share']:.4f}  "
                  f"lam_w {b['mean_lambda_star_within']:.3f} lam_x {b['mean_lambda_star_cross']:.3f}  "
                  f"kappa {k if k is None else round(k,3)}")
        for lab in ("within_fold", "cross_month"):
            v = rec[f"board_s_{lab}"]
            print(f"    {lab:12} board {v['point']:+.4f} s  [{v['lo']:+.4f}, {v['hi']:+.4f}]")
        print(f"    CROSS-MONTH clears the 0.1 s floor at its bottom: "
              f"{rec['CROSS_MONTH_clears_the_floor_at_its_bottom']}\n")


if __name__ == "__main__":
    main()
