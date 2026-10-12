"""Order 4: is the 2026 instrument's per-ROW observation better, and does that buy gain?

Desk order, 2026-09-09, after BOARD READING 15:

    a label-free measurement of the 2026 instrument's per-row observation quality against
    the fold months, as the test of the standing candidate.

**The candidate under test** (RESULTS 8.1 / 8.4).  The 2026 instrument reads better than
the 2025 folds' on every MONTH-level figure -- more rows matched, more clocks, the observed
take-off nearer the recorded one, 84 to 88 % of clocks inside a minute against 76 %.  The
standing explanation of the transfer multiplier is that a fold-fitted per-row gain is then
applied to rows whose observation is CLEANER than the ones it was priced on.  A month-level
table cannot test that: it says the populations differ, not that the difference buys gain.

**The test, in three steps, none of which reads a 2026 label or a board number.**

1.  Per-ROW observation quality, on the served rows of both sides, in columns that exist in
    2025 and 2026 alike and are computed from the aircraft's own ground run.
2.  Does quality buy gain *inside the folds*?  Each column's fold rows are cut at their own
    quintiles and the SERVED ARM's per-row MSE gain is priced inside each stratum, with the
    day-clustered interval the gate uses.  Nothing is refitted: the arm's fold vector was
    saved by `capacity.py`, and this file only re-prices it on subsets.
3.  Re-weight the fold gain to the 2026 quality mix.  Marginally, per column, by moving the
    fold strata weights to the ranking months' shares; and jointly, by a density ratio from
    a propensity model over all the quality columns at once (label-free: its target is which
    side a row is on, never the taxi-out).  The ratio to the fold's own mean gain is what
    the instrument story PREDICTS the multiplier should be.

The prediction is then set beside the four board readings.  It is a prediction, not a
revision: score-feedback rule stands and nothing served moves on it.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/instrument.py
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
ROOT = EXP.parents[1]

# `serve` FIRST: `arms` pulls in `imports.load`, which puts other experiments' code
# directories at the front of sys.path, and one of them also has a `serve.py`
import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402

#: the arm the lane ships (v15's head, and v16's base), priced over v12
ARM = "leaves_127x300_lr050"
HOLD = ("LFPG",)
NQ = 5
OUT = EXP / "analysis/instrument.json"

#: PER-ROW OBSERVATION QUALITY.  Every column is computed from the aircraft's own ground
#: run, exists on both sides, and carries no label.  `dt_takeoff` is the disagreement
#: between the observed take-off and the recorded movement time -- 8.1's "median clock
#: offset" and "within 60 s" at row grain.  The segment columns say how much of the run the
#: network actually heard, and how big its worst gap was.  `matched_on_modal` and
#: `obs_ambiguous` are the join's own confidence.  A LOW value is a BETTER observation for
#: the first five; the last three are read in their own direction and stated per column.
QUALITY = {
    "abs_dt_takeoff_s": ("|observed take-off - recorded movement time|", "low_is_better"),
    "seg_gap_max_s": ("the worst reception gap inside the ground run", "low_is_better"),
    "seg_dt_median_s": ("the median spacing between heard points", "low_is_better"),
    "seg_points": ("how many points of the run were heard", "high_is_better"),
    "n_ground_points": ("how many of them were on the ground", "high_is_better"),
    "time_to_250m": ("time from the stand-departure onset to 250 m", "low_is_better"),
    "stand_stop_s": ("how long the aircraft was seen stopped at the stand", "high_is_better"),
    "matched_on_modal": ("the join used the segment's modal callsign form", "high_is_better"),
}


def served(d: pd.DataFrame) -> np.ndarray:
    """The rows this lane may move, LFPG held out as the serve holds it."""
    return (d["usable"].to_numpy(bool)
            & ~np.isin(d["airport"].to_numpy().astype(str), list(HOLD)))


def quality_frame(d: pd.DataFrame) -> pd.DataFrame:
    """The quality columns for one side, on its own index."""
    out = pd.DataFrame(index=d.index)
    out["abs_dt_takeoff_s"] = d["dt_takeoff"].astype("float64").abs()
    for c in ("seg_gap_max_s", "seg_dt_median_s", "seg_points", "n_ground_points",
              "time_to_250m", "stand_stop_s", "matched_on_modal"):
        out[c] = d[c].astype("float64")
    return out


def strata(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Quintile index, with a separate top stratum for a missing value."""
    k = np.digitize(x, edges, right=False)
    k = np.clip(k, 0, len(edges))
    return np.where(np.isnan(x), len(edges) + 1, k)


def main() -> None:
    rng = np.random.default_rng(A.SEED)
    pan = A.panel_with_block(BA.load())
    vec = dict(np.load(EXP / f"work/d-capacity-{ARM}.npz"))

    # ---- the fold side, pooled over the three folds on the rows the lane serves -------
    fr = []
    for f in A.FOLDS:
        p = pan[f]
        m = served(p)
        q = quality_frame(p)[m]
        q["r"] = p["rbase"].to_numpy("float64")[m]
        q["d"] = np.asarray(vec[f], "float64")[m]
        q["day"] = p["day"].to_numpy()[m]
        q["airport"] = p["airport"].to_numpy().astype(str)[m]
        q["fold"] = f
        fr.append(q)
    fold = pd.concat(fr, ignore_index=True)

    # ---- the ranking side, the same columns on the same class ------------------------
    dep, _ = S.ranking_block()
    dep["usable"] = A.served_class(dep)
    rk = quality_frame(dep)[served(dep)].reset_index(drop=True)
    rk["airport"] = dep["airport"].astype(str).to_numpy()[served(dep)]

    out = {
        "what": ("the 2026 instrument's per-row observation quality against the fold "
                 "months, and whether the difference buys gain"),
        "ordered_by": "the desk, 2026-09-09, order 4, after BOARD READING 15",
        "arm_priced": ARM, "held": list(HOLD),
        "no_refit": ("the arm's fold vector was saved by capacity.py on the pinned "
                     "population; this file re-prices it on subsets and re-weights it"),
        "label_free": ("no 2026 label and no board number enters any number below; the "
                       "propensity model's target is which side a row is on"),
        "rows": {"fold_served": int(len(fold)), "ranking_served": int(len(rk))},
        "by_column": {},
    }

    # ---------------------------------------------- step 1 and 2, per quality column --
    for col, (what, sense) in QUALITY.items():
        x, z = fold[col].to_numpy("float64"), rk[col].to_numpy("float64")
        finite = x[np.isfinite(x)]
        edges = np.unique(np.quantile(finite, np.arange(1, NQ) / NQ)) if len(finite) else \
            np.array([0.0])
        kf, kr = strata(x, edges), strata(z, edges)
        rec = {"what": what, "sense": sense,
               "edges": [float(e) for e in edges],
               "fold": {"median": float(np.nanmedian(x)),
                        "missing": float(np.mean(~np.isfinite(x)))},
               "ranking": {"median": float(np.nanmedian(z)),
                           "missing": float(np.mean(~np.isfinite(z)))},
               "strata": {}}
        gains, wf, wr = [], [], []
        for k in sorted(set(kf.tolist()) | set(kr.tolist())):
            sel = kf == k
            g = (BA.paired(fold["r"].to_numpy("float64")[sel],
                           fold["d"].to_numpy("float64")[sel],
                           fold["day"].to_numpy()[sel], rng) if sel.sum() >= 200 else None)
            f_share, r_share = float(sel.mean()), float((kr == k).mean())
            rec["strata"][int(k)] = {
                "fold_rows": int(sel.sum()), "fold_share": round(f_share, 4),
                "ranking_share": round(r_share, 4),
                "mse_gain_s2_per_row": g["mse_gain_s2"] if g else None,
                "ci95": g.get("ci95") if g else None}
            if g:
                gains.append(g["mse_gain_s2"]); wf.append(f_share); wr.append(r_share)
        if gains:
            gains, wf, wr = np.array(gains), np.array(wf), np.array(wr)
            base = float(np.sum(wf * gains) / max(wf.sum(), 1e-12))
            proj = float(np.sum(wr * gains) / max(wr.sum(), 1e-12))
            rec["fold_gain_s2_per_row"] = base
            rec["reweighted_to_2026_s2_per_row"] = proj
            rec["predicted_multiplier"] = round(proj / base, 4) if base else None
        out["by_column"][col] = rec

    # ------------------------------------------------- step 3, the joint density ratio
    feat = list(QUALITY)
    X = pd.concat([fold[feat], rk[feat]], ignore_index=True)
    lab = np.r_[np.zeros(len(fold)), np.ones(len(rk))]
    joint = {}
    for tag, cols in (("quality_only", feat), ("quality_and_airport", feat + ["airport"])):
        Xi = pd.concat([fold[cols], rk[cols]], ignore_index=True).copy()
        if "airport" in cols:
            Xi["airport"] = Xi["airport"].astype("category")
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold

        def _mdl():
            return lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                                      min_child_samples=200, n_jobs=3, verbose=-1,
                                      random_state=A.SEED)

        # CROSS-FITTED (3 folds).  An in-sample propensity separates the two sides better
        # than it can out of sample, which inflates the density ratio and therefore the
        # multiplier this measurement is testing.  Every fold row's weight comes from a
        # model that never saw it, so the number below cannot be flattered that way.
        oof = np.zeros(len(Xi))
        for tr_i, te_i in StratifiedKFold(3, shuffle=True,
                                          random_state=A.SEED).split(Xi, lab):
            m = _mdl()
            m.fit(Xi.iloc[tr_i], lab[tr_i])
            oof[te_i] = m.predict_proba(Xi.iloc[te_i])[:, 1]
        mdl = _mdl()
        mdl.fit(Xi, lab)                      # kept only to report the in-sample AUC
        pr = np.clip(oof[:len(fold)], 1e-4, 1 - 1e-4)
        w = (pr / (1 - pr)) * (len(fold) / max(len(rk), 1))
        # trim the tail so one fold row cannot carry the answer
        cap = float(np.quantile(w, 0.99))
        wt = np.minimum(w, cap)
        r, d = fold["r"].to_numpy("float64"), fold["d"].to_numpy("float64")
        v = 2 * r * d - d * d
        base = float(np.mean(v))
        proj = float(np.sum(wt * v) / np.sum(wt))
        # a day-clustered interval on the RATIO, resampling days as the gate does
        codes, uniq = pd.factorize(fold["fold"].astype(str) + "|" + fold["day"].astype(str))
        k = len(uniq)
        nv = np.bincount(codes, weights=v, minlength=k)
        nc = np.bincount(codes, minlength=k)
        nw = np.bincount(codes, weights=wt * v, minlength=k)
        nwc = np.bincount(codes, weights=wt, minlength=k)
        draws = rng.integers(0, k, size=(BA.BOOT, k))
        rb = ((nw[draws].sum(1) / nwc[draws].sum(1))
              / (nv[draws].sum(1) / nc[draws].sum(1)))
        joint[tag] = {
            "auc_of_the_propensity_out_of_fold": float(
                __import__("sklearn.metrics", fromlist=["roc_auc_score"])
                .roc_auc_score(lab, oof)),
            "auc_of_the_propensity_in_sample": float(
                __import__("sklearn.metrics", fromlist=["roc_auc_score"])
                .roc_auc_score(lab, mdl.predict_proba(Xi)[:, 1])),
            "cross_fitted": "3-fold, every weight from a model that did not see the row",
            "weight_p99_cap": cap,
            "fold_gain_s2_per_row": base,
            "reweighted_to_2026_s2_per_row": proj,
            "predicted_multiplier": round(proj / base, 4) if base else None,
            "ci95": [float(x) for x in np.percentile(rb, [2.5, 97.5])],
        }
    out["joint_density_ratio"] = joint

    out["board_readings_for_comparison"] = {
        "v13": 1.75, "v14": 1.03, "v15": 2.01, "v16": 0.36,
        "note": ("a re-weighting of the SAME rows applies equally to every update served "
                 "on them, so a single instrument multiplier can only explain the part "
                 "the four readings share; it cannot make 1.75 and 0.36 at once"),
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    print(f"{'column':22s} {'fold med':>10s} {'2026 med':>10s} {'fold s2':>9s} "
          f"{'2026 s2':>9s} {'mult':>6s}")
    for c, r in out["by_column"].items():
        if "predicted_multiplier" in r:
            print(f"{c:22s} {r['fold']['median']:10.2f} {r['ranking']['median']:10.2f} "
                  f"{r['fold_gain_s2_per_row']:9.1f} "
                  f"{r['reweighted_to_2026_s2_per_row']:9.1f} "
                  f"{r['predicted_multiplier']:6.3f}")
    for t, r in joint.items():
        print(f"joint {t:28s} auc {r['auc_of_the_propensity_out_of_fold']:.3f}  "
              f"{r['fold_gain_s2_per_row']:9.1f} -> {r['reweighted_to_2026_s2_per_row']:9.1f}"
              f"  x{r['predicted_multiplier']:.3f}  ci {r['ci95'][0]:.3f} {r['ci95'][1]:.3f}")
    print("written", OUT)


if __name__ == "__main__":
    main()
