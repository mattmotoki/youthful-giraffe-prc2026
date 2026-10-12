"""Question 4: is the model's under-prediction of long taxis shrinkage?

A squared-error learner with L2 and a finite tree depth produces predictions
that are shrunk toward the conditional mean: regressing the truth on the
prediction then gives a slope above 1. This file measures that slope directly on
the saved out-of-fold predictions, overall and inside slices, so the hypothesis
is tested before anything is built on it.

The calibration slope is estimated on the CONTROL arm's own rows; it is a
description of that arm, not a correction fitted for later use. Where a slope is
quoted as a possible fix, it is cross-fitted odd day / even day, so no row's own
residual sets its own correction.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import prc.data as D  # noqa: E402
import common as C  # noqa: E402
import q1c_trimmed as q1c  # noqa: E402

OUT = HERE.parent / "analysis" / "q4_shrinkage.md"


def slope(y, p):
    """OLS slope of y on p. >1 means the predictions are shrunk toward the mean."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    v = p.var()
    return float(np.cov(y, p, bias=True)[0, 1] / v) if v > 0 else np.nan


def cv_recalibrate(y, p, day):
    """RMSE after an affine recalibration fitted odd day / even day."""
    parity = (pd.to_datetime(pd.Series(day).reset_index(drop=True)).dt.dayofyear % 2).to_numpy()
    out = p.copy()
    for q in (0, 1):
        fit, use = parity != q, parity == q
        b = slope(y[fit], p[fit])
        a = y[fit].mean() - b * p[fit].mean()
        out[use] = a + b * p[use]
    return C.rmse(y - out), out


def main():
    P = ["# exp-007 question 4 — is the under-prediction of long taxis shrinkage?\n",
         "Regressing the truth on the prediction: a slope above 1 says the predictions are",
         "compressed toward the conditional mean, which is the signature the question names.",
         "Measured on the baseline out-of-fold line with the two unreachable rows removed.\n",
         "| fold | slice | rows | RMSE | calibration slope | intercept | RMSE after an affine recalibration (odd/even day) |",
         "|---|---|---|---|---|---|---|"]
    for fold in C.FOLDS:
        d = pd.read_parquet(HERE.parent / "work" / f"rows_{fold}.parquet")
        d = d[~q1c.rollover_mask(d)]
        nl = d[d["apt"] != "LIRF"]
        core = nl[(~nl["no_nm"]) & (~nl["noac"])]
        slices = {
            "core, other nine": core,
            "  of which target ≤ 3600 s": core[core["y"] <= 3600],
            "  of which target > 1800 s": core[core["y"] > 1800],
            "  of which span > 1800 s": core[core["span"] > 1800],
            "  of which on the worst 5 % of days": None,
        }
        wd = C.worst_days(core, 0.05)
        key = pd.MultiIndex.from_frame(core[["apt", "day"]])
        slices["  of which on the worst 5 % of days"] = core[key.isin(wd)]
        for name, s in slices.items():
            b = slope(s["y"], s["pred"])
            a = s["y"].mean() - b * s["pred"].mean()
            rec, _ = cv_recalibrate(s["y"].to_numpy(), s["pred"].to_numpy(), s["day"])
            P.append(f"| {fold} | {name} | {len(s):,} | {C.rmse(s['err']):.1f} | **{b:.3f}** | "
                     f"{a:+.0f} s | {rec:.1f} |")
    P.append("")
    P.append("## What the slopes say\n")
    P.append("A slope of 1.000 is a calibrated predictor. The table above is the test; the "
             "reading is written in RESULTS.md finding 5.\n")

    # Per-airport-day: is there a day effect the model misses, and is it predictable?
    P.append("## The day effect: how much would a per-airport-day intercept be worth?\n")
    P.append("An oracle that added each airport-day's own mean residual to every row of that day. "
             "This is the ceiling on any day-severity second stage, including a random effect; it "
             "uses the validation month's labels and is not a model. The leave-one-out column "
             "removes the row's own residual from the day mean it is then corrected by, which is "
             "the honest version -- with ~500 rows a day the two barely differ, and the point of "
             "showing both is that the optimistic one is already small.\n")
    P.append("| fold | slice | rows | RMSE | after an oracle day intercept | the same, leave-one-out | days |")
    P.append("|---|---|---|---|---|---|---|")
    for fold in C.FOLDS:
        d = pd.read_parquet(HERE.parent / "work" / f"rows_{fold}.parquet")
        d = d[~q1c.rollover_mask(d)]
        nl = d[d["apt"] != "LIRF"]
        core = nl[(~nl["no_nm"]) & (~nl["noac"])].copy()
        for name, s in (("core, other nine", core),
                        ("bulk ≤3600", core[core["y"] <= 3600])):
            g = s.groupby(["apt", "day"])["err"]
            eff = g.transform("mean")
            # leave-one-out: the row's own residual removed from its own day mean,
            # which is the honest version -- the "in" column above is optimistic
            # because a row helps set the intercept it is then corrected by.
            tot, n = g.transform("sum"), g.transform("size")
            loo = (tot - s["err"]) / (n - 1).clip(lower=1)
            P.append(f"| {fold} | {name} | {len(s):,} | {C.rmse(s['err']):.1f} | "
                     f"**{C.rmse(s['err'] - eff):.1f}** | {C.rmse(s['err'] - loo):.1f} | "
                     f"{s.groupby(['apt','day']).ngroups} |")
    P.append("")
    P.append("And the honest version of the same thing: the day intercept fitted on the "
             "fold's *training* months cannot exist (there are no out-of-fold residuals for a "
             "training month), so the only implementable second stage predicts the day effect "
             "from label-free day state. `analysis/q2_daystate.md` measured how well those "
             "columns rank days; the ceiling above says what a perfect ranking would be worth.\n")
    OUT.write_text("\n".join(P))
    print(OUT.read_text())


if __name__ == "__main__":
    main()
