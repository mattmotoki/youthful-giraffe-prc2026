"""Do the day-state columns explain anything the model has NOT already got?

The paired arms in `run_ds.py` answer the question the assignment asks. This
screen answers a cheaper, sharper version of it first, and it is the same
instrument `q2j_july.md` used on the surface columns: for each day-state column,
the fraction of the baseline's NM-bulk squared error that a 20-bin function of
that column removes ON TOP of the model's prediction, cross-fitted odd day /
even day so no row's own residual sets its own correction.

A column the model already reconstructs from the row's own values scores ~0 here
even if `q2_daystate.md` showed it ranks bad days well -- which is exactly the
distinction finding 5 draws between marking a day and improving a prediction.
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
import prc.day_state_features as DS  # noqa: E402
import common as C  # noqa: E402
import q1c_trimmed as q1c  # noqa: E402
from q2j_july import cv_explained, intercept_gain  # noqa: E402

OUT = HERE.parent / "analysis" / "q3a_screen.md"


def main():
    P = ["# exp-007 question 3a — what the day-state columns add on top of the model\n",
         __doc__.strip().split("\n\n", 1)[1], "",
         "| fold | column | SSE removed | RMSE removed | Spearman with the residual |",
         "|---|---|---|---|---|"]
    summary = {}
    for fold in C.FOLDS:
        rows = pd.read_parquet(HERE.parent / "work" / f"rows_{fold}.parquet")
        rows = rows[~q1c.rollover_mask(rows)]
        f = D.FOLDS[fold]
        train, masked, _ = D.fold_frames(f)
        F = DS.DayState().fit(train).transform(masked)
        dep = masked[masked["PHASE_mvt"].eq("DEP")]
        F = F.assign(**{D.ID: dep[D.ID].to_numpy()})
        m = rows.merge(F, on=D.ID, how="left", validate="1:1")
        core = m[(m["apt"] != "LIRF") & (~m["no_nm"]) & (~m["noac"]) & (m["y"] <= 3600)]
        res = []
        r = core["err"].to_numpy()
        ig = intercept_gain(r, core["day"].to_numpy())
        summary.setdefault("intercept", {})[fold] = ig
        for c in DS.COLUMNS:
            frac, secs = cv_explained(r, core[c].to_numpy(dtype="float64"), core["day"].to_numpy())
            if np.isfinite(frac):
                res.append((c, frac, secs,
                            float(pd.Series(r).corr(pd.Series(core[c].to_numpy(dtype="float64")),
                                                    method="spearman"))))
        R = pd.DataFrame(res, columns=["column", "frac", "secs", "rho"]).sort_values("frac", ascending=False)
        summary[fold] = R
        R.to_csv(HERE.parent / "work" / f"q3a_screen_{fold}.csv", index=False)
        for _, x in R.head(6).iterrows():
            P.append(f"| {fold} | `{x['column']}` | {100*x['frac']:.2f} % | {x['secs']:.2f} s | {x['rho']:+.3f} |")
        P.append(f"| {fold} | *(best of the other {len(R)-6})* | "
                 f"{100*R['frac'].iloc[6]:.2f} % | {R['secs'].iloc[6]:.2f} s | |")
    P.append("")
    P.append("The bin means are centred, so a column can only score on VARIATION. For scale, a "
             "cross-fitted CONSTANT correction -- the model's global bias -- is worth "
             + ", ".join(f"{100*summary['intercept'][f][0]:.2f} % on {f}" for f in C.FOLDS)
             + ": that is what an uncentred screen would have added to every column, noise "
               "included.\n")
    P.append("And for scale against the rest of the ranking frame, the same centred screen over "
             "the 99 surface and own-flight columns put the best at 0.73 % of the july NM-bulk "
             "squared error (`analysis/q2j_july.md`), and that column was `span`, which the model "
             "already holds.\n")
    OUT.write_text("\n".join(P))
    print(OUT.read_text())


if __name__ == "__main__" and "--interactions" not in sys.argv:
    main()


# ---------------------------------------------------------------------------
# Interaction probe, added after the univariate screen came back near zero.
# ---------------------------------------------------------------------------
def cv_explained_2d(resid, x, z, day, nx=8, nz=8):
    """Same cross-fitted screen, but on a 2-D grid of (x, z).

    The univariate screen cannot see a day-state column used as a MODIFIER of the
    row's own value. This one can: it fits a mean residual per (x-bin, z-bin) cell
    on the odd days and applies it to the even days, and the reverse. The cell
    means are CENTRED on the fitting half, for the reason `cv_explained`
    documents: otherwise every grid removes the model's global bias and is
    credited with it.
    """
    ok = np.isfinite(x) & np.isfinite(z) & np.isfinite(resid)
    r, xv, zv, d = resid[ok], x[ok], z[ok], day[ok]
    parity = (pd.to_datetime(pd.Series(d).reset_index(drop=True)).dt.dayofyear % 2).to_numpy()
    corr = np.zeros_like(r)
    for p in (0, 1):
        fit, use = parity != p, parity == p
        ex = np.unique(np.quantile(xv[fit], np.linspace(0, 1, nx + 1)))
        ez = np.unique(np.quantile(zv[fit], np.linspace(0, 1, nz + 1)))
        if len(ex) < 3 or len(ez) < 3:
            return np.nan, np.nan
        bf = (np.clip(np.digitize(xv[fit], ex[1:-1]), 0, len(ex) - 2) * 100
              + np.clip(np.digitize(zv[fit], ez[1:-1]), 0, len(ez) - 2))
        bu = (np.clip(np.digitize(xv[use], ex[1:-1]), 0, len(ex) - 2) * 100
              + np.clip(np.digitize(zv[use], ez[1:-1]), 0, len(ez) - 2))
        means = pd.Series(r[fit]).groupby(bf).mean() - float(r[fit].mean())
        corr[use] = pd.Series(bu).map(means).fillna(0.0).to_numpy()
    before = float(np.mean(r ** 2))
    after = float(np.mean((r - corr) ** 2))
    return 1 - after / before, np.sqrt(before) - np.sqrt(after)


def interactions():
    lines = ["\n## Interaction probe\n",
             "The screen above is additive and univariate. This one fits a mean residual per",
             "(own-value bin, day-state bin) cell on the odd days and applies it to the even days,",
             "and the reverse: it can see a day-state column used as a *modifier* of the row's own",
             "value, which is the only way the paired arms could beat the univariate screen.\n",
             "| fold | own value | day state | SSE removed | RMSE removed |", "|---|---|---|---|---|"]
    pairs = [("span", "ds_span_std_z"), ("span", "ds_share_span1800_z"),
             ("span", "ds_deice_hr"), ("gap_al", "ds_gap_p90_z"),
             ("span_sched", "ds_sched_med_z"), ("span", "ds_nm_rate_z")]
    for fold in C.FOLDS:
        rows = pd.read_parquet(HERE.parent / "work" / f"rows_{fold}.parquet")
        rows = rows[~q1c.rollover_mask(rows)]
        f = D.FOLDS[fold]
        train, masked, _ = D.fold_frames(f)
        F = DS.DayState().fit(train).transform(masked)
        dep = masked[masked["PHASE_mvt"].eq("DEP")]
        F = F.assign(**{D.ID: dep[D.ID].to_numpy()})
        m = rows.merge(F, on=D.ID, how="left", validate="1:1")
        core = m[(m["apt"] != "LIRF") & (~m["no_nm"]) & (~m["noac"]) & (m["y"] <= 3600)]
        r = core["err"].to_numpy()
        # Same-shape control: the day-state axis SHUFFLED, which keeps the grid's
        # flexibility and the column's marginal distribution but destroys its link
        # to the day. Anything a real pair does not beat is grid flexibility.
        rng = np.random.default_rng(3)
        shuf = rng.permutation(core["ds_span_std_z"].to_numpy("float64"))
        base_frac, base_secs = cv_explained_2d(r, core["span"].to_numpy("float64"), shuf,
                                               core["day"].to_numpy())
        lines.append(f"| {fold} | `span` | **control: `ds_span_std_z` shuffled** | "
                     f"{100*base_frac:.2f} % | {base_secs:.2f} s |")
        for own, ds in pairs:
            frac, secs = cv_explained_2d(r, core[own].to_numpy("float64"),
                                         core[ds].to_numpy("float64"), core["day"].to_numpy())
            lines.append(f"| {fold} | `{own}` | `{ds}` | {100*frac:.2f} % | {secs:.2f} s |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__" and "--interactions" in sys.argv:
    OUT.write_text(OUT.read_text() + interactions())
    print(interactions())
