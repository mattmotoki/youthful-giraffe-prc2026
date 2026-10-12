"""Question 2J (desk re-aim): the same decomposition applied to JULY at the other nine.

July's non-LIRF RMSE is 249.7 s against the leaders' implied 200-240 band, and it
carries no unreachable rollover row, so unlike january it is a clean read of where
our model is actually losing. This file answers: which airports, which days, which
regime, whether take-off minus AOBT being long is the marker, what the model
under-predicts, and what in the ranking frame marks it.

The "what marks it" screen is honest about fitting: every candidate column is
label-free, and its explanatory power is measured by fitting a binned residual
correction on the odd days of the validation month and scoring it on the even
days (and the reverse), so no row's own residual trains its own correction.
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
from prc.surface_features import build_features as sf_build  # noqa: E402

OUT = HERE.parent / "analysis" / "q2j_july.md"


def tbl(df, by, tot_sse, label, sort="sse", top=None):
    g = df.groupby(by, dropna=False, observed=True).agg(
        n=("se", "size"), sse=("se", "sum"), rmse=("err", C.rmse), bias=("err", "mean"),
        ybar=("y", "mean"), spanbar=("span", "median"))
    g["share"] = 100 * g["sse"] / tot_sse
    g = g.sort_values("sse", ascending=False) if sort == "sse" else g.sort_index()
    if top:
        g = g.head(top)
    name = by if isinstance(by, str) else "/".join(by)
    lines = [f"**{label}**", "", f"| {name} | rows | RMSE | SSE % | mean error | mean y | median span |",
             "|---|---|---|---|---|---|---|"]
    for k, r in g.iterrows():
        lines.append(f"| {k} | {int(r['n']):,} | {r['rmse']:,.1f} | **{r['share']:.1f}** | "
                     f"{r['bias']:+,.0f} | {r['ybar']:,.0f} | {r['spanbar']:,.0f} |")
    return "\n".join(lines) + "\n"


def cv_explained(resid, x, day, nbins=20, centre=True):
    """SSE fraction of `resid` a binned function of `x` removes, odd/even-day cross-fitted.

    Returns (fraction removed, seconds of RMSE removed) on the whole slice. A
    column that only looks predictive because it is fitted on its own rows scores
    ~0 here.

    `centre` subtracts the fitting half's mean residual from the bin means, so the
    correction is mean-zero and can only capture VARIATION. Without it every
    column, including pure noise, removes the model's global bias and gets credit
    for it: on the july core that bias is about -11 s against a 227 s RMSE, i.e.
    0.21 % of the squared error, and a Gaussian noise column duly scored +0.206 %
    before this was fixed. The intercept is a real (tiny) improvement but it is
    not a marker, so it is reported separately by `intercept_gain` and removed
    from every column's score.
    """
    ok = np.isfinite(x) & np.isfinite(resid)
    if ok.sum() < 2000:
        return np.nan, np.nan
    r, xv, d = resid[ok], x[ok], day[ok]
    parity = (pd.to_datetime(d).dayofyear % 2).to_numpy()
    corr = np.zeros_like(r)
    for p in (0, 1):
        fit, use = parity != p, parity == p
        if fit.sum() < 500 or use.sum() < 500:
            return np.nan, np.nan
        edges = np.unique(np.quantile(xv[fit], np.linspace(0, 1, nbins + 1)))
        if len(edges) < 4:
            return np.nan, np.nan
        bi_fit = np.clip(np.digitize(xv[fit], edges[1:-1]), 0, len(edges) - 2)
        bi_use = np.clip(np.digitize(xv[use], edges[1:-1]), 0, len(edges) - 2)
        means = pd.Series(r[fit]).groupby(bi_fit).mean()
        if centre:
            means = means - float(r[fit].mean())
        corr[use] = pd.Series(bi_use).map(means).fillna(0.0).to_numpy()
    before = float(np.mean(r ** 2))
    after = float(np.mean((r - corr) ** 2))
    return 1 - after / before, np.sqrt(before) - np.sqrt(after)


def intercept_gain(resid, day):
    """What an odd/even-day cross-fitted CONSTANT correction alone is worth.

    This is the floor every column in the screen would have scored if the bin
    means were not centred; quoting it makes the screen's zero explicit.
    """
    r = np.asarray(resid, float)
    parity = (pd.to_datetime(pd.Series(day).reset_index(drop=True)).dt.dayofyear % 2).to_numpy()
    corr = np.zeros_like(r)
    for p in (0, 1):
        fit, use = parity != p, parity == p
        corr[use] = float(r[fit].mean())
    before, after = float(np.mean(r ** 2)), float(np.mean((r - corr) ** 2))
    return 1 - after / before, np.sqrt(before) - np.sqrt(after)


def main():
    d = pd.read_parquet(HERE.parent / "work" / "rows_july.parquet")
    nl = d[d["apt"] != "LIRF"].copy()
    tot = nl["se"].sum()
    P = [f"# exp-007 question 2J — where july's other-nine 249.7 s lives\n",
         f"Baseline `{C.BASELINE}`. {len(nl):,} non-LIRF july departures, RMSE "
         f"**{C.rmse(nl['err']):.1f} s**, against the 200-240 s band the board implies.",
         "July carries no unreachable rollover row (`analysis/q1_floor.md`), so every number here",
         "is error a model can attack.\n"]

    P.append("## By airport\n")
    P.append(tbl(nl, "apt", tot, "july, non-LIRF, by airport"))
    P.append("The three airports the desk named are the three largest contributions, and they are "
             "large for different reasons: see the regime split below.\n")

    P.append("## By regime, with the block-on-LOBT class broken out\n")
    r = nl["r"]
    on_lobt = nl["b_lobt"].abs() <= 60
    on_sched = nl["b_sched"].abs() <= 60
    away = r.abs() > 300
    reg = pd.Series("NM bulk", index=nl.index, dtype=object)
    reg[(on_lobt | on_sched) & away & nl["AOBT_3_flt"].notna()] = "block on LOBT/SCHED"
    reg[nl["tail"] & ~nl["no_nm"]] = "NM tail > 3600 s"
    reg[nl["no_nm"]] = "no-NM"
    nl["reg2"] = reg
    P.append(tbl(nl, "reg2", tot, "july, non-LIRF, by regime"))
    P.append(tbl(nl, ["apt", "reg2"], tot, "the same, per airport (top 20 cells)", top=20))

    P.append("## Is a long take-off-minus-AOBT the marker?\n")
    nl["span_bin"] = pd.cut(nl["span"], [-np.inf, 300, 600, 900, 1200, 1800, 2700, 3600, np.inf],
                            labels=["<300", "300-600", "600-900", "900-1200", "1200-1800",
                                    "1800-2700", "2700-3600", ">3600"])
    P.append(tbl(nl[~nl["no_nm"]], "span_bin", tot, "july, NM-joined non-LIRF, by take-off − AOBT",
                 sort="index"))
    P.append("`span` is label-free — both `MVT_TIME_UTC_mvt` and `AOBT_3_flt` survive the ranking "
             "mask — and it is already a feature (`features.py` `d_takeoff_aobt`, and the model's "
             "residual target is built on it). The table therefore reads as *conditional* bias: "
             "what the model still gets wrong once it knows the span.\n")

    P.append("## By airport-day\n")
    g = nl.groupby(["apt", "day"])["se"].sum().sort_values(ascending=False)
    cum = g.cumsum() / g.sum()
    rows = ["| worst | days | cumulative SSE share |", "|---|---|---|"]
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
        k = max(1, int(round(frac * len(g))))
        rows.append(f"| {100*frac:.0f} % | {k} | **{100*cum.iloc[k-1]:.1f} %** |")
    P.append(f"{len(g):,} airport-days.\n")
    P.append("\n".join(rows) + "\n")
    rows = ["**the fifteen worst july airport-days**", "",
            "| airport | day | rows | SSE % | RMSE | mean error | mean y | median span | p90 span |",
            "|---|---|---|---|---|---|---|---|---|"]
    for (a, dy), s in g.head(15).items():
        sub = nl[(nl["apt"] == a) & (nl["day"] == dy)]
        rows.append(f"| {a} | {dy:%Y-%m-%d} | {len(sub):,} | {100*s/tot:.2f} | {C.rmse(sub['err']):,.0f} | "
                    f"{sub['err'].mean():+,.0f} | {sub['y'].mean():,.0f} | {sub['span'].median():,.0f} | "
                    f"{sub['span'].quantile(0.9):,.0f} |")
    P.append("\n".join(rows) + "\n")

    P.append("## What the model under-predicts\n")
    up = nl[nl["err"] > 0]
    dn = nl[nl["err"] <= 0]
    P.append(f"Under-predicted rows: {len(up):,} ({100*len(up)/len(nl):.1f} %), "
             f"{100*up['se'].sum()/tot:.1f} % of the SSE, mean error {up['err'].mean():+.0f} s. "
             f"Over-predicted: {len(dn):,} ({100*len(dn)/len(nl):.1f} %), "
             f"{100*dn['se'].sum()/tot:.1f} %, mean {dn['err'].mean():+.0f} s. "
             f"Whole-slice bias {nl['err'].mean():+.1f} s.\n")
    P.append(tbl(nl[nl["err"] > 0], "reg2", tot, "the under-predicted half, by regime"))

    P.append("## What in the ranking frame marks it\n")
    _, masked, _ = D.fold_frames("july")
    F = sf_build(masked)                        # unfitted families only: counts and timestamps
    dep = masked[masked["PHASE_mvt"] == "DEP"]
    F = F.add_prefix("sf_")
    F.insert(0, D.ID, dep[D.ID].to_numpy())
    m = nl.merge(F, on=D.ID, how="left", validate="1:1")
    m["day_dt"] = m["day"]
    core = m[(~m["no_nm"]) & (~m["tail"])]      # the NM bulk: 81 % of the SSE
    cands = [c for c in F.columns if c != D.ID and pd.api.types.is_numeric_dtype(F[c])]
    cands += ["span", "gap_al", "span_sched"]
    res = []
    resid = core["err"].to_numpy()
    for c in cands:
        x = core[c].to_numpy(dtype="float64")
        frac, secs = cv_explained(resid, x, core["day_dt"].to_numpy())
        if np.isfinite(frac):
            res.append((c, frac, secs, float(pd.Series(resid).corr(pd.Series(x), method="spearman"))))
    R = pd.DataFrame(res, columns=["column", "frac_sse", "rmse_s", "spearman"]).sort_values(
        "frac_sse", ascending=False)
    R.to_csv(HERE.parent / "work" / "july_markers.csv", index=False)
    ig_frac, ig_secs = intercept_gain(resid, core["day_dt"].to_numpy())
    lines = ["Every column below is computable from the ranking-masked july frame alone. The score is",
             "the fraction of the NM-bulk squared error a 20-bin function of that one column removes,",
             "cross-fitted odd day / even day, *on top of the model's own prediction*, with the bin",
             "means CENTRED so the correction is mean-zero. A column the model already uses fully",
             "scores ~0, and so does pure noise -- see the floor below.\n",
             f"NM-bulk rows: {len(core):,}, RMSE {C.rmse(core['err']):.1f} s.\n",
             f"For reference, a cross-fitted CONSTANT correction -- the model's global bias, which "
             f"every column would remove if the bin means were not centred -- is worth "
             f"**{100*ig_frac:.2f} % / {ig_secs:.2f} s** on this slice. That is the amount by which "
             f"an uncentred version of this screen flatters every column, and it is why a Gaussian "
             f"noise column scored +0.21 % before the fix.\n",
             "| column | SSE removed | RMSE removed | Spearman with the residual |",
             "|---|---|---|---|"]
    for _, x in R.head(25).iterrows():
        lines.append(f"| `{x['column']}` | {100*x['frac_sse']:.2f} % | {x['rmse_s']:.2f} s | {x['spearman']:+.3f} |")
    P.append("\n".join(lines) + "\n")
    P.append(f"(Full ranking of {len(R)} columns: `work/july_markers.csv`.)\n")

    OUT.write_text("\n".join(P))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
