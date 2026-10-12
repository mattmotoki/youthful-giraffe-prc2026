"""Q1 quantitative: which groupings explain BLOCK - AOBT, and stand-level shape."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import common as C
import prc.data as d

TR = [f"2025-{m:02d}" for m in range(1, 7)]
TRD = [f"2025-{m:02d}" for m in range(1, 12)]


def two_means(x, iters=25):
    """1-D 2-means: return (separation in pooled sd, minority share, gap)."""
    x = np.sort(x)
    if len(x) < 40:
        return np.nan, np.nan, np.nan
    c = np.array([np.percentile(x, 25), np.percentile(x, 75)])
    for _ in range(iters):
        cut = c.mean()
        a, b = x[x < cut], x[x >= cut]
        if len(a) < 5 or len(b) < 5:
            return np.nan, np.nan, np.nan
        nc = np.array([a.mean(), b.mean()])
        if np.allclose(nc, c):
            break
        c = nc
    cut = c.mean()
    a, b = x[x < cut], x[x >= cut]
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(x) - 2))
    sep = (c[1] - c[0]) / pooled if pooled > 0 else np.nan
    return sep, min(len(a), len(b)) / len(x), c[1] - c[0]


def main():
    dep = pd.read_parquet(d.ROOT / "artifacts/exp-003-anatomy/dep_all.parquet")
    dep["month"] = dep["month"].astype(str)
    dep["apt_type"] = dep.apt + "|" + dep.AIRCRAFT_TYPE_mvt.astype(str)
    dep["apt_op"] = dep.apt + "|" + dep.AIRCRAFT_OPERATOR_flt.astype(str)
    dep["apt_rwy"] = dep.apt + "|" + dep.RUNWAY_mvt.astype(str)
    dep["apt_stand_rwy"] = dep.apt_stand + "|" + dep.RUNWAY_mvt.astype(str)
    dep["apt_hour"] = dep.apt + "|" + dep.hour.astype(str)
    dep["apt_stand_hour"] = dep.apt_stand + "|" + (dep.hour // 6).astype(str)
    out = []; W = out.append
    W("# Q1 quantitative: what explains `BLOCK - AOBT_3`\n")

    groups = ["apt", "apt_pfx", "apt_stand", "stand_pfx", "apt_rwy", "apt_stand_rwy",
              "AIRCRAFT_TYPE_mvt", "apt_type", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
              "apt_op", "MARKET_SEGMENT_flt", "hour", "apt_hour", "apt_stand_hour",
              "ADES_mvt"]
    # `month` is deliberately absent: the validation month's level is never in
    # the training table, so a month predictor falls back to the global median
    # on every row and reproduces it by construction. Seasonality is measured
    # in q1c_drift.py instead (Sol review, 2026-09-05).
    W("## Out-of-sample: RMSE of `offset = group median fitted on the training months`\n")
    W("Scored on the bulk (NM join, taxi <= 3600 s) of each validation month. "
      "`taxi_hat = (MVT - AOBT_3) - offset_hat`, so this is RMSE on the target. "
      "`explained %` is the share of the *variance* the grouping removes relative "
      "to the global-median predictor, out of sample.\n")
    rows = []
    for fold, trm, vam in [("july", TR, "2025-07"), ("december", TRD, "2025-12"),
                           ("january", [f"2025-{m:02d}" for m in range(2, 13)], "2025-01")]:
        tr = dep[dep.month.isin(trm)]
        va = dep[dep.month == vam]
        mt = tr.has_nm & tr.TAXITIME_SEC_mvt.le(3600)
        mv = va.has_nm & va.TAXITIME_SEC_mvt.le(3600)
        y = va.loc[mv, "TAXITIME_SEC_mvt"].to_numpy(float)
        nm = va.loc[mv, "nm_taxi"].to_numpy(float)
        g0 = tr.loc[mt, "blk_aobt"].median()
        base = float(np.sqrt(np.mean((y - (nm - g0)) ** 2)))
        for g in groups:
            t = tr.loc[mt].groupby(g, observed=True)["blk_aobt"].agg(["median", "size"])
            t = t[t["size"] >= 30]["median"]
            off = va.loc[mv, g].astype(object).map(t).astype(float).fillna(g0).to_numpy(float)
            r = float(np.sqrt(np.mean((y - (nm - off)) ** 2)))
            rows.append({"grouping": g, "fold": fold, "levels": int(len(t)),
                         "rmse": r, "gain_vs_global": base - r,
                         "explained_pct": 100 * (1 - (r / base) ** 2)})
        rows.append({"grouping": "(global median)", "fold": fold, "levels": 1,
                     "rmse": base, "gain_vs_global": 0.0, "explained_pct": 0.0})
    T = pd.DataFrame(rows)
    piv = T.pivot(index="grouping", columns="fold", values=["rmse", "explained_pct"])
    piv.columns = [f"{a} {b}" for a, b in piv.columns]
    piv["levels"] = T.groupby("grouping")["levels"].max()
    piv = piv.sort_values("rmse july")
    W(C.md_table(piv, floatfmt="{:.2f}"))
    W("")

    # in-sample eta^2 for reference
    W("## In-sample eta-squared on 2025 (share of the offset's variance between group means)\n")
    nmm = dep[(~dep.is_ranking) & dep.has_nm & dep.TAXITIME_SEC_mvt.le(3600)]
    x = nmm["blk_aobt"]
    sst = float(((x - x.mean()) ** 2).sum())
    rows = []
    for g in groups:
        gm = nmm.groupby(g, observed=True)["blk_aobt"].agg(["mean", "size"])
        ssb = float((gm["size"] * (gm["mean"] - x.mean()) ** 2).sum())
        rows.append({"grouping": g, "levels": int(len(gm)), "eta2_pct": 100 * ssb / sst})
    W(C.md_table(pd.DataFrame(rows).set_index("grouping").sort_values("eta2_pct", ascending=False),
                 floatfmt="{:.2f}"))
    W("")

    # stand-level shape
    W("## Stand-level shape of the offset (2025, NM join, taxi <= 3600 s, stands with n >= 200)\n")
    g = nmm.groupby("apt_stand", observed=True)["blk_aobt"]
    agg = g.agg(["size", "median", "std", lambda s: s.quantile(.75) - s.quantile(.25)])
    agg.columns = ["n", "median", "std", "iqr"]
    agg = agg[agg["n"] >= 200]
    sep = {k: two_means(v.to_numpy(float)) for k, v in g if len(v) >= 200}
    agg["sep_sd"] = [sep[k][0] for k in agg.index]
    agg["minor_share"] = [sep[k][1] for k in agg.index]
    agg["gap_s"] = [sep[k][2] for k in agg.index]
    agg["apt"] = [k.split("|")[0] for k in agg.index]
    W(f"{len(agg)} stands with at least 200 departures cover "
      f"{100*agg['n'].sum()/len(nmm):.1f} % of the bulk rows.\n")
    W("### distribution of the per-stand dispersion\n")
    q = agg[["std", "iqr", "sep_sd", "gap_s"]].describe(percentiles=[.05, .25, .5, .75, .95]).T
    W(C.md_table(q, floatfmt="{:.1f}"))
    W("")
    W("### per airport: how tight are its stands?\n")
    t = agg.groupby("apt").agg(stands=("n", "size"), rows=("n", "sum"),
                               med_std=("std", "median"), min_std=("std", "min"),
                               n_std_lt_120=("std", lambda s: int((s < 120).sum())),
                               med_sep=("sep_sd", "median"),
                               # 2.67 is what a unimodal Gaussian scores on this
                               # statistic and 1.78 what a t(3) scores, so the
                               # threshold has to be above 2.67 to mean anything.
                               # A large separation with a 1 % minority cluster is
                               # an outlier group, not a second mode, so the count
                               # below also requires a 15 % minority weight
                               # (gpt-5.6-sol round 3).
                               n_sep_gt3=("sep_sd", lambda s: int((s > 3.0).sum())))
    t["n_bimodal"] = agg[(agg.sep_sd > 3.0) & (agg.minor_share >= 0.15)].groupby("apt").size().reindex(t.index).fillna(0)
    W(C.md_table(t.rename_axis("airport"), floatfmt="{:.1f}"))
    W("")
    W("### the 20 tightest stands (near-constant pushback convention)\n")
    W(C.md_table(agg.nsmallest(20, "std")[["n", "median", "std", "iqr", "sep_sd"]].rename_axis("apt|stand"),
                 floatfmt="{:.1f}"))
    W("")
    W("### the 20 most clearly bimodal stands (2-means separation in pooled sd, minority share >= 15 %)\n")
    W("A unimodal Gaussian scores 2.67 on this statistic and a t(3) scores 1.78, "
      "so `n_bimodal` above counts stands over 3.0.\n")
    bim = agg[(agg.minor_share >= 0.15) & agg.sep_sd.notna()].nlargest(20, "sep_sd")
    W(C.md_table(bim[["n", "median", "std", "sep_sd", "minor_share", "gap_s"]].rename_axis("apt|stand"),
                 floatfmt="{:.2f}"))
    W("")

    # month drift with a robust location that is not on a plateau
    W("## Drift across 2025: trimmed mean of the offset (10 % each tail) by airport and month\n")
    tm = lambda s: float(s[(s >= s.quantile(.1)) & (s <= s.quantile(.9))].mean())
    piv = (dep[dep.has_nm & dep.TAXITIME_SEC_mvt.le(3600) | dep.is_ranking & dep.has_nm]
           .pivot_table(index="apt", columns="month", values="blk_aobt", aggfunc=tm, observed=True))
    W(C.md_table(piv.rename_axis("airport"), floatfmt="{:.0f}"))
    W("\n(The *median* of this offset is a bad drift statistic: the density is flat to "
      "about 0.16 % per second near the centre, so the median hops by a whole minute "
      "between adjacent months without the distribution moving. The trimmed mean above "
      "is what the drift tables in `q1_offset_structure.md` should be read against.)\n")
    C.write("q1b_decomp.md", "\n".join(out))


if __name__ == "__main__":
    main()
