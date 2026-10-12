"""Question 2 (kept short): what in the ranking frame marks the worst airport-days.

The desk's re-aim: January 2026 is harder than any 2025 fold can show
(`docs/shift-2026.md`), so a *label-free* marker of a bad airport-day is
insurance for the ranking month even where the fold gap is small.

Every candidate is computed from the ranking-masked month frame alone. The
target is label-defined -- the airport-day's RMSE under the baseline
out-of-fold line -- and is never an input. Candidates are scored by
(i) Spearman with the day's RMSE and (ii) AUC for "in the worst 5 % of days by
squared error", pooled across the three folds and reported per fold.
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
import prc.weather_features as WX  # noqa: E402

OUT = HERE.parent / "analysis" / "q2_daystate.md"


def day_state(masked: pd.DataFrame, hourly) -> pd.DataFrame:
    """Airport-day statistics from a ranking-masked month frame. No label is read.

    Legal inputs (docs/input-rule.md): every movement's take-off/landing time,
    every departure's AOBT_3_flt / LOBT_flt / SCHED_TIME_UTC_mvt, and the
    ARRIVALS' block time and taxi time. Departure BLOCK_TIME_UTC_mvt and
    TAXITIME_SEC_mvt are never touched -- they are NaT/NaN in a masked frame and
    this function does not name them.
    """
    f = masked
    isdep = f["PHASE_mvt"].eq("DEP").to_numpy()
    apt = np.where(isdep, f["ADEP_mvt"].astype(str), f["ADES_mvt"].astype(str))
    day = f["MVT_TIME_UTC_mvt"].dt.floor("D")
    dep = pd.DataFrame({
        "apt": apt[isdep], "day": day[isdep].to_numpy(),
        "span": (f.loc[isdep, "MVT_TIME_UTC_mvt"] - f.loc[isdep, "AOBT_3_flt"]).dt.total_seconds().to_numpy(),
        "gap_al": (f.loc[isdep, "AOBT_3_flt"] - f.loc[isdep, "LOBT_flt"]).dt.total_seconds().to_numpy(),
        "span_sched": (f.loc[isdep, "MVT_TIME_UTC_mvt"] - f.loc[isdep, "SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy(),
        "nm": f.loc[isdep, "FLIGHT_ID_mvt"].notna().to_numpy().astype(float),
        "noac": f.loc[isdep, "AIRCRAFT_TYPE_mvt"].isna().to_numpy().astype(float),
        "rwy": f.loc[isdep, "RUNWAY_mvt"].astype(str).to_numpy(),
        "hour": f.loc[isdep, "MVT_TIME_UTC_mvt"].dt.hour.to_numpy(),
    })
    arr = pd.DataFrame({
        "apt": apt[~isdep], "day": day[~isdep].to_numpy(),
        "taxiin": f.loc[~isdep, "TAXITIME_SEC_mvt"].astype("float64").to_numpy(),
    })
    g = dep.groupby(["apt", "day"])
    S = pd.DataFrame({
        "ds_span_med": g["span"].median(),
        "ds_span_p90": g["span"].quantile(0.90),
        "ds_span_p99": g["span"].quantile(0.99),
        "ds_span_mean": g["span"].mean(),
        "ds_span_std": g["span"].std(),
        "ds_span_iqr": g["span"].quantile(0.75) - g["span"].quantile(0.25),
        "ds_share_span1800": g["span"].apply(lambda s: float((s > 1800).mean())),
        "ds_share_span2700": g["span"].apply(lambda s: float((s > 2700).mean())),
        "ds_gap_med": g["gap_al"].median(),
        "ds_gap_p90": g["gap_al"].quantile(0.90),
        "ds_sched_med": g["span_sched"].median(),
        "ds_sched_p90": g["span_sched"].quantile(0.90),
        "ds_nm_rate": g["nm"].mean(),
        "ds_noac_rate": g["noac"].mean(),
        "ds_n_dep": g.size(),
        "ds_rwy_n": g["rwy"].nunique(),
        "ds_rwy_modal_share": g["rwy"].apply(lambda s: float(s.value_counts(normalize=True).iloc[0])),
        # gpt-5.6-sol round 1, blocking finding 6: the first version encoded only
        # how many runways and how concentrated, never WHICH configuration is
        # active or whether it changed. Entropy of the day's runway-share vector
        # and a change-of-modal-runway flag do encode that.
        "ds_rwy_entropy": g["rwy"].apply(
            lambda s: float(-(s.value_counts(normalize=True)
                              * np.log(s.value_counts(normalize=True))).sum())),
        "ds_rwy_modal": g["rwy"].apply(lambda s: s.value_counts().index[0]),
    })
    A = arr.groupby(["apt", "day"]).agg(ds_taxiin_med=("taxiin", "median"),
                                        ds_taxiin_p90=("taxiin", lambda s: s.quantile(0.90)),
                                        ds_n_arr=("taxiin", "size"))
    S = S.join(A, how="left")
    S["ds_mvt_total"] = S["ds_n_dep"] + S["ds_n_arr"]
    # movement shortfall: the day against the airport's median day in this month
    med = S.groupby(level=0)["ds_mvt_total"].transform("median")
    S["ds_mvt_shortfall"] = S["ds_mvt_total"] / med
    # runway configuration change from the day before
    S = S.sort_index()
    S["ds_rwy_modal_share_prev"] = S.groupby(level=0)["ds_rwy_modal_share"].shift(1)
    prev_modal = S.groupby(level=0)["ds_rwy_modal"].shift(1)
    S["ds_rwy_modal_changed"] = (S["ds_rwy_modal"] != prev_modal).astype(float)
    S.loc[prev_modal.isna(), "ds_rwy_modal_changed"] = np.nan
    S = S.drop(columns=["ds_rwy_modal"])
    S["ds_span_med_prev"] = S.groupby(level=0)["ds_span_med"].shift(1)
    S["ds_span_med_delta"] = S["ds_span_med"] - S["ds_span_med_prev"]

    # weather: the day's worst hour and its mean, over the airport's own rows
    wx = WX.transform(f, hourly)
    wx = wx.assign(apt=apt, day=day.to_numpy())
    wg = wx[wx["day"].notna()].groupby(["apt", "day"])
    W = pd.DataFrame({
        "ds_wx_temp_min": wg["wx_temp_c"].min(),
        "ds_wx_temp_mean": wg["wx_temp_c"].mean(),
        "ds_wx_snow6_max": wg["wx_snow_6h_mm"].max(),
        "ds_wx_precip_sum": wg["wx_precip_mm"].mean(),
        "ds_wx_freezing_share": wg["wx_freezing_precip"].mean(),
        "ds_wx_subzero_share": wg["wx_subzero"].mean(),
        "ds_wx_wind_max": wg["wx_wind_speed_ms"].max(),
        "ds_wx_dewdep_min": wg["wx_dewpoint_depression_c"].min(),
    })
    return S.join(W, how="left")


def auc(score, label):
    s, l = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, l = s[ok], l[ok]
    if l.sum() == 0 or (~l).sum() == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = l.sum(), (~l).sum()
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    hourly = WX.load_hourly()
    per_fold, pooled = {}, []
    for fold in C.FOLDS:
        rows = pd.read_parquet(HERE.parent / "work" / f"rows_{fold}.parquet")
        import q1c_trimmed as q1c
        rows = rows[~q1c.rollover_mask(rows)]           # the unreachable rows cannot be marked
        rows = rows[rows["apt"] != "LIRF"]
        tgt = rows.groupby(["apt", "day"]).agg(sse=("se", "sum"), n=("se", "size"))
        tgt["rmse"] = np.sqrt(tgt["sse"] / tgt["n"])
        tgt["worst5"] = tgt["sse"] >= tgt["sse"].quantile(1 - 0.05)
        _, masked, _ = D.fold_frames(fold)
        S = day_state(masked, hourly)
        S.index = S.index.set_levels([S.index.levels[0], pd.to_datetime(S.index.levels[1], utc=True)])
        J = tgt.join(S, how="left")
        J["fold"] = fold
        per_fold[fold] = J
        pooled.append(J)
    P = pd.concat(pooled)
    P.to_parquet(HERE.parent / "work" / "daystate.parquet")

    key0 = [P.index.get_level_values(0), P["fold"]]
    within = P.groupby(key0)["sse"].rank(ascending=False, method="first") <= 2
    cands = [c for c in P.columns if c.startswith("ds_")]
    res = []
    for c in cands:
        row = {"column": c}
        for fold in C.FOLDS:
            J = per_fold[fold]
            row[f"rho_{fold}"] = float(J["rmse"].corr(J[c], method="spearman"))
            row[f"auc_{fold}"] = auc(J[c], J["worst5"])
        # Pooled, ranked inside (airport, FOLD). Ranking inside the airport alone
        # pools a january day with a july day of the same airport, so a column that
        # merely tracks the season -- july is both later-departing and harder --
        # scores on the seasonal contrast rather than on the day contrast. The
        # first version of this table did that; the fold is in the key from here.
        key = [P.index.get_level_values(0), P["fold"]]
        z = P.groupby(key)[c].rank(pct=True)
        rr = P.groupby(key)["rmse"].rank(pct=True)
        row["rho_within_apt_fold"] = float(rr.corr(z, method="spearman"))
        # AUC for the worst 5 % of days, also computed inside (airport, fold) by
        # ranking the candidate there first
        # Two labels, because they are different questions and the first version of
        # this table described one and computed the other (gpt-5.6-sol round 1,
        # blocking finding 5). `auc_global` marks the fold's own worst 5 % of days
        # by squared error; `auc_within` marks the worst two of each airport's 31
        # days, which is the within-(airport, fold) analogue of the rho column.
        row["auc_global"] = auc(z.to_numpy(), P["worst5"].to_numpy())
        row["auc_within"] = auc(z.to_numpy(), within.to_numpy())
        row["auc_raw"] = auc(P[c], P["worst5"])
        row["n_missing"] = int(P[c].isna().sum())
        res.append(row)
    R = pd.DataFrame(res)
    R["abs_rho"] = R["rho_within_apt_fold"].abs()
    R = R.sort_values("abs_rho", ascending=False)
    R.to_csv(HERE.parent / "work" / "daystate_screen.csv", index=False)

    out = ["# exp-007 question 2 — what marks the worst airport-days, label-free\n",
           f"{len(P):,} airport-days over the three folds (non-LIRF, the two unreachable rows",
           "removed). Every candidate comes from the ranking-masked month frame; the target",
           "(the day's RMSE under the baseline out-of-fold line) is label-defined and is never",
           "an input. `rho within airport-fold` ranks days inside each (airport, fold), so a",
           "column can score neither by telling the airports apart -- the model already knows",
           "the airport -- nor by tracking the season, which pooling within the airport alone",
           "would have let it do. `AUC` is for the worst 5 % of days, on the same",
           "within-(airport, fold) ranks, under TWO labels: the fold's own worst 5 % of",
           "airport-days by squared error, and each airport's own worst two of its 31 days.",
           "The first version of this table computed the first and described the second.",
           "The per-fold rho columns are plain correlations across that fold's days and DO",
           "carry the between-airport contrast.\n",
           "| column | rho within airport-fold | AUC, fold's worst 5 % | AUC, each airport's worst 2 of 31 | rho january | rho december | rho july | missing days |",
           "|---|---|---|---|---|---|---|---|"]
    for _, x in R.iterrows():
        out.append(f"| `{x['column']}` | **{x['rho_within_apt_fold']:+.3f}** | {x['auc_global']:.3f} | "
                   f"{x['auc_within']:.3f} | {x['rho_january']:+.3f} | {x['rho_december']:+.3f} | "
                   f"{x['rho_july']:+.3f} | {int(x['n_missing'])} |")
    OUT.write_text("\n".join(out) + "\n")
    print(OUT.read_text()[:4000])


if __name__ == "__main__":
    main()
