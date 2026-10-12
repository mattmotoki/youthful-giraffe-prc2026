"""exp-043-pilot steps 2 and 3: match the observed segments and report the numbers.

Identity is `REPORT.md` §3: the modal callsign of the ground segment that ends in a
take-off, joined to the week's EHAM departures, with the observed wheels-off inside
+/- 15 min of `MVT_TIME_UTC_mvt`.  `CALLSIGN_flt` carries the join on the NM rows;
the airline-prefix map (learned off July) carries the no-NM rows.

Two readings of my own, both recorded because the assignment leaves them open:

*   The join is made on the callsign and the +/- 15 min window, NOT on
    `(callsign, date)`.  The window is the discriminating key -- date equality only
    adds a failure at midnight, where an aircraft that pushes back at 23:55 has its
    take-off in the next UTC day.  `date_agrees` counts how many matches a strict
    date join would have lost, so the assignment's version is recoverable.
*   Ambiguity is resolved by nearest wheels-off and counted, never dropped
    silently.  Both directions are counted: two departures for one segment, and
    two segments for one departure (an aircraft that returns to the stand).

  .venv/bin/python experiments/exp-043-pilot/code/match.py 2025-07-07 2025-07-08
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
import census as C          # noqa: E402
import segments as S        # noqa: E402

WINDOW = 900.0              # s, the +/- 15 min identity window
RNG = np.random.default_rng(20260908)


def rmse_centered(e: pd.Series) -> tuple[float, float, float, float]:
    """RMSE about the sample median, and the 1st/99th percentiles of the error."""
    e = e.dropna()
    if len(e) < 2:
        return (np.nan,) * 4
    m = float(e.median())
    return (float(np.sqrt(((e - m) ** 2).mean())), m,
            float(e.quantile(.01)), float(e.quantile(.99)))


def _resamples(ds: list) -> list[tuple]:
    """Every day-clustered resample.  With five or six days the whole scheme is
    small enough to enumerate exactly (5^5 = 3,125), which is honest about what a
    five-cluster interval is; beyond that fall back to sampling."""
    k = len(ds)
    if k ** k <= 200_000:
        return list(itertools.product(range(k), repeat=k))
    return [tuple(RNG.integers(0, k, k)) for _ in range(20_000)]


def day_interval(frame: pd.DataFrame, day: pd.Series, stat) -> dict:
    """Day-clustered interval on any statistic of a row set.

    Reported as a 2.5-97.5 spread of the whole resampling scheme AND as the
    leave-one-day-out range, because with five clusters the first is not a
    textbook 95 % confidence interval and should not be read as one.
    """
    ds = sorted(day.dropna().unique())
    if len(ds) < 2:
        return {"point": stat(frame), "days": len(ds)}
    idx = {d: np.flatnonzero((day == d).to_numpy()) for d in ds}
    vals = []
    for pick in _resamples(ds):
        take = np.concatenate([idx[ds[i]] for i in pick])
        v = stat(frame.iloc[take])
        if v == v:
            vals.append(v)
    loo = [stat(frame[day != d]) for d in ds]
    return {"point": stat(frame), "days": len(ds),
            "resample_lo": float(np.percentile(vals, 2.5)),
            "resample_hi": float(np.percentile(vals, 97.5)),
            "leave_one_day_out_lo": float(np.nanmin(loo)),
            "leave_one_day_out_hi": float(np.nanmax(loo)),
            "exact_scheme": len(ds) ** len(ds) <= 200_000}


def day_boot(e: pd.Series, day: pd.Series, n: int = 2000) -> tuple[float, float]:
    """The centred-RMSE spread of the day-clustered scheme (kept for the table)."""
    e, day = e.dropna(), day.loc[e.dropna().index]
    f = pd.DataFrame({"e": e.to_numpy("float64")})
    r = day_interval(f, day.reset_index(drop=True),
                     lambda d: float(np.sqrt(((d.e - d.e.median()) ** 2).mean())))
    return (r.get("resample_lo", np.nan), r.get("resample_hi", np.nan))


def build(days: list[str]) -> pd.DataFrame:
    pts, _ = C.load_slices(days)
    seg = S.segments(pts)
    seg["callsign_modal"] = seg.callsign_modal.str.strip()
    coh = C.with_callsign(pd.read_parquet(EXP / "work/cohort.parquet"))
    coh = coh[coh.day.dt.strftime("%Y-%m-%d").isin(days)].copy()
    # unit-agnostic: the files are timestamp[us, tz=UTC], so an int64 cast would
    # give microseconds, and NaT would become a large negative number
    for col, name in (("MVT_TIME_UTC_mvt", "mvt_s"), ("BLOCK_TIME_UTC_mvt", "blk_s"),
                      ("AOBT_3_flt", "aobt_s")):
        coh[name] = (coh[col] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()

    j = coh.merge(seg, left_on="callsign", right_on="callsign_modal", how="left")
    j["dt_takeoff"] = j.obs_wheels_off - j.mvt_s
    j = j[j.dt_takeoff.abs() <= WINDOW]
    # nearest wheels-off wins in both directions, and both are counted
    j["absdt"] = j.dt_takeoff.abs()
    n_multi_seg = int((j.groupby("MVT_ID_mvt").size() > 1).sum())
    # stable, so a tie in |wheels-off - MVT_TIME| resolves the same way everywhere
    j = j.sort_values("absdt", kind="stable").drop_duplicates("MVT_ID_mvt")
    n_multi_dep = int((j.groupby(["icao", "obs_wheels_off"]).size() > 1).sum())
    j = j.sort_values("absdt", kind="stable").drop_duplicates(["icao", "obs_wheels_off"])
    j["date_agrees"] = (pd.to_datetime(j.obs_wheels_off, unit="s", utc=True)
                        .dt.floor("D") == j.day)
    m = coh.merge(j.drop(columns=[c for c in coh.columns if c != "MVT_ID_mvt"]),
                  on="MVT_ID_mvt", how="left")
    m.attrs["n_multi_seg"] = n_multi_seg
    m.attrs["n_multi_dep"] = n_multi_dep
    m.attrs["n_segments"] = int(len(seg))
    return m


def table(m: pd.DataFrame, label: str) -> dict:
    """The paired accuracy table on one row set."""
    out = {"rows": int(len(m))}
    for name, off in (("obs_strict", "obs_offblock_strict"),
                      ("obs_loose", "obs_offblock_loose"),
                      ("obs_disp250", "obs_offblock_disp")):
        for scope, sel in (("all_matched", m[off].notna()),
                           ("paired_with_aobt", m[off].notna() & m.aobt_s.notna())):
            sub = m[sel]
            if len(sub) < 2:
                continue
            e = sub.blk_s - sub[off]
            r, med, p1, p99 = rmse_centered(e)
            lo, hi = day_boot(e, sub.day.dt.strftime("%Y-%m-%d"))
            rec = {"rows": int(len(sub)), "median_offset_s": med, "rmse_s": r,
                   "p01_s": p1, "p99_s": p99, "day_ci95": [lo, hi]}
            if scope == "paired_with_aobt":
                ea = sub.blk_s - sub.aobt_s
                ra, meda, pa1, pa99 = rmse_centered(ea)
                la, ha = day_boot(ea, sub.day.dt.strftime("%Y-%m-%d"))
                rec["aobt_on_same_rows"] = {"median_offset_s": meda, "rmse_s": ra,
                                            "p01_s": pa1, "p99_s": pa99,
                                            "day_ci95": [la, ha]}
                d = (e - e.median()) - (ea - ea.median())
                rec["paired_rmse_difference_s"] = float(r - ra)
                pf = pd.DataFrame({"e": e.to_numpy("float64"), "a": ea.to_numpy("float64")})
                rec["paired_rmse_difference_day_interval"] = day_interval(
                    pf, sub.day.dt.strftime("%Y-%m-%d").reset_index(drop=True),
                    lambda d: float(np.sqrt(((d.e - d.e.median()) ** 2).mean())
                                    - np.sqrt(((d.a - d.a.median()) ** 2).mean())))
                rec["paired_mse_difference_s2"] = float(
                    ((e - e.median()) ** 2).mean() - ((ea - ea.median()) ** 2).mean())
                rec["corr_obs_aobt_errors"] = float(np.corrcoef(e, ea)[0, 1])
                rec["rmse_of_difference_s"] = float(np.sqrt((d ** 2).mean()))
            out[f"{name}.{scope}"] = rec
    return {label: out}


def main(days: list[str], tag: str = "") -> None:
    m = build(days)
    matched = m.obs_wheels_off.notna()
    dk = m.day.dt.strftime("%Y-%m-%d")
    reach_ci = day_interval(pd.DataFrame({"h": matched.to_numpy()}), dk.reset_index(drop=True),
                            lambda d: float(d.h.mean()))
    res = {
        "days": days, "window_s": WINDOW,
        "segments_observed": m.attrs["n_segments"],
        "reach": {
            "departures": int(len(m)),
            "matched": int(matched.sum()),
            "matched_share": round(float(matched.mean()), 4),
            "matched_share_day_interval": reach_ci,
            "matched_nm": round(float(matched[m.has_nm].mean()), 4),
            "matched_no_nm": (round(float(matched[~m.has_nm].mean()), 4)
                              if (~m.has_nm).any() else None),
            "no_nm_rows": int((~m.has_nm).sum()),
            "with_obs_offblock_strict": int(m.obs_offblock_strict.notna().sum()),
            "with_obs_offblock_loose": int(m.obs_offblock_loose.notna().sum()),
            "with_obs_offblock_disp": int(m.obs_offblock_disp.notna().sum()),
            "strict_share_of_all": round(float(m.obs_offblock_strict.notna().mean()), 4),
            "ambiguous_two_segments_one_departure": m.attrs["n_multi_seg"],
            "ambiguous_two_departures_one_segment": m.attrs["n_multi_dep"],
            "date_agrees_on_matched": int(m.date_agrees.sum(skipna=True)),
            "callsign_source_matched": m.loc[matched, "callsign_source"]
                                        .value_counts().to_dict(),
        },
        "clock_check_wheels_off_minus_mvt_s": {
            "rows": int(matched.sum()),
            "median": float(m.dt_takeoff.median()),
            "mean_abs": float(m.dt_takeoff.abs().mean()),
            # the denominator is the MATCHED rows: an unmatched row has no clock
            "share_within_60s": float((m.loc[matched, "dt_takeoff"].abs() <= 60).mean()),
            "within_60s_rows": int((m.loc[matched, "dt_takeoff"].abs() <= 60).sum()),
            "p01": float(m.dt_takeoff.quantile(.01)),
            "p99": float(m.dt_takeoff.quantile(.99)),
        },
        "accuracy": {},
        "segment_quality": {
            "seg_dt_median_s": float(m.seg_dt_median_s.median()),
            "seg_points_median": float(m.seg_points.median()),
            "stand_stop_observed_share": float(m.stand_stop_s.notna().mean()),
        },
    }
    res["accuracy"] |= table(m, "all_rows")
    res["accuracy"] |= table(m[m.has_nm], "nm_rows")
    if (~m.has_nm).any():
        res["accuracy"] |= table(m[~m.has_nm], "no_nm_rows")

    (EXP / "work").mkdir(exist_ok=True)
    suf = f"-{tag}" if tag else ""
    m.to_parquet(EXP / f"work/matched{suf}.parquet", index=False)
    (EXP / f"analysis/match{suf}.json").write_text(json.dumps(res, indent=2, sort_keys=True,
                                                              default=float))
    print(json.dumps(res, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    tag = ""
    if a and a[0].startswith("--tag="):
        tag, a = a[0][6:], a[1:]
    main(a or ["2025-07-07", "2025-07-08"], tag)
