"""Why LFPG's 2026-01 observation is twice as wide as any month the fit has seen.

The desk, 2026-09-08: "LFPG's hold -- find the cause of the 2026-01 doubling label-free,
to clear it or keep it with a reason."

`analysis/lfpgwhy.json` established the SIZE of the problem and ruled out two causes: the
excess is not the day's receipt (the spread is lowest where the match share is lowest) and
not the fragment length (the segments are 7-point on every month).  It left the cause open.
This asks four label-free questions the first pass did not:

1. **Does the fit ever see LFPG in January?**  `lfpgwhy.json` records `january: n = 0` for
   LFPG in one line and does not draw the consequence.  If LFPG has no with-clock row in
   the 2025-01 fold, then no rotation has ever fitted an LFPG January row, and the served
   update at LFPG in 2026-01 is an extrapolation from July and December LFPG alone.
2. **Is the width one month, or a handful of DAYS?**  A snow or de-icing event is a few
   days; a changed airport is a month.  The two want different rules, and only one of them
   makes the hold the right instrument.
3. **What separates a wide row from a normal one**, in the observation's own columns --
   the segment, the stand, the reception, the match quality -- and at what hour?
4. **What would a narrower rule cost?**  The airport hold returns 1,612 ranking rows to the
   base; a day-level or row-level rule would return fewer.  Priced in rows and in update,
   never on a board reading (score-feedback rule).

Nothing here is fitted and nothing is served.  The wide/normal split is defined from the
FOLD months' own quantiles, so no ranking-month statistic and no label enters the rule.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/lfpgjan.py
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

AP = "LFPG"
OUT = EXP / "analysis/lfpg-january.json"
#: the observation's own columns, all label-free, that could separate a wide row
SEP = ["obs_taxi", "obs_taxi_motion", "obs_seg_taxi", "seg_points", "n_ground_points",
       "seg_span_s", "seg_gap_max_s", "seg_dt_median_s", "stand_dist_km",
       "seen_before_move_s", "stand_stop_s", "time_to_250m", "dt_takeoff",
       "callsign_n", "obs_n_candidates", "arr_present", "arr_taxi_in_s",
       "arr_stand_occupancy_s", "arr_gap_reception_s", "clock_ok", "matched_on_modal",
       "obs_ambiguous", "turnaround_anchored"]


def spread(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype="float64")
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    med = float(np.median(v))
    return {"n": int(len(v)), "median": med,
            "rms_about_the_median": float(np.sqrt(np.mean((v - med) ** 2))),
            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
            "p95": float(np.percentile(v, 95)),
            "deciles": [round(float(x), 1) for x in np.percentile(v, np.arange(10, 100, 10))]}


def summarise(d: pd.DataFrame) -> dict:
    return {c: spread(d[c].to_numpy("float64")) for c in ("obs_taxi", "obs_taxi_motion",
                                                          "obs_seg_taxi")
            if c in d}


def main() -> None:
    pan = A.panel_with_block(BA.load())
    dep, _ = S.ranking_block()
    out = {"what": __doc__.strip().splitlines()[0], "airport": AP,
           "rule_sha256": A.serving_rule()[0], "matcher_sha256": A.MATCHER_SHA}

    # --- 1. the coverage census, every airport, every month ------------------
    cov = {}
    for f in A.FOLDS:
        p = pan[f]
        cov[A.FOLD_MONTH[f]] = {
            str(a): {"rows": int(len(g)), "with_clock": int(g.usable.sum()),
                     "clock_share": round(float(g.usable.mean()), 4)}
            for a, g in p.groupby("airport", observed=True)}
    for m, g0 in dep.groupby("rank_month"):
        cov[str(m)] = {
            str(a): {"rows": int(len(g)), "with_clock": int(g.usable.sum()),
                     "clock_share": round(float(g.usable.mean()), 4)}
            for a, g in g0.groupby("airport", observed=True)}
    out["coverage_by_airport_month"] = cov
    dark = {m: sorted(a for a, r in c.items() if r["with_clock"] == 0)
            for m, c in cov.items()}
    out["airports_with_no_with_clock_row"] = dark
    out["the_fit_never_sees_this_airport_in_january"] = bool(
        AP in dark.get("2025-01", []))

    # --- 2. the width, and whether it is a few days --------------------------
    fold_rows = {A.FOLD_MONTH[f]: pan[f][(pan[f].airport.astype(str) == AP)
                                         & pan[f].usable.to_numpy(bool)].copy()
                 for f in A.FOLDS}
    rank_rows = {str(m): g[(g.airport.astype(str) == AP) & g.usable.to_numpy(bool)].copy()
                 for m, g in dep.groupby("rank_month")}
    allm = {**fold_rows, **rank_rows}
    out["width_by_month"] = {m: summarise(d) for m, d in allm.items()}

    days = {}
    for m, d in allm.items():
        if not len(d):
            days[m] = {"days": 0}
            continue
        day = (d.day.astype(str) if "day" in d else
               d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")).to_numpy()
        v = d.obs_taxi.to_numpy("float64")
        ok = np.isfinite(v)
        v, day = v[ok], day[ok]
        med = float(np.median(v))
        sq = (v - med) ** 2
        per = pd.DataFrame({"day": day, "sq": sq}).groupby("day").agg(
            rows=("sq", "size"), ss=("sq", "sum"))
        per = per.sort_values("ss", ascending=False)
        tot = float(per.ss.sum())
        days[m] = {
            "days": int(len(per)), "rows": int(len(v)),
            "share_of_the_squared_spread_in_the_worst_1_3_5_days": [
                round(float(per.ss.head(k).sum() / tot), 4) if tot else None
                for k in (1, 3, 5)],
            "worst_days": [{"day": str(i), "rows": int(r.rows),
                            "rms_about_the_month_median": round(float(np.sqrt(r.ss / r.rows)), 1)}
                           for i, r in per.head(6).iterrows()],
            "median_day_rms": round(float(np.median(np.sqrt(per.ss / per.rows))), 1),
        }
    out["is_it_a_few_days"] = days

    # --- 3. what separates a wide row, at the FOLD months' own quantile ------
    ref = pd.concat([d for m, d in fold_rows.items() if len(d)], ignore_index=True)
    cut = float(np.nanpercentile(ref.obs_taxi.to_numpy("float64"), 95))
    out["wide_rule"] = {
        "expression": f"obs_taxi > {cut:.1f} s",
        "learned_on": ("the p95 of obs_taxi over this airport's with-clock rows in the "
                       "FITTING months only; no ranking-month statistic and no label"),
        "fold_rows_behind_it": int(len(ref)),
    }
    sep = {}
    for m, d in allm.items():
        if not len(d):
            continue
        w = d.obs_taxi.to_numpy("float64") > cut
        rec = {"rows": int(len(d)), "wide_rows": int(w.sum()),
               "wide_share": round(float(w.mean()), 4)}
        for c in SEP:
            if c not in d:
                continue
            v = pd.to_numeric(d[c], errors="coerce").to_numpy("float64")
            if not np.isfinite(v).any():
                continue
            rec[c] = {"wide_median": (round(float(np.nanmedian(v[w])), 2)
                                      if w.any() and np.isfinite(v[w]).any() else None),
                      "normal_median": (round(float(np.nanmedian(v[~w])), 2)
                                        if (~w).any() and np.isfinite(v[~w]).any() else None)}
        if "MVT_TIME_UTC_mvt" in d:
            hr = d.MVT_TIME_UTC_mvt.dt.hour.to_numpy()
            rec["hour_of_day"] = {
                "wide_median": int(np.median(hr[w])) if w.any() else None,
                "normal_median": int(np.median(hr[~w])) if (~w).any() else None,
                "wide_share_by_hour": {int(h): round(float(w[hr == h].mean()), 3)
                                       for h in sorted(set(hr.tolist()))
                                       if (hr == h).sum() >= 20},
            }
        sep[m] = rec
    out["wide_against_normal"] = sep

    # --- 4. what a narrower rule would cost ---------------------------------
    lf = dep[(dep.airport.astype(str) == AP)].copy()
    served = lf.usable.to_numpy(bool) & ~lf.noac.to_numpy(bool)
    wide = (lf.obs_taxi.to_numpy("float64") > cut) & served
    by_month = {}
    for m, g in lf.groupby("rank_month"):
        sm = g.usable.to_numpy(bool) & ~g.noac.to_numpy(bool)
        wm = (g.obs_taxi.to_numpy("float64") > cut) & sm
        by_month[str(m)] = {"served_rows_if_the_hold_were_lifted": int(sm.sum()),
                            "rows_the_wide_rule_would_still_hold": int(wm.sum()),
                            "rows_the_wide_rule_would_release": int((sm & ~wm).sum())}
    out["what_a_narrower_rule_would_cost"] = {
        "airport_hold_holds": int(served.sum()),
        "wide_rule_holds": int(wide.sum()),
        "wide_rule_releases": int((served & ~wide).sum()),
        "by_month": by_month,
        "note": ("rows only.  What the released rows would be WORTH cannot be read on the "
                 "ranking side without a label; the fold side prices the hold at +8.8729 "
                 "-> +8.8576 s (analysis/serve-v14.json), which is what LFPG's fold rows "
                 "are worth, not what its 2026-01 rows would be"),
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print(f"LFPG january fold rows: "
          f"{cov.get('2025-01', {}).get(AP, {}).get('with_clock')}  "
          f"(the fit never sees it: {out['the_fit_never_sees_this_airport_in_january']})")
    for m in sorted(allm):
        w = out["width_by_month"][m].get("obs_taxi", {})
        d = days[m]
        print(f"  {m}  n {w.get('n', 0):5d}  obs_taxi rms {w.get('rms_about_the_median', 0):7.1f} "
              f"iqr {w.get('iqr', 0):7.1f}  days {d.get('days', 0):3d}  "
              f"worst 1/3/5 {d.get('share_of_the_squared_spread_in_the_worst_1_3_5_days')}")
    print(json.dumps(out["what_a_narrower_rule_would_cost"]["by_month"], indent=1))
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
