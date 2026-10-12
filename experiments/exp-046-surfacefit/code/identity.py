"""exp-046-surfacefit: WHY the matcher loses 25-41 points at EDDF, EGLL and LIRF.

The complete 2026-01 (`analysis/reachwhy.json`) says those three airports are HEARD on
the ground at 91.7 / 93.1 / 91.6 % and strictly MATCHED at 50.5 / 68.4 / 57.1, so what
loses their rows is identification and not reception (RESULTS.md §4).  The desk's line of
2026-09-08 15:05 bounds this diagnosis: label-free, on 2026-01, about two hours, along
the axes it named -- **the callsign form** (prefix, digits, leading zeros), **the ±15 min
take-off window**, **the date boundary**, **the `ADEP` field** -- with the pilot's
**geometric key** (take-off instant, one-to-one, placebo-tested; `exp-043` §4d) as the
fallback identity.

Nothing here is a repair.  It measures, per axis, how many of the rows the strict matcher
leaves behind that axis would recover, so the repair that follows is chosen by a number.

The one structural suspect the schema itself carries: a ground segment records
`callsign_modal` AND `callsigns_all`, and **23 % of segments carry more than one form**
(an aircraft that lands as `DLH1319` and departs as `DLH4YF` is one ground run with two
names, and the modal can be the wrong one).  `code/block.py` joins on the modal alone.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/identity.py [month]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
import arms as A                                             # noqa: E402
import block as B                                            # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

WINDOW = B.WINDOW                       # 900 s, the strict take-off window
WIDE = B.WIDE_WINDOW                    # 3600 s, the desk's wide window
DAY_S = 6 * 3600.0                      # "the same day" for the heard/not-heard test
PLACEBO_SHIFT_S = 7200.0
_ZEROS = re.compile(r"^([A-Z]+)0+(\d.*)$")
_KEEP = re.compile(r"[^A-Z0-9]")


def norm(s: pd.Series) -> pd.Series:
    """Callsign, comparable across the two sources.

    Upper case, punctuation and spaces removed, and the leading zeros of the numeric run
    stripped: the file writes `BAW07A` where the transponder sends `BAW7A`.  The airline
    prefix itself is never rewritten here -- a prefix repair would be a different claim
    and it is measured separately, through the reconstruction `block.cohort()` already
    does.
    """
    u = s.fillna("").astype(str).str.upper().str.replace(_KEEP, "", regex=True)
    return u.str.replace(_ZEROS, r"\1\2", regex=True)


def forms(seg: pd.DataFrame) -> pd.DataFrame:
    """One row per (segment, callsign form): `callsigns_all` is `|`-delimited."""
    s = seg[["seg_key", "airport", "summary_day", "kind", "obs_wheels_off",
             "obs_offblock", "callsign_modal", "callsigns_all"]].copy()
    s["form"] = s.callsigns_all.fillna(s.callsign_modal).astype(str).str.split("|")
    f = s.explode("form")
    f["form"] = f.form.str.strip()
    f = f[f.form.ne("") & ~f.form.str.fullmatch(r"[@?_]+")]
    f["form_norm"] = norm(f.form)
    f["is_modal"] = f.form.eq(f.callsign_modal.str.strip())
    return f


def one_to_one(cand: pd.DataFrame) -> int:
    """Rows recovered when each segment may serve at most one row, nearest take-off first."""
    if not len(cand):
        return 0
    c = cand.sort_values("absdt").drop_duplicates("MVT_ID_mvt").drop_duplicates("seg_key")
    return int(len(c))


def ladder(rows: pd.DataFrame, f: pd.DataFrame, free_keys: set[int],
           points: pd.DataFrame) -> dict:
    """The FIRST reason that applies to each unmatched row, so the counts partition it.

    Ordered from the source outward: no name on the file side, then no ground run under
    that name, then a run that is not a departure segment, then a departure segment that
    does not carry the name as its MODAL (which is the join `code/block.py` does), then
    the ±15 min window, and what is left is contention -- a segment another row took.
    """
    out = {"unmatched_rows": int(len(rows))}
    if not len(rows):
        return out
    fx = f[["seg_key", "airport", "summary_day", "form", "form_norm", "is_modal",
            "kind", "obs_wheels_off"]]
    dep_forms = fx[fx.kind.eq("departure") & fx.obs_wheels_off.notna()]

    def hit(src: pd.DataFrame, win: float | None = None,
            same_day: bool = True) -> pd.DataFrame:
        """Rows joined to segments by name.

        **Scoped to the row's own day and its two neighbours.**  A callsign recurs every
        day, and the month's summaries are all in `f`, so an unscoped join answers "this
        name flies from here" and not "this flight was heard": without the scope a row
        whose own day carries no run at all is credited with the same name on another
        day and lands under the window instead of under reception.
        """
        j = rows.merge(src, left_on=["airport", "callsign_norm"],
                       right_on=["airport", "form_norm"], how="inner",
                       suffixes=("", "_seg"))
        if same_day:
            off = (pd.to_datetime(j.summary_day) - pd.to_datetime(j.takeoff_day)).dt.days
            j = j[off.abs() <= 1]
        j["dt"] = j.obs_wheels_off - j.mvt_s
        j["absdt"] = j.dt.abs()
        return j if win is None else j[j.absdt <= win]

    any_seg = hit(fx)                       # any segment of any kind, any time
    any_dep = hit(dep_forms)                # a departure segment with a take-off
    dep_win = any_dep[any_dep.absdt <= WINDOW]
    modal_win = dep_win[dep_win.is_modal]

    no_name = rows.callsign.isna() | rows.callsign.eq("")
    seen_any = rows.MVT_ID_mvt.isin(any_seg.MVT_ID_mvt)
    seen_dep = rows.MVT_ID_mvt.isin(any_dep.MVT_ID_mvt)
    in_win = rows.MVT_ID_mvt.isin(dep_win.MVT_ID_mvt)
    as_modal = rows.MVT_ID_mvt.isin(modal_win.MVT_ID_mvt)

    reason = np.where(
        no_name, "1_no_callsign_on_the_file_side",
        np.where(~seen_any, "2_no_ground_run_under_that_name",
                 np.where(~seen_dep, "3_run_is_not_a_departure_segment",
                          np.where(~in_win, "4_departure_segment_outside_the_window",
                                   np.where(~as_modal, "5_name_is_not_the_segment_modal",
                                            "6_contention_or_ambiguity")))))
    out["why_not_matched"] = pd.Series(reason).value_counts().to_dict()

    # what each repair would recover, one-to-one, on top of the strict rule
    out["repair_join_on_every_broadcast_form"] = one_to_one(dep_win[~dep_win.is_modal])
    out["repair_widen_window_to_3600s"] = one_to_one(
        any_dep[(any_dep.absdt > WINDOW) & (any_dep.absdt <= WIDE)
                & ~any_dep.MVT_ID_mvt.isin(dep_win.MVT_ID_mvt)])
    out["recovered_from_a_neighbour_day_file"] = int(
        dep_win[dep_win.summary_day.ne(dep_win.takeoff_day)].MVT_ID_mvt.nunique())

    # the ADEP axis: the same name on a departure segment at ANOTHER airport
    oth = rows.merge(dep_forms, left_on="callsign_norm", right_on="form_norm",
                     how="inner", suffixes=("", "_seg"))
    oth = oth[oth.airport.ne(oth.airport_seg)]
    oth["absdt"] = (oth.obs_wheels_off - oth.mvt_s).abs()
    out["found_at_another_airport_in_window"] = int(
        oth[oth.absdt <= WINDOW].MVT_ID_mvt.nunique())
    # the same name on a departure segment, but on ANOTHER day: the callsign flies this
    # route daily and this row's own day carries no run for it
    cross = hit(dep_forms, same_day=False)
    cross = cross[~cross.MVT_ID_mvt.isin(any_dep.MVT_ID_mvt)]
    out["name_only_on_another_day"] = int(cross.MVT_ID_mvt.nunique())

    # how thin the ground presence is, for the rows with no run under their name
    if len(points):
        pn = points.groupby("cn", observed=True).n_points.max()
        for lab, sel in (("no_ground_run_under_that_name", ~seen_any),
                         ("matched_elsewhere_in_this_month", seen_dep)):
            v = pn.reindex(rows.callsign_norm[sel].to_numpy()).dropna()
            out[f"ground_points_median_{lab}"] = float(v.median()) if len(v) else None
            out[f"has_ground_points_share_{lab}"] = (
                float(rows.callsign_norm[sel].isin(pn.index).mean()) if sel.any() else None)

    # --- the geometric fallback (exp-043 §4d): take-off instant, one-to-one ---
    free = dep_forms[dep_forms.seg_key.isin(free_keys)].drop_duplicates("seg_key")
    g = rows.merge(free[["seg_key", "airport", "obs_wheels_off"]], on="airport",
                   how="inner")
    g["dt"] = g.obs_wheels_off - g.mvt_s
    g["absdt"] = g.dt.abs()
    gw = g[g.absdt <= WINDOW]
    out["geometric_one_to_one"] = one_to_one(gw)
    gp = g.copy()
    gp["absdt"] = (gp.dt - PLACEBO_SHIFT_S).abs()
    out["geometric_one_to_one_placebo_plus_2h"] = one_to_one(gp[gp.absdt <= WINDOW])
    out["free_departure_segments"] = int(len(free))
    return out


def main(month: str = "2026-01") -> None:
    src = (D.load_ranking() if month.startswith("2026")
           else pd.read_parquet(D.month_file(month)))
    dep = src[src.PHASE_mvt == "DEP"].copy()
    dep = dep[dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").eq(month)]
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in A.prefix_months_for(month)], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")

    rule, rule_report = B.one_rule([(sorted(set(dep.ADEP_mvt.dropna()) & set(B.AIRPORTS)),
                                     A.month_days(month))])
    blk = B.build(dep, tr, require_rule=rule)
    coh = B.cohort(dep, B.prefix_map(tr))
    coh = coh.merge(blk[["MVT_ID_mvt", "obs_matched", "obs_has_clock", "reach_class"]],
                    on="MVT_ID_mvt", how="left", validate="one_to_one")
    coh["obs_matched"] = coh.obs_matched.astype("boolean").fillna(False).astype(bool)
    coh["callsign_norm"] = norm(coh.callsign)
    coh["airport"] = coh.ADEP_mvt.astype(str)

    # the held airport-days are held by instruction, not lost by the matcher: their rows
    # leave this population and are reported on their own
    held = pd.Series(list(zip(coh.airport, coh.takeoff_day))).isin(
        set(B.HELD_AIRPORT_DAYS)).to_numpy()
    coh["held_day"] = held
    seg = B.load_summaries(sorted(coh.airport.unique()),
                           sorted(coh.takeoff_day.unique()), require_rule=rule)
    seg = seg.reset_index(drop=True)
    seg["seg_key"] = np.arange(len(seg))
    f = forms(seg)
    # the segments the strict matcher already used, by (airport, modal, take-off): the
    # block does not hand back its segment key, so the used pool is reconstructed from
    # the matched rows' own observation
    used = blk[blk.obs_matched.fillna(False).astype(bool)]
    used = used.merge(coh[["MVT_ID_mvt", "airport", "callsign"]], on="MVT_ID_mvt")
    used_keys = set(f.merge(used, left_on=["airport", "form"],
                            right_on=["airport", "callsign"], how="inner")
                    .assign(dt=lambda x: (x.obs_wheels_off_x - x.obs_wheels_off_y).abs())
                    .query("dt <= 1.0").seg_key)

    # the harvest's own ground-point census, to say whether a name with no run was
    # heard at all: artifacts/exp-045/callsigns/callsigns_<day>.parquet
    pts = []
    for d in sorted(coh.takeoff_day.unique()):
        f_pts = ROOT / f"artifacts/exp-045/callsigns/callsigns_{d}.parquet"
        if f_pts.exists():
            pts.append(pd.read_parquet(f_pts).assign(day=d))
    points = pd.concat(pts, ignore_index=True) if pts else pd.DataFrame(
        columns=["airport", "callsign", "n_points", "day"])
    if len(points):
        points["cn"] = norm(points.callsign)

    rep = {"month": month, "rule": rule, "rule_cover": rule_report,
           "rows_on_a_held_day": int(coh.held_day.sum()),
           "window_s": WINDOW, "wide_window_s": WIDE,
           "held_airport_days": [f"{a} {d}" for (a, d) in B.HELD_AIRPORT_DAYS],
           "segments": int(len(seg)),
           "segments_with_more_than_one_callsign_form":
               float((seg.callsign_n > 1).mean()),
           "by_airport": {}}
    for a, g_all in coh.groupby("airport"):
        g = g_all[~g_all.held_day]
        rows = g[~g.obs_matched][["MVT_ID_mvt", "airport", "callsign", "callsign_norm",
                                  "mvt_s", "takeoff_day"]]
        fa = f[f.airport.eq(a)] if a in set(f.airport) else f.iloc[:0]
        pa = points[points.airport.eq(a)] if len(points) else points
        rec = {"departures": int(len(g)), "rows_on_a_held_day": int(g_all.held_day.sum()),
               "matched": int(g.obs_matched.sum()),
               "matched_share": round(float(g.obs_matched.mean()), 4) if len(g) else None}
        rec |= ladder(rows, fa if len(rows) else fa.iloc[:0], used_keys, pa)
        rec["free_departure_segments_at_this_airport"] = int(
            fa[fa.kind.eq("departure") & ~fa.seg_key.isin(used_keys)]
            .seg_key.nunique())
        rep["by_airport"][a] = rec
        w = rec.get("why_not_matched", {})
        print(f"{a}: {rec['departures']:6d} rows, matched {rec['matched_share'] or 0:6.1%}, "
              f"unmatched {rec['unmatched_rows']:5d} | "
              + ", ".join(f"{k[0]}:{v}" for k, v in sorted(w.items()))
              + f" | repair modal->all-forms {rec.get('repair_join_on_every_broadcast_form', 0):4d}"
              f", window {rec.get('repair_widen_window_to_3600s', 0):4d}"
              f", geometric {rec.get('geometric_one_to_one', 0):4d}"
              f" (placebo {rec.get('geometric_one_to_one_placebo_plus_2h', 0)})", flush=True)

    (EXP / "analysis/identity.json").write_text(
        json.dumps(rep, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
