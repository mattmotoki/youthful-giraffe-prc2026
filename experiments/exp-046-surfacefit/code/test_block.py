"""exp-046-surfacefit step 1: the block's match, tested against the pilot's matcher.

Three tests, because there are three things that can be wrong and they need separating.

**A0. The rule, and its version.**  The harvest's `all_segments()` against the pilot's
`segments()`, per day, on all seven days the pilot stored.  The pilot worker's Sol round 1
repaired the rule on 2026-09-08 (a `first_airborne` point with a NULL altitude is not a
take-off) after four months had been harvested under the shipped one; the harvest has
since **adopted** the repair and backfills already-run months from their stored points
(`exp-045-harvest/manifest.json`, `rule_note`), and every summary now carries
`airborne_alt_ok`, so either rule can be read off it.  This stage therefore (a) runs one
rule everywhere, asserted identical between the two implementations, (b) NORMALISES any
summary written before the repair through `airborne_alt_ok` (`block.load_summaries`), and
(c) keeps the divergence itself as a live measurement off the summaries: the number of
`departure` segments the repair removes.  The historical impact on matched departures,
measured while the two rules were both in the tree, is `analysis/rule-divergence.json`.

**A. The matcher.**  `exp-043-pilot/code/match.py:build()` is run and compared row by row
with the block's, on the same cohort rows.  Every difference must be attributable to a
named mechanism with a count:

  1. the block cuts segments at the UTC day boundary because the harvest kernel reads one
     daily archive at a time, while the pilot ran `segments()` over seven days of points
     at once — and adsb.lol's daily archives overlap, so that concatenation carries
     duplicate `(icao, t)` rows and a duplicated AIRBORNE point splits one ground run
     into two candidate departure segments;
  2. the null-altitude repair above, which removes candidate segments the block still has;
  3. a departure that blocked off inside the week but took off outside it has no summary
     under the block's take-off-day key.

**B. The published counts.**  The pilot's `analysis/match-*.json` reach numbers are
reported beside the block's with their deltas.  They are **reported, not asserted**: the
pilot is repairing its own experiment and its published counts move, and pinning this
test to a moving file would make it a test of the pilot's edit schedule.

    .venv/bin/python experiments/exp-046-surfacefit/code/test_block.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
import block as B                                            # noqa: E402
from imports import load as _load                            # noqa: E402

PC = _load("exp-043-pilot", "census")        # the pilot's `census`, not the harvest's
M = _load("exp-043-pilot", "match")
_seg = _load("exp-043-pilot", "segments").segments
_all = _load("exp-045-harvest", "allsegments", ("exp-043-pilot",)).all_segments

WEEK = [f"2025-07-{d:02d}" for d in range(7, 14)]
HELDOUT = WEEK[2:]


def pilot_frame() -> pd.DataFrame:
    """The pilot's own cohort rows, with the ranking-legal columns the block reads."""
    coh = pd.read_parquet(ROOT / "experiments/exp-043-pilot/work/cohort.parquet")
    return coh.assign(PHASE_mvt="DEP")


def prefix_train() -> pd.DataFrame:
    """The eleven 2025 months that are not July — the pilot's own map source."""
    frames = []
    for p in sorted((ROOT / "data/raw").glob("training_2025-*.parquet")):
        if "2025-07-01" in p.name:
            continue
        t = pd.read_parquet(p, columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
        frames.append(t[t.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt"))
    return pd.concat(frames, ignore_index=True)


def rule_divergence() -> tuple[dict, set, set]:
    """A0: the two implementations per day, the divergence the repair accounts for, and
    the segments the pilot's rule finds when it is applied ONE DAY AT A TIME."""
    perday, only_h, pilot_perday, only_p, differing = {}, set(), set(), 0, 0
    repair_removes = 0
    tied_aircraft, tied_rows, tied_icaos = 0, 0, set()
    for d in WEEK:
        pts = pd.read_parquet(ROOT / f"artifacts/exp-043/points_{d}.parquet")
        o = _seg(pts)
        n = _all(pts.assign(airport="EHAM"))
        nd = n[n.kind == "departure"]
        # An aircraft whose trace carries two points at the same instant orders itself
        # differently under a stable and an unstable sort, so the ground run splits at a
        # different point.  The harvest sorts stably as of exp-045 commit afc77ee; the
        # pilot's `segments.py` is frozen and sorts unstably.  The comparison therefore
        # SETS THOSE AIRCRAFT ASIDE and asserts identity on the rest, exactly as
        # `exp-045-harvest/code/check_rule.py` does, and reports how many it set aside.
        tie = pts.duplicated(subset=["icao", "t"], keep=False)
        tied = set(pts.loc[tie, "icao"].unique())
        tied_aircraft += len(tied)
        tied_icaos |= tied
        tied_rows += int(tie.sum())
        # `pilot_perday` is the pilot's own per-day population and stays WHOLE: it is the
        # attribution set for section A and a tied aircraft is a real segment there.  The
        # identity comparison below is what sets the tied aircraft aside.
        pilot_perday |= set(zip(o.icao, o.obs_wheels_off))
        ot = o[~o.icao.isin(tied)]
        ndt = nd[~nd.icao.isin(tied)]
        ko = set(zip(ot.icao, ot.obs_wheels_off))
        kn = set(zip(ndt.icao, ndt.obs_wheels_off))
        only_h |= (kn - ko)
        only_p += len(ko - kn)
        shared = ndt.merge(ot, on=["icao", "obs_wheels_off"], suffixes=("_h", "_p"))
        same = np.isclose(shared.obs_offblock_disp_h.to_numpy(float),
                          shared.obs_offblock_disp_p.to_numpy(float),
                          rtol=0, atol=0, equal_nan=True)
        differing += int((~same).sum())
        assert same.all(), (
            f"{d}: {int((~same).sum())} shared off-block instants differ on aircraft "
            "whose points carry no equal-instant tie")
        perday[d] = {"harvest": int(len(nd)), "pilot": int(len(o)),
                     "shared": int(len(shared))}
        # The repair, measured live off the artifact: departures the shipped rule would
        # have kept and the repaired one drops.  NOTE (round 2 FALLBACK, non-blocking):
        # now that every summary on disk is written under the repaired rule, a segment
        # that ends airborne always carries `airborne_alt_ok`, so this counter is zero by
        # construction and measures nothing.  It is kept as a tripwire: a non-zero value
        # would mean a pre-repair file had reappeared under a post-repair rule version.
        # The repair's real size, measured while both rules were in the tree, is
        # analysis/rule-divergence.json.
        sm = pd.read_parquet(ROOT / f"artifacts/exp-046/summaries/EHAM/{d}.parquet")
        if "airborne_alt_ok" in sm.columns:
            repair_removes += int((sm.ends_airborne.fillna(False).astype(bool)
                                   & ~sm.airborne_alt_ok.fillna(False).astype(bool)).sum())
    rep = {
        "departure_rows_per_day": perday,
        "shared_rows_off_block_identical": True,
        "shared_rows_differing": differing,
        "implementations_disagree_on": len(only_h) + only_p,
        "departures_the_repair_removes": repair_removes,
        "aircraft_set_aside_for_an_equal_instant_tie": tied_aircraft,
        "tied_points_over_the_week": tied_rows,
        "tie_note": (
            "exp-045 commit afc77ee: 92 of one gate day's 553,178 points share an "
            "(airport, icao, t) with another and 22 of those pairs disagree about "
            "on_ground, so an unstable sort lands the pair either way round and the "
            "ground run splits at a different point. The harvest now sorts stably; the "
            "pilot's segments.py is frozen and does not. Those aircraft are set aside "
            "here and the identity is asserted on the rest."),
        "divergence_note": (
            "the two implementations now run ONE rule and must agree exactly; "
            "`departures_the_repair_removes` is the repair's own size, measured live off "
            "the summaries through `airborne_alt_ok`. Its effect on MATCHED departures, "
            "measured while both rules were in the tree, is analysis/rule-divergence.json "
            "(6 segments of 5,441 and 5 matched departures of 4,930)."),
    }
    return rep, only_h, pilot_perday, tied_icaos


#: The placebo is run at SEVERAL shifts on purpose.  A two-hour shift alone passes for
#: the wrong reason: a row's own pushback then has an implied taxi of its real taxi plus
#: 7,200 s, which the [60 s, 7,200 s] bound rejects on its own, so the segment-end guard
#: is never exercised (round 2 FALLBACK, blocking 3).  1.0 h and 1.5 h sit INSIDE the
#: bound, so there the guard is the only thing standing between the rule and a wrong
#: movement, and what survives it is the rule's false-match rate on a cohort that
#: contains no true match at all.
PLACEBO_SHIFTS_H = [1.0, 1.5, 2.0]
#: the largest number of wide matches any placebo shift may produce, as a share of the
#: cohort.  At the measured 1,800 s segment-end window the worst shift gives 1 row of
#: 5,131 (0.02 %); 0.1 % is 5 rows, which leaves room for a different week without
#: leaving room for the 24 rows the old 3,600 s window produced.
PLACEBO_MAX_WIDE_SHARE = 0.001
#: the same bound on the STRICT rule.  It used to be exactly zero, and it was zero while
#: the strict join used the modal callsign alone.  Joining on every broadcast form
#: (`block.FORMS`, 2026-09-08, `analysis/identity.json`) buys 1,222 rows at EDDF and 49
#: on this week, and costs one shifted row of 5,131 at +1.0 h: a callsign whose two
#: rotations that day are about an hour apart, so a row moved an hour reaches the other
#: one inside the +/- 15 min window.  The test now measures that price rather than
#: forbidding it, and `strict_modal_only` proves where it comes from.
PLACEBO_MAX_STRICT_SHARE = 0.001

#: THE WHOLE-MONTH PER-AIRPORT BOUND (rebuild, 2026-09-09).  The two constants above are
#: asserted on the PILOT WEEK, at EHAM, on 5,131 rows, and they are right there.  What was
#: missing is any assertion at all on the other nine airports -- which is how EDDM came to
#: read 0.00120 on a whole ranking month (round 3 finding 7) while this suite stayed green.
#:
#: The desk routed *"a look at EDDM's callsign pool"* here.  Measured, the pool is not the
#: cause: EDDM broadcasts 1.609 forms per departure segment, FOURTH of nine, and LEMD, with
#: the largest pool of all, has the second smallest false share; Spearman against the false
#: share is +0.35.  What tracks it is SEGMENT DENSITY -- all-kind ground segments per
#: departure ranks EDDM, EHAM and LSZH first, second and third, the same three airports in
#: the same order as the false share, Pearson +0.884 (`analysis/eddm-density.json`).  A
#: shifted row lands falsely when the airport offers it more segments inside the window, so
#: the bound belongs per airport and scaled by that, not lane-wide.
#:
#: This is a DIFFERENT quantity on a DIFFERENT cohort from the two constants above, and it
#: does not replace them.  The floor covers an airport whose reception is too thin to scale
#: from -- LTFM delivers no summaries at all; no airport rides it today.
MONTH_FALSE_MATCH_FLOOR = 0.0003
MONTH_FALSE_MATCH_PER_DENSITY = 0.0005


def month_false_match_bound(density: float) -> float:
    """The per-airport whole-month bound, from the airport's own segment density.

    `density` is all-kind ground segments per departure, the quantity that tracks the
    measured false-match share; `code/finding7.py` measures it and asserts this bound, and
    `test_month_false_match_bound` below re-reads the record so the number cannot drift.
    """
    return max(MONTH_FALSE_MATCH_FLOOR,
               MONTH_FALSE_MATCH_PER_DENSITY * float(density))


def check_month_false_match_bound() -> dict:
    """Re-read `analysis/finding7.json` and re-assert the per-airport whole-month bound.

    The measurement itself is `code/finding7.py`, which rebuilds a whole ranking month
    three times; this is the cheap re-read that keeps the RECORDED number and the bound
    from drifting apart in silence.  It reports `unmeasured` rather than passing when the
    record predates the bound, so a stale file cannot make the suite green.
    """
    f7 = EXP / "analysis/finding7.json"
    if not f7.exists():
        return {"state": "unmeasured", "why": "analysis/finding7.json is absent"}
    rec = json.loads(f7.read_text())
    worst, dens = rec.get("worst_by_airport") or {}, rec.get("density_by_airport") or {}
    if not worst or not dens:
        return {"state": "unmeasured",
                "why": ("analysis/finding7.json predates the per-airport bound; re-run "
                        "code/finding7.py")}
    out = {"state": "checked", "floor": MONTH_FALSE_MATCH_FLOOR,
           "per_density": MONTH_FALSE_MATCH_PER_DENSITY, "by_airport": {}}
    for a, v in sorted(worst.items()):
        d = dens.get(a)
        if d is None:                     # an airport that delivers no summaries at all
            continue
        share = float(v["worst_false_share_across_shifts"])
        bound = month_false_match_bound(d)
        out["by_airport"][a] = {"density": d, "false_share": share, "bound": bound,
                                "headroom": round(bound / share, 2) if share else None}
        assert share <= bound, (
            f"{a} matched {share:.6f} of its rows on a cohort with no true match left, "
            f"above the {bound:.6f} its segment density of {d:.3f} allows")
    return out


def placebo(coh_raw: pd.DataFrame, train: pd.DataFrame, rule: str | None,
            real: pd.DataFrame) -> dict:
    """Does the match carry identity, or would it match anything?

    Every departure's `MVT_TIME` is moved forward, and the block is rebuilt on a cohort
    in which no row has a true match left.  Anything the rule still matches is a false
    match, and the funnel says which guard refused the rest.
    """
    def counts(b):
        m = b.match_mode
        return int((m == 0).sum()), int((m > 0).sum())

    rs, rw = counts(real)
    rows = int(len(real))
    nonmodal = int((real.matched_on_modal == 0).sum()) if "matched_on_modal" in real else 0
    out = {"cohort_rows": rows, "real_strict": rs, "real_wide": rw,
           "real_wide_funnel": real.attrs.get("wide_funnel"),
           "real_matched_on_a_non_modal_form": nonmodal,
           "max_wide_share_asserted": PLACEBO_MAX_WIDE_SHARE, "by_shift": {},
           "note": ("a rule that carried no identity would match the shifted rows too. "
                    "The funnel says which guard does the work, and a shift inside the "
                    "implied-taxi bound is the only one that tests the segment-end "
                    "guard at all.")}
    out["max_strict_share_asserted"] = PLACEBO_MAX_STRICT_SHARE
    for h in PLACEBO_SHIFTS_H:
        sh = coh_raw.copy()
        sh["MVT_TIME_UTC_mvt"] = sh.MVT_TIME_UTC_mvt + pd.Timedelta(hours=h)
        pl = B.build(sh, train, airports=["EHAM"], require_rule=rule)
        ps, pw = counts(pl)
        fn = pl.attrs.get("wide_funnel") or {}
        out["by_shift"][f"{h:+.1f}h"] = {
            "strict": ps, "wide": pw, "wide_share": pw / max(rows, 1),
            "funnel": fn,
            "inside_the_implied_taxi_bound": bool(h * 3600 < B.WIDE_MAX_TAXI),
            "the_segment_end_guard_removed": (fn.get("after_implied_taxi", 0)
                                              - fn.get("after_segment_end_window", 0)),
        }
    worst = max((v["wide"] for v in out["by_shift"].values()), default=0)
    out["worst_wide_across_shifts"] = worst
    out["worst_wide_share"] = worst / max(rows, 1)
    # WHERE the price comes from: the same shift with the modal-only join, which is
    # what this rule was before 2026-09-08.  If that reads 0 the whole placebo cost is
    # the form join's, and the test says so rather than leaving it to be inferred.
    h0 = PLACEBO_SHIFTS_H[0]
    sh0 = coh_raw.copy()
    sh0["MVT_TIME_UTC_mvt"] = sh0.MVT_TIME_UTC_mvt + pd.Timedelta(hours=h0)
    keep_forms = B.FORMS
    try:
        B.FORMS = False
        pl0 = B.build(sh0, train, airports=["EHAM"], require_rule=rule)
        s0, w0 = counts(pl0)
    finally:
        B.FORMS = keep_forms
    out["strict_modal_only"] = {
        "shift_h": h0, "strict": s0, "wide": w0,
        "note": ("the modal-only join at the shift that bites: everything the strict "
                 "placebo now shows is the price of joining on every broadcast form")}
    return out


def main() -> None:
    coh_raw = pilot_frame()
    train = prefix_train()
    # the pilot week's own rule version, resolved and required: the loader is fail-closed
    week_rule, week_rule_report = B.one_rule([(["EHAM"], WEEK)])
    blk = B.build(coh_raw, train, airports=["EHAM"], require_rule=week_rule)
    pm = B.prefix_map(train)
    coh = B.cohort(coh_raw, pm)

    a0, only_harvest_segments, pilot_perday, tied_icaos = rule_divergence()
    # the same pilot rule on seven days at once, against the same rule per day: the
    # harvest kernel reads one daily archive, so the per-day assembly is not a choice
    wk, _ = PC.load_slices(WEEK)
    wkseg = _seg(wk)
    pilot_week = set(zip(wkseg.icao, wkseg.obs_wheels_off))
    a0["assembly"] = {
        "week_points_after_the_pilot_dedup": int(len(wk)),
        "pilot_rule_departures_per_day": len(pilot_perday),
        "pilot_rule_departures_week_concatenated": len(pilot_week),
        "only_per_day": len(pilot_perday - pilot_week),
        "only_week_concatenated": len(pilot_week - pilot_perday),
        "note": ("the SAME pilot rule on both sides. A ground run that crosses midnight "
                 "is one segment in the week view and two in the per-day view, and a "
                 "day file carries a few points of the previous UTC day, so the two "
                 "assemblies do not produce the same segment population. The harvest "
                 "kernel reads one daily archive, so the per-day assembly is what the "
                 "served observation is."),
    }

    # ---- A. against the pilot's own matcher, on the same rows ----------------
    pil = M.build(WEEK)[["MVT_ID_mvt", "obs_wheels_off", "obs_offblock_disp", "mvt_s",
                         "icao"]]
    pil = pil.rename(columns={"obs_wheels_off": "p_wo", "obs_offblock_disp": "p_off",
                              "icao": "p_icao"})
    j = (coh[["MVT_ID_mvt", "takeoff_day", "has_nm"]]
         .merge(blk[["MVT_ID_mvt", "obs_icao", "obs_wheels_off", "obs_taxi",
                     "dt_takeoff", "obs_matched", "obs_has_clock", "match_mode"]],
                on="MVT_ID_mvt", how="left")
         .merge(pil, on="MVT_ID_mvt", how="left"))
    j["p_matched"] = j.p_wo.notna()
    j["p_taxi"] = j.mvt_s - j.p_off
    j["obs_matched"] = j.obs_matched.astype("boolean").fillna(False).astype(bool)

    both = j.p_matched & j.obs_matched
    same_seg = ((j.loc[both, "obs_icao"] == j.loc[both, "p_icao"])
                & np.isclose(j.loc[both, "obs_wheels_off"], j.loc[both, "p_wo"],
                             rtol=0, atol=1e-6))
    same_taxi = np.isclose(j.loc[both, "obs_taxi"], j.loc[both, "p_taxi"],
                           rtol=0, atol=1e-6, equal_nan=True)

    only_block = j[j.obs_matched & ~j.p_matched].copy()
    only_pilot = j[j.p_matched & ~j.obs_matched].copy()
    only_block["from_divergent_segment"] = [
        (i, w) in only_harvest_segments
        for i, w in zip(only_block.obs_icao, only_block.obs_wheels_off)]
    only_block["from_a_per_day_segment"] = [
        (i, w) in pilot_perday
        for i, w in zip(only_block.obs_icao, only_block.obs_wheels_off)]
    # the desk's wide rule (2026-09-08): a departure the strict rule leaves unmatched,
    # matched on the callsign plus a plausible off-block, with the clock check carried as
    # a quality feature.  The pilot has no such pass, so these rows are its by design.
    only_block["from_the_wide_match"] = only_block.match_mode.fillna(0) > 0
    # an aircraft whose points carry two readings at one instant splits its ground run
    # differently under the harvest's stable sort and the pilot's frozen unstable one
    only_block["from_a_tied_aircraft"] = only_block.obs_icao.isin(tied_icaos)
    # THIS lane's own change, and the first mechanism the attribution tries: the strict
    # join uses every callsign form the ground run broadcast, where the pilot used the
    # modal one alone (`block.FORMS`, 2026-09-08).  Measured by rebuilding the week with
    # the modal-only join rather than inferred from `matched_on_modal`, because the
    # normalisation of leading zeros is part of the same change.
    keep_forms = B.FORMS
    try:
        B.FORMS = False
        blk_modal = B.build(coh_raw, train, airports=["EHAM"], require_rule=week_rule)
    finally:
        B.FORMS = keep_forms
    modal_matched = set(blk_modal.loc[blk_modal.obs_matched, "MVT_ID_mvt"])
    only_block["from_the_form_join"] = ~only_block.MVT_ID_mvt.isin(modal_matched)
    only_pilot["takeoff_outside_week"] = ~only_pilot.takeoff_day.isin(WEEK)

    # ---- the shared rows that differ, attributed the same way as the unshared ----
    # Sol round 1 (blocking 7): "different segment chosen" and "differing obs_taxi" were
    # reported and then guarded by an arbitrary `<= 20`.  Each such row is now assigned a
    # named mechanism and the count of rows with NO mechanism is asserted to be zero.
    perday_only = pilot_perday - pilot_week
    week_only = pilot_week - pilot_perday
    sh = j[both].copy()

    def _in(sset, ic, wo):
        return np.array([(i, w) in sset for i, w in zip(ic, wo)], dtype=bool)

    sh["blk_segment_is_per_day_only"] = _in(perday_only, sh.obs_icao, sh.obs_wheels_off)
    sh["pilot_segment_is_week_only"] = _in(week_only, sh.p_icao, sh.p_wo)
    sh["blk_segment_is_rule_divergent"] = _in(only_harvest_segments, sh.obs_icao,
                                              sh.obs_wheels_off)
    sh["blk_used_the_wide_match"] = sh.match_mode.fillna(0) > 0
    sh["from_a_tied_aircraft"] = sh.obs_icao.isin(tied_icaos)
    sh["attributed"] = (sh.blk_segment_is_per_day_only | sh.pilot_segment_is_week_only
                        | sh.blk_segment_is_rule_divergent | sh.blk_used_the_wide_match
                        | sh.from_a_tied_aircraft)
    sh["same_segment"] = np.asarray(same_seg)
    sh["same_taxi"] = np.asarray(same_taxi)

    diff_seg = sh[~sh.same_segment]
    diff_taxi = sh[~sh.same_taxi]
    diff_seg_divergent = int(diff_seg.blk_segment_is_rule_divergent.sum())
    shared_diff = {
        "different_segment_chosen": int(len(diff_seg)),
        "different_segment_from_a_per_day_segment":
            int(diff_seg.blk_segment_is_per_day_only.sum()),
        "different_segment_from_a_week_only_pilot_segment":
            int(diff_seg.pilot_segment_is_week_only.sum()),
        "different_segment_from_the_rule_repair": diff_seg_divergent,
        "different_segment_from_the_wide_match":
            int(diff_seg.blk_used_the_wide_match.sum()),
        "different_segment_from_a_tied_aircraft":
            int(diff_seg.from_a_tied_aircraft.sum()),
        "different_segment_unattributed": int((~diff_seg.attributed).sum()),
        "differing_obs_taxi": int(len(diff_taxi)),
        "differing_obs_taxi_from_a_per_day_segment":
            int(diff_taxi.blk_segment_is_per_day_only.sum()),
        "differing_obs_taxi_from_a_week_only_pilot_segment":
            int(diff_taxi.pilot_segment_is_week_only.sum()),
        "differing_obs_taxi_from_the_rule_repair":
            int(diff_taxi.blk_segment_is_rule_divergent.sum()),
        "differing_obs_taxi_from_the_wide_match":
            int(diff_taxi.blk_used_the_wide_match.sum()),
        "differing_obs_taxi_from_a_tied_aircraft":
            int(diff_taxi.from_a_tied_aircraft.sum()),
        "differing_obs_taxi_unattributed": int((~diff_taxi.attributed).sum()),
    }

    # ---- B. beside the published reach numbers ------------------------------
    def reach(days: list[str]) -> dict:
        sel = coh[coh.takeoff_day.isin(days)]
        b = blk[blk.MVT_ID_mvt.isin(sel.MVT_ID_mvt)]
        m = sel[["MVT_ID_mvt", "has_nm"]].merge(b, on="MVT_ID_mvt", how="left")
        mt = m.obs_matched.astype("boolean").fillna(False)
        return {"departures": int(len(m)), "matched": int(mt.sum()),
                "with_clock": int(m.obs_has_clock.astype("boolean").fillna(False).sum()),
                "matched_share": round(float(mt.mean()), 4),
                "matched_nm": round(float(mt[m.has_nm].mean()), 4),
                "matched_no_nm": round(float(mt[~m.has_nm].mean()), 4),
                "no_nm_rows": int((~m.has_nm).sum())}

    def published(name: str) -> dict:
        p = json.loads((ROOT / f"experiments/exp-043-pilot/analysis/{name}.json").read_text())
        return {k: p["reach"][k] for k in
                ("departures", "matched", "matched_share", "matched_nm", "matched_no_nm",
                 "no_nm_rows", "with_obs_offblock_disp")}

    a0["rule_required"] = week_rule
    a0["rule_cover"] = week_rule_report
    a0["rule_versions_used"] = blk.attrs.get("rules_seen")
    a0["days_rejected_for_a_different_rule"] = len(blk.attrs.get("days_wrong_rule", []))
    a0["normalised_by_airborne_alt_ok"] = blk.attrs.get("normalised_by_airborne_alt_ok")
    assert len(a0["rule_versions_used"] or []) <= 1, (
        f"the block mixed rule versions: {a0['rule_versions_used']}")
    assert a0["implementations_disagree_on"] == 0, (
        f"the two implementations of one rule disagree on "
        f"{a0['implementations_disagree_on']} segments")

    rep = {
        "A0_the_rule": a0,
        "A_against_pilot_matcher": {
            "cohort_rows": int(len(coh)),
            "pilot_matched": int(j.p_matched.sum()),
            "block_matched": int(j.obs_matched.sum()),
            "both_matched": int(both.sum()),
            "same_segment_chosen": int(same_seg.sum()),
            "different_segment_chosen": int((~same_seg).sum()),
            "of_which_the_divergent_segment": diff_seg_divergent,
            "identical_obs_taxi": int(same_taxi.sum()),
            **shared_diff,
            "only_block_matched": int(len(only_block)),
            "only_block_from_a_divergent_segment": int(only_block.from_divergent_segment.sum()),
            "only_block_from_a_per_day_segment": int(only_block.from_a_per_day_segment.sum()),
            "only_block_from_the_wide_match": int(only_block.from_the_wide_match.sum()),
            "only_block_from_a_tied_aircraft": int(only_block.from_a_tied_aircraft.sum()),
            "only_block_from_the_form_join": int(only_block.from_the_form_join.sum()),
            "only_block_matched_by_the_modal_join_too": int(len(modal_matched)),
            "only_block_unattributed": int((~only_block.from_divergent_segment
                                            & ~only_block.from_a_per_day_segment
                                            & ~only_block.from_the_wide_match
                                            & ~only_block.from_a_tied_aircraft
                                            & ~only_block.from_the_form_join).sum()),
            "wide_matches_total": blk.attrs.get("n_wide_matches"),
            "only_pilot_matched": int(len(only_pilot)),
            "only_pilot_takeoff_outside_week": int(only_pilot.takeoff_outside_week.sum()),
        },
        "B_reach": {
            "week_block": reach(WEEK), "week_published": published("match-week"),
            "heldout_block": reach(HELDOUT), "heldout_published": published("match-heldout"),
            "note": ("reported, not asserted: the pilot is repairing its own experiment "
                     "and its published counts move with the rule."),
        },
        "ambiguity": {"n_multi_seg_block": blk.attrs.get("n_multi_seg"),
                      "n_multi_dep_block": blk.attrs.get("n_multi_dep")},
        "clock_check": {"median": float(blk.dt_takeoff.median()),
                        "share_within_60s": float((blk.dt_takeoff.abs() <= 60).mean(skipna=True))},
        "C_placebo": placebo(coh_raw, train, week_rule, blk),
        "summaries_days_missing": [f"{a} {d}" for a, d in blk.attrs.get("days_missing", [])],
    }

    # --- the assertions: every difference is attributed, none is unexplained ---
    a = rep["A_against_pilot_matcher"]
    assert a["only_block_unattributed"] == 0, (
        f"{a['only_block_unattributed']} rows the block matched and the pilot did not are "
        "explained by NEITHER the null-altitude repair NOR the per-day assembly NOR the "
        "wide match")
    assert a["only_pilot_matched"] == a["only_pilot_takeoff_outside_week"], (
        f"{a['only_pilot_matched'] - a['only_pilot_takeoff_outside_week']} rows the pilot "
        "matched and the block did not are NOT explained by the take-off-day key")
    assert a["different_segment_unattributed"] == 0, (
        f"{a['different_segment_unattributed']} shared rows choose a different segment "
        "for no named reason")
    assert a["differing_obs_taxi_unattributed"] == 0, (
        f"{a['differing_obs_taxi_unattributed']} shared rows carry a different obs_taxi "
        "for no named reason")
    # the published counts, asserted as a band and not as a point: the pilot repairs its
    # own experiment and its counts move with the rule, but they cannot move far
    for k, (got, pub) in {
        "week": (rep["B_reach"]["week_block"]["matched"],
                 rep["B_reach"]["week_published"]["matched"]),
        "heldout": (rep["B_reach"]["heldout_block"]["matched"],
                    rep["B_reach"]["heldout_published"]["matched"]),
    }.items():
        assert abs(got - pub) / max(pub, 1) <= 0.01, (
            f"{k}: the block matches {got} against the pilot's published {pub}, more "
            "than 1 % apart")
    a["verdict"] = (
        f"The block matches {a['block_matched']} of {a['cohort_rows']} departures against "
        f"the pilot's {a['pilot_matched']}; {a['both_matched']} are matched by both and "
        f"{a['same_segment_chosen']} of those choose the same segment, at an identical "
        f"obs_taxi on {a['identical_obs_taxi']}. EVERY difference carries a named "
        "mechanism and the count with no mechanism is zero in all three tables. "
        f"{a['only_pilot_matched']} rows only the pilot matches. "
        f"{a['only_block_matched']} only the block matches: "
        f"{a['only_block_from_the_form_join']} through the strict join reading every "
        "callsign form the ground run broadcast where the pilot read the modal one alone, "
        f"{a['only_block_from_the_wide_match']} through the desk's wide match, "
        f"{a['only_block_from_a_tied_aircraft']} on an aircraft whose points carry two "
        "readings at one instant (the harvest sorts stably as of exp-045 commit afc77ee "
        "and the pilot's frozen segments.py does not), "
        f"{a['only_block_from_a_per_day_segment']} through a segment the pilot's own rule "
        "finds one day at a time but not over seven days at once "
        f"({a0['assembly']['only_per_day']} such departure events over the week against "
        f"{a0['assembly']['only_week_concatenated']} the other way), and "
        f"{a['only_block_from_a_divergent_segment']} through a segment the null-altitude "
        "repair removes; the sets overlap, which is why they sum to more than "
        f"{a['only_block_matched']}. Of the {a['different_segment_chosen']} shared rows "
        f"that choose a different segment, all {a['different_segment_from_a_tied_aircraft']} "
        f"are on a tied aircraft; so are all {a['differing_obs_taxi_from_a_tied_aircraft']} "
        f"of the {a['differing_obs_taxi']} that carry a different obs_taxi. The harvest "
        "kernel reads one daily archive, so the per-day assembly is the served "
        "observation and that difference is a property of the source, not a defect.")

    c = rep["C_placebo"]
    for k, v in c["by_shift"].items():
        assert v["strict"] / max(c["cohort_rows"], 1) <= PLACEBO_MAX_STRICT_SHARE, (
            f"the strict rule matched {v['strict']} of {c['cohort_rows']} rows whose "
            f"take-off was moved {k}, above the {PLACEBO_MAX_STRICT_SHARE:.1%} this test "
            "allows outside its +/- 15 min window")
        assert v["wide_share"] <= PLACEBO_MAX_WIDE_SHARE, (
            f"the wide rule matched {v['wide']} of {c['cohort_rows']} rows whose take-off "
            f"was moved {k}, above the {PLACEBO_MAX_WIDE_SHARE:.1%} this test allows")
    # and at least one shift must actually exercise the segment-end guard, or the test
    # is only measuring the implied-taxi bound
    assert any(v["the_segment_end_guard_removed"] > 0 for v in c["by_shift"].values()), (
        "no placebo shift reached the segment-end guard: every candidate was already "
        "refused by the implied-taxi bound, so the guard is untested")

    rep["month_false_match_bound"] = check_month_false_match_bound()

    (EXP / "analysis/block-test.json").write_text(json.dumps(rep, indent=2, sort_keys=True,
                                                             default=float))
    print(json.dumps(rep, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
