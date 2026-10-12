"""Why LFPG's clocked segments go unclaimed -- label-free, before the reach spec is written.

Desk, 2026-09-09, after the ceiling:

    LFPG identity read FIRST, then the reach block ... an identity failure that leaves 82 %
    of LFPG's clocked segments unclaimed is a specific cause, and the same cause may sit
    inside EDDF's 41 % -- reading it before writing the spec makes the block a repair rather
    than a search. Read it label-free: for LFPG's 7,266 unclaimed clocked segments, what the
    broadcast forms look like against the ranking callsigns (airline prefixes, form lengths,
    time-of-day, day boundary, box), and the same breakdown at EDDF for comparison; report
    the cause candidates with counts.

**How a segment is known to be unclaimed, without re-implementing the matcher.**  The block
carries `obs_icao` and `obs_taxi`, and `obs_offblock` is `mvt_s - obs_taxi`, so
`(icao, obs_offblock)` identifies the segment the matcher actually took.  Every clocked
departure segment outside that set is unclaimed by the matcher's own account rather than by
this file's opinion of it.

**The cause ladder**, each step conditional on the one before failing, so the counts add up:

  1. `no_such_callsign`  no form of the segment appears among the airport's ranking join
     keys at all.  This bucket is then SPLIT, because it turned out not to be one thing:
     `is_a_ranking_arrival` (the form is a ranking ARRIVAL into the same airport -- a
     departure run still broadcasting the inbound identity) against `neither`.
  2. `wrong_day`         a form matches, but on no ranking day within a day either side.
  3. `outside_window`    a form matches on the day, but no ranking row of that form sits
     within the wide rule's +/- 3600 s of the segment's take-off.
  4. `contended`         a form matches inside the window, so a row WAS available and the
     segment lost it: the departure took a nearer segment, or the segment lost the
     one-to-one dedup.  This is the only bucket a matcher change can convert.

Everything here is a count on inputs.  No label, no fold, no fit, nothing served.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/identityread.py
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
sys.path.insert(0, str(ROOT / "src"))

import arms as A                                                      # noqa: E402
import block as B                                                     # noqa: E402
import prc.data as D                                                  # noqa: E402

#: `--all` reads every airport, for the reach spec's expected class diff, and writes its
#: own file so the two-airport reading RESULTS 19 cites is not overwritten
ALL = "--all" in sys.argv
OUT = EXP / ("analysis/identity-read-all.json" if ALL else "analysis/identity-read.json")
AIRPORTS = None if ALL else ["LFPG", "EDDF"]
MONTHS = ["2026-01", "2026-07"]
DAY = 86400.0


def top(s: pd.Series, n: int = 12) -> dict:
    return {str(k): int(v) for k, v in s.value_counts().head(n).items()}


def main() -> None:
    rank = D.load_ranking().copy()          # BOTH phases: the ARR side is the finding
    rank["month"] = rank.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in D.MONTHS], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")

    out = {"what": ("why clocked ground segments at LFPG go unclaimed, and whether the same "
                    "cause sits inside EDDF's gap; counts on inputs, no label and no fit"),
           "ordered_by": "the desk, 2026-09-09, LFPG identity read before the reach spec",
           "matcher_sha256": A.MATCHER_SHA,
           "cause_ladder": ["no_such_callsign", "wrong_day", "outside_window", "contended"],
           "by_airport_month": {}}

    for month in MONTHS:
        dep_all = rank[rank.month.eq(month) & rank.PHASE_mvt.eq("DEP")].copy()
        rule, _ = B.one_rule([(sorted(set(dep_all.ADEP_mvt.dropna()) & set(B.AIRPORTS)),
                               sorted(dep_all.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
                                      .unique()))])
        blk = B.build(dep_all, tr, require_rule=rule)
        coh = B.cohort(dep_all, B.prefix_map(tr))
        m = coh[["MVT_ID_mvt", "ADEP_mvt", "callsign", "mvt_s"]].merge(
            blk[["MVT_ID_mvt", "obs_icao", "obs_taxi"]], on="MVT_ID_mvt", how="left")
        m["join_key"] = B.norm_callsign(m.callsign)
        m["offblock"] = m.mvt_s - m.obs_taxi

        days = sorted(dep_all.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").unique())
        for a in (AIRPORTS or sorted(set(dep_all.ADEP_mvt.dropna()) & set(B.AIRPORTS))):
            r = m[m.ADEP_mvt.eq(a)]
            # dropna on the two columns SEPARATELY misaligns them and empties the set;
            # the pair has to survive as a pair
            cl = r[["obs_icao", "offblock"]].dropna()
            claimed = set(zip(cl.obs_icao.astype(str),
                              cl.offblock.round(0).astype("int64")))
            rk = r[r.join_key.ne("")]
            keys_all = set(rk.join_key)
            by_key_day: dict[str, set] = {}
            by_key_t: dict[str, np.ndarray] = {}
            for k, g in rk.groupby("join_key"):
                by_key_day[k] = set((g.mvt_s // DAY).astype("int64"))
                by_key_t[k] = g.mvt_s.to_numpy("float64")

            seg = B.load_summaries([a], days, require_rule=rule)
            if not len(seg):                 # LTFM january: files exist, segments do not
                out["by_airport_month"][f"{month} {a}"] = {
                    "clocked_departure_segments": 0, "claimed_by_the_matcher": 0,
                    "unclaimed": 0, "note": "no segment under the resolved rule"}
                print(f"=== {month} {a}: no segments under the resolved rule", flush=True)
                continue
            seg = seg[seg.kind.eq("departure") & seg.obs_offblock.notna()].copy()
            if not len(seg):                 # LTFM july: segments, but none with a clock
                out["by_airport_month"][f"{month} {a}"] = {
                    "clocked_departure_segments": 0, "claimed_by_the_matcher": 0,
                    "unclaimed": 0, "note": "segments exist but none carries a clock"}
                print(f"=== {month} {a}: no clocked departure segment", flush=True)
                continue
            seg["seg_key"] = range(len(seg))
            smon = pd.to_datetime(seg.seg_start, unit="s", utc=True).dt.strftime("%Y-%m")
            seg = seg[smon.eq(month)]
            is_claimed = np.array([(i, o) in claimed for i, o in
                                   zip(seg.icao.astype(str),
                                       seg.obs_offblock.round(0).astype("int64"))],
                                  dtype=bool)
            un = seg[~is_claimed].copy()

            forms = B.broadcast_forms(un)
            g = forms.groupby("seg_key")
            cause, ex = {}, {}
            for sk, gg in g:
                ks = list(gg.join_key)
                hit = [k for k in ks if k in keys_all]
                if not hit:
                    cause[sk] = "no_such_callsign"; ex[sk] = ks[0]; continue
                t = float(un.loc[un.seg_key.eq(sk), "obs_offblock"].iloc[0])
                wo = un.loc[un.seg_key.eq(sk), "obs_wheels_off"].iloc[0]
                d0 = int(t // DAY)
                onday = [k for k in hit if by_key_day[k] & {d0 - 1, d0, d0 + 1}]
                if not onday:
                    cause[sk] = "wrong_day"; ex[sk] = hit[0]; continue
                anchor = float(wo) if pd.notna(wo) else t
                inwin = [k for k in onday
                         if np.any(np.abs(by_key_t[k] - anchor) <= B.WIDE_WINDOW)]
                cause[sk] = "contended" if inwin else "outside_window"
                ex[sk] = (inwin or onday)[0]
            un["cause"] = un.seg_key.map(cause).fillna("no_join_key")

            # ---- what `no_such_callsign` actually is -------------------------------
            # The ranking file has an ARR side of the same size as its DEP side.  A
            # departure run whose transponder never switched off the inbound callsign
            # carries an ARRIVAL's identity, and no callsign join can ever claim it.
            akeys = set(B.norm_callsign(
                rank[(rank.PHASE_mvt == "ARR") & rank.ADES_mvt.eq(a)
                     & rank.month.eq(month)].CALLSIGN_flt).dropna()) - {""}
            per = forms.groupby("seg_key").join_key.agg(set)
            is_arr = per.apply(lambda ks: bool(ks & akeys))
            nsc = un[un.cause.eq("no_such_callsign")]
            arr_seg = nsc[nsc.seg_key.map(is_arr).fillna(False)]
            rec_arr = {
                "no_such_callsign": int(len(nsc)),
                "of_them_a_ranking_arrival_into_this_airport": int(len(arr_seg)),
                "share": round(len(arr_seg) / max(1, len(nsc)), 4),
                "neither_a_departure_nor_an_arrival": int(len(nsc) - len(arr_seg))}

            # ---- and whether they are REAL unclaimed departures --------------------
            # A time-only candidate: an unclaimed ranking departure whose MVT_TIME sits
            # within w of the segment's take-off.  Uniqueness inside the window is the
            # false-match control -- one candidate and one only, or the segment is not
            # identified by time either.
            um = np.sort(r.loc[r.obs_taxi.isna(), "mvt_s"].to_numpy("float64"))
            wo = arr_seg.obs_wheels_off.dropna().to_numpy("float64")
            windows = {}
            for w in (60, 300, 900):
                n = (np.searchsorted(um, wo + w) - np.searchsorted(um, wo - w))
                windows[f"pm{w}s"] = {
                    "segments_with_a_candidate": int((n >= 1).sum()),
                    "segments_with_exactly_one": int((n == 1).sum()),
                    "share_exactly_one": round(float((n == 1).mean()) if len(n) else 0.0, 4)}
            rec_arr["unclaimed_ranking_departures_at_this_airport"] = int(len(um))
            rec_arr["time_only_candidates"] = windows
            un["form_len"] = un.callsign_modal.fillna("").astype(str).str.len()
            un["prefix"] = un.callsign_modal.fillna("").astype(str).str.extract(
                r"^([A-Z]+)")[0]
            hour = pd.to_datetime(un.obs_offblock, unit="s", utc=True).dt.hour
            key = f"{month} {a}"
            rec = {
                "clocked_departure_segments": int(len(seg)),
                "claimed_by_the_matcher": int(len(seg) - len(un)),
                "unclaimed": int(len(un)),
                "cause_counts": top(un.cause, 8),
                "cause_share": {k: round(v / max(1, len(un)), 4)
                                for k, v in top(un.cause, 8).items()},
                "unclaimed_prefixes": top(un.prefix),
                "claimed_prefixes": top(seg.loc[~seg.seg_key.isin(un.seg_key), "callsign_modal"]
                                        .fillna("").astype(str).str.extract(r"^([A-Z]+)")[0]),
                "ranking_prefixes": top(rk.join_key.str.extract(r"^([A-Z]+)")[0]),
                "unclaimed_form_length": top(un.form_len, 10),
                "unclaimed_by_hour_utc": top(hour, 24),
                "unclaimed_with_no_callsign_at_all":
                    int(un.callsign_modal.isna().sum() + un.callsign_modal.eq("").sum()),
                "unclaimed_with_no_wheels_off": int(un.obs_wheels_off.isna().sum()),
                "unclaimed_missing_a_stand_position": int(un.stand_lat.isna().sum()),
                "claimed_missing_a_stand_position":
                    int(seg.loc[~seg.seg_key.isin(un.seg_key), "stand_lat"].isna().sum()),
            }
            rec["what_no_such_callsign_really_is"] = rec_arr
            out["by_airport_month"][key] = rec
            print(f"=== {key}: {rec['clocked_departure_segments']} clocked segments, "
                  f"{rec['claimed_by_the_matcher']} claimed, {rec['unclaimed']} unclaimed",
                  flush=True)
            for k, v in rec["cause_counts"].items():
                print(f"      {k:20s} {v:6d}  ({100 * rec['cause_share'][k]:5.1f} %)",
                      flush=True)
            w60 = rec_arr["time_only_candidates"]["pm60s"]
            print(f"      of no_such_callsign, a ranking ARRIVAL into {a}: "
                  f"{rec_arr['of_them_a_ranking_arrival_into_this_airport']} "
                  f"({100 * rec_arr['share']:.1f} %), neither "
                  f"{rec_arr['neither_a_departure_nor_an_arrival']}", flush=True)
            print(f"      of those, exactly one unclaimed departure within 60 s: "
                  f"{w60['segments_with_exactly_one']} "
                  f"({100 * w60['share_exactly_one']:.1f} %)", flush=True)
            print(f"      no wheels_off {rec['unclaimed_with_no_wheels_off']:6d}   "
                  f"no stand position {rec['unclaimed_missing_a_stand_position']:6d} "
                  f"(claimed: {rec['claimed_missing_a_stand_position']})", flush=True)

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\nwritten", OUT)


if __name__ == "__main__":
    main()
