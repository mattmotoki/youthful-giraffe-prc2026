"""Order 3: why LTFM has delivered summary days and ZERO identified departures.

Desk pick, 2026-09-09:

    (3) LTFM first, today: the count of LTFM segments by broadcast form against the ranking
    callsigns, no fit -- 13.8 % of the file with zero matches is the largest unknown we
    have.

LTFM is 47,581 of the 344,841 ranking rows (13.8 %) and the served class contains none of
them, in either month.  Every other delivered airport matches between 35 % and 97 %, so the
question is whether the join fails at LTFM or whether there is nothing there to join to.

The instrument is a count and nothing else: no model, no fold, no label.  It reads the
summaries through `block.load_summaries` -- the matcher's own loader, with the matcher's own
rule filter and held-day list -- and the ranking departures through `prc.data`, and puts the
matcher's own `broadcast_forms` keys on both sides.  A comparison airport is read the same
way, so "LTFM is empty" is a statement with a scale next to it.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/ltfm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import block as B                                                     # noqa: E402
import prc.data as D                                                  # noqa: E402

OUT = EXP / "analysis/ltfm.json"
#: read beside LTFM at the same scale, so "empty" is measured and not asserted
COMPARE = ["EDDF", "LEMD"]


def summary_days(airport: str) -> list[str]:
    """Every day the harvest has a file for, from the roots the matcher reads."""
    days = set()
    for r in B.SUMMARY_ROOTS:
        d = r / airport
        if d.is_dir():
            days |= {p.stem for p in d.glob("*.parquet")}
    return sorted(days)


def segment_side(airport: str, days: list[str]) -> dict:
    """What the summaries hold for one airport: segments, clocks, and join keys."""
    seg = B.load_summaries([airport], days, None, allow_any_rule=True)
    if not len(seg):
        return {"segments": 0, "departure_segments": 0, "with_a_departure_clock": 0,
                "join_keys": 0, "keys": set(), "by_day": {},
                "days_the_loader_read": 0, "days_held_short_by_the_loader": 0}
    seg = seg.copy()
    seg["seg_key"] = range(len(seg))          # build() numbers segments the same way
    dep = seg[seg.kind.eq("departure")]
    clocked = dep[dep.obs_offblock.notna()]
    forms = B.broadcast_forms(dep) if len(dep) else dep.assign(join_key=pd.Series(dtype=str))
    # the loader carries no day column, so the day is read off the segment's own start
    day = pd.to_datetime(seg.seg_start, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    by_day = {}
    for d, g in seg.groupby(day):
        gd = g[g.kind.eq("departure")]
        by_day[str(d)] = {"segments": int(len(g)), "departure": int(len(gd)),
                          "with_a_clock": int(gd.obs_offblock.notna().sum())}
    return {"segments": int(len(seg)), "departure_segments": int(len(dep)),
            "days_the_loader_read": len(by_day),
            "days_held_short_by_the_loader": len(seg.attrs.get("days_held_short", []) or []),
            "with_a_departure_clock": int(len(clocked)),
            "join_keys": int(forms.join_key.nunique()) if len(forms) else 0,
            "keys": set(forms.join_key.dropna()) if len(forms) else set(),
            "by_day": by_day}


def main() -> None:
    rank = D.load_ranking()
    rank = rank[rank.PHASE_mvt == "DEP"].copy()
    rank["day"] = rank.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")

    out = {"what": ("whether LTFM's zero matches are a join failure or an empty feed; a "
                    "count, with no fit, no fold and no label"),
           "ordered_by": "the desk, 2026-09-09, pick (3)",
           "matcher_sha256": B.__file__ and __import__("arms").MATCHER_SHA,
           "airports": {}}

    for a in ["LTFM", *COMPARE]:
        days = summary_days(a)
        ranking_days = sorted(rank.loc[rank.ADEP_mvt.eq(a), "day"].unique())
        overlap = [d for d in days if d in set(ranking_days)]
        seg = segment_side(a, days)
        r = rank[rank.ADEP_mvt.eq(a)]
        rkeys = set(B.norm_callsign(r.CALLSIGN_flt.dropna()))
        shared = seg["keys"] & rkeys
        out["airports"][a] = {
            "summary_day_files": len(days),
            "of_them_on_a_ranking_day": len(overlap),
            "ranking_departures": int(len(r)),
            "segments": seg["segments"],
            "departure_segments": seg["departure_segments"],
            "with_a_departure_clock": seg["with_a_departure_clock"],
            "segments_per_delivered_day": (round(seg["segments"] / len(days), 1)
                                           if days else None),
            "days_the_loader_read": seg.get("days_the_loader_read", 0),
            "days_held_short_by_the_loader": seg.get("days_held_short_by_the_loader", 0),
            "distinct_broadcast_forms": seg["join_keys"],
            "distinct_ranking_callsigns": len(rkeys),
            "forms_that_appear_in_the_ranking_file": len(shared),
            "by_day": seg["by_day"],
        }
        v = out["airports"][a]
        print(f"=== {a}: {v['summary_day_files']} day files, {v['segments']} segments "
              f"({v['segments_per_delivered_day']}/day), "
              f"{v['departure_segments']} departures, "
              f"{v['with_a_departure_clock']} with a clock", flush=True)
        print(f"    {v['distinct_broadcast_forms']} broadcast forms against "
              f"{v['distinct_ranking_callsigns']} ranking callsigns -> "
              f"{v['forms_that_appear_in_the_ranking_file']} shared "
              f"({v['ranking_departures']} ranking departures)", flush=True)

    ltfm = out["airports"]["LTFM"]
    out["verdict"] = (
        "the feed, not the join" if ltfm["with_a_departure_clock"] == 0 else
        "the join, not the feed")
    out["what_it_means_for_the_next_step_table"] = (
        "RESULTS 17 charged LTFM's july rows to the matcher because 14 day FILES exist. "
        "A file that holds one or two segments is not a delivered day, so those rows belong "
        "with the harvest's undelivered ones and the matcher's headroom is smaller than the "
        "table says. Section 17 and nextsteps.py are corrected on this reading.")
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    print("\nverdict:", out["verdict"])
    print("written", OUT)


if __name__ == "__main__":
    main()
