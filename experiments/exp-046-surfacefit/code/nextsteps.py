"""Order 7: where the ranking file's unserved rows are, and what each block is worth.

Desk order, 2026-09-09:

    then write the lane's next-step table in RESULTS ... every candidate next block -- the
    526 occupied-unwritten rows, a direct label model on the observation instead of a
    residual, the fold-side push items of 9, reach (rows with an observation the matcher
    does not identify), anything else -- with kind of work, cost, upside at face with the
    currency named, what it gives up, revival line.

This file is the arithmetic behind the counts and the faces in RESULTS section 17, so the
table cites a path rather than a paragraph.  It reads two committed records and fits
nothing:

  * `analysis/class-manifest-v16.json` -- rows and `obs_has_clock` per airport per ranking
    month, on the served vector's own bytes;
  * `analysis/ranking-population.json` -- the airport-days v16's block actually read, which
    is what separates "the harvest has not delivered this day" from "the day is here and the
    matcher does not identify the departure".

**The split's one assumption, stated.**  An airport-month missing `k` of its 31 days is
charged `rows * k / 31` rows to the harvest, pro rata, and the remainder to the matcher.
Days are not equal, so the split is approximate at the airport-month level; it is exact
where a month is whole (every july airport but LIRF and LTFM) and where it is empty (LTFM
january).  The boundary day of the month before is dropped so a whole month reads 31/31.

**The currency, and why the face is an upper bound.**  Board seconds, not fold seconds:
section 16 measured the conversion.  `seconds(n) = L - sqrt(L**2 - g*n/N)` with `g` the
served arm's mean squared gain per served row and `L` v16's board level.  A row is unmatched
BECAUSE its evidence is weak, so it cannot be worth what the average matched row is worth --
the counts are exact, the per-row value is borrowed from rows that are easier.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/nextsteps.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent

import arms as A                                                      # noqa: E402

OUT = EXP / "analysis/nextsteps.json"
#: v16's board RMSE -- a reading, score-feedback rule: nothing is fitted to it and no arm is chosen by it
BOARD_LEVEL_S = 246.83
#: the two clean readings of section 16.3 agree here; the level alone is about x1.2
FULL_TRANSFER = 1.492
DAYS_IN_MONTH = 31


def main() -> None:
    man = json.loads((EXP / "analysis/class-manifest-v16.json").read_text())
    cpath = EXP / "analysis/reach-ceiling.json"
    ceiling = json.loads(cpath.read_text()) if cpath.exists() else None
    pop = json.loads((EXP / "analysis/ranking-population.json").read_text())["latest"]
    sf = json.loads((EXP / "analysis/searchfit.json").read_text())

    n_rows = int(man["totals"]["rows"])
    served = int(man["totals"]["served_by_exp_046"])
    g_all = float(sf["what_is_left"]["per_step"]["v13"]["s2_gain_of_the_served_arm"])
    per_row = g_all * n_rows / served          # s2 per SERVED row, not per file row

    def face(n: int) -> dict:
        g = per_row * n / n_rows
        lvl = BOARD_LEVEL_S - (BOARD_LEVEL_S ** 2 - g) ** 0.5
        return {"rows": n, "board_s_at_the_level_currency": lvl,
                "board_s_at_the_two_clean_readings": lvl * FULL_TRANSFER}

    buckets: collections.Counter = collections.Counter()
    detail = []
    for month, mrec in man["months"].items():
        days = collections.defaultdict(set)
        for x in pop["months"][month]["days"]:
            if x["day"].startswith(month):     # drop the boundary day of the month before
                days[x["airport"]].add(x["day"])
        for a, r in sorted(mrec["by_airport"].items()):
            got = len(days.get(a, ()))
            gap = int(r["rows"]) - int(r["obs_has_clock"])
            undelivered = min(round(r["rows"] * (DAYS_IN_MONTH - got) / DAYS_IN_MONTH), gap)
            matcher = gap - undelivered
            key = "lfpg_held_at_the_base" if a == "LFPG" else "matcher_does_not_identify"
            buckets["harvest_has_no_summary_day"] += undelivered
            buckets[key] += matcher
            detail.append({"month": month, "airport": a, "rows": int(r["rows"]),
                           "days_delivered": got, "no_clock": gap,
                           "harvest_has_no_summary_day": undelivered,
                           "matcher_does_not_identify": matcher,
                           "clock_share": r["obs_has_clock"] / r["rows"]})

    out = {
        "what": ("where the ranking file's 154,182 clockless rows are, split into the ones "
                 "the harvest has not delivered and the ones a delivered day carries but "
                 "the matcher does not identify; and what each is worth at the currency"),
        "ordered_by": "the desk, 2026-09-09, order 7",
        "reads": ["analysis/class-manifest-v16.json", "analysis/ranking-population.json",
                  "analysis/searchfit.json"],
        "fits_nothing": True,
        "matcher_sha256": A.MATCHER_SHA,
        "currency": {
            "board_level_s": BOARD_LEVEL_S, "full_transfer": FULL_TRANSFER,
            "s2_per_served_row": per_row,
            "d11": ("the board level is a READING used to convert a fold s2 gain into board "
                    "seconds; nothing is fitted to it and no arm is selected by it"),
            "the_face_is_an_upper_bound": ("a row is unmatched because its evidence is weak, "
                                           "so it cannot be worth what a matched row is"),
        },
        # SUPERSEDED, 2026-09-09: the pro-rata split charges an airport-month's clockless
        # rows to the harvest in proportion to its MISSING day FILES, which counted LTFM's
        # 14 near-empty july files as delivered (analysis/ltfm.json: 32 segments across 16
        # days, none with a clock) and so charged 11,374 rows to a matcher that cannot
        # reach them.  reach-ceiling.json measures the same thing with no threshold in it:
        # clocked departure segments that no ranking row claims.  Kept because RESULTS 17
        # cited it before the ceiling was measured.
        "totals_superseded_by_the_ceiling": {k: face(v) for k, v in buckets.items()},
        "totals_superseded_why": (
            "a day FILE is not a delivered day; see analysis/ltfm.json and "
            "analysis/reach-ceiling.json, which bound the matcher's headroom directly"),
        "totals": ceiling["totals"] if ceiling else None,
        "occupied_unwritten": face(
            int(man["totals"]["occupied_rows_at_the_price_base_the_guard_does_not_hold"])),
        # the rows RESULTS section 17 names one by one, so the table cites this file for
        # every figure it prints rather than recomputing any of them in prose
        "named_in_the_table": {
            "ltfm_july_delivered_but_unmatched": face(
                next(d["matcher_does_not_identify"] for d in detail
                     if d["airport"] == "LTFM" and d["month"] == "2026-07")),
            "ltfm_both_months_if_january_lands": face(
                sum(d["rows"] for d in detail if d["airport"] == "LTFM")),
            "lfpg_with_a_clock_but_held": face(
                sum(d["rows"] - d["no_clock"] for d in detail if d["airport"] == "LFPG")),
        },
        "by_airport_month": sorted(detail, key=lambda d: -d["matcher_does_not_identify"]),
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    print(f"served {served} of {n_rows} ranking rows; {per_row:.0f} s2 per served row\n")
    for k, v in (out["totals"] or {}).items():
        print(f"CEILING {k:20s} {v['rows']:7d} rows -> "
              f"{v['board_s_at_the_level_currency']:+.3f} s level, "
              f"{v['board_s_at_the_two_clean_readings']:+.3f} s at x{FULL_TRANSFER}")
    print("(superseded pro-rata split:)")
    for k, v in out["totals_superseded_by_the_ceiling"].items():
        print(f"  {k:26s} {v['rows']:7d} rows -> "
              f"{v['board_s_at_the_level_currency']:+.3f} s level")
    ou = out["occupied_unwritten"]
    print(f"{'occupied_unwritten':28s} {ou['rows']:7d} rows -> "
          f"{ou['board_s_at_the_level_currency']:+.3f} s level, "
          f"{ou['board_s_at_the_two_clean_readings']:+.3f} s at x{FULL_TRANSFER}")
    top = [d for d in out["by_airport_month"] if d["airport"] != "LFPG"][:6]
    share = sum(d["matcher_does_not_identify"] for d in top) / \
        buckets["matcher_does_not_identify"]
    print(f"\nthe six airport-months carrying {100 * share:.0f} % of the matcher's "
          f"headroom (LFPG is its own bucket):")
    for d in top:
        print(f"  {d['month']} {d['airport']}  {d['days_delivered']:2d}/31 days, "
              f"clock {100 * d['clock_share']:5.1f} %, "
              f"matcher misses {d['matcher_does_not_identify']:6d}")
    for k, v in out["named_in_the_table"].items():
        print(f"{k:36s} {v['rows']:7d} rows -> "
              f"{v['board_s_at_the_level_currency']:+.3f} s level, "
              f"{v['board_s_at_the_two_clean_readings']:+.3f} s at x{FULL_TRANSFER}")
    print("\nwritten", OUT)


if __name__ == "__main__":
    main()
