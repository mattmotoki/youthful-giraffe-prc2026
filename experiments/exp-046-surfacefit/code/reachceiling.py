"""Order 2, step one: the CEILING on reach, before any matcher is touched.

Desk pick, 2026-09-09:

    then (2) reach, ceiling first -- at EDDF 2026-07 count the ground segments carrying a
    departure clock and no identified MVT_ID, bounding the 13,342 from above; report the
    ceiling per airport-month for the six that carry 79 % before building anything.

**What bounds what.**  A ranking row can only join the served class if some ground segment
carries a departure clock for it.  So for one airport-month the extra rows any matcher
could ever reach is at most

    min(ranking rows with no clock today, clocked departure segments no MVT_ID claims)

and the second term is the one this file measures.  It is a real ceiling and not an
estimate: a segment already claimed by a departure cannot be claimed again, and a segment
with no clock carries nothing this lane serves.  Nothing here says the ceiling is
ATTAINABLE -- an unclaimed segment may belong to a departure outside the ranking file, to
one of the 388 `noac` rows, or to no departure at all.  It says what the ceiling is.

**It also replaces the pro-rata split of `nextsteps.py`.**  That file charged an
airport-month's clockless rows to the harvest in proportion to its MISSING day files, which
counted LTFM's 14 near-empty july files as delivered and so charged 11,374 rows to the
matcher that no matcher can reach (`analysis/ltfm.json`: 32 segments across 16 days, zero
with a clock).  Segments do not need a threshold, so this measure has no arbitrary cut in
it at all.

No fit, no fold, no label -- one `block.build` per month for the match side, which is the
matcher's own answer rather than a re-implementation of it.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/reachceiling.py
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

OUT = EXP / "analysis/reach-ceiling.json"
MONTHS = ["2026-01", "2026-07"]
#: v16's board level and the section 16.3 transfer, both READINGS -- score-feedback rule: nothing is fitted
BOARD_LEVEL_S, FULL_TRANSFER = 246.83, 1.492


def main() -> None:
    man = json.loads((EXP / "analysis/class-manifest-v16.json").read_text())
    sf = json.loads((EXP / "analysis/searchfit.json").read_text())
    n_rows = int(man["totals"]["rows"])
    per_row = (float(sf["what_is_left"]["per_step"]["v13"]["s2_gain_of_the_served_arm"])
               * n_rows / int(man["totals"]["served_by_exp_046"]))

    def face(n: int) -> dict:
        g = per_row * n / n_rows
        lvl = BOARD_LEVEL_S - (BOARD_LEVEL_S ** 2 - g) ** 0.5
        return {"rows": int(n), "board_s_at_the_level_currency": lvl,
                "board_s_at_the_two_clean_readings": lvl * FULL_TRANSFER}

    rank = D.load_ranking()
    rank = rank[rank.PHASE_mvt == "DEP"].copy()
    rank["month"] = rank.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")

    rows = []
    for month in MONTHS:
        dep = rank[rank.month.eq(month)].copy()
        tr = pd.concat([pd.read_parquet(D.month_file(m),
                                        columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                        for m in D.MONTHS], ignore_index=True)
        tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
        rule, _ = B.one_rule([(sorted(set(dep.ADEP_mvt.dropna()) & set(B.AIRPORTS)),
                               sorted(dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").unique()))])
        blk = B.build(dep, tr, require_rule=rule)
        m = dep[["MVT_ID_mvt", "ADEP_mvt"]].merge(blk, on="MVT_ID_mvt", how="left",
                                                  validate="one_to_one")
        m["obs_matched"] = m.obs_matched.astype("boolean").fillna(False).astype(bool)
        m["has_clock"] = m.obs_has_clock.astype("boolean").fillna(False).astype(bool)

        # the segment side, read through the matcher's own loader and rule
        days = sorted(dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").unique())
        for a, g in m.groupby("ADEP_mvt"):
            seg = B.load_summaries([a], days, require_rule=rule)
            if len(seg):
                sd = seg[seg.kind.eq("departure") & seg.obs_offblock.notna()]
                # a segment belongs to this month when its own day is in the month
                day = pd.to_datetime(sd.seg_start, unit="s", utc=True).dt.strftime("%Y-%m")
                clocked = int((day == month).sum())
            else:
                clocked = 0
            no_clock = int((~g.has_clock).sum())
            claimed = int(g.has_clock.sum())
            unclaimed = max(clocked - claimed, 0)
            rows.append({
                "month": month, "airport": a,
                "ranking_rows": int(len(g)),
                "rows_with_a_clock_today": claimed,
                "rows_with_no_clock_today": no_clock,
                "clocked_departure_segments": clocked,
                "segments_no_row_claims": unclaimed,
                "ceiling": int(min(no_clock, unclaimed)),
                "ceiling_share_of_the_gap": (round(min(no_clock, unclaimed) / no_clock, 4)
                                             if no_clock else None)})
            r = rows[-1]
            print(f"  {month} {a}  gap {no_clock:6d}  clocked segments {clocked:6d}  "
                  f"unclaimed {unclaimed:6d}  -> CEILING {r['ceiling']:6d}"
                  f"  ({100 * (r['ceiling_share_of_the_gap'] or 0):5.1f} % of the gap)",
                  flush=True)

    tot = int(sum(r["ceiling"] for r in rows))
    six = sorted([r for r in rows if r["airport"] != "LFPG"],
                 key=lambda r: -r["rows_with_no_clock_today"])[:6]
    out = {"what": ("the ceiling on reach: clocked departure segments that no ranking row "
                    "claims, which bounds from above every row any matcher could add"),
           "ordered_by": "the desk, 2026-09-09, pick (2), ceiling first",
           "not_attainable_only_a_bound": ("an unclaimed segment may belong to a departure "
                                           "outside the ranking file, to one of the 388 "
                                           "noac rows, or to no departure at all"),
           "matcher_sha256": A.MATCHER_SHA, "fits_nothing": True,
           "currency": {"board_level_s": BOARD_LEVEL_S, "full_transfer": FULL_TRANSFER,
                        "d11": "board level is a reading; nothing is fitted to it"},
           "by_airport_month": sorted(rows, key=lambda r: -r["ceiling"]),
           "totals": {"ceiling_all": face(tot),
                      "ceiling_without_lfpg":
                          face(sum(r["ceiling"] for r in rows if r["airport"] != "LFPG")),
                      "the_six_that_carried_79_percent":
                          face(sum(r["ceiling"] for r in six))}}
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\ntotals:")
    for k, v in out["totals"].items():
        print(f"  {k:32s} {v['rows']:7d} rows -> "
              f"{v['board_s_at_the_level_currency']:+.3f} s level, "
              f"{v['board_s_at_the_two_clean_readings']:+.3f} s at x{FULL_TRANSFER}")
    print("written", OUT)


if __name__ == "__main__":
    main()
