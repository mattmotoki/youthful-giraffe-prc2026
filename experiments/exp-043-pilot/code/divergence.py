"""exp-043-pilot: price the current segment rule against the one the harvest shipped.

`exp-045-harvest` first ran `segments.py` as it stood on 2026-09-07, which reads any
first non-ground point after a ground run as a take-off and orders points with an
unstable sort.  Two repairs have been made since, and this file prices each one and
both together over the same points:

  `alt`     Sol round 1: the first non-ground point must also carry a real altitude,
            because readsb writes `null` for a position with no altitude at all and
            a null is not evidence of flight.  The harvest adopted this at 8906ea1.
  `stable`  Sol round 4: points that share an (icao, t) exist at the UTC-midnight
            overlap and a few disagree about `on_ground`, so an unstable sort can
            split a ground run in two places.  The harvest measured 14 and 17
            segments moving between machines on its two stored days and adopted a
            stable sort at afc77ee; the pilot ran unstable until round 4 found it.

The `stable` column costs nothing, and `analysis/ordering.json` says why: the pilot
de-duplicates `(icao, t)` before `segments()` runs, so no tie is left for a sort to
break and the segment table is identical under a stable sort, an unstable sort and a
shuffled input.  The repair is a guard for an input that DOES carry ties -- a raw day
file, which is what the harvest reads, and where it measured 14 and 17 segments
moving between two machines.

  .venv/bin/python experiments/exp-043-pilot/code/divergence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "code"))
import census as C          # noqa: E402
import match as M           # noqa: E402
import segments as S        # noqa: E402

WEEK = [f"2025-07-{d:02d}" for d in range(7, 14)]
HELD = WEEK[2:]


def numbers(m: pd.DataFrame) -> dict:
    s = m[m.obs_offblock_disp.notna() & m.aobt_s.notna() & m.has_nm]
    e, ea = s.blk_s - s.obs_offblock_disp, s.blk_s - s.aobt_s
    return {"matched": int(m.obs_wheels_off.notna().sum()),
            "rows_scored": int(len(s)),
            "rmse_obs_s": round(M.rmse_centered(e)[0], 2),
            "rmse_aobt_s": round(M.rmse_centered(ea)[0], 2),
            "paired_difference_s": round(M.rmse_centered(e)[0] - M.rmse_centered(ea)[0], 2)}


def main() -> None:
    out = {"why": ("prices the two repairs made since exp-045-harvest's first kernels ran "
                   "segments.py as it stood on 2026-09-07.  The harvest has adopted both "
                   "(8906ea1, afc77ee) and re-derives earlier months from stored points."),
           "rule": {
               "shipped": ("shipped-2026-09-07: any first non-ground point after a ground "
                           "run is a take-off, points ordered with an unstable sort"),
               "alt": ("repair-2026-09-08: that point must also carry a real altitude, "
                       "not readsb's null"),
               "stable": ("repair-2026-09-08+stable-order: the same, and the points are "
                          "ordered stably, so the same points give one answer on every "
                          "machine"),
               "current": "what this experiment and the harvest both run now: alt + stable"},
           "stable_column_caveat": (
               "measured on this laptop, where pandas' quicksort is deterministic for a "
               "given input, so a small number here is not the size of the defect.  The "
               "defect is that the unstable answer is a property of the machine: the "
               "harvest measured 14 and 17 segments moving between this laptop and the "
               "Kaggle image on the same points (exp-045-harvest/analysis/machine_check.json)")}
    # (name, REQUIRE_ALT, STABLE); `repair` is kept as a name so the earlier record
    # and this one line up, and `current` is what every other number in this
    # experiment is measured under.
    ARMS = (("shipped", False, False), ("repair", True, False),
            ("stable_only", False, True), ("current", True, True))

    pts, _ = C.load_slices(WEEK)
    segs = {}
    for name, alt, stable in ARMS:
        S.REQUIRE_ALT, S.STABLE = alt, stable
        g = S.segments(pts)
        g["key"] = S.seg_key(g)
        segs[name] = g
    S.REQUIRE_ALT, S.STABLE = True, True

    a, b = segs["shipped"], segs["current"]
    only_a, only_b = set(a.key) - set(b.key), set(b.key) - set(a.key)
    out["segments"] = {name: int(len(g)) for name, g in segs.items()}
    out["segments"] |= {"only_in_shipped": len(only_a), "only_in_current": len(only_b),
                        "shared": int(len(set(a.key) & set(b.key)))}
    out["segments_by_arm_against_shipped"] = {
        name: {"only_in_shipped": len(set(a.key) - set(g.key)),
               "only_in_this_arm": len(set(g.key) - set(a.key))}
        for name, g in segs.items() if name != "shipped"}
    d = a[a.key.isin(only_a)][["icao", "callsign_modal", "obs_wheels_off", "t_airborne",
                               "seg_points", "n_ground_points"]].copy()
    d["takeoff_utc"] = pd.to_datetime(d.obs_wheels_off, unit="s", utc=True).astype(str)
    out["segments_dropped_by_the_repair"] = d.drop(columns="obs_wheels_off").to_dict("records")

    for tag, days in (("week", WEEK), ("heldout", HELD)):
        arm = {}
        for name, alt, stable in ARMS:
            S.REQUIRE_ALT, S.STABLE = alt, stable
            arm[name] = numbers(M.build(days))
        S.REQUIRE_ALT, S.STABLE = True, True
        arm["match_rows_changed"] = abs(arm["shipped"]["matched"] - arm["current"]["matched"])
        arm["rmse_moved_by_s"] = round(
            arm["current"]["rmse_obs_s"] - arm["shipped"]["rmse_obs_s"], 2)
        out[tag] = arm

    (EXP / "analysis/divergence.json").write_text(json.dumps(out, indent=2, sort_keys=True,
                                                             default=str))
    print(json.dumps({k: v for k, v in out.items()
                      if k != "segments_dropped_by_the_repair"}, indent=2, default=str))


if __name__ == "__main__":
    main()
