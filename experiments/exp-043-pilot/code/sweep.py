"""exp-043-pilot: why the observation wins, and how far the rule can be pushed.

The headline numbers use the assignment's rule (ground speed above 3 kt held for
30 s).  Its median offset is about -190 s: the airport's recorded off-block is
three minutes BEFORE the observed motion onset.  Pushback runs at 2-5 kt, so a 3 kt
gate can only fire once the tug is done and the aircraft taxis on its own.  This
sweep asks whether a lower gate, or a displacement gate that does not depend on the
broadcast ground speed at all, measures the same event earlier and tighter.

Nothing here replaces the headline: it is reported as a named refinement with its
own numbers, so the assignment's rule and the refinement are both readable.

  .venv/bin/python experiments/exp-043-pilot/code/sweep.py 2025-07-07 2025-07-08
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


def run(pts, coh, move_kt: float, hold_s: float, stand_s: float) -> dict:
    S.MOVE_KT, S.HOLD_S, S.STAND_STOP = move_kt, hold_s, stand_s
    seg = S.segments(pts)
    seg["callsign_modal"] = seg.callsign_modal.str.strip()
    j = coh.merge(seg, left_on="callsign", right_on="callsign_modal", how="left")
    j["dt_takeoff"] = j.obs_wheels_off - j.mvt_s
    j = j[j.dt_takeoff.abs() <= M.WINDOW]
    j = j.sort_values("dt_takeoff", key=abs, kind="stable")
    j = j.drop_duplicates("MVT_ID_mvt").drop_duplicates(["icao", "obs_wheels_off"])
    out = {"move_kt": move_kt, "hold_s": hold_s, "stand_stop_s": stand_s,
           "matched": int(len(j))}
    for name in ("obs_offblock_strict", "obs_offblock_loose", "obs_offblock_disp"):
        s = j[j[name].notna()]
        if len(s) < 20:
            continue
        e = s.blk_s - s[name]
        r, med, p1, p99 = M.rmse_centered(e)
        ea = (s.blk_s - s.aobt_s).dropna()
        ra = M.rmse_centered(ea)[0]
        out[name] = {"rows": int(len(s)), "share": round(len(s) / len(coh), 4),
                     "median_offset_s": round(med, 1), "rmse_s": round(r, 1),
                     "p01_s": round(p1, 1), "p99_s": round(p99, 1),
                     "aobt_rmse_same_rows_s": round(ra, 1)}
    return out


def main(days: list[str]) -> None:
    pts, _ = C.load_slices(days)
    coh = C.with_callsign(pd.read_parquet(EXP / "work/cohort.parquet"))
    coh = coh[coh.day.dt.strftime("%Y-%m-%d").isin(days)].copy()
    for col, name in (("MVT_TIME_UTC_mvt", "mvt_s"), ("BLOCK_TIME_UTC_mvt", "blk_s"),
                      ("AOBT_3_flt", "aobt_s")):
        coh[name] = (coh[col] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()

    rows = []
    for kt in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0):
        rows.append(run(pts, coh, kt, 30.0, 60.0))
        r = rows[-1].get("obs_offblock_loose", {})
        print(f"move>{kt:>4} kt hold 30 s: loose n={r.get('rows')} "
              f"median_off={r.get('median_offset_s')} RMSE={r.get('rmse_s')} "
              f"(AOBT {r.get('aobt_rmse_same_rows_s')})")
    for kt in (3.0, 6.0, 7.0, 8.0, 9.0):
        for hold in (10.0, 20.0, 30.0, 60.0, 90.0):
            rows.append(run(pts, coh, kt, hold, 60.0))
            r = rows[-1].get("obs_offblock_loose", {})
            print(f"move>{kt:.0f} kt hold {hold:>4.0f} s: loose n={r.get('rows')} "
                  f"median_off={r.get('median_offset_s')} RMSE={r.get('rmse_s')}")
    for dm in (20.0, 35.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 500.0, 800.0, 1200.0):
        S.DISP_M = dm
        rows.append(run(pts, coh, 3.0, 30.0, 60.0) | {"disp_m": dm})
        r = rows[-1].get("obs_offblock_disp", {})
        print(f"displacement > {dm:>5} m: n={r.get('rows')} "
              f"median_off={r.get('median_offset_s')} RMSE={r.get('rmse_s')} "
              f"(AOBT {r.get('aobt_rmse_same_rows_s')})")
    S.DISP_M = 250.0
    for st in (0.0, 30.0, 120.0, 300.0, 600.0):   # 600 s is the assignment's own margin
        rows.append(run(pts, coh, 3.0, 30.0, st))
        r = rows[-1].get("obs_offblock_strict", {})
        print(f"stand margin >= {st:>5} s: strict n={r.get('rows')} "
              f"share={r.get('share')} RMSE={r.get('rmse_s')}")
    S.MOVE_KT, S.HOLD_S, S.STAND_STOP, S.DISP_M = 3.0, 30.0, 60.0, 250.0

    # How long the network sees a departing aircraft standing before it moves.  This
    # is the fact that settled the anchor rule, and the report quotes it, so it is
    # measured here rather than remembered: `stand_stop_s` is the stationary time
    # observed immediately before the motion onset, at the frozen rule.
    seg = S.segments(pts)
    seg["callsign_modal"] = seg.callsign_modal.str.strip()
    m = coh.merge(seg, left_on="callsign", right_on="callsign_modal", how="left")
    m["dt_takeoff"] = m.obs_wheels_off - m.mvt_s
    m = m[m.dt_takeoff.abs() <= M.WINDOW].sort_values("dt_takeoff", key=abs, kind="stable")
    m = m.drop_duplicates("MVT_ID_mvt").drop_duplicates(["icao", "obs_wheels_off"])
    def _med(d, col):
        v = d[col].dropna()
        return {"rows": int(len(v)), "median_s": round(float(v.median()), 1)}

    for d in (seg, m):
        d["seen_before_motion_s"] = d.obs_offblock_disp - d.seg_start
        d["seen_before_motion_loose_s"] = d.obs_offblock_loose - d.seg_start
    stand = {
        "seen_before_motion": {
            "what": ("how long the network sees the aircraft before the 250 m stand "
                     "departure: obs_offblock_disp - seg_start.  This is the number the "
                     "report quotes for 'transponders come on near pushback'."),
            "all_segments": _med(seg, "seen_before_motion_s"),
            "matched_rows": _med(m, "seen_before_motion_s")},
        "seen_before_motion_3kt": {
            "what": "the same against the 3 kt / 30 s onset: obs_offblock_loose - seg_start",
            "all_segments": _med(seg, "seen_before_motion_loose_s"),
            "matched_rows": _med(m, "seen_before_motion_loose_s")},
        "stand_stop_margin": {
            "what": ("stand_stop_s: the STATIONARY RUN observed immediately before the "
                     "onset, which a brief GPS blip cuts short, so it is smaller than "
                     "the time the aircraft is seen at all"),
            "all_segments": _med(seg, "stand_stop_s"),
            "matched_rows": _med(m, "stand_stop_s")},
        "days": days}
    print("seen before motion (250 m): all segments "
          f"{stand['seen_before_motion']['all_segments']['median_s']} s, matched "
          f"{stand['seen_before_motion']['matched_rows']['median_s']} s; "
          f"stand-stop margin matched {stand['stand_stop_margin']['matched_rows']['median_s']} s")
    (EXP / "analysis/sweep.json").write_text(json.dumps(
        {"days": days, "arms": rows, "stand_stop": stand},
        indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main(sys.argv[1:] or ["2025-07-07", "2025-07-08"])
