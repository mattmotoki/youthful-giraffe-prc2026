"""exp-045-harvest: prove that `all_segments()` leaves the pilot's frozen rule alone.

Runs `exp-043-pilot/code/segments.py` and `code/allsegments.py` over the pilot's own
stored EHAM slices and asserts that every departure row agrees on every number the
pilot's rule produces.  If this fails, the harvest is not using the frozen rule.

Both files order their points stably, so every row is asserted on.  There was an
exemption here between 12:00 and 14:00 UTC on 2026-09-08: the harvest had made
`all_segments()` stable while the pilot's `segments()` still used pandas' default,
so the aircraft whose points share a timestamp were counted apart instead of
compared.  exp-043's 09998b5 made the pilot stable too and the exemption went with
it; `analysis/rule_check.json` still counts the tied points, because their number is
the size of what the two sorts used to disagree about.

    .venv/bin/python experiments/exp-045-harvest/code/check_rule.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-043-pilot/code"))
sys.path.insert(0, str(EXP / "code"))

from allsegments import all_segments          # noqa: E402
from segments import segments                 # noqa: E402

KEYS = ["icao", "obs_wheels_off"]
SAME = ["obs_offblock_strict", "obs_offblock_loose", "obs_offblock_disp",
        "stand_stop_s", "turnaround_anchored", "t_airborne", "seg_start",
        "seg_points", "seg_span_s", "seg_gap_max_s", "seg_dt_median_s",
        "n_ground_points", "callsign_modal", "callsign_n"]


def main(days: list[str]) -> None:
    art = ROOT / "artifacts/exp-043"
    rep = {"days": days, "checked": []}
    for d in days:
        pts = pd.read_parquet(art / f"points_{d}.parquet")
        old = segments(pts)
        new = all_segments(pts.assign(airport="EHAM"))
        dep = new[new.kind == "departure"].copy()
        # counted, not exempted: how much of the day the two sorts used to differ on
        tied = set(pts.loc[pts.duplicated(subset=["icao", "t"], keep=False), "icao"])
        old_all, new_all = old, new
        assert len(old) == len(dep), f"{d}: {len(old)} pilot rows vs {len(dep)} departure rows"
        o = old.sort_values(KEYS).reset_index(drop=True)
        n = dep.sort_values(KEYS).reset_index(drop=True)
        bad = {}
        for c in SAME:
            a, b = o[c], n[c]
            if a.dtype.kind == "f":
                same = np.isclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=0, atol=0)
            else:
                same = (a.fillna("~") == b.fillna("~")).to_numpy()
            if not same.all():
                bad[c] = int((~same).sum())
        assert not bad, f"{d}: columns differ {bad}"
        rep["checked"].append({
            "day": d, "pilot_departure_rows": int(len(old_all)),
            "rows_asserted_identical": int(len(old)),
            "rows_exempted": 0,
            "harvest_rows": int(len(new_all)),
            "by_kind": new.kind.value_counts().to_dict(),
            "columns_identical": SAME,
            "aircraft_with_a_tied_timestamp": len(tied),
            "aircraft_exempted": 0,
            "tied_points": int(pts.duplicated(subset=["icao", "t"], keep=False).sum()),
            "points": int(len(pts)),
        })
        print(f"{d}: all {len(old)} of the pilot's departure rows identical on "
              f"{len(SAME)} columns; harvest adds {len(new_all) - len(old_all)} rows "
              f"({new.kind.value_counts().to_dict()}); "
              f"{len(tied)} aircraft carry a tied timestamp and both sorts now agree "
              f"on them")
    (EXP / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(EXP / "analysis/rule_check.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main(sys.argv[1:] or ["2025-07-07", "2025-07-08"])
