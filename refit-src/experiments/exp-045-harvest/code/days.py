"""exp-045-harvest: the fourteen months, in the order the assignment sets, and the
days of one of them as a json list for `push_kernel.sh`.

    python3 code/days.py            -> the month order, one per line
    python3 code/days.py 2026-01    -> ["2026-01-01", ...]
    python3 code/days.py 2026-01 --todo   -> the days of that month with no run json yet
"""
from __future__ import annotations

import calendar
import json
import os
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

# The desk reordered this on 2026-09-08 so the modelling worker `surfacefit` can start
# on a labelled month while the ranking months are still being harvested: 2025-07 in
# full first, then the two ranking months, then the other two fold months, then the rest.
ORDER = ["2025-07", "2026-07", "2026-01", "2025-12", "2025-01",
         "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
         "2025-08", "2025-09", "2025-10", "2025-11"]

# exp-186 (2026-10-08): $PRC_RANKING_MODE=final adds the two new months of the final file
# (prc/data.py RANKING_SPECS["final"]) after the fourteen; the ranking mode keeps ORDER as it
# was. This file runs under the system python3 (no pandas), so it reads the mode itself.
FINAL_NEW_MONTHS = ["2026-02", "2026-06"]
RANKING_MODE = os.environ.get("PRC_RANKING_MODE") or "ranking"
if RANKING_MODE not in ("ranking", "final"):
    raise SystemExit(f"PRC_RANKING_MODE={RANKING_MODE!r}: ranking or final")
if RANKING_MODE == "final":
    ORDER = ORDER + FINAL_NEW_MONTHS


def days(month: str) -> list[str]:
    y, m = int(month[:4]), int(month[5:7])
    return [f"{month}-{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]


def todo(month: str) -> list[str]:
    have = {p.name[4:14] for p in (EXP / "analysis/runs").glob("run_*.json")}
    return [d for d in days(month) if d not in have]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        for m in ORDER:
            print(m, len(days(m)))
    else:
        fn = todo if "--todo" in sys.argv else days
        print(json.dumps(fn(sys.argv[1])))
