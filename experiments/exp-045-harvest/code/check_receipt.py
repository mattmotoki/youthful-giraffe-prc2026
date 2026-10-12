"""exp-045-harvest: the receipt rate cannot exceed what the box observed.

A departure counted as received must have been seen as a departure segment in its
airport's box that month, so for every (airport, month)

    received <= departure segments written

is an identity of the data, not a threshold.  It is the check that caught the receipt
bug of 2026-09-08: `census.rate()` aggregated a callsign's ground appearances over the
whole month before applying the take-off window, so a flight number flown daily counted
as received on every day of the month.  The old answer broke this identity in 18 of 80
airport-months -- EGLL in March 2025 was priced at 16,023 received against 13,752
departure segments observed.  The repaired answer breaks it in none.

    .venv/bin/python experiments/exp-045-harvest/code/check_receipt.py [YYYY-MM ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ART = EXP.parents[1] / "artifacts/exp-045"


def segments(airport: str, days: list[str]) -> int:
    """Departure segments written for `airport` over `days`."""
    n = 0
    for d in days:
        p = ART / "summaries" / airport / f"{d}.parquet"
        if p.exists():
            n += int((pd.read_parquet(p, columns=["kind"])["kind"] == "departure").sum())
    return n


def main(months: list[str]) -> None:
    receipt = json.loads((EXP / "analysis/receipt.json").read_text())
    bad, checked = [], 0
    for month in sorted(months or receipt):
        if month not in receipt:
            continue
        days = receipt[month]["days_measured"]
        for airport, v in receipt[month]["by_airport"].items():
            seen = segments(airport, days)
            if not seen:
                continue
            checked += 1
            if int(v["received"]) > seen:
                bad.append((month, airport, int(v["received"]), seen))
    for month, airport, got, seen in bad:
        print(f"{month} {airport}: {got} received against {seen} departure segments observed")
    print(f"checked {checked} airport-months; {len(bad)} price more received than observed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if a[:2] == "20"])
