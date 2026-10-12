"""This lane's REACH against the harvest's corrected RECEIPT, month by month and cell by cell.

The desk, 2026-09-09: the harvest corrected its receipt rate -- a flight number flown daily
was counted as received every day, which inflated a MONTH's rate while leaving a single
day's nearly untouched.  Corrected: 2026-01 62.7 %, 2026-07 56.1 %, 2025-07 49.6 %, against
72.8 / 64.9 / 61.0 before.

Every receipt number this lane quoted came from that file, so every one is re-read here
against this lane's own match share, which is measured on the block and does not change.
The comparison matters because §4 concluded from the OLD numbers that the binding limiter
at EDDF, EGLL and LIRF was the MATCHER -- "the matcher loses 19 to 32 % against 7 to 8 %
unheard".  With the inflated denominator gone that gap is mostly gone with it.

Two readings, both like-for-like: the denominators are the same departures (the harvest's
`departures` per month is this lane's own row count to the row), and the airport tables use
the harvest's own per-airport rate.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/reachvsreceipt.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parent.parent

RECEIPT = ROOT / "experiments/exp-045-harvest/analysis/receipt.json"
TRANSFER = EXP / "analysis/transfer-why.json"
OUT = EXP / "analysis/reach-vs-receipt.json"
#: what the record quoted before the harvest's correction, so the two are side by side
OLD_MONTH = {"2025-01": 0.5413, "2025-07": 0.6095, "2025-12": 0.7558,
             "2026-01": 0.7278, "2026-07": 0.6487}


def main() -> None:
    rec = json.loads(RECEIPT.read_text())
    tw = json.loads(TRANSFER.read_text())
    out = {
        "what": __doc__.strip().splitlines()[0],
        "receipt_file": str(RECEIPT.relative_to(ROOT)),
        "receipt_sha256": hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),
        "receipt_measure": ("`receipt_rate` -- the corrected per-day measure. "
                            "`receipt_rate_anyday` is the superseded one this record "
                            "used to quote and is kept beside it"),
        "reach_measure": ("this lane's own `match_share` from analysis/transfer-why.json, "
                          "measured on the block; it does not change"),
        "by_month": {}, "by_airport_month": {},
    }
    for m in sorted(rec):
        if m not in ("2025-01", "2025-07", "2025-12", "2026-01", "2026-07"):
            continue
        r = rec[m]
        cm = tw.get("coverage_by_airport_month", {})
        reach = None
        # the month's match share is the row-weighted mean of the airport cells
        num = den = 0.0
        for k, v in tw.get("by_airport_month", {}).items():
            a, mm = k.rsplit(" ", 1)
            if mm != m or "match_share" not in v:
                continue
            rows = v.get("rows") or v.get("matched", 0) / max(v["match_share"], 1e-9)
            num += v["match_share"] * rows
            den += rows
        if den:
            reach = num / den
        out["by_month"][m] = {
            "departures": r["departures"],
            "receipt_corrected": r["receipt_rate"],
            "receipt_superseded_anyday": r["receipt_rate_anyday"],
            "receipt_this_record_quoted": OLD_MONTH.get(m),
            "reach_match_share": (round(reach, 4) if reach is not None else None),
            "reach_minus_corrected_receipt": (round(reach - r["receipt_rate"], 4)
                                              if reach is not None else None),
            "reach_minus_superseded_receipt": (
                round(reach - r["receipt_rate_anyday"], 4)
                if reach is not None else None),
            "share_of_what_the_network_heard_that_the_matcher_reaches": (
                round(reach / r["receipt_rate"], 4)
                if reach is not None and r["receipt_rate"] else None),
        }
    for k, v in sorted(tw.get("by_airport_month", {}).items()):
        a, m = k.rsplit(" ", 1)
        ba = rec.get(m, {}).get("by_airport", {}).get(a)
        if not ba or "match_share" not in v:
            continue
        out["by_airport_month"][k] = {
            "reach_match_share": round(v["match_share"], 4),
            "receipt_corrected": ba["rate"],
            "receipt_superseded_anyday": ba["rate_anyday"],
            "reach_minus_corrected": round(v["match_share"] - ba["rate"], 4),
            "reach_minus_superseded": round(v["match_share"] - ba["rate_anyday"], 4),
        }
    gaps = [x["reach_minus_corrected"] for x in out["by_airport_month"].values()]
    olds = [x["reach_minus_superseded"] for x in out["by_airport_month"].values()]
    out["reading"] = {
        "cells": len(gaps),
        "median_reach_minus_corrected_receipt": round(sorted(gaps)[len(gaps) // 2], 4),
        "median_reach_minus_superseded_receipt": round(sorted(olds)[len(olds) // 2], 4),
        "cells_where_the_matcher_loses_more_than_5_points": int(
            sum(g < -0.05 for g in gaps)),
        "cells_where_the_SUPERSEDED_number_said_so": int(sum(g < -0.05 for g in olds)),
        "note": ("a negative gap is the matcher losing rows the network heard. Under the "
                 "superseded receipt that gap was the headline of §4; under the corrected "
                 "one it is small, and the limiter is reception nearly everywhere"),
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("%-9s %10s %10s %10s %9s %9s" % ("month", "reach", "receipt", "superseded",
                                           "gap", "gap_old"))
    for m, v in out["by_month"].items():
        print("%-9s %10s %10s %10s %9s %9s" % (
            m, v["reach_match_share"], v["receipt_corrected"],
            v["receipt_superseded_anyday"], v["reach_minus_corrected_receipt"],
            v["reach_minus_superseded_receipt"]))
    print()
    print(json.dumps(out["reading"], indent=1))
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
