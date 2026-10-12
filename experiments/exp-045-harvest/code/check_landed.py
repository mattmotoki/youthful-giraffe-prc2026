"""exp-045-harvest: do the summaries on disk hold what the manifest says they hold?

The manifest's `summaries_rule` for a day comes from a record written when the day was
re-derived.  The parquet files come from whatever wrote them last.  On 2025-12 those
were two different things: the re-derive finished at 13:54:07 UTC and the fetch pull
overwrote all 279 airport-days 41 seconds later, while the manifest went on reporting
the re-derived rule.  A wrong manifest is worse than a mixed one, because nothing
downstream has any reason to doubt it.

So this compares, per day, the rows actually on disk against the row count the rule
record claims, and fails loudly on any difference.

    .venv/bin/python experiments/exp-045-harvest/code/check_landed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ART = ROOT / "artifacts/exp-045"


def main() -> int:
    recs = sorted((EXP / "analysis/rederived").glob("2*.json"))
    bad, checked = [], 0
    for rj in recs:
        rec = json.loads(rj.read_text())
        day = rec["day"]
        files = sorted(ART.glob(f"summaries/*/{day}.parquet"))
        if not files:
            bad.append((day, "no summaries on disk", rec["outputs"]["segments"]["rows"], 0))
            continue
        n = sum(len(pd.read_parquet(f)) for f in files)
        want = rec["outputs"]["segments"]["rows"]
        checked += 1
        if n != want:
            bad.append((day, rec["rule_version"], want, n))
    for day, ver, want, got in bad:
        print(f"{day}: record says {want} rows under {ver}, disk holds {got}")
    print(f"checked {checked} re-derived days against their records; "
          f"{len(bad)} disagree")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
