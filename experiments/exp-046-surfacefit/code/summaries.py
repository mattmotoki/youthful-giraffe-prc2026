"""exp-046-surfacefit: re-express the pilot's stored EHAM week as harvest summaries.

The feature block (`code/block.py`) reads exactly one schema — `exp-045-harvest`'s
per-airport, per-day segment summary — so that the pipeline unit-tested here on the
pilot's week is the pipeline that will serve on the harvest's months.  The pilot
stored POINTS (`artifacts/exp-043/points_<day>.parquet`), not summaries, so this
script runs the harvest's own `all_segments()` over them and writes the result in the
harvest's layout under `artifacts/exp-046/summaries/EHAM/<day>.parquet`.

Nothing is re-derived: `all_segments()` is the harvest's file, which
`exp-045/code/check_rule.py` already proves reproduces the pilot's frozen rule row by
row on these same days.  This script only changes where the numbers live.

    .venv/bin/python experiments/exp-046-surfacefit/code/summaries.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
from imports import load as _load             # noqa: E402

all_segments = _load("exp-045-harvest", "allsegments", ("exp-043-pilot",)).all_segments

SRC = ROOT / "artifacts/exp-043"
OUT = ROOT / "artifacts/exp-046/summaries"
DAYS = [f"2025-07-{d:02d}" for d in range(7, 14)]
AIRPORT = "EHAM"


def rule_sha256() -> str:
    """The identity of the rule these summaries were written by.

    The pilot's reviewer repaired the rule on 2026-09-08 and the harvest now records a
    `summaries_rule_sha256` per day so that a fold table cannot mix two rules
    (`exp-045-harvest/manifest.json`).  These local summaries need the same handle, and
    it is the hash of the two files that ARE the rule.
    """
    h = hashlib.sha256()
    for rel in ("experiments/exp-043-pilot/code/segments.py",
                "experiments/exp-045-harvest/code/allsegments.py"):
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()


def main(days: list[str]) -> None:
    rep = {"source": "artifacts/exp-043 points (pilot), via exp-045 all_segments()",
           "airport": AIRPORT, "rule_sha256": rule_sha256(), "days": {}}
    (OUT / AIRPORT).mkdir(parents=True, exist_ok=True)
    for d in days:
        pts = pd.read_parquet(SRC / f"points_{d}.parquet")
        seg = all_segments(pts.assign(airport=AIRPORT))
        # the harvest writes one file per airport and drops the constant column
        seg = seg.drop(columns=["airport"])
        p = OUT / AIRPORT / f"{d}.parquet"
        seg.to_parquet(p, index=False)
        rep["days"][d] = {"rows": int(len(seg)),
                          "by_kind": seg.kind.value_counts().to_dict(),
                          "bytes": p.stat().st_size}
        print(f"{d}: {len(seg)} segments {seg.kind.value_counts().to_dict()}")
    (EXP / "analysis/summaries-pilotweek.json").write_text(
        json.dumps(rep, indent=2, sort_keys=True))
    # the sidecar `block.rule_by_day()` reads, so these days carry a rule version too
    (OUT / "rule.json").write_text(json.dumps(
        {"rule_sha256": rep["rule_sha256"], "days": days,
         "note": "sha256 of exp-043-pilot/code/segments.py + exp-045-harvest/code/"
                 "allsegments.py as they were when these summaries were written"},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:] or DAYS)
