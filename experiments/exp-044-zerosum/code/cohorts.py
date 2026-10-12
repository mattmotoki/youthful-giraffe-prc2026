"""Cohort sizes the report quotes: eligible shares and fitting-set rows per design.

Sol round 1, non-blocking: the checker validated only five headline values, so
prose numbers could drift.  `tests/check.py` now audits every number in the
report against the record, and these are the numbers it needed that no other
script wrote.  Panel only, no features, no fit.

    python cohorts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zs                                   # noqa: E402
import learner as L                         # noqa: E402
import controls as C                        # noqa: E402

EXP = zs.EXP
FOLDS = zs.FOLDS


def main() -> None:
    pan = zs.V.load()
    S = {f: L.fold_state(f, pan[f]) for f in FOLDS}
    out = {"note": "eligible shares and fitting-set sizes per cross-fitting design; "
                   "the numbers RESULTS.md quotes for the size argument of §4.",
           "folds": {}}
    n_elig = {}
    for f in FOLDS:
        s = S[f]
        ei = np.flatnonzero(s["elig"])
        n_elig[f] = len(ei)
        blk = s["blk"][ei]
        day = s["days"][ei]
        half = C.half_index(day) == 0          # controls.py's own split rule
        out["folds"][f] = {
            "n_rows": int(len(s["elig"])),
            "n_eligible": int(len(ei)),
            "eligible_pct": round(100 * len(ei) / len(s["elig"]), 2),
            "n_scored_cohort": int(s["m0"].sum()),
            "dayblock_fit_rows": {"min": int(min((blk != b).sum() for b in range(zs.BLOCKS))),
                                  "max": int(max((blk != b).sum() for b in range(zs.BLOCKS)))},
            "halfmonth_fit_rows": {"min": int(min(half.sum(), (~half).sum())),
                                   "max": int(max(half.sum(), (~half).sum()))}}
    for f in FOLDS:
        out["folds"][f]["xmonth_fit_rows"] = int(sum(n_elig[g] for g in FOLDS if g != f))
    out["dayblock_fit_rows_range"] = [min(out["folds"][f]["dayblock_fit_rows"]["min"]
                                          for f in FOLDS),
                                      max(out["folds"][f]["dayblock_fit_rows"]["max"]
                                          for f in FOLDS)]
    out["halfmonth_fit_rows_range"] = [min(out["folds"][f]["halfmonth_fit_rows"]["min"]
                                           for f in FOLDS),
                                       max(out["folds"][f]["halfmonth_fit_rows"]["max"]
                                           for f in FOLDS)]
    out["xmonth_fit_rows_range"] = [min(out["folds"][f]["xmonth_fit_rows"] for f in FOLDS),
                                    max(out["folds"][f]["xmonth_fit_rows"] for f in FOLDS)]
    den = json.loads((EXP / "analysis" / "density.json").read_text())
    out["scored_rows_by_grid"] = {
        g: int(sum(den["folds"][f"{g}|{f}"]["n_scored_cohort"] for f in FOLDS))
        for g in ("300", "600", "1200")}
    out["cohort_growth_1200_over_600_pct"] = round(
        100 * (out["scored_rows_by_grid"]["1200"] / out["scored_rows_by_grid"]["600"] - 1), 1)
    (EXP / "analysis" / "cohorts.json").write_text(json.dumps(out, indent=2))
    for f in FOLDS:
        v = out["folds"][f]
        print(f"  [{f:>8}] eligible {v['n_eligible']:,} ({v['eligible_pct']:.1f} %)  "
              f"dayblock fit {v['dayblock_fit_rows']['min']:,}-{v['dayblock_fit_rows']['max']:,}"
              f"  halfmonth {v['halfmonth_fit_rows']['min']:,}-{v['halfmonth_fit_rows']['max']:,}"
              f"  xmonth {v['xmonth_fit_rows']:,}")
    print(f"  scored rows by grid {out['scored_rows_by_grid']}, "
          f"1200 s cohort +{out['cohort_growth_1200_over_600_pct']} %")
    print(f"  written {EXP / 'analysis' / 'cohorts.json'}")


if __name__ == "__main__":
    main()
