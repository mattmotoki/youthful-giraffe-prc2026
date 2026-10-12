"""Fit the final blend weights on all OOF months and write the submission candidate.

The rotation in `blend_eval.py` estimates what a blend scores on a month it has
not seen; it is not how the shipped weights are chosen. At serve time there is
no held-out month, so the shipped weights are fitted on **every** available
validation month, each represented once by its ranking-analogue fold
(`july_loo`, `december`, `january`) — using both `july` and `july_loo` would
count 2025-07 twice. The rotated table is the honest estimate of what those
weights will do; this script produces them and the parquet.

  python make_candidate.py members.json --mode nnls --out exp-008-blend
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# ORDER MATTERS: exp-001 has its own `blend_eval.py`, and putting its directory
# ahead of this one on sys.path shadows ours and executes its module-level code
# at import time (it prints "no paired predictions yet" and exits 1, which looked
# for a while like an external guard). This directory goes first; exp-001's is
# only needed for `guard`.
sys.path.insert(0, str(ROOT / "experiments" / "exp-001-retro-v1"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402
import guard as guard_mod  # noqa: E402
import blend_eval as BE  # noqa: E402

FIT_FOLDS = ("july_loo", "december", "january")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--mode", default="nnls", choices=["nnls", "ols", "flat"])
    ap.add_argument("--members", default=None)
    ap.add_argument("--min-rows", type=int, default=50)
    ap.add_argument("--per-regime", default="1")
    ap.add_argument("--name", default="exp-008-blend")
    ap.add_argument("--ranking", default=None,
                    help="JSON mapping member -> ranking prediction parquet")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    files = cfg["members"]
    if a.members:
        files = {k: files[k] for k in a.members.split(",")}
    data, names = BE.load(files, folds=FIT_FOLDS)
    fit = pd.concat([data[f] for f in FIT_FOLDS], ignore_index=True)
    P = fit[names].to_numpy("float64")
    y = fit["y_true"].to_numpy("float64")
    per_regime = a.per_regime not in ("0", "false", "no")
    W = (B.fit_per_regime(P, y, fit["regime"].to_numpy(), a.mode, min_rows=a.min_rows)
         if per_regime else {"_all": B.fit_weights(P, y, a.mode)})
    weights = {k: dict(zip(names, map(float, v))) for k, v in W.items()}
    print("weights fitted on " + "+".join(FIT_FOLDS) + f" ({len(fit)} rows, mode {a.mode}):")
    print(pd.DataFrame(weights).round(3).to_string())

    lo, hi = guard_mod.bounds_from_months(D.MONTHS, 0.0, 1.0)
    out = {"members": list(names), "mode": a.mode, "per_regime": per_regime,
           "min_rows": a.min_rows, "fit_folds": list(FIT_FOLDS), "fit_rows": int(len(fit)),
           "weights": weights, "floor": B.FLOOR, "guard_bounds": [lo, hi],
           "prepare_members": True,
           "late_cut": B.LATE_CUT, "blend_version": B.VERSION}
    if a.ranking:
        out["ranking"] = json.loads(Path(a.ranking).read_text())
        rec = B.build_candidate(out, ROOT / "submissions" / "candidates" / f"{a.name}.parquet", ROOT)
        out["candidate"] = rec
        print("\ncandidate: " + json.dumps(rec, indent=2))
    (HERE / f"blend-config-{a.name}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote blend-config-{a.name}.json")


if __name__ == "__main__":
    main()
