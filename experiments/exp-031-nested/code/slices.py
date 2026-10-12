"""COMMON.md's decomposition for every candidate on every column, both anchors.

The ledger's `july tail>3600` and `july no-NM` columns.  `y > 3600` and the
strict no-NM mask (`exp-028/code/review1.no_nm_mask`, "every `_flt` column
null") on the CLEAN rows, the same population as the point estimate.

Sol round 2, finding 7: this script reads whatever `oof/` holds, and `oof/`
holds two of the eight (column x anchor) families -- `oof/README.md` says which
and how to rebuild the rest.  So it MERGES into the committed
`analysis/slices.json` instead of replacing it, and refuses to drop a family
that is already recorded.  The committed file was produced while all seven
written families were on disk (the shipped-anchor `published` column is never
written: `rotate.py` skips it, because those bytes are already in git).

    python slices.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
sys.path.insert(0, str(ROOT))
import review1 as R1  # noqa: E402
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
ARMS = ("stack_5", "v6", "v7", "K2", "refit_bulk")


def main() -> None:
    dst = EXP / "analysis" / "slices.json"
    out = json.loads(dst.read_text()) if dst.exists() else {}
    before = {(a, c) for a in out for c in out[a]}
    for p in sorted(EXP.glob("oof/exp-031*.parquet")):
        stem = p.stem.removeprefix("exp-031")
        fold = next(f for f in FOLDS if stem.endswith(f))
        col = stem[:-len(fold) - 1].strip("-")
        anchor = "nested" if col.startswith("nestedanchor") else "rotated"
        col = col.removeprefix("nestedanchor-")
        d = pd.read_parquet(p)
        y = d["y_true"].to_numpy("float64")
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(fold)).to_numpy()
        strict = R1.no_nm_mask(fold, ids)[ok]
        yy = y[ok]
        for a in ARMS:
            e = ((d[f"y_pred_{a}"].to_numpy("float64") - y) ** 2)[ok]
            rr = lambda m: round(float(np.sqrt(e[m].mean())), 2)
            out.setdefault(anchor, {}).setdefault(col, {}).setdefault(a, {})[fold] = {
                "n": int(ok.sum()), "le3600": rr(yy <= 3600), "gt3600": rr(yy > 3600),
                "no_nm": rr(strict), "n_no_nm": int(strict.sum())}
    after = {(a, c) for a in out for c in out[a]}
    if not before <= after:
        raise SystemExit(f"refusing to drop recorded families: {sorted(before - after)}")
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n  {len(after)} (anchor, column) families recorded, "
          f"{len(after - before)} new this run")
    for anchor in out:
        for col in out[anchor]:
            for a in ARMS:
                r = out[anchor][col][a]["july_loo"]
                print(f"{anchor:8s} {col:14s} {a:11s} july  gt3600 {r['gt3600']:9.2f}  "
                      f"no_nm {r['no_nm']:9.2f}")
    print("\n  written: analysis/slices.json")


if __name__ == "__main__":
    main()
