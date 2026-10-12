"""Frame identity: rebuild one month and compare it with the cached frame.

`code/member.py gate` compares model PERFORMANCE against a committed vector; it
cannot see a stale cache, because `features.py build_all` skips a month whose
parquet already exists. This rebuilds a month into a scratch directory and
compares the 64 selected columns value by value against the cache.
Sol round 1, non-blocking 1.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/exp-003-anatomy/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-011-bulkfloor/code"))
import prc.data as d                     # noqa: E402
import features as FT                    # noqa: E402
import q1c_seeds as Q                    # noqa: E402

MONTH = sys.argv[1] if len(sys.argv) > 1 else "2025-07"
OUT = ROOT / "experiments/exp-034-bulk/analysis/verify-frame.json"


def main() -> int:
    t0 = time.time()
    cached = pd.read_parquet(FT.ART / f"{MONTH}.parquet")
    fresh = FT.build_month(d.load_month(MONTH))
    rec = {"month": MONTH, "cached_shape": list(cached.shape), "fresh_shape": list(fresh.shape),
           "n_features": len(Q.FEATURES)}

    if not cached["MVT_ID_mvt"].equals(fresh["MVT_ID_mvt"]):
        rec["ok"] = False
        rec["reason"] = "the row order or the id set differs"
        OUT.write_text(json.dumps(rec, indent=2))
        print(json.dumps(rec, indent=2))
        return 1

    diffs = {}
    for c in Q.FEATURES:
        a, b = cached[c], fresh[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy("float64"), b.to_numpy("float64")
            bad = int((~(np.isclose(x, y, rtol=0, atol=0, equal_nan=True))).sum())
        else:
            bad = int((a.astype(str).to_numpy() != b.astype(str).to_numpy()).sum())
        if bad:
            diffs[c] = bad
    rec["columns_equal"] = len(Q.FEATURES) - len(diffs)
    rec["columns_different"] = diffs
    rec["ok"] = not diffs
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
