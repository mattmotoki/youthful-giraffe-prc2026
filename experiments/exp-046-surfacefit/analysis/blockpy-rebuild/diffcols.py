"""Every emitted column, both variants, per month: gained / lost / CHANGED WHERE BOTH PRESENT.

The claim the spec makes is "no existing value moves", so the sweep is over the whole
emitted frame and not over a chosen subset.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
H = HERE.parents[1] / "work/blockpy-rebuild"      # the class frames
out, worst = {}, 0
for month in ["2025-01", "2025-07", "2025-12", "2026-01", "2026-07"]:
    a = pd.read_parquet(H / f"class-{month}-shipped.parquet")
    b = pd.read_parquet(H / f"class-{month}-v2.parquet")
    assert a.MVT_ID_mvt.equals(b.MVT_ID_mvt), month
    assert list(a.columns) == list(b.columns), month
    rec = {}
    for c in a.columns:
        if c in ("MVT_ID_mvt", "airport"):
            continue
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            x, y = x.astype("float64"), y.astype("float64")
            both = x.notna() & y.notna()
            chg = int((both & ~np.isclose(x, y, equal_nan=True)).sum())
        else:
            x, y = x.astype(str).where(a[c].notna()), y.astype(str).where(b[c].notna())
            both = x.notna() & y.notna()
            chg = int((both & x.ne(y)).sum())
        rec[c] = {"gained": int((x.isna() & y.notna()).sum()),
                  "lost": int((x.notna() & y.isna()).sum()),
                  "changed_where_both_present": chg}
        worst = max(worst, chg)
    out[month] = rec
    moved = {c: v for c, v in rec.items() if any(v.values())}
    print(month, json.dumps(moved), flush=True)

out["_columns_swept"] = sorted(out["2025-01"])
out["_worst_changed_where_both_present"] = worst
(HERE / "interface-columns.json").write_text(json.dumps(out, indent=2, sort_keys=True))
print("columns swept:", len(out["_columns_swept"]), out["_columns_swept"])
print("worst changed_where_both_present:", worst)
