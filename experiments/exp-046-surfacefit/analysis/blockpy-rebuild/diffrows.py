"""The ROW set that moves, and how much of it is spillover onto rows whose own match held.

`diffcols.py` counts per column; this counts per row, because the question a consumer asks
is "which of my rows change", and a feature that is computed from the NEIGHBOURHOOD (the
arrival at the same stand) can move on a row whose own match did not.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
H = HERE.parents[1] / "work/blockpy-rebuild"      # the class frames
SKIP = {"MVT_ID_mvt", "airport"}
out = {}
for month in ["2025-01", "2025-07", "2025-12", "2026-01", "2026-07"]:
    a = pd.read_parquet(H / f"class-{month}-shipped.parquet")
    b = pd.read_parquet(H / f"class-{month}-v2.parquet")
    diff = np.zeros(len(a), bool)
    per_col = {}
    for c in a.columns:
        if c in SKIP:
            continue
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            xf, yf = x.astype("float64"), y.astype("float64")
            m = (xf.isna() != yf.isna()) | (xf.notna() & yf.notna()
                                            & ~np.isclose(xf, yf, equal_nan=True))
        else:
            m = x.astype(str).ne(y.astype(str)) | (x.isna() != y.isna())
        m = m.to_numpy()
        per_col[c] = int(m.sum())
        diff |= m
    cls = (a.obs_matched.fillna(False).astype(bool).to_numpy()
           != b.obs_matched.fillna(False).astype(bool).to_numpy())
    cls |= (a.obs_has_clock.fillna(False).astype(bool).to_numpy()
            != b.obs_has_clock.fillna(False).astype(bool).to_numpy())
    spill = diff & ~cls
    out[month] = {
        "rows": int(len(a)),
        "rows_with_any_difference": int(diff.sum()),
        "rows_whose_class_changed": int(cls.sum()),
        "spillover_rows_whose_class_held": int(spill.sum()),
        "spillover_by_airport": {str(k): int(v) for k, v in
                                 pd.Series(a.airport[spill]).value_counts().items()},
        "spillover_columns": {c: int((m := (
            (a[c].astype(str).ne(b[c].astype(str))) | (a[c].isna() != b[c].isna())
        ).to_numpy() & spill).sum()) for c in a.columns if c not in SKIP
            and (((a[c].astype(str).ne(b[c].astype(str)))
                  | (a[c].isna() != b[c].isna())).to_numpy() & spill).any()},
        "per_column_rows": {c: n for c, n in per_col.items() if n},
    }
    print(month, json.dumps({k: v for k, v in out[month].items()
                             if k != "per_column_rows"}), flush=True)
(HERE / "row-diff.json").write_text(json.dumps(out, indent=2, sort_keys=True))
print("written", HERE / "row-diff.json")
