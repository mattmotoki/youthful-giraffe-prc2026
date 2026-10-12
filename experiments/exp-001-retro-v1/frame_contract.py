"""The v2 feature frame's published contract, for exp-008's sibling models.

Prints the exact column list, dtypes and a content fingerprint. Run it after any
change to `features.py` and compare: if the fingerprint moves, the consumers of
the frame have to be told before the change lands.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import prc.data as D  # noqa: E402
from features import FRAME_VERSION, build_features  # noqa: E402

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
MONTH = "2025-07"

X = build_features(D.load_month(MONTH), VERSION)
cols = list(X.columns)
schema = "|".join(f"{c}:{X[c].dtype}" for c in cols)
h = hashlib.sha256(schema.encode()).hexdigest()[:16]
vals = hashlib.sha256(
    pd.util.hash_pandas_object(X.select_dtypes("number").round(6), index=False).values.tobytes()
).hexdigest()[:16]
print(f"frame {VERSION} (FRAME_VERSION={FRAME_VERSION[VERSION]}) on {MONTH}: {X.shape[0]} rows, {len(cols)} columns")
print(f"  schema fingerprint  {h}")
print(f"  numeric fingerprint {vals}")
print(f"  categoricals: {[c for c in cols if X[c].dtype == object]}")
print("  columns:")
for c in cols:
    print(f"    {c:28s} {X[c].dtype}")
