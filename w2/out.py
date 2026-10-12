"""Write a 2026 candidate file in the v64 schema and row order; return its SHA-256."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "iso"))
import finalmode as FM  # noqa: E402
ID, COL = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def write(ids, values, path: Path) -> str:
    tmpl = pd.read_parquet(FM.submitting_path(ROOT / "input-root/data/raw"))
    ids = np.asarray(ids, "float64"); v = np.asarray(values, "float64")
    assert np.array_equal(tmpl[ID].to_numpy("float64"), ids), "row order differs from the template"
    assert len(v) == FM.n_rows() and np.isfinite(v).all() and v.min() >= 0
    q = pd.DataFrame({ID: ids, COL: v})
    buf = io.BytesIO(); q.to_parquet(buf, index=False)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()
