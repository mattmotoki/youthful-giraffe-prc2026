"""Step 5: v9 with the noac rows changed and nothing else.

The correction is the same object §4 measured: `y_chain - y_base` from two
heads fitted on all twelve 2025 months, applied only where `mvt_sched <
BAND_MIN` (above it the shipped value is the exp-009 band table, not the
learned mixture).  Every other row of v9 is copied byte for byte.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
from prc.tail_model import BAND_MIN, MIN_TAXI  # noqa: E402

V9 = ROOT / "submissions" / "candidates" / "exp-027-k2.parquet"
OUT = ROOT / "submissions" / "candidates" / "exp-037-tail.parquet"


def build() -> tuple:
    v9 = pd.read_parquet(V9)
    tmpl = D.load_submitting()
    assert list(v9.MVT_ID_mvt) == list(tmpl.MVT_ID_mvt), "v9 is not in template order"
    r = pd.read_parquet(EXP / "work" / "ranking_head.parquet")
    corr = np.where(r.mvt_sched.to_numpy(float) < BAND_MIN,
                    r.y_chain.to_numpy(float) - r.y_base.to_numpy(float), 0.0)
    s = pd.Series(corr, index=r.MVT_ID_mvt.to_numpy("float64"))
    out = v9.copy()
    add = s.reindex(out.MVT_ID_mvt.to_numpy("float64")).fillna(0.0).to_numpy()
    new = np.clip(out.TAXITIME_SEC_mvt.to_numpy("float64") + add, MIN_TAXI, None)
    moved = new != out.TAXITIME_SEC_mvt.to_numpy("float64")
    out["TAXITIME_SEC_mvt"] = new
    d = new - v9.TAXITIME_SEC_mvt.to_numpy("float64")
    return out, v9, d, moved


def main() -> None:
    out, v9, d, moved = build()
    n_noac = 388
    print("rows:", len(out), "moved:", int(moved.sum()),
          "identical:", int((~moved).sum()))
    print("shift RMS over the 388 noac rows: %.2f s" % float(np.sqrt(np.sum(d ** 2) / n_noac))) 
    print("max |shift|: %.2f s" % float(np.abs(d).max()))
    print("finite:", bool(np.isfinite(out.TAXITIME_SEC_mvt).all()),
          "min:", float(out.TAXITIME_SEC_mvt.min()))
    tmpl = D.load_submitting()
    assert list(out.MVT_ID_mvt) == list(tmpl.MVT_ID_mvt)
    assert out.TAXITIME_SEC_mvt.dtype == np.float64
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print("wrote", OUT, h)
    # the rebuild check: rebuild in memory and compare to the file on disk
    back = pd.read_parquet(OUT)
    print("max abs difference file vs rebuild: %.6g"
          % float(np.abs(back.TAXITIME_SEC_mvt.to_numpy() - out.TAXITIME_SEC_mvt.to_numpy()).max()))
    unchanged = int((back.TAXITIME_SEC_mvt.to_numpy() == v9.TAXITIME_SEC_mvt.to_numpy()).sum())
    print("rows bit-identical to v9:", unchanged)


if __name__ == "__main__":
    main()
