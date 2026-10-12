"""exp-054-arrivalrepr: the review-repair readers (Sol round 1, finding 1; SPEC amendment A1).

`e54.py` stays byte for byte as registered: its sha is in every record.  Its `arrival_outcomes()`
reads the two label columns (`BLOCK_TIME_UTC_mvt`, `TAXITIME_SEC_mvt`) of the WHOLE month file and
only then keeps the ARR rows, so a pandas frame holds the month's departure labels for a moment.
The readers here ask pyarrow for the ARR rows only (`filters=[("PHASE_mvt", "==", "ARR")]`), so no
departure row, and so no departure label, is in any frame they return.  (pyarrow decodes the file's
pages to apply the filter, as the file sha256 reads every byte; no departure value leaves it.)

`arrival_outcomes_arr()` is `e54.arrival_outcomes()` with that one change: the same identity check,
clip, uniqueness check and id order.  `audit2.py` and `represent_r.py` use these readers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e54 as E                                                       # noqa: E402

D = E.D
ARR_ONLY = [("PHASE_mvt", "==", "ARR")]


def arrival_rows(month: str, columns: list[str] | None = None) -> pd.DataFrame:
    """The ARR rows of one month (all columns when `columns` is None), times parsed as
    `prc.data.load_month` parses them.  Refuses a frame that holds any other phase."""
    if columns is not None and "PHASE_mvt" not in columns:
        columns = ["PHASE_mvt"] + list(columns)
    df = pd.read_parquet(D.month_file(month), columns=columns, filters=ARR_ONLY)
    ph = df["PHASE_mvt"].astype(str).to_numpy()
    if not (ph == "ARR").all():
        raise SystemExit(f"{month}: the ARR-only read returned {int((ph != 'ARR').sum())} other rows")
    df = D._parse_times(df.reset_index(drop=True))
    df["month"] = month
    return df


def phase_counts(month: str) -> dict:
    """Row counts of one month by phase, from the PHASE_mvt column alone."""
    ph = pd.read_parquet(D.month_file(month), columns=["PHASE_mvt"])["PHASE_mvt"].astype(str).to_numpy()
    return {"rows": int(len(ph)), "arrivals": int((ph == "ARR").sum()),
            "departures": int((ph == "DEP").sum())}


def arrival_outcomes_arr(month: str, ids: np.ndarray) -> np.ndarray:
    """`e54.arrival_outcomes(month, ids)` from an ARR-only read: arrival taxi-in `B_arr - T_arr`
    (asserted equal to TAXITIME_SEC_mvt on every ARR row, a missing value on both sides counting
    as equal), clipped to E.Y_CLIP, in `ids` order; NaN where the arrival has no outcome."""
    a = arrival_rows(month, ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"] + E.ARR_TARGET_COLS)
    alg = (a["BLOCK_TIME_UTC_mvt"] - a["MVT_TIME_UTC_mvt"]).dt.total_seconds().to_numpy("float64")
    tin = a["TAXITIME_SEC_mvt"].to_numpy("float64")
    same = (alg == tin) | (np.isnan(alg) & np.isnan(tin))
    if not same.all():
        raise SystemExit(f"{month}: TAXITIME_SEC_mvt != BLOCK - MVT on {int((~same).sum())} ARR rows")
    s = pd.Series(np.clip(tin, *E.Y_CLIP), index=a["MVT_ID_mvt"].to_numpy("float64"))
    if not s.index.is_unique:
        raise SystemExit(f"{month}: arrival ids are not unique")
    missing = ~np.isin(np.asarray(ids, dtype="float64"), s.index.to_numpy("float64"))
    if missing.any():
        raise SystemExit(f"{month}: {int(missing.sum())} requested ids are not arrivals of the month")
    return s.reindex(ids).to_numpy("float64")
