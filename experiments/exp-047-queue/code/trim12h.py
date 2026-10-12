"""exp-047: the registered trimmed-12h companion mask.

RECORD ONLY, NEVER A GATE -- desk ruling 2026-09-09, registered in
`analysis/trimmed-12h-registration.json` and in RESULTS 7be.  Every read this lane
makes prints the raw number, which is the gate, and this companion beside it.

The row set is FROZEN in the registration file.  This module reads it from there
rather than recomputing `y > 43200`, so a panel that later gains or loses a row
cannot silently change what "trimmed-12h" means.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REG = Path(__file__).resolve().parent.parent / "analysis/trimmed-12h-registration.json"
CUT_S = 43200.0


def registered() -> dict:
    return json.loads(REG.read_text())


def excluded_ids(fold: str) -> np.ndarray:
    """The frozen MVT ids this companion removes on one fold."""
    return np.array([r["MVT_ID_mvt"] for r in registered()["excluded_rows"]
                     if r["fold"] == fold], dtype="float64")


def keep(fold: str, ids: np.ndarray) -> np.ndarray:
    """A boolean mask over a fold panel's rows: True for the rows the companion keeps."""
    return ~np.isin(ids, excluded_ids(fold))
