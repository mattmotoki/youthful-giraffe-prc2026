"""K2's window block, built for `prc.tail_model`'s frames, one month at a time.

Nothing is built here. `experiments/exp-018-fresh/code/window_features.py` is
K2's block, unedited, and this module is the adapter that gives it to a second
member: it enforces the one-month-at-a-time contract, re-blanks the departure
label columns defensively before the block sees the frame, and returns a frame
indexed like the input frame's departure rows so
`prc.tail_model.with_window` can join it onto the signals.

The leakage contract is K2's, restated as code rather than as a claim
(`experiments/exp-027-step/SPEC.md`, "Leakage contract"):

- a departure's own `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are set to NaN
  before anything is built, so an upstream masking fault cannot become a leak;
- the arrival block uses an ARRIVAL row's taxi-in, which `docs/input-rule.md`
  states is a legal input and which the ranking file carries unmasked;
- one month at a time: the ranking file holds January and July 2026 six months
  apart and a window spanning them is meaningless. The training frame is a
  concat of eleven months and is grouped the same way `signals_by_month` groups
  it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/exp-018-fresh/code"))
import window_features as WF  # noqa: E402

#: K2's own re-blanked columns, `exp-018-fresh/code/features.py` and
#: `exp-027-step/code/cell_reach.py`.
_LABEL_COLS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")


def _one_month(frame: pd.DataFrame) -> pd.DataFrame:
    dep = (frame["PHASE_mvt"] == "DEP").to_numpy()
    f = frame.copy()
    for c in _LABEL_COLS:
        if c in f.columns:
            f.loc[dep, c] = pd.NaT if f[c].dtype.kind == "M" else np.nan
    return WF.from_frame(f)


def build(frame: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    """The 68-column block for every departure row of `frame`, per month."""
    if month_col not in frame.columns:
        # Sol round 1, finding 9: a silent whole-frame fallback would break the
        # one-month rule for a future caller. Both callers here pass `month`;
        # a frame that spans more than one month and does not say so is a bug,
        # not a default, so it is refused rather than quietly concatenated.
        # Sol round 2: keying only on BLOCK_TIME_UTC_mvt let a multi-month
        # frame with no block timestamps fall through. Every time column the
        # frame carries is checked, so a month boundary cannot hide in a
        # column this one happened not to look at.
        stamps = [c for c in frame.columns if frame[c].dtype.kind == "M"]
        seen = set()
        for c in stamps or ["BLOCK_TIME_UTC_mvt"]:
            seen |= set(pd.to_datetime(frame[c]).dt.to_period("M").dropna().unique())
        if len(seen) > 1:
            raise ValueError(
                f"build() got a frame with no {month_col!r} column spanning "
                f"{len(seen)} months ({sorted(str(m) for m in seen)[:4]}...); "
                "the window is meaningless across a month boundary -- pass a "
                "month column or one month at a time")
        return _one_month(frame)
    parts = [_one_month(g) for _, g in frame.groupby(month_col, sort=True)]
    out = pd.concat(parts)
    return out.astype("float32")


if __name__ == "__main__":
    import time
    sys.path.insert(0, str(ROOT))
    import prc.data as D
    m = sys.argv[1] if len(sys.argv) > 1 else "2025-07"
    t0 = time.time()
    fr = D.load_month(m)
    t1 = time.time()
    W = build(fr)
    print(f"{m}: frame {fr.shape} in {t1-t0:.0f}s; block {W.shape} in {time.time()-t1:.0f}s; "
          f"{W.memory_usage(deep=True).sum()/1e6:.0f} MB")
    print(list(W.columns)[:6], "...", list(W.columns)[-3:])
