"""The residual guard: clip the implied block-versus-AOBT gap, not the taxi time.

exp-001 and exp-002 both measured a ceiling on the *taxi time* and both found it
loses, because July 2025's true maximum is 88,132 s and a model's large
predictions are earned. A clip on `r = BLOCK - AOBT` is a different object. The
prediction implies `r_hat = (take-off - AOBT) - y_hat`; over 2025's labelled
departures `r` runs from -20,657 to +9,069 s and only 0.031 % of rows sit outside
[-10800, +3600] (exp-006 finding 8). A prediction whose implied `r` leaves that
range is the model extrapolating. Because the clip acts on the residual, the
prediction can still be 87,000 s wherever the known `take-off - AOBT` is.

It is still a clip, and it is applied to 2026 rows on a range measured in 2025.
If a 2026 month genuinely runs outside that range the guard will misfire, so
`count_touched` reports how many rows it moves.
"""
from __future__ import annotations

import numpy as np

LO, HI = -10800.0, 3600.0  # exp-006's bounds, kept only as a reference point


def bounds_from_months(months, q_lo: float = 0.0, q_hi: float = 1.0) -> tuple[float, float]:
    """Guard bounds from the training months' observed r = BLOCK - AOBT.

    exp-006 chose [-10800, +3600] by looking at 2025 as a whole, which for a fold
    includes its validation month, so the bounds were selected with labels the
    fold is not allowed to see. This derives them from the fold's training months
    alone. The default quantiles are 0 and 1 — the observed range, with no free
    parameter to tune — because the guard's job is to reject a prediction whose
    implied r is outside anything training ever saw.
    """
    import sys
    from pathlib import Path as _P

    import pandas as pd

    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    import prc.data as D

    d = pd.concat([D.load_month(m, columns=["PHASE_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"])
                   for m in months])
    d = d[d.PHASE_mvt == "DEP"]
    r = ((d.BLOCK_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()).dropna().to_numpy()
    return float(np.quantile(r, q_lo)), float(np.quantile(r, q_hi))


def apply(y_hat: np.ndarray, d_takeoff_aobt: np.ndarray, lo: float = LO, hi: float = HI,
          floor: float | None = 60.0) -> np.ndarray:
    """Clip the implied r on rows that have an NM off-block.

    Rows without one keep their prediction, except that the output floor below
    applies to every row — so a row with no NM off-block and a prediction under
    the floor is still lifted. That cannot arise in this pipeline, where
    predictions are floored before the guard runs, but the function does not
    promise otherwise.
    """
    y = np.asarray(y_hat, dtype="float64").copy()
    a = np.asarray(d_takeoff_aobt, dtype="float64")
    ok = np.isfinite(a)
    r = a[ok] - y[ok]
    y[ok] = a[ok] - np.clip(r, lo, hi)
    if floor is not None:
        y = np.maximum(y, floor)
    return y


def count_touched(y_hat: np.ndarray, d_takeoff_aobt: np.ndarray, lo: float = LO, hi: float = HI) -> int:
    a = np.asarray(d_takeoff_aobt, dtype="float64")
    ok = np.isfinite(a)
    r = a[ok] - np.asarray(y_hat, dtype="float64")[ok]
    return int(((r < lo) | (r > hi)).sum())


def write_guarded(name: str, folds=("july", "december", "january"), suffix: str = "-guard") -> list[str]:
    """Write guarded copies of a run's out-of-fold predictions, so the guard can be
    read with the same paired bootstrap as any other mechanism."""
    import sys
    from pathlib import Path

    import pandas as pd

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[1]))
    import prc.data as D

    written = []
    for fold in folds:
        p = here / "oof" / f"{name}-{fold}.parquet"
        if not p.exists():
            continue
        o = pd.read_parquet(p).set_index("MVT_ID_mvt")
        d = D.load_month(D.FOLDS[fold].valid_month,
                         columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
        d = d[d.PHASE_mvt == "DEP"].set_index("MVT_ID_mvt")
        a = ((d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()).reindex(o.index).to_numpy()
        o["y_pred"] = apply(o.y_pred.to_numpy(), a)
        out = here / "oof" / f"{name}{suffix}-{fold}.parquet"
        o.reset_index().to_parquet(out, index=False)
        written.append(str(out))
    return written


if __name__ == "__main__":
    import sys
    for f in write_guarded(sys.argv[1] if len(sys.argv) > 1 else "e0-base"):
        print("wrote", f)
