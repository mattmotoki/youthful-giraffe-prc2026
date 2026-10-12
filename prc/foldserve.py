"""One post-process for fold and serve (exp-147).

Every stage of the served chain turns fitted outputs into a prediction with the functions
here, on the fold side and on the ranking side alike, so the fold price describes the
vector that ships. They take arrays in one row order and hold no state.

* `head_correction` -- the surface head's bag: the MEAN OF RAW member corrections, then
  the class restriction, the guard and airport holds, then the 60 s floor. The old fold
  path processed each member before the mean (exp-046 `chainfold.py:306-324`), the
  ranking path after it (`serve.py:1284, 2489-2504`); the two do not commute (P5 item 1).
* `blend` -- `V + w (L - V)` on the rows a stage serves, with V the SERVED base on both
  sides. exp-134's folds blended on p12 and served on v22 (T1 L5).
* `floor_moved` -- the 60 s floor, applied only where a stage moves a row, so an
  unmoved row keeps its base bytes (exp-046 `arms.apply_floor`).
"""
from __future__ import annotations

import numpy as np

FLOOR = 60.0


def floor_moved(base: np.ndarray, d: np.ndarray, floor: float = FLOOR) -> np.ndarray:
    """Correction `d` with `base + d >= floor` wherever `d != 0`; zeros stay zero."""
    base = np.asarray(base, dtype="float64")
    d = np.asarray(d, dtype="float64")
    if base.shape != d.shape:
        raise ValueError("base and d must have one shape")
    out = d.copy()
    moved = d != 0
    out[moved] = np.maximum(base[moved] + d[moved], floor) - base[moved]
    return out


def head_correction(base: np.ndarray, members, *, in_class: np.ndarray,
                    held: np.ndarray | None = None, floor: float = FLOOR) -> np.ndarray:
    """The bag's correction for one scored frame.

    members: K raw member corrections (rows aligned with `base`), as fitted: no mask,
    guard or floor applied. in_class: the rows the stage may move (the frozen class).
    held: rows the guard or an airport hold keeps at the base.
    """
    stack = np.vstack([np.asarray(m, dtype="float64") for m in members])
    if stack.shape[1] != len(base):
        raise ValueError(f"members have {stack.shape[1]} rows, base has {len(base)}")
    if not np.isfinite(stack).all():
        raise ValueError("a member correction is not finite")
    d = np.where(np.asarray(in_class, dtype=bool), stack.mean(0), 0.0)
    if held is not None:
        d[np.asarray(held, dtype=bool)] = 0.0
    return floor_moved(base, d, floor)


def blend(V: np.ndarray, L: np.ndarray, w: float, rows: np.ndarray,
          floor: float = FLOOR) -> np.ndarray:
    """`V + w (L - V)` on `rows`, `V` elsewhere, floored where it moves."""
    V = np.asarray(V, dtype="float64")
    L = np.asarray(L, dtype="float64")
    rows = np.asarray(rows, dtype=bool)
    if not (V.shape == L.shape == rows.shape):
        raise ValueError("V, L and rows must have one shape")
    if not np.isfinite(V).all() or not np.isfinite(L[rows]).all():
        raise ValueError("a blend input is not finite")
    d = np.where(rows, w * (L - V), 0.0)
    return V + floor_moved(V, d, floor)
