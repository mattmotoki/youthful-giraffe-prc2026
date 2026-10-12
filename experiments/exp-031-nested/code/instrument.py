"""The four columns of the instrument, as a member-column swap on one frame.

`exp-026-rebuild/code/frames.fold_frame` is the canonical fold frame: the member
out-of-fold predictions, floored and residual-guarded exactly as
`prc.blend.build_candidate` prepares them, joined to the five gate columns and
to `t`.  This module does not rebuild it.  It builds it once per fold from the
published addresses and then **replaces a member's column** with the same
member's prediction from `exp-025-cloud/oof-nested/`, re-floored and re-guarded
by the same two calls on the same `d_takeoff_aobt`.

That is why the `published` column here is bit-identical to `exp-027-step`'s by
construction rather than by a gate: it is the same object, untouched.

Four columns:

| name | the three refittable members read from |
|---|---|
| `published` | the committed workstation OOF (the record) |
| `kaggle` | `oof-nested/<member>--<fold>.parquet`, the same members' ORDINARY folds re-run on Kaggle |
| `nested` | fold `f` scored: `f` itself from the Kaggle baseline, the two FITTING folds from `<g>_nx_<mon(f)>` |
| `nested-f1held` | `nested`, with `F1-fresh` held at its Kaggle baseline everywhere |

`published -> kaggle` is the machine term; `kaggle -> nested` is the nesting.
The `oof-nested/` README requires the middle column: a shift measured from the
published files is mostly CatBoost on a different GPU (RESULTS section 10.6).

`nested-f1held` exists because `K2-window` has no nested refit.  Without it the
`K2` arm would have two members nested against `v7`'s three and their difference
would carry one member's change of instrument.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import frames as F  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.blend as B  # noqa: E402

FOLDS = F.FOLDS
#: `prc.data.NESTED_FOLDS` read the other way round: which extra month a fold's
#: training set must drop so that it may serve the rotation of `scored`.
MON = {"july_loo": "jul", "december": "dec", "january": "jan"}
#: The three members `exp-025-cloud` could refit.  `V3-allin` is absent because
#: its frame needs the Copernicus ERA5 snapshot (RESULTS section 10.3); `C2-cat`,
#: `D1-winter` and `N1-nmtail` were not in that experiment's scope.
NESTABLE = ("F1-fresh", "T-tailv4", "L2-hub8725")
ND = ROOT / "experiments/exp-025-cloud/oof-nested"
COLUMNS = ("published", "kaggle", "nested", "nested-f1held")


def _kaggle(member: str, foldfile: str) -> pd.DataFrame:
    """One `oof-nested/` file, ids as float64.  The `fold` column is its own name."""
    d = pd.read_parquet(ND / f"{member}--{foldfile}.parquet")
    got = set(pd.unique(d["fold"]))
    if got != {foldfile}:
        raise SystemExit(f"{member}--{foldfile}: the fold column says {got}")
    d = d[["MVT_ID_mvt", "y_pred"]].copy()
    d["MVT_ID_mvt"] = d["MVT_ID_mvt"].astype("float64")
    return d


def _swap(m: pd.DataFrame, member: str, src: pd.DataFrame) -> None:
    """Replace `member`'s prepared column in place, from `src`'s `y_pred`."""
    v = (src.set_index("MVT_ID_mvt")["y_pred"]
            .reindex(m["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64"))
    if not np.isfinite(v).all():
        raise SystemExit(f"{member}: {int((~np.isfinite(v)).sum())} frame rows are "
                         f"not in the substituted file")
    a = m["d_takeoff_aobt"].to_numpy("float64")
    v = np.maximum(v, B.FLOOR)
    m[member] = B.apply_residual_guard(v, a, *F.GUARD, B.FLOOR)


def source(column: str, member: str, slot: str, scored: str) -> str:
    """Which `oof-nested/` file member `member` reads in fold `slot`, or "" for published.

    `slot` is the fold whose rows are being filled; `scored` is the fold the
    rotation is about to score.  A fitting fold (`slot != scored`) on a nested
    column drops the scored fold's month as well, which is exactly what closes
    the indirect label channel `internal decision record` nested validation rule names.
    """
    if member not in NESTABLE or column == "published":
        return ""
    if column == "kaggle":
        return slot
    if column == "nested-f1held" and member == "F1-fresh":
        return slot
    if column not in ("nested", "nested-f1held"):
        raise SystemExit(f"unknown instrument column {column!r}")
    return slot if slot == scored else f"{slot}_nx_{MON[scored]}"


def base_frames(names: tuple[str, ...]) -> dict:
    """`frames.fold_frame` on the published addresses, one per fold."""
    return {f: F.fold_frame(f, names) for f in FOLDS}


def build(column: str, scored: str, names: tuple[str, ...], base: dict) -> dict:
    """The three fold frames the rotation of `scored` runs on, for one column."""
    if column == "published":
        return base
    out = {}
    for slot in FOLDS:
        m = base[slot].copy()
        for member in names:
            src = source(column, member, slot, scored)
            if src:
                _swap(m, member, _kaggle(member, src))
        out[slot] = m
    return out


def describe(column: str, scored: str, names: tuple[str, ...]) -> dict:
    return {m: {slot: (source(column, m, slot, scored) or "published") for slot in FOLDS}
            for m in names if m in NESTABLE}


if __name__ == "__main__":
    names = ("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh",
             "V3-allin", "N1-nmtail")
    base = base_frames(names)
    for col in COLUMNS:
        for scored in FOLDS:
            d = build(col, scored, names, base)
            print(f"{col:14s} scored={scored:9s} "
                  + "  ".join(f"{m}:" + "/".join(
                      d and (source(col, m, s, scored) or "pub") for s in FOLDS)
                      for m in NESTABLE))
