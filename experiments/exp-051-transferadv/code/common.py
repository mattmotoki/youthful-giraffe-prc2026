"""exp-051 shared construction, taken from exp-046 without change.

The panels, member 1's placebo perturbation, the extension frames and the pricing function
are exp-046's own (`onemonth.py`, `onemonthbag.py`, `rung511.price`).  This file only
assembles them, so that a board-like fit differs from the plain member-1 stack in ONE thing:
half of the scored fold month's own days in the fitting set.

Large vectors go to `artifacts/exp-051/work/` (git-ignored); their SHA-256 is written into
the analysis JSON that cites them.  Every artifact is write-once (exp-046 `nooverwrite`).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
E46 = ROOT / "experiments/exp-046-surfacefit"
sys.path.insert(0, str(E46 / "code"))
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import classplacebo as CP                                             # noqa: E402
import rung511 as R                                                   # noqa: E402
import onemonth as OM                                                 # noqa: E402
import withinsplit as WS                                              # noqa: E402
import nooverwrite as NO                                              # noqa: E402

WORK = ROOT / "artifacts/exp-051/work"
ANALYSIS = EXP / "analysis"
LEAVES = 255
SEED_MEMBER = 1
HALF_NAME = {0: "A", 1: "B"}

#: The three uploaded with-clock steps, stacked in uploaded order.  Level 0 is member 1 alone.
#: v22's month per recorded rule's seasonal-head fold rule: 2025-08 on july, 2025-02 on december and
#: january (exp-046 `onemonthbag.STACKS["seasonal"]`).
LEVELS = {
    0: {f: [] for f in A.FOLDS},
    1: {f: ["2025-11"] for f in A.FOLDS},
    2: {f: ["2025-11", "2025-10", "2025-03"] for f in A.FOLDS},
    3: {"july_loo": ["2025-11", "2025-10", "2025-03", "2025-08"],
        "december": ["2025-11", "2025-10", "2025-03", "2025-02"],
        "january": ["2025-11", "2025-10", "2025-03", "2025-02"]},
}
STEP = {1: "v20", 2: "v21", 3: "v22"}

#: The saved plain member-1 stacks (K = 1, placebo seed 1), each the member 1 of the bag
#: that shipped at that step (exp-046 `onemonthbag.STACKS`).
PLAIN = {0: E46 / "work/d-bundle-placebo1-255-free.npz",
         1: E46 / "work/d-onemonth-screen-255-free.npz",
         2: E46 / "work/d-onemonth-twomonth-plus03-255-free.npz",
         3: E46 / "work/d-seasonal-m1-255-free.npz"}

#: Read-only board comparators (docs/SUBMISSIONS.md rows 20-22).  score-feedback rule: nothing is fitted,
#: selected, scaled or revised on these.  Transfer = board gain / K = 10 fold face.
BOARD = {"v20": {"board_gain_s": 245.7752 - 245.2901, "face_s": 0.513473},
         "v21": {"board_gain_s": 245.2901 - 245.094, "face_s": 0.55876},
         "v22": {"board_gain_s": 245.094 - 245.0207, "face_s": 0.398006}}
for _v in BOARD.values():
    _v["transfer"] = _v["board_gain_s"] / _v["face_s"]
#: the assignment's rounded figures, kept beside the exact ones
BOARD_ASSIGNMENT = {"v20": 0.946, "v21": 0.352, "v22": 0.185}


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ids_sha16(ids) -> str:
    return hashlib.sha256(np.sort(np.asarray(ids, dtype="float64")).tobytes()).hexdigest()[:16]


def load_vec(p: Path) -> dict:
    return {f: v.astype("float64") for f, v in np.load(p).items()}


def build_panels() -> tuple[dict, dict, dict, list]:
    """`pan_ship` (priced on), `pan` (member 1's placebo-1 fitting panel), `ship`, `scols`.

    Exactly `onemonth.py`'s construction: the ungated blocks restricted at R.THR and the
    step-0 gap cut; member 1's perturbation drawn with seed 1 over the fold months in
    A.FOLDS order.
    """
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        mth = A.FOLD_MONTH[f]
        b = CF.build_month(mth, A.prefix_months_for(mth), rule)
        R.assert_mode3(b, mth, "shipped")
        blk[mth] = b
    pan_ship = CF.panel_for(blk, R.THR, {mth: CF.GAP_CUT_S for mth in blk})
    ship, scols = A.shipped_cols(pan_ship)
    rng = np.random.default_rng(SEED_MEMBER)
    blocks = {}
    for mth, b in blk.items():
        pb, _ = CP.placebo_block(b, R.THR, CF.GAP_CUT_S, CP.MOVED[mth], rng)
        blocks[mth] = pb
    real = A.month_block
    A.month_block = (lambda bs: (lambda month, prefix_months, rule=None: bs[month]))(blocks)
    try:
        pan = A.panel_with_block(BA.load())
    finally:
        A.month_block = real
    for f in A.FOLDS:
        if not np.array_equal(pan[f].MVT_ID_mvt.to_numpy("float64"),
                              pan_ship[f].MVT_ID_mvt.to_numpy("float64")):
            raise SystemExit(f"{f}: member 1's panel is not in the shipped panel's row order")
        if not pan[f].day.equals(pan_ship[f].day):
            raise SystemExit(f"{f}: member 1's panel carries different days")
    return pan_ship, pan, ship, scols


def half_of_rows(pan_ship: dict, f: str) -> np.ndarray:
    """0 = half A (day blocks 0 and 2), 1 = half B (blocks 1 and 3), for every fold row.

    `withinsplit.halves` over the fold's own days: four contiguous calendar-day blocks."""
    return WS.halves(pan_ship[f].day)


def half_fit_rows(pan: dict, ship: dict, f: str, h: int, hv: np.ndarray) -> pd.DataFrame:
    """Half `h` of fold `f`'s own usable rows on member 1's panel, shipped columns joined
    exactly as `chainfold.fit_arm` joins them to member 1's fitting rows, and carrying member
    1's target y - pbase (the served head's fold months carry the same target)."""
    p = pan[f]
    sub = p[p.usable.to_numpy(bool) & (hv == h)].reset_index(drop=True)
    return sub.join(ship[f].reindex(sub.MVT_ID_mvt.to_numpy("float64")).reset_index(drop=True))


def ext_frame(scols: list, months: list, cache: dict) -> pd.DataFrame | None:
    """The extension months in the given order, as `onemonthbag._stack_frame` builds them."""
    frame = None
    for mo in months:
        if mo not in cache:
            r = OM.nov_rows(scols, month=mo)
            cache[mo] = r[r.usable.to_numpy(bool)].reset_index(drop=True)
        rf = cache[mo]
        if frame is None:
            frame = rf
        else:
            miss = [c for c in frame.columns if c not in rf.columns]
            if miss:
                raise SystemExit(f"{mo} lacks {len(miss)} column(s) the stack carries: {miss[:8]}")
            frame = pd.concat([frame, rf.reindex(columns=frame.columns)], ignore_index=True)
    return frame


def fit_path(f: str, h: int | None, level: int) -> tuple[Path, Path]:
    tag = (f"bl-{f}-{HALF_NAME[h]}-L{level}" if h is not None
           else f"plainrepro-{f}-L{level}")
    return WORK / f"{tag}.npz", ANALYSIS / "fits" / f"{tag}.json"


def write_json(p: Path, obj: dict, overwrite: bool = False) -> None:
    NO.refuse_existing([p], overwrite=overwrite)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float))
