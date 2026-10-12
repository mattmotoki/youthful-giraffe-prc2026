"""Feature blocks at more than one window width, in one stacker.

exp-028 measured the window family one width at a time: a block name (`depx`)
and a single width (`PRC_WIN`).  This experiment's first two arms need **two and
three aggregate sets side by side in the same stacker**, so a block name here may
carry its own width:

    depx            the 600 s block exactly as exp-028 builds it, same columns
    depx_w300       the same nine aggregates and four lags at 300 s, every
                    window column suffixed `_w300`, the five row-context columns
                    (`hour`, `dow`, `apt`, `own_ta`, `own_al`) NOT repeated
    arrk            the arrivals keyed on runway and stand (`arrk.py`), arm 4

The width-tagged form drops the row-context columns because they do not depend
on the width: repeating them would hand the stacker five identical columns per
extra width and change what `feature_fraction` samples, which is a different
construction from "the same block at a second width".

**Every departure block still comes out of `exp-028/code/stack5w.block_for` and
`stack5w.ranking_blocks`.**  This module only chooses the width, renames, and
concatenates, so the fold side and the serving side stay the one function pair
exp-028 checked against the real months (`tests/panel_vs_raw.py`, RESULTS §8b).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
W28 = ROOT / "experiments/exp-028-window/code"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(W28))
import arrk as Ak  # noqa: E402
import features as Fe  # noqa: E402
import stack5w as SW  # noqa: E402

#: exp-028's own two builders, captured at import.  `rotate.py` rebinds
#: `stack5w.block_for` to this module's, so the names must be taken here or the
#: rebound name would make `block_for` below call itself.
_BLOCK_FOR = SW.block_for
_RANKING_BLOCKS = SW.ranking_blocks

#: `<block>` or `<block>_w<seconds>`.
TAGGED = re.compile(r"^(?P<base>.+?)_w(?P<win>\d+)$")


def parse(name: str, default_win: float) -> tuple[str, float, bool]:
    """(base block name, its width, whether the width was named in the name)."""
    m = TAGGED.match(name)
    if not m:
        return name, float(default_win), False
    return m.group("base"), float(m.group("win")), True


def _tag(block: pd.DataFrame, win: float) -> pd.DataFrame:
    """Drop the width-independent row context and suffix what is left."""
    keep = [c for c in block.columns if c not in Fe.ROW_ONLY]
    return block[keep].add_suffix(f"_w{int(win)}")


def widths(spec: dict) -> list[float]:
    return [parse(n, spec["win"])[1] for n in spec["blocks"]]


def block_for(fold: str, dep: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """The fold-side block for `spec = {"blocks": [...], "win": float}`.

    Signature-compatible with `stack5w.block_for`, which is what lets
    `rotate.py` hand it to exp-028's own rotation unchanged.
    """
    pan = SW.panel_of(dep) if "base" not in dep.columns else dep
    parts = []
    for name in spec["blocks"]:
        base, win, tagged = parse(name, spec["win"])
        if base == "arrk":
            b = Ak.for_fold(fold, pan, win)
        else:
            b = _BLOCK_FOR(fold, pan, {"blocks": [base], "win": win})
        parts.append(_tag(b, win) if tagged else b)
    return _join(parts)


def ranking_blocks(rk: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """The serving-side block, in `rk`'s row order.  Same names, same order."""
    parts = []
    for name in spec["blocks"]:
        base, win, tagged = parse(name, spec["win"])
        if base == "arrk":
            b = Ak.for_ranking(rk, win)
        else:
            b = _RANKING_BLOCKS(rk, {"blocks": [base], "win": win})
        parts.append(_tag(b, win) if tagged else b)
    return _join(parts)


def _join(parts: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    dup = [c for c in out.columns[out.columns.duplicated()]]
    if dup:
        raise SystemExit(f"the block has duplicate column names: {sorted(set(dup))}")
    return out
