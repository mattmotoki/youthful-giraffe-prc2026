"""The two claims `RESULTS.md` §3 makes about the multi-width block, checked.

1. **A width-tagged arm strictly contains v6's block.** `blocks.block_for` with
   `["depx"]` at 600 s must equal `stack5w.block_for`'s own output value for
   value and name for name, and the first 50 columns of `["depx", "depx_w300"]`
   must be that same frame — so arm 1's delta against v6 is "what a second
   resolution adds" and not "what a different block does".
2. **The tagged part is the same builder at the other width**, with the five
   width-independent row-context columns dropped and nothing else changed.

Also checked, because it is the fit/serve property the whole family rests on:
the fold-side and serving-side builders return the **same column names in the
same order** for every spec this experiment measured. Values cannot be compared
across the two — they are different months — but a name or order mismatch is the
failure that silently feeds a stacker the wrong column, and `serve.py` raises on
it at serve time; this makes it a test rather than a runtime hope.

    python tests/block_identity.py          # needs data/raw for the serving side
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
import blocks as B2  # noqa: E402
import features as Fe  # noqa: E402
import panel as P  # noqa: E402
import stack5w as SW  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import frames as F  # noqa: E402
import regime_table as RT  # noqa: E402
import serve_stack5 as SS  # noqa: E402
import stack5 as S  # noqa: E402

FOLD = "july_loo"
SPECS = [{"blocks": ["depx", "depx_w300"], "win": 600.0},
         {"blocks": ["depx", "depx_w300", "depx_w1800"], "win": 600.0},
         {"blocks": ["depx", "arrk"], "win": 600.0}]


def equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return (list(a.columns) == list(b.columns)
            and np.array_equal(a.to_numpy("float64"), b.to_numpy("float64"), equal_nan=True))


def main() -> None:
    pan = SW.panel_of(P.load_fold(FOLD))
    v6 = SW.block_for(FOLD, pan, {"blocks": ["depx"], "win": 600.0})
    mine = B2.block_for(FOLD, pan, {"blocks": ["depx"], "win": 600.0})
    assert equal(v6, mine), "blocks.block_for changes exp-028's own 600 s block"
    print(f"[1] untagged depx is exp-028's own block, {v6.shape[1]} columns, values equal")

    two = B2.block_for(FOLD, pan, {"blocks": ["depx", "depx_w300"], "win": 600.0})
    assert equal(two.iloc[:, :v6.shape[1]], v6), "arm 1 does not contain v6's block"
    print(f"[1] arm 1's first {v6.shape[1]} columns are v6's block, values equal")

    raw300 = SW.block_for(FOLD, pan, {"blocks": ["depx"], "win": 300.0})
    keep = [c for c in raw300.columns if c not in Fe.ROW_ONLY]
    tagged = raw300[keep].add_suffix("_w300")
    assert equal(two.iloc[:, v6.shape[1]:], tagged), "the tagged part is not the 300 s builder"
    print(f"[2] arm 1's remaining {tagged.shape[1]} columns are the same builder at 300 s, "
          f"row context dropped")

    if not (ROOT / "data" / "raw" / "ranking.parquet").exists():
        print("[serve] data/raw absent; the name-and-order check is skipped")
        return
    if not RT.OUT.exists():
        RT.build()
    rk = SS.ranking_frame(S.B5, RT.load())
    data = {FOLD: F.fold_frame(FOLD, S.B5)}
    for spec in SPECS:
        a = B2.block_for(FOLD, data[FOLD], spec)
        b = B2.ranking_blocks(rk, spec)
        assert list(a.columns) == list(b.columns), f"{spec['blocks']}: names or order differ"
        assert len(b) == len(rk), f"{spec['blocks']}: {len(b)} rows against {len(rk)}"
        print(f"[serve] {'+'.join(spec['blocks']):28s} {a.shape[1]:3d} columns, "
              f"fold and ranking builders agree on names and order")
    print("\nblock_identity PASS")


if __name__ == "__main__":
    main()
