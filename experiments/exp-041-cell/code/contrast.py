"""The PAIRED contrast on the PRIMARY endpoint, arm against arm.

Sol round 2, blocking finding 3: `order.py` and `twofactor.py` bootstrap each arm
against v10 on the ranking-weighted board, and every arm-against-arm contrast
against a per-fold, eligible-row endpoint.  The advance clauses in
`assignments/cell.md` are written on the pair -- "ordered beats unordered", "the
two-factor arm beats the one-factor arm" -- so the interval that decides them has
to be on the same primary endpoint: the ranking-weighted all-row january/july
board, with 1/2/3-day moving blocks drawn inside each fold.

This screen fits nothing.  It reads the saved out-of-fold vectors of
`oof/exp-041-*.parquet`, which carry `y_true` and `y_pred` for every row of every
fold, and re-uses `base.board_boot` with the two arms in place of (v10, arm), so
the contrast is reported as `gain_over_v10(A) - gain_over_v10(B)`, so a
positive number means A is the better arm.

    python contrast.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import base as B  # noqa: E402

EXP = B.EXP

#: every pair an advance clause or a stated sentence in RESULTS rests on
PAIRS = [
    # step 2 -- the assignment's "ordered beats unordered at equal capacity"
    ("order-ord", "order-unord"),
    ("order-ord", "order-unord2"),
    ("order-ord", "order-ord_c"),
    ("order-ord", "order-h0"),
    ("order-unord", "order-h0"),
    ("order-unord2", "order-h0"),
    # step 2's second half -- "the pair beats v10" is the arm-vs-v10 interval
    # already in order.json; here the arms are compared with each other.
    # step 3 -- "two factors beat one at matched capacity"
    ("twofactor-two", "twofactor-one"),
    ("twofactor-two", "twofactor-g0"),
    ("twofactor-two", "twofactor-two_c"),
    ("twofactor-one", "twofactor-g0"),
    ("twofactor-two_q04", "twofactor-one_q04"),
    ("twofactor-two_q04", "twofactor-q04h1"),
    ("twofactor-one_q04", "twofactor-q04h1"),
    # the control
    ("crossmonth-base_within", "crossmonth-base_cross"),
    ("crossmonth-two_within", "crossmonth-two_cross"),
]


def load(name: str) -> pd.DataFrame:
    p = EXP / "oof" / f"exp-041-{name}.parquet"
    return pd.read_parquet(p, columns=["fold", "date", "y_true", "y_pred"])


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(B.SEED)
    cache: dict[str, pd.DataFrame] = {}
    for a, b in PAIRS:
        for n in (a, b):
            if n not in cache:
                cache[n] = load(n)
    out = {}
    for a, b in PAIRS:
        da, db = cache[a], cache[b]
        assert len(da) == len(db), (a, b)
        parts = {}
        for f in da["fold"].unique():
            ja = da["fold"] == f
            jb = db["fold"] == f
            y = da.loc[ja, "y_true"].to_numpy("float64")
            assert np.array_equal(y, db.loc[jb, "y_true"].to_numpy("float64"))
            sa = (y - da.loc[ja, "y_pred"].to_numpy("float64")) ** 2
            sb = (y - db.loc[jb, "y_pred"].to_numpy("float64")) ** 2
            parts[str(f)] = (da.loc[ja, "date"].to_numpy(), sa, sb)
        r = B.board_boot(parts, rng)
        #  `board_boot` returns `board(A) - board(B)` in RMSE, where a SMALLER
        #  board is the better arm.  Everything else in this experiment reports
        #  gains over v10, where bigger is better, so the sign is flipped here:
        #  `gain_a_minus_b` is `gain_over_v10(A) - gain_over_v10(B)`, positive
        #  when A is the better arm.  The interval is flipped with it.
        def flip(c):
            return [round(-c[1], 4), round(-c[0], 4)]
        out[f"{a} - {b}"] = {
            "board_rmse_a": r["board_v10"], "board_rmse_b": r["board_arm"],
            "gain_a_minus_b": round(-r["gain"], 4),
            "ci_block_1d": flip(r["ci_block_1d"]),
            "ci_block_2d": flip(r["ci_block_2d"]),
            "ci_block_3d": flip(r["ci_block_3d"]),
            "clear_of_zero_3d": bool(r["ci_block_3d"][0] > 0
                                     or r["ci_block_3d"][1] < 0),
            "note": "positive: A is the better arm on the primary endpoint"}
        print(f"  {a} - {b}: {-r['gain']:+.4f} s  3d "
              f"{flip(r['ci_block_3d'])}", flush=True)
    rec = {"screen": "contrast",
           "endpoint": "ranking-weighted ALL-ROW january/july board; "
                       "gain = board(A) - board(B), positive means A is better",
           "note": "computed from the saved out-of-fold vectors; nothing is "
                   "fitted here.  Blocks are drawn inside each fold.",
           "pairs": out, "wall_s": round(time.time() - t0, 1)}
    p = EXP / "analysis" / "contrasts.json"
    p.write_text(json.dumps(rec, indent=2))
    print(f"\n  written: {p}  ({rec['wall_s']} s)")


if __name__ == "__main__":
    main()
