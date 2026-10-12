"""Arm 3 priced in exp-030 arm 1's own form: no EHAM mask on either side.

RESULTS §5 compares "what a second window width is worth beside K2" against
exp-030's "what it is worth beside F1-fresh".  exp-030 arm 1 and its base v6 both
carry **every** airport; arm 3 and K2 both hold EHAM at stack-5.  A comparison
across those two populations is not a comparison, so this file re-prices arm 3
from the `y_pred_nomask` column of both rotations, which is the same construction
exp-030 measured.

    python nomask.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import bases as Bs  # noqa: E402

FOLDS = Bs.FOLDS
#: exp-030 RESULTS §2: arm 1 (600 + 300 s) against v6, both with every airport.
#: Quoted from that file rather than recomputed here, and gated against its own
#: committed record so it cannot drift (Sol round 1, non-blocking 5).
EXP030_ARM1 = 0.0734
EXP030_ARM1_REC = (Bs.ROOT / "experiments/exp-030-window2/analysis/arm-w600w300.json")
SIDES = {"k2_600_nomask": Bs.E27 / "oof" / "exp-027-k2-k2-hold-{fold}.parquet",
         "k2_w300_nomask": EXP / "oof" / "exp-036-k2-w300-{fold}.parquet"}


def main() -> None:
    ref, _, _ = Bs.load()
    out = {"note": "arm 3 in exp-030 arm 1's own form: the EHAM mask is off on both "
                   "sides, so the two experiments' deltas are on the same population."}
    board = {}
    for name, path in SIDES.items():
        # Sol round 1, non-blocking 5: this read `y_pred_nomask` positionally.
        # Both rotations are in the reference's row order, but a comparison that
        # assumes it rather than checking it is one file move away from being
        # silently wrong.
        v = {}
        for f in FOLDS:
            o = pd.read_parquet(str(path).format(fold=f))
            if not np.array_equal(o["MVT_ID_mvt"].to_numpy("float64"),
                                  ref[f]["MVT_ID_mvt"].to_numpy("float64")):
                raise SystemExit(f"{f}: {name} is not in the reference's row order")
            v[f] = o["y_pred_nomask"].to_numpy("float64")
        c = {f: Bs.D.rmse_clean(ref[f]["y_true"], v[f], ref[f]["MVT_ID_mvt"], f)
             for f in FOLDS}
        board[name] = Bs.Sc.P.board(c)
        out[name] = {f: round(c[f], 4) for f in FOLDS}
        out[f"{name}_board"] = round(board[name], 4)
        out[f"{name}_clean"] = c
    out["gain_board"] = round(board["k2_600_nomask"] - board["k2_w300_nomask"], 4)
    out["gain_per_fold"] = {f: round(out["k2_600_nomask_clean"][f]
                                     - out["k2_w300_nomask_clean"][f], 4) for f in FOLDS}
    for k in list(out):
        if k.endswith("_clean"):
            del out[k]
    got = json.loads(EXP030_ARM1_REC.read_text())["gain_board_vs_v6"]
    if abs(got - EXP030_ARM1) > 1e-9:
        raise SystemExit(f"exp-030 arm 1 records {got}, not the quoted {EXP030_ARM1}")
    out["exp030_arm1_gain_board_vs_v6"] = EXP030_ARM1
    out["exp030_arm1_record"] = str(EXP030_ARM1_REC.relative_to(Bs.ROOT))
    p = EXP / "analysis" / "arm-k2-w300-nomask.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n  written: {p}")


if __name__ == "__main__":
    main()
