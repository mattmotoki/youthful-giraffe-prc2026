"""Score one rotated arm against v6 and write its record.

    python measure.py <arm> [--blocks depx+depx_w300] [--win 600] [--note "..."]

Reads the rotation `rotate.py` wrote (`oof/exp-028-stack5w-<arm>-{fold}.parquet`,
column `y_pred` — the arm's keep-noac vector) and hands it to `score.score`,
which is the one place an arm's numbers are computed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import score as Sc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--win", type=float, default=600.0)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    ref = Sc.reference()
    preds = {}
    for f in Sc.FOLDS:
        o = pd.read_parquet(EXP / "oof" / f"exp-028-stack5w-{a.arm}-{f}.parquet")
        if not np.array_equal(o["MVT_ID_mvt"].to_numpy("float64"),
                              ref[f]["MVT_ID_mvt"].to_numpy("float64")):
            raise SystemExit(f"{f}: the arm's rotation is not in the reference's row order")
        if float(np.abs(o["y_pred_stack5"].to_numpy("float64")
                        - ref[f]["y_pred_stack5"].to_numpy("float64")).max()) > 1e-9:
            raise SystemExit(f"{f}: the arm's no-window rotation is not exp-028's stack-5")
        preds[f] = o["y_pred"].to_numpy("float64")
    rec = Sc.score(a.arm, preds, ref, note=a.note)
    if a.blocks:
        rec["blocks"] = a.blocks.split("+")
        rec["win"] = a.win
    Sc.show(rec)
    print(f"\n  written: {Sc.save(rec)}")


if __name__ == "__main__":
    main()
