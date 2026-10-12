"""Arm 3: EHAM out of the served mask, on the v6 form.

exp-028 §5 measured the `depx` block by airport and found EHAM losing on **all
three folds** (−0.097 / −0.086 / −0.178 s of fold RMSE) — the one airport
`docs/shift-2026.md` puts outside its own 2025 regime in January 2026, and the
one with the lowest NM-join rate in the data.  exp-028 left the exclusion to the
desk because it is a fourth cut on the same three folds; `assignments/window2.md`
takes it as arm 3.

**Nothing is refitted.**  The arm is arithmetic on the saved rotation: every EHAM
departure keeps stack-5's own value and every other row keeps v6's.  On the
serving side the same statement is arithmetic on two candidate files that are
already verified and uploaded — `exp-028-window.parquet` (v6) and
`exp-026-stack-5.parquet` — which is why this arm needs no stacker run and no new
serving path.

    python eham.py                 # on v6, which is arm 3
    python eham.py --on <arm>      # the same exclusion on another rotated arm
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

AIRPORT = "EHAM"


def block_side(ref: dict, on: str | None) -> tuple[dict, str]:
    """The vector the exclusion is applied to: v6's, or a rotated arm's."""
    if not on:
        return {f: ref[f]["y_pred_v6"].to_numpy("float64") for f in Sc.FOLDS}, "v6"
    out = {}
    for f in Sc.FOLDS:
        o = pd.read_parquet(EXP / "oof" / f"exp-028-stack5w-{on}-{f}.parquet")
        if not np.array_equal(o["MVT_ID_mvt"].to_numpy("float64"),
                              ref[f]["MVT_ID_mvt"].to_numpy("float64")):
            raise SystemExit(f"{f}: {on} is not in the reference's row order")
        out[f] = o["y_pred"].to_numpy("float64")
    return out, on


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", default=None, help="the rotated arm to exclude EHAM from")
    ap.add_argument("--name", default=None)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--win", type=float, default=600.0)
    a = ap.parse_args()

    ref = Sc.reference()
    side, label = block_side(ref, a.on)
    preds, n = {}, {}
    for f in Sc.FOLDS:
        d = ref[f]
        out = d["airport"].astype(str).to_numpy() == AIRPORT
        preds[f] = np.where(out, d["y_pred_stack5"].to_numpy("float64"), side[f])
        n[f] = int(out.sum())
    rec = Sc.score(a.name or (f"{label}-eham-out" if a.on else "eham-out"), preds, ref,
                   note=f"{label} with {AIRPORT} served from stack-5; rows taken back "
                        + ", ".join(f"{f} {n[f]}" for f in Sc.FOLDS)
                        + ". No refit: arithmetic on the saved rotation.")
    rec["airport_excluded"] = AIRPORT
    rec["applied_to"] = label
    if not a.on:
        # the serving contract for a derived arm, pinned by SHA-256 (Sol round 1,
        # blocker 1).  `serve.py` refuses any --from / --base / --regimes that is
        # not exactly this, so an arm's score cannot be attached to other bytes.
        rec["serve"] = {"mode": "derived",
                        "from": Sc.CANDIDATES["v6"][0],
                        "from_sha256": Sc.CANDIDATES["v6"][1],
                        "base": Sc.CANDIDATES["stack-5"][0],
                        "base_sha256": Sc.CANDIDATES["stack-5"][1],
                        "regimes": list(Sc.REGIMES),
                        "exclude_airports": [AIRPORT]}
    rec["n_rows_taken_from_stack5"] = n
    if a.blocks:
        rec["blocks"] = a.blocks.split("+")
        rec["win"] = a.win
    Sc.show(rec)
    print(f"\n  written: {Sc.save(rec)}")


if __name__ == "__main__":
    main()
