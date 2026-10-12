"""The whole WITHOUT the rung: the corrected class and the LFPG rider, over v18's bytes.

`code/rung511.py` read A -- the 511 rung alone on the shipped class -- at **-0.06281
[-0.12236, -0.00916]**, clear below zero on july_loo. The ladder turned over, so the
candidate section 33 composed (rung + correction + rider) carries a negative part, and the
composition the desk should see beside it is the same candidate with the rung dropped:

    E = 255 leaves, CORRECTED class, LFPG RELEASED

**This costs no fit.** `rung511.fit_free` already saved that exact vector as
`work/d-bundle-km1299-255-free.npz` -- the fit for price B, before the airport hold was
applied -- so E is a re-price of a vector that exists, and the rider inside it is the same
fit as its own held twin and therefore carries no refit noise.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/fallbackwhole.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]

import arms as A                                                      # noqa: E402
import chainfold as CF                                                # noqa: E402
import planekill as PK                                                # noqa: E402
import rung511 as R                                                   # noqa: E402

OUT = EXP / "analysis/fallback-whole.json"


def main() -> None:
    t0 = time.time()
    if R.sha256(R.HEAD) != R.HEAD_SHA:
        raise SystemExit(f"{R.HEAD.name} is not the vector v18's bytes ship")
    rule = A.serving_rule()[0]
    blk_km = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        k = PK.build_km(m, rule)
        R.assert_mode3(k, m, "corrected")
        blk_km[m] = k
    print("mode-3 asserted on the corrected builds", flush=True)
    pan = CF.panel_for(blk_km, R.THR, {m: CF.GAP_CUT_S for m in blk_km})

    zh = np.load(R.HEAD)
    H = {f: zh[f].astype("float64") for f in A.FOLDS}
    zf = np.load(R.FREE["B"])
    E = {f: zf[f].astype("float64") for f in A.FOLDS}
    B = R.hold_lfpg(pan, E)

    out = {"what": __doc__.strip().splitlines()[0],
           "pre_registered_in": "RESULTS.md section 33; this composition is section 34's",
           "vector": str(R.FREE["B"].relative_to(ROOT)),
           "head": {"file": str(R.HEAD.relative_to(ROOT)), "sha256": R.HEAD_SHA},
           "no_fit": ("rung511.py saved this vector as the unheld half of price B; E is a "
                      "re-price, and its rider is the same fit as its held twin"),
           "prices": {}}
    for key, d_new, d_head, label in (
            ("E", E, H, "THE WHOLE WITHOUT THE RUNG: 255 leaves, corrected class, "
                        "LFPG released, over v18's bytes"),
            ("rider_on_B", E, B, "the LFPG rider alone, on top of the corrected class "
                                 "at 255 leaves")):
        rec = R.price(pan, d_new, d_head, label)
        if key == "rider_on_B":
            rec["lit_rule"] = {
                "lit_folds": [f for f in A.FOLDS if rec["rows_moved"][f] > 0],
                "dark_folds": rec["folds_with_no_moved_row"],
                "every_moving_fold_interval_clear":
                    rec["every_moving_fold_interval_clear"]}
        out["prices"][key] = rec
        print(f"\n[{key}] {label}", flush=True)
        print(R.row(rec), flush=True)
        print(R.fold_rows(rec), flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\nwritten to", OUT.relative_to(ROOT), f"({out['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
