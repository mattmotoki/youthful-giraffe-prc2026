"""The bundle's own whole-update fold gate, per fold, read only (round 7, finding 1).

`serve.fold_gate` returns the board-level reading and the per-fold verdict, not the
per-fold numbers, and the round-7 repair was ordered as a one-line change plus a re-emit --
not as a widening of the record. So the per-fold numbers §29.3 quotes are taken here, by
the same call `fold_gate` makes (`base.price` on the panel at this serve's pins, seed 46,
the shipped vector with LFPG zeroed), and written to their own address rather than asserted
in prose.

Nothing is fitted and no lane file is written except this file's own JSON.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/foldgatecheck.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402

HOLD = ("LFPG",)
THR = 1299.0
VECTORS = {
    "the_bundle_that_ships": EXP / "work/d-bundle-ungated1299-guard995.npz",
    "v18s_fit_on_the_shipped_block": EXP / "work/d-guard-leaves_255x300_lr050-guard995.npz",
}
OUT = EXP / "analysis/foldgate-byfold.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rule = A.serving_rule()[0]
    blocks = {A.FOLD_MONTH[f]: CF.build_month(A.FOLD_MONTH[f],
                                              A.prefix_months_for(A.FOLD_MONTH[f]), rule)
              for f in A.FOLDS}
    pan = CF.panel_for(blocks, THR, {m: CF.GAP_CUT_S for m in blocks})
    out = {"what": __doc__.strip().splitlines()[0], "hold": list(HOLD),
           "panel": "the v19 serve's own panel: the ungated blocks at the 28.2 gate",
           "seed": A.SEED, "readings": {}}
    for name, p in VECTORS.items():
        z = np.load(p)
        kept, zeroed = {}, {}
        for f in A.FOLDS:
            dv = z[f].astype("float64").copy()
            m = np.isin(pan[f].airport.to_numpy().astype(str), list(HOLD))
            zeroed[f] = int((dv[m] != 0).sum())
            dv[m] = 0.0
            kept[f] = dv
        pr = BA.price(pan, kept, np.random.default_rng(A.SEED))
        out["readings"][name] = {
            "vector": str(p.relative_to(ROOT)), "sha256": sha256(p),
            "board_all_row_gain_s": pr["board_all_row"]["gain_s"],
            "board_interval_ci95": pr["board_all_row"]["interval"]["ci95"],
            "board_clean_gain_s": pr["board_clean"]["gain_s"],
            "every_fold_interval_clear": bool(pr["every_fold_interval_clear"]),
            "lfpg_rows_zeroed_by_the_hold": zeroed,
            "by_fold": {f: pr["by_fold"][f]["all_row"] for f in A.FOLDS}}
        r = out["readings"][name]
        ci = r["board_interval_ci95"]
        print(f"{name:34s} {r['board_all_row_gain_s']:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}]  "
              f"clear {r['every_fold_interval_clear']}  zeroed {zeroed}", flush=True)
        for f in A.FOLDS:
            v = r["by_fold"][f]
            print(f"   {f:10s} {v['mse_gain_s2']:>+10.1f} "
                  f"[{v['ci95'][0]:>+9.1f},{v['ci95'][1]:>+9.1f}]  moved {v['rows_moved']}",
                  flush=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
