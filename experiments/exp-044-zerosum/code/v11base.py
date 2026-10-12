"""A secondary reading the desk needs: the same update on top of v11, not v10.

The assignment fixes v10 as the base.  The board moved to **v11**
(`v10 + 0.217*(stack-5 - v10)`, 263.0298, BOARD READING 10) before this
experiment ran, so a candidate built on v10 starts 0.0166 s behind the file the
board already holds.  This prices the same update on the v11 residual, on the
fold side only, so the desk can see whether the two are additive.  Nothing here
is refitted: `d` is the ridge's update at the served amplitude, and only the
base it is added to changes.

    OMP_NUM_THREADS=3 python v11base.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zs                                   # noqa: E402
import learner as L                         # noqa: E402

EXP = zs.EXP
FOLDS = zs.FOLDS
S5_AMP = 0.217          # exp-038 section 2c, the scalar v11 was served at


def main() -> None:
    pan = zs.V.load()
    S = {f: L.fold_state(f, pan[f]) for f in FOLDS}
    amps = json.load(open(EXP / "analysis" / "learner.json"))["amplitudes"]
    a_pooled = amps["ridge"]["pooled"]
    H = np.load(EXP / "analysis" / "h_vectors.npz")
    _, vec, _ = zs.V.Bs.load()
    rng = np.random.default_rng(zs.SEED)

    out = {"served_amplitude_pooled": round(a_pooled, 4), "stack5_amplitude": S5_AMP,
           "note": "Sol round 1, blocking 4: the first version scored every fold at the "
                   "POOLED amplitude, which is fitted on all three folds including the one "
                   "it scores. The headline is now the per-fold CROSS-MONTH amplitude, "
                   "fitted on the other two folds only, exactly as the v10 table is; the "
                   "pooled row is kept beside it and labelled, because it is the scalar the "
                   "served file carries.",
           "arms": {}}
    for cf in ("dayblock", "xmonth"):
      for amp_kind in ("cross_month", "pooled"):
          rec, ex = {}, {}
          for f in FOLDS:
              a = amps["ridge"][f] if amp_kind == "cross_month" else a_pooled
              Sf = S[f]
              m0 = Sf["m0"]
              cm, ucm = pd.factorize(Sf["code"][m0])
              d = np.zeros(len(m0))
              d[m0] = a * zs.project(H[f"{f}|ridge|{cf}"][m0].astype("float64"),
                                     cm, len(ucm))
              p11 = Sf["p10"] + S5_AMP * (vec["stack-5"][f] - Sf["p10"])
              r11 = Sf["y"] - p11
              v = 2 * r11[m0] * d[m0] - d[m0] ** 2
              pt, lo, hi = zs.boot_days(v, np.ones(int(m0.sum())), Sf["days"][m0], rng)
              ex[f] = {"all_v11": float(zs.D.rmse(Sf["y"], p11)),
                       "all_new": float(zs.D.rmse(Sf["y"], p11 + d)),
                       "all_v10": float(zs.D.rmse(Sf["y"], Sf["p10"]))}
              rec[f] = {"amplitude": round(a, 4),
                        "mse_gain_s2": round(pt, 3), "ci": [round(lo, 3), round(hi, 3)],
                        "gain_all_over_v11": round(ex[f]["all_v11"] - ex[f]["all_new"], 4),
                        "v11_minus_v10_all": round(ex[f]["all_v10"] - ex[f]["all_v11"], 4)}
          b11 = zs.V.board({f: ex[f]["all_v11"] for f in FOLDS})
          bnew = zs.V.board({f: ex[f]["all_new"] for f in FOLDS})
          b10 = zs.V.board({f: ex[f]["all_v10"] for f in FOLDS})
          out["arms"][f"{cf}|{amp_kind}"] = {"folds": rec,
                             "board_v10": round(b10, 4), "board_v11": round(b11, 4),
                             "board_v11_plus_d": round(bnew, 4),
                             "gain_over_v11": round(b11 - bnew, 4),
                             "gain_over_v10_total": round(b10 - bnew, 4)}
          print(f"  ridge|{cf}|{amp_kind:11s} v10 {b10:.4f}  v11 {b11:.4f}  v11+d {bnew:.4f}  "
                f"(+{b11 - bnew:.4f} over v11, +{b10 - bnew:.4f} over v10)  folds "
                + " ".join(f"{rec[f]['gain_all_over_v11']:+.4f}" for f in FOLDS)
                + "  2A-B " + " ".join(
                    f"{rec[f]['mse_gain_s2']:+.1f}[{rec[f]['ci'][0]:+.0f},{rec[f]['ci'][1]:+.0f}]"
                    for f in FOLDS))
    (EXP / "analysis" / "v11base.json").write_text(json.dumps(out, indent=2))
    print(f"  written {EXP / 'analysis' / 'v11base.json'}")


if __name__ == "__main__":
    main()
