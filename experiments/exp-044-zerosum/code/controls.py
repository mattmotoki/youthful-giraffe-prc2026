"""Two controls for the dayblock-to-xmonth collapse found in step 2.

Every arm's within-month number (Q03's design) is far above its cross-month one,
and the gap grows with capacity.  Two readings fit that, and only one of them is
about the world:

* the CROSS-MONTH fit is genuinely harder, and Q03's within-month design is
  optimistic because the fitting days share a month with the scored days;
* something in the cross-month IMPLEMENTATION costs the gain -- the level list
  frozen on the fitting months, or the pooling of two months' cells.

`dayblock_frozen` is the implementation control: Q03's within-month day-block
cross-fitting, unchanged, with the categorical level list frozen on the OTHER
two months -- the serving vocabulary.  If it reads near the within-month number,
the vocabulary is not what costs the gain.

`halfmonth` puts the same question on a time axis: fit on one contiguous half of
the month's days and score the other half, both ways.  Same month, same
vocabulary, same learner -- only the distance in time between the fitting rows
and the scored rows grows.

    OMP_NUM_THREADS=3 python controls.py
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
import zs                                   # noqa: E402
import learner as L                         # noqa: E402

EXP = zs.EXP
FOLDS = zs.FOLDS
ARMS = ["ridge", "lgb_shipped", "lgb_shipped_wide"]


def half_index(days: np.ndarray) -> np.ndarray:
    u = np.unique(days)
    first = set(u[:int(np.ceil(len(u) / 2))])
    return np.array([0 if d in first else 1 for d in days])


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(zs.SEED)
    pan = zs.V.load()
    S = {f: L.fold_state(f, pan[f]) for f in FOLDS}
    amps = json.load(open(EXP / "analysis" / "learner.json"))["amplitudes"]

    H = {f: {} for f in FOLDS}
    for f in FOLDS:
        Sf = S[f]
        ids, ei = Sf["ids"], np.flatnonzero(Sf["elig"])
        others = [g for g in FOLDS if g != f]
        frozen = zs.levels_of([zs.dep(S[g]["month"]) for g in others])
        local = zs.levels_of([zs.dep(Sf["month"])])
        hi = half_index(Sf["days"])
        for shipped in (False, True):
            arms = [a for a in ARMS if (a in L.SHIPPED) == shipped]
            if not arms:
                continue
            Xfr, _ = L.month_matrix(Sf["month"], frozen, shipped, ids)
            Xlo, _ = L.month_matrix(Sf["month"], local, shipped, ids)
            for arm in arms:
                base = L.BASE[arm]
                h = np.zeros(len(ids))
                for b in range(zs.BLOCKS):
                    fit, sc = ei[Sf["blk"][ei] != b], ei[Sf["blk"][ei] == b]
                    h[sc] = L.fit_h(base, Xfr[fit], Sf["r"][fit], Sf["code"][fit], Xfr[sc])
                H[f][f"{arm}|dayblock_frozen"] = h
                h2 = np.zeros(len(ids))
                for side in (0, 1):
                    fit, sc = ei[hi[ei] != side], ei[hi[ei] == side]
                    h2[sc] = L.fit_h(base, Xlo[fit], Sf["r"][fit], Sf["code"][fit], Xlo[sc])
                H[f][f"{arm}|halfmonth"] = h2
                print(f"    [{f}] {arm:18s} controls {time.time() - t0:6.1f}s", flush=True)
            del Xfr, Xlo

    recs, boards = {}, {}
    for arm in ARMS:
        for ctl in ("dayblock_frozen", "halfmonth"):
            key = f"{arm}|{ctl}"
            recs[key] = {}
            for f in FOLDS:
                m0 = S[f]["m0"]
                cm, ucm = pd.factorize(S[f]["code"][m0])
                d0 = np.zeros(len(m0))
                d0[m0] = zs.project(H[f][key][m0], cm, len(ucm))
                a = amps[arm][f]
                recs[key][f] = L.score(S[f], a * d0, H[f][key], a, rng, True)
            ex = lambda k: {f: recs[key][f]["rmse_exact"][k] for f in FOLDS}
            av, an = zs.V.board(ex("all_v10")), zs.V.board(ex("all_new"))
            boards[key] = {"all_row_v10": round(av, 4), "all_row_arm": round(an, 4),
                           "gain_all_row": round(av - an, 4)}
            print(f"  {key:34s} board {av - an:+.4f} s  folds "
                  + " ".join(f"{recs[key][f]['gain_all']:+.4f}" for f in FOLDS)
                  + "  2A-B " + " ".join(
                      f"{recs[key][f]['mse_gain_s2']:+.1f}[{recs[key][f]['ci'][0]:+.0f},"
                      f"{recs[key][f]['ci'][1]:+.0f}]" for f in FOLDS), flush=True)

    out = {"step": "2 controls", "arms": ARMS, "folds": recs, "board": boards,
           "wall_s": round(time.time() - t0, 1)}
    (EXP / "analysis" / "controls.json").write_text(json.dumps(out, indent=2))
    # Sol round 2, blocking 4: kept so code/records.py can slice every arm.
    np.savez_compressed(EXP / "analysis" / "h_controls.npz",
                        **{f"{f}|{k}": H[f][k].astype("float32")
                           for f in FOLDS for k in H[f]})
    print(f"  written {EXP / 'analysis' / 'controls.json'} ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
