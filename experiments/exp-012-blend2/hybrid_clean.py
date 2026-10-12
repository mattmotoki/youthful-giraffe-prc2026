"""Clean-population score of the hybrid rule, in the desk's currency.

`make_candidate_hybrid.py` reports all-rows RMSE because that is what the board
did to us before the calibration existed. Since the 2026-09-06 board read the
predictor is CLEAN board + 4 s, so the candidate's own score has to be quoted
that way too. Same rotation, same weights, same regime split; only the metric
changes, and both are printed.
"""
import os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
for p in (str(HERE.parents[1]), str(HERE)):
    sys.path.insert(0, p)
import blend as BL, nmtail_regime as NG, nmtail_rotate as NR, prc.blend as B, prc.data as D

TARGET, OUTER = "nm_late", os.environ.get("OUTER", "nnls")
data = {f: NG.enriched(f, "phi") for f in BL.FOLDS}
names = data[BL.FOLDS[0]]["names"]; k = len(names)
sel = {f: (data[f]["regime"] == TARGET) & data[f]["ok"] for f in BL.FOLDS}
allr, cln = {}, {}
for f in BL.FOLDS:
    src = [g for g in BL.FOLDS if g != f]
    wl = B.fit_weights(np.concatenate([data[g]["P"][sel[g]] for g in src]),
                       np.concatenate([data[g]["y"][sel[g]] for g in src]), "nnls")
    wo = (np.full(k - 1, 1.0 / (k - 1)) if OUTER == "flat" else
          B.fit_weights(np.concatenate([data[g]["P"][~sel[g]][:, :k-1] for g in src]),
                        np.concatenate([data[g]["y"][~sel[g]] for g in src]), OUTER))
    d, s = data[f], sel[f]
    p = np.empty(len(d["y"])); p[s] = d["P"][s] @ wl; p[~s] = d["P"][~s][:, :k-1] @ wo
    p = np.maximum(p, B.FLOOR)
    allr[f] = float(np.sqrt(((d["y"] - p) ** 2).mean()))
    # `enriched` carries no ids; the row order is the one NR.load_one builds the
    # frame on, so the ids come from the same call rather than from a re-read that
    # might order differently.
    ids = NR.load_one(NR.CORE["L2-hub"], f, "L2-hub")["MVT_ID_mvt"].to_numpy()
    assert len(ids) == len(d["y"]), f"{f}: {len(ids)} ids for {len(d['y'])} rows"
    cln[f] = float(D.rmse_clean(d["y"], p, ids, f))
BOARD = {"july_loo": 0.5571, "january": 0.4429}
ba = float(np.sqrt(sum(BOARD[f] * allr[f] ** 2 for f in BOARD)))
bc = float(np.sqrt(sum(BOARD[f] * cln[f] ** 2 for f in BOARD)))
print(f"  outer={OUTER}  " + "  ".join(f"{f} {allr[f]:.2f}/{cln[f]:.2f}" for f in BL.FOLDS))
print(f"  board all {ba:.3f}   board clean {bc:.3f}   PREDICT {bc+4:.2f}")
