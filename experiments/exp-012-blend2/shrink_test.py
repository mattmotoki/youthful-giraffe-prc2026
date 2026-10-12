"""Is the best blend between flat and fitted, and can the mix be chosen honestly?

With three members, rotated NNLS (318.81) beats a flat average (318.98). Add a
fourth member of comparable strength and it inverts: flat 318.21, NNLS 318.61.
So the fitted weights are overfitting the fit months once the members are
equally good -- D1-winter's fitted weight swings 0.088 to 0.268 depending on
which two months are in the fit set.

The natural question is whether a convex mix w(L) = L*flat + (1-L)*NNLS beats
both. Choosing L by looking at the rotated board estimate would be selection on
the scored folds, which is the bias exp-008's SPEC already had to disclose. So L
is chosen INSIDE the rotation instead: for held-out fold f, the other two folds
are the fit set, and L is picked by fitting NNLS on one of them and scoring the
other, both directions averaged. No fold's own labels ever touch the L applied
to it. The inner rotation is thin -- one month fits, one month scores -- and
that is stated rather than hidden.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parents[1]), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import blend as BL          # exp-012's own
import prc.blend as B

LAMBDAS = np.array([float(x) for x in __import__("os").environ.get("LAM","0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0").split(",")])


def main(cfg):
    data, names = BL.load(BL.json.loads(Path(cfg).read_text())["members"])
    print("members:", names)
    P = {f: data[f][names].to_numpy("float64") for f in BL.FOLDS}
    Y = {f: data[f]["y_true"].to_numpy("float64") for f in BL.FOLDS}
    rows = {}
    for held in BL.FOLDS:
        src = [f for f in BL.FOLDS if f != held]
        # --- inner rotation: choose L using only the two fit folds
        scores = np.zeros(len(LAMBDAS))
        for a, b in ((0, 1), (1, 0)):
            Pa, ya = P[src[a]], Y[src[a]]
            Pb, yb = P[src[b]], Y[src[b]]
            w_fit = B.fit_weights(Pa, ya, "nnls")
            w_flat = np.full(len(names), 1.0 / len(names))
            for i, L in enumerate(LAMBDAS):
                w = L * w_flat + (1 - L) * w_fit
                scores[i] += np.sqrt(((yb - np.maximum(Pb @ w, B.FLOOR)) ** 2).mean())
        L_star = float(LAMBDAS[int(np.argmin(scores))])
        # --- apply: fit on both src folds, shrink by the chosen L, score held
        Pfit = np.concatenate([P[f] for f in src])
        yfit = np.concatenate([Y[f] for f in src])
        w_fit = B.fit_weights(Pfit, yfit, "nnls")
        w_flat = np.full(len(names), 1.0 / len(names))
        Ph, yh = P[held], Y[held]
        out = {}
        for lab, w in (("nnls", w_fit), ("flat", w_flat),
                       ("shrunk", L_star * w_flat + (1 - L_star) * w_fit)):
            # the floor is part of the estimator, as in blend.rotate
            out[lab] = float(np.sqrt(((yh - np.maximum(Ph @ w, B.FLOOR)) ** 2).mean()))
        rows[held] = (L_star, out)
        print(f"{held:>9}  L*={L_star:.1f}  nnls={out['nnls']:7.3f} "
              f"flat={out['flat']:7.3f} shrunk={out['shrunk']:7.3f}")
    print()
    for lab in ("nnls", "flat", "shrunk"):
        est = BL.board_estimate({f: rows[f][1][lab] for f in BL.SHARES})
        print(f"  board estimate, {lab:>6}: {est:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
