"""WHY the NM-tail head is worth 7.48 s against R0 and nothing against the core.

exp-010 measured the head against two single base models and found seven gains
in seven comparisons. Against exp-012's four-member core the honestly rotated
value is -0.015 s. Both are true, and this is the diagnosis: the head's marginal
value as a function of how much ensemble it is being added to.

The curve is measured IN SAMPLE at every core size, which is an upper bound at
every point and therefore comparable across points; the rotated number decides
the shipping question and is reported separately in `nmtail_rotate.py`.
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
import numpy as np  # noqa: E402
import prc.blend as B  # noqa: E402
import nmtail_rotate as NR  # noqa: E402

# R0 is exp-010's base; it is not in exp-012's core but it anchors their number.
LADDER = [
    ("R0 alone (exp-010's base)", ["R0-retro"]),
    ("T-tailv2 alone", ["T-tailv2"]),
    ("+ L2-hub", ["T-tailv2", "L2-hub"]),
    ("+ C2-cat", ["T-tailv2", "L2-hub", "C2-cat"]),
    ("+ D1-winter (the core)", ["T-tailv2", "L2-hub", "C2-cat", "D1-winter"]),
]


def main(fold="july_loo", head_mode="phi"):
    import pandas as pd, json
    d = NR.fold_data(fold, head_mode)
    names = d["names"]
    # add R0 as an extra column
    r0 = NR.load_one("experiments/exp-012-blend2/oof/R0-retro-{fold}.parquet", fold, "R0-retro")
    base = NR.load_one(NR.CORE["L2-hub"], fold, "L2-hub")[["MVT_ID_mvt"]]
    j = base.merge(r0[["MVT_ID_mvt", "R0-retro"]], on="MVT_ID_mvt", how="left")
    lo, hi = json.loads((NR.E8 / "guard_bounds.json").read_text())[fold]
    a = NR.load_one(NR.CORE["L2-hub"], fold, "L2-hub")["d_takeoff_aobt"].to_numpy("float64")
    r0v = B.apply_residual_guard(np.maximum(j["R0-retro"].to_numpy("float64"), B.FLOOR),
                                 a, float(lo), float(hi), B.FLOOR)
    P, y, ok = d["P"], d["y"], d["ok"]
    col = {n: P[:, i] for i, n in enumerate(names)}
    col["R0-retro"] = r0v
    h = col["N1-nmtail"]
    print(f"=== {fold}, head mode {head_mode}, in-sample (an upper bound at every row)")
    print(f"{'core':>28} {'core rmse':>10} {'+N1':>9} {'gain':>7} {'w(N1)':>7} {'corr':>6}")
    for label, members in LADDER:
        M = np.stack([col[n] for n in members], axis=1)
        w = B.fit_weights(M[ok], y[ok], "nnls")
        p0 = M[ok] @ w
        M2 = np.concatenate([M[ok], h[ok, None]], axis=1)
        w2 = B.fit_weights(M2, y[ok], "nnls")
        p1 = M2 @ w2
        r = lambda v: float(np.sqrt(((y[ok] - v) ** 2).mean()))
        c = float(np.corrcoef(y[ok] - p0, y[ok] - h[ok])[0, 1])
        print(f"{label:>28} {r(p0):10.2f} {r(p1):9.2f} {r(p0)-r(p1):7.3f} {w2[-1]:7.3f} {c:6.3f}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
