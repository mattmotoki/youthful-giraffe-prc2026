"""Compare two member sets that differ in ONE member, by the blend criterion.

`screen_by_blend.py` answers "what does ADDING this candidate buy?". The Huber
budget question is different: the member set is fixed and one member is REPLACED
by a variant of itself. Adding and swapping are not the same measurement — a swap
cannot buy diversity by widening the set, it can only trade one member's errors
for another's, so the gain has to come from where those errors sit.

Per fold, weights are fitted only on the OTHER folds (the same rotation
`screen_by_blend.py` uses), so no fold pays for its own weights.

  python screen_swap.py <base.json> <variant.json> [mode ...]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import blend as BL  # noqa: E402
import prc.blend as B  # noqa: E402

FOLDS = ("july_loo", "december", "january")
# The board estimate is a quadrature of july_loo and january; december is carried
# because a three-fold rotation is what the desk's budget rule asks for.
BOARD = {"july_loo": 0.5571, "january": 0.4429}


def rotated(files: dict, mode: str) -> dict:
    data, names = BL.load(files, folds=FOLDS)
    out = {}
    for f in FOLDS:
        fit = pd.concat([data[g] for g in FOLDS if g != f], ignore_index=True)
        w = B.fit_weights(fit[names].to_numpy("float64"),
                          fit["y_true"].to_numpy("float64"), mode)
        d = data[f]
        p = np.maximum(d[names].to_numpy("float64") @ w, B.FLOOR)
        out[f] = float(np.sqrt(((d["y_true"].to_numpy("float64") - p) ** 2).mean()))
        out[f"w_{f}"] = dict(zip(names, np.round(w, 4)))
    out["board"] = float(np.sqrt(sum(BOARD[f] * out[f] ** 2 for f in BOARD)))
    return out


def main(base_cfg: str, var_cfg: str, *modes: str) -> int:
    import provenance
    print(provenance.stamp())
    base = json.loads((HERE / base_cfg).read_text())["members"]
    var = json.loads((HERE / var_cfg).read_text())["members"]
    changed = [k for k in base if base.get(k) != var.get(k)]
    added, dropped = set(var) - set(base), set(base) - set(var)
    if added or dropped or len(changed) != 1:
        print(f"  not a one-member swap: changed={changed} added={added} dropped={dropped}")
        return 2
    m = changed[0]
    print(f"  swapping {m}\n    from {base[m]}\n    to   {var[m]}\n")
    for mode in (modes or ("nnls", "flat")):
        a, b = rotated(base, mode), rotated(var, mode)
        print(f"  [{mode}]  {'fold':>10} {'base':>9} {'variant':>9} {'delta':>8}")
        for f in FOLDS:
            print(f"  {'':6} {f:>10} {a[f]:9.3f} {b[f]:9.3f} {b[f]-a[f]:+8.3f}")
        print(f"  {'':6} {'board':>10} {a['board']:9.3f} {b['board']:9.3f} "
              f"{b['board']-a['board']:+8.3f}   "
              f"({'variant better' if b['board'] < a['board'] else 'base better'})")
        wa = {f: a[f'w_{f}'].get(m) for f in FOLDS}
        wb = {f: b[f'w_{f}'].get(m) for f in FOLDS}
        print(f"  {'':6} weight on {m}: base {wa}  variant {wb}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
