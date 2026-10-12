"""§11: the regime hold, applied to `C2-cat` WITHOUT being chosen again.

    PRC_THREADS=4 .venv/bin/python experiments/exp-040-chain/code/regime_test.py

§4c chose the subset `nm_bulk`+`nm_late` by reading `L2-hub8725`'s slice table on
these three folds, which is why §9b calls that result discovery evidence: the
subset choice was held out of the fold that scored it, the choice of a REGIME
FAMILY could not be.  `C2-cat` is the other retro-v2 member and rotates on its
own arms, so **the rule can be applied to it without being chosen again**, and
that is the cleanest test of the family this experiment can run.

**The rule is fixed here, in code, before any `C2-cat` fold is read**, and this
file is committed before the december and january arms land.  It takes no
arguments and enumerates nothing:

    serve the chain arm on `nm_bulk` and `nm_late`; hold `nonm` at v10;
    hold `noac` at v10 (the `d11-noac` variant, which is what a candidate ships)

The prediction, stated in the same commit: if the family is real rather than a
reading of `L2-hub8725`'s noise, `C2-cat`'s held arm gains over v10 on all three
folds and **december's interval clears zero** -- which is exactly what december
failed to do for the ungated arm.  A `nonm` that gains here, or a december that
crosses, is evidence against the family and is reported as such.

Writes `analysis/regime-test.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-038-packet/code"))
sys.path.insert(0, str(ROOT))
import align as A  # noqa: E402

FOLDS = A.FOLDS
#: THE RULE, fixed before the data. Not a search, not an argument.
HELD = ("nm_bulk", "nm_late")
BASE = "v10 replay"
ARM = "chain d11-noac"
PACKET = "q11-{fold}.parquet"          # the C2-cat run of `code/q09.py`
#: the same three numbers on `L2-hub8725`, for the comparison §11 is about
L2 = {"july_loo": 0.1497, "december": 0.1071, "january": 0.1293, "board": 0.1393}


def price(data: dict, keep) -> dict:
    st, parts = {}, {}
    for f in FOLDS:
        d = data[f]
        y = d["y_true"].to_numpy("float64")
        b = d[BASE].to_numpy("float64")
        on = d["regime"].isin(list(keep)).to_numpy() if keep is not None else \
            np.ones(len(d), bool)
        p = np.where(on, d[ARM].to_numpy("float64"), b)
        pr = A.per_row(y, b, p)
        st[f] = A.fold_stats(pr["r"], pr["d"], np.ones(len(y), bool))
        parts[f] = (A.blocks(d["day"].to_numpy("int64"), 1),
                    pr["r"] ** 2, (pr["r"] - pr["d"]) ** 2)
    g = A.board_gain(st)
    lo, hi = A.boot_board(parts, 2000)
    out = {"served_on": list(keep) if keep is not None else "all rows",
           "gain_board": g["gain_board"], "ci": [lo, hi], "folds": {}}
    for f in FOLDS:
        ci = A.boot_fold(*parts[f])
        w3 = A.boot_fold(A.blocks(data[f]["day"].to_numpy("int64"), 3),
                         parts[f][1], parts[f][2])
        out["folds"][f] = {"gain_rmse": st[f]["gain_rmse"], "ci": list(ci),
                           "ci_3day": list(w3), "A": st[f]["A"], "B": st[f]["B"],
                           "scale": st[f]["A"] / st[f]["B"],
                           "clear": bool(ci[0] > 0.0), "clear_3day": bool(w3[0] > 0.0)}
    out["every_fold_clear"] = bool(all(out["folds"][f]["clear"] for f in FOLDS))
    out["every_fold_clear_3day"] = bool(all(out["folds"][f]["clear_3day"] for f in FOLDS))
    return out


def main() -> None:
    miss = [f for f in FOLDS if not (EXP / "packet" / PACKET.format(fold=f)).exists()]
    if miss:
        raise SystemExit(f"the C2-cat serving run has not produced {miss}; run "
                         f"code/q09.py --member C2-cat --tag q11 first")
    data = {f: pd.read_parquet(EXP / "packet" / PACKET.format(fold=f)) for f in FOLDS}
    for f in FOLDS:
        for c in (BASE, ARM, "regime", "day", "y_true"):
            if c not in data[f].columns:
                raise SystemExit(f"packet {f} has no column {c!r}")

    rec = {"rule": f"serve on {'+'.join(HELD)}, hold the rest at v10",
           "preregistered": "code/regime_test.py, committed before the C2-cat "
                            "december and january arms landed",
           "member": "C2-cat", "l2_hub8725_for_comparison": L2,
           "held": price(data, HELD), "ungated": price(data, None)}
    #: the two single regimes, reported because the family's claim is about their
    #: SIGNS -- `nm_bulk` gains, `nonm` loses -- not about the best subset here.
    for r in ("nm_bulk", "nm_late", "nonm"):
        rec[f"only_{r}"] = price(data, (r,))
    rec["verdict"] = (
        "the family holds on a member that did not choose it"
        if rec["held"]["every_fold_clear"] and rec["only_nonm"]["gain_board"] < 0
        else "the family does NOT reproduce on a member that did not choose it")

    (EXP / "analysis" / "regime-test.json").write_text(json.dumps(rec, indent=2))
    for k in ("held", "ungated", "only_nm_bulk", "only_nm_late", "only_nonm"):
        r = rec[k]
        print(f"\n[{k}] served on {r['served_on']}")
        print(f"  board {r['gain_board']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
        for f in FOLDS:
            x = r["folds"][f]
            print(f"  {f:9s} {x['gain_rmse']:+.4f} 1d [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}]"
                  f" 3d [{x['ci_3day'][0]:+.4f}, {x['ci_3day'][1]:+.4f}]"
                  f"  {'CLEAR' if x['clear'] else 'crosses zero'}")
        print(f"  every fold clear: 1-day {r['every_fold_clear']}  "
              f"3-day {r['every_fold_clear_3day']}")
    print(f"\nVERDICT: {rec['verdict']}")


if __name__ == "__main__":
    main()
