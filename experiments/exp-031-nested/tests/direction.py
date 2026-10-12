"""The substitution goes into the FITTING folds, and nowhere else.

Two checks that no RMSE gate can make, because both are about which file landed
in which slot rather than about what the numbers came to.

1. `prc.data.NESTED_FOLDS` maps `<base>_nx_<mon>` to the scored fold it serves.
   `instrument.source` must agree with it on all six nested folds, in both
   directions: the file it picks for slot `g` while scoring `f` must be the one
   `NESTED_FOLDS` says serves `f`, and every file `NESTED_FOLDS` assigns to `f`
   must be picked.
2. On the `nested` column the SCORED fold's member column must equal the
   `kaggle` column's (both are the member's ordinary fold, refit on nothing
   extra) and the two FITTING folds' must differ from it. If the substitution
   were the other way round -- the scored fold nested and the fitting folds not
   -- the label channel would stay open and every number in this experiment
   would be wrong while every gate still passed.

    python direction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import instrument as I  # noqa: E402
sys.path.insert(0, str(I.ROOT))
import prc.data as D  # noqa: E402

NAMES = ("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh",
         "V3-allin", "N1-nmtail")


def main() -> None:
    # ---- 1. against prc.data.NESTED_FOLDS ---------------------------------
    picked = {}
    for scored in I.FOLDS:
        for slot in I.FOLDS:
            s = I.source("nested", "F1-fresh", slot, scored)
            if "_nx_" not in s:
                assert slot == scored and s == scored, f"{scored}/{slot}: {s}"
                continue
            serves, validates = D.NESTED_FOLDS[s]
            assert serves == scored, f"{s} serves {serves}, used to score {scored}"
            assert validates == slot, f"{s} validates {validates}, used in slot {slot}"
            picked[s] = (scored, slot)
    assert set(picked) == set(D.NESTED_FOLDS), \
        f"unused nested folds: {set(D.NESTED_FOLDS) - set(picked)}"
    print(f"[ok ] all six of prc.data.NESTED_FOLDS are used, each in the slot and "
          f"the rotation it declares")

    # ---- 2. the scored fold is NOT nested, the fitting folds are ----------
    base = I.base_frames(NAMES)
    kag = {f: I.build("kaggle", f, NAMES, base) for f in I.FOLDS}
    for scored in I.FOLDS:
        nst = I.build("nested", scored, NAMES, base)
        for member in I.NESTABLE:
            for slot in I.FOLDS:
                a = nst[slot][member].to_numpy("float64")
                b = kag[scored][slot][member].to_numpy("float64")
                same = bool(np.array_equal(a, b))
                if slot == scored:
                    assert same, (f"scoring {scored}: {member}'s own fold is NOT its "
                                  f"ordinary Kaggle baseline -- the substitution is "
                                  f"the wrong way round")
                else:
                    assert not same, (f"scoring {scored}: {member}'s fitting fold "
                                      f"{slot} was not substituted")
        print(f"[ok ] scoring {scored:9s}: the scored fold holds the ordinary baseline, "
              f"both fitting folds are substituted, for all three members")

    # ---- 3. the unnested members never move ------------------------------
    for scored in I.FOLDS:
        nst = I.build("nested", scored, NAMES, base)
        for member in NAMES:
            if member in I.NESTABLE:
                continue
            for slot in I.FOLDS:
                a = nst[slot][member].to_numpy("float64")
                b = base[slot][member].to_numpy("float64")
                assert np.array_equal(a, b) or (np.isnan(a) & np.isnan(b)).all() or \
                    bool((np.isnan(a) == np.isnan(b)).all() and
                         np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])), \
                    f"{member} moved on the nested column"
    print("[ok ] the four unnested members are byte-identical to the published column")
    print("\nPASS: the substitution closes the fitting folds' channel and nothing else.")


if __name__ == "__main__":
    main()
