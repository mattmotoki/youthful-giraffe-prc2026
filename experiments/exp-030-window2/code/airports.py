"""What arm 3 is made of, and whether EHAM is special.

Arm 3 excludes one airport from a served mask after that airport was named by a
per-airport table computed on the same three folds (exp-028 §5).  That is a
selection on the scored folds and this file does not pretend otherwise; what it
can do is say whether the gain is **EHAM's** or whether *any* exclusion would pay,
which is the difference between "the block does not transfer at this airport" and
"serving the block on fewer rows is simply safer".

Two tables, both arithmetic on exp-028's committed w600 rotation, nothing refitted:

  * per airport, v6's fold RMSE against stack-5's **inside that airport's rows**;
  * the board gain over v6 of excluding each airport in turn, one at a time.

    python airports.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import score as Sc  # noqa: E402
sys.path.insert(0, str(Sc.ROOT))
import prc.data as D  # noqa: E402


def main() -> None:
    ref = Sc.reference()
    apts = sorted(set(ref["july_loo"]["airport"].astype(str)))
    inside, excl = [], []
    for apt in apts:
        row_in = {"airport": apt}
        preds = {}
        for f in Sc.FOLDS:
            d = ref[f]
            a = d["airport"].astype(str).to_numpy() == apt
            y = d["y_true"].to_numpy("float64")
            s5 = d["y_pred_stack5"].to_numpy("float64")
            v6 = d["y_pred_v6"].to_numpy("float64")
            ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy() & a
            r5 = float(np.sqrt(np.mean((s5[ok] - y[ok]) ** 2)))
            r6 = float(np.sqrt(np.mean((v6[ok] - y[ok]) ** 2)))
            row_in[f"n_{f}"] = int(ok.sum())
            row_in[f"stack5_{f}"] = round(r5, 3)
            row_in[f"v6_{f}"] = round(r6, 3)
            row_in[f"gain_{f}"] = round(r5 - r6, 3)
            preds[f] = np.where(a, s5, v6)
        inside.append(row_in)
        rec = Sc.score(f"leave-out-{apt}", preds, ref, note="one-airport exclusion on v6")
        excl.append({"airport": apt,
                     **{f"vs_v6_{f}": rec["vs_v6"][f] for f in Sc.FOLDS},
                     "gain_board_vs_v6": rec["gain_board_vs_v6"],
                     "worst_fold_vs_v6": rec["worst_fold_vs_v6"],
                     "gate": rec["gate"]})
    A = pd.DataFrame(inside)
    E = pd.DataFrame(excl).sort_values("gain_board_vs_v6", ascending=False)
    A.to_csv(EXP / "analysis" / "airport_inside.csv", index=False)
    E.to_csv(EXP / "analysis" / "airport_leave_out.csv", index=False)
    print("=== v6 against stack-5 inside each airport's own rows (clean; + is v6 better) ===")
    print(A[["airport"] + [f"gain_{f}" for f in Sc.FOLDS]
            + [f"n_{f}" for f in Sc.FOLDS]].to_string(index=False))
    print("\n=== excluding one airport from v6's served mask, board gain over v6 ===")
    print(E.to_string(index=False))
    (EXP / "analysis" / "airports.json").write_text(
        json.dumps({"inside": inside, "leave_out": excl}, indent=2))


if __name__ == "__main__":
    main()
