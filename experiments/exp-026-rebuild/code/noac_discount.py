"""What is stack-5 worth if the noac gain is given back? Measured, not adjusted.

`docs/SUBMISSIONS.md` carries stack-5 in two currencies: the clean+4 rule at
face value, and the same with blend-5's noac-concentrated fold gain discounted.
The discount can be computed rather than estimated: §5a says three quarters of
blend-5's gain over blend-4 sits on the `noac` regime, and stack-5 serves its
noac rows from blend-5's anchor unchanged, so substituting **blend-4's** rotated
prediction on those rows and re-scoring gives the number stack-5 would hold if
the noac swaps were worth exactly nothing.

That is a pessimistic bound, not a forecast: it assumes the whole noac gain is
an artefact of 402 fitting rows. The face value assumes it all transfers. The
board is somewhere between, and `blend-5 -T-tailv4` is the upload that says
where.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
import ladder as L  # noqa: E402
import rule as R  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.data as D  # noqa: E402


def main() -> None:
    cache: dict = {}
    b4 = L.run(L.B4, cache)
    b5 = L.run(L.B5, cache)
    rows = {}
    for f in F.FOLDS:
        m = F.fold_frame(f, L.B5)
        m4 = F.fold_frame(f, L.B4)
        if not np.array_equal(m["MVT_ID_mvt"].to_numpy(), m4["MVT_ID_mvt"].to_numpy()):
            raise SystemExit(f"{f}: member sets do not align")
        d = pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet")
        if not np.array_equal(d["MVT_ID_mvt"].to_numpy("float64"),
                              m["MVT_ID_mvt"].to_numpy("float64")):
            raise SystemExit(f"{f}: stack-5 oof does not align with the frame")
        y = m["y_true"].to_numpy("float64")
        ids = m["MVT_ID_mvt"].to_numpy("float64")
        noac = m["regime"].to_numpy() == "noac"
        variants = {
            "blend-4": b4[f]["pred"],
            "blend-5": b5[f]["pred"],
            "blend-5, noac on blend-4": np.where(noac, b4[f]["pred"], b5[f]["pred"]),
            "stack-5 keep-noac": d["y_pred"].to_numpy("float64"),
            "stack-5, noac on blend-4": np.where(noac, b4[f]["pred"],
                                                 d["y_pred"].to_numpy("float64")),
        }
        for k, p in variants.items():
            rows.setdefault(k, {})[f] = D.rmse_both(y, p, ids, f)["rmse_clean"]
        rows.setdefault("_n_noac", {})[f] = int(noac.sum())

    T = pd.DataFrame({k: v for k, v in rows.items() if not k.startswith("_")}).T
    T = T[list(F.FOLDS)]
    T["clean board"] = [F.board_estimate({f: T.loc[k, f] for f in F.SHARES}) for k in T.index]
    T["predict"] = T["clean board"] + 4.0
    T["vs blend-5"] = T["clean board"] - T.loc["blend-5", "clean board"]
    print("=== clean rotation, with and without the noac gain ===")
    print(T.round(3).to_string())
    print(f"\nnoac rows per fold: {rows['_n_noac']}; the ranking file holds 388")
    disc = T.loc["stack-5, noac on blend-4", "clean board"] - T.loc["stack-5 keep-noac", "clean board"]
    print(f"\nthe whole noac gain is worth {disc:+.3f} s to stack-5: "
          f"{T.loc['stack-5 keep-noac','predict']:.2f} at face value, "
          f"{T.loc['stack-5, noac on blend-4','predict']:.2f} discounted to zero")
    T.round(4).to_csv(F.EXP / "analysis" / "noac_discount.csv")
    (F.EXP / "analysis" / "noac_discount.json").write_text(
        json.dumps({"table": json.loads(T.round(4).to_json(orient="index")),
                    "n_noac": rows["_n_noac"], "discount_s": round(float(disc), 4)}, indent=2))


if __name__ == "__main__":
    main()
