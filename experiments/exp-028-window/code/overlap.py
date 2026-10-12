"""Did stack-5 already make the move the arm makes on blend-5?

The oracle and the sharing statistic are the same on both bases
(`diagnose.py`), so the transient component is NOT consumed — which leaves one
explanation for the arm gaining +0.64 s on blend-5 and +0.07 s on stack-5: the
part of the component the window aggregates can REACH is the part stack-5
already moved, and what the oracle still prices is the part neither can reach.

The test is direct.  Fit the `win_only` arm on blend-5's residual, rotated as
usual, and compare its adjustment with the move stack-5 itself made
(`blend-5 - stack-5`) on the same rows:

  * their correlation, overall and inside `nm_bulk`;
  * the regression slope of the arm's adjustment on stack-5's move;
  * what the arm's blend-5 adjustment is worth when it is applied to stack-5
    instead — if stack-5 has already made the move, adding it again over-corrects.

    python overlap.py
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
import arm as A  # noqa: E402
import panel as P  # noqa: E402
sys.path.insert(0, str(P.ROOT))
import prc.data as D  # noqa: E402


def main() -> None:
    t0 = time.time()
    raw = P.load()
    X = A.design(raw, A.CANDIDATE["win"])
    b5 = A.rebase(raw, "blend5")
    _, adj_pred = A.run(b5, X, arm="win_only", **A.CANDIDATE)

    rows = []
    for f in P.FOLDS:
        d = raw[f]
        y = d["y_true"].to_numpy("float64")
        blend = d["anchor"].to_numpy("float64")
        stack = d["base"].to_numpy("float64")
        adj = blend - adj_pred[f]              # what the arm subtracts from blend-5
        move = blend - stack                   # what stack-5 subtracted from blend-5
        nb = d["regime"].to_numpy() == "nm_bulk"
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        den = float((move ** 2).sum())
        rows.append(dict(
            fold=f, n=len(d),
            corr_all=float(np.corrcoef(adj, move)[0, 1]),
            corr_bulk=float(np.corrcoef(adj[nb], move[nb])[0, 1]),
            sd_arm=float(adj.std()), sd_stack=float(move.std()),
            slope=float((adj * move).sum() / den) if den else np.nan,
            # blend-5 + the arm, against stack-5 and against stack-5 + the same arm
            blend5=D.rmse_clean(y, blend, ids, f),
            blend5_arm=D.rmse_clean(y, blend - adj, ids, f),
            stack5=D.rmse_clean(y, stack, ids, f),
            stack5_blendarm=D.rmse_clean(y, stack - adj, ids, f)))
    T = pd.DataFrame(rows)
    for c in ("blend5", "blend5_arm", "stack5", "stack5_blendarm"):
        T.attrs[c] = P.board({f: T.set_index("fold").loc[f, c] for f in P.SHARES})
    print(T.round(4).to_string(index=False))
    print("\nclean board:  " + "   ".join(f"{c} {T.attrs[c]:.4f}"
                                          for c in ("blend5", "blend5_arm", "stack5",
                                                    "stack5_blendarm")))
    print(f"the arm on blend-5 is worth {T.attrs['blend5'] - T.attrs['blend5_arm']:+.3f} s; "
          f"the SAME adjustment carried onto stack-5 is worth "
          f"{T.attrs['stack5'] - T.attrs['stack5_blendarm']:+.3f} s")
    T.round(4).to_csv(P.EXP / "analysis" / "overlap.csv", index=False)
    (P.EXP / "analysis" / "overlap.json").write_text(json.dumps(
        {"per_fold": json.loads(T.round(4).to_json(orient="records")),
         "board": {c: round(float(T.attrs[c]), 4) for c in
                   ("blend5", "blend5_arm", "stack5", "stack5_blendarm")},
         "wall_s": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
