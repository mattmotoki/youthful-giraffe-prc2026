"""How much would the stacker's board estimate move if the CatBoost OOF changed?

exp-012's reproducibility work has three possible outcomes. In outcome 2 --
CatBoost is not bit-reproducible on GPU at a fixed thread count, so the member
moves to CPU -- both sides of the member move, the stacker is refitted, and
stack-3's 317.215 does not carry.

That is worth bounding BEFORE anyone spends GPU-hours, and it can be bounded
with files already on disk. `exp-008` left two CatBoost OOF sets on all three
folds:

    C2-d8p   5,325 rounds, the shipped member          july_loo 315.17
    C1-d8      600 rounds, the truncated budget        july_loo 330.62

They differ by **15.5 s of member quality** (`exp-012` RESULTS), which is far
larger than any numerical difference a GPU-versus-CPU refit of the same
configuration could produce. So substituting C1 for C2 and re-running the
rotation gives a generous UPPER BOUND on what outcome 2 costs the stacker.

No new fits of any member; only the stacker itself is refitted, which is
seconds. Nothing is written to `submissions/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import members as M  # noqa: E402
import stack as S  # noqa: E402
import verify_r1 as V  # noqa: E402
sys.path.insert(0, str(M.ROOT))
import prc.blend as B  # noqa: E402

OUT = M.EXP / "analysis"
ALT = "experiments/exp-008-diversity/oof/C1-d8-{fold}.parquet"


def rotation(label: str) -> dict:
    """stack-3's configuration: members-only arm, five-seed bag, noac offload."""
    data = {f: M.fold_frame(f) for f in M.FOLDS}
    per_fold, member = {}, {}
    for f in M.FOLDS:
        ap = data[f]
        fit = pd.concat([data[g] for g in M.FOLDS if g != f], ignore_index=True)
        y = ap["y_true"].to_numpy("float64")
        ref = S.fit_blend2(fit, ap)
        bag = np.maximum(np.mean([V.arm(fit, ap, s, V.NOGATE)[0] for s in V.SEEDS], axis=0),
                         B.FLOOR)
        reg = ap["regime"].to_numpy()
        p = np.where(reg == "noac", ref, bag)
        per_fold[f] = dict(stack=S.rmse(y, p), blend2=S.rmse(y, ref),
                           member=S.rmse(y, np.maximum(ap["C2-cat"].to_numpy("float64"), B.FLOOR)))
        member[f] = per_fold[f]["member"]
        print(f"  [{label}] {f:9s} member {per_fold[f]['member']:7.2f}  "
              f"blend2 {per_fold[f]['blend2']:7.3f}  stack {per_fold[f]['stack']:7.3f}", flush=True)
    board = {k: M.board_estimate({f: per_fold[f][k] for f in M.SHARES})
             for k in ("stack", "blend2", "member")}
    return {"per_fold": per_fold, "board": board}


def main() -> None:
    M.use_config(M.EXP / "members-blend3.json")
    print("=== shipped C2-cat (C2-d8p, 5,325 rounds) ===")
    base = rotation("C2-d8p")
    shipped = M.OOF["C2-cat"]

    M.OOF["C2-cat"] = ALT
    M.PINS = M.EXP / "pins-sensitivity-c1.json"
    print("\n=== substituted C1-d8 (600 rounds, 15.5 s worse member) ===")
    alt = rotation("C1-d8")
    M.OOF["C2-cat"] = shipped

    rows = []
    for k, name in (("member", "C2-cat member alone"),
                    ("blend2", "blend-3 rule"),
                    ("stack", "stack-3 form")):
        rows.append(dict(quantity=name, shipped=base["board"][k], substituted=alt["board"][k],
                         delta=alt["board"][k] - base["board"][k]))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "sensitivity_cat.csv", index=False)
    print("\n=== board estimate under a 15.5 s degradation of the CatBoost member ===")
    print(T.round(3).to_string(index=False))
    dm = T.loc[T.quantity == "C2-cat member alone", "delta"].iloc[0]
    ds = T.loc[T.quantity == "stack-3 form", "delta"].iloc[0]
    print(f"\n  member moves {dm:+.2f} s, the stack-3 form moves {ds:+.2f} s "
          f"-> retention {100*ds/dm:.1f} %")
    print("  A GPU-versus-CPU refit of the SAME configuration is a numerical difference,")
    print("  orders below a 15.5 s budget change, so this is a generous upper bound on")
    print("  what exp-012's outcome 2 would cost this candidate's board estimate.")


if __name__ == "__main__":
    main()
