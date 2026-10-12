"""What the surviving arm is worth on the board, in the ledger's own arithmetic.

`experiments/exp-026-rebuild/code/noac_cells.py` equation (1): with `s` the
shift a change makes on a row and `e = base - y` the error the base makes there,

    dSSE = sum s^2 + 2 * sum s * e,     rho = sum s*e / sum s^2

so a change pays only when rho < -0.5.  rho is NOT a correlation coefficient and
is NOT bounded below by -1 (Sol round 1, S9): it is a ratio of two sums, and the
fold's own value here is -1.293.  rho = -1 is only the point where the change is
worth exactly its own size in the other direction; the real floor is
Cauchy-Schwarz, rho >= -sqrt(sum e^2 / sum s^2).  The fold measures rho directly,
because on the 2025 rows both `s` and `e` are known.  Four numbers per arm: what
the fold's own rho would be worth on the board, what an uncorrelated change of
that size costs, what rho = -1 would be worth, and the Cauchy-Schwarz ceiling for
a shift of that size against blend-5's 3,542 s.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction as DIR  # noqa: E402
from describe import LAB  # noqa: E402
from direction import apply_head, fit_head, rmse  # noqa: E402

N_BOARD = 344841
V7 = 263.4623
SIG_NOAC = 3542.0        # blend-5's error on the 388 (exp-026 §13, exp-027 S2)


def score(dsse):
    return float(np.sqrt(V7 ** 2 + dsse / N_BOARD) - V7)


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    a["_m"] = DIR.matched_mask(a, b)
    h_all, h_mat = fit_head(a), fit_head(a[a._m])
    b["corr"] = apply_head(h_mat, b) - apply_head(h_all, b)
    v = pd.read_parquet(EXP / "work" / "fold_arms.parquet")
    v["corr"] = v.delta - v.y_pred

    rows = []
    for bands, tag in ((["1-2h"], "1-2h only"), (["1-2h", "2-4h"], "1-2h + 2-4h"),
                       (LAB, "every band")):
        # the fold's own rho, on all noac rows of the three fold months
        s = np.where(v.band.isin(bands), v["corr"], 0.0)
        e = (v.y_pred - v.y_true).to_numpy(float)
        ss = float((s ** 2).sum())
        rho = float((s * e).sum() / ss)
        # the same change on the 388
        s26 = np.where(b.band.isin(bands), b["corr"], 0.0)
        ss26 = float((s26 ** 2).sum())
        ceiling = score(ss26 - 2 * np.sqrt(ss26) * np.sqrt(388 * SIG_NOAC ** 2))
        rows.append({
            "arm": tag, "n_moved_388": int((s26 != 0).sum()),
            "shift_rms_388": float(np.sqrt(np.mean(s26 ** 2))),
            "fold_rho": rho,
            "board_if_rho_transfers": score(ss26 * (1 + 2 * rho)),
            "board_if_uncorrelated": score(ss26),
            "board_at_rho_minus_1": score(-ss26),
            "board_ceiling_cauchy_schwarz": ceiling,
            "fold_gain_all_noac": rmse(v.y_true, v.y_pred) - rmse(v.y_true, v.y_pred + s),
        })
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "board_price.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
