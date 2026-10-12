"""Persist the numbers RESULTS quotes that no other CSV holds (Sol round 2, R2-7).

Four records:

  arm_summary.csv        every arm x scope: RMSE, gain, the day-clustered
                         interval, the shift on the fold and on the 388
  mixture_coordinate.csv the implied mixture coordinate's range and its
                         out-of-[0,1] share, per band
  band8_arithmetic.csv   the 8h+ hypothetical §8 quotes: what moving those 22
                         rows from the shipped coordinate to the 2025 rate would
                         cost in shift and in board, and what the oracle is worth
  band_table.csv         gains an `ALL` row, so the aggregate the table shows is
                         in the file (written by `table.py`)
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
from describe import LAB  # noqa: E402
from direction import (apply_head, fit_head, matched_mask,  # noqa: E402
                       paired_rmse_diff, rmse)

N_BOARD = 344841
V7 = 263.4623
SIG_NOAC = 3542.0
ARMS = {"1-2h only": ["1-2h"], "1-2h + 2-4h": ["1-2h", "2-4h"], "every band": LAB}


def score(dsse: float) -> float:
    return float(np.sqrt(V7 ** 2 + dsse / N_BOARD) - V7)


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    a["_m"] = matched_mask(a, b)
    h_all, h_mat = fit_head(a), fit_head(a[a._m])
    b["corr"] = apply_head(h_mat, b) - apply_head(h_all, b)
    v = pd.read_parquet(EXP / "work" / "fold_arms.parquet")
    v["corr"] = v.delta - v.y_pred

    rows = []
    for tag, bands in ARMS.items():
        s = np.where(v.band.isin(bands), v["corr"], 0.0)
        s26 = np.where(b.band.isin(bands), b["corr"], 0.0)
        for scope in ("matched", "all"):
            sel = np.flatnonzero(v._m.to_numpy() if scope == "matched"
                                 else np.ones(len(v), bool))
            y, base = v.y_true.to_numpy()[sel], v.y_pred.to_numpy()[sel]
            arm = base + s[sel]
            lo, hi = paired_rmse_diff(y, base, arm, v.day.to_numpy()[sel])
            rows.append({"arm": tag, "scope": scope, "n": len(sel),
                         "rmse_shipped": rmse(y, base), "rmse_arm": rmse(y, arm),
                         "gain": rmse(y, base) - rmse(y, arm), "lo": lo, "hi": hi,
                         "fold_shift_rms": float(np.sqrt(np.mean(s[sel] ** 2))),
                         "n_moved_388": int((s26 != 0).sum()),
                         "shift_rms_388": float(np.sqrt(np.mean(s26 ** 2))),
                         "budget_500_ok": bool(np.sqrt(np.mean(s26 ** 2)) <= 500)})
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "arm_summary.csv", index=False)
    print(t.round(3).to_string(index=False))

    # --- the implied mixture coordinate ------------------------------------
    rows = []
    for lab in LAB:
        q = b[b.band == lab]
        if not len(q):
            continue
        p = q.p_v7.to_numpy(float)
        rows.append({"band": lab, "n2026": len(q), "median": float(np.median(p)),
                     "min": float(p.min()), "max": float(p.max()),
                     "n_outside_0_1": int(((p < 0) | (p > 1)).sum())})
    m = pd.DataFrame(rows)
    m.loc[len(m)] = {"band": "ALL", "n2026": int(m.n2026.sum()),
                     "median": float(np.median(b.p_v7)), "min": float(b.p_v7.min()),
                     "max": float(b.p_v7.max()),
                     "n_outside_0_1": int(m.n_outside_0_1.sum())}
    m.to_csv(EXP / "analysis" / "mixture_coordinate.csv", index=False)
    print("\n" + m.round(3).to_string(index=False))

    # --- the 8h+ hypothetical §8 quotes ------------------------------------
    q = b[b.band == "8h+"]
    p_v7 = float(np.median(q.p_v7))
    p25 = h_all["8h+"][0]
    oph, onp = h_all["8h+"][1], h_all["8h+"][2]
    per_row = abs((p_v7 - p25) * (onp - oph))
    ss = len(q) * per_row ** 2
    orac = np.sqrt(V7 ** 2 - 388 * SIG_NOAC ** 2 / N_BOARD) - V7
    d = pd.DataFrame([{
        "band": "8h+", "n2026": len(q),
        "coordinate_v7_median": p_v7, "rate_2025_band": p25,
        "o_ph": oph, "o_np": onp,
        "shift_per_row_s": per_row,
        "shift_rms_388_s": float(np.sqrt(ss / 388)),
        "budget_500_ok": bool(np.sqrt(ss / 388) <= 500),
        "board_if_uncorrelated_s": score(ss),
        "board_at_rho_minus_1_s": score(-ss),
        "board_oracle_all_388_s": float(orac),
    }])
    d.to_csv(EXP / "analysis" / "band8_arithmetic.csv", index=False)
    print("\n" + d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
