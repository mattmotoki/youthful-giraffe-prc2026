"""Step 3: what 142 s of RMS on 388 rows needs, as a distance Matt can read.

The record's own board arithmetic (`experiments/exp-026-rebuild/code/
noac_cells.py` eq. 1, used by `exp-033/code/board_price.py`): with `s` the shift
on a ranking row and `e = base - y` the error the base makes there,

    dSSE = sum s^2 + 2 sum s e,   board = sqrt(V^2 + dSSE/N) - V

`rho = sum s e / sum s^2` is the record's alignment ratio; in this experiment's
convention `d = s` and `r = y - base = -e`, so `rho = -A/B = -scale`.  A change
pays only when `rho < -0.5`, i.e. `scale > 0.5`.

Board head V = v12 = 262.7418 (BOARD READING 11); the leader is 260.931, so the
gap is 1.811 s.  `SIG_NOAC = 3542 s` is the record's estimate of the served
error on the 388 (exp-026 section 13, exp-027 S2).  Nothing here reads a board
cell or fits anything to one (score-feedback rule): it is arithmetic on numbers already in the
record plus this experiment's own measured alignment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "code"))
from arms import FIT_MONTHS  # noqa: E402
from read import READ_MONTHS, endpoint  # noqa: E402

N_BOARD = 344841
V12 = 262.7418
LEADER = 260.931
GAP = V12 - LEADER
SIG_NOAC = 3542.0        # the served error on the 388, exp-026 section 13
N388 = 388
BUDGET = 500.0           # the movement budget on the 388, s RMS


def board(dsse: float) -> float:
    """Board seconds gained (negative = the score falls = better)."""
    return float(np.sqrt(V12 ** 2 + dsse / N_BOARD) - V12)


def dsse_needed(gain_s: float) -> float:
    return float(N_BOARD * ((V12 - gain_s) ** 2 - V12 ** 2))


def main() -> None:
    s388 = pd.read_parquet(EXP / "work" / "arms_388.parquet")
    a = pd.read_parquet(EXP / "work" / "arms_2025.parquet")
    arms = [c[2:] for c in s388.columns if c.startswith("d_")]
    rd = a[a.month.isin(READ_MONTHS)]
    r9 = (rd.y_true - rd.base).to_numpy(float)

    # --- table 1: the 388 by band, and where each arm's movement lands -------
    rows = []
    tot = {k: float((s388[f"d_{k}"] ** 2).sum()) for k in arms}
    for bnd, g in s388.groupby("band", observed=True):
        row = {"band": bnd, "n": len(g), "share_388": len(g) / N388,
               "mvt_sched_median_h": float(g.mvt_sched.median() / 3600.0),
               "v7_median_s": float(g.v7.median())}
        for k in arms:
            row[f"rms_d_{k}"] = float(np.sqrt(np.mean(g[f"d_{k}"] ** 2)))
            row[f"share_energy_{k}"] = float((g[f"d_{k}"] ** 2).sum() / tot[k])
        rows.append(row)
    t1 = pd.DataFrame(rows).sort_values("n", ascending=False)
    t1.to_csv(EXP / "analysis" / "anatomy_bands.csv", index=False)

    # --- table 2: what each arm is worth on the board, at four alignments ---
    rows = []
    for k in arms:
        s = s388[f"d_{k}"].to_numpy(float)
        ss = float((s ** 2).sum())
        S = float(np.sqrt(ss / N388))
        e9 = endpoint(r9, rd[f"d_{k}"].to_numpy(float))
        rho9 = -e9["scale"] if e9["scale"] is not None else np.nan   # record convention
        ceil_rho = -SIG_NOAC / S
        rows.append({
            "arm": k, "rms_d_388": S, "n_moved": int((s != 0).sum()),
            "nine_month_scale": e9["scale"], "nine_month_rho_board": rho9,
            "board_at_measured_rho": board(ss * (1 + 2 * rho9)),
            "board_if_uncorrelated": board(ss),
            "board_at_rho_minus_1": board(-ss),
            "board_ceiling_cs": board(ss * (1 + 2 * ceil_rho))})
    t2 = pd.DataFrame(rows)
    t2.to_csv(EXP / "analysis" / "anatomy_board.csv", index=False)

    # --- table 3: the distance ---------------------------------------------
    need = dsse_needed(GAP)                       # dSSE that closes the gap
    # correlation between the correction and the base error, rho_c = A/(sig*S)
    def rho_c_needed(S):
        # 388*(S^2 - 2*rho_c*S*SIG) = need
        return float((S ** 2 - need / N388) / (2.0 * S * SIG_NOAC))
    S_opt_unbounded = None
    rho_opt = float(np.sqrt(-need / (N388 * SIG_NOAC ** 2)))     # best case over S
    S_opt_unbounded = rho_opt * SIG_NOAC
    ladder = []
    for S in (100.0, 142.0, 250.0, BUDGET, 918.1):
        ladder.append({"rms_d_388": S,
                       "rho_c_needed_to_close_gap": rho_c_needed(S),
                       "board_at_perfect_aim_rho_minus1": board(-N388 * S ** 2),
                       "board_ceiling_cs": board(N388 * S ** 2 - 2 * S * SIG_NOAC * N388)})
    t3 = pd.DataFrame(ladder)
    t3.to_csv(EXP / "analysis" / "anatomy_ladder.csv", index=False)

    # what the measured alignment is worth at its own optimal amplitude, with a
    # day-clustered interval on the alignment itself
    ft = a[a.month.isin(FIT_MONTHS)]
    rf = (ft.y_true - ft.base).to_numpy(float)
    days9 = rd.day.to_numpy().astype("datetime64[D]").astype("int64")
    rng = np.random.default_rng(11)
    uq, inv = np.unique(days9, return_inverse=True)
    pos = [np.flatnonzero(inv == i) for i in range(len(uq))]
    draws = [np.concatenate([pos[j] for j in rng.integers(0, len(uq), len(uq))])
             for _ in range(2000)]

    def worth(rc):
        return board(-N388 * (rc ** 2) * SIG_NOAC ** 2) if rc > 0 else 0.0

    meas = []
    for k in arms:
        dv = rd[f"d_{k}"].to_numpy(float)
        rc = endpoint(r9, dv)["rho"]
        bs = np.array([np.corrcoef(r9[i], dv[i])[0, 1] for i in draws])
        rcf = endpoint(rf, ft[f"d_{k}"].to_numpy(float))["rho"]
        meas.append({"arm": k, "rho_c_nine_month": rc,
                     "rho_lo": float(np.percentile(bs, 2.5)),
                     "rho_hi": float(np.percentile(bs, 97.5)),
                     "rho_c_fit_in_sample": rcf,
                     "optimal_rms_d": rc * SIG_NOAC,
                     "board_at_optimal_amplitude": worth(rc),
                     "board_at_rho_hi": worth(float(np.percentile(bs, 97.5))),
                     "board_at_in_sample_rho": worth(rcf),
                     "fraction_of_gap": abs(worth(rc)) / GAP,
                     "fraction_of_gap_at_rho_hi": abs(worth(float(np.percentile(bs, 97.5)))) / GAP})
    t4 = pd.DataFrame(meas)
    t4.to_csv(EXP / "analysis" / "anatomy_measured.csv", index=False)

    rec = {"V12": V12, "leader": LEADER, "gap_s": GAP, "N_board": N_BOARD,
           "sig_noac": SIG_NOAC, "dsse_to_close_gap": need,
           "rho_c_needed_best_case": rho_opt,
           "rms_d_at_that_rho": S_opt_unbounded,
           "rho_c_needed_within_500s_budget": rho_c_needed(BUDGET),
           "board_if_388_were_perfect": board(-N388 * SIG_NOAC ** 2),
           "bands": t1.to_dict("records"), "board": t2.to_dict("records"),
           "ladder": t3.to_dict("records"), "measured": t4.to_dict("records")}
    (EXP / "analysis" / "anatomy.json").write_text(json.dumps(rec, indent=2, default=float))

    pd.set_option("display.width", 220)
    print(f"board head v12 {V12}, leader {LEADER}, gap {GAP:.3f} s; "
          f"all 388 perfect is worth {board(-N388 * SIG_NOAC ** 2):+.2f} s")
    print(f"closing the gap on the 388 alone needs dSSE {need:,.0f} s^2 -> "
          f"correlation {rho_opt:.3f} at {S_opt_unbounded:.0f} s RMS, "
          f"or {rho_c_needed(BUDGET):.3f} inside the 500 s budget")
    print("\n=== the 388 by band, and where each arm's movement lands ===")
    print(t1.round(3).to_string(index=False))
    print("\n=== what each arm is worth on the board ===")
    print(t2.round(4).to_string(index=False))
    print("\n=== the ladder: what a move of this size can buy ===")
    print(t3.round(4).to_string(index=False))
    print("\n=== what the measured nine-month alignment is worth ===")
    print(t4.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
