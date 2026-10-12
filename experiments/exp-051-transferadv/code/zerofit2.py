"""exp-051, before the iteration-2 registration: two zero-fit quantities it needs.  NO FIT.

(a) The PLAIN K = 3 bag (members 1-3 of each shipped stack, averaged by exp-046's own
    `bagarm.build_bag`) priced step by step, per fold and on the board.  Iteration 2 changes
    only july_loo, so T2's january part is fixed by these numbers.  From them: the lowest
    T2_L the design can reach if the july carrier step were exactly zero, and the july step
    ratio the design would need to reach each board transfer.  Both are arithmetic on the
    plain arm; no carrier fit exists when this runs.

(b) RECORD ONLY, proxy limit 7 of SPEC I1: how noisy is a board transfer?  The board gain of a
    step is one realisation on the 2026 rows.  Its sampling spread is proxied by the day
    bootstrap of the K = 10 plain face on july_loo + january (the same draws as price.py,
    seed 5101).  It changes no rule and no verdict.

    .venv/bin/python experiments/exp-051-transferadv/code/zerofit2.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import common as C
from common import A, BA, NO, R
import bagarm as BAG                                                  # noqa: E402
import memberspread as MS

OUT = C.ANALYSIS / "zerofit2.json"
SEED = 5101
N_BOOT = 2000
BOARD_FOLDS = ("july_loo", "january")
K3 = [1, 2, 3]


def board_pt(mse: dict) -> float:
    return float(np.sqrt(sum(BA.SHARES[f] * mse[f] for f in BOARD_FOLDS)))


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    NO.refuse_existing([OUT], overwrite=a.overwrite)
    pan_ship, _, _, _ = C.build_panels()
    r0 = {f: pan_ship[f].y_true.to_numpy("float64") - pan_ship[f].pbase.to_numpy("float64")
          for f in A.FOLDS}

    # ---- (a) the plain K = 3 bag
    bag3 = {}
    for L in range(4):
        mem = {s: C.load_vec(MS.member_path(L, s)) for s in K3}
        bag3[L], _ = BAG.build_bag(mem, pan_ship, K3)
    out = {"section": "exp-051 zero fit before the I2 registration", "K3_members": K3,
           "K3_plain": {}, "board_noise_record_only": {}}
    for L in (1, 2, 3):
        s = C.STEP[L]
        pr = R.price(pan_ship, bag3[L], bag3[L - 1], f"plain K=3 {s}")
        g = {f: pr["by_fold"][f]["mse_gain_s2"] for f in A.FOLDS}
        mse0 = {f: float(np.mean((r0[f] - bag3[L - 1][f]) ** 2)) for f in A.FOLDS}
        mse1 = {f: float(np.mean((r0[f] - bag3[L][f]) ** 2)) for f in A.FOLDS}
        gain = board_pt(mse0) - board_pt(mse1)
        # the july carrier step at exactly zero: july's MSE stays at its predecessor's
        floor_ = (board_pt(mse0) - board_pt({"july_loo": mse0["july_loo"], "january": mse1["january"]})) / gain
        # the july step ratio rho (carrier / plain, s2 per row) at which T2_L equals the board's
        # transfer, holding january plain: solve on the board-seconds scale by bisection
        def t_of(rho: float) -> float:
            m1 = {"july_loo": mse0["july_loo"] - rho * g["july_loo"], "january": mse1["january"]}
            return (board_pt(mse0) - board_pt(m1)) / gain
        need = {}
        for key, tgt in (("board_exact", C.BOARD[s]["transfer"]),):
            lo, hi = -5.0, 5.0
            for _ in range(200):
                mid = (lo + hi) / 2
                if t_of(mid) < tgt:
                    lo = mid
                else:
                    hi = mid
            need[key] = (lo + hi) / 2
        out["K3_plain"][s] = {
            "by_fold_s2": g, "board_s": gain,
            "board_s_rung511": pr["board_all_row"]["gain_s"],
            "board_ci95": pr["board_all_row"]["interval"]["ci95"],
            "T2_floor_if_july_carrier_step_is_zero": floor_,
            "july_rho_needed_for_board_transfer": need["board_exact"],
            "board_transfer_read_only": C.BOARD[s]["transfer"]}
        print(f"{s}: plain K3 board {gain:+.4f} s  july {g['july_loo']:+.1f}  jan {g['january']:+.1f}"
              f"  dec {g['december']:+.1f}  | T2 floor {floor_:.3f}  july rho needed "
              f"{need['board_exact']:.3f} for {C.BOARD[s]['transfer']:.3f}")

    # ---- (b) record only: the board-side noise proxy from the K = 10 faces
    rng = np.random.default_rng(SEED)
    draws, codes, cnt = {}, {}, {}
    for f in A.FOLDS:
        cd, uq = pd.factorize(pan_ship[f].day.to_numpy())
        codes[f], k = cd, len(uq)
        draws[f] = rng.integers(0, k, size=(N_BOOT, k))
        c = np.bincount(cd, minlength=k).astype("float64")
        cnt[f] = c[draws[f]].sum(1)
    bag10 = {L: C.load_vec(MS.BAGS[L]) for L in range(4)}
    mseb = {}
    for L in range(4):
        mseb[L] = {}
        for f in BOARD_FOLDS:
            e = (r0[f] - bag10[L][f]) ** 2
            sm = np.bincount(codes[f], weights=e, minlength=draws[f].shape[1])
            mseb[L][f] = sm[draws[f]].sum(1) / cnt[f]
    for L in (1, 2, 3):
        s = C.STEP[L]
        b0 = np.sqrt(sum(BA.SHARES[f] * mseb[L - 1][f] for f in BOARD_FOLDS))
        b1 = np.sqrt(sum(BA.SHARES[f] * mseb[L][f] for f in BOARD_FOLDS))
        gd = b0 - b1
        sd = float(gd.std(ddof=1))
        face, bg = C.BOARD[s]["face_s"], C.BOARD[s]["board_gain_s"]
        out["board_noise_record_only"][s] = {
            "K10_face_s": face, "K10_face_day_bootstrap_sd_s": sd,
            "K10_face_ci95": [float(np.percentile(gd, 2.5)), float(np.percentile(gd, 97.5))],
            "board_gain_s": bg, "board_transfer": bg / face,
            "board_transfer_pm_1p96_sd_over_face": [(bg - 1.96 * sd) / face, (bg + 1.96 * sd) / face],
            "assumes": ("the 2026 board rows carry the day-level spread of the 2025 fold rows "
                        "for the same step; the fold face is held fixed, so only the board "
                        "side's noise is in this band"),
            "gates": False}
        print(f"{s}: K10 face day-bootstrap sd {sd:.4f} s -> board transfer "
              f"{bg/face:.3f} +- {1.96*sd/face:.3f} (record only)")
    out["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, out, overwrite=a.overwrite)
    print(f"written to {OUT.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
