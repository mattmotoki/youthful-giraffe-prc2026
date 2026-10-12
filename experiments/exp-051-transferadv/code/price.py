"""exp-051 iteration 1, items 2-3: price the board-like stack against the plain one. ZERO FIT.

For fold f and level L, the board-like vector S_L is STITCHED: a row in half B takes the fit
that added half A's days, a row in half A the fit that added half B's.  Every fold row is
scored exactly once, by a fit that never saw its day.  The plain vector P_L is exp-046's saved
member-1 stack.  Each step is priced over its own predecessor, on identical rows:

    G_X(L) = mean over the fold of 2 r inc - inc^2,  r = y - pbase - X_{L-1},  inc = X_L - X_{L-1}

with the record's pricing function `rung511.price` (per fold, day-clustered, and the board
all-row gain over july_loo + january at base.SHARES).  The registered statistic is

    T_L = board gain of S_L over S_{L-1}  /  board gain of P_L over P_{L-1}

with a PAIRED day bootstrap: one set of day draws per fold, shared by both arms and all steps.
PASS (SPEC I1.3) iff for each of v20, v21, v22 the 95 % interval of T_L contains the read-only
board transfer.  December carries no board weight and is reported beside, never in, T_L.

    .venv/bin/python experiments/exp-051-transferadv/code/price.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common as C
from common import A, BA, NO, R

OUT = C.ANALYSIS / "boardlike.json"
SEED = 5101
N_BOOT = 2000
BOARD_FOLDS = ("july_loo", "january")


def stitched(f: str, level: int, hv: np.ndarray, fits: dict) -> np.ndarray:
    vA = C.load_vec(C.fit_path(f, 0, level)[0])[f]       # half A added -> scores half B
    vB = C.load_vec(C.fit_path(f, 1, level)[0])[f]       # half B added -> scores half A
    for h, v in ((0, vA), (1, vB)):
        npz, js = C.fit_path(f, h, level)
        rec = json.loads(js.read_text())
        if rec["vector_sha256"] != C.sha256(npz):
            raise SystemExit(f"{npz.name}: sha256 is not the one its fit record carries")
        fits[npz.stem] = {"sha256": rec["vector_sha256"], "seconds_fit": rec["seconds_fit"],
                          "total_fit_rows": rec["fit_arm_provenance"]["total_fit_rows"],
                          "half_rows": rec["rows"].get("half_rows"),
                          "ext_rows": rec["rows"].get("ext_rows", 0),
                          "load_1min_at_launch": rec["load_1min_at_launch"]}
    return np.where(hv == 1, vA, vB)


def board_from_mse(mse: dict) -> np.ndarray:
    return np.sqrt(sum(BA.SHARES[f] * mse[f] for f in BOARD_FOLDS))


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    NO.refuse_existing([OUT], overwrite=a.overwrite)
    pan_ship, _, _, _ = C.build_panels()
    hv = {f: C.half_of_rows(pan_ship, f) for f in A.FOLDS}
    fits: dict = {}
    X = {"BL": {L: {} for L in range(4)}, "P": {}}
    for f in A.FOLDS:
        for L in range(4):
            X["BL"][L][f] = stitched(f, L, hv[f], fits)
    X["P"] = {L: C.load_vec(C.PLAIN[L]) for L in range(4)}

    # ---- the record's pricing, per step and arm (and L0 board-like over member 1, record only)
    priced = {}
    for arm in ("BL", "P"):
        for L in (1, 2, 3):
            priced[f"{arm}-{C.STEP[L]}"] = R.price(pan_ship, X[arm][L], X[arm][L - 1],
                                                   f"{arm} {C.STEP[L]} over its predecessor")
    priced["BL0-over-P0"] = R.price(pan_ship, X["BL"][0], X["P"][0],
                                    "the half-days over member 1 (record only)")

    # ---- per half and per fold, s2/row on the rows each half's fit scored
    per_half = {}
    for f in A.FOLDS:
        p = pan_ship[f]
        y, pb = p.y_true.to_numpy("float64"), p.pbase.to_numpy("float64")
        per_half[f] = {}
        for L in (1, 2, 3):
            ent = {}
            for arm in ("BL", "P"):
                d0, d1 = X[arm][L - 1][f], X[arm][L][f]
                r = y - pb - d0
                inc = d1 - d0
                v = 2 * r * inc - inc * inc
                ent[arm] = {C.HALF_NAME[h] + "_rows": float(v[hv[f] == h].mean()) for h in (0, 1)}
                ent[arm]["whole_fold"] = float(v.mean())
                ent[arm]["rows_moved"] = int((inc != 0).sum())
            ent["ratio_whole_fold"] = ent["BL"]["whole_fold"] / ent["P"]["whole_fold"]
            ent["ratio_by_half"] = {k: ent["BL"][k] / ent["P"][k] for k in ("A_rows", "B_rows")}
            per_half[f][C.STEP[L]] = ent

    # ---- the paired day bootstrap
    rng = np.random.default_rng(SEED)
    draws, codes, cnt = {}, {}, {}
    for f in A.FOLDS:
        cd, uq = pd.factorize(pan_ship[f].day.to_numpy())
        codes[f], k = cd, len(uq)
        draws[f] = rng.integers(0, k, size=(N_BOOT, k))
        c = np.bincount(cd, minlength=k).astype("float64")
        cnt[f] = (c.sum(), c[draws[f]].sum(1))
    mse = {arm: {L: {} for L in range(4)} for arm in ("BL", "P")}
    mse_pt = {arm: {L: {} for L in range(4)} for arm in ("BL", "P")}
    for arm in ("BL", "P"):
        for L in range(4):
            for f in A.FOLDS:
                p = pan_ship[f]
                e = (p.y_true.to_numpy("float64") - p.pbase.to_numpy("float64") - X[arm][L][f]) ** 2
                s = np.bincount(codes[f], weights=e, minlength=draws[f].shape[1])
                mse_pt[arm][L][f] = float(s.sum() / cnt[f][0])
                mse[arm][L][f] = s[draws[f]].sum(1) / cnt[f][1]
    gain, gain_pt, fold_g, fold_g_pt = {}, {}, {}, {}
    for arm in ("BL", "P"):
        for L in (1, 2, 3):
            b0 = board_from_mse(mse[arm][L - 1])
            b1 = board_from_mse(mse[arm][L])
            gain[(arm, L)] = b0 - b1
            gain_pt[(arm, L)] = float(np.sqrt(sum(BA.SHARES[f] * mse_pt[arm][L - 1][f] for f in BOARD_FOLDS))
                                      - np.sqrt(sum(BA.SHARES[f] * mse_pt[arm][L][f] for f in BOARD_FOLDS)))
            rp = priced[f"{arm}-{C.STEP[L]}"]["board_all_row"]["gain_s"]
            if abs(gain_pt[(arm, L)] - rp) > 1e-9:
                raise SystemExit(f"{arm} {C.STEP[L]}: the bootstrap's point {gain_pt[(arm, L)]!r} "
                                 f"is not rung511.price's {rp!r}")
            for f in A.FOLDS:
                fold_g[(arm, L, f)] = mse[arm][L - 1][f] - mse[arm][L][f]
                fold_g_pt[(arm, L, f)] = mse_pt[arm][L - 1][f] - mse_pt[arm][L][f]

    def ci(v):
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    steps = {}
    T_draws = {}
    for L in (1, 2, 3):
        s = C.STEP[L]
        T_draws[L] = gain[("BL", L)] / gain[("P", L)]
        Tpt = gain_pt[("BL", L)] / gain_pt[("P", L)]
        Tci = ci(T_draws[L])
        obs = C.BOARD[s]["transfer"]
        steps[s] = {
            "levels": {f: C.LEVELS[L][f] for f in A.FOLDS},
            "board_s": {"BL": gain_pt[("BL", L)], "P": gain_pt[("P", L)],
                        "BL_ci95": ci(gain[("BL", L)]), "P_ci95": ci(gain[("P", L)]),
                        "P_share_of_draws_nonpositive": float((gain[("P", L)] <= 0).mean())},
            "T_board_like_over_plain": Tpt, "T_ci95": Tci,
            "board_transfer_read_only": obs,
            "board_transfer_assignment_rounded": C.BOARD_ASSIGNMENT[s],
            "contains_board_transfer": bool(Tci[0] <= obs <= Tci[1]),
            "contains_assignment_rounded": bool(Tci[0] <= C.BOARD_ASSIGNMENT[s] <= Tci[1]),
            "predicted_board_gain_s_at_K10_face_record_only": Tpt * C.BOARD[s]["face_s"],
            "board_gain_s_read_only": C.BOARD[s]["board_gain_s"],
            "by_fold_s2_per_row": {
                f: {"BL": fold_g_pt[("BL", L, f)], "P": fold_g_pt[("P", L, f)],
                    "BL_ci95": ci(fold_g[("BL", L, f)]), "P_ci95": ci(fold_g[("P", L, f)]),
                    "ratio": fold_g_pt[("BL", L, f)] / fold_g_pt[("P", L, f)],
                    "ratio_ci95": ci(fold_g[("BL", L, f)] / fold_g[("P", L, f)]),
                    "board_weight": BA.SHARES.get(f, 0.0)} for f in A.FOLDS},
            "per_half": {f: per_half[f][s] for f in A.FOLDS},
            "rows_moved": {arm: priced[f"{arm}-{s}"]["rows_moved"] for arm in ("BL", "P")},
        }
    # SPEC I1.3, record only: the member-inflated companion interval.  It cannot turn a FAIL
    # into a PASS; it says whether a FAIL is within fit-to-fit noise.
    ms = json.loads((C.ANALYSIS / "memberspread.json").read_text())["steps"]
    for L in (1, 2, 3):
        s = C.STEP[L]
        e = steps[s]
        lo, hi = e["T_ci95"]
        sdl = ms[s]["sd_log"]
        if lo > 0 and hi > 0 and sdl is not None and e["T_board_like_over_plain"] > 0:
            se_day = (np.log(hi) - np.log(lo)) / 3.92
            se_tot = float(np.sqrt(se_day ** 2 + 2 * sdl ** 2))
            t = e["T_board_like_over_plain"]
            iv = [float(t * np.exp(-1.96 * se_tot)), float(t * np.exp(1.96 * se_tot))]
            e["member_inflated_record_only"] = {
                "se_day_log": float(se_day), "sd_log_member_plain": sdl, "se_total_log": se_tot,
                "ci95": iv, "contains_board_transfer": bool(iv[0] <= e["board_transfer_read_only"] <= iv[1]),
                "gates": False}
        else:
            e["member_inflated_record_only"] = {"ci95": None, "why": "a non-positive bound", "gates": False}
    disc = T_draws[2] / T_draws[1]
    d_obs = C.BOARD["v21"]["transfer"] / C.BOARD["v20"]["transfer"]
    dci = ci(disc)
    passed = all(steps[s]["contains_board_transfer"] for s in steps)
    out = {
        "section": "exp-051 SPEC I1.2-I1.3", "seed": SEED, "n_boot": N_BOOT,
        "K": "K = 1 on both arms (member 1, placebo seed 1); the board steps were K = 10 bags",
        "statistic": ("T_L = board-seconds gain of the board-like step over its predecessor / "
                      "the plain member-1 step's, july_loo + january at base.SHARES"),
        "steps": steps,
        "v21_discriminator": {"T21_over_T20": float(steps["v21"]["T_board_like_over_plain"]
                                                    / steps["v20"]["T_board_like_over_plain"]),
                              "ci95": dci, "board_read_only": d_obs,
                              "contains_board": bool(dci[0] <= d_obs <= dci[1]),
                              "interval_below_1": bool(dci[1] < 1.0)},
        "verdict": "PASS" if passed else "FAIL",
        "steps_failing": [s for s in steps if not steps[s]["contains_board_transfer"]],
        "half_days_over_member1_record_only": {
            "board_s": priced["BL0-over-P0"]["board_all_row"]["gain_s"],
            "board_ci95": priced["BL0-over-P0"]["board_all_row"]["interval"]["ci95"],
            "by_fold_s2": {f: priced["BL0-over-P0"]["by_fold"][f]["mse_gain_s2"] for f in A.FOLDS},
            "by_fold_ci95": {f: priced["BL0-over-P0"]["by_fold"][f]["ci95"] for f in A.FOLDS}},
        "rung511_price": {k: {"board_all_row": v["board_all_row"], "board_clean": v["board_clean"],
                              "by_fold": v["by_fold"], "by_fold_clean": v["by_fold_clean"],
                              "every_fold_interval_clear": v["every_fold_interval_clear"],
                              "rows_moved": v["rows_moved"]} for k, v in priced.items()},
        "fits": fits,
        "plain": {L: {"path": str(C.PLAIN[L].relative_to(C.ROOT)), "sha256": C.sha256(C.PLAIN[L])}
                  for L in range(4)},
        "seconds": round(time.time() - t0, 1)}
    C.write_json(OUT, out, overwrite=a.overwrite)
    for s, e in steps.items():
        print(f"{s}: BL {e['board_s']['BL']:+.4f} {['%+.4f' % x for x in e['board_s']['BL_ci95']]}  "
              f"P {e['board_s']['P']:+.4f}  T {e['T_board_like_over_plain']:.3f} "
              f"[{e['T_ci95'][0]:.3f}, {e['T_ci95'][1]:.3f}]  board {e['board_transfer_read_only']:.3f}"
              f"  contains {e['contains_board_transfer']}  (member-inflated, record only: "
              f"{e['member_inflated_record_only']['ci95']})")
        for f in A.FOLDS:
            g = e["by_fold_s2_per_row"][f]
            print(f"     {f:9} BL {g['BL']:+8.2f}  P {g['P']:+8.2f}  ratio {g['ratio']:.3f} "
                  f"[{g['ratio_ci95'][0]:.3f}, {g['ratio_ci95'][1]:.3f}]")
    print(f"v21 discriminator T21/T20 {out['v21_discriminator']['T21_over_T20']:.3f} {dci} "
          f"board {d_obs:.3f}")
    print(f"VERDICT {out['verdict']}  failing {out['steps_failing']}")
    print(f"written to {OUT.relative_to(C.ROOT)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
