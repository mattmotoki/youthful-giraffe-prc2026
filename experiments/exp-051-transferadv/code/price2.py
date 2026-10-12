"""exp-051 iteration 2: price the June-carrier stack against the plain one (SPEC I2).  ZERO FIT.

Arms (july_loo carries the only change; january and december take the plain member of the
same seed at the same level, so their increments are identical in both arms):

    P   plain members s = 1..3 (`memberspread.member_path`)
    C   June carrier, s = 1..3 (`carrier.py`; C0 at s = 1 is exp-046's only202506)
    S   the sham, s = 1: June's rbase permuted within airport

K = 3 bags are built by exp-046's `bagarm.build_bag`, the definition every shipped bag used.
The registered statistics (SPEC I2):

    rho_L = G_july(C3, L) / G_july(P3, L)          G = mean(2 r inc - inc^2), s2 per row
    T2_L  = board gain of the C3 step / board gain of the P3 step   (july_loo + january, SHARES)

with a paired day bootstrap, seed 5102, 2,000 draws, shared by every arm and step.  Record only:
the K = 1 readings, the three members' own rho, and the June step over v22's july stack under
the plain arm and under iteration 1's half-day arm.

    .venv/bin/python experiments/exp-051-transferadv/code/price2.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common as C
from common import A, BA, NO, R
import bagarm as BAG                                                  # noqa: E402
import memberspread as MS
import carrier as CR

OUT = C.ANALYSIS / "carrier.json"
SEED = 5102
N_BOOT = 2000
BOARD_FOLDS = ("july_loo", "january")
K3 = [1, 2, 3]
F = "july_loo"
CUT_R1 = 0.5


def fit_vec(tag: str) -> np.ndarray:
    npz, js = CR.paths(tag)
    rec = json.loads(js.read_text())
    if rec["vector_sha256"] != C.sha256(npz):
        raise SystemExit(f"{npz.name}: sha256 is not the one its fit record carries")
    return np.load(npz)[F].astype("float64")


def arm_vec(kind: str, s: int, L: int) -> dict:
    out = dict(C.load_vec(MS.member_path(L, s)))
    if kind == "P":
        return out
    if kind == "C":
        out[F] = (C.load_vec(CR.ONLY0306)[F] if (s == 1 and L == 0) else fit_vec(f"C-s{s}-L{L}"))
    elif kind == "S":
        out[F] = fit_vec(f"S-s1-L{L}")
    return out


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    NO.refuse_existing([OUT], overwrite=a.overwrite)
    for tag in ("integrity-C0-s1", "integrity-P1-s2"):
        rep = json.loads((C.ANALYSIS / f"i2-{tag}.json").read_text())
        if not rep["passes_1e-9"]:
            raise SystemExit(f"{tag} did not pass; nothing is priced")
    pan_ship, _, _, _ = C.build_panels()
    r0 = {f: pan_ship[f].y_true.to_numpy("float64") - pan_ship[f].pbase.to_numpy("float64")
          for f in A.FOLDS}

    V = {}                                            # V[name][L] = {fold: vector}
    for kind in ("P", "C"):
        for s in K3:
            V[f"{kind}{s}"] = {L: arm_vec(kind, s, L) for L in range(4)}
    V["S1"] = {L: arm_vec("S", 1, L) for L in range(4)}
    for kind in ("P", "C"):
        V[f"{kind}3bag"] = {L: BAG.build_bag({s: V[f"{kind}{s}"][L] for s in K3}, pan_ship, K3)[0]
                            for L in range(4)}
    for L in range(4):                                # january and december are the same arm
        for f in ("january", "december"):
            if not np.array_equal(V["C3bag"][L][f], V["P3bag"][L][f]):
                raise SystemExit(f"L{L} {f}: the carrier bag differs from the plain bag off july")
    # June, record only: P4 = P3 + June; BL4 = iteration 1's BL3 + June, stitched as in price.py
    hv = C.half_of_rows(pan_ship, F)
    bl3 = np.where(hv == 1, C.load_vec(C.fit_path(F, 0, 3)[0])[F], C.load_vec(C.fit_path(F, 1, 3)[0])[F])
    bl4 = np.where(hv == 1, fit_vec("June-BL4-A"), fit_vec("June-BL4-B"))
    P3 = V["P1"][3]
    V["J-P"] = {3: P3, 4: {**P3, F: fit_vec("June-P4-s1")}}
    V["J-BL"] = {3: {**P3, F: bl3}, 4: {**P3, F: bl4}}

    # ---- the paired day bootstrap: per-fold day sums, one set of draws for everything
    rng = np.random.default_rng(SEED)
    draws, codes, cnt, npt = {}, {}, {}, {}
    for f in A.FOLDS:
        cd, uq = pd.factorize(pan_ship[f].day.to_numpy())
        codes[f], k = cd, len(uq)
        draws[f] = rng.integers(0, k, size=(N_BOOT, k))
        c = np.bincount(cd, minlength=k).astype("float64")
        cnt[f], npt[f] = c[draws[f]].sum(1), c.sum()

    def mse(vec: dict) -> tuple[dict, dict]:
        pt, bs = {}, {}
        for f in A.FOLDS:
            e = (r0[f] - vec[f]) ** 2
            sm = np.bincount(codes[f], weights=e, minlength=draws[f].shape[1])
            pt[f], bs[f] = float(sm.sum() / npt[f]), sm[draws[f]].sum(1) / cnt[f]
        return pt, bs

    M = {name: {L: mse(v) for L, v in lv.items()} for name, lv in V.items()}

    def board(m: dict):
        return np.sqrt(sum(BA.SHARES[f] * m[f] for f in BOARD_FOLDS))

    def step(name: str, L: int):
        (p0, b0), (p1, b1) = M[name][L - 1], M[name][L]
        g = {f: (p0[f] - p1[f], b0[f] - b1[f]) for f in A.FOLDS}
        return g, (float(board(p0) - board(p1)), board(b0) - board(b1))

    def ci(v):
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    steps, rho_d, T_d = {}, {}, {}
    for L in (1, 2, 3):
        s = C.STEP[L]
        gC, bC = step("C3bag", L)
        gP, bP = step("P3bag", L)
        for name, (bpt, _) in (("C3bag", bC), ("P3bag", bP)):
            rp = R.price(pan_ship, V[name][L], V[name][L - 1], f"{name} {s}")["board_all_row"]["gain_s"]
            if abs(bpt - rp) > 1e-9:
                raise SystemExit(f"{name} {s}: bootstrap point {bpt!r} is not rung511.price's {rp!r}")
        rho_d[L] = gC[F][1] / gP[F][1]
        T_d[L] = bC[1] / bP[1]
        rho_pt = gC[F][0] / gP[F][0]
        T_pt = bC[0] / bP[0]
        obs = C.BOARD[s]["transfer"]
        Tci = ci(T_d[L])
        # K = 1 and per-member readings, record only
        mem = {}
        for m in K3:
            gCm, bCm = step(f"C{m}", L)
            gPm, bPm = step(f"P{m}", L)
            mem[m] = {"rho": gCm[F][0] / gPm[F][0], "T2": bCm[0] / bPm[0],
                      "july_s2": {"C": gCm[F][0], "P": gPm[F][0]}}
        gS, bS = step("S1", L)
        gP1, bP1 = step("P1", L)
        rhoS_d = gS[F][1] / gP1[F][1]
        steps[s] = {
            "july_s2_per_row": {"C3": gC[F][0], "P3": gP[F][0], "C3_ci95": ci(gC[F][1]),
                                "P3_ci95": ci(gP[F][1]),
                                "P3_clears_zero": bool(ci(gP[F][1])[0] > 0)},
            "january_s2_per_row_both_arms": gP["january"][0],
            "december_s2_per_row_both_arms": gP["december"][0],
            "rho": rho_pt, "rho_ci95": ci(rho_d[L]),
            "board_s": {"C3": bC[0], "P3": bP[0], "C3_ci95": ci(bC[1]), "P3_ci95": ci(bP[1])},
            "T2": T_pt, "T2_ci95": Tci, "board_transfer_read_only": obs,
            "contains_board_transfer": bool(Tci[0] <= obs <= Tci[1]),
            "contains_assignment_rounded": bool(Tci[0] <= C.BOARD_ASSIGNMENT[s] <= Tci[1]),
            "members_record_only": mem,
            "members_rho_sd": float(np.std([mem[m]["rho"] for m in K3], ddof=1)),
            "K1_member1_record_only": mem[1],
            "sham_K1": {"rho_S": gS[F][0] / gP1[F][0], "rho_S_ci95": ci(rhoS_d),
                        "contains_1": bool(ci(rhoS_d)[0] <= 1.0 <= ci(rhoS_d)[1]),
                        "july_s2": {"S": gS[F][0], "P1": gP1[F][0]},
                        "T2_S": bS[0] / bP1[0], "T2_S_ci95": ci(bS[1] / bP1[1])},
        }
    # ---- the registered readings
    e20 = steps["v20"]
    lo, hi = e20["rho_ci95"]
    r1 = ("NOT READ (kill line)" if not e20["july_s2_per_row"]["P3_clears_zero"] else
          "PRESERVES" if lo >= CUT_R1 else "CRUSHES" if hi <= CUT_R1 else "STRADDLES")
    drho = rho_d[1] - rho_d[3]
    dci = ci(drho)
    r1b = "CONTENT-ORDERED" if dci[0] > 0 else "REVERSED" if dci[1] < 0 else "UNIFORM"
    below = [s for s in steps if steps[s]["sham_K1"]["rho_S_ci95"][1] < 1.0]
    above = [s for s in steps if steps[s]["sham_K1"]["rho_S_ci95"][0] > 1.0]
    r2 = ("INFORMATIONAL" if not below and not above else
          f"CAPACITY PART on {below}" if below else f"SHAM RAISES {above}")
    passed = all(steps[s]["contains_board_transfer"] for s in steps)
    disc = T_d[2] / T_d[1]
    d_obs = C.BOARD["v21"]["transfer"] / C.BOARD["v20"]["transfer"]
    # carrier's own value: C0 over P0, july
    own = {}
    for name_c, name_p, k in (("C3bag", "P3bag", "K3"), ("C1", "P1", "K1_member1")):
        p0, b0 = M[name_p][0]
        p1, b1 = M[name_c][0]
        own[k] = {"july_s2": p0[F] - p1[F], "july_ci95": ci(b0[F] - b1[F])}
    # June, record only
    june = {}
    for name in ("J-P", "J-BL"):
        (p0, b0), (p1, b1) = M[name][3], M[name][4]
        bg = board(b0) - board(b1)
        june[name] = {"july_s2": p0[F] - p1[F], "july_ci95": ci(b0[F] - b1[F]),
                      "board_s": float(board(p0) - board(p1)), "board_ci95": ci(bg),
                      "clears_0p1_floor_at_interval_bottom": bool(ci(bg)[0] > 0.1)}
    out = {
        "section": "exp-051 SPEC I2", "seed": SEED, "n_boot": N_BOOT, "K": "K = 3 (members 1-3)",
        "steps": steps,
        "R1_v20_cause": {"rho_v20": e20["rho"], "ci95": e20["rho_ci95"], "cut": CUT_R1,
                         "reading": r1, "iteration1_half_day_rho_v20_july": 0.18725},
        "R1b_content_order": {"rho_v20_minus_rho_v22": float(steps["v20"]["rho"] - steps["v22"]["rho"]),
                              "ci95": dci, "reading": r1b},
        "R2_sham": {"reading": r2, "steps_capacity_part": below, "steps_sham_raises": above},
        "R3_rule": {"verdict": "PASS" if passed else "FAIL",
                    "steps_failing": [s for s in steps if not steps[s]["contains_board_transfer"]]},
        "R4_v21_discriminator": {"T2_v21_over_T2_v20": float(steps["v21"]["T2"] / steps["v20"]["T2"]),
                                 "ci95": ci(disc), "board_read_only": d_obs,
                                 "contains_board": bool(ci(disc)[0] <= d_obs <= ci(disc)[1]),
                                 "interval_below_1": bool(ci(disc)[1] < 1.0)},
        "carrier_own_value_record_only": own,
        "june_record_only": june,
        "seconds": round(time.time() - t0, 1)}
    C.write_json(OUT, out, overwrite=a.overwrite)
    for s, e in steps.items():
        print(f"{s}: july C3 {e['july_s2_per_row']['C3']:+8.2f}  P3 {e['july_s2_per_row']['P3']:+8.2f}"
              f"  rho {e['rho']:.3f} [{e['rho_ci95'][0]:.3f}, {e['rho_ci95'][1]:.3f}]  "
              f"T2 {e['T2']:.3f} [{e['T2_ci95'][0]:.3f}, {e['T2_ci95'][1]:.3f}] board "
              f"{e['board_transfer_read_only']:.3f} contains {e['contains_board_transfer']}  "
              f"members rho {[round(e['members_record_only'][m]['rho'], 3) for m in K3]}  "
              f"sham rho {e['sham_K1']['rho_S']:.3f} {[round(x, 3) for x in e['sham_K1']['rho_S_ci95']]}")
    print(f"R1 {r1} (rho_v20 {e20['rho']:.3f} {e20['rho_ci95']})  R1b {r1b} {dci}  R2 {r2}  "
          f"R3 {out['R3_rule']}  R4 {out['R4_v21_discriminator']['T2_v21_over_T2_v20']:.3f} {ci(disc)}")
    print(f"carrier own value {own}")
    print(f"June {june}")
    print(f"written to {OUT.relative_to(C.ROOT)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
