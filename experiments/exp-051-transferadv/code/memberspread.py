"""exp-051 iteration 1, record only: the K = 1 limit, measured on the PLAIN side. ZERO FIT.

The board steps were K = 10 bags; this experiment's ratio is read at K = 1 (member 1).  The
day bootstrap does not carry fit-to-fit noise, so this reads it where it can be read without a
fit: the ten plain members of each shipped bag, each priced as its own K = 1 step over its own
predecessor member, and the K = 10 bags themselves (which must reproduce the recorded faces).

    member s:  L0 d-bundle-placebo{s}   L1 d-onemonth-m{s}   L2 d-threemonth-m{s}   L3 d-seasonal-m{s}
    (member 1's L1 and L2 are the screen and plus03 vectors, as onemonthbag.STACKS declares)

It enters no gate.  SPEC I1.3 uses it for one record-only companion interval.

    .venv/bin/python experiments/exp-051-transferadv/code/memberspread.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import common as C
from common import A, BA, NO

OUT = C.ANALYSIS / "memberspread.json"
W = C.E46 / "work"
TEN = list(range(1, 11))
BAGS = {0: W / "d-bag10-255-free.npz", 1: W / "d-onemonth-bag10-255-free.npz",
        2: W / "d-threemonth-bag10-255-free.npz", 3: W / "d-seasonal-bag10-255-free.npz"}
FACES = {1: 0.513473, 2: 0.55876, 3: 0.398006}


def member_path(L: int, s: int):
    if L == 0:
        return W / f"d-bundle-placebo{s}-255-free.npz"
    if s == 1 and L in (1, 2):
        return C.PLAIN[L]
    return W / {1: f"d-onemonth-m{s}-255-free.npz", 2: f"d-threemonth-m{s}-255-free.npz",
                3: f"d-seasonal-m{s}-255-free.npz"}[L]


def step_gain(pan: dict, d0: dict, d1: dict) -> tuple[float, dict]:
    mse0, mse1, per = {}, {}, {}
    for f in A.FOLDS:
        p = pan[f]
        r = p.y_true.to_numpy("float64") - p.pbase.to_numpy("float64")
        mse0[f] = float(np.mean((r - d0[f]) ** 2))
        mse1[f] = float(np.mean((r - d1[f]) ** 2))
        per[f] = mse0[f] - mse1[f]
    b = lambda m: np.sqrt(sum(BA.SHARES[f] * m[f] for f in ("july_loo", "january")))
    return float(b(mse0) - b(mse1)), per


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    NO.refuse_existing([OUT], overwrite=a.overwrite)
    pan_ship, _, _, _ = C.build_panels()
    out = {"section": "exp-051 SPEC I1 record only: plain members as K = 1 steps", "steps": {}}
    for L in (1, 2, 3):
        g, per = [], []
        for s in TEN:
            gs, ps_ = step_gain(pan_ship, C.load_vec(member_path(L - 1, s)),
                                C.load_vec(member_path(L, s)))
            g.append(gs)
            per.append(ps_)
        g = np.array(g)
        k10, per10 = step_gain(pan_ship, C.load_vec(BAGS[L - 1]), C.load_vec(BAGS[L]))
        out["steps"][C.STEP[L]] = {
            "member_board_s": g.tolist(), "mean": float(g.mean()), "sd": float(g.std(ddof=1)),
            "sd_log": float(np.log(g).std(ddof=1)) if (g > 0).all() else None,
            "member1": float(g[0]), "member1_rank_of_10_ascending": int((g < g[0]).sum() + 1),
            "by_fold_s2_member_mean": {f: float(np.mean([p[f] for p in per])) for f in A.FOLDS},
            "by_fold_s2_member_sd": {f: float(np.std([p[f] for p in per], ddof=1)) for f in A.FOLDS},
            "by_fold_s2_member1": per[0],
            "K10_face_board_s": k10, "K10_face_recorded": FACES[L],
            "K10_face_reproduces": bool(abs(k10 - FACES[L]) < 5e-6),
            "K10_by_fold_s2": per10,
            "mean_member_over_K10": float(g.mean() / k10)}
        e = out["steps"][C.STEP[L]]
        print(f"{C.STEP[L]}: members {e['mean']:.4f} +- {e['sd']:.4f} (sd_log {e['sd_log']}), "
              f"member 1 {e['member1']:.4f} (rank {e['member1_rank_of_10_ascending']}), "
              f"K10 {k10:.5f} (recorded {FACES[L]}, reproduces {e['K10_face_reproduces']})")
    out["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, out, overwrite=a.overwrite)
    print(f"written to {OUT.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
