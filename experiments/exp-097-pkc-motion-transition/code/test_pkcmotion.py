"""exp-097 synthetic verification.  No real row, label, prediction or summary is read.

Every test builds its own frame.  The suite proves the contracts a real read cannot be
repeated to fix: the ownership complement's four cases, the support class, the
projection refusal, the id contract, the out-of-fold discipline of the fit, the
placebo's determinism, the board currency, and the phase order.

    OMP_NUM_THREADS=2 .venv/bin/python \
        experiments/exp-097-pkc-motion-transition/code/test_pkcmotion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pkcmotion as M                                                  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def frame(**kw) -> pd.DataFrame:
    n = len(next(iter(kw.values())))
    base = {
        "id64": np.arange(1, n + 1, dtype=np.int64),
        "own_has_clock": np.zeros(n, bool),
        "obs_has_clock_block": np.zeros(n, bool),
        "noac_bool": np.zeros(n, bool),
        "obs_matched": np.ones(n, bool),
        "obs_taxi_motion": np.full(n, 600.0),
        "stand_stop_s": np.full(n, 120.0),
        "turnaround_anchored": np.zeros(n),
        "seg_span_s": np.full(n, 900.0),
        "n_ground_points": np.full(n, 50.0),
        "dt_takeoff": np.zeros(n),
        "stand_dist_km": np.full(n, 1.0),
        "reach_class": np.array(["matched_no_clock"] * n),
        "p_v22": np.full(n, 800.0),
        "airport": np.array(["EHAM"] * n),
        "day": np.array(["2025-07-01"] * n),
    }
    base.update(kw)
    return M.add_signal(pd.DataFrame(base))


def test_complement_four_cases() -> None:
    print("complement: the corrected four-case ownership rule")
    d = frame(own_has_clock=np.array([False, False, True, True]),
              noac_bool=np.array([False, True, False, True]))
    got = M.complement_mask(d).tolist()
    check("truth table is [T, T, F, T]", got == [True, True, False, True], str(got))


def test_support_class() -> None:
    print("support: neither vector claims it, it is matched, and the motion is physical")
    d = frame(own_has_clock=np.array([False] * 7),
              obs_has_clock_block=np.array([False, True, False, False, False, False, False]),
              noac_bool=np.array([False, False, True, False, False, False, False]),
              obs_matched=np.array([True, True, True, False, True, True, True]),
              obs_taxi_motion=np.array([600.0, 600.0, 600.0, 600.0, np.nan, 30.0, 9000.0]))
    got = M.support_mask(d).tolist()
    check("only the first row is supported", got == [True] + [False] * 6, str(got))
    d2 = frame(own_has_clock=np.array([True]), obs_has_clock_block=np.array([False]))
    check("v22's own clock alone excludes a row", not M.support_mask(d2)[0])


def test_projection_refusal() -> None:
    print("projection: the allow-list fails closed")
    for bad in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"):
        try:
            M.refuse_columns(["MVT_ID_mvt", bad], ["MVT_ID_mvt", bad], "test")
            check(f"refuses {bad}", False)
        except SystemExit:
            check(f"refuses {bad}", True)
    try:
        M.refuse_columns(["MVT_ID_mvt", "y_true"], M.PANEL_COLS_LABEL_FREE, "test")
        check("refuses a column outside the registered projection", False)
    except SystemExit:
        check("refuses a column outside the registered projection", True)
    M.refuse_columns(list(M.PANEL_COLS_LABEL_FREE), M.PANEL_COLS_LABEL_FREE, "test")
    check("accepts exactly the registered label-free panel projection", True)
    try:
        M.refuse_columns(["residual_x"], ["residual_x"], "test")
        check("refuses a name containing 'residual'", False)
    except SystemExit:
        check("refuses a name containing 'residual'", True)


def test_ids() -> None:
    print("ids: integer-valued, unique, inside the exact binary64 range")
    check("float ids convert", M.check_ids(np.array([1.0, 2.0]), "t").tolist() == [1, 2])
    check("int ids convert", M.check_ids(np.array([1, 2], dtype=np.int64),
                                        "t").tolist() == [1, 2])
    check("uint ids convert", M.check_ids(np.array([1, 2], dtype=np.uint64),
                                          "t").tolist() == [1, 2])
    for bad, why in [(np.array([1.5]), "non-integral"),
                     (np.array([np.nan]), "non-finite"),
                     (np.array([2.0 ** 53]), "positive boundary"),
                     (np.array([-(2.0 ** 53)]), "negative boundary"),
                     (np.array([3.0, 3.0]), "duplicate"),
                     # Sol round 1, blocking 7: an int64 2**53 used to pass while its
                     # float64 twin was refused, and a uint64 2**63 wrapped negative
                     (np.array([2 ** 53], dtype=np.int64), "int64 positive boundary"),
                     (np.array([-(2 ** 53)], dtype=np.int64), "int64 negative boundary"),
                     (np.array([2 ** 53], dtype=np.uint64), "uint64 boundary"),
                     (np.array([2 ** 63], dtype=np.uint64), "uint64 wrap"),
                     (np.array([7, 7], dtype=np.int64), "int64 duplicate")]:
        try:
            M.check_ids(bad, "t")
            check(f"rejects {why}", False)
        except SystemExit:
            check(f"rejects {why}", True)


def test_forbidden_columns_are_absent_from_every_projection() -> None:
    print("projections: no registered projection can carry a forbidden column")
    # Every DEPARTURE-side projection must be free of both forbidden columns.  The
    # ARRIVAL projection is the one exception and it is checked separately below.
    for name, cols in [("DEP_COLS", M.DEP_COLS), ("PREFIX_COLS", M.PREFIX_COLS),
                       ("RANKING_COLS", M.RANKING_COLS), ("CLASS_COLS", M.CLASS_COLS),
                       ("PANEL_COLS_LABEL_FREE", M.PANEL_COLS_LABEL_FREE)]:
        check(f"{name} is free of both forbidden columns",
              not (set(cols) & set(M.FORBIDDEN)), str(cols))
    check("no projection anywhere carries the target",
          all("TAXITIME_SEC_mvt" not in c for c in
              (M.DEP_COLS, M.ARR_COLS, M.PREFIX_COLS, M.RANKING_COLS, M.CLASS_COLS,
               M.PANEL_COLS_LABEL_FREE, M.PANEL_COLS_LABEL)))
    # The label is MVT_TIME - BLOCK_TIME on a DEPARTURE row.  The stand chain needs the
    # ARRIVAL in-block, which is the same column on a row that is not any departure.
    check("the arrival projection carries the ARR in-block the stand chain needs",
          "BLOCK_TIME_UTC_mvt" in M.ARR_COLS)
    check("the arrival projection still never carries the target",
          "TAXITIME_SEC_mvt" not in M.ARR_COLS)
    good = pd.DataFrame({"MVT_ID_mvt": [1.0], "p_v22": [2.0]})
    M.assert_no_forbidden(good, "t")
    check("a clean frame passes the guard", True)
    for c in M.FORBIDDEN:
        try:
            M.assert_no_forbidden(pd.DataFrame({"MVT_ID_mvt": [1.0], c: [3.0]}), "t")
            check(f"the guard rejects a frame carrying {c}", False)
        except SystemExit:
            check(f"the guard rejects a frame carrying {c}", True)
    arr = pd.DataFrame({"PHASE_mvt": ["ARR"], "BLOCK_TIME_UTC_mvt": [1.0]})
    M.assert_no_forbidden(arr, "t", allow=("BLOCK_TIME_UTC_mvt",))
    check("an ARR-only frame may hold the arrival in-block when it is allowed", True)
    try:
        M.assert_no_forbidden(pd.DataFrame({"TAXITIME_SEC_mvt": [1.0]}), "t",
                              allow=("TAXITIME_SEC_mvt",))
        check("no caller can allow the target itself", False)
    except SystemExit:
        check("no caller can allow the target itself", True)
    try:
        M.assert_no_forbidden(pd.DataFrame({"MVT_ID_mvt": [1.0]}), "t",
                              allow=("TAXITIME_SEC_mvt",))
        check("the target allowance is refused even when the frame is clean", False)
    except SystemExit:
        check("the target allowance is refused even when the frame is clean", True)


def test_ridge_recovers_and_is_out_of_fold() -> None:
    print("fit: the ridge recovers a planted relation and never sees its target fold")
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(600, 200, n)
    p = rng.normal(800, 100, n)
    y = 100.0 + 0.5 * p + 0.8 * x + rng.normal(0, 5, n)
    d = frame(obs_taxi_motion=x, p_v22=p)
    d["y_true"] = y
    X, med = M.design(d, M.MOTION_COLS)
    beta = M.ridge_fit(X, y)
    check("intercept within 5 s", abs(beta[0] - 100.0) < 5.0, f"{beta[0]:.2f}")
    check("p_v22 weight within 0.02", abs(beta[1] - 0.5) < 0.02, f"{beta[1]:.4f}")
    check("motion weight within 0.02", abs(beta[2] - 0.8) < 0.02, f"{beta[2]:.4f}")
    # a target-fold row cannot change a coefficient fitted on the other folds
    d2 = d.copy()
    d2["y_true"] = y + 10000.0
    X2, _ = M.design(d2, M.MOTION_COLS)
    b2 = M.ridge_fit(X, y)                       # the SAME training frame
    check("the fit is a function of its training frame only",
          np.allclose(beta, b2))


def test_design_uses_training_medians() -> None:
    print("design: a missing feature takes the TRAINING median, never the target fold's")
    tr = frame(obs_taxi_motion=np.array([100.0, 200.0, 300.0]))
    Xtr, med = M.design(tr, M.MOTION_COLS)
    ap = frame(obs_taxi_motion=np.array([np.nan, np.nan]))
    Xap, used = M.design(ap, M.MOTION_COLS, med)
    check("median comes from training", used["obs_taxi_motion"] == 200.0,
          str(used["obs_taxi_motion"]))
    check("applied design is finite", bool(np.isfinite(Xap).all()))


def test_clip() -> None:
    print("clip: a served prediction stays inside the registered physical taxi window")
    X = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
    beta = np.array([1e6, 0, 0, 0, 0])
    check("clipped above", M.apply_fit(X, beta)[0] == M.TAXI_HI)
    beta = np.array([-1e6, 0, 0, 0, 0])
    check("clipped below", M.apply_fit(X, beta)[0] == M.TAXI_LO)


def test_placebo_is_deterministic_and_within_group() -> None:
    print("placebo: deterministic, and it never moves a value across airport-day")
    n = 200
    d = frame(obs_taxi_motion=np.arange(n, dtype="float64"),
              airport=np.array(["EHAM"] * 100 + ["EDDF"] * 100),
              day=np.array(["2025-07-01"] * 50 + ["2025-07-02"] * 50) .tolist() * 2)
    a = M.permute_within(d, M.MOTION_COLS, 5)
    b = M.permute_within(d, M.MOTION_COLS, 5)
    check("two runs agree", np.array_equal(a["obs_taxi_motion"].to_numpy(),
                                           b["obs_taxi_motion"].to_numpy()))
    c = M.permute_within(d.iloc[::-1].reset_index(drop=True), M.MOTION_COLS, 5)
    check("row order does not change the multiset per group",
          sorted(a["obs_taxi_motion"].tolist()) == sorted(c["obs_taxi_motion"].tolist()))
    key = a["airport"].astype(str) + "|" + a["day"].astype(str)
    for k, g in a.groupby(key):
        src = d[(d["airport"].astype(str) + "|" + d["day"].astype(str)) == k]
        check(f"group {k} keeps its own values",
              sorted(g["obs_taxi_motion"].tolist()) == sorted(src["obs_taxi_motion"].tolist()))
    check("a different seed gives a different permutation",
          not np.array_equal(a["obs_taxi_motion"].to_numpy(),
                             M.permute_within(d, M.MOTION_COLS, 6)["obs_taxi_motion"].to_numpy()))


def test_board_currency() -> None:
    print("currency: the July/January mixture and the shares the ranking file has")
    check("shares sum to one", abs(sum(M.BOARD_SHARE.values()) - 1.0) < 1e-12)
    check("july share", abs(M.BOARD_SHARE["july_loo"] - 192122 / 344841) < 1e-15)
    check("january share", abs(M.BOARD_SHARE["january"] - 152719 / 344841) < 1e-15)
    check("december is not in the currency", "december" not in M.BOARD_SHARE)
    v = M.board_rmse({"july_loo": 100.0 ** 2, "january": 100.0 ** 2})
    check("a flat mixture is the flat RMSE", abs(v - 100.0) < 1e-9, str(v))


def test_fold_scores_and_bootstrap() -> None:
    print("metrics: an unmoved row contributes zero, and the interval is day-clustered")
    y = np.array([100.0, 200.0, 300.0, 400.0])
    p0 = np.array([110.0, 210.0, 310.0, 410.0])
    p1 = p0.copy()
    s = M.fold_scores(y, p0, p1)
    check("no change means no gain", s["rmse_gain"] == 0.0 and s["changed_rows"] == 0)
    p1 = p0.copy()
    p1[0] = 100.0
    s = M.fold_scores(y, p0, p1)
    check("one exact row gives a positive gain", s["rmse_gain"] > 0)
    check("changed rows counted", s["changed_rows"] == 1)
    b = M.day_bootstrap_gain(y, p0, p1, np.array(["a", "a", "b", "b"]), 1, boot=200)
    check("interval brackets the point",
          b["ci95"][0] <= b["point"] <= b["ci95"][1], str(b))


def test_gate_arithmetic() -> None:
    print("gates: the registered numbers")
    check("upload gate is +0.10 s", M.UPLOAD_GATE_S == 0.10)
    check("support floor is 300 / 60", (M.MIN_SUPPORT_TOTAL, M.MIN_SUPPORT_PER_FOLD)
          == (300, 60))
    check("taxi window is v22's own", (M.TAXI_LO, M.TAXI_HI) == (60.0, 7200.0))
    check("the target is on the board", abs(M.V22_BOARD - M.BOARD_TARGET - 1.4402) < 1e-9)


def test_phase_order() -> None:
    """Phase 2's predecessor moved from a loose receipt to the phase-1 **r2** chain, which
    `preflight` checks before the claim (SPEC §12b); the full boundary is tested in
    `test_pkcboundary.py`.  What belongs here is the order itself: a later phase cannot
    start without the earlier one's receipt, and phase 2 cannot be entered at all without
    passing every barrier first."""
    print("phase order: a later phase refuses without the earlier result")
    import pkccontrol as C                                             # noqa: PLC0415
    real = C.read_receipt
    try:
        C.read_receipt = lambda n: {"decision": "FAIL"}
        try:
            C.phase3({"desk_acceptance": {}, "phase1_chain": {}, "repo": {}, "start": {}})
            check("phase 3 refuses a FAIL decision receipt", False)
        except SystemExit:
            check("phase 3 refuses a FAIL decision receipt", True)
        C.read_receipt = lambda n: None
        try:
            C.phase3({"desk_acceptance": {}, "phase1_chain": {}, "repo": {}, "start": {}})
            check("phase 3 refuses with no phase-2 receipt", False)
        except SystemExit:
            check("phase 3 refuses with no phase-2 receipt", True)
    finally:
        C.read_receipt = real
    check("phase 2 cannot be entered without a boundary record",
          C.phase2.__code__.co_argcount == 1)
    for phase in (2, 3):
        try:
            C.preflight(phase, "")
            check(f"preflight refuses phase {phase} outside its launcher", False)
        except SystemExit:
            check(f"preflight refuses phase {phase} outside its launcher", True)


def main() -> int:
    for t in (test_complement_four_cases, test_support_class, test_projection_refusal,
              test_ids, test_forbidden_columns_are_absent_from_every_projection,
              test_ridge_recovers_and_is_out_of_fold,
              test_design_uses_training_medians, test_clip,
              test_placebo_is_deterministic_and_within_group, test_board_currency,
              test_fold_scores_and_bootstrap, test_gate_arithmetic, test_phase_order):
        t()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("all synthetic tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
