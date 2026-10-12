"""RESULTS section 47.66: lever C stages 0-2 -- the cell-constant residual, priced with NO fit.

recorded rule hands this lane the with-clock cell-constant residual on the shipped airport x 10-minute
grid.  47.65 registered the design and the argument that a realisation factor BORROWED from
queue's arm N (a different class, a different relation) is the wrong instrument, because the
recoverable fraction of a cell-constant ceiling has a closed form:

    r_i = mu_c + e_i in cell c,  tau^2 = Var(mu),  sigma^2 = Var(e)
    s     = tau^2 / (tau^2 + sigma^2)                 the ORACLE ceiling (mu_c known)
    rho   = s / (1 - s)
    lambda*(n) = (n-1) rho / ((n-1) rho + 1)          the recoverable fraction at cell size n

`lambda*` is applied PER ROW at that row's own cell size and then averaged, never by plugging a
mean n into a non-linear function.  Rows in singleton cells recover nothing and are carried at
lambda* = 0 rather than dropped, because dropping them would price a population nobody serves.

Stage 0 counts, stage 1 estimates tau^2 with queue's own self-excluded estimator (imported from
exp-047 by path, not reimplemented), stage 2 turns the two into a board figure.  No LightGBM.

    .venv/bin/python experiments/exp-046-surfacefit/code/leverc.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import base as BA                                                       # noqa: E402

#: queue's estimators, imported BY PATH so the two `base` modules cannot shadow each other.
_spec = importlib.util.spec_from_file_location(
    "queue_base", ROOT / "experiments/exp-047-queue/code/base.py")
QB = importlib.util.module_from_spec(_spec)
sys.modules["queue_base"] = QB
_spec.loader.exec_module(QB)

ARM = EXP / "work/d-seasonal-bag10-255-free.npz"
OUT = EXP / "analysis/leverc-stage012.json"
BIN = QB.BIN                        # 600 s, the shipped grid's cell
SEED = 47
BODY = 3600.0
#: 47.30.2's board proxy: december carries no board share.
CJ = {"july_loo": 0.0009318464575330257, "january": 0.00074096, "december": 0.0}
FLOOR_S = 0.1


def main() -> None:
    if BIN != 600:
        raise SystemExit(f"queue's BIN is {BIN}, not the 10-minute grid this lever names")
    pan = BA.load()
    arm = np.load(ARM)
    rng = np.random.default_rng(SEED)
    out = {"what": __doc__.strip().splitlines()[0], "bin_s": BIN, "seed": SEED,
           "arm": str(ARM.relative_to(ROOT)), "board_proxy": CJ, "by_fold": {}}

    for f in BA.FOLDS:
        p = pan[f]
        y = p.y_true.to_numpy("float64")
        d_arm = arm[f].astype("float64")
        if len(d_arm) != len(p):
            raise SystemExit(f"{f}: arm has {len(d_arm)} rows, the panel {len(p)}")
        # MY class: the rows this lane's arm actually serves, body only.  Stated as mine --
        # it is NOT queue's row-2 population, so `s` here is the same STATISTIC on a
        # different population and is not required to reproduce the census's 0.0622/0.0769.
        cls = (d_arm != 0) & (y <= BODY)
        r = p.rbase.to_numpy("float64") - d_arm          # what the head AND arm still leave
        apt = p.airport.astype(str).to_numpy()
        bins = np.floor(p.t.to_numpy("float64") / BIN).astype("int64")
        code, ncode = QB.cellcode(apt[cls], bins[cls])
        rr = r[cls]
        days = p.day.to_numpy()[cls]

        # ---- stage 0: how many rows share a cell -------------------------------------
        cnt = np.bincount(code, minlength=ncode)
        per_row = cnt[code]                              # this row's own cell size
        st0 = {"class_rows": int(cls.sum()), "fold_rows": int(len(p)), "cells": int(ncode),
               "mean_rows_per_cell": float(cnt.mean()),
               "median_rows_per_cell": float(np.median(cnt)),
               "row_weighted_mean_cell_size": float(per_row.mean()),
               "singleton_cells": int((cnt == 1).sum()),
               "rows_in_singleton_cells": int((per_row == 1).sum()),
               "by_airport": {}}
        for a in sorted(set(apt[cls])):
            m = apt[cls] == a
            st0["by_airport"][a] = {
                "rows": int(m.sum()), "cells": int(len(set(code[m]))),
                "row_weighted_mean_cell_size": float(per_row[m].mean())}

        # ---- stage 1: the shared component, queue's estimator, unchanged --------------
        sh = QB.shared(rr, code, ncode, days, rng)
        mean_r2 = float(np.mean(rr ** 2))
        tau2 = {"point": float(sh["value"]), "lo": float(sh["ci"][0]),
                "hi": float(sh["ci"][1])}
        st1 = {"shared_component_s2": sh, "mean_r2_s2": mean_r2,
               "reachable_share": {k: v / mean_r2 for k, v in tau2.items()},
               "census_row2_for_contrast": [0.0622, 0.0769]}

        # ---- stage 2: lambda* per row, then the board figure --------------------------
        st2 = {}
        for lab, t2 in tau2.items():
            s = t2 / mean_r2
            rho = s / (1.0 - s)
            lam = (per_row - 1) * rho / ((per_row - 1) * rho + 1.0)   # 0 in singletons
            gain_class = float(np.mean(lam) * t2)          # s^2 per CLASS row
            gain_fold = gain_class * st0["class_rows"] / st0["fold_rows"]
            st2[lab] = {"reachable_share": s, "rho": rho,
                        "mean_lambda_star": float(lam.mean()),
                        "lambda_star_at_the_row_weighted_n": float(
                            (st0["row_weighted_mean_cell_size"] - 1) * rho
                            / ((st0["row_weighted_mean_cell_size"] - 1) * rho + 1.0)),
                        "recoverable_s2_per_class_row": gain_class,
                        "recoverable_s2_per_fold_row": gain_fold,
                        "oracle_ceiling_s2_per_fold_row": t2 * st0["class_rows"]
                                                          / st0["fold_rows"]}
        out["by_fold"][f] = {"stage0_count": st0, "stage1_shared": st1, "stage2_lambda": st2}

    # ---- the board figure, 47.30.2's share weighting ---------------------------------
    board = {}
    for lab in ("point", "lo", "hi"):
        board[lab] = sum(CJ[f] * out["by_fold"][f]["stage2_lambda"][lab]
                         ["recoverable_s2_per_fold_row"] for f in BA.FOLDS)
    ceil = {lab: sum(CJ[f] * out["by_fold"][f]["stage2_lambda"][lab]
                     ["oracle_ceiling_s2_per_fold_row"] for f in BA.FOLDS)
            for lab in ("point", "lo", "hi")}
    # THE CORRECTION 47.65 NEEDED.  `lambda*` bounds a predictor that estimates a cell's
    # constant from the OTHER LABELLED ROWS OF THAT CELL.  The ranking months are 2026-01
    # and 2026-07; the training data is 2025-01..2025-12; and 0 ranking ids appear in any
    # training file.  A ranking row's cell therefore contains NO labelled row, so the
    # oracle bound is a FOLD-SIDE quantity that the board cannot reach even in principle.
    # What the board can reach is a predictor of the cell constant from FEATURES that exist
    # on unlabelled rows -- which is what arm N is -- and the only calibration this
    # programme owns for that is arm N's realised fraction of its own ceiling.
    ARM_N = (0.049, 0.089)                       # queue 7ca vi, the corrected factor
    board_rel = {"lo": ceil["lo"] * ARM_N[0], "point_lo": ceil["point"] * ARM_N[0],
                 "point_hi": ceil["point"] * ARM_N[1], "hi": ceil["hi"] * ARM_N[1]}
    out["lever_c_board_s"] = {
        "oracle_ceiling": ceil,
        "fold_side_oracle_bound": dict(
            board, note=("what a shrunk LOO CELL-MEAN arm would read IN FOLD. It uses the "
                         "cell's other LABELLED rows and is NOT reachable on the board")),
        "board_relevant_band": dict(
            board_rel, basis="my measured ceiling x arm N's realised fraction (7ca vi)",
            note="a predictor of the cell constant from features available on unlabelled rows"),
        "ranking_rows_with_a_labelled_neighbour": 0,
        "floor_s": FLOOR_S,
        "fold_side_bound_clears_the_floor": bool(board["lo"] >= FLOOR_S),
        "BOARD_RELEVANT_clears_the_floor_at_its_bottom": bool(board_rel["lo"] >= FLOOR_S),
        "registered_as": ("RESULTS 47.66, the prediction for lever C before any fit; the "
                          "board-relevant band is the one the floor is applied to")}
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))

    print(f"written {OUT.relative_to(ROOT)}\n")
    print("STAGE 0 -- rows per occupied cell on my class (arm != 0, body y <= 3600)")
    for f in BA.FOLDS:
        s0 = out["by_fold"][f]["stage0_count"]
        print(f"  {f:10} class {s0['class_rows']:7,} of {s0['fold_rows']:7,}  cells "
              f"{s0['cells']:6,}  mean {s0['mean_rows_per_cell']:5.2f}  median "
              f"{s0['median_rows_per_cell']:4.1f}  ROW-WEIGHTED n {s0['row_weighted_mean_cell_size']:6.2f}"
              f"  singleton rows {s0['rows_in_singleton_cells']:5,}")
    print("\nSTAGE 1 -- the shared component, queue's self-excluded estimator")
    for f in BA.FOLDS:
        s1 = out["by_fold"][f]["stage1_shared"]
        sh, rs = s1["shared_component_s2"], s1["reachable_share"]
        print(f"  {f:10} tau^2 {sh['value']:9.1f} [{sh['ci'][0]:.1f}, {sh['ci'][1]:.1f}]  "
              f"mean r^2 {s1['mean_r2_s2']:9.1f}  s = {rs['point']:.4f} "
              f"[{rs['lo']:.4f}, {rs['hi']:.4f}]   pairs {sh['pairs']:,}")
    print("\nSTAGE 2 -- lambda* per row, and the board figure")
    for f in BA.FOLDS:
        s2 = out["by_fold"][f]["stage2_lambda"]["point"]
        print(f"  {f:10} mean lambda* {s2['mean_lambda_star']:.4f}  "
              f"recoverable {s2['recoverable_s2_per_class_row']:8.2f} s2/class row  "
              f"-> {s2['recoverable_s2_per_fold_row']:8.2f} s2/fold row")
    lc = out["lever_c_board_s"]
    print(f"\n  ORACLE CEILING  {ceil['point']:+.4f} s  [{ceil['lo']:+.4f}, {ceil['hi']:+.4f}]")
    print(f"  RECOVERABLE     {board['point']:+.4f} s  [{board['lo']:+.4f}, {board['hi']:+.4f}]")
    br = lc["board_relevant_band"]
    print(f"     ^ fold-side ORACLE bound: uses the cell's other LABELLED rows.")
    print(f"       0 of the ranking rows have a labelled neighbour (2026 months, 2025 "
          f"training), so the board cannot reach it even in principle.")
    print(f"  BOARD-RELEVANT   +{br['lo']:.4f} .. +{br['hi']:.4f} s  "
          f"(ceiling x arm N's realised 0.049-0.089)")
    print(f"  floor {FLOOR_S} s -- fold-side bound clears: "
          f"{lc['fold_side_bound_clears_the_floor']};  BOARD-RELEVANT clears at its bottom: "
          f"{lc['BOARD_RELEVANT_clears_the_floor_at_its_bottom']}")


if __name__ == "__main__":
    main()
