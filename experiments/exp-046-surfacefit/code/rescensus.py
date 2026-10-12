"""Where the remaining error sits after the K = 10 bag -- a residual census (desk, 2026-09-09).

Read-only on candidates and serve records; no fits, no serve, no candidate.  The point is to
find the next arm rather than to guess it, and in particular to answer one question the lane
cannot answer by looking at its own arm: **is the biggest remaining error inside the class
this lane writes, or in the complement queue owns?**  If it is the complement, the next block
routes to queue and not here, and this file says so.

**The prediction being censused** is the one v8 ships on the folds: `pbase + bag10`, with
`bag10` the K = 10 bag, which is zero outside the shipped class -- so the complement's residual
is `y - pbase`, exactly the error queue's arm would have to move.

**Board weighting** follows `base.board_interval`: board MSE is `sum(share_f * MSE_f)` over the
two ranking-share folds, so a cell's board contribution is `sum_f share_f * SSE_cell,f / n_f`
and its SHARE of the board is that over the whole board MSE.  December carries share 0 and is
reported for completeness, never counted.

**The ranking side has no label.**  The only in-sample signal there is what the ten members
disagree about, so the same cuts are reported on the writable population with per-row member
variance in place of squared error.  That is a measure of where the bag is UNSURE, which is
not the same thing as where it is WRONG, and it is labelled as such.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/rescensus.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import guard as G                                                     # noqa: E402
import rung511 as R                                                   # noqa: E402
import bagarm as BAG                                                  # noqa: E402

TEN = list(range(1, 11))
OUT = EXP / "analysis/residual-census.json"
TAIL_S = 3600.0
#: the ranking months the two board shares belong to
RANK_MONTH_SHARE = {"2026-07": BA.SHARES["july_loo"], "2026-01": BA.SHARES["january"]}


def cells_for(df: pd.DataFrame, owned: np.ndarray, guard_held: np.ndarray,
              noac: np.ndarray, y: np.ndarray | None, edges: np.ndarray | None = None,
              pred: np.ndarray | None = None, pedges: np.ndarray | None = None) -> dict:
    """The four cuts, as boolean masks keyed by `cut/name`."""
    out = {}
    # 1. clock regime
    out["regime/owned_written"] = owned & ~guard_held
    out["regime/guard_held"] = guard_held
    out["regime/noac"] = noac
    out["regime/complement_no_clock"] = ~owned & ~guard_held & ~noac
    # 2. airport
    ap = df["airport"].astype(str).to_numpy()
    for a in pd.unique(ap):
        out[f"airport/{a}"] = ap == a
    # 3. month of fold
    mo = df["month"].astype(str).to_numpy()
    for m in pd.unique(mo):
        out[f"month/{m}"] = mo == m
    # 4. taxi band -- deciles of the label, plus the >3,600 s tail as its own band.
    # **The edges are COMMON to every fold**, computed once on the pooled board-share
    # folds and passed in: fold-specific quantiles would give each fold its own cell
    # names, and the board aggregation below would then split one band across two keys
    # instead of summing it.  The first run of this file did exactly that.
    if y is not None and edges is not None:
        tail = y > TAIL_S
        body = ~tail
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            last = i == len(edges) - 2
            sel = body & (y >= lo) & ((y <= hi) if last else (y < hi))
            out[f"taxi_decile/{i + 1:02d} [{lo:.0f},{hi:.0f}]"] = sel
        out[f"taxi_decile/tail >{TAIL_S:.0f}"] = tail
    # 4b. THE SAME BANDS ON THE PREDICTION, and this is not decoration.  Binning by the
    # LABEL and reading the mean signed error is a trap: conditioning on high `y` selects
    # rows whose error is positive for ANY imperfect predictor, so "the top decile of y
    # under-predicts by +199 s" is guaranteed by regression dilution and says nothing about
    # calibration.  The cut that can support an arm bins by the PREDICTION, where a non-zero
    # mean signed error IS a calibration defect the model could fix.  Both are reported; the
    # label-binned one is kept for the error-share accounting only.
    if pred is not None and pedges is not None:
        ptail = pred > TAIL_S
        pbody = ~ptail
        for i in range(len(pedges) - 1):
            lo, hi = pedges[i], pedges[i + 1]
            last = i == len(pedges) - 2
            sel = pbody & (pred >= lo) & ((pred <= hi) if last else (pred < hi))
            out[f"pred_decile/{i + 1:02d} [{lo:.0f},{hi:.0f}]"] = sel
        out[f"pred_decile/tail >{TAIL_S:.0f}"] = ptail
        writable = owned | guard_held
        out["xp/writable & pred_top_decile"] = writable & (pred > pedges[-2]) & pbody
        out["xp/owned_written & pred_top_decile"] = (owned & ~guard_held
                                                     & (pred > pedges[-2]) & pbody)
        out[f"xp/writable & pred>{TAIL_S:.0f}"] = writable & ptail

    # 5. CROSS CUTS, restricted to what this lane can actually write.  A single-cut census
    # cannot name an arm: "the tail is 49.7 % of the board" is useless if the tail sits in
    # the complement.  These are the only cells a next arm here could address.
    writable = owned | guard_held
    if y is not None and edges is not None:
        for nm, sel in (("owned_written", owned & ~guard_held), ("guard_held", guard_held),
                        ("writable", writable)):
            out[f"x/{nm} & tail>{TAIL_S:.0f}"] = sel & (y > TAIL_S)
            out[f"x/{nm} & top_decile"] = sel & (y > edges[-2]) & (y <= TAIL_S)
    # The airport cross-cut runs on BOTH sides.  On the ranking side the single-cut
    # `airport/*` rms is diluted by the complement rows in that airport, which carry
    # exactly zero disagreement -- the same shape of artefact as the label binning above,
    # so the writable-only figure is the one an airport comparison may use.
    ap = df["airport"].astype(str).to_numpy()
    for a in pd.unique(ap):
        out[f"x/writable & {a}"] = writable & (ap == a)
    return out


def main() -> None:
    t0 = time.time()
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})
    print("mode-3 asserted on the shipped builds", flush=True)

    mem = {s: {f: v.astype("float64") for f, v in
               np.load(EXP / f"work/d-bundle-placebo{s}-255-free.npz").items()}
           for s in TEN}
    bag10 = BAG.build_bag(mem, pan, TEN)[0]
    on_disk = {f: v.astype("float64") for f, v in
               np.load(EXP / "work/d-bag10-255-free.npz").items()}
    if not all(np.array_equal(bag10[f], on_disk[f]) for f in A.FOLDS):
        raise SystemExit("the rebuilt K=10 bag is not the vector v8 ships")
    print("the K=10 bag rebuilds bit for bit -- censusing the vector v8 ships", flush=True)

    # common band edges, from the pooled body of the two board-share folds
    pool = np.concatenate([pan[f].y_true.to_numpy("float64") for f in BA.SHARES])
    pool = pool[pool <= TAIL_S]
    EDGES = np.quantile(pool, np.linspace(0, 1, 11))
    EDGES[0], EDGES[-1] = min(EDGES[0], 0.0), TAIL_S
    ppool = np.concatenate([(pan[f].pbase.to_numpy("float64") + bag10[f])
                            for f in BA.SHARES])
    ppool = ppool[ppool <= TAIL_S]
    PEDGES = np.quantile(ppool, np.linspace(0, 1, 11))
    PEDGES[0], PEDGES[-1] = min(PEDGES[0], 0.0), TAIL_S

    rec = {"what": __doc__.strip().splitlines()[0], "tail_s": TAIL_S,
           "shares": BA.SHARES, "taxi_band_edges_s": [float(x) for x in EDGES],
           "taxi_band_edges_from": ("the pooled body (y <= 3600) of july_loo and january, "
                                    "so every fold uses the SAME cells and the board "
                                    "aggregation sums a band instead of splitting it"),
           "folds": {}, "board": {}}
    per_fold_cells, mse_f, n_f = {}, {}, {}
    for f in A.FOLDS:
        p = pan[f]
        y = p.y_true.to_numpy("float64")
        pb = p.pbase.to_numpy("float64")
        pred = pb + bag10[f]
        e = y - pred
        se = e * e
        n = len(p)
        usable = p.usable.to_numpy(bool)
        noac = p.noac.to_numpy(bool)
        cut = G.cut(pan, [g for g in A.FOLDS if g != f])
        gmask = G.mask(pb, p.obs_taxi, cut) & usable
        cells = cells_for(p, usable, gmask, noac, y, EDGES, pred, PEDGES)
        per_fold_cells[f] = cells
        mse_f[f], n_f[f] = float(se.mean()), n
        rows = {}
        tot = float(se.sum())
        for k, m in cells.items():
            c = int(m.sum())
            if not c:
                continue
            rows[k] = {"rows": c, "sse": float(se[m].sum()),
                       "share_of_fold_sse": float(se[m].sum() / tot),
                       "rms_s": float(np.sqrt(se[m].mean())),
                       "mean_signed_error_s": float(e[m].mean())}
        rec["folds"][f] = {"rows": n, "rmse_s": float(np.sqrt(mse_f[f])),
                           "guard_cut_s": float(cut), "cells": rows}
        print(f"  {f:10s} rows {n:7d}  rmse {np.sqrt(mse_f[f]):8.2f} s  "
              f"cells {len(rows)}", flush=True)

    # --- board-weighted: a cell's contribution is sum_f share_f * SSE_f / n_f -----------
    board_mse = sum(BA.SHARES[f] * mse_f[f] for f in BA.SHARES)
    keys = sorted({k for f in BA.SHARES for k in per_fold_cells[f]})
    board_cells = {}
    for k in keys:
        contrib, rws = 0.0, 0
        for f in BA.SHARES:
            m = per_fold_cells[f].get(k)
            if m is None:
                continue
            p = pan[f]
            e = p.y_true.to_numpy("float64") - p.pbase.to_numpy("float64") - bag10[f]
            contrib += BA.SHARES[f] * float((e[m] ** 2).sum()) / n_f[f]
            rws += int(m.sum())
        board_cells[k] = {"board_mse_contribution_s2": contrib,
                          "share_of_board_mse": contrib / board_mse,
                          "rows_in_the_two_share_folds": rws,
                          "board_rmse_s_if_this_cell_were_perfect":
                              float(np.sqrt(max(board_mse - contrib, 0.0)))}
    rec["board"] = {
        "board_mse_s2": board_mse, "board_rmse_s": float(np.sqrt(board_mse)),
        "december_share": 0.0,
        "cells": board_cells,
        "note": ("a cell's contribution is sum_f share_f * SSE_cell,f / n_f, matching "
                 "base.board_interval; cuts OVERLAP by construction (an airport cell and a "
                 "regime cell contain the same rows), so shares are comparable WITHIN a cut "
                 "and do not sum to 1 across cuts"),
    }

    # --- the question that routes the next block ----------------------------------------
    own = board_cells.get("regime/owned_written", {}).get("share_of_board_mse", 0.0)
    held = board_cells.get("regime/guard_held", {}).get("share_of_board_mse", 0.0)
    comp = board_cells.get("regime/complement_no_clock", {}).get("share_of_board_mse", 0.0)
    nac = board_cells.get("regime/noac", {}).get("share_of_board_mse", 0.0)
    rec["who_owns_the_remaining_error"] = {
        "this_lane_owned_and_written": own, "this_lane_guard_held": held,
        "queue_complement_no_clock": comp, "noac": nac,
        "the_complement_is_the_largest": bool(comp > max(own, held, nac)),
        "routes_the_next_block_to": ("exp-047-queue -- the complement carries the largest "
                                     "share of board MSE and this lane cannot write those "
                                     "rows" if comp > max(own, held, nac)
                                     else "exp-046 -- the class this lane writes carries "
                                          "the largest share"),
    }

    # --- the ranking side: no label, so member disagreement is the only signal -----------
    import serve as S
    S._UNGATED = True
    dep, _ = S.ranking_block()
    vecs = np.vstack([np.load(EXP / f"work/d-rank-placebo{s}-255.npz")["d"].astype("float64")
                      for s in TEN])
    var = vecs.var(0)
    usable_r = dep.usable.to_numpy(bool)
    noac_r = dep.noac.to_numpy(bool)
    moved = (vecs.mean(0) != 0)
    cells_r = cells_for(dep, usable_r, usable_r & ~moved, noac_r, None)
    tot_var = float(var.sum())
    rrows = {}
    for k, m in cells_r.items():
        c = int(m.sum())
        if not c:
            continue
        rrows[k] = {"rows": c, "sum_member_variance_s2": float(var[m].sum()),
                    "share_of_total_disagreement": float(var[m].sum() / tot_var),
                    "member_disagreement_rms_s": float(np.sqrt(var[m].mean()))}
    by_month = {}
    for rm, sh in RANK_MONTH_SHARE.items():
        m = (dep.rank_month.astype(str).to_numpy() == rm)
        if m.sum():
            by_month[rm] = {"rows": int(m.sum()), "share": sh,
                            "member_disagreement_rms_s": float(np.sqrt(var[m].mean()))}
    rec["ranking_side"] = {
        "what_this_is": ("member disagreement, NOT error -- there is no label on the "
                         "ranking side. It measures where the bag is UNSURE, which is not "
                         "the same as where it is wrong, and no gain may be claimed from it"),
        "rows": int(len(dep)), "writable": int(usable_r.sum()),
        "total_disagreement_s2": tot_var,
        "by_rank_month": by_month, "cells": rrows,
    }

    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    top = sorted(board_cells.items(), key=lambda kv: -kv[1]["share_of_board_mse"])[:12]
    print("\n  TOP CELLS BY BOARD-WEIGHTED SQUARED ERROR")
    for k, v in top:
        print(f"    {k:34s} {v['share_of_board_mse']*100:6.2f}%  rows "
              f"{v['rows_in_the_two_share_folds']:7d}  board rmse if perfect "
              f"{v['board_rmse_s_if_this_cell_were_perfect']:8.3f} s")
    print("\n  who owns the remaining error:",
          json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in rec["who_owns_the_remaining_error"].items()
                      if not isinstance(v, str)}))
    print("  routes to:", rec["who_owns_the_remaining_error"]["routes_the_next_block_to"])
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
