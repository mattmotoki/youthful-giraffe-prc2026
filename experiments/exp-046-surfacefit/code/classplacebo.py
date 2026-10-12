"""The resolution floor of the fold instrument: what an ARBITRARY class change of the same
size is worth (desk decision, 2026-09-09, after read B).

Read B priced the plane-metric correction -- a **19-row net** change in a 2,497-row added
class -- at **-0.06087 [-0.12411, -0.00618]** over v18's bytes, july_loo clear below zero.
The sign pattern says that number is not the correction:

    2025-01  net  -4  ( 4 out,  0 in)   ->  january  +25.23 s2/row
    2025-12  net +23  ( 3 out, 26 in)   ->  december +23.21
    2025-07  net  +0  ( 8 out,  8 in)   ->  july_loo **-85.43**, interval clear below zero

The fold whose class COUNT did not change carries the whole loss.  A correction that improves
placements cannot systematically harm the one fold it left the same size; what it can do is
change the OTHER two folds, which are july's fit set, and move july's vector chaotically --
boosted trees are not continuous in their training set.

Ruled out first, so this is not a guess: the panel has not moved (`artifacts/exp-047/block`
is older than the bundle fit), and a refit at the IDENTICAL setting on the IDENTICAL class is
bit-identical to the shipped vector off LFPG (`analysis/lfpg-rider.json`, 0 rows differing).
So the sensitivity is to the FIT SET and not to run-to-run noise.

**The measurement.** Perturb the SHIPPED class by exactly the counts the correction moved, but
choose the rows arbitrarily inside the same stratum, refit at 255 leaves, and price over v18's
bytes.  A placebo drawn from the same stratum is the matched comparison: rows forced OUT come
from the kept set, rows forced IN come from the mode-4 rows that pass step 0 and check (i) and
fail check (ii) -- which is exactly where the correction's joins come from.

If the placebo reads the same order as read B, the day-clustered interval is not pricing the
thing that moves, and neither read B nor read A resolves what it claims to.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/classplacebo.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]

import arms as A                                                      # noqa: E402
import chainfold as CF                                                # noqa: E402
import rung511 as R                                                   # noqa: E402

#: the counts the plane-metric correction moved, per month (`analysis/plane-kill.json`)
MOVED = {"2025-07": {"out": 8, "in": 8},
         "2025-12": {"out": 3, "in": 26},
         "2025-01": {"out": 4, "in": 0}}
#: THREE draws, not one: the desk, 2026-09-09 -- "three arbitrary 33-row perturbations of
#: the shipped class (different rows each; recorded), refit at 255 with the shipped setting,
#: each priced over v18's bytes per fold, raw and clean ... a read counts on a fold only
#: where it lies outside the placebo envelope there".  33 is july's FIT SET: december's 29
#: state changes plus january's 4.
SEEDS = [1, 2, 3]
OUT = EXP / "analysis/class-placebo.json"


def placebo_block(blk, thr: float, gap_cut: float, counts: dict, rng):
    """`chainfold.restrict`, with `counts` rows forced out of and into the kept set.

    The two pools are the correction's own strata: OUT is drawn from the rows the gate
    keeps, IN from the mode-4 rows that pass step 0 and check (i) and fail check (ii).
    """
    b = blk.copy()
    m4 = b.match_mode.to_numpy("float64") == 4.0
    step = (m4 & (b.chain_gap_s.to_numpy("float64") > gap_cut)
            & (b.chain_abs_dt_offblock.to_numpy("float64") <= thr))
    agrees = b.chain_runway_agrees.fillna(0).to_numpy("float64") == 1.0
    keep = step & agrees
    out_pool = np.flatnonzero(keep)
    in_pool = np.flatnonzero(step & ~agrees)
    n_out = min(counts["out"], len(out_pool))
    n_in = min(counts["in"], len(in_pool))
    forced_out = rng.choice(out_pool, n_out, replace=False) if n_out else np.array([], int)
    forced_in = rng.choice(in_pool, n_in, replace=False) if n_in else np.array([], int)
    keep[forced_out] = False
    keep[forced_in] = True
    ids = {"forced_out": sorted(b.MVT_ID_mvt.to_numpy()[forced_out].tolist()),
           "forced_in": sorted(b.MVT_ID_mvt.to_numpy()[forced_in].tolist())}
    drop = m4 & ~keep
    cols = [c for c in (["obs_icao", "obs_wheels_off", *CF.BU.FEATURES]) if c in b.columns]
    b.loc[drop, cols] = np.nan
    b.loc[drop, "obs_matched"] = False
    b.loc[drop, "obs_has_clock"] = False
    b.loc[drop, "reach_class"] = "unmatched"
    return b, {"kept": int(keep.sum()), "forced_out": int(n_out), "forced_in": int(n_in),
               "out_pool": int(len(out_pool)), "in_pool": int(len(in_pool)),
               "rows_moved": ids}


def main() -> None:
    t0 = time.time()
    if R.sha256(R.HEAD) != R.HEAD_SHA:
        raise SystemExit(f"{R.HEAD.name} is not the vector v18's bytes ship")
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    print("mode-3 asserted on the shipped builds", flush=True)
    H = {f: v.astype("float64") for f, v in np.load(R.HEAD).items()}

    out = {"what": __doc__.strip().splitlines()[0],
           "counts_matched_to": "analysis/plane-kill.json, per month",
           "moved": MOVED, "seeds": SEEDS,
           "read_B_for_comparison": {
               "all_row_gain_s": -0.06087, "all_row_ci95": [-0.12411, -0.00618],
               "by_fold_s2_per_row": {"july_loo": -85.43, "december": +23.21,
                                      "january": +25.23}},
           "draws": {}}
    for seed in SEEDS:
        t1 = time.time()
        rng = np.random.default_rng(seed)
        blocks, census = {}, {}
        for m, b in blk.items():
            pb, cen = placebo_block(b, R.THR, CF.GAP_CUT_S, MOVED[m], rng)
            blocks[m], census[m] = pb, cen
        real = A.month_block

        def mb(month, prefix_months, rule=None):
            return blocks[month]
        A.month_block = mb
        try:
            import base as BA
            pan = A.panel_with_block(BA.load())
        finally:
            A.month_block = real
        cache = EXP / f"work/d-bundle-placebo{seed}-255-free.npz"
        if cache.exists():
            d_free = {f: v.astype("float64") for f, v in np.load(cache).items()}
            refit = False
        else:
            setting_was, hold_was = CF.SETTING, CF.HOLD
            CF.SETTING, CF.HOLD = {"num_leaves": 255}, ()
            try:
                d_free = CF.fit_arm(pan)
            finally:
                CF.SETTING, CF.HOLD = setting_was, hold_was
            np.savez_compressed(cache, **d_free)
            refit = True
        d_held = R.hold_lfpg(pan, d_free)
        rec = R.price(pan, d_held, H, f"placebo draw {seed}: an arbitrary class change of "
                                      f"the correction's own size, 255 leaves")
        rec["census"] = census
        rec["refitted_this_run"] = refit
        rec["seconds"] = round(time.time() - t1, 1)
        out["draws"][str(seed)] = rec
        print(f"\n[placebo {seed}]  ({rec['seconds']:.0f}s, refit {refit})", flush=True)
        print(R.row(rec), flush=True)
        print(R.fold_rows(rec), flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # the draws must actually be different rows, proved rather than asserted
    overlap = {}
    for i, a in enumerate(SEEDS):
        for b_ in SEEDS[i + 1:]:
            for m in MOVED:
                ca = out["draws"][str(a)]["census"][m]["rows_moved"]
                cb = out["draws"][str(b_)]["census"][m]["rows_moved"]
                for side in ("forced_out", "forced_in"):
                    overlap[f"{m} {side} draw{a}^draw{b_}"] = len(
                        set(ca[side]) & set(cb[side]))
    out["draws_share_rows"] = overlap
    out["draws_are_distinct"] = bool(max(overlap.values()) < max(
        c["out"] + c["in"] for c in MOVED.values())) if overlap else None

    env, env_c = {}, {}
    for f in A.FOLDS:
        env[f] = [out["draws"][str(s)]["by_fold"][f]["mse_gain_s2"] for s in SEEDS]
        env_c[f] = [out["draws"][str(s)]["by_fold_clean"][f]["mse_gain_s2"] for s in SEEDS]
    board = [out["draws"][str(s)]["board_all_row"]["gain_s"] for s in SEEDS]
    out["envelope"] = {
        "what": ("what an ARBITRARY class change of the correction's own size is worth. "
                 "A read counts on a fold only where it lies outside this."),
        "by_fold_s2_per_row_raw": env, "by_fold_s2_per_row_clean": env_c,
        "board_all_row_s": board,
        "range_by_fold_raw": {f: [min(v), max(v)] for f, v in env.items()},
        "range_by_fold_clean": {f: [min(v), max(v)] for f, v in env_c.items()},
        "widest_abs_by_fold_raw": {f: max(abs(x) for x in v) for f, v in env.items()},
        "board_range_s": [min(board), max(board)],
        "widest_abs_board_s": max(abs(x) for x in board),
    }
    out["re_reading"] = {}
    for name, g_s, by in (
            ("A_the_511_rung", -0.06281,
             {"july_loo": -62.58, "december": 12.54, "january": -6.13}),
            ("B_the_plane_metric_correction", -0.06087,
             {"july_loo": -85.43, "december": 23.21, "january": 25.23})):
        lo, hi = out["envelope"]["board_range_s"]
        out["re_reading"][name] = {
            "all_row_gain_s": g_s,
            "board_outside_the_placebo_range": bool(g_s < lo or g_s > hi),
            "by_fold": {f: {
                "s2_per_row": by[f],
                "placebo_range": out["envelope"]["range_by_fold_raw"][f],
                "outside_the_envelope": bool(
                    by[f] < out["envelope"]["range_by_fold_raw"][f][0]
                    or by[f] > out["envelope"]["range_by_fold_raw"][f][1])}
                for f in A.FOLDS},
        }
        out["re_reading"][name]["folds_where_the_read_counts"] = [
            f for f in A.FOLDS
            if out["re_reading"][name]["by_fold"][f]["outside_the_envelope"]]
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\nPLACEBO ENVELOPE (three arbitrary class changes of the same size):", flush=True)
    for f in A.FOLDS:
        print(f"  {f:10s} raw {[round(x, 2) for x in env[f]]}  range "
              f"[{min(env[f]):+.2f},{max(env[f]):+.2f}]   clean "
              f"{[round(x, 2) for x in env_c[f]]}", flush=True)
    print(f"  board all-row {[round(x, 5) for x in board]}  range "
          f"[{min(board):+.5f},{max(board):+.5f}]", flush=True)
    print("  draws share rows:", max(overlap.values()) if overlap else 0,
          "| distinct:", out["draws_are_distinct"], flush=True)
    print("\nRE-READING against the envelope:", flush=True)
    for name, r in out["re_reading"].items():
        print(f"  {name}: board {r['all_row_gain_s']:+.5f} outside "
              f"{r['board_outside_the_placebo_range']}; counts on "
              f"{r['folds_where_the_read_counts'] or 'NO FOLD'}", flush=True)
    print("written to", OUT.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
