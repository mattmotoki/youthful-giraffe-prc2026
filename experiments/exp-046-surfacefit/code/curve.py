"""The learning curve inside the current panel -- RESULTS section 47.15, registered at f714c16.

Section 47.14 left the sample lever resting on ONE point, and that point was a *reweighting*:
the Kish control lost 0.10998 board seconds at an effective-sample fraction of 0.5626, and
`k = 0.1912` board s per nat was read off it.  Every band in section 47.12 hangs on that.  This
measures the curve directly, by DELETING rows instead of down-weighting them.

Member 1's perturbation, K = 1, the shipped recipe unchanged.  Fitting rows at 25 / 50 / 75 /
100 %, drawn inside each source fold month at a fixed seed so the two contributing months keep
their proportions.  Two readings: over v8's head (the lane's currency, and the currency section
47.12's screen will report in) and over the 100 % member (the direct cost of the deleted rows,
with the K = 1 bagging deficit exactly cancelled).  The slope is fitted to the second.

**K = 1 is the right instrument, not a saving.**  A single member sits below the K = 10 head by
a bagging deficit that is the SAME at every level, so it cancels in the difference between
levels.  Ten times the fits would buy an offset that cancels.

**The band is propagated.**  One day-clustered bootstrap prices all four levels on the SAME
resample and refits k inside each draw; the interval is the percentiles of the 2,000 k values.
Pairing is the point -- the levels share their scoring rows.

**The frac = 1.0 level is refitted, not loaded**, and must reproduce the cached member to the
tolerance section 47.15 registered in advance: max|diff| <= 1e-9 s, expecting section 47.11's
0 / 0 / 2.274e-13.  That is the `wmap` discipline of section 47.9 applied to a second optional
argument: a subsample branch that quietly kept every row would make all four levels read the
same and look like a flat curve.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/curve.py
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
import classplacebo as CP                                             # noqa: E402
import rung511 as R                                                   # noqa: E402
import bagarm as BAG                                                  # noqa: E402

SEED_MEMBER = 1                      # the perturbation, v8's member 1
LEAVES = 255
FRACS = [1.00, 0.75, 0.50, 0.25]     # 1.00 first: nothing is read if its check fails
TEN = list(range(1, 11))
HEAD10 = EXP / "work/d-bag10-255-free.npz"
CACHED_M1 = EXP / "work/d-bundle-placebo1-255-free.npz"
OUT = EXP / "analysis/curve.json"
REPRO_TOL_S = 1e-9                   # registered in section 47.15 BEFORE this ran
REPRO_EXPECTED = {"july_loo": 0.0, "december": 0.0, "january": 2.274e-13}   # section 47.11

#: RESULTS section 47.15, committed at f714c16 BEFORE this file was written.
PRE = {
    "registered_in": "RESULTS.md section 47.15, commit f714c16, before this file existed",
    "reading_the_slope_is_fitted_to": "increment over the 100 % member",
    "hundred_pct_over_v8_head_s": -0.20,
    "hundred_pct_over_v8_head_band_s": [-0.35, -0.08],
    "hundred_pct_reasoning": ("section 41.4 fitted B(K) = a + b/K to the member churn and "
                              "section 44 read K = 5 -> 10 at +0.02250. Under that law "
                              "K = 1 -> 10 is b(1 - 1/10) against b(1/5 - 1/10), a factor of "
                              "9, so one member should sit about 0.20 s below the bag"),
    "slope_k_board_s_per_nat": 0.25,
    "slope_k_band_s": [0.05, 0.60],
    "slope_k_reasoning": ("two arguments point opposite ways and both are recorded: ABOVE "
                          "0.1912 because a down-weighted row still enters every split "
                          "statistic and a deleted one does not; BELOW 0.1912 because the "
                          "control's weights are a permutation and distort composition just "
                          "as violently, so part of its -0.10998 is section 47.11's "
                          "-0.14163 placement effect being charged to rows"),
    "every_subsampled_level_negative": True,
    "monotone_in_rows": True,
    "control_anchor_k": 0.1912,
    "covmatch_anchor_k": 0.4375,
    "refutes_it": ("k's interval crossing zero kills the sample lever at this size; any "
                   "subsampled level positive and clear over the 100 % member means the "
                   "K = 1 churn dominates and nothing is extrapolated; the frac = 1.0 check "
                   "missing its tolerance means no level is read at all"),
    "what_it_cannot_do": ("subsampling the same two months is not the experiment of adding "
                          "nine new ones. Section 47.6 measured clock coverage moving by tens "
                          "of points between months, so the marginal new row need not be "
                          "worth the marginal deleted one -- in EITHER direction"),
}


def curve_bootstrap(ph: dict, incs: dict, keys: list) -> dict:
    """Board gains for every level on ONE set of day resamples, `base.board_interval`'s law.

    The draw `dr` is taken once per fold, before the level loop, and reused for every level.
    Without that the levels would differ by their draws as much as by their rows, and the
    slope's band would be an artefact of the bootstrap rather than a statement about it.
    """
    rng = np.random.default_rng(BA.SEED)
    num0, cnt, num1 = {}, {}, {k: {} for k in keys}
    for f in BA.SHARES:
        p = ph[f]
        y = p["y_true"].to_numpy("float64")
        pb = p["pbase"].to_numpy("float64")
        codes, uniq = pd.factorize(p["day"].to_numpy())
        k_days = len(uniq)
        s0 = np.bincount(codes, weights=(y - pb) ** 2, minlength=k_days)
        c = np.bincount(codes, minlength=k_days).astype("float64")
        dr = rng.integers(0, k_days, size=(BA.BOOT, k_days))
        num0[f], cnt[f] = s0[dr].sum(1), c[dr].sum(1)
        for key in keys:
            dd = np.asarray(incs[key][f], dtype="float64")
            s1 = np.bincount(codes, weights=(y - pb - dd) ** 2, minlength=k_days)
            num1[key][f] = s1[dr].sum(1)
    mse0 = sum(BA.SHARES[f] * num0[f] / cnt[f] for f in BA.SHARES)
    out = {}
    for key in keys:
        mse1 = sum(BA.SHARES[f] * num1[key][f] / cnt[f] for f in BA.SHARES)
        out[key] = np.sqrt(mse0) - np.sqrt(mse1)
    return out


def board_point(ph: dict, inc: dict) -> float:
    """The point board gain of one increment over the head panel it is priced against."""
    rm0, rm1 = {}, {}
    for f in A.FOLDS:
        p = ph[f]
        y = p["y_true"].to_numpy("float64")
        pb = p["pbase"].to_numpy("float64")
        dd = np.asarray(inc[f], dtype="float64")
        rm0[f] = float(np.sqrt(np.mean((y - pb) ** 2)))
        rm1[f] = float(np.sqrt(np.mean((y - pb - dd) ** 2)))
    return float(BA.board(rm0) - BA.board(rm1))


def fit_slope(x: np.ndarray, g: np.ndarray) -> float:
    """`g = k*x` through the origin: the exact zero at frac = 1 is the anchor."""
    return float((x * g).sum() / (x * x).sum())


def main() -> None:
    t0 = time.time()

    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    print("mode-3 asserted on the shipped builds", flush=True)
    pan_ship = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})

    # v8's head, rebuilt from its own ten members and checked against the shipped vector
    head_members = {s: {f: v.astype("float64") for f, v in
                        np.load(EXP / f"work/d-bundle-placebo{s}-255-free.npz").items()}
                    for s in TEN}
    bag10, _ = BAG.build_bag(head_members, pan_ship, TEN)
    on_disk = {f: v.astype("float64") for f, v in np.load(HEAD10).items()}
    if not all(np.array_equal(bag10[f], on_disk[f]) for f in A.FOLDS):
        raise SystemExit("the rebuilt K=10 head is not the vector v8 ships")
    print("v8's head rebuilds bit for bit from its ten members", flush=True)

    # member 1's perturbed panel, the same draw and the same order as v8's head used
    rng = np.random.default_rng(SEED_MEMBER)
    blocks = {}
    for m, b in blk.items():
        pb, _ = CP.placebo_block(b, R.THR, CF.GAP_CUT_S, CP.MOVED[m], rng)
        blocks[m] = pb
    real = A.month_block
    A.month_block = (lambda blocks: (
        lambda month, prefix_months, rule=None: blocks[month]))(blocks)
    try:
        pan = A.panel_with_block(BA.load())
    finally:
        A.month_block = real

    levels, census = {}, {}
    for frac in FRACS:
        t1 = time.time()
        pct = int(round(frac * 100))
        cache = EXP / f"work/d-curve{pct}-255-free.npz"
        side = cache.with_suffix(".prov.json")
        if cache.exists() and side.exists():
            rec = json.loads(side.read_text())
            if abs(rec.get("subsample_seed_used", -1) - SEED_MEMBER) > 0:
                raise SystemExit(f"{pct} %: cached at seed {rec.get('subsample_seed_used')}")
            levels[frac] = {f: v.astype("float64") for f, v in np.load(cache).items()}
            census[pct] = {**rec, "refitted_this_run": False}
            print(f"{pct:3d} %: from cache", flush=True)
        else:
            setting_was, hold_was = CF.SETTING, CF.HOLD
            CF.SETTING, CF.HOLD = {"num_leaves": LEAVES}, ()
            prov: dict = {}
            try:
                d = CF.fit_arm(pan, prov=prov, frac=frac, sub_seed=SEED_MEMBER)
            finally:
                CF.SETTING, CF.HOLD = setting_was, hold_was
            missing = [f for f in A.FOLDS if "subsample_frac" not in prov.get(f, {})]
            if missing:
                raise SystemExit(f"{pct} %: no subsample provenance for {missing} -- the fit "
                                 f"ran on the full panel while being labelled a subsample")
            np.savez_compressed(cache, **d)
            side.write_text(json.dumps({"folds": prov, "subsample_seed_used": SEED_MEMBER,
                                        "seconds": round(time.time() - t1, 1)},
                                       indent=2, sort_keys=True))
            levels[frac] = d
            census[pct] = {"folds": prov, "refitted_this_run": True,
                           "seconds": round(time.time() - t1, 1)}
            print(f"{pct:3d} %: fitted ({time.time()-t1:.0f}s)  rows "
                  + " ".join(f"{f[:4]} {prov[f]['fit_rows']}" for f in A.FOLDS), flush=True)

        # the check that makes every other level worth reading, at the registered tolerance
        if frac >= 1.0:
            cached = {f: v.astype("float64") for f, v in np.load(CACHED_M1).items()}
            diffs = {f: float(np.abs(levels[frac][f] - cached[f]).max()) for f in A.FOLDS}
            ident = {f: bool(np.array_equal(levels[frac][f], cached[f])) for f in A.FOLDS}
            census[pct]["reproduction_check"] = {
                "max_abs_diff_s": diffs, "bit_for_bit": ident,
                "tolerance_s": REPRO_TOL_S, "expected_s": REPRO_EXPECTED,
                "passes": all(v <= REPRO_TOL_S for v in diffs.values())}
            for f in A.FOLDS:
                print(f"      repro {f:9s} max|diff| {diffs[f]:.3e}  identical {ident[f]}",
                      flush=True)
            if not census[pct]["reproduction_check"]["passes"]:
                raise SystemExit(
                    f"the frac=1.0 refit does not reproduce {CACHED_M1.name} within "
                    f"{REPRO_TOL_S:g} s -- {diffs}. Section 47.15 registered this tolerance "
                    f"before the run, so no level is read. The new subsample argument moved "
                    f"the shipped path and every level below it is priced on a different fit.")

    # ---- reading 1: each level as an increment over v8's head, the lane's currency
    over_head = {}
    for frac in FRACS:
        over_head[f"{int(round(frac*100))}pct"] = R.price(
            pan_ship, levels[frac], bag10,
            f"K=1 member 1 at {frac:.0%} of the fitting rows, over v8's head")

    # ---- reading 2: each level over the 100 % member -- the direct cost of deleted rows
    d100 = levels[1.00]
    ph = R.head_panel(pan_ship, d100)
    incs = {f"{int(round(fr*100))}pct": {f: levels[fr][f] - d100[f] for f in A.FOLDS}
            for fr in FRACS}
    keys = [f"{int(round(fr*100))}pct" for fr in FRACS]
    over_full = {k: {"board_gain_s": board_point(ph, incs[k]),
                     "rows_moved": {f: int((incs[k][f] != 0).sum()) for f in A.FOLDS}}
                 for k in keys}
    if over_full["100pct"]["board_gain_s"] != 0.0:
        raise SystemExit("the 100 % level is not exactly zero against itself")

    # ---- the slope, with a band propagated through paired day draws
    draws = curve_bootstrap(ph, incs, keys)
    sub = [fr for fr in FRACS if fr < 1.0]
    x = np.log(np.array(sub))
    g_pt = np.array([over_full[f"{int(round(fr*100))}pct"]["board_gain_s"] for fr in sub])
    k_pt = fit_slope(x, g_pt)
    G = np.stack([draws[f"{int(round(fr*100))}pct"] for fr in sub])      # (3, BOOT)
    k_draws = (x[:, None] * G).sum(0) / (x * x).sum()
    k_lo, k_hi = (float(v) for v in np.percentile(k_draws, [2.5, 97.5]))

    base_over_head = over_head["100pct"]["board_all_row"]["gain_s"]
    def predict(mult: float) -> dict:
        return {"rows_multiple": mult, "ln_rows": float(np.log(mult)),
                "slope_term_s": k_pt * float(np.log(mult)),
                "screen_reading_s": base_over_head + k_pt * float(np.log(mult)),
                "screen_ci95": [base_over_head + k_lo * float(np.log(mult)),
                                base_over_head + k_hi * float(np.log(mult))],
                "gain_over_the_current_panel_s": k_pt * float(np.log(mult)),
                "gain_ci95": [k_lo * float(np.log(mult)), k_hi * float(np.log(mult))]}

    rec = {"what": __doc__.strip().splitlines()[0],
           "member": SEED_MEMBER, "leaves": LEAVES, "fracs": FRACS,
           "pre_registration": PRE, "levels": census,
           "reading_1_over_v8_head": over_head,
           "reading_2_over_the_100pct_member": over_full,
           "bootstrap": {"draws": BA.BOOT, "seed": BA.SEED,
                         "paired": "one set of day resamples prices every level"},
           "slope": {"k_board_s_per_nat": k_pt, "ci95": [k_lo, k_hi],
                     "fitted_on": [f"{int(round(fr*100))}pct" for fr in sub],
                     "form": "g = k * ln(frac), through the exact zero at frac = 1",
                     "share_of_draws_positive": float((k_draws > 0).mean()),
                     "crosses_zero": bool(k_lo <= 0.0 <= k_hi)},
           "hundred_pct_over_v8_head": {
               "board_gain_s": base_over_head,
               "ci95": over_head["100pct"]["board_all_row"]["interval"]["ci95"]},
           "predictions_for_the_extended_panel": {
               "x2": predict(2.0), "x3": predict(3.0), "x5": predict(5.0),
               "x5_5_the_extended_panel": predict(5.5)},
           "anchors_it_is_scored_against": {
               "control_k_0_1912_gives_at_5_5x": 0.326,
               "covmatch_k_0_4375_gives_at_5_5x": 0.746,
               "section_47_12_registered_band_s": [0.13, 0.58],
               "section_47_12_superseded_band_s": [0.56, 1.01]},
           }

    v = {"slope_positive_and_clear": bool(k_lo > 0),
         "every_subsampled_level_negative": all(
             over_full[f"{int(round(fr*100))}pct"]["board_gain_s"] < 0 for fr in sub),
         "monotone_in_rows": bool(
             over_full["25pct"]["board_gain_s"] < over_full["50pct"]["board_gain_s"]
             < over_full["75pct"]["board_gain_s"] < 0),
         "k_inside_the_registered_band": bool(PRE["slope_k_band_s"][0] <= k_pt
                                              <= PRE["slope_k_band_s"][1]),
         "hundred_pct_inside_its_registered_band": bool(
             PRE["hundred_pct_over_v8_head_band_s"][0] <= base_over_head
             <= PRE["hundred_pct_over_v8_head_band_s"][1]),
         "k_above_the_control_anchor": bool(k_pt > PRE["control_anchor_k"]),
         "control_anchor_inside_k_interval": bool(k_lo <= PRE["control_anchor_k"] <= k_hi),
         "covmatch_anchor_inside_k_interval": bool(k_lo <= PRE["covmatch_anchor_k"] <= k_hi)}
    rec["verdict"] = v
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    print()
    print("  level   fitting rows   over v8 head                       over the 100 % member")
    for fr in FRACS:
        k = f"{int(round(fr*100))}pct"
        oh = over_head[k]["board_all_row"]
        ci = oh["interval"]["ci95"]
        n = sum(census[int(round(fr*100))]["folds"][f]["fit_rows"] for f in A.FOLDS)
        print(f"  {fr:5.0%}  {n:11,d}   {oh['gain_s']:+.5f} "
              f"[{ci[0]:+.5f},{ci[1]:+.5f}]   {over_full[k]['board_gain_s']:+.5f}")
    print()
    print(f"  slope k = {k_pt:+.5f} [{k_lo:+.5f}, {k_hi:+.5f}] board s per nat of rows "
          f"({float((k_draws > 0).mean())*100:.1f} % of draws positive)")
    print(f"  registered {PRE['slope_k_board_s_per_nat']:+.3f} in "
          f"[{PRE['slope_k_band_s'][0]:+.2f}, {PRE['slope_k_band_s'][1]:+.2f}] -> "
          f"inside {v['k_inside_the_registered_band']}")
    print(f"  control anchor 0.1912 inside the interval "
          f"{v['control_anchor_inside_k_interval']} | covmatch anchor 0.4375 inside "
          f"{v['covmatch_anchor_inside_k_interval']}")
    print()
    print(f"  100 % member over v8's head {base_over_head:+.5f} (registered "
          f"{PRE['hundred_pct_over_v8_head_s']:+.2f}, inside "
          f"{v['hundred_pct_inside_its_registered_band']})")
    for nm, mult in (("x2", 2.0), ("x3", 3.0), ("x5", 5.0), ("x5.5", 5.5)):
        p = predict(mult)
        print(f"  {nm:4s} rows: gain over the current panel "
              f"{p['gain_over_the_current_panel_s']:+.4f} "
              f"[{p['gain_ci95'][0]:+.4f}, {p['gain_ci95'][1]:+.4f}]   "
              f"screen would read {p['screen_reading_s']:+.4f}")
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
