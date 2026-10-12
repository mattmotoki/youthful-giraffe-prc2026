"""`c_lgb_covmatch` -- the composition-matched head, pre-registered in RESULTS section 47.9.

The folds this lane measures on are not the population it serves: clock coverage moves by
tens of points between months, so the writable population is 25.06 % total-variation distant
between the folds and the ranking side (section 47.6, `analysis/coverage.json`).  This arm is
v8's head, unchanged in features and in every hyperparameter, refitted with one per-row sample
weight `w(airport) = p_rank / p_fold` taken from that table.  Nothing else moves.

**The weights carry no label and no fold measurement** -- they are counts of rows.  The desk
ruled (2026-09-09) that rule (a) therefore applies: the gate is `every_fold_interval_clear` on
the IN-FOLD increment over v8's head, with the cross-fitted figure reported beside it.

**The weighted path must PROVE it ran weighted.**  An optional argument that preserves the
default is exactly the shape that produced section 45.3's five silent defaults, and here the
silent branch would price as +0.00 and read as "the arm does nothing" -- a plausible artifact,
which is what makes that class of defect invisible.  So: the map is checked against the
registered extremes and Kish fraction before a single fit; every member records the sha256 of
the weight vector LightGBM actually consumed, per fold, with its extremes and its weighted-fit
count; a cached member without that sidecar is refused; and a constant weight vector is fatal
inside `chainfold.fit_arm` rather than quiet here.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/covmatch.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

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

TEN = list(range(1, 11))
LEAVES = 255
HEAD10 = EXP / "work/d-bag10-255-free.npz"          # v8's head, the thing this improves on
COV = EXP / "analysis/coverage.json"
CTL = EXP / "analysis/control-weights.json"
OUT_VEC = EXP / "work/d-covmatch10-255-free.npz"
OUT = EXP / "analysis/covmatch.json"
OUT_VEC_CTL = EXP / "work/d-kishctl10-255-free.npz"
OUT_CTL = EXP / "analysis/kish-control.json"

#: RESULTS section 47.9, written and committed (c86ae65) BEFORE this file existed, and ruled
#: by the desk before it was run.  Nothing below is chosen after a number.
PRE = {
    "registered_in": "RESULTS.md section 47.9, commit c86ae65, before this file was written",
    "ruled_by_the_desk": ("rule (a): the weights are row counts from ranking-side metadata, "
                          "no fold gain enters the form and no alternative was picked on a "
                          "fold reading, so there is no section 33.3 selection premium to "
                          "price. Gate = every_fold_interval_clear on the IN-FOLD increment "
                          "over v8's head; the cross-fitted figure reported beside it"),
    "reading_the_gate_applies_to": "in_fold",
    "point_estimate_board_s": 0.05,
    "band_s": [-0.05, 0.20],
    "expected_interval_half_width_s": 0.03,
    "every_fold_clear_expected": False,
    "fold_expected_to_fail": "january",
    "expected_below_the_upload_floor_of_0_1_s": True,
    "why_small": ("the per-airport amplitude in arm_lgb is already free and a composition "
                  "weight is constant within an airport, so it cannot move it: only the "
                  "shared tree changes. And section 44 measured 511 leaves over 255 at "
                  "+0.01738 [-0.01443, +0.04283] cross-fitted, so capacity is near saturated "
                  "and losing 44 % of the effective sample should not buy anything"),
    "the_risk": ("the composition being matched is not stationary -- coverage at LIRF moves "
                 "87.0 % -> 0.0 % across two fold months and 0.1 % -> 51.5 % across the two "
                 "ranking months, so an arm that wins here may be winning on a composition "
                 "that has already changed"),
    "kish_effective_fraction_of_the_map": 0.563,
    "weight_extremes": {"min": 0.35, "max": 5.23},
}


#: RESULTS section 47.13, committed at 112fc7e BEFORE this ran.  The control is the SAME nine
#: weights permuted at seed 47 and rescaled by one exponent to the IDENTICAL Kish fraction, so
#: the only thing that differs from `c_lgb_covmatch` is whether the weights mean anything.
PRE_CONTROL = {
    "registered_in": "RESULTS.md section 47.13, commit 112fc7e, before this file ran it",
    "reading_the_gate_applies_to": "in_fold",
    "point_estimate_board_s": -0.20,
    "band_s": [-0.35, -0.05],
    "reasoning": ("if section 47.11's loss is the SAMPLE, the control reads near -0.25; if it "
                  "is the COMPOSITION, near 0. I expect mostly sample, because that mechanism "
                  "is a 44 % cut in effective rows and nothing in the argument used the "
                  "airports' identity"),
    "what_it_cannot_settle": ("Kish is matched but PLACEMENT is not -- the control's heaviest "
                              "weight sits on LFPG's few hundred rows where covmatch's sat on "
                              "LEMD's 4,182 -- so a control that loses much less has two "
                              "explanations and one permutation cannot choose between them"),
    "kish_effective_fraction_of_the_map": 0.562580,
    "weight_extremes": {"min": 0.1553, "max": 4.2609},
    "map_sha256_prefix": "396541472f4fdee8",
    "alpha": 1.222501, "permutation_seed": 47,
}


def control_map() -> tuple[dict, dict]:
    """The composition-neutral control, read from the file section 47.13 committed."""
    rec = json.loads(CTL.read_text())
    wmap = {a: float(w) for a, w in rec["weights"].items()}
    lo, hi = min(wmap.values()), max(wmap.values())
    blob = json.dumps({a: round(w, 12) for a, w in sorted(wmap.items())},
                      sort_keys=True).encode()
    got = hashlib.sha256(blob).hexdigest()
    for name, g, want in (("min", lo, PRE_CONTROL["weight_extremes"]["min"]),
                          ("max", hi, PRE_CONTROL["weight_extremes"]["max"]),
                          ("Kish", rec["kish"],
                           PRE_CONTROL["kish_effective_fraction_of_the_map"])):
        if abs(g - want) > 0.001:
            raise SystemExit(f"the control map does not match section 47.13: {name} is "
                             f"{g:.6f} against the registered {want:.6f}")
    if not got.startswith(PRE_CONTROL["map_sha256_prefix"]):
        raise SystemExit(f"the control map sha is {got[:16]} and section 47.13 registered "
                         f"{PRE_CONTROL['map_sha256_prefix']} -- this is not the map that was "
                         f"registered, and its band does not apply to it")
    return wmap, {"map": {a: float(w) for a, w in sorted(wmap.items())},
                  "map_sha256": got, "min": float(lo), "max": float(hi),
                  "kish_effective_fraction_of_the_map": float(rec["kish"]),
                  "airports_not_named_get_weight": 1.0,
                  "alpha": rec["alpha"], "permutation_seed": rec["seed"],
                  "source": str(CTL.relative_to(ROOT))}


def weight_map() -> tuple[dict, dict]:
    """The composition weights, read from `analysis/coverage.json` and CHECKED.

    Nothing here is typed in from the section 47.6 table: the table and this map come from
    the same file, and the registered extremes are used as an assertion on it, never as the
    source of the numbers.
    """
    cov = json.loads(COV.read_text())
    comp = cov["writable_composition"]
    wmap = {a: v["weight_rank_over_fold"] for a, v in comp.items()
            if v["weight_rank_over_fold"] is not None}
    if not wmap:
        raise SystemExit(f"{COV} carries no usable weights")
    lo, hi = min(wmap.values()), max(wmap.values())
    kish = cov["matching_cost"]["kish_effective_sample_fraction"]
    for name, got, want in (("min weight", lo, PRE["weight_extremes"]["min"]),
                            ("max weight", hi, PRE["weight_extremes"]["max"]),
                            ("Kish fraction", kish, PRE["kish_effective_fraction_of_the_map"])):
        if abs(got - want) > 0.01:
            raise SystemExit(
                f"the composition map does not match what section 47.9 registered: {name} "
                f"is {got:.4f} against the registered {want:.4f}. coverage.json has moved "
                f"since the registration was committed -- re-run coverage.py, decide whether "
                f"the registration still stands, and do NOT price this arm against a band "
                f"written for a different map.")
    blob = json.dumps({a: round(w, 12) for a, w in sorted(wmap.items())},
                      sort_keys=True).encode()
    return wmap, {"map": {a: float(w) for a, w in sorted(wmap.items())},
                  "map_sha256": hashlib.sha256(blob).hexdigest(),
                  "min": float(lo), "max": float(hi),
                  "kish_effective_fraction_of_the_map": float(kish),
                  "airports_not_named_get_weight": 1.0,
                  "source": str(COV.relative_to(ROOT))}


def check_prov(seed: int, prov: dict, wprov: dict) -> None:
    """Every fold of every member must show a weight vector that actually reached LightGBM."""
    missing = [f for f in A.FOLDS if f not in prov]
    if missing:
        raise SystemExit(f"member {seed}: no weight provenance for {missing} -- the fit ran "
                         f"UNWEIGHTED on those folds and its number would be a +0.00 that "
                         f"reads as 'the arm does nothing'")
    for f in A.FOLDS:
        p = prov[f]
        if p["distinct_weights"] < 2:
            raise SystemExit(f"member {seed} {f}: {p['distinct_weights']} distinct weights")
        if p["n_weighted_lgb_fits"] < 2:
            raise SystemExit(f"member {seed} {f}: only {p['n_weighted_lgb_fits']} weighted "
                             f"LightGBM fits -- the OOF pass or the full fit ran unweighted")
        # normalisation-invariant, so it checks the MAP reached the rows rather than the
        # scale: both extreme airports are in every fold's fitting set, so the spread of the
        # vector LightGBM saw must be the spread of the map.
        spread, want = p["max"] / p["min"], wprov["max"] / wprov["min"]
        if abs(spread - want) > 0.05 * want:
            raise SystemExit(f"member {seed} {f}: the weight vector spans {spread:.2f} and "
                             f"the map spans {want:.2f} -- an airport at an extreme of the "
                             f"map is missing from the fitting rows, so this member is "
                             f"weighted by a DIFFERENT map from the one registered")
        if p["kish_effective_fraction"] > 0.99:
            raise SystemExit(f"member {seed} {f}: Kish fraction {p['kish_effective_fraction']:.4f} "
                             f"-- the weights are effectively flat")


def main() -> None:
    t0 = time.time()
    ps = argparse.ArgumentParser()
    ps.add_argument("--control", action="store_true",
                    help="run section 47.13's composition-neutral control instead")
    args = ps.parse_args()
    pre = PRE_CONTROL if args.control else PRE
    out_json = OUT_CTL if args.control else OUT
    out_vec = OUT_VEC_CTL if args.control else OUT_VEC
    tag = "kishctl" if args.control else "covmatch"
    wmap, wprov = control_map() if args.control else weight_map()
    print(f"weights: {len(wmap)} airports, min {wprov['min']:.3f} max {wprov['max']:.3f}, "
          f"Kish {wprov['kish_effective_fraction_of_the_map']*100:.1f} %, "
          f"sha256 {wprov['map_sha256'][:16]}", flush=True)

    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    print("mode-3 asserted on the shipped builds", flush=True)
    pan_ship = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})

    # the head this arm is an increment over, rebuilt from ITS members and checked
    head_members = {s: {f: v.astype("float64") for f, v in
                        np.load(EXP / f"work/d-bundle-placebo{s}-255-free.npz").items()}
                    for s in TEN}
    bag10, spread_head = BAG.build_bag(head_members, pan_ship, TEN)
    on_disk = {f: v.astype("float64") for f, v in np.load(HEAD10).items()}
    if not all(np.array_equal(bag10[f], on_disk[f]) for f in A.FOLDS):
        raise SystemExit("the rebuilt K=10 head is not the vector v8 ships")
    print("v8's head rebuilds bit for bit from its ten members", flush=True)

    members, census = {}, {}
    for seed in TEN:
        t1 = time.time()
        cache = EXP / f"work/d-bundle-{tag}{seed}-255-free.npz"
        side = cache.with_suffix(".prov.json")
        if cache.exists() and side.exists():
            rec = json.loads(side.read_text())
            if rec.get("map_sha256") != wprov["map_sha256"]:
                raise SystemExit(f"member {seed}: the cached fit used weight map "
                                 f"{rec.get('map_sha256', '')[:16]} and this run's map is "
                                 f"{wprov['map_sha256'][:16]} -- delete the cache and refit")
            check_prov(seed, rec["folds"], wprov)
            members[seed] = {f: v.astype("float64") for f, v in np.load(cache).items()}
            census[seed] = {"refitted_this_run": False, "weight_provenance": rec["folds"]}
            print(f"member {seed}: from cache (weighted, checked)", flush=True)
            continue
        if cache.exists() and not side.exists():
            raise SystemExit(f"member {seed}: {cache.name} exists with no weight-provenance "
                             f"sidecar, so it cannot be shown to have run weighted. Delete "
                             f"it and refit; do NOT average it into the bag.")
        # the SAME ten perturbations as v8's head, drawn in the same order
        rng = np.random.default_rng(seed)
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
        setting_was, hold_was = CF.SETTING, CF.HOLD
        CF.SETTING, CF.HOLD = {"num_leaves": LEAVES}, ()
        prov: dict = {}
        try:
            d = CF.fit_arm(pan, wmap=wmap, prov=prov)
        finally:
            CF.SETTING, CF.HOLD = setting_was, hold_was
        check_prov(seed, prov, wprov)
        np.savez_compressed(cache, **d)
        side.write_text(json.dumps({"map_sha256": wprov["map_sha256"], "seed": seed,
                                    "folds": prov}, indent=2, sort_keys=True))
        members[seed] = d
        census[seed] = {"refitted_this_run": True, "weight_provenance": prov,
                        "seconds": round(time.time() - t1, 1)}
        print(f"member {seed}: fitted weighted ({time.time()-t1:.0f}s)  "
              f"kish {prov['july_loo']['kish_effective_fraction']:.3f}", flush=True)

    bag_cm, spread_cm = BAG.build_bag(members, pan_ship, TEN)
    np.savez_compressed(out_vec, **bag_cm)

    rec = {"what": __doc__.strip().splitlines()[0], "seeds": TEN, "leaves": LEAVES,
           "arm": f"c_lgb_{tag}@leaves_255x300_lr050+guard995",
           "pre_registration": pre, "weights": wprov, "members": census,
           "which": tag,
           "member_spread_covmatch": spread_cm, "member_spread_head": spread_head,
           "head_rebuilds_bit_for_bit": True,
           "vector": str(out_vec.relative_to(ROOT)), "prices": {}}

    inc = R.price(pan_ship, bag_cm, bag10, f"c_lgb_{tag} as an increment over v8's head")
    rec["prices"]["in_fold_over_v8_head"] = inc
    rec["prices"]["cross_fitted_beside_it"] = R.crossfit(
        pan_ship, bag_cm, bag10, "the same increment, cross-fitted, reported beside it")

    g = inc["board_all_row"]["gain_s"]
    ci = inc["board_all_row"]["interval"]["ci95"]
    clear = {f: bool(inc["by_fold"][f]["ci95"][0] > 0) for f in A.FOLDS}
    rec["verdict"] = {
        "reading_the_gate_applies_to": "in_fold",
        "increment_board_s": g,
        "increment_ci95": ci,
        "interval_half_width_s": float((ci[1] - ci[0]) / 2),
        "every_fold_interval_clear": bool(inc["every_fold_interval_clear"]),
        "folds_clear": clear,
        "inside_the_registered_band": bool(pre["band_s"][0] <= g <= pre["band_s"][1]),
        "above_the_upload_floor_0_1_s": bool(g > 0.1),
        "the_arm_passes_its_gate": bool(inc["every_fold_interval_clear"] and g > 0),
        "the_registration_was_right": None,     # filled below, both halves named
    }
    v = rec["verdict"]
    v["the_registration_was_right"] = bool(
        v["inside_the_registered_band"]
        and (v["every_fold_interval_clear"] == pre["every_fold_clear_expected"]
             if "every_fold_clear_expected" in pre else True))
    v["fold_that_failed"] = [f for f, c in clear.items() if not c]
    v["fold_expected_to_fail_was_right"] = (
        bool(pre["fold_expected_to_fail"] in v["fold_that_failed"])
        if v["fold_that_failed"] and "fold_expected_to_fail" in pre else None)
    rec["seconds"] = round(time.time() - t0, 1)
    out_json.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    print()
    print(R.row(inc))
    print(R.fold_rows(inc))
    cf = rec["prices"]["cross_fitted_beside_it"]
    cf["note_on_the_labels"] = ("crossfit() names a two-point ladder, so its 'leaves_511' "
                                "reads c_lgb_covmatch here and 'leaves_255 (the head)' "
                                "reads v8's head. The rule applied is unchanged.")
    x = cf["cross_fitted"]
    print(f"  cross-fitted beside it: {x['all_row_gain_s']:+.5f} "
          f"[{x['all_row_ci95'][0]:+.5f}, {x['all_row_ci95'][1]:+.5f}]  every-fold-clear "
          f"{x['every_fold_interval_clear']}  premium "
          f"{cf['selection_premium_s']:+.5f}")
    print()
    print(f"  registered: {pre['point_estimate_board_s']:+.3f} in "
          f"[{pre['band_s'][0]:+.2f}, {pre['band_s'][1]:+.2f}]")
    print(f"  read:       {g:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}], every-fold-clear "
          f"{v['every_fold_interval_clear']}, failed {v['fold_that_failed']}")
    print(f"  inside the band {v['inside_the_registered_band']} | "
          f"arm passes its gate {v['the_arm_passes_its_gate']} | "
          f"above the 0.1 s floor {v['above_the_upload_floor_0_1_s']}")
    print("written to", out_json.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
