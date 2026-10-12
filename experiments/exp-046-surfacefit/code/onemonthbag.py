"""RESULTS section 47.20's K = 10 extended bag: the ten perturbations on the extended panel.

Each member is v8's head recipe fitted on the current panel PLUS 2025-11's 99,709 writable
rows.  Member 1 already exists -- it is the section 47.17 screen's vector -- and is reused
after being checked, not refitted.  2025-11 carries NO perturbation in any member: the
perturbation is a class draw on the fold blocks, and the extra month is new rows, which is
the point.

Gate, registered at section 47.17 before its data and ruled rule (a) by the desk: the
**in-fold increment over v8's head, every fold clear**.  Prediction registered at section
47.20 before this ran: **+0.49, band [+0.32, +0.62]**, NOT the naive (ii) + offset of
+0.59236 -- that offset was measured at 523,550 fitting rows and a bag over less churn buys
less.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/onemonthbag.py
"""
from __future__ import annotations

import argparse
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
import onemonth as OM                                                 # noqa: E402

TEN = list(range(1, 11))
LEAVES = 255
HEAD10 = EXP / "work/d-bag10-255-free.npz"
OUT_VEC = EXP / "work/d-onemonth-bag10-255-free.npz"
OUT = EXP / "analysis/onemonth-bag10.json"

PRE_THREE = {
    "registered_in": "RESULTS.md section 47.33, before this file ran on the three-month stack",
    "point_estimate_board_s": 0.468, "band_s": [0.36, 0.58],
    "per_fold_s2": {"july_loo": 336.0, "december": 316.0, "january": 210.0},
    "per_fold_band_s2": {"july_loo": [230, 450], "december": [200, 440], "january": [120, 300]},
    "the_naive_figure_deliberately_not_registered": 0.83932,
    "why": ("the SAME churn-scaling argument section 47.20 registered and got right (+0.49 "
            "[+0.32,+0.62] against a read of +0.51347). The bag benefit is measured twice: "
            "+0.28613 at 1.000x rows (47.16) and +0.20724 at 1.575x (v9 minus the one-month "
            "K=1, both over v8's head). Those two points give benefit ~ n^-0.7105, so at the "
            "three-month 2.672x the bag buys about +0.1424, not the +0.28613 the naive figure "
            "would add. A bag over less churn buys less"),
    "falsifier_for_the_argument": ("a measured bag benefit at or above +0.20724 -- the "
                                   "one-month value -- says more rows did NOT reduce the "
                                   "churn a bag averages away, and the scaling is wrong"),
    "every_fold_clear_expected": True,
    "fold_expected_to_fail": "january",
    "gate": "in_fold increment over v9's head, every fold clear, rule (a)",
}

PRE_ONEMONTH = {
    "registered_in": "RESULTS.md section 47.20, commit 467c89f, before this file ran",
    "point_estimate_board_s": 0.49, "band_s": [0.32, 0.62],
    "the_naive_figure_deliberately_not_registered": 0.59236,
    "why": ("the +0.28613 bag offset was measured at 523,550 fitting rows; at 1.577x the rows "
            "each member is individually more stable, so a bag over less churn buys less. If "
            "churn variance falls as 1/n the offset scales by ~0.63 to about +0.18"),
    "every_fold_clear_expected": True,
    "fold_expected_to_fail": "january",
    "gate": "in_fold increment over v8's head, every fold clear, rule (a)",
}

# SECTION 47.55.  The seasonal bag's OWN registration, from 47.49.3 (commit f435d25, written
# before 2025-08 existed).  It exists because `pre: None` used to fall through to the
# module-level PRE -- section 45.3's silent default -- and this stack's report therefore
# claimed section 47.20's band, which belongs to the ONE-MONTH bag over v8's head.  No board
# BAND was registered for the seasonal bag, only a point and per-fold points, so `band_s` is
# None here and the file makes no band claim rather than borrowing one.
PRE_SEASONAL = {
    "registered_in": ("RESULTS.md section 47.49.3, commit f435d25, before 2025-08 existed "
                      "and before any Aug read"),
    "point_estimate_board_s": 0.63115, "band_s": None,
    "per_fold_s2": {"january": 224.90, "december": 66.01, "july_loo": 498.47},
    "per_fold_band_s2": {"january": [167.93, 281.83], "december": [0.92, 148.21],
                         "july_loo": None},
    "companion_after_the_Aug_read": {
        "july_loo": 374.70,
        "why": ("section 47.51: Aug's measured alone value +628.73 falsified the +818.05 the "
                "registered point was scaled from. The companion is the same conversion at "
                "the measured value; the REGISTERED point does not move")},
    "the_naive_figure_deliberately_not_registered": None,
    "every_fold_clear_expected": True,
    "fold_expected_to_fail": None,
    "fold_expected_to_decide_it": "december, at a scaled lower bound of +0.92",
    "gate": ("in_fold increment over v10's head, every fold clear, rule (a), read under "
             "D23's discounted gate credited = min(read, prediction)"),
}


#: Each stack declares, in ONE place, the months it carries, the K=1 vector that IS its
#: member 1, the head its increment is priced over, and the prediction registered before the
#: fit.  A stack is never assembled from flags at the call site: that is how a bag ends up
#: priced against the wrong head.
PRICE_KEY_NOTE = (
    "the single key under `prices` NAMES the head, and `priced_over` says the same thing in "
    "words: the two cannot disagree because the key is derived from it (section 47.43.4). "
    "Before that repair the key was hard-coded `in_fold_over_v8_head` on every stack, "
    "including the three-month one priced over v9's head -- queue's v10 audit caught it")

# SECTION 47.58.  The five-month July head's registration, from 47.57 (commit ce7fbfd,
# before any 2025-06 read).  Its gate is JULY-ONLY by construction and that is stated here
# rather than discovered at the read: 47.57's ruling changes the July head and nothing else,
# so january's and december's increments over v11's arm are EXACTLY zero -- not "clear", not
# "small", zero -- because their members are REUSED, not refitted.  A rule (a) that demanded
# every fold clear on this arm could never pass, and a gate that cannot pass is not a gate.
PRE_SEASONAL5 = {
    "registered_in": "RESULTS.md 47.57, commit ce7fbfd, before any 2025-06 read",
    "point_estimate_board_s": 0.79256 - 0.398005522,   # the june increment alone
    "band_s": None,
    "per_fold_s2": {"july_loo": 423.42, "december": 0.0, "january": 0.0},
    "per_fold_band_s2": {"july_loo": None, "december": [0.0, 0.0], "january": [0.0, 0.0]},
    "companion_at_the_measured_d1_value": {"july_loo": 308.57,
                                           "board_increment": 0.68555 - 0.398005522},
    "the_naive_figure_deliberately_not_registered": None,
    "every_fold_clear_expected": False,
    "fold_expected_to_decide_it": "july_loo, the only fold whose head changed",
    "gate": ("in_fold increment over v11's arm, JULY-ONLY clear of zero, rule (a) restricted "
             "to the fold whose head changed; january and december must be EXACTLY zero and "
             "a non-zero there is a defect, not a reading"),
}

STACKS = {
    "onemonth": {"months": ["2025-11"],
                 "member1": "d-onemonth-screen-255-free.npz",
                 "member_prefix": "d-onemonth",
                 "head": "d-bag10-255-free.npz", "head_name": "v8's head",
                 "price_key": "in_fold_over_v8_head",
                 "out_vec": "d-onemonth-bag10-255-free.npz",
                 "out": "onemonth-bag10.json", "pre": PRE_ONEMONTH},
    "threemonth": {"months": ["2025-11", "2025-10", "2025-03"],
                   "member1": "d-onemonth-twomonth-plus03-255-free.npz",
                   "member_prefix": "d-threemonth",
                   "head": "d-onemonth-bag10-255-free.npz", "head_name": "v9's head",
                   "price_key": "in_fold_over_v9_head",
                   "out_vec": "d-threemonth-bag10-255-free.npz",
                   "out": "threemonth-bag10.json", "pre": PRE_THREE},
    # SECTION 47.49, the SEASONAL HEAD (recorded rule, 47.46).  `months` is a PER-FOLD map, not a list:
    # the served months are antipodal, so the fit that scores a january-like fold carries the
    # months near 2026-01 and the fit that scores july carries the months near 2026-07.  There
    # is no cached member 1 -- the Feb screen's member is a DIFFERENT set on july -- so
    # `member1` is None and all ten members are fitted.
    "seasonal": {"months": {"january": ["2025-11", "2025-10", "2025-03", "2025-02"],
                            "december": ["2025-11", "2025-10", "2025-03", "2025-02"],
                            "july_loo": ["2025-11", "2025-10", "2025-03", "2025-08"]},
                 "member1": None,
                 "member_prefix": "d-seasonal",
                 "head": "d-threemonth-bag10-255-free.npz", "head_name": "v10's head",
                 "price_key": "in_fold_over_v10_head",
                 "out_vec": "d-seasonal-bag10-255-free.npz",
                 "out": "seasonal-bag10.json", "pre": PRE_SEASONAL},
    # SECTION 47.49 (c): the seasonal stack's MECHANICS, exercisable before 2025-08 exists.
    # July's set is the THREE-month one -- a placeholder where 2025-08's will go -- so the
    # per-fold path is exercised with genuinely different sets and different row counts
    # without inventing a month.  `plan_only` refuses everything but --plan.
    "seasonal5": {"months": {"january": ["2025-11", "2025-10", "2025-03", "2025-02"],
                            "december": ["2025-11", "2025-10", "2025-03", "2025-02"],
                            "july_loo": ["2025-11", "2025-10", "2025-03", "2025-08",
                                         "2025-06"]},
                  "member1": None, "member_prefix": "d-seasonal5",
                  # only july's head changed, so only july is fitted; the other two folds
                  # are taken from the SAME member of the stack this one is priced over
                  "fit_folds": ["july_loo"],
                  "reuse_members": "d-seasonal-m{seed}-255-free.npz",
                  "reuse_stack": "seasonal",
                  "head": "d-seasonal-bag10-255-free.npz",
                  "head_name": "v11's arm",
                  "price_key": "in_fold_over_v11_arm",
                  "out_vec": "d-seasonal5-bag10-255-free.npz",
                  "out": "seasonal5-bag10.json", "pre": PRE_SEASONAL5},
    # 47.58's mechanics rehearsal: seasonal5 with July's FIFTH month absent, so the run
    # reads no 2025-06 row. plan_only refuses everything but --plan.
    "seasonal5plan": {"months": {"january": ["2025-11", "2025-10", "2025-03", "2025-02"],
                                 "december": ["2025-11", "2025-10", "2025-03", "2025-02"],
                                 "july_loo": ["2025-11", "2025-10", "2025-03", "2025-08"]},
                      "member1": None, "member_prefix": "d-seasonal5",
                      "fit_folds": ["july_loo"],
                      "reuse_members": "d-seasonal-m{seed}-255-free.npz",
                      "reuse_stack": "seasonal",
                      "head": "d-seasonal-bag10-255-free.npz", "head_name": "v11's arm",
                      "price_key": "in_fold_over_v11_arm",
                      "out_vec": "d-seasonal5plan-bag10-255-free.npz",
                      "out": "seasonal5plan-bag10.json", "pre": None, "plan_only": True},
    "seasonalplan": {"months": {"january": ["2025-11", "2025-10", "2025-03", "2025-02"],
                                "december": ["2025-11", "2025-10", "2025-03", "2025-02"],
                                "july_loo": ["2025-11", "2025-10", "2025-03"]},
                     "plan_only": True,
                     "member1": None, "member_prefix": "d-seasonalplan",
                     "head": "d-threemonth-bag10-255-free.npz", "head_name": "v10's head",
                     "price_key": "in_fold_over_v10_head",
                     "out_vec": "d-seasonalplan-bag10-255-free.npz",
                     "out": "seasonalplan-bag10.json", "pre": None},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="onemonth", choices=sorted(STACKS),
                    help="which extension stack to bag; 'onemonth' reproduces v9 exactly")
    ap.add_argument("--plan", action="store_true",
                    help="section 47.49 (c): build the extension frame(s), print the row "
                         "counts and the member cache names, and STOP before any fit")
    args = ap.parse_args()
    st = STACKS[args.stack]
    if st.get("plan_only") and not args.plan:
        raise SystemExit(
            f"stack {args.stack!r} is PLAN ONLY (section 47.49 (c)): it stands a placeholder "
            f"month set where a real one will go, so a bag built from it would be fitted on "
            f"the wrong months. Run it with --plan, or use the real stack")
    global OUT_VEC, OUT, PRE
    OUT_VEC = EXP / f"work/{st['out_vec']}"
    OUT = EXP / f"analysis/{st['out']}"
    # SECTION 47.55.  A stack's registration is DECLARED, never inherited.  It used to fall
    # through to a module-level PRE when a stack left it None, so the seasonal bag's report
    # carried section 47.20's band -- the one-month bag's, over v8's head -- and computed
    # `inside_the_registered_band` against it.  That is the fourth instance of the 47.43.4
    # class, so the repair is structural: no default, and the registration must NAME the head
    # this stack is priced over, exactly as `price_key` must.
    PRE = st["pre"]
    if PRE is not None and st["head_name"] not in PRE["gate"]:
        raise SystemExit(
            f"stack {args.stack!r} is priced over {st['head_name']!r} but its registration "
            f"gate says {PRE['gate']!r}. A registration that names another head is a "
            f"registration for another run (section 47.55)")
    t0 = time.time()
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    pan_ship = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})
    _, scols = A.shipped_cols(pan_ship)

    head_members = {s: {f: v.astype("float64") for f, v in
                        np.load(EXP / f"work/d-bundle-placebo{s}-255-free.npz").items()}
                    for s in TEN}
    bag10, spread_head = BAG.build_bag(head_members, pan_ship, TEN)
    if not all(np.array_equal(bag10[f], np.load(HEAD10)[f].astype("float64"))
               for f in A.FOLDS):
        raise SystemExit("the rebuilt K=10 head is not the vector v8 ships")
    print("v8's head rebuilds bit for bit from its ten members", flush=True)

    def _stack_frame(months: list, label: str, cache: dict) -> "tuple":
        """Concatenate a month list into one fitting frame, proving the concat by row count."""
        frame, rows = None, {}
        for mo in months:
            if mo not in cache:
                r = OM.nov_rows(scols, month=mo)
                cache[mo] = r[r.usable.to_numpy(bool)].reset_index(drop=True)
            rf = cache[mo]
            want = OM.MONTHS[mo].get("rows")
            if want is not None and len(rf) != want:
                raise SystemExit(f"{mo} gives {len(rf)} writable rows, not the {want} registered")
            if frame is None:
                frame = rf
            else:
                miss = [c for c in frame.columns if c not in rf.columns]
                if miss:
                    raise SystemExit(f"{mo} lacks {len(miss)} column(s) the stack carries: "
                                     f"{miss[:8]}")
                frame = pd.concat([frame, rf.reindex(columns=frame.columns)], ignore_index=True)
            rows[mo] = int(len(rf))
            print(f"{label}{mo} writable rows: {len(rf):,}  ->  extension total "
                  f"{len(frame):,}", flush=True)
        if sum(rows.values()) != len(frame):
            raise SystemExit(f"{label}the stack did not concatenate to the sum of its months")
        return frame, rows

    _cache: dict = {}
    if isinstance(st["months"], dict):
        # SECTION 47.49: a per-fold map. Every fold must be named -- a fold missing from it
        # would fit with no extension at all while the run is labelled seasonal.
        miss = [f for f in A.FOLDS if f not in st["months"]]
        if miss:
            raise SystemExit(f"stack {args.stack!r} is a per-fold month map that names no set "
                             f"for {miss} -- those folds would fit unextended")
        nov_fit, month_rows = {}, {}
        for f in A.FOLDS:
            nov_fit[f], month_rows[f] = _stack_frame(st["months"][f], f"{f}: ", _cache)
        if len({tuple(sorted(v)) for v in month_rows.values()}) == 1:
            raise SystemExit("every fold was given the SAME month set through the per-fold "
                             "path; that is the single-set case and must be run as one, so "
                             "the record cannot claim a seasonal head that is not one")
    else:
        nov_fit, month_rows = _stack_frame(st["months"], "", _cache)

    if args.plan:
        rows = (month_rows if isinstance(nov_fit, dict) else {"(one set)": month_rows})
        plan = {"what": ("section 47.49 (c): the seasonal stack's MECHANICS, no fit. The "
                         "extension frames are BUILT -- every month's base6 and block are "
                         "checked by sha to get this far -- and the run stops before the "
                         "first member"),
                "stack": args.stack, "months": st["months"],
                "months_are_per_fold": isinstance(st["months"], dict),
                "extension_rows": ({f: int(len(v)) for f, v in nov_fit.items()}
                                   if isinstance(nov_fit, dict) else int(len(nov_fit))),
                "month_rows": rows,
                "member_caches": [
                    str((EXP / f"work/{st['member1']}" if (s == 1 and st["member1"])
                         else EXP / f"work/{st['member_prefix']}-m{s}-255-free.npz")
                        .relative_to(ROOT)) for s in TEN],
                "head": st["head"], "head_name": st["head_name"],
                "price_key": st["price_key"], "out_vec": st["out_vec"]}
        out = EXP / f"analysis/{st['out'].replace('.json', '')}-plan.json"
        out.write_text(json.dumps(plan, indent=2, sort_keys=True, default=float))
        print(json.dumps(plan, indent=2, sort_keys=True, default=float))
        print(f"\nwritten {out.relative_to(ROOT)} -- PLAN ONLY, nothing was fitted",
              flush=True)
        return

    members, census = {}, {}
    for seed in TEN:
        t1 = time.time()
        cache = (EXP / f"work/{st['member1']}" if (seed == 1 and st["member1"])
                 else EXP / f"work/{st['member_prefix']}-m{seed}-255-free.npz")
        if cache.exists():
            members[seed] = {f: v.astype("float64") for f, v in np.load(cache).items()}
            census[seed] = {"refitted_this_run": False, "from": cache.name}
            print(f"member {seed}: from cache ({cache.name})", flush=True)
            continue
        rng = np.random.default_rng(seed)
        blocks = {}
        for m, b in blk.items():
            pb, _ = CP.placebo_block(b, R.THR, CF.GAP_CUT_S, CP.MOVED[m], rng)
            blocks[m] = pb
        real = A.month_block
        A.month_block = (lambda bs: (lambda month, prefix_months, rule=None: bs[month]))(blocks)
        try:
            pan = A.panel_with_block(BA.load())
        finally:
            A.month_block = real
        setting_was, hold_was = CF.SETTING, CF.HOLD
        CF.SETTING, CF.HOLD = {"num_leaves": LEAVES}, ()
        prov: dict = {}
        try:
            d = CF.fit_arm(pan, prov=prov, extra=nov_fit,
                           folds=tuple(st["fit_folds"]) if st.get("fit_folds") else None)
            if st.get("fit_folds"):
                # SECTION 47.58: the folds this stack does not fit come from the SAME member
                # of the stack it is priced over. Reuse is only correct when the month set is
                # identical, so that is asserted here rather than trusted.
                src = EXP / f"work/{st['reuse_members'].format(seed=seed)}"
                if not src.exists():
                    raise SystemExit(
                        f"member {seed}: {src.name} does not exist, so the folds this stack "
                        f"does not fit have nothing to come from. Build {st['reuse_stack']!r} "
                        f"first -- do NOT refit them here, because a refit reproduces them "
                        f"only to the ULP class (section 47.50) and the increment on an "
                        f"unchanged fold must be EXACTLY zero")
                other = STACKS[st["reuse_stack"]]["months"]
                reused = {f: v.astype("float64") for f, v in np.load(src).items()}
                for f in A.FOLDS:
                    if f in st["fit_folds"]:
                        continue
                    if st["months"][f] != other[f]:
                        raise SystemExit(
                            f"member {seed}: fold {f} is being REUSED from "
                            f"{st['reuse_stack']!r} but its month set here is "
                            f"{st['months'][f]} against {other[f]} there. Reusing a fold "
                            f"whose set changed would ship a head nobody fitted")
                    if f not in reused:
                        raise SystemExit(f"member {seed}: {src.name} carries no {f}")
                    d[f] = reused[f]
        finally:
            CF.SETTING, CF.HOLD = setting_was, hold_was
        _fitted = tuple(st["fit_folds"]) if st.get("fit_folds") else tuple(A.FOLDS)
        if any("extra_rows" not in prov.get(f, {}) for f in _fitted):
            raise SystemExit(f"member {seed}: no extra-row provenance -- it fitted the plain "
                             f"panel while being labelled an extension member")
        np.savez_compressed(cache, **d)
        members[seed] = d
        census[seed] = {"refitted_this_run": True, "seconds": round(time.time() - t1, 1),
                        "folds_fitted": list(_fitted),
                        "folds_reused_from": (None if not st.get("fit_folds")
                                              else st["reuse_members"].format(seed=seed)),
                        "extra_rows": {f: prov[f]["extra_rows"] for f in _fitted},
                        "total_fit_rows": {f: prov[f]["total_fit_rows"] for f in _fitted}}
        _p0 = prov[_fitted[0]]
        print(f"member {seed}: fitted ({time.time()-t1:.0f}s) folds {list(_fitted)}  "
              f"{_p0['total_fit_rows']:,} {_fitted[0]} rows "
              f"({_p0['extra_multiple']:.3f}x)", flush=True)

    bag, spread = BAG.build_bag(members, pan_ship, TEN)
    np.savez_compressed(OUT_VEC, **bag)

    head_vec = ({f: v.astype("float64") for f, v in np.load(EXP / f"work/{st['head']}").items()}
                if st["head"] != "d-bag10-255-free.npz" else bag10)
    inc = R.price(pan_ship, bag, head_vec,
                  f"the K=10 extended bag as an increment over {st['head_name']}")
    g = inc["board_all_row"]["gain_s"]
    ci = inc["board_all_row"]["interval"]["ci95"]

    # SECTION 47.43.4.  The price key NAMES the head it is priced over.  A key hard-coded to
    # v8 outlived the head it described -- the three-month stack is priced over v9's head --
    # and queue's v10 audit had to read `priced_over` to discover it.  The key is declared on
    # the stack beside `head_name` and the two are asserted to agree, so the key cannot lie
    # about its own content.
    price_key = st["price_key"]
    if price_key != f"in_fold_over_{st['head_name'].split(chr(0x27))[0]}_head":
        raise SystemExit(
            f"stack {args.stack!r} declares price_key {price_key!r} but head_name "
            f"{st['head_name']!r} -- the key and priced_over would disagree in the record, "
            f"which is the defect queue caught on threemonth-bag10.json")
    rec = {"what": __doc__.strip().splitlines()[0], "seeds": TEN, "leaves": LEAVES,
           "pre_registration": PRE, "members": census,
           "nov_writable_rows": ({f: int(len(v)) for f, v in nov_fit.items()}
                                 if isinstance(nov_fit, dict) else int(len(nov_fit))),
           "prices_key_note": PRICE_KEY_NOTE,
           "member_spread_extended": spread, "member_spread_head": spread_head,
           "head_rebuilds_bit_for_bit": True,
           "vector": str(OUT_VEC.relative_to(ROOT)),
           "prices": {price_key: inc},
           "stack": args.stack, "months": st["months"], "month_rows": month_rows,
           "months_are_per_fold": isinstance(st["months"], dict),
           "priced_over": st["head_name"],
           "cross_fitted_beside_it": R.crossfit(
               pan_ship, bag, head_vec, "the same increment, cross-fitted, beside it",
               arm_name=f"the {len(st['months'])}-month bag at {LEAVES} leaves",
               head_name=st["head_name"]),
           "verdict": {
               "board_gain_s": g, "ci95": ci,
               "every_fold_interval_clear": bool(inc["every_fold_interval_clear"]),
               "folds_clear": {f: bool(inc["by_fold"][f]["ci95"][0] > 0) for f in A.FOLDS},
               "passes_the_gate": bool(inc["every_fold_interval_clear"] and g > 0),
               "inside_the_registered_band": (
                   None if PRE is None or PRE.get("band_s") is None
                   else bool(PRE["band_s"][0] <= g <= PRE["band_s"][1])),
               "registered_point_board_s": None if PRE is None else PRE["point_estimate_board_s"],
               "at_the_naive_figure": (
                   None if PRE is None
                   or PRE.get("the_naive_figure_deliberately_not_registered") is None
                   else bool(abs(g - PRE["the_naive_figure_deliberately_not_registered"]) < 0.03)),
               "above_the_upload_floor_0_1_s": bool(g > 0.1)}}
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    print(); print(R.row(inc)); print(R.fold_rows(inc))
    v = rec["verdict"]
    if PRE is None:
        print("\n  NO registration is declared for this stack: no band claim is made")
    elif PRE.get("band_s") is None:
        print(f"\n  registered point {PRE['point_estimate_board_s']:+.5f} "
              f"({PRE['registered_in']}); NO board band was registered, so no band claim")
    else:
        print(f"\n  registered {PRE['point_estimate_board_s']:+.2f} in "
              f"[{PRE['band_s'][0]:+.2f}, {PRE['band_s'][1]:+.2f}]")
    print(f"  read       {g:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]  inside "
          f"{v['inside_the_registered_band']}  | at the naive figure "
          f"{v['at_the_naive_figure']}")
    print(f"  every-fold-clear {v['every_fold_interval_clear']}  GATE "
          f"{v['passes_the_gate']}  above the 0.1 s floor "
          f"{v['above_the_upload_floor_0_1_s']}")
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
