"""The next candidate priced APART and then whole: the 511 rung, the plane-metric class,
and the LFPG rider, each as an increment over v18's bytes (desk, 2026-09-09).

Pre-registered in RESULTS section 33, committed before this file was run. Four vectors,
three fits:

    A  511 leaves, SHIPPED class      the rung alone
    B  255 leaves, CORRECTED class    the plane-metric correction alone
    C  511 leaves, CORRECTED class    rung + correction
    D  C with the LFPG hold released  the whole, with the rider

`chainfold.fit_arm` applies the airport hold as its LAST step, after the fit, the floor and
the guard, so **C and D are one fit**: it is run with `HOLD = ()` and D is that vector, C is
that vector with LFPG zeroed. Fitting twice would put LightGBM's threaded non-reproducibility
(section 24) inside the rider, which is the one thing the rider must not carry.

Everything is priced against **H**, the vector v18's bytes ship
(`work/d-bundle-ungated1299-guard995.npz`), so H->A attributes the rung, A->C the correction,
C->D the rider, and H->D is the whole the registered every-fold-clear gate applies to.

The rung is selected on the folds, so it is also read CROSS-FITTED by `crossfit.py`'s rule:
for each scored fold the rung is taken only if, on both other folds, its increment interval
is clear of zero AND its registered whole-update tail clause passes.

**Constraints, from the order.** The shipped probe and its `-ungated` caches are read and
never written; the corrected block is the `-ungated-km` sibling section 32.1 built. Mode-3
pins 1,170 / 861 / 731 are asserted before every build and a miss is fatal.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/rung511.py
"""
from __future__ import annotations

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
import planekill as PK                                                # noqa: E402

THR = 1299.0
HOLD = ("LFPG",)
HEAD = EXP / "work/d-bundle-ungated1299-guard995.npz"
HEAD_SHA = ("5dbbcb8c7d14831fd75debde479071cd"
            "2a2eff9b7ff9b7469b6625ce00238684")
#: the free (LFPG unheld) fit per stage; the held vectors are derived from these, never
#: fitted a second time
FREE = {"A": EXP / "work/d-bundle-ungated1299-511-free.npz",
        "B": EXP / "work/d-bundle-km1299-255-free.npz",
        "C": EXP / "work/d-bundle-km1299-511-free.npz"}
HELD = {"A": EXP / "work/d-bundle-ungated1299-511-guard995.npz",
        "B": EXP / "work/d-bundle-km1299-255-guard995.npz",
        "C": EXP / "work/d-bundle-km1299-511-guard995.npz"}
OUT = EXP / "analysis/rung511.json"
LEVEL = 246.3605                    # v18's board reading; nothing is fitted to it (score-feedback rule)
N_RANKING = 344841


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_mode3(blk, month: str, which: str) -> int:
    n3 = int((blk.match_mode.fillna(-1) == 3).sum())
    if n3 != CF.MODE3_REFERENCE[month]:
        raise SystemExit(f"{which} {month}: mode-3 rows {n3} != "
                         f"{CF.MODE3_REFERENCE[month]} -- a shipped placement moved")
    return n3


def hold_lfpg(pan: dict, d: dict) -> dict:
    """The airport hold, applied exactly where `chainfold.fit_arm` applies it: last."""
    out = {}
    for f in A.FOLDS:
        v = d[f].astype("float64").copy()
        v[np.isin(pan[f].airport.to_numpy().astype(str), list(HOLD))] = 0.0
        out[f] = v
    return out


def fit_free(pan: dict, leaves: int, cache: Path) -> tuple[dict, bool]:
    """One fit at `leaves`, LFPG NOT held, cached under its own name."""
    if cache.exists():
        z = np.load(cache)
        return {f: z[f].astype("float64") for f in A.FOLDS}, False
    setting_was, hold_was = CF.SETTING, CF.HOLD
    CF.SETTING, CF.HOLD = {"num_leaves": leaves}, ()
    try:
        d = CF.fit_arm(pan)
    finally:
        CF.SETTING, CF.HOLD = setting_was, hold_was
    np.savez_compressed(cache, **d)
    return d, True


def head_panel(pan: dict, d_head: dict) -> dict:
    """`pbase + head`, the base an increment over v18's bytes is read against."""
    out = {}
    for f in A.FOLDS:
        p = pan[f].copy()
        p["pbase"] = p.pbase.to_numpy("float64") + d_head[f]
        p["rbase"] = p.rbase.to_numpy("float64") - d_head[f]
        out[f] = p
    return out


def price(pan: dict, d_new: dict, d_head: dict, label: str) -> dict:
    """`price_increment`, with the two tail readings named apart."""
    rec = CF.price_increment(pan, d_new, d_head, np.random.default_rng(A.SEED))
    ph = head_panel(pan, d_head)
    rec["tail_clause_on_the_whole_update"] = rec.pop("tail")
    rec["tail_clause_on_the_increment"] = {
        f: A.tail_clauses(ph[f], d_new[f] - d_head[f])["registered_clause_y_over_3600s"]
        for f in A.FOLDS}
    rec["label"] = label
    return rec


def crossfit(pan: dict, d_rung: dict, d_head: dict, label: str,
             arm_name: str = "leaves_511",
             head_name: str = "leaves_255 (the head)") -> dict:
    """`crossfit.py`'s rule, on a two-point ladder: keep 255, or climb to 511.

    For each scored fold the rung is taken only if, on BOTH other folds, its increment
    interval over the head is clear of zero and its registered whole-update tail clause
    passes.  Otherwise the increment on that fold is exactly zero.

    ROUND 12, N2: `arm_name` and `head_name` default to the 511/255 rung this was written for,
    so section 41's readings are unchanged -- but every OTHER caller was recording
    `"chosen": "leaves_511"` on a 255-leaf arm.  No value depended on it and it was a label,
    which is exactly the class of defect 47.43.4 is about.  The caller names its own arm.
    """
    rec = CF.price_increment(pan, d_rung, d_head, np.random.default_rng(A.SEED))
    tails = {f: A.tail_clauses(pan[f], d_rung[f])["registered_clause_y_over_3600s"]
             for f in A.FOLDS}
    picks, d_cross = {}, {}
    for f in A.FOLDS:
        sel = [g for g in A.FOLDS if g != f]
        ok = {g: {"interval_clear": bool(rec["by_fold"][g]["ci95"][0] > 0),
                  "tail_passes": bool(tails[g]["passes_the_clause"]),
                  "mse_gain_s2_per_row": rec["by_fold"][g]["mse_gain_s2"]} for g in sel}
        elig = all(v["interval_clear"] and v["tail_passes"] for v in ok.values())
        picks[f] = {"selected_on": sel, "eligible": elig,
                    "chosen": arm_name if elig else head_name,
                    "evidence_on_the_selection_folds": ok}
        d_cross[f] = d_rung[f] if elig else d_head[f]
    rec_x = CF.price_increment(pan, d_cross, d_head, np.random.default_rng(A.SEED))
    return {"label": label,
            "in_fold": {"all_row_gain_s": rec["board_all_row"]["gain_s"],
                        "all_row_ci95": rec["board_all_row"]["interval"]["ci95"],
                        "clean_gain_s": rec["board_clean"]["gain_s"],
                        "every_fold_interval_clear": rec["every_fold_interval_clear"]},
            "cross_fitted": {"all_row_gain_s": rec_x["board_all_row"]["gain_s"],
                             "all_row_ci95": rec_x["board_all_row"]["interval"]["ci95"],
                             "clean_gain_s": rec_x["board_clean"]["gain_s"],
                             "every_fold_interval_clear": rec_x["every_fold_interval_clear"],
                             "by_fold": rec_x["by_fold"]},
            "out_of_fold_selection": picks,
            "selection_premium_s": (rec["board_all_row"]["gain_s"]
                                    - rec_x["board_all_row"]["gain_s"]),
            "rule": ("crossfit.py: the rung is taken on a fold only if, on both other "
                     "folds, its increment interval is clear of zero and its registered "
                     "whole-update tail clause passes")}


def face(g_s2_per_row: float, n: int) -> dict:
    """Section 16.3's level currency, on v18's board reading."""
    gg = g_s2_per_row * n / N_RANKING
    lvl = LEVEL - (LEVEL ** 2 - gg) ** 0.5 if LEVEL ** 2 > gg else float("nan")
    return {"rows": n, "board_s_at_x1_2": lvl, "board_s_at_x1_5": lvl * 1.492}


def row(rec: dict) -> str:
    b = rec["board_all_row"]
    return ("  all-row %+8.5f [%+8.5f,%+8.5f]  clean %+8.5f  every-fold-clear %s"
            % (b["gain_s"], b["interval"]["ci95"][0], b["interval"]["ci95"][1],
               rec["board_clean"]["gain_s"], rec["every_fold_interval_clear"]))


def fold_rows(rec: dict) -> str:
    out = []
    for f in A.FOLDS:
        r, c = rec["by_fold"][f], rec["by_fold_clean"][f]
        t = rec["tail_clause_on_the_whole_update"][f]
        out.append("    %-9s moved %6d  raw %+9.2f [%+8.2f,%+8.2f]  clean %+9.2f "
                   "[%+8.2f,%+8.2f]  tail %+8.3f %s"
                   % (f, rec["rows_moved"][f], r["mse_gain_s2"], r["ci95"][0], r["ci95"][1],
                      c["mse_gain_s2"], c["ci95"][0], c["ci95"][1],
                      t["rmse_gain_s"], "PASS" if t["passes_the_clause"] else "FAIL"))
    return "\n".join(out)


def main() -> None:
    t0 = time.time()
    if sha256(HEAD) != HEAD_SHA:
        raise SystemExit(f"{HEAD.name} is not the vector v18's bytes ship")
    rule = A.serving_rule()[0]
    out = {"what": __doc__.strip().splitlines()[0],
           "pre_registered_in": "RESULTS.md section 33 (commit a68e5e3)",
           "head": {"file": str(HEAD.relative_to(ROOT)), "sha256": HEAD_SHA,
                    "is": "v18's bytes on the fold side: 255 leaves, guard995, floor, "
                          "LFPG held, ungated class at 1299 s"},
           "threshold_s": THR, "hold": list(HOLD), "level_s": LEVEL,
           "mode3_reference": CF.MODE3_REFERENCE, "prices": {}}

    # ------------------------------------------------------------------ the two classes
    blk_ship, blk_km, mode3 = {}, {}, {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        blk_ship[m] = b
        n3 = assert_mode3(b, m, "shipped")
        k = PK.build_km(m, rule)
        blk_km[m] = k
        assert_mode3(k, m, "corrected")
        mode3[m] = n3
        print(f"{m}: mode-3 {n3} asserted on both builds  ({time.time()-t0:.0f}s)",
              flush=True)
    out["mode3_asserted"] = mode3

    gaps = {m: CF.GAP_CUT_S for m in blk_ship}
    pan_ship = CF.panel_for(blk_ship, THR, gaps)
    pan_km = CF.panel_for(blk_km, THR, gaps)
    added = {}
    for name, pan in (("shipped", pan_ship), ("corrected", pan_km)):
        added[name] = {f: int(((pan[f].match_mode.fillna(-1) == 4)
                               & pan[f].usable.to_numpy(bool)).sum()) for f in A.FOLDS}
    out["rows_added_to_the_class"] = added
    out["class_change"] = {f: added["corrected"][f] - added["shipped"][f] for f in A.FOLDS}
    print("class added:", added, "change:", out["class_change"], flush=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    zh = np.load(HEAD)
    H = {f: zh[f].astype("float64") for f in A.FOLDS}

    # -------------------------------------------------------------------- the three fits
    stages = [("A", "the rung alone: 511 leaves on the SHIPPED class", pan_ship, 511),
              ("B", "the correction alone: 255 leaves on the CORRECTED class", pan_km, 255),
              ("C", "rung + correction: 511 leaves on the CORRECTED class", pan_km, 511)]
    free, kept = {}, {}
    for key, what, pan, leaves in stages:
        t1 = time.time()
        d_free, refit = fit_free(pan, leaves, FREE[key])
        d_held = hold_lfpg(pan, d_free)
        np.savez_compressed(HELD[key], **d_held)
        free[key], kept[key] = d_free, d_held
        rec = price(pan, d_held, H, what)
        rec["leaves"] = leaves
        rec["class"] = "corrected" if pan is pan_km else "shipped"
        rec["vector"] = str(HELD[key].relative_to(ROOT))
        rec["refitted_this_run"] = refit
        rec["seconds"] = round(time.time() - t1, 1)
        out["prices"][key] = rec
        print(f"\n[{key}] {what}   ({rec['seconds']:.0f}s, refit {refit})", flush=True)
        print(row(rec), flush=True)
        print(fold_rows(rec), flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # ------------------------------------------------------- D: the whole, with the rider
    D = free["C"]                       # the same fit, the hold not applied
    rec = price(pan_km, D, H, "THE WHOLE: 511 leaves, corrected class, LFPG released")
    rec["leaves"], rec["class"] = 511, "corrected"
    rec["vector"] = str(FREE["C"].relative_to(ROOT))
    out["prices"]["D"] = rec
    print("\n[D] THE WHOLE over v18's bytes", flush=True)
    print(row(rec), flush=True)
    print(fold_rows(rec), flush=True)

    rid = price(pan_km, D, kept["C"], "the LFPG rider alone, on top of C")
    rid["lit_rule"] = {
        "lit_folds": [f for f in A.FOLDS if rid["rows_moved"][f] > 0],
        "dark_folds": rid["folds_with_no_moved_row"],
        "every_moving_fold_interval_clear": rid["every_moving_fold_interval_clear"]}
    out["prices"]["rider"] = rid
    print("\n[rider] LFPG released, on top of C", flush=True)
    print(row(rid), flush=True)
    print("    lit", rid["lit_rule"]["lit_folds"], "dark", rid["lit_rule"]["dark_folds"],
          "moving-clear", rid["every_moving_fold_interval_clear"], flush=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # ------------------------------------------------------------------ the cross-fits
    out["cross_fit"] = {
        "the_rung_on_the_shipped_class": crossfit(pan_ship, kept["A"], H,
                                                  "A over H: 511 vs 255, shipped class"),
        "the_rung_on_the_corrected_class": crossfit(pan_km, kept["C"], kept["B"],
                                                    "C over B: 511 vs 255, corrected class"),
    }
    for k, v in out["cross_fit"].items():
        print(f"\n[cross-fit] {v['label']}", flush=True)
        print("    in-fold      %+8.5f [%+8.5f,%+8.5f]"
              % (v["in_fold"]["all_row_gain_s"], *v["in_fold"]["all_row_ci95"]), flush=True)
        print("    CROSS-FITTED %+8.5f [%+8.5f,%+8.5f]  clear %s"
              % (v["cross_fitted"]["all_row_gain_s"], *v["cross_fitted"]["all_row_ci95"],
                 v["cross_fitted"]["every_fold_interval_clear"]), flush=True)
        print("    premium      %+8.5f" % v["selection_premium_s"], flush=True)
        for f, p in v["out_of_fold_selection"].items():
            print(f"      scored {f:10s} -> {p['chosen']}", flush=True)

    # ------------------------------------------------------------------ additivity check
    parts = (out["prices"]["A"]["board_all_row"]["gain_s"]
             + out["prices"]["B"]["board_all_row"]["gain_s"]
             + out["prices"]["rider"]["board_all_row"]["gain_s"])
    out["additivity"] = {
        "sum_of_the_parts_s": parts,
        "the_whole_s": out["prices"]["D"]["board_all_row"]["gain_s"],
        "difference_s": out["prices"]["D"]["board_all_row"]["gain_s"] - parts,
        "pre_registered_band_s": 0.02,
        "the_parts_interact": bool(
            abs(out["prices"]["D"]["board_all_row"]["gain_s"] - parts) > 0.02)}
    print("\nadditivity: parts %+8.5f  whole %+8.5f  difference %+8.5f  interact %s"
          % (parts, out["additivity"]["the_whole_s"], out["additivity"]["difference_s"],
             out["additivity"]["the_parts_interact"]), flush=True)

    # the stop rule, evaluated rather than left to the reader
    board_share = ["july_loo", "january"]
    tw = out["prices"]["A"]["tail_clause_on_the_whole_update"]
    out["stop_rule"] = {
        "rule": ("desk, 2026-09-09: if the rung fails the registered whole-update tail "
                 "clause on a board-share fold (july_loo or january; december carries no "
                 "board share), the table says so and the work stops"),
        "board_share_folds": board_share,
        "the_rung_A": {f: {"rmse_gain_s": tw[f]["rmse_gain_s"],
                           "passes": tw[f]["passes_the_clause"]} for f in A.FOLDS},
        "triggered": bool(not all(tw[f]["passes_the_clause"] for f in board_share))}
    print("\nSTOP RULE triggered:", out["stop_rule"]["triggered"], flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\nwritten to", OUT.relative_to(ROOT), f"({out['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
