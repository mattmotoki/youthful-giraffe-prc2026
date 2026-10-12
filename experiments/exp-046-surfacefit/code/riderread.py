"""Read E: v18's bytes plus the LFPG rider ALONE -- the one increment with no refit inside it.

The desk, 2026-09-09, after the placebo's first draw:

> every fresh fit carries ~20 s/row of chaos whose energy costs ~0.06 s unless aligned gain
> exceeds it -- the rider is the one increment on the table with no refit inside it.

`work/d-rider-lfpg-on-v19.npz` is not a new fit.  It is **H's own fit** -- the vector v18's
bytes ship -- with the LFPG hold lifted: `lfpgrider.py` took the same fit with `HOLD = ()` and
spliced its LFPG entries into H, so the increment is nonzero on LFPG rows and **exactly zero
everywhere else by construction**.  That is checked here before it is priced, not asserted:
the provenance test is that the rider equals H off LFPG on 0 rows differing, on all three
folds.  With B = mean(inc**2) confined to a few hundred rows per fold, there is no chaos
energy to pay for.

Priced over v18's bytes per fold, raw and clean, with both tail readings, under the **lit
rule** (section 26.5): an EMPTY lit fold is not a crossing, so january -- where LFPG moves no
row by instrument -- is dark and the gate is `every_moving_fold_interval_clear` over july and
december.

Pre-registered expectation, from section 30: **+0.0194 [+0.0078, +0.0338]**, lit folds clear.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/riderread.py
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

RIDER = EXP / "work/d-rider-lfpg-on-v19.npz"
AIRPORT = "LFPG"
OUT = EXP / "analysis/rider-read.json"
PRE = {"all_row_gain_s": 0.0194, "ci95": [0.0078, 0.0338],
       "source": "RESULTS section 30 / analysis/lfpg-rider.json",
       "expectation": "lit folds clear under the lit rule"}


def main() -> None:
    t0 = time.time()
    if R.sha256(R.HEAD) != R.HEAD_SHA:
        raise SystemExit(f"{R.HEAD.name} is not the vector v18's bytes ship")
    H = {f: v.astype("float64") for f, v in np.load(R.HEAD).items()}
    Z = {f: v.astype("float64") for f, v in np.load(RIDER).items()}

    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})

    # ------------------------------------------------------------------ provenance FIRST
    prov = {}
    for f in A.FOLDS:
        lf = pan[f].airport.to_numpy().astype(str) == AIRPORT
        off = ~lf
        prov[f] = {
            "rows": int(len(H[f])),
            "lfpg_rows": int(lf.sum()),
            "rows_differing_from_the_head_OFF_lfpg": int((Z[f][off] != H[f][off]).sum()),
            "identical_off_lfpg": bool(np.array_equal(Z[f][off], H[f][off])),
            "lfpg_rows_the_head_moves": int((H[f][lf] != 0).sum()),
            "lfpg_rows_the_rider_moves": int((Z[f][lf] != 0).sum()),
            "increment_nonzero_rows": int((Z[f] != H[f]).sum()),
        }
    ok = all(p["identical_off_lfpg"] for p in prov.values())
    print("PROVENANCE: the rider is H off LFPG on every fold:", ok, flush=True)
    for f, p in prov.items():
        print(f"  {f:10s} {p['rows_differing_from_the_head_OFF_lfpg']} rows differ off LFPG "
              f"| LFPG {p['lfpg_rows']} rows, head moves {p['lfpg_rows_the_head_moves']}, "
              f"rider moves {p['lfpg_rows_the_rider_moves']}", flush=True)
    if not ok:
        raise SystemExit("the rider is not H off LFPG; it carries a refit and read E is void")

    # ------------------------------------------------------------------------- the price
    rec = R.price(pan, Z, H, "read E: v18's bytes + the LFPG rider alone, no refit")
    lit = [f for f in A.FOLDS if rec["rows_moved"][f] > 0]
    dark = rec["folds_with_no_moved_row"]
    dec = {}
    for f in A.FOLDS:
        r = rec["by_fold"][f]
        two_a, b = 2 * r["A_mean_rd"], r["B_mean_dd"]
        dec[f] = {"two_A_alignment": two_a, "B_churn": b,
                  "ratio_2A_over_B": (two_a / b) if b else None,
                  "rms_increment_s": float(np.sqrt(b)),
                  "gain_s2_per_row": r["mse_gain_s2"]}
    out = {"what": __doc__.strip().splitlines()[0],
           "vector": str(RIDER.relative_to(ROOT)),
           "head": {"file": str(R.HEAD.relative_to(ROOT)), "sha256": R.HEAD_SHA},
           "no_refit": ("H's own fit with the LFPG hold lifted and spliced; the increment is "
                        "nonzero on LFPG rows and exactly zero elsewhere, checked above"),
           "provenance": prov, "provenance_passes": ok,
           "pre_registered": PRE,
           "price_over_v18_bytes": rec,
           "decomposition": dec,
           "lit_rule": {
               "ruling": ("section 26.5: an EMPTY lit fold is not a crossing; a cell dark by "
                          "instrument is priced on its lit folds (>= 2, december among them "
                          "for a winter ranking month)"),
               "lit_folds": lit, "dark_folds": dark,
               "every_moving_fold_interval_clear": rec["every_moving_fold_interval_clear"],
               "every_fold_interval_clear": rec["every_fold_interval_clear"],
               "gate_under_the_lit_rule": bool(
                   rec["every_moving_fold_interval_clear"] and len(lit) >= 2
                   and rec["board_all_row"]["gain_s"] > 0
                   and rec["board_all_row"]["interval"]["ci95"][0] > 0)},
           "against_the_pre_registration": {
               "read_all_row_s": rec["board_all_row"]["gain_s"],
               "expected_s": PRE["all_row_gain_s"],
               "difference_s": rec["board_all_row"]["gain_s"] - PRE["all_row_gain_s"],
               "inside_the_expected_interval": bool(
                   PRE["ci95"][0] <= rec["board_all_row"]["gain_s"] <= PRE["ci95"][1])},
           "seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\n[E] the LFPG rider alone over v18's bytes", flush=True)
    print(R.row(rec), flush=True)
    print(R.fold_rows(rec), flush=True)
    print("  decomposition (2A alignment vs B churn):", flush=True)
    for f, d in dec.items():
        rr = d["ratio_2A_over_B"]
        print(f"    {f:10s} 2A {d['two_A_alignment']:+10.4f}  B {d['B_churn']:10.4f}  "
              f"2A/B {('%.2f' % rr) if rr else 'n/a':>6}  rms |inc| "
              f"{d['rms_increment_s']:.3f} s", flush=True)
    print("  lit", lit, "dark", dark, "| moving-clear",
          rec["every_moving_fold_interval_clear"],
          "| gate", out["lit_rule"]["gate_under_the_lit_rule"], flush=True)
    print("  vs pre-registered +0.0194 [+0.0078,+0.0338]: inside =",
          out["against_the_pre_registration"]["inside_the_expected_interval"], flush=True)
    print("written to", OUT.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
