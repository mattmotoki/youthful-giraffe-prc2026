"""The LFPG rider, priced on top of v19 under the lit rule (desk, 2026-09-09).

v19 ships with LFPG held at the price base, as v16 and v18 did. The rider is the question
the desk deferred: what is it worth to RELEASE that hold on top of the candidate as it
stands? Section 26 priced the release on the v18-era shipped block and read +0.0138
[+0.0014, +0.0294] over the hold. v19's class is bigger -- the turnaround pass and the
gated ungated chain rows, 285 of which are LFPG -- so the number has to be taken again on
the class the candidate actually carries.

**The lit rule** (desk, 2026-09-09, section 26.5): *an EMPTY lit fold is not a crossing --
a cell dark by instrument is priced on its lit folds (>= 2, december among them for a
winter ranking month)*. LFPG january moves no row, so the gate is july and december, and
`base.price`'s `every_moving_fold_interval_clear` is the reading, with
`folds_with_no_moved_row` beside it saying which fold was dark.

**Why nothing is refitted from scratch.** `chainfold.fit_arm` applies the airport hold as
its LAST step, after the LightGBM fit, the floor and the guard. So the rider and v19 are
one and the same fit and differ only in that line. The fit is re-run here only to recover
the LFPG entries the saved vector zeroed, and then SPLICED: every non-LFPG row takes v19's
own shipped value, so the increment is nonzero on LFPG rows and nowhere else by
construction rather than by agreement. The agreement is checked anyway and reported --
LightGBM's threaded leaf summation is not bit-reproducible (section 24), so the refit is
expected to differ from the shipped vector by about a ULP off LFPG and by nothing more.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/lfpgrider.py
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
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import chainfold as CF                                                # noqa: E402

AIRPORT = "LFPG"
THR = 1299.0
V19 = EXP / "work/d-bundle-ungated1299-guard995.npz"
V19_SHA = ("5dbbcb8c7d14831fd75debde479071cd"
           "2a2eff9b7ff9b7469b6625ce00238684")
#: the UNHELD fit -- what is cached, because it is the whole cost of this file.  Caching
#: the SPLICED rider instead would make the agreement check below vacuous on a re-run: the
#: spliced vector equals v19 off LFPG by construction, so reading it back and comparing
#: would prove nothing.  The unheld fit is the thing the check has to see.
UNHELD = EXP / "work/d-lfpg-unheld-fit.npz"
RIDER = EXP / "work/d-rider-lfpg-on-v19.npz"
OUT = EXP / "analysis/lfpg-rider.json"
LEVEL = 246.6571
N_RANKING = 344841
#: LFPG's ranking class under v19, read off `analysis/ranking-reach.json` (written by the
#: v19 serve): every LFPG departure the matcher gives a clock of its own
LFPG_RANKING = {"2026-01": 1136, "2026-07": 1210}


def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    if sha256(V19) != V19_SHA:
        raise SystemExit(f"{V19.name} is not the vector v19 shipped")
    rule = A.serving_rule()[0]
    blocks = {A.FOLD_MONTH[f]: CF.build_month(A.FOLD_MONTH[f],
                                              A.prefix_months_for(A.FOLD_MONTH[f]), rule)
              for f in A.FOLDS}
    gaps = {m: CF.GAP_CUT_S for m in blocks}
    pan = CF.panel_for(blocks, THR, gaps)
    z = np.load(V19)
    d_v19 = {f: z[f].astype("float64") for f in A.FOLDS}

    # the same fit with the hold lifted, for the LFPG entries only.  Saved, because the
    # fit is the whole cost of this file and the reading beside it is arithmetic on the
    # panel: a re-run picks the vector up rather than fitting a second, slightly different
    # one (LightGBM's threaded leaf summation is not bit-reproducible, section 24).
    if UNHELD.exists():
        zr = np.load(UNHELD)
        d_free = {f: zr[f].astype("float64") for f in A.FOLDS}
        refit = False
    else:
        held_was = CF.HOLD
        CF.HOLD = ()
        try:
            d_free = CF.fit_arm(pan)
        finally:
            CF.HOLD = held_was
        np.savez_compressed(UNHELD, **d_free)
        refit = True

    lf, d_rider, agree = {}, {}, {}
    for f in A.FOLDS:
        m = pan[f].airport.to_numpy().astype(str) == AIRPORT
        lf[f] = m
        v = d_v19[f].copy()
        v[m] = d_free[f][m]
        d_rider[f] = v
        off = ~m
        diff = np.abs(d_free[f][off] - d_v19[f][off])
        agree[f] = {
            "rows_off_lfpg": int(off.sum()),
            "refit_differs_from_the_shipped_vector_on": int((diff > 0).sum()),
            "max_abs_difference_s": float(diff.max()) if off.any() else 0.0,
            "max_relative_difference": float(
                (diff[diff > 0] / np.abs(d_v19[f][off][diff > 0])).max())
            if (diff > 0).any() else 0.0,
            "lfpg_rows": int(m.sum()),
            "lfpg_rows_the_rider_moves": int((v[m] != 0).sum()),
            "lfpg_rows_v19_moves": int((d_v19[f][m] != 0).sum()),
        }
    np.savez_compressed(RIDER, **d_rider)

    rec = CF.price_increment(pan, d_rider, d_v19, np.random.default_rng(A.SEED))

    # the face, measured on the LIT folds' released rows and never borrowed
    num = den = 0
    lit = [f for f in A.FOLDS if rec["rows_moved"][f] > 0]
    dark = [f for f in A.FOLDS if rec["rows_moved"][f] == 0]
    num = 0.0
    for f in lit:
        p = pan[f]
        inc = d_rider[f] - d_v19[f]
        sel = inc != 0
        r = p.rbase.to_numpy("float64") - d_v19[f]
        num += float((2 * r[sel] * inc[sel] - inc[sel] ** 2).sum())
        den += int(sel.sum())
    g = num / den if den else 0.0
    # **THE DENOMINATOR IS THE CLASS, not the airport.**  The ranking count this share is
    # projected onto (2,346) is LFPG's WITH-CLOCK rows, so the fold-side share has to be
    # taken over LFPG's with-clock fold rows too.  Taken over every LFPG fold row it reads
    # 4.3 % and understates the projection by a factor of about twenty.
    fold_lfpg = sum(int((lf[f] & pan[f].usable.to_numpy(bool)).sum()) for f in lit)
    fold_lfpg_all_rows = sum(int(lf[f].sum()) for f in lit)
    released_share = den / fold_lfpg if fold_lfpg else 0.0
    n_class = sum(LFPG_RANKING.values())
    face = {}
    for label, n in (("every LFPG class row", n_class),
                     ("projected at the lit folds' release share",
                      int(round(n_class * released_share)))):
        gg = g * n / N_RANKING
        lvl = LEVEL - (LEVEL ** 2 - gg) ** 0.5 if LEVEL ** 2 > gg else float("nan")
        face[label] = {"rows": n, "board_s_at_x1_2": lvl, "board_s_at_x1_5": lvl * 1.492}

    out = {
        "what": __doc__.strip().splitlines()[0],
        "airport": AIRPORT,
        "on_top_of": {"candidate": "submissions/candidates/exp-046-surface-v5.parquet",
                      "vector": str(V19.relative_to(ROOT)), "vector_sha256": V19_SHA},
        "rider_vector": str(RIDER.relative_to(ROOT)),
        "unheld_fit_vector": str(UNHELD.relative_to(ROOT)),
        "unheld_fit_was_refitted_this_run": refit,
        "construction": (
            "v19's own vector everywhere except LFPG, where the same fit with the hold "
            "lifted is spliced in; the increment is nonzero on LFPG rows and nowhere else "
            "by construction"),
        "refit_agrees_with_the_shipped_vector_off_lfpg": agree,
        "lit_rule": {
            "ruling": ("desk, 2026-09-09, section 26.5: an empty lit fold is not a "
                       "crossing; a cell dark by instrument is priced on its lit folds "
                       "(>= 2, december among them for a winter ranking month)"),
            "lit_folds": lit, "dark_folds": dark,
            "folds_with_no_moved_row": rec["folds_with_no_moved_row"],
            "every_moving_fold_interval_clear": rec["every_moving_fold_interval_clear"],
            "every_fold_interval_clear": rec["every_fold_interval_clear"],
            "gate_under_the_lit_rule": bool(
                rec["every_moving_fold_interval_clear"] and len(lit) >= 2
                and rec["board_all_row"]["gain_s"] > 0
                and rec["board_all_row"]["interval"]["ci95"][0] > 0)},
        "increment_over_v19": rec,
        "face": {
            "level_s": LEVEL,
            "level_is": "v17's board reading; nothing is fitted to it (D11)",
            "s2_per_released_row_measured_on": lit,
            "s2_per_released_row": g,
            "released_fold_rows": den,
            "lfpg_fold_rows_in_the_class_on_the_lit_folds": fold_lfpg,
            "lfpg_fold_rows_of_any_kind_on_the_lit_folds": fold_lfpg_all_rows,
            "release_share_on_the_lit_folds": released_share,
            "release_share_is": ("released fold rows / LFPG WITH-CLOCK fold rows on the "
                                 "lit folds: what the guard and the floor leave of the "
                                 "class, which is the fraction the ranking class would "
                                 "release too"),
            "lfpg_ranking_class_under_v19": LFPG_RANKING,
            "at": face},
        "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    b = rec["board_all_row"]
    print(f"rider over v19: {b['gain_s']:+.5f} "
          f"[{b['interval']['ci95'][0]:+.5f},{b['interval']['ci95'][1]:+.5f}]  "
          f"clean {rec['board_clean']['gain_s']:+.5f}", flush=True)
    for f in A.FOLDS:
        r = rec["by_fold"][f]
        print(f"  {f:10s} moved {rec['rows_moved'][f]:>6}  s2/row {r['mse_gain_s2']:>+9.2f}"
              f"  ci [{r['ci95'][0]:>+8.2f},{r['ci95'][1]:>+8.2f}]", flush=True)
    print("lit", lit, "dark", dark,
          "| moving-clear", rec["every_moving_fold_interval_clear"],
          "| gate", out["lit_rule"]["gate_under_the_lit_rule"])
    print("face", json.dumps(face, default=float))
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
