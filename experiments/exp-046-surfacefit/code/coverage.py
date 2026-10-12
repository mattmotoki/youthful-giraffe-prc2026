"""The coverage process: the folds this lane measures on are not the population it serves.

§47.6.  Whether a row is WRITABLE here is decided by whether its airport had an observation
clock in that month, and that is not a stable property.  This file measures it three ways and
writes them all to `analysis/coverage.json`:

  1. `clock_coverage`  -- the fraction of each airport's rows that carry a clock, in each of
     the three fold months and the two ranking months.  A cell moving 87 % -> 0 % between two
     fold months is not sampling noise.
  2. `writable_composition` -- the board-weighted airport composition of the writable
     population on the folds against the ranking side, and `w = rank / fold`, the weight a
     composition-matched head would carry (§47.9's `c_lgb_covmatch`).
  3. `matching_cost` -- the Kish effective sample size that weighting would leave, which is the
     cost of the arm and is registered before it is fitted, not after.

No labels, no fits, no candidate is read: this is a census of which rows exist.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/exp-046-surfacefit"
sys.path.insert(0, str(EXP / "code"))
sys.path.insert(0, str(ROOT))

import arms as A  # noqa: E402
import chainfold as CF  # noqa: E402
import rung511 as R  # noqa: E402
import serve as S  # noqa: E402

OUT = EXP / "analysis/coverage.json"

#: the board shares, fold side and ranking side.  december carries none, by registration.
FOLD_SHARE = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}
RANK_SHARE = {"2026-07": 0.5571321275602379, "2026-01": 0.4428678724397621}


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

    # the ranking side is read UNGATED, the class v8 was served on
    S._UNGATED = True
    dep, _ = S.ranking_block()
    rap = dep.airport.astype(str).to_numpy()
    rrm = dep.rank_month.astype(str).to_numpy()
    rus = dep.usable.to_numpy(bool)

    airports = sorted(set(np.unique(rap)) |
                      {a for f in A.FOLDS
                       for a in np.unique(pan[f].airport.astype(str).to_numpy())})

    # 1. clock coverage, every airport in every month
    cov = {}
    for a in airports:
        row = {}
        for f in A.FOLDS:
            p = pan[f]
            ap = p.airport.astype(str).to_numpy()
            us = p.usable.to_numpy(bool)
            n = int((ap == a).sum())
            row[A.FOLD_MONTH[f]] = {"rows": n, "with_clock": int((us & (ap == a)).sum()),
                                    "coverage": (float((us & (ap == a)).sum() / n)
                                                 if n else 0.0)}
        for m in RANK_SHARE:
            sel = rrm == m
            n = int((sel & (rap == a)).sum())
            row[m] = {"rows": n, "with_clock": int((sel & rus & (rap == a)).sum()),
                      "coverage": (float((sel & rus & (rap == a)).sum() / n) if n else 0.0)}
        cov[a] = row

    # 2. board-weighted composition of the WRITABLE population, both sides
    fold_mass, rank_mass = {}, {}
    for a in airports:
        fold_mass[a] = sum(
            sh * float((pan[f].usable.to_numpy(bool)
                        & (pan[f].airport.astype(str).to_numpy() == a)).sum()) / len(pan[f])
            for f, sh in FOLD_SHARE.items())
        rank_mass[a] = sum(
            sh * float(((rrm == m) & rus & (rap == a)).sum()) / float((rrm == m).sum())
            for m, sh in RANK_SHARE.items())
    fden, rden = sum(fold_mass.values()), sum(rank_mass.values())
    comp = {}
    for a in airports:
        pf, pr = fold_mass[a] / fden, rank_mass[a] / rden
        comp[a] = {"share_of_writable_folds": pf, "share_of_writable_ranking": pr,
                   "weight_rank_over_fold": (pr / pf) if pf > 0 else None,
                   "row_mass_folds": fold_mass[a], "row_mass_ranking": rank_mass[a]}
    tv = 0.5 * sum(abs(comp[a]["share_of_writable_folds"]
                       - comp[a]["share_of_writable_ranking"]) for a in airports)

    # 3. what matching would cost: Kish effective sample size under w = p_rank / p_fold
    chi = sum(comp[a]["share_of_writable_ranking"] ** 2 / comp[a]["share_of_writable_folds"]
              for a in airports if comp[a]["share_of_writable_folds"] > 0)
    board_rows = sum(int(pan[f].usable.to_numpy(bool).sum()) for f in FOLD_SHARE)

    rec = {
        "what": ("§47.6 -- clock coverage is not stable by airport or by month, so the "
                 "writable population the folds measure is not the writable population the "
                 "board is scored on. No labels and no fits are read here."),
        "clock_coverage": cov,
        "writable_composition": comp,
        "writable_row_mass": {"folds": fden, "ranking": rden, "ratio": rden / fden},
        "total_variation_distance": tv,
        "matching_cost": {
            "sum_p_rank_squared_over_p_fold": chi,
            "kish_effective_sample_fraction": 1.0 / chi,
            "writable_rows_on_the_board_folds": board_rows,
            "effective_rows_after_matching": int(board_rows / chi),
            "what": ("the cost of §47.9's c_lgb_covmatch, registered before it is fitted: "
                     "matching the composition throws away this much of the effective sample"),
        },
        "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    print(f"{'airport':8s}" + "".join(f"{A.FOLD_MONTH[f]:>10s}" for f in A.FOLDS)
          + "".join(f"{m:>10s}" for m in RANK_SHARE) + "   clock coverage")
    for a in airports:
        print(f"{a:8s}" + "".join(f"{cov[a][A.FOLD_MONTH[f]]['coverage']*100:9.1f}%"
                                  for f in A.FOLDS)
              + "".join(f"{cov[a][m]['coverage']*100:9.1f}%" for m in RANK_SHARE))
    print(f"\n{'airport':8s} {'fold %':>8s} {'rank %':>8s} {'w':>7s}")
    for a in sorted(airports, key=lambda a: -comp[a]["share_of_writable_ranking"]):
        w = comp[a]["weight_rank_over_fold"]
        print(f"{a:8s} {comp[a]['share_of_writable_folds']*100:7.2f}% "
              f"{comp[a]['share_of_writable_ranking']*100:7.2f}% "
              + (f"{w:7.2f}" if w is not None else "      -"))
    print(f"\nwritable row mass  folds {fden*100:.2f}%  ranking {rden*100:.2f}%  "
          f"ratio {rden/fden:.3f}")
    print(f"total variation distance {tv*100:.2f}%")
    print(f"Kish effective sample fraction {100/chi:.1f}%  "
          f"({board_rows} -> {int(board_rows/chi)} rows)")
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
