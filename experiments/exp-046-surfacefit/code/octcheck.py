"""An ID PRE-CHECK on queue's base6 2025-10, against this lane's own 2025-10 surface block.

Queue's 2025-10 base6 vector exists (sha 2bfc9a6959281d6e); its NEIGHBOUR block failed
verification at 185 -> 63 columns and is being re-emitted.  So no fit can run.  What CAN be
settled now is whether the base vector's ids line up with the block this lane builds itself:
one-to-one, nothing missing, every derived attribute present -- and, as a by-product, the
TRUE writable row count that section 47.23.4 caveat 3 pre-registered a re-evaluation on.

It also PROVES the refactor that made `onemonth.nov_rows` take a month: 2025-11 is rebuilt
through the new default path and must still give exactly 99,709 writable rows, the number
every fit since section 47.18 carried, and `validate_derivation` must still reproduce
july_loo exactly.  A refactor that quietly moved the 2025-11 path would invalidate v9.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/octcheck.py
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
import chainfold as CF                                                # noqa: E402
import rung511 as R                                                   # noqa: E402
import onemonth as OM                                                 # noqa: E402
import prc.data as D                                                  # noqa: E402

OCT = "2025-10"
OUT = EXP / "analysis/octcheck.json"


def main() -> None:
    global OCT, OUT
    if len(sys.argv) > 1:                       # the same pre-check, on any registered month
        OCT = sys.argv[1]
        OUT = EXP / f"analysis/idcheck-{OCT}.json"
        if OCT not in OM.MONTHS:
            raise SystemExit(f"{OCT} is not a registered extension month; "
                             f"known: {sorted(OM.MONTHS)}")
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

    # --- the refactor is proved before it is used ---------------------------
    OM.validate_derivation()
    nov = OM.nov_rows(scols)
    n_nov = int(nov.usable.sum())
    if n_nov != 99709:
        raise SystemExit(f"the refactored path gives {n_nov} writable 2025-11 rows, not the "
                         f"99,709 every fit since section 47.18 carried -- v9's head was "
                         f"fitted on the old path and this would mean they differ")
    print(f"refactor proved: 2025-11 still gives {n_nov:,} writable rows, "
          f"derivations still exact", flush=True)

    # --- the pre-check, with NO neighbour block ------------------------------
    d = OM.nov_rows(scols, month=OCT, with_neighbour=False)
    b6 = pd.read_parquet(OM.OOF / f"base6--{OCT}.parquet")
    mv = D.load_month(OCT)
    surf = CF.restrict(CF.build_month(OCT, A.prefix_months_for(OCT), rule),
                       R.THR, CF.GAP_CUT_S)

    ids6 = b6["MVT_ID_mvt"].astype("float64").to_numpy()
    idsm = mv["MVT_ID_mvt"].astype("float64").to_numpy()
    idss = surf["MVT_ID_mvt"].astype("float64").to_numpy()
    rec = {
        # ROUND 12, N4: the docstring names 2025-10, the month this file was written
        # for, so every later month recorded a "what" that named the WRONG month while
        # carrying the right values. The month is a parameter; the sentence must be too.
        "what": (f"An ID PRE-CHECK on queue's base6 {OCT}, against this lane's own "
                 f"{OCT} surface block."),
        "month": OCT,
        "base6": {"file": str((OM.OOF / f"base6--{OCT}.parquet").relative_to(ROOT)),
                  "sha16": OM.MONTHS[OCT]["base6_sha16"],
                  "rows": int(len(b6)), "unique_ids": int(len(np.unique(ids6))),
                  "ids_are_unique": bool(len(np.unique(ids6)) == len(ids6))},
        "movement_table": {"rows": int(len(mv)),
                           "base6_ids_absent_from_it": int((~np.isin(ids6, idsm)).sum())},
        "surface_block": {"rows": int(len(surf)),
                          "ids_are_unique": bool(len(np.unique(idss)) == len(idss)),
                          "base6_ids_absent_from_it": int((~np.isin(ids6, idss)).sum()),
                          "block_ids_absent_from_base6": int((~np.isin(idss, ids6)).sum())},
        "derived": {
            "rows": int(len(d)),
            "airport_missing": int(d.airport.isna().sum() + (d.airport.astype(str) == "nan").sum()),
            "time_missing": int(pd.isna(d["t"]).sum()),
            "target_not_finite": int((~np.isfinite(d.rbase.to_numpy("float64"))).sum()),
            "noac_rows": int(d.noac.sum()),
            "matched": int(d.obs_matched.sum()),
            "with_clock": int(d.obs_has_clock.sum()),
            "days": int(d.day.nunique()),
            "airports": sorted(set(d.loc[d.usable.to_numpy(bool), "airport"].astype(str))),
        },
        "WRITABLE_ROWS": int(d.usable.sum()),
        "writable_by_airport": {str(a): int(n) for a, n in
                                d.loc[d.usable.to_numpy(bool), "airport"].astype(str)
                                .value_counts().sort_index().items()},
        "neighbour_block": "NOT read -- queue is re-emitting it; no fit may use this frame. The SHIPPED block IS joined and checked here, because it is a separate dependency",
        "shipped_columns_joined": int(sum(c in d.columns for c in scols)),
        "shipped_columns_expected": len(scols),
        "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    print(f"\nbase6 {OCT}: {rec['base6']['rows']:,} rows, unique ids "
          f"{rec['base6']['ids_are_unique']}")
    print(f"  absent from the movement table : {rec['movement_table']['base6_ids_absent_from_it']}")
    print(f"  absent from the surface block  : {rec['surface_block']['base6_ids_absent_from_it']}")
    print(f"  surface block rows not in base6: {rec['surface_block']['block_ids_absent_from_base6']}")
    print(f"  airport/time/target missing    : {rec['derived']['airport_missing']} / "
          f"{rec['derived']['time_missing']} / {rec['derived']['target_not_finite']}")
    print(f"  shipped columns joined         : {rec['shipped_columns_joined']}/{rec['shipped_columns_expected']}")
    print(f"\n  WRITABLE ROWS: {rec['WRITABLE_ROWS']:,}   over {rec['derived']['days']} days, "
          f"{len(rec['derived']['airports'])} airports")
    print("  by airport:", ", ".join(f"{k} {v:,}" for k, v in rec["writable_by_airport"].items()))
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
