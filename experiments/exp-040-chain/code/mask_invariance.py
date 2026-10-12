"""The leakage check the size of this gain demands: the chain columns cannot see a label.

    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-040-chain/code/mask_invariance.py

`experiments/exp-008-diversity/frames.py::verify_mask_invariance` makes this
check for the v2 columns: the masked validation frame must produce the same
feature values as the unmasked training frame, and only then is one cached build
per month legitimate.  The chain columns are new, so they get the same check,
column by column, on a summer month and a winter month.

If any chain column differed between the two builds, it would be reading a
departure's own `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt` -- the two fields the
ranking file blanks -- and every number in this experiment would be a leak.

Writes `analysis/mask-invariance.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import prc.data as D  # noqa: E402
from chain_cols import ARMS, columns  # noqa: E402

COLS = list(ARMS["chain"])


def main() -> None:
    out = {}
    for m in ("2025-07", "2025-12"):
        full = D.load_month(m)
        masked, _ = D.mask_like_ranking(full)
        # The `chain` arm's three columns are `prc/tail_features.py`'s, and they
        # are built here from BOTH frames.  The `derived` arm's five come from
        # exp-037's `stand_chain`, which REFUSES a frame carrying departure
        # labels, so its guarantee is structural and is demonstrated instead.
        a = columns(full, "chain", m)        # the UNMASKED month
        b = columns(masked, "chain", m)      # the ranking-masked month
        try:
            columns(full, "derived", m)
            refused = False
        except ValueError:
            refused = True
        rec = {"n_rows": int(len(a)), "same_index": bool(a.index.equals(b.index)),
               "stand_chain_refuses_an_unmasked_frame": refused}
        for c in COLS:
            u = a[c].to_numpy("float64")
            v = b[c].to_numpy("float64")
            bad = ~((u == v) | (np.isnan(u) & np.isnan(v)))
            rec[c] = {"n_differ": int(bad.sum()),
                      "max_abs_diff": float(np.nanmax(np.abs(u[bad] - v[bad]))) if bad.any() else 0.0}
        # and the departure fields that must be gone from the masked frame
        dep = masked["PHASE_mvt"] == "DEP"
        rec["masked_dep_block_time_present"] = int(masked.loc[dep, "BLOCK_TIME_UTC_mvt"].notna().sum())
        rec["masked_dep_taxitime_present"] = int(masked.loc[dep, "TAXITIME_SEC_mvt"].notna().sum())
        rec["unmasked_dep_block_time_present"] = int(full.loc[dep, "BLOCK_TIME_UTC_mvt"].notna().sum())
        out[m] = rec
        print(m, json.dumps({k: v for k, v in rec.items() if k != "same_index"}), flush=True)
    (EXP / "analysis" / "mask-invariance.json").write_text(json.dumps(out, indent=2))
    worst = max(out[m][c]["n_differ"] for m in out for c in COLS)
    print("\nrows differing between the masked and unmasked build, worst column:", worst)
    if worst:
        raise SystemExit("a chain column changes under the ranking mask: it reads a label")
    if not all(out[m]["stand_chain_refuses_an_unmasked_frame"] for m in out):
        raise SystemExit("stand_chain accepted a frame carrying departure labels")
    print("MASK INVARIANCE OK: the three chain columns are identical built from the "
          "masked and the unmasked month, and stand_chain refuses a labelled frame")


if __name__ == "__main__":
    main()
