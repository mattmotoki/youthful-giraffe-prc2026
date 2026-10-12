"""The same null applied to step 2's label difference.

If a random query set of the same band mix also drives P(placeholder) down by
0.06 in the 1-2 h band, then the label difference is a property of taking a
k-nearest-neighbour SUBSET of a band -- which is a systematic operation, not a
random one -- and not a property of the 388.  Step 3's arm gets this test in
`placebo2.py`; step 2's estimand needs it for the same reason.

Same construction as `placebo2.py`, including its day-block query sampling: 31
winter and 31 summer 2025 days, the noac rows on them, capped per band at the
388's counts; a fake query may not take a neighbour from its own calendar day;
the estimand is the primary one (distinct rows, unweighted).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction as DIR  # noqa: E402
from describe import LAB  # noqa: E402
from placebo2 import fake_queries, query_match  # noqa: E402

NREP = 200


def diffs(a: pd.DataFrame, mm: np.ndarray) -> dict:
    out = {}
    for lab in LAB:
        m = (a.band == lab).to_numpy()
        if not m.sum():
            continue
        sel = m & mm
        out[lab] = (float(a.ph.to_numpy()[sel].mean()) - float(a.ph.to_numpy()[m].mean())
                    if sel.sum() else np.nan)
    return out


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    d_aa = np.load(EXP / "work" / "d_aa_rowlevel.npy")
    real = diffs(a, DIR.matched_mask(a, b))
    counts = {lab: int((b.band == lab).sum()) for lab in LAB}
    rng = np.random.default_rng(707)
    reps = []
    for r in range(NREP):
        qi = fake_queries(a, counts, rng)
        d = diffs(a, query_match(a, d_aa, qi))
        reps.append({"rep": r, "n_queries": len(qi),
                     "n_query_days": int(len(np.unique(a.day.to_numpy()[qi]))), **d})
    rp = pd.DataFrame(reps)
    rp.to_csv(EXP / "analysis" / "placebo_step2_reps.csv", index=False)
    rows = []
    for lab in LAB:
        col = rp[lab].to_numpy()
        rows.append({"band": lab, "diff_real": real[lab], "placebo_mean": col.mean(),
                     "placebo_sd": col.std(),
                     "placebo_p05": float(np.percentile(col, 5)),
                     "percentile": float((col < real[lab]).mean()),
                     "add_one_p_two_sided": float(min(1.0, 2 * min(
                         (1 + (col <= real[lab]).sum()) / (1 + len(col)),
                         (1 + (col >= real[lab]).sum()) / (1 + len(col)))))})
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "placebo_step2.csv", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
