"""Step 2's premise, checked on the population Q09 is scored on.

    OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-040-chain/code/equality.py

exp-037 §0 measured `g_any == gap_arr1` and `g_prev2 == gap_arr2` at 0 s on the
1,485 chained 2025 noac rows and the 386 chained ranking rows.  The assignment
uses that equality to say the chain arm and the shipped `gap_arr1..3` are the
same instrument, and Q09 is scored on the ORDINARY rows, so it is checked here on
every departure of every month, training and ranking.

Writes `analysis/equality.csv`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import prc.data as D  # noqa: E402
from chain_cols import equality  # noqa: E402


def main() -> None:
    rows = []
    for m in D.MONTHS:
        t0 = time.time()
        masked, _ = D.mask_like_ranking(D.load_month(m))
        rec = {"month": m, **equality(masked), "seconds": round(time.time() - t0, 1)}
        rows.append(rec)
        print(rec, flush=True)
    rank = D.load_ranking()
    for m, g in rank.groupby("month", sort=True):
        t0 = time.time()
        rec = {"month": str(m), **equality(g), "seconds": round(time.time() - t0, 1)}
        rows.append(rec)
        print(rec, flush=True)
    out = pd.DataFrame(rows)
    (EXP / "analysis").mkdir(parents=True, exist_ok=True)
    out.to_csv(EXP / "analysis" / "equality.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
