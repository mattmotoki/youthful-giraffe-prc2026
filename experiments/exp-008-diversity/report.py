"""Regenerate the per-learner four-fold table from result-*.json as markdown.

Tables in RESULTS.md are produced by this, not typed, so a number in the write-up
cannot drift from the number in the artifact it came from.

  python report.py                 # every arm
  python report.py L0-lin C1-d8    # a subset, in the order given
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
#: Ranking-month shares of the 344,841 scored departures (exp-001 board.py).
SHARES = {"2026-07": 0.5571321275602379, "2026-01": 0.4428678724397621}
FOLDS = ("july", "july_loo", "december", "january")
COLS = [("rmse", "rmse"), ("rmse_bulk", "≤3600"), ("rmse_tail", ">3600"),
        ("rmse_no_nm", "no-NM"), ("rmse_noac", "noac")]


def rows(names):
    out = []
    for n in names:
        p = HERE / f"result-{n}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        for f in FOLDS:
            r = rec["folds"].get(f)
            if r:
                out.append({"arm": n, "fold": f, "rounds": r["rounds"],
                            **{k: r[k] for k, _ in COLS}})
    return pd.DataFrame(out)


def markdown(d: pd.DataFrame) -> str:
    head = "| arm | fold | rounds | " + " | ".join(lbl for _, lbl in COLS) + " |"
    sep = "|" + "---|" * (3 + len(COLS))
    lines = [head, sep]
    for _, r in d.iterrows():
        lines.append(f"| `{r.arm}` | {r.fold} | {int(r.rounds)} | "
                     + " | ".join(f"{r[k]:.1f}" for k, _ in COLS) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem[len("result-"):] for p in HERE.glob("result-*.json"))
    d = rows(names)
    if d.empty:
        raise SystemExit("no result-*.json yet")
    print(markdown(d))
    print("\n\n### overall RMSE, arm by fold\n")
    piv = d.pivot(index="arm", columns="fold", values="rmse").reindex(columns=list(FOLDS))
    piv = piv.reindex([n for n in names if n in piv.index])
    print("| arm | " + " | ".join(FOLDS) + " | board est. |")
    print("|" + "---|" * (len(FOLDS) + 2))
    for arm, r in piv.iterrows():
        # Board estimate, same convention as exp-001's board.py: the two
        # ranking-analogue folds combined in QUADRATURE (RMSE does not average
        # linearly), weighted by each ranking month's share of departures.
        be = (float(np.sqrt(SHARES["2026-07"] * r["july_loo"] ** 2
                            + SHARES["2026-01"] * r["january"] ** 2))
              if pd.notna(r.get("july_loo")) and pd.notna(r.get("january")) else float("nan"))
        cells = " | ".join("–" if pd.isna(v) else f"{v:.1f}" for v in r)
        print(f"| `{arm}` | {cells} | " + ("–" if pd.isna(be) else f"{be:.1f}") + " |")
