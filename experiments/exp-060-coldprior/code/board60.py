"""exp-060-coldprior: the board-recombined COMMON.md table (generated from `analysis/oos.json`).

Each COMMON.md column is recombined from its two ranking folds at the fixed ranking shares,
`sqrt(sum_r share_r * RMSE_r^2)`, which is exp-056's convention (`analysis/common-tables-r1.md`).
dec has no board weight and is reported beside it, never inside it.  Writes
`analysis/board-tables.md`.

    .venv/bin/python experiments/exp-060-coldprior/code/board60.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e60 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

COLS = (("rmse", "RMSE"), ("rmse_clean", "clean"), ("rmse_le3600", "≤ 3,600"),
        ("rmse_gt3600", "> 3,600"), ("rmse_nonm", "no-NM"))
ARMS = ("base", "zero", "true", "const", "gmean", "donor", "shuffle")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = E.ANALYSIS / "board-tables.md"
    NO.refuse_existing([out], a.overwrite)
    oos = E.load_json(E.ANALYSIS / "oos.json")
    L = ["# exp-060-coldprior — the board-recombined COMMON.md table (`code/board60.py`)\n",
         "\nBoard = jul and jan at the ranking shares "
         f"{oos['board_shares']['jul']:.4f} / {oos['board_shares']['jan']:.4f}; each column is "
         "recombined from its two fold RMSEs. dec carries no board weight.\n"]
    for lev in ("B", "A"):
        L.append(f"\n**Level {lev}** (level A is descriptive)\n")
        L.append("| arm | " + " | ".join(c[1] for c in COLS) + " | dec RMSE |")
        L.append("|---" * (len(COLS) + 2) + "|")
        for arm in ARMS:
            key = "base" if arm == "base" else f"{arm}|{lev}"
            vals = []
            for c, _ in COLS:
                v = sum(oos["board_shares"][r] * oos["tables"][r][key][c] ** 2
                        for r in ("jul", "jan"))
                vals.append(f"{np.sqrt(v):.4f}")
            L.append(f"| {arm} | " + " | ".join(vals) + " | "
                     f"{oos['tables']['dec'][key]['rmse']:.4f} |")
    out.write_text("\n".join(L) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
