"""The board blend recomputed on the desk's (july_loo, january) analogue pair.

`prc.data` VERSION 2 adds `july_loo`: the other eleven months against July 2025.
The submission model trains on all twelve, so July 2026 is predicted with July
and August 2025 in training; the forward `july` fold has no same-season data and
is pessimistic, exactly as `january` is the analogue for January 2026. So
`(july_loo, january)` is the fold pair that matches the ranking situation, and it
is the pair this file blends.

Every number comes from ONE line's out-of-fold files, so the folds share a model
configuration and a set of `prc/` module versions. The two unreachable
date-rollover rows (`analysis/q1_rollover.md`) are removed, because they are a
lottery every team draws from rather than a difference in skill; the floor they
put on a board score is finding 3's separate arithmetic.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import prc.data as D  # noqa: E402
import common as C  # noqa: E402

# ranking-file composition, from docs/shift-2026.md and exp-006
N_JAN, N_JUL = 152_719, 192_122
LIRF_JAN, LIRF_JUL = 11_061, 15_838
NINE_JAN, NINE_JUL = N_JAN - LIRF_JAN, N_JUL - LIRF_JUL
W_NINE_JUL, W_NINE_JAN = NINE_JUL / (NINE_JUL + NINE_JAN), NINE_JAN / (NINE_JUL + NINE_JAN)
W_LIRF_JUL, W_LIRF_JAN = LIRF_JUL / (LIRF_JUL + LIRF_JAN), LIRF_JAN / (LIRF_JUL + LIRF_JAN)
LIRF_SHARE = (LIRF_JUL + LIRF_JAN) / (N_JAN + N_JUL)

ROLL_IDS: set = set()


def _rollover_ids(fold: str) -> set:
    d = pd.read_parquet(HERE.parent / "work" / f"rows_{fold}.parquet")
    import q1c_trimmed as q1c
    return set(d.loc[q1c.rollover_mask(d), D.ID])


def numbers(dirname: str, line: str, fold: str) -> dict:
    p = HERE.parents[1] / dirname / "oof" / f"{line}-{fold}.parquet"
    d = pd.read_parquet(p)
    d = d[d["y_true"].notna() & ~d[D.ID].isin(ROLL_IDS)]
    e = d["y_true"] - d["y_pred"]
    nl = d["airport"] != "LIRF"
    return {"nine": C.rmse(e[nl]), "lirf": C.rmse(e[~nl]), "n": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="exp-007-winter/oof", help="unused, kept for clarity")
    ap.add_argument("--lines", default="exp-007-winter:d1-base-s7,exp-007-winter:d1-ds-s7",
                    help="comma-separated <experiment dir>:<line name>")
    a = ap.parse_args()
    global ROLL_IDS
    ROLL_IDS = _rollover_ids("january") | _rollover_ids("july")

    out = ["# exp-007 — the board blend on the (july_loo, january) analogue pair\n",
           __doc__.strip().split("\n\n", 1)[1], "",
           f"Ranking composition: the other nine carry {NINE_JUL:,} July and {NINE_JAN:,} January",
           f"departures ({100*W_NINE_JUL:.1f} % / {100*W_NINE_JAN:.1f} %); LIRF carries {LIRF_JUL:,}",
           f"and {LIRF_JAN:,} ({100*W_LIRF_JUL:.1f} % / {100*W_LIRF_JAN:.1f} %) and is",
           f"{100*LIRF_SHARE:.2f} % of the file.\n",
           "| line | July fold used | other nine | LIRF | implied total |", "|---|---|---|---|---|"]
    for spec in a.lines.split(","):
        dirname, line = spec.split(":")
        try:
            jan = numbers(dirname, line, "january")
        except FileNotFoundError:
            continue
        for jul_fold in ("july", "july_loo"):
            try:
                jul = numbers(dirname, line, jul_fold)
            except FileNotFoundError:
                continue
            nine = np.sqrt(W_NINE_JUL * jul["nine"] ** 2 + W_NINE_JAN * jan["nine"] ** 2)
            lirf = np.sqrt(W_LIRF_JUL * jul["lirf"] ** 2 + W_LIRF_JAN * jan["lirf"] ** 2)
            tot = np.sqrt((1 - LIRF_SHARE) * nine ** 2 + LIRF_SHARE * lirf ** 2)
            mark = " **(the analogue pair)**" if jul_fold == "july_loo" else ""
            out.append(f"| `{line}` | `{jul_fold}`{mark} | **{nine:.1f}** | {lirf:.1f} | {tot:.1f} |")
    out.append("")
    out.append("The leaders' band, from exp-006's arithmetic at a board top of 277.3 s, is 200 to "
               "240 s on the other nine against 550 to 720 s at LIRF.\n")
    p = HERE.parent / "analysis" / "q6_blend.md"
    p.write_text("\n".join(out))
    print(p.read_text())


if __name__ == "__main__":
    main()
