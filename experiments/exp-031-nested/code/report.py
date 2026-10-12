"""`analysis/rotation.json` as the tables the assignment asks for.

Three tables and one decomposition:

  1. candidate x fold, clean RMSE, one block per column of the instrument
  2. the chain, gain over predecessor, per fold and on the board, with the
     day-clustered 95 % intervals exp-028 uses
  3. `K2 - refit-bulk`, the ordering question, on every column
  4. each gain split into its machine term (`published -> kaggle`) and its
     nesting term (`kaggle -> nested`)

    python report.py > ../analysis/tables.md
    python report.py --anchor nested > ../analysis/tables-nestedanchor.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
FOLDS = ("july_loo", "december", "january")
ARMS = ("stack-5", "v6", "v7", "K2", "refit-bulk")
PAIRS = ("v6 - stack-5", "v7 - v6", "K2 - v7", "refit-bulk - v7", "K2 - refit-bulk")
COLS = ("published", "kaggle", "nested", "nested-f1held")
LABEL = {"published": "published", "kaggle": "kaggle (unnested)",
         "nested": "nested (3 members)", "nested-f1held": "nested (2 members)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", choices=("rotated", "nested"), default="rotated")
    a = ap.parse_args()
    tag = "" if a.anchor == "rotated" else "-nestedanchor"
    d = json.loads((EXP / "analysis" / f"rotation{tag}.json").read_text())
    print(f"<!-- anchor: {d['anchor']} — {d['anchor_note']} -->\n")
    # Sol round 1, finding 5: every difference below is taken from the UNROUNDED
    # fold RMSE and board that `rotate.py` stores beside the rounded ones.
    cl, bd, gn = d["clean"], d["board_exact"], d["gain"]
    cols = [c for c in COLS if c in cl]

    print("## 1. Candidate x fold, clean RMSE\n")
    print("| candidate | column | july_loo | december | january | board |")
    print("|---|---|---:|---:|---:|---:|")
    for a in ARMS:
        for c in cols:
            print(f"| {a} | {LABEL[c]} | " + " | ".join(f"{cl[c][a][f]:.4f}" for f in FOLDS)
                  + f" | {bd[c][a]:.4f} |")

    print("\n## 2. The chain: each candidate over its predecessor "
          "(positive = the new candidate is better)\n")
    for p in PAIRS:
        print(f"\n**{p}**\n")
        print("| column | july_loo | december | january | board | "
              "july_loo 95 % | january 95 % | board 95 % |")
        print("|---|---:|---:|---:|---:|---|---|---|")
        for c in cols:
            g = gn[c][p]
            ci = lambda f: f"[{g['ci'][f][1]:+.3f}, {g['ci'][f][2]:+.3f}]"
            b = g["board_ci"]
            print(f"| {LABEL[c]} | "
                  + " | ".join(f"{g['per_fold'][f]:+.4f}" for f in FOLDS)
                  + f" | {g['board']:+.4f} | {ci('july_loo')} | {ci('january')} | "
                    f"[{b[1]:+.3f}, {b[2]:+.3f}] |")

    print("\n## 3. The two terms of the change of instrument, on the board\n")
    print("| quantity | published | kaggle | nested (3) | machine "
          "(kag - pub) | nesting (nest - kag) |")
    print("|---|---:|---:|---:|---:|---:|")
    for a in ARMS:
        r = [bd[c][a] for c in ("published", "kaggle", "nested")]
        print(f"| {a} board | " + " | ".join(f"{x:.4f}" for x in r)
              + f" | {r[1]-r[0]:+.4f} | {r[2]-r[1]:+.4f} |")
    for p in PAIRS:
        new, old = p.split(" - ")
        r = [bd[c][old] - bd[c][new] for c in ("published", "kaggle", "nested")]
        print(f"| gain {p} | " + " | ".join(f"{x:+.4f}" for x in r)
              + f" | {r[1]-r[0]:+.4f} | {r[2]-r[1]:+.4f} |")

    print("\n## 4. Sign check: does any shipped gain reverse?\n")
    print("| gain | published | nested (3) | nested (2) | sign holds |")
    print("|---|---:|---:|---:|---|")
    for p in PAIRS:
        new, old = p.split(" - ")
        v = {c: bd[c][old] - bd[c][new] for c in cols}
        same = all((v[c] > 0) == (v["published"] > 0) for c in cols)
        print(f"| {p} | {v['published']:+.4f} | {v['nested']:+.4f} | "
              f"{v['nested-f1held']:+.4f} | {'yes' if same else '**NO**'} |")


if __name__ == "__main__":
    main()
