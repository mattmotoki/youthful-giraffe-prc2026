"""Gate: does the rule on committed OOF reproduce the published rotations?

Nothing downstream is believed until both lines below are ~0.  blend-4 and
blend-5 each published three rotated fold numbers in their config; the rule
here is re-fitted from the committed out-of-fold files and must land on them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
import rule as R  # noqa: E402

SETS = {
    "blend-4": (("T-tailv2", "C2-cat", "L2-hub", "D1-winter", "F1-fresh", "N1-nmtail"),
                {"july_loo": 289.3189, "december": 207.7722, "january": 340.8188}),
    "blend-5": (("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh", "V3-allin",
                 "N1-nmtail"),
                {"july_loo": 286.3274, "december": 207.0083, "january": 339.0173}),
}


def main() -> None:
    out = {}
    for label, (names, published) in SETS.items():
        data = {f: F.fold_frame(f, names) for f in F.FOLDS}
        res = R.rotate(data, names)
        row = {}
        for f in F.FOLDS:
            got = res[f]["rmse"]
            row[f] = dict(mine=round(got, 4), published=published[f],
                          delta=round(got - published[f], 4),
                          clean=round(res[f]["rmse_clean"], 4),
                          n_excluded=res[f]["n_excluded"])
        row["board_all"] = round(res["board_all"], 4)
        row["board_clean"] = round(res["board_clean"], 4)
        row["predict"] = round(res["predict"], 4)
        out[label] = row
        print(f"=== {label} ===")
        for f in F.FOLDS:
            d = row[f]
            print(f"  {f:9s} {d['mine']:9.4f}  (published {d['published']:.4f}, "
                  f"delta {d['delta']:+.4f})   clean {d['clean']:9.4f} "
                  f"[{d['n_excluded']} excluded]")
        print(f"  board all {row['board_all']:.4f}   clean {row['board_clean']:.4f}   "
              f"clean+4 {row['predict']:.4f}")
    (F.EXP / "analysis" / "reproduce.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
