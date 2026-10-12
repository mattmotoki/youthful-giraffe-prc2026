"""Clean board estimates for the cost subsets in `docs/QUEUE-2026-09-07.md` section 1.

The rebuild table asks which members a 12-hour Kaggle session can rebuild.  That
question is only useful next to the score each subset would serve, and Sol's
round-1 review (blocking finding 5, `docs/reviews/queue-round1.md`) found the
document gave the costs without the scores.  This computes them.

It reads nothing but the committed member out-of-fold parquets: no raw data, no
`artifacts/`, no fit.  The rotation, the per-regime NNLS, the residual guard and
the 60 s floor all come from exp-026's modules, so these numbers sit on the same
instrument as `analysis/ladder_*.csv` and are directly comparable with them.

    .venv/bin/python scripts/queue_subsets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "exp-026-rebuild" / "code"))
import frames as F  # noqa: E402
import ladder as L  # noqa: E402
import rule as R  # noqa: E402

#: The four subsets section 1 prices, cheapest first.  Names are blend-5's.
SUBSETS = [
    ("F1+V3+T+N", ("F1-fresh", "V3-allin", "T-tailv4", "N1-nmtail")),
    ("+C2-cat", ("F1-fresh", "V3-allin", "T-tailv4", "N1-nmtail", "C2-cat")),
    ("+L2-hub8725", ("F1-fresh", "V3-allin", "T-tailv4", "N1-nmtail", "C2-cat",
                     "L2-hub8725")),
    ("all seven (= blend-5)", L.B5),
]


def main() -> None:
    cache: dict = {}
    rows = []
    for label, names in SUBSETS:
        r = L.run(names, cache)
        rec = {"subset": label, "n_members": len(names), "servable": L.servable(names)}
        for f in F.FOLDS:
            rec[f] = round(r[f]["rmse_clean"], 3)
        rec["clean board"] = round(r["board_clean"], 3)
        # exp-026's own `predict` is the desk's +4.0 convention; the queue
        # document reads every prediction at the measured +4.2762 offset, so both
        # are carried and the document quotes the second.
        rec["predict (+4.0)"] = round(r["predict"], 2)
        rec["predict (+4.2762)"] = round(r["board_clean"] + 4.2762, 2)
        rows.append(rec)
    T = pd.DataFrame(rows)
    base = T.loc[T.subset == "all seven (= blend-5)", "clean board"].iloc[0]
    T["vs blend-5"] = (T["clean board"] - base).round(3)
    print(T.to_string(index=False))
    out = ROOT / "docs" / "queue-subsets-2026-09-07.csv"
    T.to_csv(out, index=False)
    print("\nwritten:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
