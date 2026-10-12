"""Clean day-clustered intervals for stack-5 against blend-5, all three folds.

Sol's round-1 review (blocking finding 3, `docs/reviews/queue-round1.md`) found
that `docs/QUEUE-2026-09-07.md` quoted clean point estimates beside intervals
that `experiments/exp-026-rebuild/RESULTS.md` labels "all rows", and called the
`rmse_clean` kill passed on that pair.  This computes the clean intervals.

It refits nothing: the stacker's out-of-fold predictions are committed
(`exp-026-rebuild/oof/stack5-{fold}.parquet`, `y_pred` is the served keep-noac
arm and `y_pred_anchor` is blend-5), and the bootstrap and the airport-day
clustering are exp-026's own functions.  The only change is that
`prc.data.excluded_ids(fold)` rows are dropped before the draw.

    .venv/bin/python scripts/queue_clean_interval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "exp-026-rebuild" / "code"))
import frames as F  # noqa: E402
import stack5 as S  # noqa: E402

import prc.data as D  # noqa: E402


def main() -> None:
    rows = []
    for fold in F.FOLDS:
        oof = pd.read_parquet(F.EXP / "oof" / f"stack5-{fold}.parquet")
        meta = F.fold_frame(fold, ("F1-fresh",))[["MVT_ID_mvt", "airport", "t"]]
        m = oof.merge(meta, on="MVT_ID_mvt", how="left", validate="1:1")
        drop = set(D.excluded_ids(fold))
        keep = ~m["MVT_ID_mvt"].isin(drop)
        # reset_index matters: exp-026's `clusters()` builds its key by adding two
        # Series, and a filtered frame keeps its old index, so the airport half
        # and the day half would align on different labels and produce NaN keys.
        for label, sub in (("all rows", m), ("clean", m[keep].reset_index(drop=True))):
            cl = S.clusters(sub)
            d, lo, hi = S.boot_delta(sub["y_true"], sub["y_pred"], sub["y_pred_anchor"], cl)
            rows.append({"fold": fold, "population": label, "n": len(sub),
                         "n_excluded": int((~keep).sum()) if label == "clean" else 0,
                         "delta": round(float(d), 3),
                         "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                         "excludes zero": bool(hi < 0 or lo > 0)})
    T = pd.DataFrame(rows)
    print(T.to_string(index=False))
    out = ROOT / "docs" / "queue-clean-intervals-2026-09-07.csv"
    T.to_csv(out, index=False)
    print("\nwritten:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
