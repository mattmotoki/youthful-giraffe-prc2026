"""Which input differs, for the rows where the two lanes' class boundary disagrees.

The desk, 2026-09-08 19:05: `exp-047-queue` lists the ranking rows where its 2026 blocks
and this lane's served dry run disagree about the reach class, at one and the same
`block.py` (`ba41935e`).  This reads that list against THIS lane's own ranking block and
says, per row, which quantity differs: the match itself, the clock, `obs_taxi`, the
`noac` hold, or only the reading of `reach_class` into a served class.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parent.parent

# `serve` FIRST: `arms` pulls in `imports.load`, which puts other experiments' code
# directories at the front of sys.path, and one of them also has a `serve.py`
import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402

LIST = ROOT / "experiments/exp-047-queue/analysis/boundary-diff.json"


def main() -> None:
    want = json.loads(LIST.read_text())
    rows = pd.DataFrame(want["rows"])
    dep, _ = S.ranking_block()
    cols = [c for c in ("MVT_ID_mvt", "obs_matched", "obs_has_clock", "obs_taxi",
                        "match_mode", "obs_ambiguous", "noac", "usable", "airport",
                        "month", "obs_n_candidates") if c in dep.columns]
    mine = dep[cols].copy()
    j = rows.merge(mine, on="MVT_ID_mvt", how="left", suffixes=("", "_mine"))
    j["mine_class"] = np.where(
        j.get("noac", False).astype(bool), "noac_hold",
        np.where(~j.obs_matched.astype(bool), "unmatched",
                 np.where(~j.obs_has_clock.astype(bool), "matched_no_clock",
                          np.where(j.obs_taxi.isna(), "matched_no_obs_taxi", "with_clock"))))
    j["mechanism"] = np.where(
        j.queue_obs_matched.astype(bool) != j.obs_matched.astype(bool),
        "the MATCH itself differs",
        np.where(j.queue_obs_has_clock.astype(bool) != j.obs_has_clock.astype(bool),
                 "the CLOCK differs on a row both lanes matched",
                 "the match and the clock AGREE; only the reading into a served class differs"))
    out = {
        "read": str(LIST.relative_to(ROOT)),
        "rows": int(len(j)),
        "by_mechanism": j.mechanism.value_counts().to_dict(),
        "by_mechanism_and_disagreement": (
            j.groupby(["disagreement", "mechanism"]).size().unstack(fill_value=0)
             .to_dict()),
        "by_queue_class_and_mechanism": (
            j.groupby(["queue_reach_class", "mechanism"]).size().unstack(fill_value=0)
             .to_dict()),
        "detail": j[[c for c in ("MVT_ID_mvt", "airport", "month", "day", "disagreement",
                                 "queue_reach_class", "queue_obs_matched",
                                 "queue_obs_has_clock", "obs_matched", "obs_has_clock",
                                 "obs_taxi", "match_mode", "usable", "mine_class",
                                 "mechanism") if c in j.columns]]
             .to_dict(orient="records"),
    }
    (EXP / "analysis/boundary-diff-read.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, indent=1, default=str))


if __name__ == "__main__":
    main()
