"""Step 1's second gate: the `none` arm's ranking member against the committed one.

    .venv/bin/python experiments/exp-040-chain/code/rank_gate.py

The assignment says to gate the rebuilt ranking member against the committed
`L2-hub8725-ranking.parquet` (SHA-256 `dd0e8710...`) and to "state the offset; do
not proceed on an unexplained one".  The offset will not be zero and the reason is
already measured: the committed file was fitted on the workstation and this one on
Kaggle, and exp-025 §10.6 puts this member's machine term at +0.096 / −0.003 /
+0.471 s of fold RMSE with a sign that changes.  What makes the offset EXPLAINED
rather than unexplained is that the ranking-side difference has to be the same
size as the fold-side difference between the same two machines, which is what this
file measures: the `none` arm's fold OOFs against the published fold OOFs, row for
row, beside the ranking column against the ranking column.

An arm's ranking column is also compared against `none`'s, which is the
machine-matched increment and the number the candidate's diagnostic quotes.

Writes `analysis/rank-gate.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
ARMS = ("none", "chain", "derived")
PUB_RANK = ROOT / "experiments/exp-012-blend2/published/L2-hub8725-ranking.parquet"
PUB_RANK_SHA = "dd0e8710000ab29b930ec72d8b058b00e6ed86ba3240cad01402c13813527882"
PUB_OOF = "experiments/exp-012-blend2/published/L2-hub8725-{fold}.parquet"
ARM_OOF = "experiments/exp-040-chain/oof/{arm}--{fold}.parquet"
ARM_RANK = "experiments/exp-040-chain/oof/{arm}--ranking.parquet"


def pair(a: pd.DataFrame, b: pd.DataFrame, key: str = "MVT_ID_mvt") -> np.ndarray:
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    if not np.array_equal(a[key].to_numpy("float64"), b[key].to_numpy("float64")):
        raise SystemExit("the two files are not over the same departures")
    return a["y_pred"].to_numpy("float64") - b["y_pred"].to_numpy("float64")


def stat(d: np.ndarray) -> dict:
    return {"n": int(len(d)), "moved": int((d != 0).sum()),
            "rms": float(np.sqrt((d ** 2).mean())), "max_abs": float(np.abs(d).max()),
            "mean": float(d.mean())}


def main() -> None:
    got = D.sha256(PUB_RANK)
    if got != PUB_RANK_SHA:
        raise SystemExit(f"the committed ranking member has changed: {got[:8]}")
    rec = {"committed": {"path": str(PUB_RANK.relative_to(ROOT)), "sha256": got},
           "machine_term_fold": {}, "machine_term_ranking": {}, "arm_vs_none": {}}

    for f in FOLDS:
        p = ROOT / ARM_OOF.replace("{arm}", "none").format(fold=f)
        if not p.exists():
            continue
        a = pd.read_parquet(p)
        b = pd.read_parquet(ROOT / PUB_OOF.format(fold=f))
        rec["machine_term_fold"][f] = stat(pair(a, b)) | {
            "rmse_kaggle": float(((a["y_true"] - a["y_pred"]) ** 2).mean() ** 0.5),
            "rmse_published": float(((b["y_true"] - b["y_pred"]) ** 2).mean() ** 0.5)}
        rec["machine_term_fold"][f]["rmse_offset"] = (
            rec["machine_term_fold"][f]["rmse_kaggle"]
            - rec["machine_term_fold"][f]["rmse_published"])

    pub = pd.read_parquet(PUB_RANK)
    cols = {}
    for arm in ARMS:
        p = ROOT / ARM_RANK.replace("{arm}", arm)
        if not p.exists():
            print(f"[rank_gate] {arm}: {p.name} is absent", flush=True)
            continue
        d = pd.read_parquet(p)
        cols[arm] = d
        rec["machine_term_ranking"][arm] = stat(pair(d, pub)) | {
            "sha256": D.sha256(p),
            "min": float(d["y_pred"].min()), "median": float(d["y_pred"].median()),
            "finite": bool(np.isfinite(d["y_pred"].to_numpy("float64")).all())}
    for arm in ARMS:
        if arm != "none" and arm in cols and "none" in cols:
            rec["arm_vs_none"][arm] = stat(pair(cols[arm], cols["none"]))

    #: the gate: the ranking-side shift between the two machines must be the same
    #: SIZE as the fold-side shift between the same two machines.  A ranking file
    #: that differed by much more would be a different fit, not the machine.
    if rec["machine_term_fold"] and "none" in rec["machine_term_ranking"]:
        fold_rms = [rec["machine_term_fold"][f]["rms"] for f in rec["machine_term_fold"]]
        r = rec["machine_term_ranking"]["none"]["rms"]
        rec["gate"] = {"ranking_rms": r, "fold_rms_range": [min(fold_rms), max(fold_rms)],
                       "ratio_to_fold_mean": r / float(np.mean(fold_rms)),
                       "explained": bool(0.5 <= r / float(np.mean(fold_rms)) <= 2.0)}
    (EXP / "analysis" / "rank-gate.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
