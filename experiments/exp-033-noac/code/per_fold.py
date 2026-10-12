"""The per-band correction, fold by fold. A gain that lives in one fold is not
a direction, whatever the pooled interval says."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from describe import LAB  # noqa: E402
from direction import rmse  # noqa: E402


def main() -> None:
    v = pd.read_parquet(EXP / "work" / "fold_arms.parquet")
    v["corr"] = v.delta - v.y_pred
    rows = []
    for scope in ("matched", "all"):
        s0 = v[v._m] if scope == "matched" else v
        for lab in LAB:
            r = {"scope": scope, "band": lab}
            for fold in ("july_loo", "december", "january"):
                s = s0[s0.fold == fold]
                arm = s.y_pred + np.where(s.band == lab, s["corr"], 0.0)
                r[fold] = rmse(s.y_true, s.y_pred) - rmse(s.y_true, arm)
                r[f"n_{fold}"] = int((s.band == lab).sum())
            r["signs"] = "".join("+" if r[f] > 0 else "-"
                                 for f in ("july_loo", "december", "january"))
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "per_fold_bands.csv", index=False)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
