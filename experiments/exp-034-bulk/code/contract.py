"""The frame contract: what the 64 columns are, so a later drift is visible.

This lane has already had one silent frame drift (`SOURCES_V3F` appending
`fresh` mid-programme gave `V3-wins3600` four extra columns and stopped it
being comparable), which is why exp-001 grew `frame_contract.py`.  The bulk
frame has no such file, so this is it: the 64 names in order, their SHA-256,
the per-column null share and mean on each fold's validation month, and the
SHA-256 of every rebuilt month frame.

    python contract.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import member as M  # noqa: E402
import prc.data as D  # noqa: E402
import prc.stand_features as sf  # noqa: E402


def main() -> None:
    cols = list(M.FEATURES)
    rec = {"name": M.NAME, "n_columns": len(cols), "columns": cols,
           "column_hash": hashlib.sha256("\n".join(cols).encode()).hexdigest(),
           "families": {"categorical": M.Q.CAT, "own_flight": M.Q.BASE_NUM,
                        "occupancy": M.Q.OCC, "airport_neighbour": M.Q.NBR},
           "stand_features_version": sf.VERSION,
           "params": {k: v for k, v in M.PARAMS.items()},
           "rounds": M.ROUNDS, "seed": M.SEED,
           "builder": "experiments/exp-003-anatomy/code/features.py (unchanged)",
           "config_source": "experiments/exp-011-bulkfloor/code/q1c_seeds.py",
           "months": {}, "per_fold_columns": {}}
    for m in list(D.MONTHS) + ["ranking"]:
        p = M.ART / f"{m}.parquet"
        rec["months"][m] = {"sha256": D.sha256(p), "bytes": p.stat().st_size}
    for f in M.FOLDS + ("ranking",):
        month = "ranking" if f == "ranking" else D.FOLDS[f].valid_month
        d = pd.read_parquet(M.ART / f"{month}.parquet", columns=cols)
        rec["per_fold_columns"][f] = {
            c: {"null": round(float(d[c].isna().mean()), 6),
                "mean": (round(float(np.asarray(d[c], "float64").mean()), 4)
                         if c not in M.Q.CAT else None),
                "levels": (int(d[c].nunique()) if c in M.Q.CAT else None)}
            for c in cols}
        rec["per_fold_columns"][f]["_rows"] = int(len(d))
    (EXP / "analysis" / "frame-contract.json").write_text(json.dumps(rec, indent=2))
    print(f"[contract] {len(cols)} columns, hash {rec['column_hash'][:12]}…, "
          f"{len(rec['months'])} month frames", flush=True)
    for f in rec["per_fold_columns"]:
        n = rec["per_fold_columns"][f]["_rows"]
        worst = sorted(((v["null"], c) for c, v in rec["per_fold_columns"][f].items()
                        if isinstance(v, dict)), reverse=True)[:3]
        print(f"  {f:9s} {n:7d} rows; most-null columns " +
              ", ".join(f"{c} {s:.3f}" for s, c in worst), flush=True)


if __name__ == "__main__":
    main()
