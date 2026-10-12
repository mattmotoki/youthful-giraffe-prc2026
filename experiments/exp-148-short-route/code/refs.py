"""exp-148: REFERENCE vectors of the served route from its saved month-fold vectors (no refit).

    S=/path/to/complete-source-tree
    PRC_SOURCE_ROOT=$S $S/.venv/bin/python experiments/exp-148-short-route/code/refs.py --out DIR

v22, v23 and v24 come from exp-146's replay (`dry_run.replay_vectors`: the v22 fold panel,
exp-127's repaired fold update with LIRF held, exp-134's fold learner blended on the ordinary
rows with the served weight). v32 adds exp-137's saved fold arm (`run-v3/proxy`,
`pred_raw`) on the with-clock rows of that file, non-noac: `max(v24 + 0.12797 (arm - v24), 60)`
as the served v32 composed it. These vectors carry the label paths of VALIDATION section 3
(T1 L1-L7); they are a reference, not an admissible comparator. Write-once `refs.parquet`
(MVT_ID_mvt, v22, v23, v24, v32 on the 344,419 January and July departures) and `refs.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp-146-validation/code"))
import dry_run as DR                 # noqa: E402  (exp-146 replay; defines constants only)
from prc import gscore as GS         # noqa: E402

ID = "MVT_ID_mvt"
EXP137 = ("7500f236", "experiments/exp-137-withclock-proxies/kaggle/run-v3/proxy/output/main_predictions.parquet")
W32 = 0.12797          # served v32 weight (kaggle/run-v4/final/kernel.py:27-32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pq, js = a.out / "refs.parquet", a.out / "refs.json"
    if pq.exists() or js.exists():
        raise SystemExit(f"{pq} exists; write-once")
    rows = GS.load_rows()
    rv = DR.replay_vectors(rows)
    cat = {t: pd.concat([rv[f]["vec"][t] for f in DR.FOLDS]) for t in ("v22", "v23", "v24_wserved")}
    e = pd.read_parquet(io.BytesIO(DR.git_blob(*EXP137)))
    e = e[e.fold.isin(list(DR.FOLDS))]
    e = e.set_index(e[ID].astype("int64"))
    cls = rows.set_index(ID).cls
    served = cls.reindex(e.index).to_numpy() == "clock"
    arm = e.pred_raw[served].astype("float64")
    v24 = cat["v24_wserved"]
    v32 = v24.copy()
    v32.loc[arm.index] = v24.loc[arm.index] + W32 * (arm - v24.loc[arm.index])
    v32 = np.maximum(v32, 60.0)
    out = pd.DataFrame({ID: v24.index.to_numpy("int64"), "v22": cat["v22"].to_numpy("float64"),
                        "v23": cat["v23"].to_numpy("float64"), "v24": v24.to_numpy("float64"),
                        "v32": v32.to_numpy("float64")})
    jan_jul = rows[rows.month.isin([1, 7])]
    if len(out) != len(jan_jul) or set(out[ID]) != set(jan_jul[ID]) or not np.isfinite(out.iloc[:, 1:]).all().all():
        raise SystemExit(f"reference vectors cover {len(out)} rows, not the {len(jan_jul)} January/July departures")
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pq, index=False)
    js.write_text(json.dumps({
        "what": "reference vectors of the served route from saved month-fold vectors (not admissible)",
        "rows": int(len(out)), "v32_moved_rows": int(len(arm)), "w32": W32,
        "exp137_rows_not_clock_class": int((~served).sum()),
        "inputs": {**DR.inputs, "exp137 " + EXP137[1] + "@" + EXP137[0]: hashlib.sha256(DR.git_blob(*EXP137)).hexdigest()},
        "per_fold": {f: {k: v for k, v in rv[f].items() if k != "vec"} for f in DR.FOLDS},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": GS._sha256_file(pq)}, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"rows": len(out), "v32_moved": int(len(arm)),
                      "mean_abs_v32_minus_v24": float((out.v32 - out.v24).abs().mean())}))


if __name__ == "__main__":
    main()
