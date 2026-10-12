"""Parity check P1: `prc.foldserve.head_correction` rebuilds the served v22 head (exp-147).

    S=/path/to/complete-source-tree
    PRC_SOURCE_ROOT=$S $S/.venv/bin/python experiments/exp-147-foldserve/code/headparity.py

The served v22 (upload 22, candidate `exp-046-surface-v11.parquet`, sha f11d34f8...) is v18's
bytes with the surface head written on the served class: `v12 + d` there, where d is the
mean of the ten cached ranking members of the served month's head, zeroed on guard-held
rows, then floored (exp-046 `serve.py:1284, 2471-2519`, base set v25). This script takes
the same twenty member files, the v25 class export and the v12/v18 files, calls the shared
function, and compares every row with the uploaded v22 file. The fold side of the repaired
pipeline calls the same function on raw fold members, so a match here means the fold price
and the served vector come from one code path. Reads no label. Writes
`experiments/exp-147-foldserve/results/headparity.json` (write-once).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
SRC = Path(os.environ.get("PRC_SOURCE_ROOT", str(ROOT)))
os.environ.setdefault("PRC_DATA_DIR", str(SRC / "data/raw"))

from prc import data as D           # noqa: E402
from prc import foldserve as FS      # noqa: E402

OUT = EXP / "results/headparity.json"
UP = SRC / "submissions/uploaded/youthful-giraffe_{v}.parquet"
SHA = {"v12": "48dd572d61ff2403f13614ce04bdc3a8fd0dbdb5175bef7eb143650ea520ec52",
       "v18": "4d75706bfcc13d88f7556265cd008b3d3a312555955dca8eb528e4db5325fee8",
       "v22": "f11d34f88b62a43ec2a0cdbeec967a9ee4b7d92e50a5508aa4d30e5a78ea5e1c"}
CLASS = SRC / "artifacts/exp-046/class/{m}-v25.parquet"
MEMBERS = {"2026-01": "work/d-rank-seasjan-placebo{s}-255.npz",
           "2026-07": "work/d-rank-seasjul-placebo{s}-255.npz"}
SEEDS = range(1, 11)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    inputs = {}
    rk = D.load_ranking()
    dep = rk[rk.PHASE_mvt == "DEP"].reset_index(drop=True)      # serve.py:1403 row order
    ids = dep.MVT_ID_mvt.to_numpy("float64")
    vec = {}
    for v, want in SHA.items():
        p = Path(str(UP).format(v=v))
        got = sha(p)
        if got != want:
            raise SystemExit(f"{p.name} is {got[:12]}, not {want[:12]}")
        inputs[str(p.relative_to(SRC))] = got
        f = pd.read_parquet(p).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
        vec[v] = f.reindex(ids).to_numpy("float64")
        if not np.isfinite(vec[v]).all():
            raise SystemExit(f"{v} misses a ranking departure")
    cls = pd.concat([pd.read_parquet(Path(str(CLASS).format(m=m))) for m in MEMBERS])
    for m in MEMBERS:
        p = Path(str(CLASS).format(m=m))
        inputs[str(p.relative_to(SRC))] = sha(p)
    cls = cls.set_index("MVT_ID_mvt").reindex(ids)
    served = cls.served_by_exp_046.to_numpy(bool)
    guard = cls.guard_held_by_exp_046.to_numpy(bool)
    month = dep.month.to_numpy()

    d = np.zeros(len(ids))
    per_month = {}
    for m, pat in MEMBERS.items():
        mem = []
        for s in SEEDS:
            p = SRC / "experiments/exp-046-surfacefit" / pat.format(s=s)
            inputs[str(p.relative_to(SRC))] = sha(p)
            mem.append(np.load(p)["d"].astype("float64"))
        dm = FS.head_correction(vec["v12"], mem, in_class=served, held=guard)
        rows = month == m
        d[rows] = dm[rows]
        per_month[m] = {"rows": int(rows.sum()), "served": int((served & rows).sum()),
                        "guard_held": int((guard & rows).sum()),
                        "moved": int((dm[rows] != 0).sum())}
    pred = vec["v18"].copy()
    pred[served] = (vec["v12"] + d)[served]

    diff = pred - vec["v22"]
    res = {
        "what": "shared head post-process vs the uploaded v22, every ranking departure",
        "rows": int(len(ids)), "served_rows": int(served.sum()), "guard_held": int(guard.sum()),
        "per_month": per_month,
        "bit_identical_rows": int((pred == vec["v22"]).sum()),
        "max_abs_diff_s": float(np.abs(diff).max()),
        "rows_diff_over_1e-9_s": int((np.abs(diff) > 1e-9).sum()),
        "served_rows_diff_over_1e-9_s": int((np.abs(diff[served]) > 1e-9).sum()),
        "floor_bound_rows": int(((vec["v12"] + d)[served] <= FS.FLOOR).sum()),
        "inputs": inputs,
    }
    res["pass"] = res["rows_diff_over_1e-9_s"] == 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "inputs"}, indent=1))


if __name__ == "__main__":
    main()
