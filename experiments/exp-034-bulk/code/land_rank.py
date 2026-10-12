"""Validate the bulk member's ranking column before it can reach a candidate.

exp-027's Sol round 3 (blocking 2) established why this file has to exist: the
member joins are LEFT joins and LightGBM accepts missing features, so a partial
or stale ranking column would pass every downstream check and still change the
candidate.  `serve.py` therefore requires that this ran and that the bytes it
blessed are the bytes on disk.

exp-027's `land_rank.py` demands a finite prediction on every row.  This member
is PARTIAL by construction, so the check is sharper rather than weaker: the
column must be finite on exactly the ranking rows where `d_takeoff_aobt` is
finite -- the same contract `code/candidate.py` enforces on the three folds, and
the same two independent producers (exp-003's frame builder here, the raw
movement file through `regime_table.py` there).

Checks, all fatal: exactly the submitting template's id set, ids unique, float64,
and the null pattern equal to the ranking file's own `AOBT_3_flt` null pattern.

    python land_rank.py
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
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
sys.path.insert(0, str(ROOT))
import regime_table as RT  # noqa: E402
import prc.data as D  # noqa: E402

SRC = EXP / "candidates" / "bulk64-ranking.parquet"
REC = EXP / "analysis" / "bulk-ranking.json"


def main() -> None:
    d = pd.read_parquet(SRC)
    ids = d[D.ID].astype("float64").to_numpy()
    v = d["y_pred"].astype("float64").to_numpy()

    tmpl = D.load_submitting()[D.ID].astype("float64").to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise SystemExit(f"{SRC.name}: {len(ids) - len(np.unique(ids))} duplicate ids")
    missing = int(np.isin(tmpl, ids, invert=True).sum())
    extra = int(np.isin(ids, tmpl, invert=True).sum())
    if missing or extra:
        raise SystemExit(f"{SRC.name}: id mismatch, {missing} missing, {extra} extra")

    if not RT.OUT.exists():
        RT.build()
    g = RT.load()
    ta = pd.Series(g["d_takeoff_aobt"].to_numpy("float64"),
                   index=g[D.ID].astype("float64").to_numpy()).reindex(ids).to_numpy()
    ok = np.isfinite(v)
    want = np.isfinite(ta)
    if not np.array_equal(ok, want):
        raise SystemExit(f"{SRC.name}: defined on {int(ok.sum())} rows, d_takeoff_aobt on "
                         f"{int(want.sum())}; the null patterns must be identical "
                         f"({int((ok & ~want).sum())} defined off the join, "
                         f"{int((want & ~ok).sum())} undefined on it)")

    rec = {"out": str(SRC.relative_to(ROOT)), "sha256": D.sha256(SRC),
           "n": int(len(ids)), "n_defined": int(ok.sum()),
           "n_undefined": int((~ok).sum()),
           "mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
           "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
           "n_above_3600": int((v[ok] > 3600).sum()),
           "n_below_floor": int((v[ok] < 60.0).sum()),
           "producer": "experiments/exp-034-bulk/code/member.py rank, twelve-month fit",
           "member_record": "experiments/exp-034-bulk/analysis/member-ranking.json",
           "null_contract": "finite exactly where the ranking file's d_takeoff_aobt is finite"}
    REC.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2), flush=True)


if __name__ == "__main__":
    main()
