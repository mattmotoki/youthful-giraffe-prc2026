"""Row features of April, May and September 2025 for the twelve-month head frame.

    cd <repo root on the Mac>; .venv/bin/python experiments/exp-159-clock-head/code/features12.py

Calls valid's own month builder (`features_r2.month_rows` of exp-148, read from valid's worktree,
unchanged) with the three blocks of `block12.py`. So the class rule (exp-146: the ungated block,
the served mode-4 gate, noac = no aircraft type), the 27 `obs_*` columns, the 76 shipped `shp_*`
columns and the queue columns have the construction of the nine months of `features-r4c.parquet`.
The exp-047 cross-check count of each month is measured first and then required (as valid did
for its months). Reads no label. Columns and order: the nine-month table's, less `v23_*` (valid's
R2 and R4 months have none either). Write-once: `artifacts/features-r5.parquet` and
`results/features-r5.json`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]     # no module of this folder is imported
V = Path(os.environ.get("EXP148_TREE", str(ROOT.parent / "prc-data-challenge-2026-exp147")))
os.environ.setdefault("PRC_SOURCE_ROOT", str(ROOT))      # raw data, exp-046, exp-047 blocks, exp-018 caches
sys.path.insert(0, str(V / "experiments/exp-148-short-route/code"))

import pandas as pd                   # noqa: E402
import features as FE                 # noqa: E402  (valid's, exp-148)
import features_r2 as R2              # noqa: E402

MONTHS = ("2025-04", "2025-05", "2025-09")
ART = HERE.parent / "artifacts"
OUT, REC = ART / "features-r5.parquet", HERE.parent / "results/features-r5.json"
COLUMNS = HERE.parent / "results/features-r4c-columns.json"


def main() -> None:
    if OUT.exists() or REC.exists():
        raise SystemExit(f"{OUT} or {REC} exists; write-once")
    t0 = time.monotonic()
    blocks = {}
    for m in MONTHS:
        b = json.loads((ART / f"block-{m}.json").read_text())
        p = ART / f"block-{m}-{b['key']}-ungated.parquet"
        if FE.sha(p) != b["sha256"]:
            raise SystemExit(f"{m}: the block is not the recorded one")
        R2.UNGATED[m] = os.path.relpath(p, FE.E046 / "work")
        blocks[m] = {"file": p.name, "sha256": b["sha256"]}
    _, CF, OBS = FE.e046()
    sys.path.insert(0, str(FE.SRC / "experiments/exp-044-zerosum/code"))

    probe = {"_cross_check_only": True, "_drop_outside_days": True}
    for m in MONTHS:
        R2.month_rows(m, CF, OBS, {}, probe)
    cross = {m: int(probe[m]["exp047_cross_check_rows_differ"]) for m in MONTHS}
    print("exp-047 cross-check rows that differ:", cross, flush=True)
    R2.CROSS_CHECK_EXCEPTIONS.update(cross)

    inputs, checks = {}, {"_cross_check_only": False, "_drop_outside_days": True}
    parts = [R2.month_rows(m, CF, OBS, inputs, checks) for m in MONTHS]
    dropped = checks.pop("_dropped_outside_days", {})
    checks.pop("_cross_check_only"), checks.pop("_drop_outside_days")
    out = pd.concat(parts, ignore_index=True)
    if out[FE.ID].duplicated().any():
        raise SystemExit("repeated ids")
    ref = json.loads(COLUMNS.read_text())
    ref_cols = [c for c in ref["columns"] if not c.startswith("v23_")]
    if set(out.columns) != set(ref_cols):
        raise SystemExit(f"columns differ from the nine-month table: {sorted(set(out.columns) ^ set(ref_cols))}")
    out = out[ref_cols]
    for c in out.columns:                                   # as features_r2.main
        if c.startswith(("obs_", "shp_", "q_n")) or c in ("q_taxi_o", "q_taxi_f", "q_mvt"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    cover = {g: {m: float(out.loc[out.ctx_month == int(m[-2:]), [c for c in out.columns if c.startswith(g)]]
                          .notna().mean().mean()) for m in MONTHS} for g in ("obs_", "shp_", "q_")}
    out.to_parquet(OUT, index=False)
    REC.write_text(json.dumps({
        "what": "exp-159 row features of April, May, September 2025 by valid's features_r2.month_rows (label-free)",
        "rows": int(len(out)), "columns": list(out.columns), "checks": checks, "coverage": cover,
        "exp047_cross_check_measured_first": cross, "dropped_outside_days": dropped, "blocks": blocks,
        "inputs": inputs, "valid_tree": str(V),
        "code_sha256": {"features12.py": FE.sha(Path(__file__)),
                        "features_r2.py": FE.sha(V / "experiments/exp-148-short-route/code/features_r2.py"),
                        "features.py": FE.sha(V / "experiments/exp-148-short-route/code/features.py")},
        "output_sha256": FE.sha(OUT), "seconds": round(time.monotonic() - t0, 1)}, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"rows": len(out), "checks": checks, "coverage": cover, "dropped": dropped}, indent=1, default=str))


if __name__ == "__main__":
    main()
