"""The 96: newly occupied in v20 but not changed by the v6 bytes.

The desk, 2026-09-09: "newly occupied in v20 is 2,346 but changed is 2,250 -- state
what the 96 are (guard-held / on the floor / identical value) so the reviewer weighs
it rather than finds it."

The two counts answer different questions.  `occupied_by_exp_046` says WHOSE ROW IT IS
-- this lane decided the value, whatever the value turned out to be.  `changed` says the
bytes moved.  A row this lane decides and then returns to the price base is occupied and
unchanged, and that is not a defect: it is the tail guard doing its job.  This joins the
three files row by row and puts every one of the 96 in a named bucket, in priority order,
so the buckets partition.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/gapread.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]

KEY = "MVT_ID_mvt"
VAL = "taxi_out_s"
V6 = ROOT / "submissions/candidates/exp-046-surface-v6.parquet"
V5 = ROOT / "submissions/candidates/exp-046-surface-v5.parquet"
V12 = ROOT / "submissions/candidates/exp-035-k2climb-v11member.parquet"
CLASS = ROOT / "artifacts/exp-046/class"
OUT = EXP / "analysis/gap-96.json"
FLOOR = 60.0


def col(p: Path) -> tuple[pd.Series, pd.Series]:
    d = pd.read_parquet(p)
    v = [c for c in d.columns if c not in (KEY,) and d[c].dtype.kind == "f"]
    return d[KEY], d[v[0] if VAL not in d.columns else VAL]


def main() -> None:
    k6, p6 = col(V6)
    k5, p5 = col(V5)
    k12, p12 = col(V12)
    if not (k6.equals(k5) and k6.equals(k12)):
        raise SystemExit("the three files are not in one template order; join first")
    p6, p5, p12 = (x.to_numpy("float64") for x in (p6, p5, p12))

    cls = {}
    for suf in ("v19", "v20"):
        cls[suf] = pd.concat(
            [pd.read_parquet(CLASS / f"{m}-{suf}.parquet") for m in ("2026-01", "2026-07")],
            ignore_index=True)
    idx = pd.Index(k6)
    a = cls["v19"].set_index(KEY).reindex(idx)
    b = cls["v20"].set_index(KEY).reindex(idx)
    if a.occupied_by_exp_046.isna().any() or b.occupied_by_exp_046.isna().any():
        raise SystemExit("a candidate row is missing from a class vector")

    occ19 = a.occupied_by_exp_046.to_numpy(bool)
    occ20 = b.occupied_by_exp_046.to_numpy(bool)
    held20 = b.guard_held_by_exp_046.to_numpy(bool)
    newly = occ20 & ~occ19
    changed = p6 != p5

    # the buckets, in priority order, so they partition the 96
    gap = newly & ~changed
    at_base = gap & (p6 == p12)
    guard = at_base & held20
    base_not_guard = at_base & ~held20
    floor = gap & ~at_base & (p6 == FLOOR)
    same = gap & ~at_base & ~floor

    rec = {
        "what": __doc__.strip().splitlines()[0],
        "newly_occupied_in_v20": int(newly.sum()),
        "of_those_changed_by_the_bytes": int((newly & changed).sum()),
        "the_gap": int(gap.sum()),
        "buckets": {
            "guard_held_at_the_price_base": int(guard.sum()),
            "at_the_price_base_the_guard_does_not_hold": int(base_not_guard.sum()),
            "on_the_floor_at_60_s": int(floor.sum()),
            "the_head_and_the_base_agree_to_the_bit": int(same.sum()),
        },
        "buckets_partition_the_gap": int(guard.sum() + base_not_guard.sum()
                                         + floor.sum() + same.sum()) == int(gap.sum()),
        "changed_rows_all_newly_occupied": bool((changed & ~newly).sum() == 0),
        "changed_outside_the_new_class": int((changed & ~newly).sum()),
        "guard_held_in_v20_total": int(held20.sum()),
        "guard_held_and_newly_occupied": int((held20 & newly).sum()),
        "spread_of_the_gap": {
            "min_prediction_s": float(p6[gap].min()) if gap.any() else None,
            "max_prediction_s": float(p6[gap].max()) if gap.any() else None,
        },
    }
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(json.dumps(rec, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
