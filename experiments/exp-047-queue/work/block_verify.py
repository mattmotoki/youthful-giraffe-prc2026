"""§7bk: verify the re-emitted 2025-11 block against the 185-column one it replaces.

Registered in §7bk: the re-emitted block must carry **186 columns, all populated**,
and must differ from the 185-column emission **only** in `own_obs_taxi_minus_p11`
and the twelve `{Nf_,No_}_n_res_*` columns.  Every other column must be identical.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lane                                                  # noqa: E402
MONTH = lane.MONTH
NEW = ROOT / f"artifacts/exp-047/block/{MONTH}.parquet"
OLD = ROOT / f"artifacts/exp-047/block-185/{MONTH}.parquet"
OUT = EXP / f"analysis/block-{MONTH}-verify.json"
EXPECTED_NEW = ["own_obs_taxi_minus_p11"] + [
    f"{p}_n_res_{s}" for p in ("Nf", "No")
    for s in ("cell", "rwy_cell", "apt_30m", "rwy_30m", "rwy_60m", "rwy_minus_apt")]


def main() -> None:
    new = pd.read_parquet(NEW)
    old = pd.read_parquet(OLD)
    added = [c for c in new.columns if c not in old.columns]
    dropped = [c for c in old.columns if c not in new.columns]
    shared = [c for c in old.columns if c in new.columns]

    same, differ = [], {}
    for c in shared:
        a, b = old[c].to_numpy(), new[c].to_numpy()
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            eq = np.array_equal(a, b, equal_nan=True)
        else:
            eq = np.array_equal(a.astype(str), b.astype(str))
        if eq:
            same.append(c)
        else:
            n = int((~(pd.isna(a) & pd.isna(b)) & (a != b)).sum())
            differ[c] = n

    num = [c for c in new.columns if new[c].dtype.kind in "fiu"]
    dead = sorted(c for c in num if new[c].notna().sum() == 0)
    was_dead = sorted(c for c in old.columns
                      if old[c].dtype.kind in "fiu" and old[c].notna().sum() == 0)

    # §7br repair.  Both of the checks below were vacuous: on 2025-10 the twelve
    # `_n_res_` columns were ABSENT from the new block, and an absent column is
    # neither all-NaN nor "moved", so both passed on 63 shared columns.  Presence
    # of the thirteen head-dependent columns is now asserted BEFORE they judge.
    missing_head = [c for c in EXPECTED_NEW if c not in new.columns]
    head_all_nan = [c for c in EXPECTED_NEW
                    if c in new.columns and new[c].notna().sum() == 0]
    # every head column the OLD emission already carried must differ in the new one
    must_move = sorted(c for c in EXPECTED_NEW if c in old.columns)

    rec = {
        "new": {"path": str(NEW.relative_to(ROOT)), "rows": int(len(new)),
                "columns": int(len(new.columns)),
                "sha256": hashlib.sha256(NEW.read_bytes()).hexdigest(),
                "bytes": NEW.stat().st_size},
        "old": {"path": str(OLD.relative_to(ROOT)), "rows": int(len(old)),
                "columns": int(len(old.columns)),
                "sha256": hashlib.sha256(OLD.read_bytes()).hexdigest()},
        "columns_added": added,
        "columns_dropped": dropped,
        "shared_columns_identical": len(same),
        "shared_columns_that_differ": differ,
        "all_nan_columns_now": dead,
        "all_nan_columns_before": was_dead,
        "head_columns_expected": sorted(EXPECTED_NEW),
        "head_columns_missing": missing_head,
        "head_columns_all_nan": head_all_nan,
        "head_columns_that_must_move": must_move,
        "checks": {
            "rows_unchanged": len(new) == len(old),
            "is_186": len(new.columns) == 186,
            "nothing_is_all_nan_now":
                missing_head == [] and head_all_nan == [] and dead == [],
            "added_is_exactly_own_obs_taxi_minus_p11": added == ["own_obs_taxi_minus_p11"],
            "nothing_dropped": dropped == [],
            "the_only_columns_that_moved_are_the_head_dependent_ones":
                missing_head == [] and sorted(differ) == must_move,
        },
    }
    rec["PASS"] = all(rec["checks"].values())
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True))
    for k, v in rec["checks"].items():
        print(f"  {'OK ' if v else 'FAIL'} {k}")
    print(f"  columns {len(old.columns)} -> {len(new.columns)}, "
          f"added {added}, all-NaN {len(was_dead)} -> {len(dead)}")
    print(f"  shared identical {len(same)}, moved {sorted(differ)}")
    print(f"  head columns: {len(EXPECTED_NEW) - len(missing_head)}/"
          f"{len(EXPECTED_NEW)} present, missing {missing_head}, "
          f"all-NaN {head_all_nan}, must move {len(must_move)}")
    print("wrote", OUT)
    sys.exit(0 if rec["PASS"] else 1)


if __name__ == "__main__":
    main()
