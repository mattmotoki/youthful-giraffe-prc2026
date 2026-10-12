"""The clock input of frames_v.py from the v25 class files of the final file (finaldata): one row per departure of
final_submitting.parquet, clock = served_by_exp_046 (the clock class of the route features, exp-148 features.py:228).

A month without a class file stops the run; `--missing-false` (dry runs only) sets its rows to not clock.

    python experiments/exp-187-final-lirf/code/clock_input.py --class-file <2026-01-v25.parquet> [--class-file ...] \
        --out <clock.parquet> [--missing-false]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from prc import data as D  # noqa: E402

ID = "MVT_ID_mvt"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-file", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--missing-false", action="store_true")
    a = ap.parse_args()
    tpl = pd.read_parquet(D.submitting_file("final"), columns=[ID])
    raw = pd.read_parquet(D.ranking_file("final"), columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt"])
    dep = raw[raw.PHASE_mvt.eq("DEP")].set_index(ID).loc[tpl[ID].to_numpy()]
    month = dep.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    c = pd.concat([pd.read_parquet(p) for p in a.class_file], ignore_index=True)
    assert c[ID].is_unique and set(c[ID]) <= set(tpl[ID])
    c = c.set_index(ID).reindex(tpl[ID].to_numpy())
    have = c.served_by_exp_046.notna().to_numpy()
    for m in np.unique(month):
        on = month == m
        assert have[on].all() or not have[on].any(), f"month {m}: the class file covers only part of the month"
        if not have[on].any():
            assert a.missing_false, f"month {m} has no class file"
    clock = c.served_by_exp_046.fillna(False).to_numpy(bool)
    noac = c.noac.fillna(False).to_numpy(bool)
    notype = dep.AIRCRAFT_TYPE_mvt.isna().to_numpy()
    assert not (clock & noac).any()
    assert np.array_equal(noac[have], notype[have]), "class noac differs from the missing aircraft type"
    out = pd.DataFrame({ID: tpl[ID].to_numpy("int64"), "clock": clock, "class_present": have})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    rec = {"class_files": {str(p): sha(p) for p in a.class_file}, "missing_false": a.missing_false,
           "rows": int(len(out)), "clock_by_month": {int(m): int(clock[month == m].sum()) for m in np.unique(month)},
           "class_present_by_month": {int(m): int(have[month == m].sum()) for m in np.unique(month)},
           "obs_has_clock_differs": int((c.obs_has_clock.fillna(False).to_numpy(bool) != clock).sum()),
           "out_sha256": sha(a.out)}
    a.out.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
