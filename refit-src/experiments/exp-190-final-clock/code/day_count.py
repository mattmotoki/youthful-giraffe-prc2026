"""Rows of one UTC day per airport on the paths of part `head` (exp-190 REPORT; desk, DATA_CARD 48935ed22).

    $PY experiments/exp-190-final-clock/code/day_count.py --head <head.parquet> --ranking <ranking file>
        --day 2026-06-11 --out <json>

Per airport (ADEP_mvt) of the departures whose departure time (MVT_TIME_UTC_mvt) falls on the UTC day: the
departures, the rows with a head value (frame clock rows) and the rows with no head value (no clock: the
no-observation path, served by the other parts). The head part is in template order; the ranking file gives
airport and time (DEP rows only). Reads no label. Write-once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--ranking", type=Path, required=True)
    ap.add_argument("--day", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; write-once")
    head = pd.read_parquet(a.head, columns=[ID, "pred_HB3u", "clock"])
    rk = pd.read_parquet(a.ranking, columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    rk = rk[rk.PHASE_mvt == "DEP"]
    rk[ID] = rk[ID].astype("int64")
    m = head.merge(rk[[ID, "ADEP_mvt", "MVT_TIME_UTC_mvt"]], on=ID, how="left", validate="one_to_one")
    if m.ADEP_mvt.isna().any():
        raise SystemExit("a head row has no DEP row in the ranking file")
    has = np.isfinite(m.pred_HB3u.to_numpy("float64"))
    if not np.array_equal(has, m.clock.to_numpy(bool)):
        raise SystemExit("head values and the clock flag differ")
    m["has_head"] = has
    day = pd.to_datetime(m.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    d = m[day == a.day]
    by = d.groupby(d.ADEP_mvt.astype(str))
    rows = {ap_: {"departures": int(len(g)), "head_value": int(g.has_head.sum()),
                  "no_head_value": int((~g.has_head).sum())} for ap_, g in by}
    rec = {"what": "rows of one UTC day per airport on the paths of part head", "day": a.day,
           "head": str(a.head), "ranking": str(a.ranking), "by_airport": rows,
           "total": {"departures": int(len(d)), "head_value": int(d.has_head.sum()),
                     "no_head_value": int((~d.has_head).sum())}}
    a.out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
