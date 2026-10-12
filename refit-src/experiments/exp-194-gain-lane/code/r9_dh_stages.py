"""The --stages record of the R9 dh file for check_final_candidate.py (release73's form for final-iii-dh): the stages
of the V file's record, then one stage for the dh step (rows changed by month against the V file).

    $PY -I r9_dh_stages.py <V file stages record> <V file> <dh file> <dir with final_*.parquet> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
rec_path, vfile, dhfile, raw, out = (Path(x) for x in sys.argv[1:6])
assert not out.exists(), f"{out} exists (write once)"
ids = pd.read_parquet(raw / "final_submitting.parquet", columns=[ID])[ID].to_numpy("float64")
v, h = pd.read_parquet(vfile), pd.read_parquet(dhfile)
for f in (v, h):
    assert np.array_equal(f[ID].to_numpy("float64"), ids)
r = pd.read_parquet(raw / "final_ranking.parquet", columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"])
r = r[r.PHASE_mvt.eq("DEP")].set_index(ID).reindex(ids)
month = pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
ch = v[Y].to_numpy(float) != h[Y].to_numpy(float)
base = json.loads(rec_path.read_text())
stage = {"name": "gainlane/r9-dh-file:TAXITIME_SEC_mvt",
         "changed_by_month": {f"2026-{m:02d}": int((ch & (month == m)).sum()) for m in (1, 2, 6, 7)},
         "against": f"V file {vfile}"}
out.write_text(json.dumps({"v_record": str(rec_path), "stages": base["stages"] + [stage]}, indent=1) + "\n")
print("R9DHSTAGES", json.dumps(stage))
