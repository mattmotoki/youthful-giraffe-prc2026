"""The DH stage (target final-dh): the final V file plus one step on 14,587 EDDF rows without a head value.

    python scripts/build_entry.py --target final-dh --data-root DIR --out-dir NEW --record NEW.json

value = max(60, V + w (pred_DH - V)) on the rows where w > 0, and V on every other row.

- V is the file of the entry `final` (the v73 steps on the final chain). This stage reads it from the saved values
  of that entry (release/saved-vectors/final-entries/v.json); with the IDs of final_submitting.parquet they must give
  the registered V bytes. The target `final` replays V itself.
- pred_DH is the reduced clock head DH: a LightGBM bag (3 seeds) on the 2025 clock rows without January and July,
  with no adsb.lol input (exp-194, `experiments/exp-194-gain-lane/code/dh_fit.py` and `dh_apply.py`). The stage
  reads its saved values, as the other stages read their saved member values; no fit runs.
- w is 0.164 on the EDDF rows of January, February and June 2026 that are typed, have no head value, and whose own
  adsb.lol inputs are absent on an airport-day where less than 20 % of the departures have an observed taxi (the
  label-free flag dark_b of `dh_dark_rows.py`), and 0 elsewhere. The weight 0.164 comes from held January and July
  2025 rows made dark.

The saved vectors pred_DH, has_head and w_pair, with the final_submitting IDs, must give the bytes of the part that
the final build used (the entry's `part_sha256`). The step has no BLAS or tree step, so the bytes are the same on
every platform with IEEE float64.
"""
from __future__ import annotations

import hashlib
import io
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import assembly_replay as R

ROOT = R.ROOT
ID, Y = R.ID, R.Y
FLOOR = 60.0
FINAL = R.RANKING_MODE == "final"
PART_COLUMNS = ("pred_DH", "has_head", "w_pair")


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def table_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def compose(spec: dict, book: dict, data_root: Path, out_dir: Path) -> dict:
    """Apply the DH step to the saved V values; return the bytes, the table and the facts."""
    if not FINAL or spec.get("mode") != "final":
        raise SystemExit("the DH stage is a final-mode stage; PRC_RANKING_MODE must be final")
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_v32 as legacy
    for name, expected in spec["official_sha256"].items():
        path = Path(data_root) / "data/raw" / name
        if not path.is_file() or R.sha256_file(path) != expected:
            raise SystemExit(f"wrong or missing official input: data/raw/{name}")
    for rel, expected in spec["code"].items():
        if R.sha256_file(ROOT / rel) != expected:
            raise SystemExit(f"step code differs from the registered hash: {rel}")
    base = book["entries"][spec["base"]]
    if base["sha256"] != spec["base_sha256"]:
        raise SystemExit("the DH stage base is not the registered final entry")
    ids = pd.read_parquet(Path(data_root) / R.OFFICIAL_FILES["submitting"])[ID].to_numpy("float64")
    rows = spec["rows"]
    if len(ids) != rows:
        raise SystemExit("final_submitting.parquet does not have the registered row count")

    v = legacy.saved_array(base["comparison_vector"], base["rows"])
    if sha256_bytes(table_bytes(pd.DataFrame({ID: ids, Y: v}))) != base["sha256"]:
        raise SystemExit("the saved V values do not give the registered V bytes")

    member = spec["member"]
    cols = {c: R.decode_array(ROOT / member[c]["public_vector"], member[c]["public_vector_sha256"], rows)
            for c in PART_COLUMNS}
    part = pd.DataFrame({ID: ids, "pred_DH": cols["pred_DH"], "has_head": cols["has_head"].astype(bool),
                         "w_pair": cols["w_pair"]})
    if sha256_bytes(table_bytes(part)) != member["part_sha256"]:
        raise SystemExit("the saved DH vectors do not give the bytes of the final build's DH part")

    w, dh = cols["w_pair"], cols["pred_DH"]
    on = w > 0
    if int(on.sum()) != member["scope_rows"] or not set(np.unique(w).tolist()) <= {0.0, member["w"]}:
        raise SystemExit("the DH weights are not the registered scope")
    if part["has_head"].to_numpy(bool)[on].any() or not np.isfinite(dh[on]).all():
        raise SystemExit("a DH row has a head value or no DH value")
    raw_new = v[on] + w[on] * (dh[on] - v[on])
    values = v.copy()
    values[on] = np.maximum(FLOOR, raw_new)
    body = table_bytes(pd.DataFrame({ID: ids, Y: values}))
    table = pd.read_parquet(io.BytesIO(body))
    d = values - v
    checks = {
        "template_order": bool(np.array_equal(table[ID].to_numpy("float64"), ids)),
        "finite": bool(np.isfinite(values).all()),
        "min_60": bool(values.min() >= FLOOR),
        "rows": len(table) == rows,
        "outside_scope_equal_v": bool(np.array_equal(values[~on].view(np.uint64), v[~on].view(np.uint64))),
    }
    if not all(checks.values()):
        raise SystemExit(f"stage checks failed: {sorted(k for k, ok in checks.items() if not ok)}")
    facts = {"platform": f"{platform.system()}-{platform.machine()}",
             "base_sha256": base["sha256"], "part_sha256": member["part_sha256"], "checks": checks,
             "rows_changed_against_v": int((d != 0).sum()),
             "rms_change_against_v_moved_s": float(np.sqrt(np.mean(d[d != 0] ** 2))) if (d != 0).any() else 0.0,
             "floor_60_rows": int((raw_new < FLOOR).sum()), "member_fit_ran": False, "run_folder": str(out_dir)}
    return {"body": body, "table": table, "facts": facts}
