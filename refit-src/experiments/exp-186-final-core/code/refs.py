"""The v73-era reference table for the parity harness: every saved part on the 344,841 board rows.

Run (Mac, about 1 min):
    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-186-final-core/code/refs.py \
        --export <release export root> --out <dir>

Sources, all checked by hash before use:
* `release/ZABD-PACK.json` of the release export: every ranking frame (bundle, primary_frame, pair_frame,
  rule_frame, exploit_frame, vector/*) is rebuilt with the export's own `assembly_replay.rebuild_frame` and
  must give the pack's frame digest; its saved columns enter the table as `<frame>:<column>`.
* `release/saved-vectors/*.json` and `s7c/ranking.json` of the same export (the v22 to v73 stage vectors on
  the ranking split, template order): each enters as `saved:<name>`, checked by its array SHA-256.
* The official files by the pack's SHA-256 (ranking.parquet, submitting.parquet, two 2025 months).

Writes `<out>/v73-parts.parquet` (one row per board departure in template order: `MVT_ID_mvt` int64,
`airport`, `month`, then the parts) and `<out>/v73-parts.json` (sources, digests, columns, output SHA-256).
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ID = "MVT_ID_mvt"
ROWS = 344_841


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def saved_vector(path: Path) -> tuple[np.ndarray, dict]:
    """A `release/saved-vectors` array, checked by its own array SHA-256."""
    packed = json.loads(path.read_text())
    raw = gzip.decompress(base64.b64decode(packed["data"], validate=True))
    assert hashlib.sha256(raw).hexdigest() == packed["array_sha256"], path
    assert packed["encoding"] == "gzip-base64" and packed["rows"] == ROWS, (path, packed["encoding"], packed["rows"])
    v = np.frombuffer(raw, dtype=packed["dtype"]).astype("float64")
    assert len(v) == ROWS, path
    return v, {"file": str(path), "file_sha256": sha(path), "array_sha256": packed["array_sha256"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    export = a.export.resolve()
    sys.path.insert(0, str(export / "release"))
    import assembly_replay as R  # noqa: E402

    pack_path = export / "release" / "ZABD-PACK.json"
    pack = json.loads(pack_path.read_text())
    assert pack["rows"]["ranking"] == ROWS, pack["rows"]
    official = R.official_tables(ROOT, pack["official_sha256"])
    rk = official["ranking"]
    tpl = pd.read_parquet(official["template_path"])
    assert len(rk) == ROWS and (rk[ID].to_numpy() == tpl[ID].to_numpy()).all()
    ids = rk[ID].to_numpy("float64")
    assert np.all(ids == np.round(ids))
    t = pd.to_datetime(rk["MVT_TIME_UTC_mvt"], utc=True)
    out = pd.DataFrame({ID: ids.astype("int64"), "airport": rk["ADEP_mvt"].astype(str).to_numpy(),
                        "month": t.dt.strftime("%Y-%m").to_numpy()})
    rec: dict = {"what": "exp-186 v73-era reference parts on the board rows (template order)",
                 "export": str(export), "pack_sha256": sha(pack_path),
                 "official_sha256": pack["official_sha256"], "frames": {}, "saved_vectors": {}, "columns": {}}
    for key, spec in pack["frames"].items():
        if spec["split"] != "ranking":
            continue
        f = R.rebuild_frame(spec, rk, ROWS)
        digest = R.frame_digest(f)
        assert digest == spec["digest"], (key, digest, spec["digest"])
        idc = [c["name"] for c in spec["columns"] if c["kind"] == "id"]
        assert len(idc) == 1, key
        fid = f[idc[0]].to_numpy("float64")
        assert np.array_equal(fid, ids), f"{key}: ids are not in template order"
        name = key.removeprefix("ranking/")
        rec["frames"][name] = {"digest": digest, "saved": []}
        for c in spec["columns"]:
            if c["kind"] != "saved":
                continue
            col = f"{name}:{c['name']}"
            out[col] = f[c["name"]].to_numpy()
            rec["frames"][name]["saved"].append(c["name"])
            rec["columns"][col] = {"source": f"ZABD-PACK frame {key}", "dtype": str(out[col].dtype)}
    sv = export / "release" / "saved-vectors"
    for p in sorted(sv.glob("*.json")) + [sv / "s7c" / "ranking.json"]:
        name = "s7c" if p.parent.name == "s7c" else p.stem
        try:
            v, info = saved_vector(p)
        except AssertionError as e:            # a support table or another row count: not a ranking vector
            rec["saved_vectors"][name] = {"skipped": str(e)[:200], "file": str(p)}
            continue
        out[f"saved:{name}"] = v
        rec["saved_vectors"][name] = info
        rec["columns"][f"saved:{name}"] = {"source": str(p.relative_to(export)), "dtype": "float64"}
    a.out.mkdir(parents=True, exist_ok=True)
    pq = a.out / "v73-parts.parquet"
    assert not pq.exists(), f"{pq} exists; write-once"
    out.to_parquet(pq, index=False)
    rec |= {"rows": int(len(out)), "rows_by_month": out["month"].value_counts().sort_index().to_dict(),
            "output": str(pq), "output_sha256": sha(pq), "code_sha256": sha(Path(__file__)),
            "seconds": round(time.time() - t0, 1)}
    (a.out / "v73-parts.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("rows", "rows_by_month", "output_sha256", "seconds")}, indent=1))
    print(len(rec["columns"]), "columns:", ", ".join(rec["columns"]))


if __name__ == "__main__":
    main()
