"""Replay v22 -> v23 -> v24 -> v32 from published prediction arrays.

    python scripts/build_v32.py --data-root DIR --out-dir DIR --record PATH.json

No fit runs. The export holds the v22 values and the three update arrays, with
no organizer IDs or labels. IDs come from the separately supplied challenge
submission template. release/V32-REPLAY.json fixes the input, array, value and
Parquet hashes, plus the aggregate counts from the original build receipts.
Exact Parquet replay needs pandas 2.3.3 and pyarrow 24.0.0. A fresh fit has not
yet reproduced these arrays; see docs/REPRODUCE.md.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import platform
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
N_ROWS = 344841
FLOOR_S = 60.0

if not (ROOT / "LICENSE").is_file():
    raise SystemExit("Run build_v32.py from its place in the exported source tree.")

# Public record contains no private receipt or diary dependency.
REPLAY = json.loads((ROOT / "release/V32-REPLAY.json").read_text())
BASE = REPLAY["base"]
STAGES = tuple(REPLAY["stages"])

#: Receipt fields that the replay must repeat exactly.
RECEIPT_KEYS = ("vector_sha256", "candidate_sha256", "bytes", "rows", "rows_moved",
                "rows_floored", "rows_held")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def value_sha256(frame: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(frame[ID].to_numpy("<f8").tobytes())
    h.update(frame[Y].to_numpy("<f8").tobytes())
    return h.hexdigest()


def refuse_existing(paths, overwrite: bool) -> None:
    """The write-once guard of experiments/exp-046-surfacefit/code/nooverwrite.py."""
    hit = [Path(p) for p in paths if Path(p).exists()]
    if hit and not overwrite:
        raise SystemExit("REFUSING TO OVERWRITE -- these artifacts already exist:\n  "
                         + "\n  ".join(str(p) for p in hit)
                         + "\nPass --overwrite only if replacing them is the intent.")


def recorded_sha256(board: int) -> str:
    """The published artifact digest, copied from the committed upload record."""
    return BASE["sha256"] if board == 22 else next(s["sha256"] for s in STAGES if s["board"] == board)


def saved_array(spec: dict, rows: int | None = None) -> np.ndarray:
    """Verify a published prediction array with no organizer IDs or labels; `rows` is 344,841 for the ranking
    file (N_ROWS, read at the call) and 670,790 for an entry of the final mode (the entry's `rows`)."""
    rows = N_ROWS if rows is None else rows
    path = ROOT / spec["public_vector"]
    if sha256(path) != spec["public_vector_sha256"]:
        raise SystemExit(f"saved array hash mismatch: {path}")
    packed = json.loads(path.read_text())
    if packed["encoding"] != "gzip-base64" or packed["dtype"] != "<f8" or packed["rows"] != rows:
        raise SystemExit(f"invalid saved array schema: {path}")
    raw = gzip.decompress(base64.b64decode(packed["data"], validate=True))
    if len(raw) != rows * 8 or hashlib.sha256(raw).hexdigest() != packed["array_sha256"]:
        raise SystemExit(f"saved array content mismatch: {path}")
    values = np.frombuffer(raw, dtype="<f8")
    if not np.isfinite(values).all():
        raise SystemExit(f"non-finite saved array: {path}")
    return values


def v22_chain_sha256() -> str:
    """The digest that scripts/build_v22.py records for the uploaded v22 file."""
    spec = importlib.util.spec_from_file_location("build_v22", ROOT / "scripts/build_v22.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (upload,) = [s for s in module.STAGES if s["id"] == "upload"]
    return upload["sha256"]


def compose(stage: dict, base: pd.DataFrame, v22: pd.DataFrame, meta: pd.DataFrame,
            tid: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """One step, with the checks of its original builder."""
    delta = saved_array(stage)
    assert len(delta) == N_ROWS
    assert (base[ID].to_numpy().astype(np.int64) == tid).all(), "base not in template order"

    noac = meta.AIRCRAFT_TYPE_mvt.isna().to_numpy()
    held = meta.ADEP_mvt.isin(stage["hold"]).to_numpy() & (delta != 0)
    delta = np.where(held, 0.0, delta)
    moved = delta != 0
    b = base[Y].to_numpy(dtype=np.float64)
    assert not (moved & noac).any(), "update touches noac rows"
    if stage["board"] == 23:
        assert not (moved & (meta.ADEP_mvt == "LFPG").to_numpy()).any(), "update touches LFPG"
    if stage["board"] == 24:
        clock_moved = b != v22[Y].to_numpy(dtype=np.float64)
        assert not (moved & clock_moved).any(), "delta touches rows the v23 update changed"

    if stage["floor"] == "moved":
        new = b.copy()
        new[moved] = np.maximum(b[moved] + delta[moved], FLOOR_S)
        floored = int(((b + delta) < FLOOR_S)[moved].sum())
    else:
        new = np.maximum(b + delta, FLOOR_S)
        floored = int(((b + delta) < FLOOR_S).sum())
    assert np.isfinite(new).all() and new.min() >= min(FLOOR_S, b.min())
    assert (new[~moved] == b[~moved]).all()

    out = pd.DataFrame({ID: base[ID].to_numpy(), Y: new})
    chg = new - b
    return out, dict(rows=int(len(out)), rows_moved=int(moved.sum()), rows_floored=floored,
                     rows_held=int(held.sum()), public_vector_sha256=stage["public_vector_sha256"],
                     rows_changed=int((chg != 0).sum()),
                     change_rms_moved_s=float(np.sqrt(np.mean(chg[moved] ** 2))),
                     change_max_abs_s=float(np.abs(chg).max()), min_pred_s=float(new.min()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=ROOT,
                    help="directory that holds the challenge inputs under data/raw/")
    ap.add_argument("--out-dir", type=Path, help="keep v23, v24 and v32 here")
    ap.add_argument("--record", type=Path, help="write the JSON record here")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    names = {s["board"]: f"v{s['board']}-replay.parquet" for s in STAGES}
    targets = ([a.out_dir / n for n in names.values()] if a.out_dir else []) \
        + ([a.record] if a.record else [])
    refuse_existing(targets, a.overwrite)

    data = a.data_root.resolve()
    for p in (data / "data/raw/submitting.parquet", data / "data/raw/ranking.parquet"):
        if not p.is_file():
            raise SystemExit(f"missing input {p}; pass --data-root <checkout with data/raw>")

    want22 = recorded_sha256(22)
    if want22 != v22_chain_sha256():
        raise SystemExit("public replay record and build_v22.py disagree on v22")
    for name, key in (("submitting", "input_template_sha256"), ("ranking", "ranking_sha256")):
        if sha256(data / f"data/raw/{name}.parquet") != REPLAY[key]:
            raise SystemExit(f"wrong {name}.parquet input hash")

    tmpl = pd.read_parquet(data / "data/raw/submitting.parquet")
    rank = pd.read_parquet(data / "data/raw/ranking.parquet",
                           columns=[ID, "ADEP_mvt", "AIRCRAFT_TYPE_mvt"])
    tid = tmpl[ID].to_numpy().astype(np.int64)
    meta = tmpl[[ID]].astype({ID: np.int64}).merge(
        rank.drop_duplicates(ID).astype({ID: np.int64}), on=ID, how="left")
    assert len(tmpl) == len(meta) == N_ROWS

    v22 = pd.DataFrame({ID: tmpl[ID].to_numpy(dtype=BASE["id_dtype"]), Y: saved_array(BASE)})
    # Recreate the original v22 container from template IDs and saved predictions.
    with tempfile.TemporaryDirectory() as td:
        base_path = Path(td) / "v22.parquet"
        v22.to_parquet(base_path, index=False)
        got22 = sha256(base_path)
    if got22 != want22:
        raise SystemExit(f"v22 writer mismatch: {got22}; use pandas 2.3.3 and pyarrow 24.0.0")
    assert len(v22) == N_ROWS and (v22[ID].to_numpy().astype(np.int64) == tid).all()
    record = dict(
        runtime=dict(python=platform.python_version(), pandas=pd.__version__,
                     pyarrow=__import__("pyarrow").__version__, numpy=np.__version__),
        inputs={str(p.relative_to(data)): sha256(p) for p in (
            data / "data/raw/submitting.parquet", data / "data/raw/ranking.parquet")},
        base=dict(board=22, sha256=got22, recorded_sha256=want22, value_sha256=value_sha256(v22),
                  recorded_value_sha256=BASE["value_sha256"]),
        stages=[])
    failures = []
    if record["base"]["value_sha256"] != BASE["value_sha256"]:
        failures.append("v22 values")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = a.out_dir or Path(tmp)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = v22
        for stage in STAGES:
            frame, stats = compose(stage, base, v22, meta, tid)
            path = out_dir / names[stage["board"]]
            refuse_existing([path] if a.out_dir else [], a.overwrite)
            frame.to_parquet(path, index=False)
            want = recorded_sha256(stage["board"])
            got = sha256(path)
            base = pd.read_parquet(path)
            receipt = stage["expected"]
            stats.update(candidate_sha256=got, bytes=path.stat().st_size)
            receipt_diff = {k: [receipt[k], stats[k]] for k in RECEIPT_KEYS
                            if k in receipt and k in stats and receipt[k] != stats[k]}
            row = dict(board=stage["board"], source=stage["source"], vector=stage["vector"],
                       hold=list(stage["hold"]), floor=stage["floor"],
                       public_vector=stage["public_vector"],
                       sha256=got, recorded_sha256=want, bytes_match=got == want,
                       value_sha256=value_sha256(base),
                       recorded_value_sha256=stage["value_sha256"],
                       values_match=value_sha256(base) == stage["value_sha256"],
                       receipt_differences=receipt_diff, **stats)
            if a.out_dir:
                row["file"] = str(path)
            record["stages"].append(row)
            if not row["bytes_match"]:
                failures.append(f"v{stage['board']} bytes")
            if not row["values_match"]:
                failures.append(f"v{stage['board']} values")
            if receipt_diff:
                failures.append(f"v{stage['board']} receipt {sorted(receipt_diff)}")
            print(f"v{stage['board']}: {got}  bytes {'ok' if row['bytes_match'] else 'MISMATCH'}"
                  f"  values {'ok' if row['values_match'] else 'MISMATCH'}"
                  f"  receipt {'ok' if not receipt_diff else receipt_diff}"
                  f"  moved {stats['rows_moved']}", file=sys.stderr)

    record["failures"] = failures
    record["verdict"] = "PASS" if not failures else "FAIL"
    text = json.dumps(record, indent=1)
    if a.record:
        refuse_existing([a.record], a.overwrite)
        a.record.parent.mkdir(parents=True, exist_ok=True)
        a.record.write_text(text + "\n")
    print(text)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
