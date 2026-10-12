"""Replay an exp-169 final assembly from saved member values and official files.

An entry in release/ENTRIES.json with recipe "assembly" names a pack index, for
example release/ZABD-PACK.json. The pack lists each input frame of the original
build and the kind of each column:

- id: the movement ID of the split, from the caller's official file;
- official: a column of the caller's official file, joined on the movement ID;
- label: TAXITIME_SEC_mvt from the caller's 2025 training files;
- saved: a model value or rule record that the export saves, in split row order.

Split row order: the 2025 split ("fold") is the DEP rows of the January 2025
training file, then of the July 2025 file, each in file order. The 2026 split
("ranking") is the order of the submission template. Saved arrays hold no
organizer ID and no label. Each rebuilt frame must have the content digest of
the original input before the assembly starts.

The assembly calls the original exp-169 functions in the order of
experiments/exp-169-stack-plan/code/build035.py: the level-2 refit of the
parent (pair and five-column primary weights), its stages, C71/C95/C98 and the
fixed tail (parent bytes), then the six-column primary fit with the dark18
member, its stages, the refit rule statistics, the fixed tail and one bound
projection (candidate bytes). Only build035's evidence gates are left out:
they read private review, audit and forecast records, and they change no value.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments/exp-169-stack-plan/code"
ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
OFFICIAL_FILES = {
    "submitting": "data/raw/submitting.parquet",
    "ranking": "data/raw/ranking.parquet",
    "training_2025_01": "data/raw/training_2025-01-01_2025-02-01.parquet",
    "training_2025_07": "data/raw/training_2025-07-01_2025-08-01.parquet",
}
# Final mode (exp-192 finalplan, 2026-10-08): PRC_RANKING_MODE=final reads the organizers' final ranking file and
# template (the pack's official_sha256 holds their hashes); the 2025 training files are the same.
RANKING_MODE = os.environ.get("PRC_RANKING_MODE") or "ranking"
if RANKING_MODE not in ("ranking", "final"):
    raise SystemExit(f"unknown PRC_RANKING_MODE {RANKING_MODE!r}")
if RANKING_MODE == "final":
    OFFICIAL_FILES = {**OFFICIAL_FILES, "submitting": "data/raw/final_submitting.parquet",
                      "ranking": "data/raw/final_ranking.parquet"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- digests

def _missing_kind(value) -> str | None:
    if value is None:
        return "none"
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if value is pd.NA:
        return "na"
    return None


def column_bytes(series: pd.Series) -> bytes:
    """Canonical bytes of one column: its dtype and its values."""
    dtype = series.dtype
    head = (str(series.name) + "\0" + str(dtype) + "\0").encode()
    if isinstance(dtype, pd.DatetimeTZDtype):
        values = series.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        return head + values.view("<i8").tobytes()
    kind = np.dtype(dtype).kind if not isinstance(dtype, pd.api.extensions.ExtensionDtype) else "X"
    values = series.to_numpy()
    if kind == "f":
        values = np.where(np.isnan(values), np.nan, values)  # one NaN bit pattern
        return head + values.astype(values.dtype.newbyteorder("<")).tobytes()
    if kind in "iub":
        return head + values.astype(values.dtype.newbyteorder("<")).tobytes()
    if kind == "M":
        return head + values.view("<i8").tobytes()
    if kind == "O":
        items = []
        for value in values:
            missing = _missing_kind(value)
            if missing:
                items.append([missing])
            elif isinstance(value, str):
                items.append(value)
            else:
                raise ValueError(f"unsupported object value type {type(value).__name__} in {series.name}")
        return head + json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode()
    raise ValueError(f"unsupported dtype {dtype} in {series.name}")


def frame_digest(frame: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(f"{len(frame)}\0".encode())
    for name in frame.columns:
        h.update(hashlib.sha256(column_bytes(frame[name])).digest())
    return h.hexdigest()


# ---------------------------------------------------------------- saved arrays

def encode_array(series: pd.Series) -> dict:
    """The saved-array JSON body for one column (used by the private pack builder)."""
    values = series.to_numpy()
    kind = values.dtype.kind
    if kind in "fiub":
        # Every binary array is little-endian float64, the one model-array type the
        # release reference gate decodes; rebuild_frame restores the column dtype.
        wide = values.astype("<f8")
        if kind != "f" and not np.array_equal(wide.astype(values.dtype), values):
            raise ValueError(f"saved {values.dtype} values are not exact in float64")
        raw = wide.tobytes()
        dtype = "<f8"
        encoding = "gzip-base64"
    elif kind == "O":
        items = []
        for value in values:
            missing = _missing_kind(value)
            if missing:
                items.append([missing])
            elif isinstance(value, str):
                items.append(value)
            else:
                raise ValueError(f"unsupported object value type {type(value).__name__}")
        raw = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode()
        dtype = "object"
        encoding = "gzip-json"
    else:
        raise ValueError(f"unsupported saved dtype {values.dtype}")
    return {"encoding": encoding, "dtype": dtype, "rows": int(len(values)),
            "array_sha256": hashlib.sha256(raw).hexdigest(),
            "data": base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode()}


def decode_array(path: Path, expected_sha256: str, rows: int) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise SystemExit(f"saved array file hash mismatch: {path}")
    packed = json.loads(path.read_text())
    raw = gzip.decompress(base64.b64decode(packed["data"], validate=True))
    if packed["rows"] != rows or hashlib.sha256(raw).hexdigest() != packed["array_sha256"]:
        raise SystemExit(f"saved array content mismatch: {path}")
    if packed["encoding"] == "gzip-base64":
        values = np.frombuffer(raw, dtype=packed["dtype"]).copy()
    elif packed["encoding"] == "gzip-json" and packed["dtype"] == "object":
        missing = {"none": None, "nan": np.nan, "na": pd.NA}
        values = np.array([missing[v[0]] if isinstance(v, list) else v for v in json.loads(raw)], dtype=object)
    else:
        raise SystemExit(f"invalid saved array schema: {path}")
    if len(values) != rows:
        raise SystemExit(f"saved array row count mismatch: {path}")
    return values


# ---------------------------------------------------------------- official inputs

def official_tables(data_root: Path, expected: dict) -> dict:
    """The caller's official rows in split order, checked by file hash."""
    paths = {}
    for key, rel in OFFICIAL_FILES.items():
        path = Path(data_root) / rel
        if not path.is_file() or sha256_file(path) != expected[key]:
            raise SystemExit(f"wrong or missing official input: {rel}")
        paths[key] = path
    template = pd.read_parquet(paths["submitting"])
    ranking = pd.read_parquet(paths["ranking"])
    ranking = ranking[ranking["PHASE_mvt"].eq("DEP")]
    if not ranking[ID].is_unique or not template[ID].is_unique:
        raise SystemExit("official DEP rows are not unique by movement ID")
    ranking = ranking.set_index(ID).loc[template[ID].to_numpy()].reset_index()
    parts = []
    for key in ("training_2025_01", "training_2025_07"):
        month = pd.read_parquet(paths[key])
        parts.append(month[month["PHASE_mvt"].eq("DEP")])
    fold = pd.concat(parts, ignore_index=True)
    if not fold[ID].is_unique:
        raise SystemExit("official 2025 DEP rows are not unique by movement ID")
    return {"fold": fold, "ranking": ranking, "template_path": paths["submitting"]}


def rebuild_frame(spec: dict, official: pd.DataFrame, rows: int) -> pd.DataFrame:
    columns = {}
    for col in spec["columns"]:
        name, kind, dtype = col["name"], col["kind"], col["dtype"]
        if kind == "id":
            values = official[ID].astype(dtype).to_numpy()
        elif kind == "official":
            source = official[col["source"]]
            if col.get("transform") == "month":
                source = source.dt.month
            values = source.astype(dtype)
            values = values.array if isinstance(values.dtype, pd.DatetimeTZDtype) else values.to_numpy()
        elif kind == "label":
            values = official[Y].astype(dtype).to_numpy()
        elif kind == "saved":
            values = decode_array(ROOT / col["file"], col["sha256"], rows)
            if col["dtype"] != "object":
                values = values.astype(dtype)
        else:
            raise SystemExit(f"unknown column kind {kind}")
        columns[name] = values
    frame = pd.DataFrame(columns)
    for col in spec["columns"]:
        if str(frame[col["name"]].dtype) != col["dtype"]:
            frame[col["name"]] = frame[col["name"]].astype(col["dtype"])
    if len(frame) != rows:
        raise SystemExit("rebuilt frame has the wrong row count")
    return frame


# ---------------------------------------------------------------- assembly

def _assembly_modules():
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import assemble_final_032 as A  # noqa: E402
    import dark_twin035 as T  # noqa: E402
    return A, T


def make_reader(A, frames: dict, records: dict, template_path: Path):
    class PackReader(A.Reader):
        """The original reader, serving rebuilt frames and saved records by key."""

        def path(self, rec):
            if rec["path"] == "official:submitting":
                path = template_path
            else:
                path = ROOT / records[rec["path"]]
            if sha256_file(path) != rec["sha256"]:
                raise SystemExit(f"record hash mismatch: {rec['path']}")
            self.inputs[rec["path"]] = rec["sha256"]
            return path

        #: frames whose C65 row-order offset is served as zero (the C65-out tail)
        zero_order_delta = frozenset()

        def frame(self, rec):
            if rec["path"] == "official:submitting":
                return pd.read_parquet(self.path(rec))
            frame = frames[rec["path"]]
            if frame.attrs.get("digest") != rec["sha256"]:
                raise SystemExit(f"frame digest mismatch: {rec['path']}")
            self.inputs[rec["path"]] = rec["sha256"]
            if rec["path"] in self.zero_order_delta:
                return frame.assign(order_delta=0.)
            return frame

    return PackReader()


def assemble(pack: dict, data_root: Path) -> dict:
    """Rebuild every input, run the assembly, return parent, C65 and candidate bytes."""
    # The original build ran with two BLAS threads; one thread gives the same bytes there.
    official = official_tables(data_root, pack["official_sha256"])
    rows = {split: len(official[split]) for split in ("fold", "ranking")}
    if rows != pack["rows"]:
        raise SystemExit(f"official split sizes {rows} differ from {pack['rows']}")
    frames = {}
    for key, spec in pack["frames"].items():
        frame = rebuild_frame(spec, official[spec["split"]], rows[spec["split"]])
        digest = frame_digest(frame)
        if digest != spec["digest"]:
            raise SystemExit(f"rebuilt frame differs from the original input: {key}")
        frame.attrs["digest"] = digest
        frames[key] = frame
    A, T = _assembly_modules()
    rd = make_reader(A, frames, pack["records"], official["template_path"])
    parent, extra = pack["manifest"]["parent"], pack["manifest"]["extra"]
    # build035.py, numerical lines only.
    contexts = A.load(parent, rd)
    pw = A.fit_weights(parent, contexts, rd, {"refit": True})
    old = {}
    for split, c in contexts.items():
        old[split], _, _ = A.pre_stack(c, pw, split)
    A.rules(parent, contexts, old, rd)
    A.tail(parent, contexts, old, rd)
    _, parent_body = A.serialize(contexts["ranking"], old["ranking"]["final"], rd, parent["common"]["template"])
    extras = {split: rd.aligned(extra[split], c["b"], c["scope"], year=2025 if split == "fold" else 2026)
              for split, c in contexts.items()}
    frame = rd.frame(parent["common"]["rule_frame"]["fold"])
    exposure = rd.frame(parent["common"]["exposure"])
    fw, pp = T.fit(contexts["fold"], extras["fold"], frame.y.to_numpy(), exposure.weight.to_numpy())
    stages = {}
    for split, c in contexts.items():
        assert T.zero_identity(c, extras[split], pw, split)
        stages[split], _ = T.pre(c, extras[split], pw, fw, pp, split)
    statistics, _ = A.rules(parent, contexts, stages, rd)
    A.tail(parent, contexts, stages, rd)
    table, body = A.serialize(contexts["ranking"], stages["ranking"]["final"], rd, parent["common"]["template"])
    with_c65_body = None
    if pack.get("tail_order_delta_zero"):
        # The C65-out entry: the same tail from the same stack12, with the row-order
        # offset at zero (exp-181 c65_serve.py); C66, C58 and the bounds are unchanged.
        with_c65_body = body
        rd.zero_order_delta = frozenset(parent["common"]["exploit_frame"][s]["path"] for s in ("fold", "ranking"))
        A.tail(parent, contexts, stages, rd)
        table, body = A.serialize(contexts["ranking"], stages["ranking"]["final"], rd, parent["common"]["template"])
    return {"parent_body": parent_body, "with_c65_body": with_c65_body, "body": body, "table": table,
            "weights": {"pair": pw, "dark18": {"primary_fold": fw, "primary_ranking": pp}},
            "statistics": statistics, "inputs": rd.inputs}


def runtime() -> dict:
    blas = "unknown"
    try:
        config = np.show_config(mode="dicts")
        blas = config["Build Dependencies"]["blas"].get("name", "unknown")
    except Exception:
        pass
    return {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
            "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": __import__("pyarrow").__version__,
            "blas": blas, "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}}
