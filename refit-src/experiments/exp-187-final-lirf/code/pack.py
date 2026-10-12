"""The saved ranking frames of the zabd pack (release export), decoded column by column, as parity references.

The release replay (`release/assembly_replay.py`) rebuilds a frame from saved arrays and the official file; here only the
saved and official columns are decoded, in template order (`submitting.parquet`), with the replay's own decoder.

    import pack; f = pack.frame("ranking/exploit_frame")      # DataFrame, MVT_ID_mvt int64 + every column
"""
from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXPORT = Path(os.environ.get("PRC_SOURCE_EXPORT", "/path/to/source-export"))
PACK = EXPORT / "release/ZABD-PACK.json"
ID = "MVT_ID_mvt"
sys.path.insert(0, str(EXPORT / "release"))
import assembly_replay as AR  # noqa: E402


def spec(key: str) -> dict:
    return json.loads(PACK.read_text())["frames"][key]


FOLD_FILES = ("training_2025-01-01_2025-02-01.parquet", "training_2025-07-01_2025-08-01.parquet")


def fold_ids() -> np.ndarray:
    """The 2025 split's row order: DEP rows of the January 2025 file, then of the July 2025 file (assembly_replay)."""
    parts = []
    for f in FOLD_FILES:
        t = pd.read_parquet(ROOT / "data/raw" / f, columns=[ID, "PHASE_mvt"])
        parts.append(t.loc[t.PHASE_mvt.eq("DEP"), ID])
    return pd.concat(parts, ignore_index=True).to_numpy("int64")


def frame(key: str, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Saved and id columns of one frame; official columns too when `raw` (ranking.parquet) is given (ranking split)."""
    s = spec(key)
    if s["split"] == "fold":
        ids = fold_ids()
        assert raw is None and len(ids) == 344_419
        out = {ID: ids}
        for c in s["columns"]:
            if c["kind"] == "saved":
                v = AR.decode_array(EXPORT / c["file"], c["sha256"], len(ids))
                out[c["name"]] = v if c["dtype"] == "object" else v.astype(c["dtype"])
        return pd.DataFrame(out)
    assert s["split"] == "ranking", key
    tpl = pd.read_parquet(ROOT / "data/raw/submitting.parquet", columns=[ID])
    rows = len(tpl)
    out = {ID: tpl[ID].to_numpy("int64")}
    dep = None
    if raw is not None:
        dep = raw[raw.PHASE_mvt.eq("DEP")].set_index(ID).loc[out[ID]].reset_index()
    for c in s["columns"]:
        name, kind, dtype = c["name"], c["kind"], c["dtype"]
        if kind == "saved":
            v = AR.decode_array(EXPORT / c["file"], c["sha256"], rows)
            out[name] = v if dtype == "object" else v.astype(dtype)
        elif kind == "official" and dep is not None:
            src = dep[c["source"]]
            out[name] = (src.dt.month if c.get("transform") == "month" else src).astype(dtype).to_numpy()
    return pd.DataFrame(out)


def kinds(key: str) -> dict:
    return {c["name"]: (c["kind"], c["dtype"]) for c in spec(key)["columns"]}


RANKING_FRAMES = ["ranking/bundle", "ranking/rule_frame", "ranking/exploit_frame", "ranking/pair_frame", "ranking/primary_frame"]
