"""The exp-008 v2 frame with this experiment's chain columns appended.

`experiments/exp-008-diversity/run.py` does `import frames` and reads
`fold_frames`, `all_training`, `ranking_frame`, `feature_columns`,
`cat_columns`, `FAMILY`, `ES_MONTH`, `VERSION`, `SCHEMA_FINGERPRINT` and
`NUMERIC_FINGERPRINT` from it.  `run_arm.py` registers THIS module under that
name, so the member producer is the shipped one, unedited, fitting on a frame
with three (or eight) more columns.

The v2 half is exp-008's own build, read through its own month cache: nothing
here recomputes it, so `PRC_CHAIN_ARM=none` is the shipped frame column for
column and the reproduction gate is a real gate rather than a rebuild that
happens to agree.  The chain half is cached once per month, all eight columns
together, and each arm selects from it.

`PRC_CHAIN_ARM` picks the arm: `none`, `chain` (`ts_gap_arr1..3`), `derived`
(those three plus exp-037's five).  The frame VERSION string carries the arm so
that two arms can never share a cache file.
"""
from __future__ import annotations

import importlib.util
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import prc.data as D  # noqa: E402
from chain_cols import ARMS, columns  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: exp-008's frames.py, by address.  Loaded under a private name because this
#: module is about to be registered as `frames`.
V2 = _load("_exp008_frames", ROOT / "experiments/exp-008-diversity/frames.py")

ARM = os.environ.get("PRC_CHAIN_ARM", "none")
if ARM not in ARMS:
    raise SystemExit(f"PRC_CHAIN_ARM={ARM!r} is not one of {sorted(ARMS)}")
NEW = list(ARMS[ARM])

#: Everything `run.py` reads straight through.
FOLDS = V2.FOLDS
FAMILY = V2.FAMILY
ES_MONTH = V2.ES_MONTH
VERSION = f"{V2.VERSION}+{ARM}"
#: Sol round 1, non-blocking 1.  These are v2's own fingerprints and the arm appends
#: columns to v2, so a fingerprint alone no longer identifies the frame; `VERSION`
#: carries the arm and `CODE_FINGERPRINT` carries the code that builds the new
#: columns, so a stale cache or a changed chain module cannot pass unnoticed.
SCHEMA_FINGERPRINT = f"{V2.SCHEMA_FINGERPRINT}+{ARM}"
NUMERIC_FINGERPRINT = f"{V2.NUMERIC_FINGERPRINT}+{ARM}"


def _code_fingerprint() -> str:
    """SHA-256 over the modules that decide the appended columns."""
    h = hashlib.sha256()
    for f in (Path(__file__),
              Path(__file__).parent / "chain_cols.py",
              ROOT / "experiments/exp-037-tail/code/chain.py",
              ROOT / "prc" / "tail_features.py"):
        h.update(f.read_bytes() if f.exists() else b"MISSING")
    return h.hexdigest()[:16]


CODE_FINGERPRINT = _code_fingerprint()
#: The cache key carries the code that wrote it, so a changed chain module builds a
#: new file instead of silently reusing the old one.
CACHE = ROOT / "artifacts" / "exp-040" / "chain" / CODE_FINGERPRINT
#: all eight columns, so `chain` and `derived` share one build per month
ALL_NEW = list(ARMS["derived"])


def _chain_month(month: str) -> pd.DataFrame:
    """`MVT_ID_mvt` + the eight chain columns for one month, cached."""
    p = CACHE / f"{month}.parquet"
    if not p.exists():
        raw = D.load_month(month)
        masked, _ = D.mask_like_ranking(raw)
        c = columns(masked, "derived", month)
        c.insert(0, "MVT_ID_mvt",
                 raw.loc[c.index, "MVT_ID_mvt"].to_numpy("float64"))
        p.parent.mkdir(parents=True, exist_ok=True)
        c.reset_index(drop=True).to_parquet(p, index=False)
    return pd.read_parquet(p)


def _chain_ranking() -> pd.DataFrame:
    """The same for the two ranking months, each built from its own rows alone.

    That is `frames.ranking_frame`'s contract and it matters here: every chain
    column is a within-month construction, so building the two months together
    would give a January departure a July neighbour at the same stand.
    """
    p = CACHE / "ranking.parquet"
    if not p.exists():
        r = D.load_ranking()
        out = []
        for m in sorted(r["month"].unique()):
            g = r[r["month"] == m]
            c = columns(g, "derived", str(m))
            c.insert(0, "MVT_ID_mvt", g.loc[c.index, "MVT_ID_mvt"].to_numpy("float64"))
            out.append(c.reset_index(drop=True))
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(out, ignore_index=True).to_parquet(p, index=False)
    return pd.read_parquet(p)


def _attach(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Append the arm's columns to a v2 frame, joined on `MVT_ID_mvt`.

    The two builds walk the same departures in the same order, and that is
    ASSERTED rather than assumed: an order slip here would attach one flight's
    stand history to another and the fit would still finish.
    """
    if not NEW:
        return base
    a = base["MVT_ID_mvt"].to_numpy("float64")
    b = extra["MVT_ID_mvt"].to_numpy("float64")
    out = base.copy()
    if np.array_equal(a, b):
        for c in NEW:
            out[c] = extra[c].to_numpy("float64")
        return out
    idx = pd.Index(b).get_indexer(a)
    if (idx < 0).any():
        raise SystemExit(f"{int((idx < 0).sum())} v2 rows have no chain row")
    for c in NEW:
        out[c] = extra[c].to_numpy("float64")[idx]
    return out


def load_cached(month: str) -> pd.DataFrame:
    return _attach(V2.load_cached(month), _chain_month(month))


def fold_frames(fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = D.FOLDS[fold]
    tr = pd.concat([load_cached(m) for m in f.train_months], ignore_index=True)
    va = load_cached(f.valid_month)
    return tr, va


def all_training() -> pd.DataFrame:
    return pd.concat([load_cached(m) for m in D.MONTHS], ignore_index=True)


def ranking_frame() -> pd.DataFrame:
    return _attach(V2.ranking_frame(), _chain_ranking())


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return V2.feature_columns(frame)


def cat_columns(frame: pd.DataFrame) -> list[str]:
    return V2.cat_columns(frame)
