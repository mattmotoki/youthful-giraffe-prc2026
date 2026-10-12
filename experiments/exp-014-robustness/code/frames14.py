"""Cached frames for the candidate audit: the two frames the shipped blend consumes.

`v2` is exp-008's frozen 54-column frame (`C2-cat` 0.2227 + `L2-hub` 0.3351 of
the blend). `tail` is `prc.tail_features.tail_signals`, what `T-tailv2` (0.4438)
consumes, built per month exactly as its serving path builds it (one build per
month for training, one per ranking month, because `tail_ranking.py` groups the
ranking file by month before predicting).

Nothing here fits anything; it is a cache so the exposure scripts share bytes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "exp-008-diversity"))

import prc.data as D  # noqa: E402
from prc.tail_features import tail_signals  # noqa: E402

CACHE = ROOT / "artifacts" / "exp-014" / "frames"
V2 = ROOT / "artifacts" / "exp-008" / "frames"

#: Columns that are identifiers or bookkeeping, never model inputs.
V2_DROP = ("MVT_ID_mvt", "y", "t", "month_str")


def v2_month(month: str) -> pd.DataFrame:
    return pd.read_parquet(V2 / f"v2-{month}.parquet")


def v2_training() -> pd.DataFrame:
    return pd.concat([v2_month(m) for m in D.MONTHS], ignore_index=True)


def v2_ranking() -> pd.DataFrame:
    return pd.read_parquet(V2 / "v2-ranking.parquet")


def _tail_path(key: str) -> Path:
    return CACHE / f"tail-{key}.parquet"


def tail_month(month: str) -> pd.DataFrame:
    p = _tail_path(month)
    if not p.exists():
        df = D.load_month(month)
        s = tail_signals(df)
        s["MVT_ID_mvt"] = df.loc[s.index, "MVT_ID_mvt"].to_numpy()
        s["month_str"] = month
        p.parent.mkdir(parents=True, exist_ok=True)
        s.reset_index(drop=True).to_parquet(p, index=False)
    return pd.read_parquet(p)


def tail_training() -> pd.DataFrame:
    return pd.concat([tail_month(m) for m in D.MONTHS], ignore_index=True)


def tail_ranking() -> pd.DataFrame:
    p = _tail_path("ranking")
    if not p.exists():
        rk = D.load_ranking()
        out = []
        for m, g in rk.groupby("month", sort=True):
            s = tail_signals(g)
            s["MVT_ID_mvt"] = g.loc[s.index, "MVT_ID_mvt"].to_numpy()
            s["month_str"] = m
            out.append(s.reset_index(drop=True))
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(out, ignore_index=True).to_parquet(p, index=False)
    return pd.read_parquet(p)


def candidate(path: str = "submissions/candidates/exp-012-blend-1.parquet") -> pd.Series:
    """The shipped candidate's predictions, indexed by MVT_ID_mvt."""
    d = pd.read_parquet(ROOT / path)
    return pd.Series(d[D.TARGET].to_numpy("float64"), index=d[D.ID].to_numpy("float64"))


if __name__ == "__main__":
    import time
    for m in D.MONTHS:
        t0 = time.time()
        n = len(tail_month(m))
        print(f"tail {m}: {n} rows in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    print(f"tail ranking: {len(tail_ranking())} rows in {time.time()-t0:.0f}s", flush=True)
