"""Shared helpers for exp-003 anatomy."""
from pathlib import Path
import numpy as np
import pandas as pd
import prc.data as d

CACHE = d.ROOT / "artifacts" / "exp-003-anatomy" / "mvt_all.parquet"
ANA = d.ROOT / "experiments" / "exp-003-anatomy" / "analysis"
ANA.mkdir(parents=True, exist_ok=True)

TRAIN_MONTHS = {"july": [f"2025-{m:02d}" for m in range(1, 7)],
                "december": [f"2025-{m:02d}" for m in range(1, 12)]}
VALID_MONTH = {"july": "2025-07", "december": "2025-12"}


def load_all(columns=None) -> pd.DataFrame:
    return pd.read_parquet(CACHE, columns=columns)


def airport(df: pd.DataFrame) -> pd.Series:
    """Airport of the movement: ADEP for departures, ADES for arrivals."""
    return np.where(df["PHASE_mvt"].astype(str) == "DEP",
                    df["ADEP_mvt"].astype(str), df["ADES_mvt"].astype(str))


def q(x, ps=(1, 5, 25, 50, 75, 95, 99)):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {f"p{p}": np.nan for p in ps}
    v = np.percentile(x, ps)
    return {f"p{p}": float(vi) for p, vi in zip(ps, v)}


def describe(g: pd.Series) -> dict:
    x = g.to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    out = {"n": len(x)}
    if len(x):
        out |= {"mean": x.mean(), "median": float(np.median(x)), "std": x.std(),
                "mad": float(np.median(np.abs(x - np.median(x))))}
        out |= q(x)
        out["rmse_about_med"] = float(np.sqrt(np.mean((x - np.median(x)) ** 2)))
        out["rmse_about_mean"] = float(x.std())
    return out


def md_table(df: pd.DataFrame, floatfmt="{:.1f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.to_numpy())]
    return "\n".join([head, sep] + rows)


def write(name: str, text: str):
    p = ANA / name
    p.write_text(text)
    print("wrote", p, flush=True)
