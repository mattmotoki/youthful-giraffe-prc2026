"""Frozen arithmetic and fail-closed helpers for exp-087."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = ("july_loo", "december", "january")
WEIGHTS = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621,
           "december": 0.0}
AMPLITUDE = 0.5
FINAL_FLOOR = 60.0
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 870509
OUTPUT_FIELDS = ("fold", "population", "rows", "changed_rows", "rms_movement",
                 "max_movement", "rmse_comparator", "rmse_arm", "gain", "A", "B",
                 "A_over_B", "ci95_low", "ci95_high")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_id_bytes(values) -> tuple[str, bytes]:
    a = np.asarray(values)
    if a.dtype != np.dtype("float64"):
        raise TypeError(f"ID dtype must be float64, got {a.dtype}")
    if not np.isfinite(a).all() or len(np.unique(a)) != len(a):
        raise ValueError("IDs must be finite and unique")
    return "float64", np.ascontiguousarray(a.astype("<f8", copy=False)).tobytes()


def validate_numeric(*arrays, nonempty=True) -> list[np.ndarray]:
    out=[]
    for value in arrays:
        a=np.asarray(value)
        if not np.issubdtype(a.dtype,np.number): raise TypeError("numeric input required")
        if nonempty and a.size == 0: raise ValueError("empty input")
        if not np.isfinite(a).all(): raise ValueError("non-finite input")
        out.append(a.astype("float64",copy=False))
    return out


def atomic_json(path: Path, value: dict, *, exclusive=False) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive and path.exists(): raise FileExistsError(path)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w") as stream:
            json.dump(value,stream,indent=2,allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if exclusive and path.exists(): raise FileExistsError(path)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def id_record(path: Path) -> dict:
    ids = pd.read_parquet(path, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
    dtype, raw = canonical_id_bytes(ids)
    return {"rows": len(ids), "ordered_id_dtype": dtype,
            "ordered_id_sha256": hashlib.sha256(raw).hexdigest()}


def assert_record(path: Path, expected: dict) -> None:
    if sha256_file(path) != expected["sha256"]:
        raise RuntimeError(f"source hash mismatch: {path}")
    if "rows" in expected:
        got = id_record(path)
        for key in ("rows", "ordered_id_dtype", "ordered_id_sha256"):
            if got[key] != expected[key]:
                raise RuntimeError(f"source ID binding mismatch: {path}: {key}")


def half_update(comparator, full_update) -> np.ndarray:
    """Apply exactly half of the unchanged update, then the final floor."""
    base, update = validate_numeric(comparator,full_update)
    if base.shape != update.shape:
        raise ValueError("shape mismatch")
    return np.maximum(base + AMPLITUDE * update, FINAL_FLOOR)


def fold_stats(y, comparator, arm, day, population="all") -> dict:
    y,base,pred=validate_numeric(y,comparator,arm)
    days=np.asarray(day)
    if days.size == 0 or pd.isna(days).any(): raise ValueError("invalid day cluster")
    if not (y.shape == base.shape == pred.shape == days.shape):
        raise ValueError("unaligned score inputs")
    d = pred - base
    r = y - base
    A = float(np.mean(r * d)); B = float(np.mean(d * d))
    rb = float(np.sqrt(np.mean(r * r)))
    ra = float(np.sqrt(np.mean((y - pred) ** 2)))
    lo, hi = day_bootstrap(y, base, pred, days)
    return {"fold": None, "population": population, "rows": len(y),
            "changed_rows": int(np.count_nonzero(d)),
            "rms_movement": float(np.sqrt(B)),
            "max_movement": float(np.max(np.abs(d))) if len(d) else 0.0,
            "rmse_comparator": rb, "rmse_arm": ra, "gain": rb - ra,
            "A": A, "B": B, "A_over_B": A / B if B > 0 else None,
            "ci95_low": lo, "ci95_high": hi}


def day_bootstrap(y, base, arm, days, *, draws=BOOTSTRAP_DRAWS,
                  seed=BOOTSTRAP_SEED) -> tuple[float, float]:
    if not isinstance(draws,int) or isinstance(draws,bool) or draws <= 0: raise ValueError("positive integer draws required")
    y,base,arm=validate_numeric(y,base,arm); days=np.asarray(days)
    if not (y.shape==base.shape==arm.shape==days.shape) or pd.isna(days).any(): raise ValueError("invalid bootstrap inputs")
    frame = pd.DataFrame({"y": y, "base": base, "arm": arm, "day": days})
    groups = [g for _, g in frame.groupby("day", sort=True)]
    if not groups:
        raise ValueError("no day clusters")
    rng = np.random.default_rng(seed)
    gains = np.empty(draws)
    for i in range(draws):
        picks = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[j] for j in picks], ignore_index=True)
        gains[i] = math.sqrt(np.mean((sample.y - sample.base) ** 2)) - \
            math.sqrt(np.mean((sample.y - sample.arm) ** 2))
    return tuple(float(v) for v in np.quantile(gains, (0.025, 0.975)))


def board_record(stats: dict[str, dict]) -> dict:
    if set(stats)!=set(FOLDS): raise ValueError("all folds required")
    for rec in stats.values():
        validate_numeric([rec[k] for k in ("rmse_comparator","rmse_arm","max_movement","A","B")])
    mse0 = sum(WEIGHTS[f] * stats[f]["rmse_comparator"] ** 2 for f in FOLDS)
    mse1 = sum(WEIGHTS[f] * stats[f]["rmse_arm"] ** 2 for f in FOLDS)
    A=sum(WEIGHTS[f]*stats[f]["A"] for f in FOLDS)
    B=sum(WEIGHTS[f]*stats[f]["B"] for f in FOLDS)
    return {"population": "all", "rows": sum(stats[f]["rows"] for f in FOLDS),
            "changed_rows": sum(stats[f]["changed_rows"] for f in FOLDS),
            "rms_movement": math.sqrt(B),
            "max_movement": max(stats[f]["max_movement"] for f in FOLDS),
            "rmse_comparator": math.sqrt(mse0), "rmse_arm": math.sqrt(mse1),
            "gain": math.sqrt(mse0)-math.sqrt(mse1), "A": A, "B": B,
            "A_over_B": A/B if B>0 else None}


def board_bootstrap(parts: dict, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    if set(parts)!=set(FOLDS): raise ValueError("all folds required")
    if not isinstance(draws,int) or isinstance(draws,bool) or draws<=0: raise ValueError("positive integer draws required")
    rng=np.random.default_rng(seed); e0=np.zeros(draws); e1=np.zeros(draws)
    for fold in ("july_loo","january"):
        y,base,arm,days=(np.asarray(x) for x in parts[fold]); groups=[]
        frame=pd.DataFrame({"y":y,"base":base,"arm":arm,"day":days})
        groups=[g for _,g in frame.groupby("day",sort=True)]
        for i in range(draws):
            sample=pd.concat([groups[j] for j in rng.integers(0,len(groups),len(groups))],ignore_index=True)
            e0[i]+=WEIGHTS[fold]*np.mean((sample.y-sample.base)**2)
            e1[i]+=WEIGHTS[fold]*np.mean((sample.y-sample.arm)**2)
    gains=np.sqrt(e0)-np.sqrt(e1)
    return tuple(float(v) for v in np.quantile(gains,(.025,.975)))


def status(board_gain: float, raw_points: dict[str, float], raw_lows: dict[str, float]) -> str:
    validate_numeric([board_gain],list(raw_points.values()),list(raw_lows.values()))
    if set(raw_points)!=set(FOLDS) or set(raw_lows)!=set(FOLDS): raise ValueError("all fold gates required")
    if board_gain <= 0 or any(raw_points[f] <= 0 for f in FOLDS):
        return "FAIL"
    if board_gain >= 0.10 and all(raw_lows[f] > 0 for f in FOLDS):
        return "ADVANCE"
    return "MAYBE"


def write_json_once(path: Path, value: dict) -> None:
    atomic_json(path,value,exclusive=True)
