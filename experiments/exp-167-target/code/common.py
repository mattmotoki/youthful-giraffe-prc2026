"""Paths, loaders and the fold-form predictions for exp-167 (Mac, read only outside this folder).

Run from the repo root with `.venv/bin/python experiments/exp-167-target/code/<script>.py`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/exp-167-target"
RESULTS = EXP / "results"
RAW = ROOT / "data/raw"
SAVED = ROOT / "data/exp167"            # copies of valid's and clockhead's saved fold predictions
FROZEN = ROOT / "experiments/exp-146-validation/frozen"

ID = "MVT_ID_mvt"
N_RANKING = 344_841
FLOOR = 60.0
DAY = 86_400.0
PH_TOL = 6.0                            # |BLOCK - SCHED| <= 6 s: the placeholder under the +-6 s dither
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
UNSUPPORTED_CLOCK = {("LEMD", 1), ("LIRF", 1), ("LFPG", 1)}     # January clock cells without 2025 rows
W_HEAD = 0.3
#: sha256 of the saved predictions on the workstation (copied 2026-10-02)
SAVED_SHA = {
    "refs.parquet": "a07552f7357aa33266a5fe0138b6733c986ebd177b045b60f1b1cbce0c91fd45",
    "R4Fjjrefit-month-2025-01.parquet": "0095135f56b7f465c38d707ad278e8559f27528fca0cb44d6323b33657a18708",
    "R4Fjjrefit-month-2025-07.parquet": "42dee7eb7a81fc48e4853369f697dfcfa7dc97d4180218980b0d2b9a0a5fd531",
    "HA3-jjrefit.parquet": "6ddfde67187e0555cecae105ac7cae8836ee5abe50d05513fb5bc2784909d319",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def refuse_existing(*paths: Path) -> None:
    hit = [str(p) for p in paths if Path(p).exists()]
    if hit:
        raise SystemExit("write-once: these outputs exist:\n  " + "\n  ".join(hit))


def write_json(path: Path, rec: dict) -> None:
    refuse_existing(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")


def md(df: pd.DataFrame, index: bool = True, digits: int = 1) -> str:
    """A markdown table (pandas' to_markdown needs a package the shared environment lacks)."""
    d = df.reset_index() if index else df

    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:,.{digits}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    rows = ["| " + " | ".join(str(c) for c in d.columns) + " |", "|" + "---|" * len(d.columns)]
    rows += ["| " + " | ".join(cell(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def saved(name: str) -> Path:
    p = SAVED / name
    if sha(p) != SAVED_SHA[name]:
        raise SystemExit(f"{p}: not the saved file {SAVED_SHA[name][:12]}")
    return p


def raw_2025(columns: list[str] | None = None, months=range(1, 13)) -> pd.DataFrame:
    """All rows (both phases) of the 2025 month files, with `apt` = the airport of the movement."""
    files = sorted(RAW.glob("training_2025-*.parquet"))
    files = [f for f in files if int(f.name[14:16]) in set(months)]
    t = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    t["apt"] = np.where(t.PHASE_mvt == "DEP", t.ADEP_mvt, t.ADES_mvt)
    return t


def secs(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds()


def add_regime(d: pd.DataFrame) -> pd.DataFrame:
    """Label regime of 2025 departure rows (reads BLOCK and the label: training side only).

    P: placeholder, |BLOCK - SCHED| <= 6 s.  R: a whole day in the label (y - 900 rounds to >= 1 day).
    Z: BLOCK one to three hours early against SCHED with a take-off near SCHED (a zone-shift form).
    O: the rest (a real off-block).
    """
    d = d.copy()
    d["y"] = d.TAXITIME_SEC_mvt.astype("float64")
    d["bs"] = secs(d.BLOCK_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
    d["ms"] = secs(d.MVT_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
    d["nonm"] = d.AOBT_3_flt.isna()
    d["noac"] = d.AIRCRAFT_TYPE_mvt.isna()
    d["ph"] = d.bs.abs() <= PH_TOL
    d["ydays"] = np.round((d.y - 900.0) / DAY)
    d["ku"] = ((d.MVT_TIME_UTC_mvt - pd.Timedelta(minutes=15)).dt.normalize()
               - d.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days
    d["regime"] = np.where(d.ph, "P", np.where(d.ydays >= 1, "R", "O"))
    return d


# ------------------------------------------------------------------ fold rows and the served forms
def fold_rows() -> pd.DataFrame:
    """The 344,419 January and July 2025 rows of the frozen table, with cell weights.

    `w26`: W_h / n_h for a supported cell (the row's share of G's weight), 0 otherwise.
    `w25`: 1 / (number of fold rows).
    """
    r = pd.read_parquet(FROZEN / "rows.parquet")
    r = r[r.month.isin((1, 7))].copy()
    r[ID] = r[ID].astype("int64")
    cells = pd.read_csv(FROZEN / "cells.csv").set_index(["month", "airport", "cls"])
    n = r.groupby(["month", "airport", "cls"]).size()
    key = list(zip(r.month.astype(int), r.airport.astype(str), r.cls.astype(str)))
    N = np.array([float(cells.N.get(k, 0.0)) for k in key])
    nh = np.array([float(n.get(k, 0)) for k in key])
    r["w26"] = np.where(nh >= 2, N / N_RANKING / np.maximum(nh, 1), 0.0)
    r["w25"] = 1.0 / len(r)
    return r.reset_index(drop=True)


def served_forms(r: pd.DataFrame) -> pd.DataFrame:
    """Local F and local v47 form on the fold rows (the rule of exp-159 `entry.py`, re-written).

    F = v32 on the kept rows (noac, and ordinary rows without AOBT_3), else 0.5 x R4 + 0.5 x v32
    (R4 = valid's q01 and q07 models on the refit members).  v47 form = F, and
    max(60, 0.7 F + 0.3 max(60, head)) on clock rows of supported cells.
    """
    ids = r[ID].to_numpy("int64")
    refs = pd.read_parquet(saved("refs.parquet")).set_index(ID)
    refs.index = refs.index.astype("int64")
    v32 = refs.v32.reindex(ids).to_numpy("float64")
    parts = []
    for m in ("01", "07"):
        f = pd.read_parquet(saved(f"R4Fjjrefit-month-2025-{m}.parquet"))
        parts.append(pd.Series(f["pred_Sb"].to_numpy("float64"), index=f[ID].astype("int64").to_numpy()))
    s = pd.concat(parts)
    if s.index.duplicated().any() or set(s.index) != set(ids):
        raise SystemExit("the refit rows do not cover each January and July row once")
    cls = r.cls.astype(str).to_numpy()
    keep = (cls == "noac") | ((cls == "ordinary") & ~r.aobt.to_numpy(bool))
    kno = np.where(keep, v32, s.reindex(ids).to_numpy("float64"))
    f_loc = 0.5 * kno + 0.5 * v32
    h = pd.read_parquet(saved("HA3-jjrefit.parquet")).set_index(ID)["pred_HA3"]
    h.index = h.index.astype("int64")
    head = h.reindex(ids).to_numpy("float64")
    sup = np.array([(a, int(m)) not in UNSUPPORTED_CLOCK for a, m in zip(r.airport.astype(str), r.month)])
    rule = (cls == "clock") & sup
    if not np.isfinite(head[rule]).all():
        raise SystemExit("a local clock row has no head value")
    v47 = f_loc.copy()
    v47[rule] = np.maximum(FLOOR, (1 - W_HEAD) * f_loc[rule] + W_HEAD * np.maximum(FLOOR, head[rule]))
    out = r.copy()
    out["v32"], out["F"], out["v47"], out["keep"] = v32, f_loc, v47, keep
    return out


def fold_labels(r: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Raw columns of the fold rows' departures, selected by id across the month files."""
    t = raw_2025(columns=sorted(set(columns) | {ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt"}), months=(1, 2, 7, 8))
    t = t[t.PHASE_mvt == "DEP"]
    t[ID] = t[ID].astype("int64")
    t = t.set_index(ID).reindex(r[ID].to_numpy("int64"))
    if t.PHASE_mvt.isna().any():
        raise SystemExit("a fold row has no raw departure")
    return t.reset_index()


if __name__ == "__main__":
    sys.exit("a module, not a script")
