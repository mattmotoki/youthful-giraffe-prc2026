"""The shared measurement packet every screen writes, per the queue's contract.

`QUEUE.md`, "Shared measurement contract": *Primary endpoint: paired ALL-ROW
January/July MSE at the ranking month weights.  Report clean data and December
separately.  Save row and date contributions, counts, exact vector/input hashes
and fit provenance.*  Sol round 1, blocking finding 2, found the first version of
this experiment saving a summary JSON and headlining the CLEAN board.  This file
is the repair: one packet writer, used by all four screens.

* `primary()` is ALL-ROW july_loo/january at exp-016's shares.  `clean` and
  `december` are secondary fields beside it, never the headline.
* `dates()` is the per-date contribution table -- counts, base SSE, arm SSE and
  their difference -- written to `analysis/dates-<screen>-<arm>-<fold>.csv`.
* `provenance()` pins the fold-side v10 vectors by a SHA-256 over their own
  ordered bytes (not by an RMSE, which many vectors share -- Sol round 1,
  blocking finding 1), the rotation files they came from, the month files, the
  library versions and the git commit.
* `boot_blocks()` is the multi-day sensitivity check: the same statistic under a
  moving block bootstrap at 1, 2 and 3 contiguous days.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import sklearn

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import v10 as V  # noqa: E402

ROOT = V.ROOT
FOLDS = V.FOLDS
D = V.D
BOOT = 2000


def sha_bytes(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype="float64").tobytes())
    return h.hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """True when the working tree differs from `git_commit()`.

    Sol round 2, blocking finding e1: a commit hash alone does not pin the code
    that ran when the tree is dirty, so the code files are hashed as well.
    """
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                              str(HERE)], capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except Exception:
        return True


def code_state() -> dict:
    """SHA-256 of every executable file of this experiment, dirty tree or not."""
    return {p.name: sha_file(p) for p in sorted(HERE.glob("*.py"))}


def provenance(panel: dict) -> dict:
    """Every input this experiment's numbers depend on, by hash."""
    vec = {f: sha_bytes(panel[f]["MVT_ID_mvt"].to_numpy("float64"),
                        panel[f]["p10"].to_numpy("float64")) for f in FOLDS}
    src = {}
    for name, tmpl in (("K2", "experiments/exp-027-step/oof/exp-027-k2-k2-hold-{fold}.parquet"),
                       ("refit-bulk", "experiments/exp-027-step/oof/exp-027-k2-refit-bulk-{fold}.parquet"),
                       ("v6-rotation", "experiments/exp-028-window/oof/exp-028-stack5w-w600-{fold}.parquet")):
        for f in FOLDS:
            p = ROOT / tmpl.format(fold=f)
            src[f"{name}:{f}"] = sha_file(p)
    months = {f: sha_file(D.month_file(D.FOLDS[f].valid_month)) for f in FOLDS}
    return {"v10_fold_vector_sha256": vec,
            "uploaded_v10_sha256": V.UPLOADED_SHA,
            "source_rotation_sha256": src,
            "validation_month_file_sha256": months,
            "ranking_file_sha256": sha_file(D.data_dir() / "ranking.parquet"),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "code_sha256": code_state(),
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                         "pyarrow": pa.__version__}}


def primary(panel: dict, preds: dict, fold_note: str = "") -> dict:
    """ALL-ROW january/july board first; clean and december beside it, secondary.

    `preds[fold]` is the arm's full-length prediction vector in panel row order.

    Sol round 2, blocking finding a1: the first version rounded each fold RMSE to
    four decimals BEFORE the weighted board aggregation, so the board gain
    inherited up to 1e-4 s of rounding per fold.  Everything is now aggregated on
    the raw float64 RMSE and rounded only for display; `board_*_exact` keeps the
    unrounded difference beside the rounded one.
    """
    out = {"endpoint": "paired ALL-ROW january/july MSE at exp-016's ranking "
                       "month weights; clean and december secondary",
           "note": fold_note, "all_row": {}, "clean": {}, "counts": {}}
    raw: dict = {"all_row": {}, "clean": {}}
    for f in FOLDS:
        y = panel[f]["y_true"].to_numpy("float64")
        ids = panel[f]["MVT_ID_mvt"].to_numpy("float64")
        b = panel[f]["p10"].to_numpy("float64")
        p = np.asarray(preds[f], "float64")
        raw["all_row"][f] = {"v10": float(D.rmse(y, b)), "arm": float(D.rmse(y, p))}
        raw["clean"][f] = {"v10": float(D.rmse_clean(y, b, ids, f)),
                           "arm": float(D.rmse_clean(y, p, ids, f))}
        for tag in ("all_row", "clean"):
            out[tag][f] = {k: round(v, 4) for k, v in raw[tag][f].items()}
            out[tag][f]["gain"] = round(raw[tag][f]["v10"] - raw[tag][f]["arm"], 4)
        out["all_row"][f]["mse_gain_s2"] = round(
            float(((y - b) ** 2).mean() - ((y - p) ** 2).mean()), 3)
        out["counts"][f] = {"n": int(len(y)),
                            "n_excluded": int(np.isin(ids, list(D.excluded_ids(f))).sum()),
                            "n_changed": int((p != b).sum())}
    for tag in ("all_row", "clean"):
        v10 = V.board({f: raw[tag][f]["v10"] for f in FOLDS})
        arm = V.board({f: raw[tag][f]["arm"] for f in FOLDS})
        out[f"board_{tag}"] = {"v10": round(v10, 4), "arm": round(arm, 4),
                               "gain": round(v10 - arm, 4),
                               "gain_exact": float(v10 - arm)}
    out["december_all_row"] = out["all_row"]["december"]
    out["december_clean"] = out["clean"]["december"]
    return out


def dates(screen: str, arm: str, fold: str, days: np.ndarray, y: np.ndarray,
          base: np.ndarray, new: np.ndarray) -> Path:
    """Per-date contributions: the row-level record behind every headline."""
    df = pd.DataFrame({"date": days, "n": 1,
                       "sse_v10": (y - base) ** 2, "sse_arm": (y - new) ** 2})
    g = df.groupby("date").agg(n=("n", "sum"), sse_v10=("sse_v10", "sum"),
                               sse_arm=("sse_arm", "sum")).reset_index()
    g["delta_sse"] = g["sse_v10"] - g["sse_arm"]
    p = EXP / "analysis" / f"dates-{screen}-{arm}-{fold}.csv"
    g.to_csv(p, index=False, float_format="%.6f")
    return p


def rows(screen: str, arm: str, fold: str, ids: np.ndarray, days: np.ndarray,
         y: np.ndarray, base: np.ndarray, new: np.ndarray) -> Path:
    """The FULL per-row contribution table the contract asks for.

    Sol round 2, blocking finding e1: `dates()` alone is a date aggregate, and the
    shared contract asks for row AND date contributions.  One parquet per screen,
    arm and fold, addressed by its own SHA-256 in the packet.
    """
    df = pd.DataFrame({"MVT_ID_mvt": np.asarray(ids, "float64"),
                       "date": np.asarray(days),
                       "y": np.asarray(y, "float64"),
                       "p_v10": np.asarray(base, "float64"),
                       "p_arm": np.asarray(new, "float64")})
    df["sse_v10"] = (df["y"] - df["p_v10"]) ** 2
    df["sse_arm"] = (df["y"] - df["p_arm"]) ** 2
    df["delta_sse"] = df["sse_v10"] - df["sse_arm"]
    p = EXP / "analysis" / "rows" / f"rows-{screen}-{arm}-{fold}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, compression="zstd")
    return p


def rows_manifest() -> Path:
    """The committed address of the row-contribution parquets.

    `AGENTS.md`: small artifacts in git, large ones as a committed manifest entry
    with a stable address and a verified hash.  The parquets are 64 MB and stay
    out of git; this manifest is what the record cites.
    """
    d = EXP / "analysis" / "rows"
    d.mkdir(parents=True, exist_ok=True)
    (d / ".gitignore").write_text("*.parquet\n")
    out = []
    for f in sorted(d.glob("*.parquet")):
        out.append({"file": f"experiments/exp-039-screens/analysis/rows/{f.name}",
                    "bytes": f.stat().st_size,
                    "rows": int(pd.read_parquet(f, columns=["MVT_ID_mvt"]).shape[0]),
                    "sha256": sha_file(f)})
    p = d / "MANIFEST.json"
    p.write_text(json.dumps(
        {"note": "per-row contributions: MVT_ID_mvt, date, y, p_v10, p_arm, "
                 "sse_v10, sse_arm, delta_sse.  Rebuild by rerunning the screen; "
                 "the hashes below address the files this record was written from.",
         "files": out}, indent=2))
    return p


def boot_blocks(v: np.ndarray, days: np.ndarray, rng, lengths=(1, 2, 3)) -> dict:
    """Moving-block bootstrap of `mean(v)` at 1, 2 and 3 contiguous days."""
    codes, uniq = pd.factorize(pd.Series(days))
    order = np.argsort(uniq.astype(str))
    rank = np.empty(len(uniq), int)
    rank[order] = np.arange(len(uniq))
    day_idx = rank[codes]
    k = len(uniq)
    num = np.bincount(day_idx, weights=v, minlength=k)
    cnt = np.bincount(day_idx, minlength=k).astype("float64")
    out = {"point": round(float(num.sum() / cnt.sum()), 3)}
    for L in lengths:
        nb = int(np.ceil(k / L))
        starts = rng.integers(0, max(k - L + 1, 1), size=(BOOT, nb))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(BOOT, -1)
        idx = np.minimum(idx, k - 1)[:, :k]
        b = num[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        out[f"ci_block_{L}d"] = [round(float(lo), 3), round(float(hi), 3)]
    return out


def boot_blocks_pooled(parts: list, rng, lengths=(1, 2, 3)) -> dict:
    """Moving-block bootstrap pooled over folds, blocks drawn INSIDE each fold.

    Sol round 2, blocking finding a2: the first version concatenated the three
    months into one 93-label sequence and drew contiguous runs from it, so a
    2- or 3-day block could straddle a July/December/January boundary that does
    not exist in time.  Each fold is now resampled on its own day sequence and
    the sampled numerators and counts are pooled.

    `parts` is a list of `(v, days)` pairs, one per fold.
    """
    per = []
    for v, days in parts:
        codes, uniq = pd.factorize(pd.Series(days))
        order = np.argsort(uniq.astype(str))
        rank = np.empty(len(uniq), int)
        rank[order] = np.arange(len(uniq))
        day_idx = rank[codes]
        k = len(uniq)
        per.append((np.bincount(day_idx, weights=v, minlength=k),
                    np.bincount(day_idx, minlength=k).astype("float64")))
    tot_num = sum(float(n.sum()) for n, _ in per)
    tot_cnt = sum(float(c.sum()) for _, c in per)
    out = {"point": round(tot_num / tot_cnt, 3),
           "note": "contiguous blocks drawn inside each fold, never across months"}
    for L in lengths:
        num = np.zeros(BOOT)
        cnt = np.zeros(BOOT)
        for n, c in per:
            k = len(n)
            nb = int(np.ceil(k / L))
            starts = rng.integers(0, max(k - L + 1, 1), size=(BOOT, nb))
            idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(BOOT, -1)
            idx = np.minimum(idx, k - 1)[:, :k]
            num += n[idx].sum(1)
            cnt += c[idx].sum(1)
        b = num / np.maximum(cnt, 1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        out[f"ci_block_{L}d"] = [round(float(lo), 3), round(float(hi), 3)]
    return out


def board_exact(per_fold: dict) -> float:
    """The board from UNROUNDED fold RMSE (Sol round 2, blocking finding a1)."""
    return float(V.board(per_fold))


def write(screen: str, rec: dict, panel: dict) -> Path:
    rec = dict(rec)
    rec["provenance"] = provenance(panel)
    p = EXP / "analysis" / f"packet-{screen}.json"
    p.write_text(json.dumps(rec, indent=2, default=str))
    return p
