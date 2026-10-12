"""exp-047-queue: the EXTENSION-MONTH fitting panel (§7ca iv), and nothing else.

`RESULTS.md` §7ca iv registered ONE change to v17's arm: *"the fitting panel ... the
same relation, the same hold, the same cross-fit, on a panel three times the size --
because every hold above was an interval width, and interval width falls with the
panel."*  `code/arms.py` could not express it: `ROTATION` names folds, and a fold panel
comes from `base.load()`, which knows the three fold months only.  This file is the
loader that was missing.  It is a NEW file: `code/arms.py` gains one default-empty
option and every existing invocation of it takes the same path it took before.

**What an extension month is.**  A 2025 month that is not a fold, whose block was built
by `code/emit.py` and verified by `work/block_verify.py`, and whose head is the
six-member `base6` stack fitted without it.  The target is that head's residual, exactly
as `exp-046-surfacefit`'s extension uses it, and exactly the vector the block's twelve
`{Nf_,No_}_n_res_*` columns were built against -- `work/month_chain.sh` passes
`--pred oof/base6--<month>.parquet` to `emit.py`, so the head inside the features and
the head under the target are the SAME head.  2025-05 and 2025-09 stay out (recorded rule).

**Why the block is rebuilt in memory and not read off the parquet.**  The emitted
parquet is float32 (`code/emit.py`: *"every other column a float32"*), and a float32
block moves a ridge arm by up to 9.5 s^2 -- measured in this lane, and the reason the
fold side has always been built in memory.  A fitting panel is not served bytes, but a
panel that differs from the fold side in dtype would put the difference in the fit and
not in the record.  So each month is rebuilt through the same `block.build` call
`emit.py` makes, from the same masked frame, the same prefix map and the same head, and
the parquet is used for ONE thing: as the CACHE KEY.  `reconcile()` reads the registered
parquet's ids and checks that the in-memory build carries the same rows in the same
order and the same columns, and the run records the parquet's sha256 -- so the record
names bytes on disk that a later reader can check, while the fit sees float64.

**Cost.**  About 16 s a month at `--sources F,O,H,N,geo`, so the seven months are ~2 min
in front of a fit that runs for longer than that; and only the arm's own columns and the
target are kept, so the whole panel is about 245 MB of float64 rather than 1.8 GB.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import base as BS                                            # noqa: E402
import block as BK                                           # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

#: the seven months whose block is built, verified and carries a `base6` head.  recorded rule
#: closed the programme here: *"2025-05 and 2025-09 stay unbuilt"*, and they are absent
#: from this list for that reason and not because their files are missing.
MONTHS = ["2025-02", "2025-03", "2025-04", "2025-06", "2025-08", "2025-10", "2025-11"]

VERIFY = EXP / "analysis" / "block-{}-verify.json"
HEAD = EXP / "oof" / "base6--{}.parquet"


def registered(month: str) -> dict:
    """The month's block as the record has it: path, rows, columns, sha256.

    This is the cache key.  A month whose verify record is absent, or did not PASS, is
    not a fitting month: the run stops rather than fitting on a block no section names.
    """
    p = Path(str(VERIFY).format(month))
    if not p.exists():
        raise SystemExit(f"{month}: no {p.name}; the block is not on the record")
    rec = json.loads(p.read_text())
    if not rec.get("PASS"):
        raise SystemExit(f"{month}: {p.name} did not PASS")
    new = dict(rec["new"])
    new["verify"] = str(p.relative_to(ROOT))
    return new


def head(month: str) -> tuple[pd.Series, pd.Series, dict]:
    """`base6`'s prediction and the label, keyed by `MVT_ID_mvt`."""
    p = Path(str(HEAD).format(month))
    if not p.exists():
        raise SystemExit(f"{month}: no {p.name}; the month has no base6 head")
    v = pd.read_parquet(p)
    idx = v["MVT_ID_mvt"].to_numpy("float64")
    rep = {"file": str(p.relative_to(ROOT)), "rows": int(len(v)),
           "sha256": D.sha256(p),
           "rmse": round(float(np.sqrt(np.mean(
               (v["y_true"].to_numpy("float64")
                - v["y_pred"].to_numpy("float64")) ** 2))), 4)}
    return (pd.Series(v["y_pred"].to_numpy("float64"), index=idx),
            pd.Series(v["y_true"].to_numpy("float64"), index=idx), rep)


def month_panel(month: str, sources, require_rule, cols, rcol: str = "r12"):
    """One extension month as a fitting frame: `cols`, the target, and the cell keys.

    The build is `code/emit.py::emit` with the float32 cast removed -- the same masked
    frame, the same `base.prefix_frame()` map, the same `block.build` call, and the same
    head, supplied here from the parquet `work/month_chain.sh` passed to `emit.py`.
    """
    t0 = time.time()
    reg = registered(month)
    pred, ytrue, hrep = head(month)
    raw = D.load_month(month, columns=BK.FRAME_COLS)
    frame, _ = D.mask_like_ranking(raw)
    del raw
    train = BS.prefix_frame() if (set(sources) & {"O", "H", "N", "geo"}) else None
    b, brep = BK.build(frame, train, require_rule=require_rule, sources=sources,
                       pred=pred)
    ids = b["MVT_ID_mvt"].to_numpy("float64")
    miss = [c for c in cols if c not in b.columns]
    if miss:
        raise SystemExit(f"{month}: the block lacks {miss[:4]} of the arm's columns")
    # EPOCH SECONDS, not microseconds.  `MVT_TIME_UTC_mvt` is `datetime64[us, UTC]`
    # in these files, so `.astype("int64")` is microseconds; casting the dtype to
    # seconds FIRST is the conversion that does not depend on the file's resolution.
    fid = frame["MVT_ID_mvt"].to_numpy("float64")
    secs = (pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True)
            .dt.tz_convert("UTC").dt.tz_localize(None)
            .astype("datetime64[s]").astype("int64").to_numpy())
    tmap = pd.Series(secs.astype("float64"), index=fid)
    amap = pd.Series(frame["ADEP_mvt"].astype(str).to_numpy(), index=fid)
    out = b[["MVT_ID_mvt"] + list(cols)].copy()
    out["y_true"] = pd.Series(ids).map(ytrue).to_numpy("float64")
    out["p12"] = pd.Series(ids).map(pred).to_numpy("float64")
    out[rcol] = out["y_true"].to_numpy("float64") - out["p12"].to_numpy("float64")
    out["t"] = pd.Series(ids).map(tmap).to_numpy("float64")
    out["apt"] = pd.Series(ids).map(amap).to_numpy()
    out["month"] = month
    out["day"] = BS.V.day(out["t"].to_numpy("float64"))
    out["bin"] = np.floor(out["t"].to_numpy("float64") / BS.BIN).astype("int64")
    for c in ("obs_has_clock", "obs_matched", "No_n_n_cell"):
        if c in b.columns and c not in out.columns:
            out[c] = b[c].to_numpy()
    if not np.isfinite(out["t"].to_numpy("float64")).all():
        raise SystemExit(f"{month}: rows of the block are absent from the frame")
    bad = int((~np.isfinite(out[rcol].to_numpy("float64"))).sum())
    if bad:
        raise SystemExit(f"{month}: {bad} rows have no base6 residual")
    rep = {"month": month, "rows": int(len(out)), "seconds": round(time.time() - t0, 1),
           "registered_block": {k: reg[k] for k in ("path", "rows", "columns",
                                                    "sha256")},
           "head": hrep, "summary_days": brep.get("summary_days"),
           "residual_rms": round(float(np.sqrt(np.mean(
               out[rcol].to_numpy("float64") ** 2))), 4)}
    rep["reconcile"] = reconcile(out, reg, cols)
    return out.reset_index(drop=True), rep


def reconcile(out: pd.DataFrame, reg: dict, cols) -> dict:
    """The in-memory build against the parquet the record names -- the cache key check.

    Rows, order and the arm's columns.  The values are compared at float32 tolerance
    because that is exactly the difference the two carry: the parquet was cast on the
    way out and this frame was not.
    """
    p = ROOT / reg["path"]
    got = pd.read_parquet(p, columns=["MVT_ID_mvt"] + list(cols))
    same_ids = bool(len(got) == len(out) and np.array_equal(
        got["MVT_ID_mvt"].to_numpy("float64"), out["MVT_ID_mvt"].to_numpy("float64")))
    worst, worst_col = 0.0, None
    if same_ids:
        for c in cols:
            a = out[c].to_numpy("float64")
            bq = got[c].to_numpy("float64")
            m = np.isfinite(a) & np.isfinite(bq)
            if not m.any():
                continue
            scale = max(1.0, float(np.max(np.abs(a[m]))))
            d = float(np.max(np.abs(a[m] - bq[m]))) / scale
            if d > worst:
                worst, worst_col = d, c
            if (np.isfinite(a) != np.isfinite(bq)).any():
                raise SystemExit(f"{reg['path']}: {c} differs in WHICH rows are null")
    if not same_ids:
        raise SystemExit(f"{reg['path']}: the in-memory build does not carry the "
                         "registered rows in the registered order")
    if worst > 1e-5:
        raise SystemExit(f"{reg['path']}: {worst_col} differs by {worst:.3g} relative, "
                         "which is more than a float32 round trip")
    return {"ids_identical_to_the_registered_parquet": same_ids,
            "worst_relative_column_gap": float(f"{worst:.3g}"),
            "worst_column": worst_col,
            "tolerance": 1e-5,
            "note": "the parquet is float32 and this frame is float64; the gap below "
                    "is the round trip and nothing else"}


def panel(months, sources, require_rule, cols, rcol: str = "r12"):
    """The extension months as one fitting frame, in the registered order."""
    frames, reps = [], []
    for m in months:
        d, r = month_panel(m, sources, require_rule, cols, rcol=rcol)
        print(f"[panel] {m}: {len(d):,} rows, {r['seconds']} s, "
              f"residual rms {r['residual_rms']}, "
              f"sha {r['registered_block']['sha256'][:12]}", flush=True)
        frames.append(d)
        reps.append(r)
    out = pd.concat(frames, ignore_index=True)
    return out, {"months": reps, "rows": int(len(out)),
                 "months_named": list(months),
                 "excluded_by_D25": ["2025-05", "2025-09"],
                 "target": "the six-member base6 residual, the head the block's "
                           "`*_n_res_*` columns were built against"}


def cell_keys(d: pd.DataFrame) -> np.ndarray:
    """`<airport>|<absolute 600 s window>` -- the shipped grid's cell, as a key.

    §7ca ii v: a cell is a window in ABSOLUTE UTC, 86,400 s is 144 whole 600 s bins, so
    no cell straddles a month and a fold is a month.  This makes the claim checkable
    instead of argued.
    """
    return (pd.Series(d["apt"].astype(str).to_numpy()) + "|"
            + pd.Series(np.floor(d["t"].to_numpy("float64") / BS.BIN)
                        .astype("int64")).astype(str)).to_numpy()


def assert_disjoint(scored: pd.DataFrame, fit: pd.DataFrame, name: str) -> dict:
    """The pre-fit assertion of §7ca ii v: the scored fold shares NO cell with the fit.

    *"the arm run asserts, before it prices anything, that the held-out fold shares zero
    cells with its fitting folds ... A run that cannot assert both stops and reports
    rather than serving."*
    """
    a = set(cell_keys(scored).tolist())
    b = set(cell_keys(fit).tolist())
    inter = a & b
    rep = {"scored": name, "scored_cells": len(a), "fitting_cells": len(b),
           "shared_cells": len(inter),
           "example_shared": sorted(inter)[:5]}
    if inter:
        raise SystemExit(f"PRE-FIT ASSERTION FAILED: {name} shares {len(inter)} cells "
                         f"with its fitting rows, e.g. {sorted(inter)[:3]}")
    return rep


def assert_columns(cols, registered_24) -> dict:
    """The other half of §7ca ii v: every fitted column is one of the registered 24."""
    extra = sorted(set(cols) - set(registered_24))
    missing = sorted(set(registered_24) - set(cols))
    rep = {"fitted": len(cols), "registered": len(registered_24),
           "not_registered": extra, "registered_but_absent": missing}
    if extra or missing:
        raise SystemExit(f"PRE-FIT ASSERTION FAILED: the arm fits {len(cols)} columns, "
                         f"{len(extra)} of them not among the registered "
                         f"{len(registered_24)} ({extra[:4]}), "
                         f"{len(missing)} registered ones absent ({missing[:4]})")
    return rep


#: §7ca iv, quoted: the 24 columns of v17's arm N, from `analysis/serve-lit.json`.
REGISTERED_24 = [f"{p}_n_{s}" for p in ("Nf", "No") for s in (
    "anom_cell", "anom_rwy_30m", "anom_rwy_cell", "n_cell", "n_rwy_30m", "n_rwy_cell",
    "res_apt_30m", "res_cell", "res_rwy_30m", "res_rwy_60m", "res_rwy_cell",
    "res_rwy_minus_apt")]


def digest(reps: dict) -> str:
    """One sha over the cache keys, so a run names its panel in a single string."""
    keys = [f"{m['month']}:{m['registered_block']['sha256']}:{m['head']['sha256']}"
            for m in reps.get("months", [])]
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()
