"""exp-047-queue step 3: the block as an INTERFACE, not a second candidate.

`assignments/queue.md`: *"Write the block as `artifacts/exp-047/block/<month>.parquet`
keyed by `MVT_ID_mvt` with a documented schema and a manifest with hashes, so
surfacefit can take it as an input."*  This file writes exactly that and nothing else:
no model, no candidate, no edit to `exp-046-surfacefit`.

The schema is one row per departure of the month, `MVT_ID_mvt` first, every other
column a float32 named `<source><anchor>_<state>`:

    Fto_*   the ranking file's own clocks read at the row's TAKE-OFF instant
    Fob_*   the ranking file's own clocks read at the row's OFF-BLOCK instant
    Oob_*   the adsb.lol observed clocks read at the observed off-block
    Hob_*   observed where the network heard the aircraft, file otherwise
    *_sc    the same count divided by the airport's receipt rate ON THE HARVESTED
            AIRPORT-DAYS, so a 16 % airport and a 98 % one read on one scale.  The
            denominator is the harvested days and not the month (FALLBACK review
            round 1, blocking 2): a row on a day the harvest has not delivered has no
            clock by construction, so counting it made the constant a function of the
            month's coverage rather than of the airport's receipt.  A row off a
            harvested day carries NaN in every `_sc` column rather than a count scaled
            by a rate measured elsewhere.  `analysis/interface.json` records the rate
            and the harvested-day count per airport for every month.
    route_km, route_km_rel, stand_dist_km   the stand-to-runway route, from OTHER
            aircraft's observed stand positions and open OurAirports runway geometry
    obs_matched, obs_has_clock, reach_class the identity of the observation, so a
            consumer can price the matched and the unmatched populations apart

`state.py`'s docstring defines every state name.  Nothing in the file reads a label,
the row's own ground run, or the row's own taxi span; `code/audit.py` asserts it.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-047-queue/code/emit.py 2025-07 \
        --sources F,O,H,geo --rule <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import base as BS                                            # noqa: E402
import block as BK                                           # noqa: E402
import state as S                                            # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

OUT = ROOT / "artifacts/exp-047/block"
MANIFEST = EXP / "analysis/interface.json"
#: `assignments/queue.md` and `assignments/LAPTOP.md` (Disk): the volume is nearly
#: full.  A block month is tens of megabytes, so the rule that applies is "check `df`
#: before any write over 100 MB": this refuses to start with less than `MIN_FREE_GB`
#: free, and refuses to KEEP a file that turns out to be over `BIG_MB` unless the
#: laptop rule's 8 GB of headroom is there.
MIN_FREE_GB = 0.5
BIG_MB = 100.0
BIG_NEEDS_FREE_GB = 8.0


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


#: month -> the fold whose out-of-fold v11 vector is the honest prediction for it.
#: The `Nf_*` and `No_*` columns are `neighbour's taxi - v11's prediction for the
#: neighbour`, so the interface has to say WHICH v11.  For a 2025 fold month it is the
#: fold-side vector, which never saw that month; for a ranking month it is the served
#: file itself, and a consumer passes it in.
FOLD_OF = {v: k for k, v in BS.MONTH_OF.items()}


#: the SERVED head, for the ranking months: they have no out-of-fold vector, so the
#: `*_n_res_*` columns would be absent and a consumer would see a different schema on
#: the file it actually serves.  `code/census.py` already reads this file and gates it.
HEAD_FILE = "submissions/candidates/exp-035-k2climb-v11member.parquet"
HEAD_SHA = "48dd572d61ff2403f13614ce04bdc3a8fd0dbdb5175bef7eb143650ea520ec52"


#: §7bk: a prediction supplied by path, for a month that has no fold vector and is
#: not a ranking month.  `predictions()` returns None for such a month, which is why
#: 2025-11 first emitted 185 columns with the twelve `*_n_res_*` all-NaN (§7bi
#: addendum).  Additive: unset, every existing call behaves exactly as before.
PRED_FILE: Path | None = None


def predictions(month: str) -> tuple[pd.Series | None, dict]:
    if PRED_FILE is not None:
        v = pd.read_parquet(PRED_FILE)
        col = "y_pred" if "y_pred" in v.columns else D.TARGET
        return (pd.Series(v[col].to_numpy("float64"),
                          index=v["MVT_ID_mvt"].to_numpy("float64")),
                {"source": f"--pred {Path(PRED_FILE).name}",
                 "file": str(Path(PRED_FILE)), "rows": int(len(v)),
                 "sha256": D.sha256(Path(PRED_FILE)),
                 "note": "supplied by path: this month has no out-of-fold vector of "
                         "its own, so the `*_n_res_*` columns are built from the "
                         "prediction named here and are only as good as it is"})
    fold = FOLD_OF.get(month)
    if fold is None and (month == "ranking" or month.startswith("2026-")):
        got = D.sha256(ROOT / HEAD_FILE)
        if got != HEAD_SHA:
            raise SystemExit(f"the board head is not {HEAD_SHA[:12]}…: {got[:12]}…")
        head = pd.read_parquet(ROOT / HEAD_FILE)
        return (pd.Series(head[D.TARGET].to_numpy("float64"),
                          index=head[D.ID].to_numpy("float64")),
                {"source": "v12 (D20), the SERVED head", "file": HEAD_FILE,
                 "rows": int(len(head)), "sha256": got})
    if fold is None:
        return None, {"source": None,
                      "note": "no out-of-fold v11 vector for this month; the "
                              "`*_n_res_*` columns are absent and the `*_n_anom_*` "
                              "columns, which need no prediction, stand in their place"}
    pan = BS.load()[fold]
    return (pd.Series(pan["p11"].to_numpy("float64"),
                      index=pan["MVT_ID_mvt"].to_numpy("float64")),
            {"source": f"fold-side v11 of {fold}",
             "rows": int(len(pan)),
             "rmse_clean": round(float(pan.attrs["rmse_clean"]), 4),
             "sha256": hashlib.sha256(np.ascontiguousarray(
                 pan["p11"].to_numpy("float64")).tobytes()).hexdigest()})


def emit(month: str, sources: tuple[str, ...], require_rule: str | None) -> dict:
    if month == "ranking" or month.startswith("2026-"):
        frame = D.load_ranking(columns=BK.FRAME_COLS)
        if month != "ranking":
            # a RANKING MONTH: the neighbourhood cells are airport x day x window, so a
            # month is a closed set of cells and slicing it changes no column.
            keep = frame["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").eq(month)
            frame = frame[keep.to_numpy()].reset_index(drop=True)
    else:
        raw = D.load_month(month, columns=BK.FRAME_COLS)
        frame, _ = D.mask_like_ranking(raw)
        del raw
    train = None
    if any(s in sources for s in ("O", "H")) or "geo" in sources:
        # ONE MAP IN BOTH LANES (desk, 2026-09-09): `base.prefix_frame` is the nine
        # non-fold 2025 months, departures only, identical to surfacefit's.
        train = BS.prefix_frame()
    pred, prep = predictions(month)
    b, rep = BK.build(frame, train, require_rule=require_rule, sources=sources,
                      pred=pred)
    if "reach_class" in b.columns:
        b["reach_class"] = b["reach_class"].astype(str)
    for c in b.columns:
        if c == "MVT_ID_mvt":
            b[c] = b[c].astype("float64")
        elif b[c].dtype.kind in "fiu":
            b[c] = b[c].astype("float32")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{month}.parquet"
    free = free_gb(OUT.parent if OUT.exists() else ROOT)
    if free < MIN_FREE_GB:
        raise SystemExit(f"only {free:.2f} GB free; not writing {p} "
                         "(assignments/LAPTOP.md, Disk)")
    b.to_parquet(p, index=False)
    mb = p.stat().st_size / 1e6
    if mb > BIG_MB and free < BIG_NEEDS_FREE_GB:
        p.unlink()
        raise SystemExit(f"{p.name} is {mb:.0f} MB with only {free:.2f} GB free; "
                         "removed (assignments/LAPTOP.md, Disk)")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    rec = {"month": month, "path": str(p.relative_to(ROOT)), "rows": int(len(b)),
           "bytes": p.stat().st_size, "sha256": h,
           "columns": [{"name": c, "dtype": str(b[c].dtype)} for c in b.columns],
           "sources": list(sources), "summaries_rule": require_rule,
           "prediction": prep,
           "reach_class_counts": (b["reach_class"].value_counts().to_dict()
                                  if "reach_class" in b.columns else {}),
           "free_gb_after": round(free_gb(OUT), 2), "build": rep}
    print(f"{month}: {len(b):,} rows x {len(b.columns)} cols, "
          f"{p.stat().st_size/1e6:.1f} MB, sha {h[:16]}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("months", nargs="+")
    ap.add_argument("--sources", default="F")
    ap.add_argument("--rule", default=None)
    ap.add_argument("--pred", default=None,
                    help="parquet with MVT_ID_mvt and y_pred, used where a fold "
                         "month's p11 goes (§7bk)")
    a = ap.parse_args()
    global PRED_FILE
    PRED_FILE = Path(a.pred).resolve() if a.pred else None
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"months": {}}
    for m in a.months:
        man["months"][m] = emit(m, tuple(a.sources.split(",")), a.rule)
    man["schema_note"] = __doc__.strip()
    man["state_columns"] = S.COLS
    man["written_by"] = "experiments/exp-047-queue/code/emit.py"
    # NAMED FOR WHAT THEY HASH (desk, 2026-09-08 16:12 UTC, having read `block_sha256`
    # as the hash of the BLOCK FILE and found it disagreed with the parquet on disk).
    # They are the two BUILDERS; each month's own `sha256` is its parquet.  The old
    # keys are kept as aliases so a consumer written against them still resolves.
    # the MATCHER is a third builder and it moves independently of this lane (14:45,
    # 15:02, 16:58 UTC all moved it), so the manifest pins the sha every block was
    # written against.  A consumer that finds it stale knows the class boundary moved.
    man["matcher_block_py_sha256"] = BK.sha_of(
        HERE.parents[1] / "exp-046-surfacefit/code/block.py")
    man["builder_block_py_sha256"] = BK.sha_of(HERE / "block.py")
    man["builder_state_py_sha256"] = BK.sha_of(HERE / "state.py")
    man["block_sha256"] = man["builder_block_py_sha256"]
    man["state_sha256"] = man["builder_state_py_sha256"]
    man["sha256_key_note"] = (
        "`months.<month>.sha256` is the SHA-256 of that month's parquet on disk. "
        "`builder_block_py_sha256` / `builder_state_py_sha256` (and their legacy "
        "aliases `block_sha256` / `state_sha256`) are the SHA-256 of code/block.py "
        "and code/state.py, the files that WROTE it. They are different things and "
        "are not expected to agree.")
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=True, default=str))
    print("manifest", MANIFEST)


if __name__ == "__main__":
    main()
