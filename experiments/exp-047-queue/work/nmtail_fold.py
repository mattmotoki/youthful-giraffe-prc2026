"""§7bj addendum: the N1-nmtail smoke on november_loo, cache inside this lane.

The desk: smoke N1-nmtail with `exp-010-nmtail/code/run_folds.py --folds
november_loo`, build its reocc cache under THIS lane and point the producer at
it, never a write into exp-010, capped 20 min CPU.

**The cache is built here and the producer is pointed at it by injection.**
`run_folds.REOCC` is a module constant, so `REOCC = artifacts/exp-047/reocc` is
set after import and `exp-010`'s own cache directory is never created or read.
The `log` path is a parameter of `run()`, so exp-010's `climb.log` is untouched
as well.

**One thing could not be honoured and is recorded rather than worked around.**
`run()` builds its output path as a LOCAL variable --
`OOF = d.ROOT / "experiments/exp-010-nmtail/oof"` inside the function body, not a
module constant -- so it cannot be injected the way `REOCC` and `log` can. The
run therefore writes ONE `_q-`prefixed parquet into that already-existing
directory, which this script moves into `oof/` here and deletes there. Nothing
of exp-010's record is read, changed or left behind; `git status` on exp-010 is
clean afterwards. Redirecting it properly needs a one-line edit in
`run_folds.py`, which is shared code and out of scope.

The reocc builder is `experiments/exp-010-nmtail/code/build_reocc.py`'s `one()`
**verbatim** -- it is five lines and the module hardcodes its own output
directory at import, so importing it would create `artifacts/exp-010-nmtail/`.
If `build_reocc.one` changes, this copy must be re-synced.

    OMP_NUM_THREADS=2 PRC_THREADS=2 .venv/bin/python work/nmtail_fold.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp-010-nmtail/code"))
import prc.data as d                                         # noqa: E402
from prc.nmtail import reoccupation_signals                  # noqa: E402

sys.path.insert(0, str(HERE))
import lane                                                  # noqa: E402
lane.install()

REOCC = ROOT / "artifacts/exp-047/reocc"
OOF = EXP / "oof"
SUF = "" if lane.MONTH == "2025-11" else f"-{lane.MONTH}"
OUT = EXP / f"analysis/nmtail-fold{SUF}.json"
LOG = EXP / f"logs/fold-N1-nmtail-climb{SUF}.log"
BLOCK = ROOT / f"artifacts/exp-047/block/{lane.MONTH}.parquet"
FOLD = lane.FOLD
MODE = "p"
NAME = "_q-nmtail"
CONTRACT = ["MVT_ID_mvt", "fold", "y_true", "y_pred"]
CAP_S = 20 * 60


def one(frame: pd.DataFrame) -> pd.DataFrame:
    """`build_reocc.one`, verbatim."""
    masked, _ = d.mask_like_ranking(frame)
    R = reoccupation_signals(masked)
    R = R.reset_index(drop=True)
    R["MVT_ID_mvt"] = masked.loc[masked["PHASE_mvt"] == "DEP", "MVT_ID_mvt"].to_numpy()
    return R


def build_cache() -> dict:
    REOCC.mkdir(parents=True, exist_ok=True)
    t0, built = time.time(), []
    for m in d.MONTHS:
        p = REOCC / f"reocc_{m}.parquet"
        if p.exists():
            continue
        one(d.load_month(m)).to_parquet(p, index=False)
        built.append(m)
        print(f"  reocc {m} ({time.time()-t0:.0f}s)", flush=True)
    return {"dir": str(REOCC.relative_to(ROOT)), "months_built": built,
            "wall_s": round(time.time() - t0, 1),
            "bytes": sum(q.stat().st_size for q in REOCC.glob("*.parquet"))}


def main() -> None:
    OOF.mkdir(parents=True, exist_ok=True)
    print("=== reocc cache", flush=True)
    cache = build_cache()
    print(f"    {cache['wall_s']/60:.1f} min, {cache['bytes']/1e6:.1f} MB, "
          f"{len(cache['months_built'])} months built", flush=True)

    import run_folds as RF
    RF.REOCC = REOCC                     # the injection: exp-010's cache is never read
    print(f"=== N1-nmtail {FOLD} mode={MODE} (cap {CAP_S/60:.0f} min)", flush=True)
    produced = ROOT / "experiments/exp-010-nmtail/oof" / f"{NAME}-{MODE}_{FOLD}.parquet"
    t0 = time.time()
    res = RF.run(FOLD, MODE, f"{NAME}-{MODE}", LOG, None, None, None)
    dt = time.time() - t0

    rec = {"what": "§7bj addendum: N1-nmtail smoked on november_loo",
           "reocc_cache": cache, "fold": FOLD, "mode": MODE,
           "wall_s": round(dt, 1), "wall_min": round(dt / 60, 1),
           "cap_s": CAP_S, "hit_the_cap": dt > CAP_S,
           "producer": "experiments/exp-010-nmtail/code/run_folds.py (run(), injected REOCC)",
           "config": res.get("config") if isinstance(res, dict) else None}

    if produced.exists():
        v = pd.read_parquet(produced)
        v = v[[c for c in CONTRACT if c in v.columns]].copy()
        v["fold"] = FOLD
        v = v[CONTRACT]
        dst = OOF / f"N1-nmtail--{FOLD}.parquet"
        v.to_parquet(dst, index=False)
        produced.unlink()                # nothing is left in exp-010
        blk = pd.read_parquet(BLOCK, columns=["MVT_ID_mvt"])
        import prc.blend as B
        ids_new = set(v["MVT_ID_mvt"].astype("float64"))
        ids_blk = set(blk["MVT_ID_mvt"].astype("float64"))
        try:
            frame, _ = B.stack_members({"N1-nmtail": str(dst)}, ROOT, fold=FOLD)
            ok, lost = True, int(len(v) - len(frame))
        except Exception as e:                               # noqa: BLE001
            ok, lost = f"{type(e).__name__}: {e}", None
        r = v["y_true"] - v["y_pred"]
        rec["vector"] = {
            "path": str(dst.relative_to(ROOT)), "rows": int(len(v)),
            "columns_are_the_contract": list(v.columns) == CONTRACT,
            "ids_identical_to_the_block": ids_new == ids_blk,
            "in_vector_not_in_block": len(ids_new - ids_blk),
            "in_block_not_in_vector": len(ids_blk - ids_new),
            "y_pred_nulls": int(v["y_pred"].isna().sum()),
            "y_pred_null_share": round(float(v["y_pred"].isna().mean()), 4),
            "stack_members_ok": ok, "rows_lost_to_the_join": lost,
            "rmse_where_predicted": (round(float((r ** 2).mean() ** 0.5), 4)
                                     if r.notna().any() else None),
            "n_scored": int(r.notna().sum()),
            "bytes": dst.stat().st_size,
            "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
        c = rec["vector"]
        print(f"    OK {dt/60:.1f} min  rows {c['rows']:,}  ids={c['ids_identical_to_the_block']}  "
              f"stack_members={c['stack_members_ok']} lost={c['rows_lost_to_the_join']}  "
              f"y_pred null {c['y_pred_null_share']:.1%}  sha {c['sha256'][:16]}", flush=True)
    else:
        rec["vector"] = None
        print(f"    producer returned but {produced} is absent", flush=True)

    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
