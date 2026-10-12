"""exp-054-arrivalrepr iteration 2 (SPEC section 2, desk decision recorded rule): the matched raw-key removal.

The three raw key categoricals (`apt_stand`, `apt_rwy`, `op`) are removed from every arm, so the
frozen iteration-1 factors are the only channel for stand, runway and operator structure.  No new
representation fit: the frozen `work/rep-<fold>.parquet` of iteration 1 is looked up exactly as
iteration 1 did.

Per fold, in this order:

  1. read the frozen v22 panel by column WITHOUT `y_true` (pins), freeze P (Q08's sha) and the same
     five day blocks as iteration 1;
  2. rebuild the six iteration-1 arm matrices with iteration 1's own code (`fit.matrices`, the
     held month read WITHOUT either label column, the frozen representation sha-checked) and
     require every matrix sha to equal the one in `analysis/fit-<fold>.json`; then delete the three
     raw-key columns from every arm (13 columns: 10 row features + the 3 factors).  The constant
     arm's keyless matrix must equal the row-only arm's (asserted; not fitted).  Record every
     keyless matrix sha and the unix time;
  3. only now read `y_true` and form r22 = y - p_v22 (the first label read, timed);
  4. for each of the five fitted keyless arms and each day block b: the iteration-1 learner
     (`e54.LGB_PARAMS`, seed 54) fitted on the P rows of the four other blocks with y > 0,
     predicting the P rows of block b (h = 0 outside P); every launch held while the 1-minute load
     is above 10;
  5. write every arm's h (`work/h2-<fold>.parquet`) and the fit record (`analysis/fit2-<fold>.json`).

No score is computed here.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-054-arrivalrepr/code/fit2.py july_loo december january
    # mechanics only (synthetic target, no label read, scratch outputs):
    ... fit2.py july_loo --dry-run --out-dir <scratch>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e54 as E                                                       # noqa: E402
import fit as F1                                                      # noqa: E402  (iteration 1's code, unchanged)

NO = E.NO
RAW_KEYS = ["apt_stand", "apt_rwy", "op"]
FEATURES2 = [c for c in E.ALL_FEATURES if c not in RAW_KEYS]
CATS2 = [c for c in E.ROW_CATS if c not in RAW_KEYS]
ARMS2 = ["true", "gmean", "donor", "shuffle", "rowonly"]          # fitted; const_k == rowonly_k
TAG = "_k"                                                         # keyless arm names: true_k, ...


def matrices2(fold: str, pan: pd.DataFrame) -> tuple[dict, dict]:
    """The keyless arm matrices on the panel's rows, and their record.  Label-free."""
    X1, rec1 = F1.matrices(fold, pan, E.EXP)                       # the frozen iteration-1 representation
    j1 = json.loads(E.fit_json(fold).read_text())
    same = {a: rec1["arms"][a]["matrix_sha256"] == j1["matrices"]["arms"][a]["matrix_sha256"] for a in E.ARMS}
    if not all(same.values()):
        raise SystemExit(f"{fold}: a rebuilt iteration-1 matrix differs from its record: {same}")
    drop = [E.ALL_FEATURES.index(c) for c in RAW_KEYS]
    X = {a: np.delete(X1[a], drop, axis=1) for a in E.ARMS}
    if not np.array_equal(X["const"], X["rowonly"], equal_nan=True):
        raise SystemExit(f"{fold}: the constant arm's keyless matrix differs from the row-only arm's")
    rec = {"representation_sha256": rec1["representation_sha256"],
           "representation_frozen_unix": rec1["representation_frozen_unix"],
           "iteration1_matrices_rebuilt_equal": same,
           "removed_columns": RAW_KEYS, "features": FEATURES2, "categorical": CATS2,
           "const_equals_rowonly_matrix": True,
           "arms": {a + TAG: {"matrix_sha256": E.sha_arr(X[a]),
                              "rep_nonzero_rows": rec1["arms"][a]["rep_nonzero_rows"]} for a in E.ARMS}}
    rec["frozen_unix"] = time.time()
    rec["frozen_utc"] = E.utc()
    return X, rec


def run_fold(fold: str, out_dir: Path, dry: bool, overwrite: bool) -> dict:
    t0 = time.time()
    pan = E.panel(fold, E.PANEL_LABEL_FREE)          # no y_true in this read
    pop = E.population(pan, fold)
    blk = E.blocks(pan.day.to_numpy())
    X, mrec = matrices2(fold, pan)
    rec = {"fold": fold, "dry_run": dry, "rows": int(len(pan)), "P_rows": int(pop.sum()),
           "block_rows_P": {int(b): int((pop & (blk == b)).sum()) for b in range(E.K)},
           "matrices": mrec, "params": dict(E.LGB_PARAMS)}
    # ---- the first label read ----------------------------------------------------------------
    rec["label_first_read_unix"] = time.time()
    if rec["label_first_read_unix"] <= mrec["frozen_unix"]:
        raise SystemExit("clock order: labels before the frozen matrices")
    if dry:
        y = pan.p_v22.to_numpy("float64") + np.random.default_rng(E.SEED).normal(0.0, 170.0, len(pan))
    else:
        y = E.panel(fold, ["MVT_ID_mvt", "y_true"]).y_true.to_numpy("float64")
    r = y - pan.p_v22.to_numpy("float64")
    fitrows = pop & np.isfinite(r) & (y > 0)
    cat_idx = [FEATURES2.index(c) for c in CATS2]
    H = {a: np.zeros(len(pan)) for a in ARMS2}
    rec["fits"] = {}
    for a in ARMS2:
        rec["fits"][a + TAG] = {}
        for b in range(E.K):
            tr = np.flatnonzero(fitrows & (blk != b))
            sc = np.flatnonzero(pop & (blk == b))
            lo = E.hold_load(f"{fold} {a}{TAG} block {b}")
            t1 = time.time()
            m = lgb.LGBMRegressor(**E.LGB_PARAMS)
            m.fit(X[a][tr], r[tr], categorical_feature=cat_idx)
            H[a][sc] = m.predict(X[a][sc])
            imp = m.booster_.feature_importance(importance_type="split")
            rec["fits"][a + TAG][b] = {**lo, "seconds": round(time.time() - t1, 3),
                                       "train_rows": int(len(tr)), "scored_rows": int(len(sc)),
                                       "trees": int(m.booster_.num_trees()),
                                       "splits_by_feature": {f: int(v) for f, v in zip(FEATURES2, imp) if v},
                                       "h_rms_scored": float(np.sqrt(np.mean(H[a][sc] ** 2)))}
        print(f"  [{fold}] {a + TAG:10s} 5 fits, h rms on P {np.sqrt(np.mean(H[a][pop] ** 2)):.2f} s, rep splits "
              f"{sum(rec['fits'][a + TAG][b]['splits_by_feature'].get(c, 0) for b in range(E.K) for c in E.REP_COLS)}",
              flush=True)
    rec["trainable_fits"] = E.K * len(ARMS2)
    out = pd.DataFrame({"MVT_ID_mvt": pan.MVT_ID_mvt.to_numpy("float64"), "block": blk, "pop": pop,
                        **{f"h_{a}{TAG}": H[a] for a in ARMS2}})
    p = out_dir / f"work/h2-{fold}.parquet"
    NO.refuse_existing([p], overwrite)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    rec["h_parquet"] = {"path": str(p), "sha256": E.sha_file(p),
                        "h_sha256": {a + TAG: E.sha_arr(H[a]) for a in ARMS2}}
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folds", nargs="+", choices=E.FOLDS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else E.EXP
    if args.dry_run and out_dir == E.EXP:
        raise SystemExit("a dry run writes to --out-dir only")
    paths = []
    for f in args.folds:
        paths += [out_dir / f"work/h2-{f}.parquet", out_dir / f"analysis/fit2-{f}.json"]
    NO.refuse_existing(paths, args.overwrite)
    print("guard: no existing artifact at " + ", ".join(str(p) for p in paths), flush=True)
    code = {n: E.sha_file(E.HERE / n) for n in ("e54.py", "fit.py", "fit2.py")}
    for f in args.folds:
        rec = run_fold(f, out_dir, args.dry_run, args.overwrite)
        rec["code_sha256"] = code
        j = out_dir / f"analysis/fit2-{f}.json"
        NO.refuse_existing([j], args.overwrite)
        j.parent.mkdir(parents=True, exist_ok=True)
        j.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
        print(f"written {j} ({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
