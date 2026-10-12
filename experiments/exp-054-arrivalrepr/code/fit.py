"""exp-054-arrivalrepr: the five-way cross-fitted departure correction of one or more folds
(SPEC 1.6, 1.7, 1.9).

Per fold, in this order:

  1. read the frozen v22 panel by column WITHOUT `y_true` (full-file and ordered id+p_v22 sha
     pins, refused on a miss), freeze P (Q08's sha) and the five day blocks;
  2. build the departure row-only features from the held month read WITHOUT either label column
     (`e54.departure_frame`), then look up each frozen representation (sha-checked against its
     record) on the departure's keys; build the six arm matrices (the row-only arm padded with
     constant zeros to the same width) and record every matrix sha and the unix time.  The
     constant arm's matrix must equal the row-only arm's byte for byte (SPEC 1.8);
  3. only now read `y_true` and form r22 = y - p_v22 (the first label read, timed);
  4. for each arm and each day block b: fit the fixed learner on the P rows of the four other
     blocks with y > 0, predict the P rows of block b (h = 0 outside P), holding every launch
     while the 1-minute load is above 10;
  5. write every arm's h on every panel row (`work/h-<fold>.parquet`) and the fit record.

No score is computed here: this file never compares a prediction with a label.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-054-arrivalrepr/code/fit.py july_loo december january
    # mechanics only (synthetic target, no label read, scratch outputs; needs a dry-run rep):
    ... fit.py july_loo --dry-run --out-dir <scratch>
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

D, NO = E.D, E.NO


def load_rep(fold: str, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    j = json.loads((out_dir / f"analysis/rep-{fold}.json").read_text())
    p = out_dir / f"work/rep-{fold}.parquet"
    if E.sha_file(p) != j["frozen"]["sha256"]:
        raise SystemExit(f"{p} moved since its representation record was written")
    return pd.read_parquet(p), j


def matrices(fold: str, pan: pd.DataFrame, out_dir: Path) -> tuple[dict, dict]:
    """The six arm matrices on the panel's rows, and their record.  Label-free."""
    month = E.fold_month(fold)
    frame = E.departure_frame(month)
    F = E.row_features(frame)
    ids = pan.MVT_ID_mvt.to_numpy("float64")
    if len(F) != len(pan) or not np.array_equal(np.sort(F.MVT_ID_mvt.to_numpy("float64")), np.sort(ids)):
        raise SystemExit(f"{fold}: departure rows and panel rows are not the same set")
    Fe = E.encode(F).set_index("MVT_ID_mvt").reindex(ids)
    dep = frame[frame["PHASE_mvt"].to_numpy() == "DEP"].reset_index(drop=True)
    dk = E.departure_keys(dep).set_index("MVT_ID_mvt").reindex(ids)
    rep, rj = load_rep(fold, out_dir)
    base = Fe[E.ROW_FEATURES].to_numpy("float64")
    X, rec = {}, {"representation_sha256": rj["frozen"]["sha256"],
                  "representation_frozen_unix": rj["frozen"]["unix_time"], "arms": {}}
    for arm in E.ARMS:
        cols = []
        for k in E.KEYS:
            if arm == "rowonly":
                cols.append(np.zeros(len(ids)))
                continue
            t = rep[rep["key"] == k].set_index("value")[arm]
            cols.append(t.reindex(dk[k].to_numpy(object)).fillna(0.0).to_numpy("float64"))
        X[arm] = np.column_stack([base] + cols)
        rec["arms"][arm] = {"matrix_sha256": E.sha_arr(X[arm]),
                            "rep_nonzero_rows": {k: int((c != 0).sum()) for k, c in zip(E.KEYS, cols)}}
    if not np.array_equal(X["const"], X["rowonly"], equal_nan=True):
        raise SystemExit(f"{fold}: the constant arm's matrix differs from the row-only arm's")
    rec["const_equals_rowonly_matrix"] = True
    rec["row_features_sha256"] = E.sha_arr(base)
    rec["features"] = E.ALL_FEATURES
    rec["categorical"] = E.ROW_CATS
    rec["frozen_unix"] = time.time()
    rec["frozen_utc"] = E.utc()
    return X, rec


def run_fold(fold: str, out_dir: Path, dry: bool, overwrite: bool) -> dict:
    t0 = time.time()
    pan = E.panel(fold, E.PANEL_LABEL_FREE)          # no y_true in this read
    pop = E.population(pan, fold)
    blk = E.blocks(pan.day.to_numpy())
    X, mrec = matrices(fold, pan, out_dir)
    rec = {"fold": fold, "dry_run": dry, "rows": int(len(pan)), "P_rows": int(pop.sum()),
           "block_rows_P": {int(b): int((pop & (blk == b)).sum()) for b in range(E.K)},
           "matrices": mrec, "params": {k: v for k, v in E.LGB_PARAMS.items()}}
    # ---- the first label read ----------------------------------------------------------
    rec["label_first_read_unix"] = time.time()
    if rec["label_first_read_unix"] <= mrec["frozen_unix"]:
        raise SystemExit("clock order: labels before the frozen matrices")
    if dry:
        y = pan.p_v22.to_numpy("float64") + np.random.default_rng(E.SEED).normal(0.0, 170.0, len(pan))
    else:
        y = E.panel(fold, ["MVT_ID_mvt", "y_true"]).y_true.to_numpy("float64")
    r = y - pan.p_v22.to_numpy("float64")
    fitrows = pop & np.isfinite(r) & (y > 0)
    cat_idx = [E.ALL_FEATURES.index(c) for c in E.ROW_CATS]
    H = {arm: np.zeros(len(pan)) for arm in E.ARMS}
    rec["fits"] = {}
    for arm in E.ARMS:
        rec["fits"][arm] = {}
        for b in range(E.K):
            tr = np.flatnonzero(fitrows & (blk != b))
            sc = np.flatnonzero(pop & (blk == b))
            lo = E.hold_load(f"{fold} {arm} block {b}")
            t1 = time.time()
            m = lgb.LGBMRegressor(**E.LGB_PARAMS)
            m.fit(X[arm][tr], r[tr], categorical_feature=cat_idx)
            H[arm][sc] = m.predict(X[arm][sc])
            imp = m.booster_.feature_importance(importance_type="split")
            rec["fits"][arm][b] = {**lo, "seconds": round(time.time() - t1, 3),
                                   "train_rows": int(len(tr)), "scored_rows": int(len(sc)),
                                   "trees": int(m.booster_.num_trees()),
                                   "splits_by_feature": {f: int(v) for f, v in zip(E.ALL_FEATURES, imp) if v},
                                   "h_rms_scored": float(np.sqrt(np.mean(H[arm][sc] ** 2)))}
        print(f"  [{fold}] {arm:8s} 5 fits, h rms on P {np.sqrt(np.mean(H[arm][pop] ** 2)):.2f} s, "
              f"rep splits {sum(rec['fits'][arm][b]['splits_by_feature'].get(c, 0) for b in range(E.K) for c in E.REP_COLS)}",
              flush=True)
    rec["const_equals_rowonly_h"] = bool(np.array_equal(H["const"], H["rowonly"]))
    rec["trainable_fits"] = E.K * len(E.ARMS)
    out = pd.DataFrame({"MVT_ID_mvt": pan.MVT_ID_mvt.to_numpy("float64"), "block": blk, "pop": pop,
                        **{f"h_{a}": H[a] for a in E.ARMS}})
    p = out_dir / f"work/h-{fold}.parquet"
    NO.refuse_existing([p], overwrite)
    out.to_parquet(p, index=False)
    rec["h_parquet"] = {"path": str(p), "sha256": E.sha_file(p),
                        "h_sha256": {a: E.sha_arr(H[a]) for a in E.ARMS}}
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
        paths += [out_dir / f"work/h-{f}.parquet", out_dir / f"analysis/fit-{f}.json"]
    NO.refuse_existing(paths, args.overwrite)
    print("guard: no existing artifact at " + ", ".join(str(p) for p in paths), flush=True)
    code = {n: E.sha_file(E.HERE / n) for n in ("e54.py", "fit.py")}
    for f in args.folds:
        rec = run_fold(f, out_dir, args.dry_run, args.overwrite)
        rec["code_sha256"] = code
        j = out_dir / f"analysis/fit-{f}.json"
        NO.refuse_existing([j], args.overwrite)
        j.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
        print(f"written {j} ({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
