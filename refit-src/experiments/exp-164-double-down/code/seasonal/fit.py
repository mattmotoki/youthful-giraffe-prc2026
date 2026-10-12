"""One arm of the head on one split: a K-seed LightGBM bag of z = y - obs_taxi on the clock class.

    nice -n 19 taskset -c 56-59,120-123 $PY code/fit.py --arm A0 --split janjul --threads 8 --out out/step1

Trains on the split's admitted clock rows (`split.label_mask`, checked by
`split.assert_training_frame`), scores the split's held rows (or the 2026 rows for `final` and
`jjrank`; a `jjrank` fit also writes the `jjrefit` rows). Writes `<stage>-<split>.parquet`,
`.json` and `.seeds.npz` (write-once). `pred_<stage> = obs_taxi + mean_k z_hat_k`, no floor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

import common as C
import pilot as P                     # noqa: E402  (path set in common)
from prc import weekpair as WP       # noqa: E402

LGB = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 255, "min_child_samples": 100,
       "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
       "deterministic": True, "force_row_wise": True, "verbosity": -1}
ROUNDS = 300
RANKING_SPLITS = ("final", "jjrank")
SEASON_MONTHS = {
    "jan": ("2025-02", "2025-03", "2025-10", "2025-11", "2025-12"),
    "jul": ("2025-03", "2025-08", "2025-10", "2025-11", "2025-12"),
}


def targets(f: pd.DataFrame, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """(anchor, z): the label-free anchor the head adds back, and the fitted target."""
    anchor = {"obs": f["obs_obs_taxi"], "aobt": f["shp_r_aobt"].fillna(f["obs_obs_taxi"])}[kind].to_numpy("float64")
    return anchor, f["y"].to_numpy("float64") - anchor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seeds", default="601-610")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--target", choices=["obs", "aobt"], default="obs")
    ap.add_argument("--stage", default=None, help="artifact stage name; default H<arm>")
    ap.add_argument("--track", action="store_true", help="join the track table")
    ap.add_argument("--min-coverage", type=float, default=0.05,
                    help="drop a pushback-phase input under this share of training rows")
    ap.add_argument("--save-models", action="store_true", help="keep the boosters (for the label-free reads)")
    ap.add_argument("--policy", choices=("shared", "hard", "soft"), required=True)
    ap.add_argument("--season", choices=("jan", "jul"))
    ap.add_argument("--feature-record", type=Path, required=True)
    ap.add_argument("--input-record", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.policy != 'shared' and a.season is None:
        raise SystemExit('a seasonal policy needs its season')
    if a.policy == 'shared' and a.season is not None:
        raise SystemExit('the shared control has no season')
    input_record = json.loads(a.input_record.read_text())
    for key, path in (('frame', C.FRAME), ('track', C.TRACK)):
        if C.sha(path) != input_record['source_hashes'][key]:
            raise RuntimeError(f'A3 {key} source hash differs')
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    stage = a.stage or f"H{a.arm}"
    s = P.split(a.split)
    if s.token != C.TOKENS[a.split]:
        raise SystemExit(f"split {a.split}: token {s.token} is not the contract's {C.TOKENS[a.split]}")
    names = [a.split] + (["jjrefit"] if a.split == "jjrank" else [])
    for n in names:
        C.refuse_existing(a.out / f"{stage}-{n}.parquet", a.out / f"{stage}-{n}.json")
    a.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    f = C.load_frame(with_track=a.track or a.arm.startswith("A3"))
    is25 = f.month.isin(C.MONTHS_2025).to_numpy()
    anchor, z = targets(f, a.target)
    wp = f[[C.TIME]]                                       # every frame row is a departure
    shared_train = is25 & s.label_mask(wp)
    train = shared_train.copy()
    preferred = f.month.isin(SEASON_MONTHS.get(a.season, ())).to_numpy()
    if a.policy == 'hard':
        train &= preferred
    weights = np.ones(int(train.sum()), dtype='float32')
    if a.policy == 'soft':
        weights = np.where(preferred[train], 2., 1.).astype('float32')
        weights /= weights.mean(dtype='float64')
    if not train.any() or not np.isfinite(weights).all() or abs(float(weights.mean())-1) > 1e-6:
        raise RuntimeError('empty training frame or bad normalized weights')
    s.assert_training_frame(wp[train], target=z[train], what=f"{stage} {a.split}")
    if not np.isfinite(z[train]).all():
        raise SystemExit("a training row has no target")
    score = {a.split: (~is25 if a.split in RANKING_SPLITS else is25 & s.score_mask(wp))}
    if a.split == "jjrank":
        score["jjrefit"] = is25 & P.split("jjrefit").score_mask(wp)
    for n, m in score.items():
        if (m & train).any():
            raise SystemExit(f"{n}: a scored row is a training row")

    push = None
    if a.arm.startswith("A3") and a.arm != "A3c":
        cov = f.loc[shared_train, C.TRACK_PUSH].notna().mean()
        push = [c for c in C.TRACK_PUSH if cov[c] >= a.min_coverage]
        print("pushback-phase coverage on training rows:", cov.round(4).to_dict(), "kept:", push, flush=True)
    feats = C.arm_features(a.arm, f.columns, push)
    feature_record = {'features': feats, 'features_sha256': C.list_sha(feats),
                      'selection': '5% coverage on shared janjul training rows only',
                      'frame_sha256': C.sha(C.FRAME), 'track_sha256': C.sha(C.TRACK)}
    if a.feature_record.exists():
        if json.loads(a.feature_record.read_text()) != feature_record:
            raise RuntimeError('fixed seasonal feature record differs')
    elif a.policy == 'shared':
        C.write_json(a.feature_record, feature_record)
    else:
        raise RuntimeError('shared feature record must exist before a seasonal fit')
    X = f[feats].to_numpy("float32")
    print(f"{stage} {a.split}: {int(train.sum())} training rows, {len(feats)} inputs, "
          f"{ {n: int(m.sum()) for n, m in score.items()} } scored rows, load {time.time() - t0:.0f}s", flush=True)

    training_path = a.out / f'{stage}-{a.split}-training.parquet'
    C.refuse_existing(training_path)
    training_rows = f.loc[train, [C.ID, 'month']].copy()
    training_rows['sample_weight'] = weights
    training_rows.to_parquet(training_path, index=False)
    before = {'policy': a.policy, 'season': a.season, 'preferred_months': SEASON_MONTHS.get(a.season),
              'train_split': s.manifest(), 'train_rows': int(train.sum()),
              'train_rows_by_month': f.loc[train].groupby('month').size().to_dict(),
              'train_ids_sha256': hashlib.sha256(f.loc[train, C.ID].to_numpy('<i8').tobytes()).hexdigest(),
              'weights_sha256': hashlib.sha256(weights.astype('<f4').tobytes()).hexdigest(),
              'training_rows_path': str(training_path), 'training_rows_sha256': C.sha(training_path),
              'weights_mean': float(weights.mean()), 'weight_range': [float(weights.min()), float(weights.max())],
              'weighted_rows_by_month': {m: float(weights[f.loc[train, 'month'].to_numpy() == m].sum(dtype='float64'))
                                         for m in sorted(set(f.loc[train, 'month']))},
              'features': feature_record, 'params': LGB, 'rounds': ROUNDS, 'seeds': seeds,
              'target': a.target, 'threads': a.threads, 'code_sha256': C.sha(__file__)}
    before_path = a.out / f'{stage}-{a.split}-before-fit.json'
    C.refuse_existing(before_path)
    C.write_json(before_path, before)
    dtrain = lgb.Dataset(X[train], z[train], weight=weights if a.policy == 'soft' else None,
                        feature_name=feats, free_raw_data=False)
    zs = {n: np.zeros((len(seeds), int(m.sum())), "float32") for n, m in score.items()}
    gain = np.zeros(len(feats))
    secs = []
    for k, seed in enumerate(seeds):
        t1 = time.time()
        booster = lgb.train({**LGB, "seed": seed, "num_threads": a.threads}, dtrain, num_boost_round=ROUNDS)
        if booster.num_trees() != ROUNDS:
            raise SystemExit(f"seed {seed}: {booster.num_trees()} trees, expected {ROUNDS}")
        for n, m in score.items():
            zs[n][k] = booster.predict(X[m], num_threads=a.threads)
        gain += booster.feature_importance("gain")
        if a.save_models:
            mdir = a.out / f"{stage}-{a.split}.models"
            mdir.mkdir(exist_ok=True)
            booster.save_model(str(mdir / f"seed-{seed}.txt"))
        secs.append(round(time.time() - t1, 1))
        print(f"  seed {seed}: {secs[-1]}s", flush=True)

    imp = pd.Series(gain / gain.sum(), index=feats).sort_values(ascending=False)
    base_rec = {"seasonal": before, "development_only": True,
                "before_fit_sha256": C.sha(before_path),
                "feature_record_sha256": C.sha(a.feature_record),
                "input_record_sha256": C.sha(a.input_record),
                "arm": a.arm, "target": a.target, "learner": "lightgbm", "lightgbm": lgb.__version__,
                "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version(),
                "fit": {"params": LGB, "rounds": ROUNDS, "seeds": seeds, "num_threads": a.threads,
                        "seconds_per_seed": secs},
                "features": feats, "features_sha256": C.list_sha(feats),
                "train_rows": int(train.sum()),
                "train_rows_by_month": f.loc[train].groupby("month").size().to_dict(),
                "train_split": s.manifest(), "class": "ctx_cls == clock",
                "inputs": {"frame_sha256": C.sha(C.FRAME),
                           "track_sha256": C.sha(C.TRACK) if "trk_implied_taxi" in f else None},
                "importance_gain_top20": imp.head(20).round(5).to_dict(),
                "code_sha256": {n: C.sha(Path(__file__).with_name(n)) for n in ("fit.py", "common.py")},
                "source_commit": (C.SOURCE / "SOURCE_COMMIT").read_text().strip()}
    for n, m in score.items():
        sp = P.split(n)
        pred = f.loc[m, [C.ID, "month", "utc_day", "airport"]].reset_index(drop=True)
        pred[f"pred_{stage}"] = anchor[m] + zs[n].mean(0).astype("float64")
        np.savez_compressed(a.out / f"{stage}-{n}.seeds.npz", ids=pred[C.ID].to_numpy("int64"),
                            anchor=anchor[m], z=zs[n], seeds=np.array(seeds))
        rec = dict(base_rec, domain="ranking2026" if n in RANKING_SPLITS else "held2025")
        P.write(a.out, stage, sp, n, pred, rec)
        print(f"wrote {stage}-{n}: {len(pred)} rows", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
