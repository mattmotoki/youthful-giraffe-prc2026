"""One arm of the head on one split: a K-seed bag of z = y - obs_taxi on the clock class.

    nice -n 19 taskset -c 56-59,120-123 $PY code/fit.py --arm A0 --split janjul --threads 8 --out out/step1

Trains on the split's admitted clock rows (`split.label_mask`, checked by
`split.assert_training_frame`), scores the split's held rows (or the 2026 rows for `final` and
`jjrank`; a `jjrank` fit also writes the `jjrefit` rows). Writes `<stage>-<split>.parquet`,
`.json` and `.seeds.npz` (write-once). `pred_<stage> = obs_taxi + mean_k z_hat_k`, no floor.

The learner is LightGBM. `--learner catboost` (SPEC-push addendum 10) fits CatBoost on the same
rows, inputs and target: a member of another kind for the bag. A category goes in as its integer
code (-1 for missing); `--rounds` is the number of trees.
"""
from __future__ import annotations

import argparse
import platform
import socket
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import common as C
import pilot as P                     # noqa: E402  (path set in common)
from prc import weekpair as WP       # noqa: E402

LGB = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 255, "min_child_samples": 100,
       "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
       "deterministic": True, "force_row_wise": True, "verbosity": -1}
ROUNDS = 300
#: parameters that --param may change, with their type
TUNABLE = {"num_leaves": int, "min_child_samples": int, "feature_fraction": float, "bagging_fraction": float,
           "lambda_l2": float, "min_sum_hessian_in_leaf": float, "max_depth": int}
RANKING_SPLITS = ("final", "jjrank")
CB = {"loss_function": "RMSE", "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3.0, "bootstrap_type": "Bernoulli",
      "subsample": 0.8, "rsm": 0.8, "border_count": 254, "allow_writing_files": False, "verbose": 0}
CB_TUNABLE = {"depth": int, "l2_leaf_reg": float, "subsample": float, "rsm": float, "min_data_in_leaf": int,
              "border_count": int}


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
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--checkpoints", type=int, nargs="*", default=[],
                    help="also write the artifacts of the first N trees as stage <stage>r<N>")
    ap.add_argument("--drop-months", nargs="*", default=[], help="training months left out (learning-curve read)")
    ap.add_argument("--learning-rate", type=float, default=None, help="in place of the head's 0.05")
    ap.add_argument("--param", nargs="*", default=[], metavar="NAME=VALUE",
                    help="LightGBM parameters in place of the head's, e.g. num_leaves=511 (written to the record)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--learner", choices=["lightgbm", "catboost"], default="lightgbm")
    ap.add_argument("--also-janjul", action="store_true",
                    help="with --split jjrank: write the janjul artifact too. The two splits exclude the same days, "
                         "so the fit is the janjul fit (checked here), and its held rows are the jjrefit rows")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    code_sha = {n: C.sha(Path(__file__).with_name(n)) for n in ("fit.py", "common.py")}   # the code as loaded
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    stage = a.stage or f"H{a.arm}"
    s = P.split(a.split)
    if s.token != C.TOKENS[a.split]:
        raise SystemExit(f"split {a.split}: token {s.token} is not the contract's {C.TOKENS[a.split]}")
    names = [a.split] + (["jjrefit"] if a.split == "jjrank" else [])
    if a.also_janjul:
        if a.split != "jjrank":
            raise SystemExit("--also-janjul needs --split jjrank")
        jj = P.split("janjul")
        if jj.token != C.TOKENS["janjul"] or tuple(jj.excluded) != tuple(s.excluded):
            raise SystemExit("janjul and jjrank do not exclude the same days")
        names.append("janjul")
    cuts = {f"{stage}r{c}": c for c in a.checkpoints} | {stage: a.rounds}
    for n in names:
        for st in cuts:
            C.refuse_existing(a.out / f"{st}-{n}.parquet", a.out / f"{st}-{n}.json")
    a.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    track_arm = a.arm.startswith(("A3", "A4", "A5", "A6"))  # A4u included
    f = C.load_frame(with_track=a.track or track_arm, with_extra=a.arm == "A5", with_weather=a.arm.startswith("A6"))
    is25 = f.month.isin(C.MONTHS_2025).to_numpy()
    anchor, z = targets(f, a.target)
    wp = f[[C.TIME]]                                       # every frame row is a departure
    train = is25 & s.label_mask(wp) & ~f.month.isin(a.drop_months).to_numpy()
    s.assert_training_frame(wp[train], target=z[train], what=f"{stage} {a.split}")
    if not np.isfinite(z[train]).all():
        raise SystemExit("a training row has no target")
    score = {a.split: (~is25 if a.split in RANKING_SPLITS else is25 & s.score_mask(wp))}
    if a.split == "jjrank":
        score["jjrefit"] = is25 & P.split("jjrefit").score_mask(wp)
    if a.also_janjul:
        score["janjul"] = is25 & P.split("janjul").score_mask(wp)
        if not np.array_equal(train, is25 & P.split("janjul").label_mask(wp) & ~f.month.isin(a.drop_months).to_numpy()):
            raise SystemExit("the janjul training rows are not the jjrank training rows")
    for n, m in score.items():
        if (m & train).any():
            raise SystemExit(f"{n}: a scored row is a training row")

    push = None
    if track_arm and a.arm != "A3c":
        cov = f.loc[train, C.TRACK_PUSH].notna().mean()
        push = [c for c in C.TRACK_PUSH if cov[c] >= a.min_coverage]
        print("pushback-phase coverage on training rows:", cov.round(4).to_dict(), "kept:", push, flush=True)
    feats = C.arm_features(a.arm, f.columns, push)
    X = f[feats].to_numpy("float32")
    print(f"{stage} {a.split}: {int(train.sum())} training rows, {len(feats)} inputs, "
          f"{ {n: int(m.sum()) for n, m in score.items()} } scored rows, load {time.time() - t0:.0f}s", flush=True)

    boosted = a.learner == "catboost"
    params, tunable = (dict(CB), CB_TUNABLE) if boosted else (dict(LGB), TUNABLE)
    if a.learning_rate is not None:
        params["learning_rate"] = a.learning_rate
    if a.device != "cpu":
        if boosted:
            raise SystemExit("the CatBoost learner runs on the CPU here")
        params["device_type"] = a.device
    for kv in a.param:
        k, _, v = kv.partition("=")
        if k not in tunable:
            raise SystemExit(f"--param {k}: not one of {sorted(tunable)}")
        params[k] = tunable[k](v)
    cat = [c for c in feats if c in C.CATEGORICAL_ALL]
    if boosted:
        import catboost as cb
        xf = pd.DataFrame(X, columns=feats)
        for c in cat:                                     # CatBoost takes a category as an integer: the code, -1 for missing
            xf[c] = f[c].fillna(-1).astype("int64").to_numpy()
        dtrain = cb.Pool(xf[train], z[train], cat_features=cat)
        pools = {n: cb.Pool(xf[m], cat_features=cat) for n, m in score.items()}
    else:
        dtrain = lgb.Dataset(X[train], z[train], feature_name=feats, categorical_feature=cat or "auto",
                             free_raw_data=False)
    zs = {(st, n): np.zeros((len(seeds), int(m.sum())), "float32") for st in cuts for n, m in score.items()}
    gain = np.zeros(len(feats))
    secs = []
    for k, seed in enumerate(seeds):
        t1 = time.time()
        if boosted:
            booster = cb.CatBoostRegressor(**params, iterations=a.rounds, random_seed=seed, thread_count=a.threads)
            booster.fit(dtrain)
            if booster.tree_count_ != a.rounds:
                raise SystemExit(f"seed {seed}: {booster.tree_count_} trees, expected {a.rounds}")
            for n in score:
                for st, c in cuts.items():
                    zs[st, n][k] = booster.predict(pools[n], ntree_end=c, thread_count=a.threads)
            gain += booster.get_feature_importance()
        else:
            booster = lgb.train({**params, "seed": seed, "num_threads": a.threads}, dtrain, num_boost_round=a.rounds)
            if booster.num_trees() != a.rounds:
                raise SystemExit(f"seed {seed}: {booster.num_trees()} trees, expected {a.rounds}")
            for n, m in score.items():
                for st, c in cuts.items():
                    zs[st, n][k] = booster.predict(X[m], num_threads=a.threads, num_iteration=c)
            gain += booster.feature_importance("gain")
        if a.save_models:
            mdir = a.out / f"{stage}-{a.split}.models"
            mdir.mkdir(exist_ok=True)
            booster.save_model(str(mdir / f"seed-{seed}.{'cbm' if boosted else 'txt'}"))
        secs.append(round(time.time() - t1, 1))
        print(f"  seed {seed}: {secs[-1]}s", flush=True)

    imp = pd.Series(gain / gain.sum(), index=feats).sort_values(ascending=False)
    base_rec = {"arm": a.arm, "target": a.target, "learner": a.learner, "lightgbm": lgb.__version__,
                **({"catboost": cb.__version__, "importance_kind": "PredictionValuesChange"} if boosted else {}),
                "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version(),
                "fit": {"params": params, "rounds": a.rounds, "seeds": seeds, "num_threads": a.threads,
                        "seconds_per_seed": secs, "categorical": cat, "drop_months": a.drop_months},
                "features": feats, "features_sha256": C.list_sha(feats),
                "train_rows": int(train.sum()), "frame_set": C.SET,
                "train_rows_by_month": f.loc[train].groupby("month").size().to_dict(),
                "train_split": s.manifest(), "class": "ctx_cls == clock",
                "inputs": {"frame_sha256": C.sha(C.FRAME),
                           "track_sha256": C.sha(C.TRACK) if "trk_implied_taxi" in f else None,
                           "extra_sha256": C.sha(C.EXTRA) if a.arm == "A5" else None,
                           "weather_sha256": C.sha(C.WEATHER) if a.arm.startswith("A6") else None},
                "importance_gain_top20": imp.head(20).round(5).to_dict(),
                "code_sha256": code_sha,
                "source_commit": (C.SOURCE / "SOURCE_COMMIT").read_text().strip()}
    if C.N_DEPARTURES is not None:                         # exp-186: the final mode's records name it
        base_rec["ranking_mode"] = C.RANKING_MODE
    for (st, n), zz in zs.items():
        m = score[n]
        sp = P.split(n)
        pred = f.loc[m, [C.ID, "month", "utc_day", "airport"]].reset_index(drop=True)
        pred[f"pred_{st}"] = anchor[m] + zz.mean(0).astype("float64")
        np.savez_compressed(a.out / f"{st}-{n}.seeds.npz", ids=pred[C.ID].to_numpy("int64"),
                            anchor=anchor[m], z=zz, seeds=np.array(seeds))
        rec = dict(base_rec, domain="ranking2026" if n in RANKING_SPLITS else "held2025", trees_used=cuts[st])
        P.write(a.out, st, sp, int(n[6:]) if n.startswith("outer-") else n, pred, rec)   # a week-pair fold's owner is k
        print(f"wrote {st}-{n}: {len(pred)} rows", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
