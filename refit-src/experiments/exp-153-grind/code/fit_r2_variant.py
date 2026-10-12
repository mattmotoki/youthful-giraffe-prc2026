"""Run one fixed L or Lp variation under the R2 member contract.

Uses the unchanged exp147 learner and split helper copied into this directory.
The held rows are development reads until an independent confirmation exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import learner_wp as LW

VARIANTS = {
    "control": {},
    "bag2": {"bagging_seed": 1602, "feature_fraction_seed": 1603},
    "leaves127": {"num_leaves": 127},
    "l2_25": {"lambda_l2": 25.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--learner", choices=("L", "Lp"), default="L")
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    if not 1 <= a.threads <= 10:
        raise SystemExit("threads outside the current allocation")
    if a.fit not in (*LW.P.FITS_R2, "janjul", "final"):
        raise SystemExit("R2 members require jjx, janjul or final fits")
    if a.fit.startswith("month-"):
        split = LW.WP.month_hold(a.fit.removeprefix("month-"))
    elif a.fit in ("jan-to-jul", "jul-to-jan"):
        train_month, score_month = (("2025-01", "2025-07") if a.fit == "jan-to-jul"
                                    else ("2025-07", "2025-01"))
        excluded = sorted(d for m in LW.MONTHS if m != train_month for d in LW.P._days(m))
        # Include the held-month buffer dates; this also excludes the two edge days
        # of the training month when they border another excluded month.
        for m in LW.MONTHS:
            if m != train_month:
                excluded = sorted(set(excluded) | set(LW.WP.month_hold(m).excluded))
        split = LW.WP.Split(a.fit, tuple(LW.P._days(score_month)), tuple(excluded))
    else:
        split = LW.P.split(a.fit)
    stage = a.learner + "153-" + a.variant
    if any((a.out / f"{stage}-{split.name}.{ext}").exists() for ext in ("json", "parquet")):
        raise SystemExit("output exists; use a new directory")

    setup_original = LW.setup

    def setup(root):
        env = setup_original(root)
        base = env[2]

        def fit(name, xt, xs, residual, weights, cats, anchor, model, fold):
            import lightgbm as lgb
            params = dict(model.MAIN_PARAMS)
            params.update(seed=601, bagging_seed=602, feature_fraction_seed=603,
                          deterministic=True, force_col_wise=True, num_threads=a.threads)
            params.update(VARIANTS[a.variant])
            rec = {"variant": a.variant, "fit": a.fit, "params": params,
                   "rounds": 1500, "train_split": split.manifest(),
                   "features": list(xt.columns), "categorical": cats,
                   "feature_list_sha256": LW.hashlib.sha256(json.dumps(list(xt.columns)).encode()).hexdigest(),
                   "train_rows": len(xt), "score_rows": len(xs),
                   "code_sha256": LW.sha(Path(__file__)), "development_only": True}
            a.out.mkdir(parents=True, exist_ok=True)
            frozen = a.out / f"{stage}-{split.name}-before-fit.json"
            if frozen.exists():
                raise SystemExit(f"fit record exists: {frozen}")
            frozen.write_text(json.dumps(rec, indent=1) + "\n")
            print(json.dumps({"event": "fit-start", "variant": a.variant,
                              "split": split.name, "rows": len(xt), "features": len(xt.columns)}), flush=True)
            t0 = time.monotonic()
            booster = lgb.train(params, lgb.Dataset(xt, label=residual, weight=weights,
                                categorical_feature=cats, free_raw_data=True), num_boost_round=1500)
            pred = np.maximum(30.0, anchor + booster.predict(xs, num_threads=a.threads))
            if not np.isfinite(pred).all():
                raise RuntimeError("nonfinite prediction")
            rec.update(fit_seconds=time.monotonic() - t0, features=len(xt.columns),
                       lightgbm_version=lgb.__version__)
            print(json.dumps({"event": "fit-end", "seconds": rec["fit_seconds"]}), flush=True)
            return pred, {"info": rec}

        base.fit_arm = fit
        return env

    LW.setup = setup
    LW.P.FITS = (*LW.P.FITS, a.fit)
    old_split, old_artifacts = LW.P.split, LW.P.artifacts_of
    LW.P.split = lambda name: split if name == a.fit else old_split(name)
    LW.P.artifacts_of = lambda name: [(split, a.fit)] if name == a.fit else old_artifacts(name)
    write_original = LW.P.write

    def write(out, old_stage, s, owner, pred, rec):
        pred = pred.rename(columns={"pred_" + a.learner: "pred_" + stage})
        rec.update(variant=a.variant, development_only=True, variant_code_sha256=LW.sha(Path(__file__)))
        return write_original(out, stage, s, owner, pred, rec)

    LW.P.write = write
    sys.argv = [str(Path(LW.__file__)), "--fit", a.fit, "--input-root", str(a.input_root),
                "--out", str(a.out), "--learners", a.learner, "--cache", str(a.cache or a.out / "cache")]
    LW.main()


if __name__ == "__main__":
    main()
