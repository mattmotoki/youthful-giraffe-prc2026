"""exp-018 fresh lineage: fold runner.

  .venv/bin/python experiments/exp-018-fresh/code/run.py --tag iter1 --folds july december january july_loo
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D  # noqa: E402
import model as M  # noqa: E402
from prc import data  # noqa: E402

EXP = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", nargs="+", default=["july", "december", "january", "july_loo"])
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--clip-z", type=float, default=None)
    ap.add_argument("--params", type=str, default="{}")
    ap.add_argument("--seeds", type=int, default=1)
    #: exp-035-k2climb, arm 1. `--seeds N` fits seeds 7+13*s for s in 0..N-1 and
    #: averages them inside one process; a 45-minute Kaggle fold cannot hold three
    #: of those in one 12-hour slot beside everything else, so this offsets the
    #: sequence and lets one seed be fitted per kernel and averaged afterwards.
    #: `--seed-start 0 --seeds 1` is the frozen path and still passes seed=None.
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--save-oof", action="store_true")
    ap.add_argument("--mechanism", type=str, default="")
    ap.add_argument("--curve", type=int, default=0)
    ap.add_argument("--ghost-mode", default="ratio")
    ap.add_argument("--main-model", default="lgb")
    ap.add_argument("--exact-rounds", type=int, default=400)
    a = ap.parse_args()
    params = json.loads(a.params)

    out = {}
    for fname in a.folds:
        fold = data.FOLDS[fname]
        t0 = time.time()
        train = D.load(fold.train_months)
        valid = D.load([fold.valid_month])
        # exp-035-k2climb arm 3: `PRC_EXTRAS=phi[,surface,deice]` attaches
        # exp-018b's module columns. Unset changes nothing.
        if os.environ.get("PRC_EXTRAS", "").strip():
            sys.path.insert(0, str(EXP.parent / "exp-018b-climb" / "code"))
            import extras as X  # noqa: PLC0415
            ks = X.kinds_from_env(os.environ["PRC_EXTRAS"])
            xtr, xva = X.build(fname)
            train = X.attach(train, xtr, ks, f"{fname}/train")
            valid = X.attach(valid, xva, ks, f"{fname}/valid")
        y = valid["y"].to_numpy(dtype="float64")
        preds = []
        for s in range(a.seeds):
            pred, info = M.fit_predict(train.copy(), valid.copy(), rounds=a.rounds,
                                       clip_z=a.clip_z, params=params, curve=a.curve, y_valid=y,
                                       ghost_mode=a.ghost_mode, main_model=a.main_model,
                                       exact_rounds=a.exact_rounds,
                                       seed=(7 + 13 * (a.seed_start + s))
                                       if (a.seeds > 1 or a.seed_start) else None)
            preds.append(pred)
        pred = np.mean(preds, axis=0)
        sc = M.score(y, pred)
        sc["secs"] = round(time.time() - t0, 1)
        sc["n_train"] = info["n_train"]
        out[fname] = sc
        print(f"{fname:10s} rmse={sc['rmse']:8.2f} lo={sc['rmse_lo']:7.2f} hi={sc['rmse_hi']:9.1f} "
              f"(n_hi={sc['n_hi']}) {sc['secs']}s", flush=True)
        if a.save_oof:
            od = EXP / "oof"; od.mkdir(exist_ok=True)
            pd.DataFrame({"MVT_ID_mvt": valid["MVT_ID_mvt"].to_numpy(), "fold": fname,
                          "y_true": y, "y_pred": pred}).to_parquet(od / f"{fname}.parquet", index=False)
        del train, valid

    line = " ".join(f"{f}={out[f]['rmse']:.2f}" for f in a.folds)
    rec = dict(tag=a.tag, mechanism=a.mechanism, rounds=a.rounds, clip_z=a.clip_z,
               params=params, seeds=a.seeds, seed_start=a.seed_start,
               extras=os.environ.get("PRC_EXTRAS", ""),
               ghost_mode=a.ghost_mode,
               main_model=a.main_model, exact_rounds=a.exact_rounds, folds=out)
    (EXP / "results").mkdir(exist_ok=True)
    (EXP / "results" / f"{a.tag}.json").write_text(json.dumps(rec, indent=1))
    with open(EXP / "climb.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M')} {a.tag}\t{a.mechanism}\t{line}\t"
                + " ".join(f"{f_}:lo={out[f_]['rmse_lo']:.1f},hi={out[f_]['rmse_hi']:.0f}" for f_ in a.folds) + "\n")
    print("TAG", a.tag, line)


if __name__ == "__main__":
    main()
