"""exp-018 fresh lineage: fit on all twelve 2025 months and write the candidate."""
from __future__ import annotations

import argparse, gc, json, os, sys
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
    ap.add_argument("--out", default=str(data.ROOT / "submissions" / "candidates" / "exp-018-fresh.parquet"))
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--clip-z", type=float, default=None)
    ap.add_argument("--params", type=str, default="{}")
    ap.add_argument("--seeds", type=int, default=3)
    #: exp-035-k2climb, arm 1. Offsets the seed sequence so one seed can be
    #: fitted per Kaggle session and averaged with an existing multi-seed mean
    #: afterwards. `--seed-start 0` is the frozen path.
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--ghost-mode", default="ratio")
    ap.add_argument("--main-model", default="lgb")
    ap.add_argument("--exact-rounds", type=int, default=400)
    a = ap.parse_args()

    train = D.load(data.MONTHS)
    rank = D.load(["2026-01", "2026-07"])
    if os.environ.get("PRC_EXTRAS", "").strip():
        sys.path.insert(0, str(EXP.parent / "exp-018b-climb" / "code"))
        import extras as X  # noqa: PLC0415
        ks = X.kinds_from_env(os.environ["PRC_EXTRAS"])
        xtr, xva = X.build_rank()
        train = X.attach(train, xtr, ks, "rank/train")
        rank = X.attach(rank, xva, ks, "rank/valid")
        # exp-035-k2climb: `prc-k2c-mod-rank` was SIGKILLed at 28.99 GB of 31
        # during its SECOND seed, with the 92 module columns still held here as
        # well as inside `train`. They are ~1.8 GB and nothing reads them again.
        del xtr, xva
        gc.collect()
    preds = []
    for s in range(a.seeds):
        pr, info = M.fit_predict(train.copy(), rank.copy(), rounds=a.rounds, clip_z=a.clip_z,
                                 params=json.loads(a.params), ghost_mode=a.ghost_mode,
                                 main_model=a.main_model, exact_rounds=a.exact_rounds,
                                 seed=7 + 13 * (a.seed_start + s))
        preds.append(pr)
        print(f"seed {s}: mean={pr.mean():.1f} p99={np.percentile(pr, 99):.0f} max={pr.max():.0f}", flush=True)
    pred = np.mean(preds, axis=0)
    ser = pd.Series(pred, index=rank["MVT_ID_mvt"].to_numpy())
    sha = data.write_submission(ser, a.out)
    print("wrote", a.out)
    print("sha256", sha)
    print("n", len(ser), "mean", float(pred.mean()), "median", float(np.median(pred)))
    # The repo-code tarball carries `code/` only, so `results/` need not exist:
    # the K2 rank kernel lost its whole 75-minute fit to a FileNotFoundError here
    # after `write_submission` had already succeeded (exp-027 logs/prc-k2-rank.log).
    (EXP / "results").mkdir(parents=True, exist_ok=True)
    (EXP / "results" / "submission.json").write_text(json.dumps(
        dict(path=a.out, sha256=sha, rounds=a.rounds, clip_z=a.clip_z, seeds=a.seeds,
             seed_start=a.seed_start, extras=os.environ.get("PRC_EXTRAS", ""),
             ghost_mode=a.ghost_mode, main_model=a.main_model, exact_rounds=a.exact_rounds,
             params=json.loads(a.params), n=int(len(ser)), mean=float(pred.mean())), indent=1))


if __name__ == "__main__":
    main()
