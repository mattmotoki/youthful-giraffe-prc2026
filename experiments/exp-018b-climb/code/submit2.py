"""exp-018b: fit on all twelve 2025 months through model2 and write a candidate.

Why not `exp-018-fresh/code/submit.py --seeds n`: that script averages n whole
pipelines (a fresh exact head and a fresh ghost head per seed), while `model2`
bags only the main head, CatBoost and the ghost head and fits the exact head
once. Those are different estimators. A blend owner fits weights on an OOF and
serves a candidate, so the two must come from ONE code path; this script is the
serving half of `run2.py`.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

_F = Path(__file__).resolve().parents[2] / "exp-018-fresh" / "code"
sys.path.insert(0, str(_F)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D            # noqa: E402
import model as M              # noqa: E402
import model2 as M2            # noqa: E402
from prc import data           # noqa: E402

EXP = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=1200)
    ap.add_argument("--clip-z", type=float, default=6000.0)
    ap.add_argument("--extras", nargs="*", default=[])
    ap.add_argument("--drop-unreached", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    t0 = time.time()
    train = D.load(data.MONTHS)
    rank = D.load(["2026-01", "2026-07"])
    train, (rank,) = D.add_encodings(train, [rank])
    print(f"prepared in {time.time()-t0:.0f}s  train={len(train)} rank={len(rank)}", flush=True)

    pred, info = M2.fit_predict(train, rank, "rank", seeds=a.seeds, rounds=a.rounds,
                                clip_z=a.clip_z, extras_kinds=tuple(a.extras),
                                drop_unreached=a.drop_unreached, main_model="both")
    ser = pd.Series(pred, index=rank["MVT_ID_mvt"].to_numpy())
    sha = data.write_submission(ser, a.out)
    print("wrote", a.out)
    print("sha256", sha)
    print("n", len(ser), "mean", float(pred.mean()), "median", float(np.median(pred)),
          "p99", float(np.percentile(pred, 99)), "max", float(pred.max()))
    tag = a.tag or Path(a.out).stem
    (EXP / "results" / f"submission-{tag}.json").write_text(json.dumps(
        dict(path=a.out, sha256=sha, seeds=a.seeds, rounds=a.rounds, clip_z=a.clip_z,
             extras=list(a.extras), drop_unreached=a.drop_unreached,
             n_added=info["n_added"], added=info["added"], n_train=info["n_train"],
             n_dropped=info["n_dropped"], n=int(len(ser)), mean=float(pred.mean())), indent=1))


if __name__ == "__main__":
    main()
