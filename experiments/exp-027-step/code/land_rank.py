"""Land the ranking-side K2 member, with the checks the serve path needs.

Sol round 3, blocking 3: the rank kernel writes a tagged `submission.parquet`
and `serve.py` reads `candidates/k2-window-ranking.parquet`; nothing joined the
two, and `land_k2.py` handles fold files only.  Sol round 3, blocking 2: a
partial or stale K2 ranking file would join in as nulls -- `serve_stack5`'s
member joins are left joins and LightGBM accepts missing features -- and the
candidate would still come out finite.  So the file is validated here, once,
before it can reach a candidate, and its SHA-256 is recorded for `serve.py` to
require.

Checks, all of them fatal:

  * exactly the submitting template's ID set, no missing and no extra;
  * IDs unique;
  * every prediction finite (the floor is NOT checked here: a member's ranking
    column is a raw model output and `serve_stack5.ranking_frame` floors it, as
    it does the committed F1-fresh column, which has 69 rows under the floor);
  * float64.

Writes `candidates/k2-window-ranking.parquet` (`MVT_ID_mvt`, `y_pred`) and
`analysis/k2-ranking.json` with the hash, the row count, the distribution and
the kernel's own configuration record.

    python land_rank.py [--from <parquet>]
"""
from __future__ import annotations

import os

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

KERNEL = EXP / "kernels" / "f1-window-rank" / "prc-k2-rank.py"
OUT = EXP / "candidates" / "k2-window-ranking.parquet"
REC = EXP / "analysis" / "k2-ranking.json"


def find() -> Path:
    """The kernel's ranking output, wherever `pull_when_done.sh` put it."""
    cand = sorted(set(EXP.glob("oof/k2rank*submission*.parquet"))
                  | set(EXP.glob("results/*submission*.parquet"))
                  | set(EXP.glob("oof/*k2-rank*.parquet")))
    if not cand:
        raise SystemExit(f"no ranking output found; pull {kaggle_user()}/prc-k2-rank first")
    if len(cand) > 1:
        raise SystemExit(f"{len(cand)} candidate files: {[p.name for p in cand]}")
    return cand[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=None)
    a = ap.parse_args()
    src = Path(a.src) if a.src else find()
    d = pd.read_parquet(src)
    col = "y_pred" if "y_pred" in d.columns else D.TARGET
    if D.ID not in d.columns or col not in d.columns:
        raise SystemExit(f"{src.name}: columns {list(d.columns)}")
    ids = d[D.ID].astype("float64").to_numpy()
    v = d[col].astype("float64").to_numpy()

    tmpl = D.load_submitting()[D.ID].astype("float64").to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise SystemExit(f"{src.name}: {len(ids) - len(np.unique(ids))} duplicate ids")
    missing = int(np.isin(tmpl, ids, invert=True).sum())
    extra = int(np.isin(ids, tmpl, invert=True).sum())
    if missing or extra:
        raise SystemExit(f"{src.name}: id mismatch, {missing} missing, {extra} extra")
    if not np.isfinite(v).all():
        raise SystemExit(f"{src.name}: {int((~np.isfinite(v)).sum())} non-finite predictions")
    # NOT a fatal check.  A member's ranking column is a raw model output; the
    # floor and the residual guard are applied downstream by
    # `serve_stack5.ranking_frame`, exactly as they are for the committed
    # F1-fresh column, which has 69 rows below the floor and a minimum of 30.0.
    below = int((v < B.FLOOR).sum())

    order = pd.Series(v, index=ids).reindex(tmpl)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({D.ID: tmpl, "y_pred": order.to_numpy()}).to_parquet(OUT, index=False)
    sha = D.sha256(OUT)
    # The producer's identity, and it is NOT guessed from whatever json is in
    # `results/`: the fold kernels write records there too, and a fold kernel's
    # `port.json` attached to a ranking column would be a false provenance.  A
    # `port.json` is accepted only if its own `cfg.tag` says it is this kernel's;
    # otherwise the kernel script and its log carry the provenance, by hash.
    port = {"kernel": str(KERNEL.relative_to(ROOT)), "kernel_sha256": D.sha256(KERNEL)}
    log = EXP / "logs" / "prc-k2-rank.log"
    if log.exists():
        port["log"] = str(log.relative_to(ROOT))
        port["log_sha256"] = D.sha256(log)
    for q in sorted(EXP.glob("results/*.json")):
        try:
            j = json.loads(q.read_text())
        except Exception:                                    # noqa: BLE001
            continue
        if isinstance(j, dict) and j.get("cfg", {}).get("tag") == "kaggle-k2-rank":
            port.update({"file": q.name, "cfg": j["cfg"], "versions": j.get("versions"),
                         "gpu": j.get("gpu"), "git_commit": j.get("git_commit")})
            break
    rec = {"source": str(src.relative_to(ROOT)), "out": str(OUT.relative_to(ROOT)),
           "sha256": sha, "n": int(len(tmpl)),
           "mean": float(order.mean()), "median": float(np.median(order.to_numpy())),
           "min": float(order.min()), "max": float(order.max()),
           "n_above_3600": int((order.to_numpy() > 3600).sum()),
           "n_below_floor": below, "floor": float(B.FLOOR),
           "kernel": port}
    REC.write_text(json.dumps(rec, indent=2))
    print(f"[land-rank] {src.name} -> {OUT.name}  sha256 {sha[:8]}…  n={rec['n']}  "
          f"mean {rec['mean']:.1f}  median {rec['median']:.1f}  "
          f"min {rec['min']:.1f}  max {rec['max']:.1f}  >3600 {rec['n_above_3600']}  "
          f"below the floor {below} (F1-fresh's column has 69; the floor is applied "
          f"downstream by serve_stack5.ranking_frame)")
    print(f"[land-rank] record: {REC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
