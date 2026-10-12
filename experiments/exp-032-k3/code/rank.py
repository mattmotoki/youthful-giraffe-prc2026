"""The K3 ranking-side member: `T-tailv4` on all twelve months, window block on.

`experiments/exp-012-blend2/tail_ranking_v4.py` with two lines added — the block
built for the training frame and for the ranking frame — and nothing else
changed. The pin, the module version check and the fit are that file's.

    PRC_THREADS=6 python code/rank.py            # the K3 ranking column
    PRC_THREADS=6 python code/rank.py --identity # flag off: must equal the committed file

`--identity` is the serving-path control that costs no new idea: with the block
off this script must reproduce
`experiments/exp-012-blend2/published/T-tailv4-ranking.parquet` byte for byte,
because the member is deterministic and CPU-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]

THREADS = int(os.environ.get("PRC_THREADS", "6"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import prc.data as D  # noqa: E402
from prc.tail_model import TailModel  # noqa: E402
import wblock  # noqa: E402

NOAC_PIN = "v4"
PUBLISHED = ROOT / "experiments/exp-012-blend2/published/T-tailv4-ranking.parquet"
OUT = EXP / "candidates" / "k3-tail-ranking.parquet"
N_RANK = 344841
#: The committed `T-tailv4-ranking.parquet` was fitted at
#: `prc.tail_model.DEFAULT_N_JOBS` = 16 (`exp-012-blend2/tail_ranking_v4.py`
#: passes no `n_jobs`); this worker's thread budget is 6. LightGBM's
#: `deterministic` flag promises reproducibility for a GIVEN thread count, so
#: `code/threadprobe.py` measured what a different one costs: on `july_loo`,
#: three threads against six give the same fold RMSE to four decimals
#: (301.6430) and a maximum row difference of **1.68e-9 s** -- summation order,
#: nine orders below the 0.1025 s rotation floor. The identity tolerance is set
#: from that measurement rather than at 0.0, and it is still six orders below a
#: second of taxi time.
IDENT_TOL = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    train = D.load_months(D.MONTHS)
    print(f"[k3-rank] train {train.shape} in {time.time()-t0:.0f}s "
          f"(noac head pinned to {NOAC_PIN}, identity={a.identity})", flush=True)
    wtr = None if a.identity else wblock.build(train)
    if wtr is not None:
        print(f"[k3-rank] train block {wtr.shape} at {time.time()-t0:.0f}s", flush=True)
    model = TailModel().fit(train, n_jobs=THREADS, noac_head=NOAC_PIN, window=wtr)
    del wtr, train
    print(f"[k3-rank] fitted at {time.time()-t0:.0f}s "
          f"({len(model.window_cols)} window columns)", flush=True)

    rk = D.load_ranking()
    wrk = None if a.identity else wblock.build(rk)
    parts = []
    for m, g in rk.groupby("month", sort=True):
        w = None if wrk is None else wrk.reindex(g.index)
        p = model.predict(g, window=w)
        p["MVT_ID_mvt"] = g["MVT_ID_mvt"].reindex(p.index)
        parts.append(p[["MVT_ID_mvt", "y_pred"]])
        print(f"[k3-rank] {m}: {len(p)} rows at {time.time()-t0:.0f}s", flush=True)
    out = pd.concat(parts, ignore_index=True)
    if len(out) != N_RANK:
        raise SystemExit(f"{len(out)} ranking rows, expected {N_RANK}")
    if not np.isfinite(out["y_pred"].to_numpy("float64")).all():
        raise SystemExit("non-finite ranking predictions")
    dest = OUT.parent / ("k3-tail-ranking-identity.parquet" if a.identity else OUT.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    sha = D.sha256(dest)
    rec = {"out": str(dest.relative_to(ROOT)), "sha256": sha, "n": int(len(out)),
           "identity": bool(a.identity), "threads": THREADS,
           "window_columns": len(model.window_cols),
           "lightgbm": __import__("lightgbm").__version__,
           "tail_model_version": __import__("prc.tail_model", fromlist=["VERSION"]).VERSION,
           "wall_s": round(time.time() - t0, 1)}
    if a.identity:
        pub = pd.read_parquet(PUBLISHED)
        m = out.merge(pub, on="MVT_ID_mvt", suffixes=("", "_pub"), how="inner")
        if len(m) != N_RANK:
            raise SystemExit(f"identity join covers {len(m)} of {N_RANK}")
        md = float(np.abs(m["y_pred"].to_numpy() - m["y_pred_pub"].to_numpy()).max())
        rec["max_abs_diff_vs_published"] = md
        rec["identity_tolerance"] = IDENT_TOL
        print(f"[identity] max abs difference against {PUBLISHED.name}: {md:.6g} "
              f"(tolerance {IDENT_TOL:g}, {'PASS' if md <= IDENT_TOL else 'FAIL'})", flush=True)
        if md > IDENT_TOL:
            raise SystemExit("the ranking path does not reproduce the committed T-tailv4")
    (EXP / "analysis" / f"rank{'-identity' if a.identity else ''}.json").write_text(
        json.dumps(rec, indent=2))
    print(f"[k3-rank] wrote {dest.relative_to(ROOT)} sha256 {sha[:8]}… "
          f"in {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
