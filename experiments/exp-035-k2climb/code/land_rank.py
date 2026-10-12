"""Land the climbed ranking column, with the checks `serve.py` requires.

exp-027's `code/land_rank.py` validates one kernel's ranking output. This one
does the same checks and adds the arithmetic arm 1 needs: the committed K2
ranking column is the mean of seeds 7 and 20, so the three-seed column is

    mean(p7, p20, p33) = (2 * committed + p33) / 3

exactly -- `prc.data.write_submission` does no clipping, so the committed file is
the raw mean and this is an identity, not an approximation. The identity is
checked, not assumed: both inputs must carry the template's ID set and the
combination is done on the template's order.

Checks, all fatal: exactly the submitting template's ID set, no missing and no
extra; ids unique; every prediction finite (the floor is NOT checked -- a
member's ranking column is a raw model output and `serve_stack5.ranking_frame`
floors it, as it does the committed F1-fresh column); float64.

    python land_rank.py --name seeds --new candidates/k2s2-rank.parquet \
        --with-committed 2

With `--with-committed 0` there is no arithmetic and the run is the validation
alone: arm 3's ranking column comes whole out of one kernel, and this is what
gives it the record `serve.py` pins it against.

    python land_rank.py --name mod --new candidates/k2m-rank.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

K2_RANK = ROOT / "experiments/exp-027-step/candidates/k2-window-ranking.parquet"
K2_REC = ROOT / "experiments/exp-027-step/analysis/k2-ranking.json"


def column(p: Path, ids: np.ndarray) -> np.ndarray:
    d = pd.read_parquet(p)
    idc = "MVT_ID_mvt" if "MVT_ID_mvt" in d.columns else D.ID
    vc = "y_pred" if "y_pred" in d.columns else D.TARGET
    s = pd.Series(d[vc].to_numpy("float64"), index=d[idc].to_numpy("float64"))
    if s.index.duplicated().any():
        raise SystemExit(f"{p.name}: duplicate ids")
    missing = np.setdiff1d(ids, s.index.to_numpy())
    extra = np.setdiff1d(s.index.to_numpy(), ids)
    if len(missing) or len(extra):
        raise SystemExit(f"{p.name}: {len(missing)} missing, {len(extra)} extra ids")
    v = s.reindex(ids).to_numpy("float64")
    if not np.isfinite(v).all():
        raise SystemExit(f"{p.name}: {int((~np.isfinite(v)).sum())} non-finite predictions")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="the arm; writes candidates/<name>-ranking.parquet")
    ap.add_argument("--new", required=True, nargs="+",
                    help="the new seeds' ranking parquets, under candidates/; more "
                         "than one is averaged with equal weight, which is what a "
                         "k-seed ranking column fitted one seed per session is")
    ap.add_argument("--with-committed", type=int, default=0,
                    help="combine with the committed K2 column, which is the mean of this many seeds")
    a = ap.parse_args()

    ids = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    news = []
    for one in a.new:
        q = Path(one)
        news.append(q if q.is_absolute() else EXP / q)
    cols = [column(q, ids) for q in news]
    pnew = np.mean(cols, axis=0)
    parts = [{"path": str(q.relative_to(ROOT)), "sha256": D.sha256(q), "weight": 1}
             for q in news]
    if len(news) > 1:
        print(f"[land_rank] {len(news)} new seeds averaged with equal weight: "
              + ", ".join(f"{q.name} {c.mean():.4f}" for q, c in zip(news, cols))
              + f" -> {pnew.mean():.4f}")
    if a.with_committed:
        k = a.with_committed
        pold = column(K2_RANK, ids)
        rec = json.loads(K2_REC.read_text())
        if D.sha256(K2_RANK) != rec["sha256"]:
            raise SystemExit("the committed K2 ranking column does not match its own record")
        if rec["n"] != len(ids):
            raise SystemExit(f"the committed K2 ranking column has {rec['n']} rows")
        pred = (k * pold + pnew) / (k + 1)
        parts.insert(0, {"path": str(K2_RANK.relative_to(ROOT)),
                         "sha256": rec["sha256"], "weight": k})
        if len(news) > 1:
            raise SystemExit("--with-committed combines ONE new column with the "
                             "committed mean; pass one --new or drop it")
        print(f"[land_rank] mean of {k} committed seeds and 1 new = {k + 1} seeds; "
              f"committed mean {pold.mean():.4f}, new {pnew.mean():.4f}, "
              f"combined {pred.mean():.4f}")
    else:
        pred = pnew

    out = EXP / "candidates" / f"{a.name}-ranking.parquet"
    pd.DataFrame({"MVT_ID_mvt": ids, "y_pred": pred}).to_parquet(out, index=False)
    sha = D.sha256(out)
    rr = {"out": f"experiments/exp-035-k2climb/candidates/{a.name}-ranking.parquet",
          "sha256": sha, "n": int(len(ids)), "n_seeds": a.with_committed + len(news),
          "parts": parts,
          "mean": float(pred.mean()), "median": float(np.median(pred)),
          "min": float(pred.min()), "max": float(pred.max()),
          "n_above_3600": int((pred > 3600).sum()),
          "n_below_floor": int((pred < 60.0).sum()), "floor": 60.0}
    (EXP / "analysis" / f"{a.name}-ranking.json").write_text(json.dumps(rr, indent=1))
    print(f"[land_rank] wrote {rr['out']}  sha256 {sha[:8]}…  "
          f"{rr['n']} rows, {rr['n_seeds']} seeds, {rr['n_above_3600']} above 3600 s")


if __name__ == "__main__":
    main()
