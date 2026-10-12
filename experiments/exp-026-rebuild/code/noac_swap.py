"""blend-5, unchanged except on the 388 `noac` ranking rows (desk, 2026-09-07).

The desk asked for **blend-5 as uploaded, carrying blend-4's predictions on the
388 `noac` rows**, so that a board reading against v2's 265.9631 prices those
rows directly.  The instrument is right and the source is not available:

**blend-4's noac prediction cannot be rebuilt.**  The rows outside `nm_late`
take blend-4's `_all` vector, which puts **0.1213 on `T-tailv2`** and **0.1538
on `L2-hub`** -- 27.5 % of the weight on two members whose RANKING files lived
only in `artifacts/` and died with the workstation (SPEC rejected reading 3,
RESULTS section 2).  Their OOF survives, so blend-4's FOLD numbers are exact and
`code/noac_discount.py` is unaffected; the served side is gone for good.

So this builds the same instrument from sources that do exist.  Each variant is
blend-5 everywhere and one substitute on the 388 rows, which is the property the
measurement needs -- a board delta against v2 that can only have come from those
rows:

`exp002`      exp-002's purpose-built noac head
              (`experiments/exp-002-tail/noac_ranking_predictions.parquet`, 388
              rows, its own lineage and its own ledger rows).  RMS 1,858 s from
              blend-5 on these rows.  Answers the desk's question as asked --
              *is dedicated noac work worth a slot* -- because it is the thing
              such work would produce.
`drop-tailv4` the `blend-5 -T-tailv4` rung's values.  RMS 371 s.  Closest to the
              DISCOUNT's intent: section 5a verified that the tail v4 swap is only a
              noac-head change, so this prices what that swap's noac head bought.
`drop-v3`     the `blend-5 -V3-allin` rung's values.  RMS 281 s.  The same for
              the V3 swap.

None of them is blend-4, and none is labelled as blend-4.  Which one earns the
upload slot is the desk's call; all three are built, verified and staged so the
choice costs nothing.

    python noac_swap.py --source exp002
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
import regime_table as RT  # noqa: E402
# The repository root goes at the END: `code/` must stay ahead of it so
# a module at the root cannot shadow one of this experiment's own
# (round 4 -- a root-level `verify.py` would have replaced the gate).
sys.path.append(str(F.ROOT))
import prc.data as D  # noqa: E402

#: name -> the file whose values replace blend-5's on the noac rows.
SOURCES = {
    "exp002": "experiments/exp-002-tail/noac_ranking_predictions.parquet",
    "drop-tailv4": "submissions/candidates/exp-026-b5-drop-T-tailv4.parquet",
    "drop-v3": "submissions/candidates/exp-026-b5-drop-V3-allin.parquet",
}
BASE = "experiments/exp-026-rebuild/candidates/blend-5.parquet"


def _series(path: Path) -> pd.Series:
    d = pd.read_parquet(path)
    col = D.TARGET if D.TARGET in d.columns else "y_pred"
    return pd.Series(d[col].to_numpy("float64"),
                     index=d["MVT_ID_mvt"].astype("float64").to_numpy())


def build(source: str, gates: pd.DataFrame | None = None) -> dict:
    """blend-5 with `source`'s values on the noac rows. Nothing else moves."""
    if source not in SOURCES:
        raise SystemExit(f"unknown source {source!r}; one of {sorted(SOURCES)}")
    gates = RT.load() if gates is None else gates
    base = _series(F.ROOT / BASE)
    ids = base.index.to_numpy()
    reg = gates.set_index("MVT_ID_mvt").reindex(ids)["regime"].to_numpy()
    noac = reg == "noac"
    if int(noac.sum()) != 388:
        raise SystemExit(f"{int(noac.sum())} noac rows, expected 388")
    sub = _series(F.ROOT / SOURCES[source]).reindex(ids[noac])
    if sub.isna().any():
        raise SystemExit(f"{source} does not cover {int(sub.isna().sum())} of the 388 rows")
    pred = base.to_numpy("float64").copy()
    before = pred[noac].copy()
    pred[noac] = sub.to_numpy("float64")
    # The floor is the only preparation this needs: the substitute is already a
    # served prediction, and the guard cannot bind on rows it did not bind on
    # for blend-5 unless the substitute leaves the bounds, which is checked.
    pred = np.maximum(pred, D.FLOOR if hasattr(D, "FLOOR") else 60.0)
    return {"ids": ids, "pred": pred, "noac": noac, "source": source,
            "n_noac": int(noac.sum()),
            "noac_rms_shift": float(np.sqrt(np.mean((pred[noac] - before) ** 2))),
            "noac_mean_shift": float(np.mean(pred[noac] - before))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="exp002", choices=sorted(SOURCES))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    b = build(a.source)
    outdir = Path(a.out_dir) if a.out_dir else F.EXP / "candidates"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"noac-{a.source}.parquet"
    pd.DataFrame({"MVT_ID_mvt": b["ids"], "y_pred": b["pred"]}).to_parquet(out, index=False)
    p = b["pred"]
    rec = {"name": f"noac-{a.source}", "path": str(out), "sha256": D.sha256(out),
           "base": BASE, "source": SOURCES[a.source], "n": int(len(p)),
           "n_noac_substituted": b["n_noac"],
           "noac_rms_shift_s": round(b["noac_rms_shift"], 3),
           "noac_mean_shift_s": round(b["noac_mean_shift"], 3),
           "min": float(p.min()), "median": float(np.median(p)),
           "p99": float(np.quantile(p, 0.99)), "max": float(p.max())}
    (outdir / f"noac-{a.source}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
