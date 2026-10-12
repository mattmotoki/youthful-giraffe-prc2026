"""The served blend: `p = v10 + s*(stack-5 - v10)`, two hashed files and a scalar.

RESULTS.md section 2a, next step 2.  There is no model, no frame and no block
between the bytes and the answer: two committed parquet files are read by their
recorded SHA-256, put in `submitting.parquet`'s row order, and combined at ONE
amplitude that is stated in the record before the file is written.

The amplitude is passed on the command line and copied into the record with the
reason for it.  It is not fitted here, and nothing in this file reads a board
score: `assignments`/desk order, "no amplitude may be chosen or revised on a
board reading".

    python serve.py --s 0.27 --why "..." --name exp-038-v10-stack5-blend
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
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

DEST = ROOT / "submissions" / "candidates"
BASE = ("submissions/uploaded/youthful-giraffe_v10.parquet",
        "ad962763236e9b40ace45a5dac12945568bbce441ed0547ff8e814d248c72c78")
SRC = ("submissions/candidates/exp-026-stack-5.parquet",
       "94f18e47a73265a6db66811ededfc3c935fb2a37fac04bf0cd34d45af708a2a1")


def column(rel: str, sha: str) -> np.ndarray:
    """One source column in `submitting.parquet`'s row order, hash checked."""
    p = ROOT / rel
    got = D.sha256(p)
    if got != sha:
        raise SystemExit(f"{p.name}: sha256 {got[:8]} against the record's {sha[:8]}")
    d = pd.read_parquet(p)
    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    ids = d["MVT_ID_mvt"].to_numpy("float64")
    if np.array_equal(ids, tmpl):
        return d[D.TARGET].to_numpy("float64")
    v = d.set_index("MVT_ID_mvt")[D.TARGET].reindex(tmpl)
    if v.isna().any():
        raise SystemExit(f"{p.name}: {int(v.isna().sum())} template rows are missing")
    return v.to_numpy("float64")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--why", required=True, help="where the amplitude comes from")
    ap.add_argument("--name", default="exp-038-v10-stack5-blend")
    a = ap.parse_args()
    if not 0.0 <= a.s <= 1.0:
        raise SystemExit(f"the amplitude must be in [0, 1], not {a.s}")

    base = column(*BASE)
    src = column(*SRC)
    v = base + a.s * (src - base)
    if not np.isfinite(v).all():
        raise SystemExit("the blend is not finite")
    if v.min() < B.FLOOR:
        raise SystemExit(f"the blend goes below the floor: {v.min()}")

    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    out = pd.DataFrame({"MVT_ID_mvt": tmpl, D.TARGET: v.astype("float64")})
    path = DEST / f"{a.name}.parquet"
    out.to_parquet(path, index=False)
    rec = {"name": a.name, "path": str(path.relative_to(ROOT)),
           "rule": "p = v10 + s * (stack-5 - v10)", "s": a.s, "s_reason": a.why,
           "base_file": BASE[0], "base_sha256": BASE[1],
           "source_file": SRC[0], "source_sha256": SRC[1],
           "sha256": D.sha256(path),
           "n_rows": int(len(out)), "n_moved_vs_v10": int((v != base).sum()),
           "rms_move_vs_v10": round(float(np.sqrt(((v - base) ** 2).mean())), 4),
           "max_abs_move_vs_v10": round(float(np.abs(v - base).max()), 4),
           "min": float(v.min()), "max": float(v.max())}
    (EXP / "analysis" / f"served-{a.name}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
