"""Build `exp-016-stack-4.parquet`: the stacker anchored on blend-4.

Differs from `submit.py` in one thing, and it is the thing that decides the
result: the residual is taken from **blend-4's fitted NNLS**, not from the flat
average. Under blend-4's member set flat is 1.92 s worse than the blend, so a
flat-anchored stacker starts in a hole it does not climb out of -- measured at
+4.11 s against the blend, against **-0.50 s** for the same stacker anchored on
the blend (`analysis/blend4_rotation.csv`).

Serving construction:

  anchor at serve time = blend-4's per-regime NNLS fitted on ALL THREE folds'
                         OOF and applied to the members' RANKING predictions
  stacker              = fitted on the three folds' ROTATED anchors, so its
                         training anchor is out of sample on every row
  noac rows            = fall back to the anchor, as in stack-1..3

The ranking inputs are exactly blend-4's own (`blend-config-4.json`), so this
candidate inherits blend-4's serving choices and no others.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blend4 as B4  # noqa: E402
import members as M  # noqa: E402
import stack as S  # noqa: E402
import verify_r1 as V  # noqa: E402
sys.path.insert(0, str(M.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402
sys.path.insert(0, str(M.ROOT / "experiments" / "exp-001-retro-v1"))
import guard as guard_mod  # noqa: E402

OUT_PATH = M.ROOT / "submissions" / "candidates" / "exp-016-stack-4.parquet"
ROTATED = {"july_loo": 288.846, "december": 207.266, "january": 340.295}
BOARD = 312.677


def main() -> None:
    M.config_from_argv()
    bad = M.verify(require_ranking=True)
    if bad:
        raise SystemExit("member set is not servable:\n  " + "\n  ".join(bad))

    data = {f: M.fold_frame(f) for f in M.FOLDS}
    fit = pd.concat([data[f] for f in M.FOLDS], ignore_index=True)
    rk = M.ranking_frame()
    print(f"[submit4] fit {fit.shape} ranking {rk.shape}", flush=True)

    # anchor on the fit rows: the ROTATED blend, out of sample on every fold
    anchor_fit = np.concatenate([
        B4.fit_blend4(pd.concat([data[h] for h in M.FOLDS if h != g], ignore_index=True),
                      data[g])[0] for g in M.FOLDS])
    # anchor at serve time: the blend fitted on all three folds, applied to ranking
    anchor_rk, w6, w5 = B4.fit_blend4(fit, rk)
    print(f"[submit4] blend-4 anchor: nm_late weights {np.round(w6, 4).tolist()}, "
          f"elsewhere {np.round(w5, 4).tolist()}", flush=True)

    preds = [B4.arm_anchored(fit, rk, s, V.NOGATE, anchor_fit, anchor_rk) for s in V.SEEDS]
    stacked = np.maximum(np.mean(preds, axis=0), B.FLOOR)

    reg = rk["regime"].to_numpy()
    off = reg == "noac"
    pred = np.where(off, np.maximum(anchor_rk, B.FLOOR), stacked)
    a = rk["d_takeoff_aobt"].to_numpy("float64")
    lo, hi = -20657.0, 9069.0
    n_touch = guard_mod.count_touched(pred, a, lo, hi)
    pred = guard_mod.apply(pred, a, lo, hi, floor=B.FLOOR)
    print(f"[submit4] {int(off.sum())} noac rows on the anchor; guard touched {n_touch}",
          flush=True)

    sha = D.write_submission(pd.Series(pred, index=rk["MVT_ID_mvt"].to_numpy("float64")),
                             OUT_PATH)
    tpl = D.load_submitting()
    got = pd.read_parquet(OUT_PATH)
    rec = {
        "path": str(OUT_PATH.relative_to(M.ROOT)), "sha256": sha, "n": int(len(got)),
        "members": list(M.NAMES), "anchor": "blend-4 per-regime NNLS (not the flat average)",
        "stacker": "a5 members-only, 5-seed bag, 300 rounds, noac on the anchor",
        "fit_rows": int(len(fit)), "fit_folds": list(M.FOLDS),
        "rotated": ROTATED, "board_estimate_rotated": BOARD,
        "beats": {"blend-4": 313.173, "gain": round(313.173 - BOARD, 3)},
        "ids_match_template": bool(np.array_equal(got[D.ID].to_numpy("float64"),
                                                  tpl[D.ID].to_numpy("float64"))),
        "min": float(pred.min()), "median": float(np.median(pred)),
        "p99": float(np.quantile(pred, 0.99)), "max": float(pred.max()),
        "n_below_floor": int((pred < B.FLOOR - 1e-9).sum()),
        "n_nonfinite": int((~np.isfinite(pred)).sum()),
        "guard_touched": int(n_touch), "pins": M.PINS.name,
        "ranking_inputs": {n: D.sha256(M.ROOT / p)[:16] for n, p in M.RANKING.items()},
    }
    other = pd.read_parquet(M.ROOT / "submissions" / "candidates" / "exp-012-blend-4.parquet") \
        if (M.ROOT / "submissions" / "candidates" / "exp-012-blend-4.parquet").exists() else None
    if other is not None:
        d = got[D.TARGET].to_numpy("float64") - other[D.TARGET].to_numpy("float64")
        rec["vs blend-4"] = {"rms": float(np.sqrt((d ** 2).mean())), "max": float(d.max()),
                             "min": float(d.min()), "n_differ": int((d != 0).sum())}
    (M.EXP / "candidate-4.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2), flush=True)


if __name__ == "__main__":
    main()
