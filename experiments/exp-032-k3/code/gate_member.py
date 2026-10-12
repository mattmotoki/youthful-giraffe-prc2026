"""The repair: the window arm serves the bulk, the committed member serves the tail.

**Why this repair and not another.** `--arm window` improves `T-tailv4`'s
july_loo fold by +2.048 s and moves its `y > 3600` rows 7.33 s the WRONG way,
which fails the assignment's 1 s tail clause. `analysis/member-decompose.csv`
and `RESULTS.md` section 4 locate the damage: it is the `nm` head's 503 tail
rows (`nm_bulk` -25.2 s on 119, `nm_late` -14.3 s on 384), the `noac` head is
untouched at exactly 0.000 by construction, and the whole gain is the bulk.
That is the same shape exp-027 section 13 measured on the K1 slices -- "the arm
takes 1.0-1.8 s off `y <= 3600` on every fold while moving the rows above 3600 s
the wrong way" -- and `exp-027/SPEC.md` registered the repair for it before any
of these numbers existed: **"Gate the arm to `y <= 3600`. The tail lane owns
that population."**

`y` is a label, so the servable form of that rule gates on the BASE member's own
prediction, which is a function of inputs alone and is committed for every
ranking row (`published/T-tailv4-ranking.parquet`):

    K3g = window arm where base <= 3600 s, base elsewhere.

The threshold is the record's, not a tuned one. `--sweep` prints 1800 and 7200
beside it; both are read as sensitivity and neither is taken, because a
threshold chosen after seeing three fold numbers is a second cut on the same
data in exactly the sense `exp-030` section 4 and `exp-027` section 16e record.

    python code/gate_member.py            # the three fold vectors, and the sweep
    python code/gate_member.py --rank     # the ranking column
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

FOLDS = ("july_loo", "december", "january")
#: `exp-027/SPEC.md`, "Gate the arm to `y <= 3600`", and the same file's second
#: kill clause, which is written on `y > 3600`. Registered there before this
#: experiment existed.
THRESHOLD = 3600.0
SWEEP = (1800.0, 3600.0, 7200.0)
BASE_RANK = ROOT / "experiments/exp-012-blend2/published/T-tailv4-ranking.parquet"
WIN_RANK = EXP / "candidates" / "k3-tail-ranking.parquet"
OUT_RANK = EXP / "candidates" / "k3-tail-gated-ranking.parquet"
N_RANK = 344841


def gate(base: np.ndarray, win: np.ndarray, t: float = THRESHOLD) -> np.ndarray:
    return np.where(base <= t, win, base)


def folds() -> None:
    rows, sweep = [], []
    for fold in FOLDS:
        b = pd.read_parquet(EXP / "oof" / f"k3-tail-base-{fold}.parquet")
        w = pd.read_parquet(EXP / "oof" / f"k3-tail-window-{fold}.parquet")
        m = b.merge(w[["MVT_ID_mvt", "y_pred"]], on="MVT_ID_mvt",
                    suffixes=("_base", "_win"))
        if len(m) != len(b) or len(m) != len(w):
            raise SystemExit(f"{fold}: join covers {len(m)} of {len(b)}/{len(w)}")
        y = m["y_true"].to_numpy("float64")
        a = m["y_pred_base"].to_numpy("float64")
        c = m["y_pred_win"].to_numpy("float64")
        tail = y > 3600.0
        r = lambda p, k=None: float(np.sqrt(np.mean(((y - p) ** 2)[
            np.ones(len(y), bool) if k is None else k])))
        for t in SWEEP:
            p = gate(a, c, t)
            sweep.append({"fold": fold, "threshold": t, "n_window_served": int((a <= t).sum()),
                          "rmse": round(r(p), 4), "gain": round(r(a) - r(p), 4),
                          "le3600": round(r(p, ~tail), 4),
                          "gt3600": round(r(p, tail), 2),
                          "gt3600_gain": round(r(a, tail) - r(p, tail), 2),
                          "taken": t == THRESHOLD})
        p = gate(a, c)
        out = pd.DataFrame({"MVT_ID_mvt": m["MVT_ID_mvt"].to_numpy(),
                            "fold": fold, "y_true": y, "y_pred": p})
        path = EXP / "oof" / f"k3-tail-gated-{fold}.parquet"
        out.to_parquet(path, index=False)
        rows.append({"fold": fold, "n": int(len(y)),
                     "n_window_served": int((a <= THRESHOLD).sum()),
                     "base": round(r(a), 4), "window": round(r(c), 4),
                     "gated": round(r(p), 4),
                     "gain_window": round(r(a) - r(c), 4),
                     "gain_gated": round(r(a) - r(p), 4),
                     "gt3600_base": round(r(a, tail), 2),
                     "gt3600_window": round(r(c, tail), 2),
                     "gt3600_gated": round(r(p, tail), 2),
                     "gt3600_gain_window": round(r(a, tail) - r(c, tail), 2),
                     "gt3600_gain_gated": round(r(a, tail) - r(p, tail), 2),
                     "n_gt3600": int(tail.sum()),
                     "path": str(path.relative_to(ROOT)), "sha256": D.sha256(path)})
        print(f"[{fold}] base {rows[-1]['base']:9.4f}  window {rows[-1]['window']:9.4f}  "
              f"gated {rows[-1]['gated']:9.4f}   fold gain {rows[-1]['gain_gated']:+7.4f}   "
              f"tail {rows[-1]['gt3600_gain_gated']:+7.2f} s "
              f"(ungated {rows[-1]['gt3600_gain_window']:+7.2f})", flush=True)
    pd.DataFrame(sweep).to_csv(EXP / "analysis" / "gate-sweep.csv", index=False)
    (EXP / "analysis" / "member-gated.json").write_text(json.dumps(
        {"threshold": THRESHOLD, "sweep": SWEEP, "rows": rows}, indent=2))
    with pd.option_context("display.width", 200):
        print("\n" + pd.DataFrame(sweep).to_string(index=False))
    print(f"\n-> {(EXP / 'analysis' / 'member-gated.json').relative_to(ROOT)}")


def rank() -> None:
    b = pd.read_parquet(BASE_RANK)
    w = pd.read_parquet(WIN_RANK)
    for d, n in ((b, BASE_RANK.name), (w, WIN_RANK.name)):
        if len(d) != N_RANK:
            raise SystemExit(f"{n} has {len(d)} rows, expected {N_RANK}")
    m = b.rename(columns={"y_pred": "base"}).merge(
        w.rename(columns={"y_pred": "win"}), on="MVT_ID_mvt", how="inner")
    if len(m) != N_RANK:
        raise SystemExit(f"the ranking join covers {len(m)} of {N_RANK}")
    a = m["base"].to_numpy("float64")
    c = m["win"].to_numpy("float64")
    p = gate(a, c)
    if not np.isfinite(p).all():
        raise SystemExit("non-finite gated ranking predictions")
    out = pd.DataFrame({"MVT_ID_mvt": m["MVT_ID_mvt"].to_numpy(), "y_pred": p})
    out.to_parquet(OUT_RANK, index=False)
    rec = {"out": str(OUT_RANK.relative_to(ROOT)), "sha256": D.sha256(OUT_RANK),
           "n": int(len(out)), "threshold": THRESHOLD,
           "n_window_served": int((a <= THRESHOLD).sum()),
           "n_base_served": int((a > THRESHOLD).sum()),
           "base_ranking": str(BASE_RANK.relative_to(ROOT)),
           "base_ranking_sha256": D.sha256(BASE_RANK),
           "window_ranking": str(WIN_RANK.relative_to(ROOT)),
           "window_ranking_sha256": D.sha256(WIN_RANK),
           "mean": float(p.mean()), "min": float(p.min()), "max": float(p.max()),
           "n_above_3600": int((p > 3600).sum())}
    (EXP / "analysis" / "rank-gated.json").write_text(json.dumps(rec, indent=2))
    print(f"[rank] {rec['n_window_served']} rows on the window arm, "
          f"{rec['n_base_served']} on the committed member; sha256 {rec['sha256'][:8]}…")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", action="store_true")
    a = ap.parse_args()
    rank() if a.rank else folds()
