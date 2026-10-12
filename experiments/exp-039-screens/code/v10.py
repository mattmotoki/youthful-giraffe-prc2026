"""The one measurement base every screen in this experiment reads: **v10**.

`assignments/screens.md` fixes the contract: residual alignment against the v10
vector, `submissions/uploaded/youthful-giraffe_v10.parquet`.  That file is the
RANKING side and carries no labels, so a residual cannot be formed on it.  The
fold side of the same candidate is exp-036 arm 2 -- refit-bulk served at the eight
mask airports, v7 held at EDDF and EHAM -- and the served ranking file it produced
is byte-identical to the uploaded v10 (`analysis/served-exp-036-refitbulk-mask.json`,
SHA-256 `ad962763...`, checked here).  So:

    p10[fold]  = np.where(airport in KEEP, refit-bulk, v7)      the fold-side v10
    r          = y_true - p10                                   the current residual

Nothing is refitted.  `exp-036/code/bases.load` gates stack-5, v6, v7, K2 and
refit-bulk against their own published rotations first; `load()` here adds arm 2's
own three published fold numbers and its clean board as a gate, so a screen cannot
score against a vector that has silently moved.

`d_takeoff_aobt` and `d_aobt_lobt` come from `exp-028/code/panel.load_fold`, which
exp-036's reference drops.  Both are legal inputs: `docs/input-rule.md` and
`prc.data.mask_like_ranking` blank departure `BLOCK_TIME_UTC_mvt` and
`TAXITIME_SEC_mvt` only, so the take-off clock and every `_flt` column are inputs
and `AOBT_3_flt` is not the masked B.

    python v10.py        # prints the gate table
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-036-mask/code"))
import bases as Bs  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
import panel as P  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = Bs.FOLDS
Sc = Bs.Sc
GATE = 1e-3

#: exp-036 `analysis/mask.json`: the eight airports where the K2-minus-v7 change
#: is served, and the two held at v7.  Read from the file, not retyped.
_MASK = json.loads((ROOT / "experiments/exp-036-mask/analysis/mask.json").read_text())
KEEP = tuple(_MASK["keep"])
HOLD = tuple(_MASK["hold_at_v7"])

#: exp-036 arm 2's published rotation and the uploaded file it produced.
PUBLISHED = {"july_loo": 284.1572, "december": 204.9218, "january": 218.1506}
#: SHA-256 over the fold-side vector's own ordered bytes -- `MVT_ID_mvt` then
#: `p10`, both float64.  An RMSE gate cannot tell two different vectors apart
#: (Sol round 1, blocking finding 1); this can.  Pinned from the first gated run
#: and checked on every later one.
VECTOR_SHA = {
    "july_loo": "69c90a5af928de6b4c480159be605068dbc55421d04389bed74be45a280af017",
    "december": "c1452e90ed097b0a28f8d6a5f3cbf9f691f6d9d9d64698d559ec84e379b5c9a2",
    "january": "edf0ac093e6b6e260fb64975f1e27c9fa2165fc6c8f0eddaecbdff379e64bc65",
}
PUBLISHED_BOARD = 257.0248
UPLOADED = ROOT / "submissions/uploaded/youthful-giraffe_v10.parquet"
UPLOADED_SHA = "ad962763236e9b40ace45a5dac12945568bbce441ed0547ff8e814d248c72c78"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(check_upload: bool = True) -> dict[str, pd.DataFrame]:
    """Per fold: the panel, the fold-side v10 vector and the current residual.

    Columns: `MVT_ID_mvt airport t regime nm_missing noac y_true p10 r
    d_takeoff_aobt d_aobt_lobt clean` -- `clean` is False on the fold's
    unreached-rollover rows, which `prc.data.rmse_clean` drops.
    """
    if check_upload:
        got = _sha256(UPLOADED)
        if got != UPLOADED_SHA:
            raise SystemExit(f"{UPLOADED.name} is {got[:12]}..., not the v10 this "
                             f"experiment measures ({UPLOADED_SHA[:12]}...)")
    ref, vec, _ = Bs.load()
    out, clean = {}, {}
    for f in FOLDS:
        d = ref[f]
        keep = np.isin(d["airport"].astype(str).to_numpy(), KEEP)
        p10 = np.where(keep, vec["refit-bulk"][f], vec["v7"][f])
        pan = P.load_fold(f)[["MVT_ID_mvt", "d_takeoff_aobt", "d_aobt_lobt"]]
        pan["MVT_ID_mvt"] = pan["MVT_ID_mvt"].astype("float64")
        m = d[["MVT_ID_mvt", "airport", "t", "regime", "nm_missing", "noac",
               "y_true"]].copy()
        m["p10"] = p10
        m = m.merge(pan, on="MVT_ID_mvt", how="left", validate="one_to_one")
        if len(m) != len(d) or not np.array_equal(m["MVT_ID_mvt"].to_numpy(),
                                                  d["MVT_ID_mvt"].to_numpy()):
            raise SystemExit(f"{f}: the panel join lost or reordered rows")
        m["r"] = m["y_true"].to_numpy("float64") - m["p10"].to_numpy("float64")
        m["clean"] = ~m["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
        m["fold"] = f
        out[f] = m.reset_index(drop=True)
        clean[f] = D.rmse_clean(m["y_true"], m["p10"], m["MVT_ID_mvt"], f)
    off = {f: round(clean[f] - PUBLISHED[f], 4) for f in FOLDS}
    if max(abs(v) for v in off.values()) > GATE:
        raise SystemExit(f"the fold-side v10 does not reproduce exp-036 arm 2: {off}")
    board = Sc.P.board(clean)
    if abs(board - PUBLISHED_BOARD) > GATE:
        raise SystemExit(f"fold-side v10 board {board:.4f} against {PUBLISHED_BOARD:.4f}")
    got = {f: vector_sha(out[f]) for f in FOLDS}
    if not any(v.startswith("PIN_") for v in VECTOR_SHA.values()):
        bad = {f: got[f][:16] for f in FOLDS if got[f] != VECTOR_SHA[f]}
        if bad:
            raise SystemExit(f"the fold-side v10 vector has moved: {bad} against "
                             f"{{f: VECTOR_SHA[f][:16] for f in bad}}")
    else:
        print("[v10] VECTOR_SHA is unpinned; pin these:\n"
              + json.dumps({f: got[f] for f in FOLDS}, indent=4))
    return out


def vector_sha(df: pd.DataFrame) -> str:
    """SHA-256 over the fold vector's own ordered bytes: ids then `p10`, float64."""
    h = hashlib.sha256()
    for col in ("MVT_ID_mvt", "p10"):
        h.update(np.ascontiguousarray(df[col].to_numpy("float64")).tobytes())
    return h.hexdigest()


def board(rmse_by_fold: dict[str, float]) -> float:
    """exp-016's ranking-month weights: july_loo 55.71 %, january 44.29 %."""
    return Sc.P.board(rmse_by_fold)


def day(t) -> np.ndarray:
    """The UTC calendar date of a take-off clock, as `YYYY-MM-DD` strings."""
    return pd.to_datetime(np.asarray(t), unit="s", utc=True).strftime("%Y-%m-%d").to_numpy()


def main() -> None:
    d = load()
    print(f"v10 = refit-bulk at {', '.join(KEEP)}; v7 at {', '.join(HOLD)}")
    print(f"uploaded {UPLOADED.name} sha256 {UPLOADED_SHA[:16]}... verified\n")
    rows = []
    for f in FOLDS:
        x = d[f]
        c = x[x["clean"]]
        rows.append({"fold": f, "n": len(x), "n_clean": int(c.shape[0]),
                     "rmse_clean": round(float(np.sqrt((c["r"] ** 2).mean())), 4),
                     "published": PUBLISHED[f],
                     "mean_r": round(float(c["r"].mean()), 4),
                     "sse_clean_Ms2": round(float((c["r"] ** 2).sum()) / 1e6, 2)})
    T = pd.DataFrame(rows)
    print(T.to_string(index=False))
    print(f"\nclean board {board({f: rows[i]['rmse_clean'] for i, f in enumerate(FOLDS)}):.4f} "
          f"against exp-036 arm 2's {PUBLISHED_BOARD}")


if __name__ == "__main__":
    main()
