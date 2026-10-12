"""Sol round 1, blocking 5: the four RMSE columns `COMMON.md` requires, per fold.

    PRC_THREADS=4 .venv/bin/python experiments/exp-040-chain/code/common_slices.py

`assignments/COMMON.md` line 98: "Every RESULTS table must show RMSE overall,
RMSE on rows with target at or below 3600 s, RMSE on rows above, and RMSE on
no-NM rows, for each fold."  Tables two, three and four gave overall only.  This
file adds the other three for every vector those tables name -- the three member
arms, and the head vectors on both serving variants -- plus the clean endpoint
(`prc.data.rmse_clean`) the held gate omitted.

The regime label comes from `packet/q09-{fold}.parquet`, which is the same row
set in the same order as the member OOF files; that is checked, not assumed.

Writes `analysis/common-slices.csv`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
import review1 as R1  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
ARMS = ("none", "chain", "derived")
TAIL = 3600.0
HELD = ("nm_bulk", "nm_late")


def rmse(y, p, m=None) -> float:
    if m is not None:
        y, p = y[m], p[m]
    return float(np.sqrt(((y - p) ** 2).mean())) if len(y) else float("nan")


def row(kind, name, fold, ids, y, p, nn) -> dict:
    """Sol round 2, blocking 2: `nn` is the STRICT no-NM mask.

    The first version used `regime == "nonm"`, which is not what `COMMON.md`
    line 88 defines -- no-NM is "all `_flt` columns null", and the two differ
    (3,938 / 1,649 / 1,435 rows strict against 3,595 / 1,554 / 1,369 by regime).
    `review1.no_nm_mask` is exp-028's corrected mask and is the one the rest of
    the record uses (`exp-030/code/score.py:129`).
    """
    lo, hi = y <= TAIL, y > TAIL
    return {"kind": kind, "vector": name, "fold": fold, "n": int(len(y)),
            "rmse": rmse(y, p), "rmse_le3600": rmse(y, p, lo),
            "rmse_gt3600": rmse(y, p, hi), "rmse_no_nm": rmse(y, p, nn),
            "n_le3600": int(lo.sum()), "n_gt3600": int(hi.sum()), "n_no_nm": int(nn.sum()),
            "rmse_clean": D.rmse_clean(pd.Series(y), pd.Series(p), pd.Series(ids), fold)}


def main() -> None:
    out = []
    for f in FOLDS:
        pk = pd.read_parquet(EXP / "packet" / f"q09-{f}.parquet")
        ids = pk["MVT_ID_mvt"].to_numpy("float64")
        y = pk["y_true"].to_numpy("float64")
        reg = pk["regime"].astype(str).to_numpy()
        nn = R1.no_nm_mask(f, ids)
        loose = reg == "nonm"
        print(f"[{f}] no-NM strict {int(nn.sum())} rows, by regime {int(loose.sum())}, "
              f"both {int((nn & loose).sum())}", flush=True)

        # --- the member arms (tables two and three) --------------------------
        for arm in ARMS:
            d = pd.read_parquet(EXP / "oof" / f"{arm}--{f}.parquet")
            d = d.set_index("MVT_ID_mvt").reindex(ids)
            if d["y_pred"].isna().any():
                raise SystemExit(f"{arm}--{f}: the OOF is not over the packet's rows")
            if not np.allclose(d["y_true"].to_numpy("float64"), y):
                raise SystemExit(f"{arm}--{f}: y_true disagrees with the packet")
            out.append(row("member", arm, f, ids, y,
                           d["y_pred"].to_numpy("float64"), nn))

        # --- the head vectors (table four and the held gate) -----------------
        v10 = pk["v10 replay"].to_numpy("float64")
        heads = {"v10 replay": v10}
        for c in ("none d11-noac", "chain d11-noac", "derived d11-noac",
                  "chain", "derived"):
            if c in pk.columns:
                heads[c] = pk[c].to_numpy("float64")
        on = np.isin(reg, list(HELD))
        heads["chain held (nm_bulk+nm_late)"] = np.where(
            on, pk["chain d11-noac"].to_numpy("float64"), v10)
        for name, p in heads.items():
            out.append(row("head", name, f, ids, y, p, nn))

    T = pd.DataFrame(out)
    T.to_csv(EXP / "analysis" / "common-slices.csv", index=False)
    for kind in ("member", "head"):
        print(f"\n=== {kind} ===")
        print(T[T["kind"] == kind]
              .pivot_table(index="vector", columns="fold",
                           values=["rmse", "rmse_le3600", "rmse_gt3600", "rmse_no_nm"])
              .round(4).to_string())


if __name__ == "__main__":
    main()
