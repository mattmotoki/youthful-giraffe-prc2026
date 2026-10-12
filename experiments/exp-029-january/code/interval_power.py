"""Can the day-clustered interval decide this gate at all? Measured, not assumed.

L4's first kill, as the queue now words it, is "the clustered interval includes
zero". The correction moves rows inside **2 of the january fold's 310
airport-day clusters**. A cluster bootstrap that resamples all 310 draws neither
of those two in `(1 - 2/310)^310` of its draws, and in every such draw the gain
is exactly zero, so the resampling distribution carries a point mass at zero
that is far above the 2.5 % quantile. The lower bound is then <= 0 for **any**
effect size, and the kill fires on arithmetic rather than on evidence.

This file measures that mass, and reports the only interval that could have
power instead: resample the touched clusters alone, holding the untouched part
of the fold fixed, which is exact because the correction cannot change it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import correct as C  # noqa: E402
import envelope as E  # noqa: E402
import prc.data as D  # noqa: E402

ARMS = ("step|lofo|winsor", "step-half|lofo|winsor", "stress|lofo|winsor")
DRAWS = 2000


def conditional_interval(y, p_new, p_ref, cl, touched, draws=DRAWS, seed=7):
    """Resample only the clusters the correction touches; the rest is fixed.

    The fold's squared error splits exactly into a part the correction cannot
    reach and a part it can, so an interval on the whole-fold RMSE that
    resamples only the reachable clusters is the paired interval with the
    untouched variance removed rather than resampled.
    """
    e_new = (p_new - y) ** 2
    e_ref = (p_ref - y) ** 2
    inside = np.isin(cl, touched)
    base_sse, base_n = float(e_ref[~inside].sum()), int((~inside).sum())
    sub = pd.factorize(cl[inside])[0]
    k = sub.max() + 1
    order = np.argsort(sub, kind="stable")
    starts = np.searchsorted(sub[order], np.arange(k))
    ends = np.searchsorted(sub[order], np.arange(k), side="right")
    sn = np.add.reduceat(e_new[inside][order], starts)
    sr = np.add.reduceat(e_ref[inside][order], starts)
    cnt = (ends - starts).astype("float64")
    rng = np.random.default_rng(seed)
    W = rng.multinomial(k, np.full(k, 1.0 / k), size=draws).astype("float64")
    n = base_n + W @ cnt
    out = (np.sqrt((base_sse + W @ sr) / n) - np.sqrt((base_sse + W @ sn) / n))
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)), k


def main() -> None:
    R = pd.read_parquet(EXP / "analysis" / "corrections.parquet")
    R["day"] = pd.to_datetime(R["day"])
    rec = []
    for arm in ARMS:
        for f in E.FOLDS:
            g = R[R["fold"] == f]
            y = g["y_true"].to_numpy("float64")
            ref = g["y_pred"].to_numpy("float64")
            c = g[f"c::{arm}"].to_numpy("float64")
            new = np.maximum(ref + c, 60.0)
            ids = g["MVT_ID_mvt"].to_numpy("float64")
            ex = D.excluded_ids(f)
            keep = ~np.isin(ids, list(ex)) if ex else np.ones(len(g), bool)
            cl = pd.factorize(np.asarray(C.clusters(g))[keep])[0]
            k = cl.max() + 1
            touched = np.unique(cl[c[keep] != 0])
            rng = np.random.default_rng(7)
            W = rng.multinomial(k, np.full(k, 1.0 / k), size=DRAWS)
            zero = float(1.0 - (W[:, touched].sum(axis=1) > 0).mean()) if len(touched) else 1.0
            gain, lo, hi = C.boot_gain(y, new, ref, C.clusters(g), ids, f)
            clo, chi, kt = conditional_interval(y[keep], new[keep], ref[keep], cl, touched)
            rec.append(dict(arm=arm, fold=f, clusters=k, touched=len(touched),
                            zero_effect_draws=zero, gain=gain,
                            lo_all=lo, hi_all=hi, lo_touched=clo, hi_touched=chi))
    T = pd.DataFrame(rec)
    T.to_csv(EXP / "analysis" / "interval_power.csv", index=False)
    print("=== the clustered interval, and what it can and cannot decide ===")
    print(T.round(4).to_string(index=False), flush=True)
    print("\nA fold whose correction touches `touched` of `clusters` day-clusters has a")
    print("point mass of `zero_effect_draws` exactly at zero in the all-cluster")
    print("bootstrap, so its 2.5 % quantile cannot be positive whenever that mass")
    print("exceeds 0.025. The `_touched` interval removes the untouched variance")
    print("instead of resampling it, and is reported for completeness: with 1-2")
    print("clusters to resample it has no power either.")
    (EXP / "analysis" / "interval_power.json").write_text(
        json.dumps({"draws": DRAWS, "rows": T.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
