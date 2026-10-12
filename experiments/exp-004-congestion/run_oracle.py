"""Ceiling instrument: base model vs base + oracle queue count.

The oracle feature is the TRUE number of same-runway (and same-airport)
take-offs between this departure's true BLOCK_TIME_UTC_mvt and its take-off.
It is illegal at prediction time (block time is blanked in the ranking file);
it bounds what a perfect queue feature could give a model that already has the
own-flight deltas.
"""
from __future__ import annotations

import sys, time, json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import prc.data as D
from common import base_features, epoch, metrics, fmt, fit_predict, BASE_CAT, TARGET, ID


def oracle_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """True take-offs between own block time and own take-off (oracle only)."""
    d = frame[frame["PHASE_mvt"] == "DEP"]
    t = epoch(d["MVT_TIME_UTC_mvt"])
    b = epoch(d["BLOCK_TIME_UTC_mvt"])
    apt = d["ADEP_mvt"].astype(str).to_numpy()
    rwy = d["RUNWAY_mvt"].astype(str).to_numpy()
    out = pd.DataFrame(index=d.index)
    for name, key in (("oracle_q_apt", apt), ("oracle_q_rwy", np.char.add(np.char.add(apt, ":"), rwy))):
        res = np.full(len(d), np.nan)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        bounds = np.searchsorted(ks, np.unique(ks))
        edges = list(bounds) + [len(ks)]
        for i in range(len(edges) - 1):
            idx = order[edges[i]:edges[i + 1]]
            tt = np.sort(t[idx])
            res[idx] = (np.searchsorted(tt, t[idx], "left")
                        - np.searchsorted(tt, b[idx], "right"))
        out[name] = res
    out["oracle_taxi_span"] = t - b
    return out


def run_fold(fold_name: str, with_span: bool = False) -> dict:
    t0 = time.time()
    fold = D.FOLDS[fold_name]
    # Features are built one month at a time on both sides: the ranking file
    # holds its two months in isolation, so a training row must not see
    # neighbours across a file boundary either.
    Xtr_parts, ytr_parts, otr_parts = [], [], []
    for name in fold.train_months:
        month = D.load_month(name)
        B = base_features(month)
        Xtr_parts.append(B)
        ytr_parts.append(month.loc[B.index, TARGET])
        otr_parts.append(oracle_counts(month).loc[B.index])
    Xtr = pd.concat(Xtr_parts, ignore_index=True)
    ytr = pd.concat(ytr_parts, ignore_index=True).to_numpy(float)
    otr = pd.concat(otr_parts, ignore_index=True)

    valid_full = D.load_month(fold.valid_month)
    valid_masked, _ = D.mask_like_ranking(valid_full)
    Xva = base_features(valid_masked)
    yva = valid_full.loc[Xva.index, TARGET].to_numpy(float)
    has_nm_va = Xva["has_nm"].to_numpy()
    noac_va = valid_full.loc[Xva.index, "AIRCRAFT_TYPE_mvt"].isna().to_numpy()
    print(f"[{fold_name}] train {Xtr.shape} valid {Xva.shape} load+feat {time.time()-t0:.0f}s", flush=True)

    res = {}
    _, p = fit_predict(Xtr, ytr, Xva, BASE_CAT)
    res["base"] = metrics(yva, p, has_nm_va, noac_va)
    print(f"[{fold_name}] base   {fmt(res['base'])}", flush=True)

    cols = ["oracle_q_rwy", "oracle_q_apt"] + (["oracle_taxi_span"] if with_span else [])
    ova = oracle_counts(valid_full)  # the oracle reads the hidden block time on purpose
    Xtr2 = pd.concat([Xtr, otr[cols]], axis=1)
    Xva2 = pd.concat([Xva, ova.loc[Xva.index, cols]], axis=1)
    _, p2 = fit_predict(Xtr2, ytr, Xva2, BASE_CAT)
    res["oracle"] = metrics(yva, p2, has_nm_va, noac_va)
    print(f"[{fold_name}] oracle {fmt(res['oracle'])}", flush=True)
    print(f"[{fold_name}] done in {time.time()-t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    folds = sys.argv[1:] or ["july", "december", "january"]
    all_res = {}
    for f in folds:
        all_res[f] = run_fold(f)
        (HERE / "oracle_bound.json").write_text(json.dumps(all_res, indent=2))
    print(json.dumps(all_res, indent=2))
