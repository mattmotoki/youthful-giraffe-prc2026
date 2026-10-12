"""exp-017 label hygiene, the design in SPEC.md, per head.

Arm 0 and every hygiene arm must produce IDENTICAL band cells: `keep` reaches
the learned heads only (prc.tail_model 2.4.0). That check runs and prints
before any fold RMSE, per the desk. If it fails the run aborts, because the
first screen's whole table was void for exactly this reason.
"""
import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "exp-009-noac2"))
from threads import THREADS, gate
import numpy as np, pandas as pd
import prc.data as D, prc.tail_model as TM
from prc.tail_model import TailModel

EXP = D.ROOT / "experiments/exp-017-labels"
FOLDS = ("july", "january", "july_loo")
SEEDS = (7, 23, 51)
ARMS = [
    ("0 control",     dict()),
    ("A rollovers",   dict(rollover="correct")),
    ("A1 unreached",  dict(unreached=True)),
    ("C non-physical", dict(rollover="keep", nonpositive="drop", duplicates="drop")),
    ("A+C",           dict(rollover="correct", nonpositive="drop", duplicates="drop")),
    ("W recency",     dict(recency=180.0)),
    ("A+C+W",         dict(rollover="correct", nonpositive="drop",
                          duplicates="drop", recency=180.0)),
]
UNREACHED_IDS = {r["id"] for rows in D.UNREACHED_ROLLOVER.values() for r in rows}


def build(tr, spec):
    """(keep, corrected_y_or_None, weights_or_None) for one arm on one training frame."""
    dep = tr[tr["PHASE_mvt"] == "DEP"]
    w = (D.recency_weights(tr, spec["recency"]) if spec.get("recency") else None)
    if spec.get("unreached"):
        return ~dep["MVT_ID_mvt"].isin(UNREACHED_IDS), None, w
    pol = {k: spec[k] for k in ("rollover", "nonpositive", "duplicates") if k in spec}
    if not pol:
        return pd.Series(True, index=dep.index), None, w
    # the noac rollovers are the noac heads' own structure, and 11 of them are
    # the entire training signal of the band mixture: never drop them (SPEC).
    art = D.label_artefacts(tr)
    protect = dep["AIRCRAFT_TYPE_mvt"].isna() & art["rollover_undetectable"]
    pol.setdefault("rollover", "keep")
    pol.setdefault("nonpositive", "keep")
    pol.setdefault("duplicates", "keep")
    keep, y = D.clean_training_labels(tr, protect=protect, **pol)
    orig = dep[D.TARGET].astype(float)
    changed = y[(y != orig) & y.notna() & orig.notna()]
    return keep, (changed if len(changed) else None), w


def band_of(sig, y, mask=None):
    lab = y.notna() if mask is None else (y.notna() & mask.reindex(y.index).fillna(True))
    na = TM.regime_of(sig[lab]).to_numpy() == "noac"
    if not na.sum():
        return {}
    return TailModel._fit_band(sig[lab][na], y[lab][na].to_numpy(float))


def norm(b):
    return json.dumps({"offset": round(b.get("offset", float("nan")), 6),
                       "bands": {str(k): round(v, 9) for k, v in sorted(b.get("bands", {}).items())},
                       "cells": {str(k): round(v, 9) for k, v in sorted(b.get("cells", {}).items())}},
                      sort_keys=True)


# ---------------------------------------------------------- the band check
print("=" * 78)
print("BAND-CELL IDENTITY CHECK (desk): arm 0 vs every hygiene arm, before any RMSE")
print("=" * 78, flush=True)
REF, frames = {}, {}
for fold in FOLDS:
    tr, va, yv = D.fold_frames(fold)
    frames[fold] = (tr, va, yv)
    sig = TM.signals_by_month(tr)
    y = tr[D.TARGET].reindex(sig.index).astype(float)
    REF[fold] = band_of(sig, y)
    print(f"\n{fold}: band on all labelled rows (what fit uses)")
    print(f"   {norm(REF[fold])}")
    for name, spec in ARMS[1:]:
        keep, _, _ = build(tr, spec)
        b = band_of(sig, y, keep)
        same = norm(b) == norm(REF[fold])
        print(f"   {name:16s} keep drops {int((~keep).sum()):4d}  "
              f"band if keep DID reach it: {'identical' if same else 'DIFFERENT -> ' + norm(b)}")
    print(f"   -> shipped fit ignores keep for the band, so all arms use the line above",
          flush=True)

# ----------------------------------------------------------------- the arms
rows = []
for fold in FOLDS:
    tr, va, yv = frames[fold]
    ids = va["MVT_ID_mvt"].to_numpy()
    for name, spec in ARMS:
        keep, ycorr, w = build(tr, spec)
        for seed in SEEDS:
            gate()
            t0 = time.time()
            saved = None
            if ycorr is not None:                      # apply, fit, restore
                saved = tr.loc[ycorr.index, D.TARGET].copy()
                tr.loc[ycorr.index, D.TARGET] = ycorr.values
            try:
                m = TailModel().fit(tr, n_jobs=THREADS, seed=seed, keep=keep, weights=w)
            finally:
                if saved is not None:
                    tr.loc[saved.index, D.TARGET] = saved.values
            assert norm(m.band) == norm(REF[fold]), f"BAND MOVED: {fold}/{name}/{seed}"
            p = m.predict(va)
            y = yv.reindex(p.index).astype(float)
            k = y.notna()
            rows.append(dict(fold=fold, arm=name, seed=seed,
                             dropped=int((~keep).sum()),
                             corrected=0 if ycorr is None else len(ycorr),
                             rmse=D.rmse(y[k], p.y_pred[k]),
                             rmse_clean=D.rmse_clean(y[k], p.y_pred[k],
                                                     p.MVT_ID_mvt[k], fold),
                             secs=round(time.time() - t0)))
            print(f"{fold:9s} {name:16s} seed={seed:3d} rmse={rows[-1]['rmse']:7.2f} "
                  f"clean={rows[-1]['rmse_clean']:7.2f} drop={rows[-1]['dropped']:4d} "
                  f"{rows[-1]['secs']}s", flush=True)
            pd.DataFrame(rows).to_csv(EXP / "work" / "arms.csv", index=False)

df = pd.DataFrame(rows)
print("\n=== mean over seeds (spread in brackets) ===")
for metric in ("rmse", "rmse_clean"):
    g = df.groupby(["fold", "arm"])[metric].agg(["mean", "std"]).round(2)
    print(f"\n-- {metric} --")
    print(g.unstack(0).to_string())
