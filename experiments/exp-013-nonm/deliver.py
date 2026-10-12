"""Publish the no-NM head as a versioned blend member.

Writes OOF for the four folds and predictions for both ranking months, each
seed-averaged over three seeds, and prints the SHA-256 of every artefact.
Ranking months are transformed one at a time: the ranking file holds January
and July 2026 as isolated months, so a neighbour feature must not reach across
the gap between them.
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import prc.data as D
from prc.nonm import NoNMHead, segment_mask, VERSION
from segment import rmse
from report import rollover_ids

OUT = HERE / "oof"
FOLDS = ["july_loo", "december", "january", "july"]


def sha(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def run_fold(fold: str, summary: dict):
    t0 = time.time()
    f = D.FOLDS[fold]
    head = NoNMHead().fit([D.load_month(m) for m in f.train_months])
    valid = D.load_month(f.valid_month)
    pred = head.predict(valid)
    dep = valid.loc[pred.index]
    seg = segment_mask(dep)
    y = dep[D.TARGET].to_numpy(float)
    out = pd.DataFrame({D.ID: dep[D.ID].to_numpy(), "fold": fold, "y_true": y,
                        "y_pred": pred.to_numpy(), "seg": seg})
    p = OUT / f"nonm-v{VERSION}-{fold}.parquet"
    out.to_parquet(p, index=False)
    bad = rollover_ids(fold)
    keep = seg & ~out[D.ID].isin(bad).to_numpy()
    summary[fold] = {"n_segment": int(seg.sum()),
                     "segment": round(rmse(y[seg], pred.to_numpy()[seg]), 1),
                     "segment_trim": round(rmse(y[keep], pred.to_numpy()[keep]), 1),
                     "all_rows": round(rmse(y, pred.to_numpy()), 1),
                     "sha256": sha(p), "secs": round(time.time() - t0)}
    print(f"[{fold}] segment={summary[fold]['segment_trim']} (trim) "
          f"all={summary[fold]['all_rows']} sha={summary[fold]['sha256'][:8]}", flush=True)


def run_ranking(summary: dict):
    t0 = time.time()
    head = NoNMHead().fit([D.load_month(m) for m in D.MONTHS])
    rank = D.load_ranking()
    parts = []
    for month, g in rank.groupby("month", sort=True):
        pred = head.predict(g)
        dep = g.loc[pred.index]
        parts.append(pd.DataFrame({D.ID: dep[D.ID].to_numpy(), "month": month,
                                   "y_pred": pred.to_numpy(),
                                   "seg": segment_mask(dep)}))
    out = pd.concat(parts, ignore_index=True)
    p = OUT / f"nonm-v{VERSION}-ranking.parquet"
    out.to_parquet(p, index=False)
    summary["ranking"] = {"n": int(len(out)), "n_segment": int(out["seg"].sum()),
                          "sha256": sha(p), "secs": round(time.time() - t0)}
    print(f"[ranking] rows={len(out)} segment={int(out['seg'].sum())} "
          f"sha={summary['ranking']['sha256'][:8]}", flush=True)


if __name__ == "__main__":
    s = {}
    for fold in FOLDS:
        run_fold(fold, s)
        (HERE / "deliver.json").write_text(json.dumps(s, indent=2))
    run_ranking(s)
    (HERE / "deliver.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
