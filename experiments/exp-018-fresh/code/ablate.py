"""exp-018 fresh lineage: run several configurations against one prepared fold.

The target encodings and the category vocabulary are built once, so each extra
configuration costs only its own fits.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D  # noqa: E402
import model as M  # noqa: E402
from prc import data  # noqa: E402

EXP = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="july")
    ap.add_argument("--configs", required=True, help="path to a json list of {name, kwargs}")
    ap.add_argument("--log", default="ablate")
    a = ap.parse_args()
    cfgs = json.loads(Path(a.configs).read_text())
    fold = data.FOLDS[a.fold]
    t0 = time.time()
    train = D.load(fold.train_months)
    valid = D.load([fold.valid_month])
    y = valid["y"].to_numpy(dtype="float64")
    train, (valid,) = D.add_encodings(train, [valid])
    print(f"prepared in {time.time() - t0:.0f}s  train={train.shape} valid={valid.shape}", flush=True)
    rows = []
    for c in cfgs:
        t1 = time.time()
        kw = dict(c.get("kwargs", {}))
        pred, info = M.fit_predict(train, valid, prepared=True, y_valid=y, **kw)
        sc = M.score(y, pred)
        sc.update(name=c["name"], secs=round(time.time() - t1, 1))
        rows.append(sc)
        print(f"{c['name']:28s} rmse={sc['rmse']:8.2f} lo={sc['rmse_lo']:7.2f} hi={sc['rmse_hi']:9.1f} "
              f"{sc['secs']}s", flush=True)
    (EXP / "results").mkdir(exist_ok=True)
    (EXP / "results" / f"{a.log}-{a.fold}.json").write_text(json.dumps(rows, indent=1))
    with open(EXP / "climb.log", "a") as f:
        for r in rows:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M')} ablate/{a.fold}\t{r['name']}\t"
                    f"{a.fold}={r['rmse']:.2f}\tlo={r['rmse_lo']:.1f} hi={r['rmse_hi']:.0f}\n")


if __name__ == "__main__":
    main()
