"""exp-018b runner: one prepared fold, many configurations, OOF for each.

  PYTHONPATH=. .venv/bin/python experiments/exp-018b-climb/code/run2.py \
      --folds july july_loo december january --configs code/cfg-modules.json --tag mod
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

_F = Path(__file__).resolve().parents[2] / "exp-018-fresh" / "code"
sys.path.insert(0, str(_F)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D            # noqa: E402
import model as M              # noqa: E402
import model2 as M2            # noqa: E402
from prc import data           # noqa: E402

EXP = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-oof", action="store_true")
    a = ap.parse_args()
    cfgs = json.loads((EXP / a.configs).read_text() if not Path(a.configs).exists()
                      else Path(a.configs).read_text())
    # Validate every arm against the signature BEFORE the first fold is prepared:
    # a typo in one arm used to surface 53 s in, after the prepare, and killed the
    # whole chained run. (2026-09-06: `ghost_mode` in cfg-s1/cfg-s3 did exactly that.)
    import inspect
    allowed = set(inspect.signature(M2.fit_predict).parameters) - {"train", "valid", "fold_name"}
    for c in cfgs:
        extra = sorted(set(c.get("kwargs", {})) - allowed)
        if extra:
            raise SystemExit(f"config {c['name']!r} passes {extra}, which "
                             f"model2.fit_predict does not accept; allowed: {sorted(allowed)}")
    rows = []
    for fname in a.folds:
        fold = data.FOLDS[fname]
        t0 = time.time()
        train = D.load(fold.train_months)
        valid = D.load([fold.valid_month])
        y = valid["y"].to_numpy(dtype="float64")
        ids = valid["MVT_ID_mvt"].to_numpy()
        train, (valid,) = D.add_encodings(train, [valid])
        base_feats = [c for c in train.columns if c not in M.DROP + M.HIGH_CARD]
        p_exact, exact_t = M2.fit_exact_head(train, valid, base_feats)
        print(f"[{fname}] prepared + exact head in {time.time()-t0:.0f}s", flush=True)
        for c in cfgs:
            t1 = time.time()
            pred, info = M2.fit_predict(train.copy(), valid.copy(), fname,
                                        p_exact=p_exact, exact_t=exact_t,
                                        **c.get("kwargs", {}))
            both = data.rmse_both(y, pred, ids, fname)
            sc = M.score(y, pred)
            rec = dict(fold=fname, name=c["name"], rmse=both["rmse"],
                       rmse_clean=both["rmse_clean"], lo=sc["rmse_lo"], hi=sc["rmse_hi"],
                       n_added=info["n_added"], n_dropped=info["n_dropped"],
                       seeds=info["seeds"], secs=round(time.time() - t1, 1))
            rows.append(rec)
            print(f"[{fname}] {c['name']:22s} rmse={rec['rmse']:8.2f} clean={rec['rmse_clean']:8.2f} "
                  f"lo={rec['lo']:7.2f} hi={rec['hi']:8.1f} +{rec['n_added']}cols "
                  f"-{rec['n_dropped']}rows {rec['secs']}s", flush=True)
            if a.save_oof:
                od = EXP / "oof"; od.mkdir(exist_ok=True)
                pd.DataFrame({"MVT_ID_mvt": ids, "fold": fname, "y_true": y,
                              "y_pred": pred}).to_parquet(od / f"{a.tag}-{c['name']}-{fname}.parquet",
                                                          index=False)
        del train, valid
        (EXP / "results" / f"{a.tag}.json").write_text(json.dumps(rows, indent=1))
    with open(EXP / "climb.log", "a") as f:
        for r in rows:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M')} {a.tag}/{r['name']}\t{r['fold']}="
                    f"{r['rmse']:.2f} clean={r['rmse_clean']:.2f}\tlo={r['lo']:.1f} hi={r['hi']:.0f} "
                    f"+{r['n_added']}cols -{r['n_dropped']}rows seeds={r['seeds']}\n")


if __name__ == "__main__":
    main()
