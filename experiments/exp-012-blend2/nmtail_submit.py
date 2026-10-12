"""Submission path for exp-010's NM-tail head, which their code did not have.

exp-010 shipped out-of-fold predictions at a pinned configuration but no ranking
predictions (`code/` has no submission mode; only `build_reocc.py` touches the
ranking file, and it caches the re-occupation signals rather than a prediction).
This fits the same head on all twelve 2025 months and predicts the ranking file.

Held to the same rules as the other members:

* **The configuration is `prc.nmtail` VERSION 2's defaults, untouched** --
  mode `phi`, 900 estimators, lr 0.04, 63 leaves, `num_threads=8`. Nothing is
  retuned while serving, because the weight this member earns was fitted on an
  out-of-fold prediction drawn at exactly these settings. `n_estimators` is a
  declared constant with no early stopping, so unlike the tree members there is
  no round-count draw to freeze.
* **Three seeds**, averaged raw. The head is a plain-tree LightGBM with no
  linear leaves, so there is no thread-count reproducibility hazard.
* **The two ranking months are already isolated** in both inputs: exp-003's
  feature frame is built per month, and `reocc_ranking.parquet` is built by
  `build_reocc.py` with a `groupby(month)`, so no signal crosses the six-month
  gap between January and July 2026.
* **`y_pred` is null on non-NM rows**, exactly as the out-of-fold files are.
  The head predicts NM-joined departures only; inventing a value for the rest
  would make the served member a different object from the measured one. The
  blend gives this member weight only inside `nm_late`, every row of which has an
  NM join by definition, so the nulls are never reached.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "experiments/exp-010-nmtail/code"),
          str(ROOT / "experiments/exp-003-anatomy/code")):
    if p not in sys.path:
        sys.path.insert(0, p)

import prc.data as D  # noqa: E402
import prc.nmtail as nt  # noqa: E402
from features import CAT, BASE_NUM, OCC, NBR  # noqa: E402

FEAT = ROOT / "artifacts/exp-003-anatomy/feat"
REOCC = ROOT / "artifacts/exp-010-nmtail"
EXTRA = ["gap", "log_gap", "screen_1h", "screen_30m", "reocc_x_screen",
         "reocc_x_screen1h", "mvt_lobt_ratio"]
FEATURES = [c for c in CAT + BASE_NUM + OCC + NBR] + list(nt.SIGNALS) + EXTRA
#: Overridable for the bit-stability separation, which needs the same code path
#: at a fraction of the cost: determinism is a property of the path, so one seed
#: at a reduced budget tests it as well as three at 900 and finishes in minutes.
SEEDS = [int(x) for x in os.environ.get("PRC_SEP_SEEDS", "7,107,207").split(",")]
N_ESTIMATORS = int(os.environ.get("PRC_SEP_ROUNDS", "900"))
#: `prc.nmtail.NMTailModel` hardcodes `num_threads = 8` and never reads the
#: environment, so before this line the release config could DECLARE 6 threads
#: for this member while 8 was what actually ran. A declared count that is not
#: the real count is worse than no declaration: it makes the manifest lie in the
#: one field the reproducibility contract rests on. The caller sets it
#: explicitly rather than changing the module default, which other lanes read.
N_THREADS = int(os.environ.get("PRC_THREADS", "8"))


def derive(f: pd.DataFrame) -> pd.DataFrame:
    """exp-010 `run_folds.frame`'s derived columns, verbatim."""
    f = f.copy()
    f["gap"] = -f["aobt_lobt"]
    g = f["gap"].to_numpy(float)
    rd = f["reocc_lobt_aobt_diff"].fillna(0).to_numpy(float)
    f["log_gap"] = np.sign(g) * np.log1p(np.abs(g))
    f["screen_1h"] = (g < -3600).astype("float32")
    f["screen_30m"] = (g < -1800).astype("float32")
    f["reocc_x_screen"] = rd * (g < -1800)
    f["reocc_x_screen1h"] = rd * (g < -3600)
    f["mvt_lobt_ratio"] = f["mvt_lobt"] / np.maximum(f["nm_taxi"], 60.0)
    return f


def month_frame(m: str) -> pd.DataFrame:
    f = pd.read_parquet(FEAT / f"{m}.parquet")
    r = pd.read_parquet(REOCC / f"reocc_{m}.parquet")
    return derive(f.merge(r, on="MVT_ID_mvt", how="left", validate="one_to_one"))


def ranking_frame() -> pd.DataFrame:
    f = pd.read_parquet(FEAT / "ranking.parquet")
    r = pd.read_parquet(REOCC / "reocc_ranking.parquet")
    return derive(f.merge(r, on="MVT_ID_mvt", how="left", validate="one_to_one"))


def main() -> None:
    t0 = time.time()
    tr = pd.concat([month_frame(m) for m in D.MONTHS], ignore_index=True)
    tr_nm = tr[tr.has_nm & tr.TAXITIME_SEC_mvt.notna()]
    va = ranking_frame()
    missing = [c for c in FEATURES if c not in va.columns]
    if missing:
        raise SystemExit(f"ranking frame is missing {missing}")
    has_nm = va["has_nm"].to_numpy().astype(bool)
    print(f"[submit] train {tr.shape} -> NM {tr_nm.shape}; ranking {va.shape}, "
          f"{int(has_nm.sum()):,} NM rows; built in {time.time()-t0:.0f}s", flush=True)

    per_seed, cfg = [], None
    for s in SEEDS:
        model = nt.NMTailModel(mode="phi", features=FEATURES, categoricals=list(CAT), seed=s,
                               n_estimators=N_ESTIMATORS, num_threads=N_THREADS)
        model.fit(tr_nm, tr_nm["TAXITIME_SEC_mvt"].to_numpy(float))
        per_seed.append(model.predict(va))
        if s == SEEDS[0]:
            # Report the count the LIBRARY holds, not the one we asked for. The
            # release manifest records this and the build refuses if it disagrees
            # with the declared value (desk rule, 2026-09-06).
            obs = model.model_.params.get("num_threads", "unknown")
            print(f"[threads] declared={os.environ.get('PRC_THREADS','unset')} "
                  f"observed={obs} library=lightgbm", flush=True)
        cfg = {"mode": model.mode, "n_estimators": model.n_estimators,
               "learning_rate": model.learning_rate, "num_leaves": model.num_leaves,
               "min_data_in_leaf": model.min_data_in_leaf,
               "num_threads": model.num_threads, "n_features": len(FEATURES),
               "nmtail_VERSION": nt.VERSION}
        print(f"[submit] seed {s} done at {time.time()-t0:.0f}s", flush=True)

    p = np.mean(per_seed, axis=0)
    p = np.where(has_nm, p, np.nan)          # the OOF convention, exactly
    cols = {"MVT_ID_mvt": va["MVT_ID_mvt"].to_numpy(), "y_pred": p,
            "has_nm": has_nm.astype("int8")}
    for i, q in enumerate(per_seed):
        cols[f"y_pred_s{i}"] = np.where(has_nm, q, np.nan)
    out = ROOT / "artifacts" / "exp-012" / "pred" / (os.environ.get("PRC_SEP_NAME", "N1-nmtail") + "-ranking.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_parquet(out, index=False)
    rec = {"name": "N1-nmtail", "seeds": SEEDS, "n": int(len(p)),
           "n_nm": int(has_nm.sum()), "config": cfg,
           "seed_spread_nm": float(np.std([q[has_nm].mean() for q in per_seed])),
           "path": str(out.relative_to(ROOT)), "sha256": D.sha256(out),
           "seconds": round(time.time() - t0)}
    (HERE / "submit-N1-nmtail.json").write_text(json.dumps(rec, indent=2, default=float))
    print("[submit] " + json.dumps(rec, default=float), flush=True)


if __name__ == "__main__":
    main()
