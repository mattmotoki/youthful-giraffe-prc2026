"""The exp-034 bulk member: exp-003's recommended 64-column set at 600 rounds.

One configuration, three modes, no tuning:

    gate    refit seed 7 on `july_loo` and compare against exp-011's COMMITTED
            `oof/seeds_july_loo.parquet` column `p_s7` -- the same 64 columns,
            the same 600 rounds, the same 127 leaves.  This is the frame gate:
            if the rebuilt frame were not exp-003's frame the number would move
            by seconds, not by hundredths.
    folds   the member's out-of-fold prediction on the three ROTATION folds
            (`july_loo`, `december`, `january`), in the exp-027 harness's
            contract: `MVT_ID_mvt, fold, y_true, y_pred`, one file per fold,
            every departure of the validation month.
    rank    the twelve-month fit, applied to the 344,841 ranking departures.

The configuration is imported from `experiments/exp-011-bulkfloor/code/q1c_seeds.py`
rather than restated, so there is exactly one address for "what the queue's
227.8 s describes".  `FEATURES` is `CAT(13) + BASE_NUM(27) + OCC(18) + NBR(6)`,
which is exp-003 `RESULTS.md`'s recommended set, column for column.

**The member is PARTIAL and that is not a defect.**  It predicts
`taxi = max(nm_taxi - offset, 0)` with `nm_taxi = MVT_TIME_UTC_mvt - AOBT_3_flt`,
so where there is no NM join there is no `nm_taxi` and no prediction.  That is
2.06 % of a fold's rows, the same population `N1-nmtail` is null on, and the
blend/stack machinery already carries partial members (`frames.PARTIAL`,
`stack5.member_features`'s nan-aware aggregates, LightGBM's native missing).
Inventing a second head for the no-NM rows would be a different member from the
one the queue priced.

    PRC_THREADS=6 python member.py gate
    PRC_THREADS=6 python member.py folds
    PRC_THREADS=6 python member.py rank
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp-003-anatomy/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-011-bulkfloor/code"))
import prc.data as D  # noqa: E402
import q1c_seeds as Q  # noqa: E402  the frozen configuration, imported not restated

FEATURES = Q.FEATURES
PARAMS = Q.PARAMS
ROUNDS = Q.ROUNDS
SEED = int(os.environ.get("PRC_SEED", "7"))
THREADS = int(os.environ.get("PRC_THREADS", "6"))
FOLDS = ("july_loo", "december", "january")
ART = Q.ART                       # artifacts/exp-003-anatomy/feat
NAME = "B1-bulk64"
#: The columns each mode needs out of the 99-column frame, so a twelve-month
#: concatenation stays under a gigabyte.
COLS = list(dict.fromkeys(FEATURES + ["MVT_ID_mvt", "BLOCK_TIME_UTC_mvt",
                                      "AOBT_3_flt", "TAXITIME_SEC_mvt", "has_nm"]))
#: exp-011 `oof/seeds_july_loo.parquet`, seed 7, recomputed from the committed
#: file rather than quoted: the gate prints both.
GATE_REF = ROOT / "experiments/exp-011-bulkfloor/oof/seeds_july_loo.parquet"
#: exp-003 `RESULTS.md`: "the run-to-run spread of this harness is about 0.05 s
#: and no difference below ~0.2 s is real".  The committed vector was fitted on
#: the workstation (Linux, another LightGBM build, another thread count) and
#: `PARAMS` sets neither `deterministic` nor `force_row_wise`, so a row-exact
#: gate would fail for reasons that have nothing to do with the frame.
GATE_TOL = 0.05


def frame(m: str) -> pd.DataFrame:
    return pd.read_parquet(ART / f"{m}.parquet", columns=COLS)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in Q.CAT:
        X[c] = X[c].astype("category")
    return X


def _label(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["off"] = (f.BLOCK_TIME_UTC_mvt - f.AOBT_3_flt).dt.total_seconds()
    f["y"] = f.TAXITIME_SEC_mvt.astype(float)
    return f


def fit_predict(train_months, valid_frame: pd.DataFrame, seed: int = SEED,
                rounds: int = ROUNDS, threads: int = THREADS):
    """exp-011 `q1c_seeds.run`'s fit, factored so `folds` and `rank` share it."""
    tr = _label(pd.concat([frame(m) for m in train_months], ignore_index=True))
    # exp-011's training population: an NM join, a bulk label, a defined offset,
    # not a noac row.
    mtr = tr.has_nm & tr.y.le(3600) & tr.off.notna() & tr.noac.ne(1)
    p = dict(PARAMS); p["seed"] = seed; p["num_threads"] = threads
    t0 = time.time()
    mdl = lgb.train(p, lgb.Dataset(prep(tr[mtr]), label=tr.loc[mtr, "off"]),
                    num_boost_round=rounds)
    secs = time.time() - t0
    n_tr = int(mtr.sum())
    del tr
    off_hat = mdl.predict(prep(valid_frame))
    pred = np.maximum(valid_frame["nm_taxi"].to_numpy("float64") - off_hat, 0.0)
    return pred, off_hat, {"n_train": n_tr, "fit_s": round(secs, 1),
                           "rounds": rounds, "seed": seed, "threads": threads}


def bulk_rmse(v: pd.DataFrame, pred: np.ndarray) -> tuple[float, int]:
    """exp-011's diagnostic bulk slice: NM join, label <= 3600 s, not noac."""
    m = (v.has_nm & v.y.le(3600) & v.noac.ne(1)).to_numpy()
    y = v["y"].to_numpy("float64")
    return float(np.sqrt(np.mean((y[m] - pred[m]) ** 2))), int(m.sum())


def gate() -> dict:
    """Refit seed 7 on july_loo and read it against exp-011's committed vector."""
    ref = pd.read_parquet(GATE_REF)
    F = D.FOLDS["july_loo"]
    va = _label(frame(F.valid_month))
    pred, _, info = fit_predict(F.train_months, va)
    got, n = bulk_rmse(va, pred)

    m = pd.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"].to_numpy("float64"),
                      "new": pred}).merge(
        ref[["MVT_ID_mvt", "y_true", "bulk", "p_s7"]].assign(
            MVT_ID_mvt=lambda d: d["MVT_ID_mvt"].astype("float64")),
        on="MVT_ID_mvt", how="inner")
    if len(m) != len(ref):
        raise SystemExit(f"the rebuilt frame covers {len(m)} of {ref.shape[0]} committed rows")
    b = m["bulk"].to_numpy()
    y = m["y_true"].to_numpy("float64")
    want = float(np.sqrt(np.mean((y[b] - m["p_s7"].to_numpy("float64")[b]) ** 2)))
    d = m["new"].to_numpy("float64")[b] - m["p_s7"].to_numpy("float64")[b]
    rec = {"gate": "exp-011 seeds_july_loo p_s7, 64 columns, 600 rounds, seed 7",
           "n_bulk": n, "n_rows": int(len(m)),
           "rmse_new": round(got, 4), "rmse_committed": round(want, 4),
           "delta": round(got - want, 4), "tolerance": GATE_TOL,
           "row_diff_mean": round(float(d.mean()), 4),
           "row_diff_sd": round(float(d.std()), 4),
           "row_diff_p99_abs": round(float(np.quantile(np.abs(d), 0.99)), 4),
           "row_corr": round(float(np.corrcoef(m["new"].to_numpy("float64")[b],
                                               m["p_s7"].to_numpy("float64")[b])[0, 1]), 6),
           "pass": bool(abs(got - want) <= GATE_TOL), **info}
    print(json.dumps(rec, indent=2), flush=True)
    (EXP / "analysis" / "gate-frame.json").write_text(json.dumps(rec, indent=2))
    if not rec["pass"]:
        raise SystemExit(f"frame gate FAILS: {got:.4f} against {want:.4f}")
    return rec


def folds() -> None:
    out = []
    for f in FOLDS:
        F = D.FOLDS[f]
        va = _label(frame(F.valid_month))
        pred, off_hat, info = fit_predict(F.train_months, va)
        got, n = bulk_rmse(va, pred)
        y = va["y"].to_numpy("float64")
        ids = va["MVT_ID_mvt"].to_numpy("float64")
        full = float(np.sqrt(np.mean((y[np.isfinite(pred)] - pred[np.isfinite(pred)]) ** 2)))
        cln = D.rmse_clean(y, np.where(np.isfinite(pred), pred, 0.0), ids, f)
        rec = dict(fold=f, n=len(va), n_bulk=n, rmse_bulk=round(got, 4),
                   rmse_defined=round(full, 4), n_undefined=int((~np.isfinite(pred)).sum()),
                   rmse_clean_zerofill=round(cln, 4), **info)
        print("[member] " + json.dumps(rec), flush=True)
        out.append(rec)
        pd.DataFrame({"MVT_ID_mvt": ids, "fold": f, "y_true": y,
                      "y_pred": pred, "off_pred": off_hat}).to_parquet(
            EXP / "oof" / f"bulk64-{f}.parquet", index=False)
        with open(EXP / "climb.log", "a") as fh:
            fh.write(json.dumps({"name": "member", **rec}) + "\n")
    (EXP / "analysis" / "member-folds.json").write_text(json.dumps(out, indent=2))


def rank() -> None:
    rk = pd.read_parquet(ART / "ranking.parquet", columns=COLS)
    rk["y"] = np.nan
    pred, off_hat, info = fit_predict(D.MONTHS, rk)
    n_null = int((~np.isfinite(pred)).sum())
    o = EXP / "candidates" / "bulk64-ranking.parquet"
    pd.DataFrame({"MVT_ID_mvt": rk["MVT_ID_mvt"].to_numpy("float64"),
                  "y_pred": pred}).to_parquet(o, index=False)
    rec = {"rows": int(len(rk)), "undefined": n_null, "sha256": D.sha256(o),
           "out": str(o.relative_to(ROOT)), **info,
           "pred_min": round(float(np.nanmin(pred)), 3),
           "pred_med": round(float(np.nanmedian(pred)), 3),
           "pred_max": round(float(np.nanmax(pred)), 3)}
    print(json.dumps(rec, indent=2), flush=True)
    (EXP / "analysis" / "member-ranking.json").write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    {"gate": gate, "folds": folds, "rank": rank}[sys.argv[1]]()
