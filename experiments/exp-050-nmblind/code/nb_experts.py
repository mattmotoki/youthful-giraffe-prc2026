"""exp-050-nmblind: the matched NM-aware and NM-blind experts for one fold.

Both experts are ONE LightGBM recipe (`PARAMS`, `ROUNDS`, `SEED`) fitted on the SAME rows:
the fold's training months (`prc.data.FOLDS[fold].train_months`), regime `nm_bulk`
(the ordinary joined flight), label > 0 and `keep_reachable`.  They differ only by the
input block: BLIND = `nb_features.BLIND_FEATURES`, AWARE = BLIND + `NM_FEATURES`.  Each
predicts every `nm_bulk` departure of the fold's validation month; the router file
restricts to the registered population.

Categorical codes come from `work/vocab.json`, a sorted vocabulary per categorical
column over the twelve 2025 months' departure rows (inputs only, no label).  The blind
columns' vocabulary is built from blind columns only.

Also here: `prove_blind()`, RESULTS §1.11's mutation audit.  It runs right after the blind
fit and BEFORE the aware fit and before any label of the validation month is read; a
failure is fatal (q08 step 3).  Every fit launch waits while the 1-minute load average is
above 10 (`hold_load`, LAPTOP.md), and the outputs are write-once (exp-046
`nooverwrite.refuse_existing`, `--overwrite` to replace).

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-050-nmblind/code/nb_experts.py july_loo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp-046-surfacefit/code"))
import nb_features as NF                                              # noqa: E402
import nooverwrite as NO                                              # noqa: E402
import prc.data as D                                                  # noqa: E402

THREADS = 2
SEED = 50
ROUNDS = 1000
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=127,
              min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, max_bin=255, cat_smooth=10.0,
              min_data_per_group=200, max_cat_threshold=64, deterministic=True,
              force_row_wise=True, verbose=-1)
VOCAB = EXP / "work/vocab.json"
WORK = EXP / "work"
LOAD_LIMIT = 10.0


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hold_load(what: str) -> dict:
    """LAPTOP.md: hold every fit launch while the 1-minute load average is above 10."""
    while True:
        l1 = os.getloadavg()[0]
        if l1 <= LOAD_LIMIT:
            break
        print(f"  {utc()} load {l1:.2f} > {LOAD_LIMIT:g}: holding the {what} fit", flush=True)
        time.sleep(60)
    print(f"  {utc()} load {l1:.2f} <= {LOAD_LIMIT:g}: launching the {what} fit", flush=True)
    return {"launch_utc": utc(), "launch_load_1m": round(l1, 2)}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype="float64")).tobytes()
                          ).hexdigest()


def load_month(month: str) -> pd.DataFrame:
    rec = json.loads((EXP / "analysis/features.json").read_text())["months"][month]
    p = NF.feat_path(month)
    if sha_file(p) != rec["parquet_sha256"]:
        raise SystemExit(f"{p.name} moved since analysis/features.json was written")
    return pd.read_parquet(p)


def build_vocab() -> dict:
    """Sorted vocabulary per categorical column over the twelve months (inputs only)."""
    if VOCAB.exists():
        return json.loads(VOCAB.read_text())
    vals: dict[str, set] = {c: set() for c in NF.CATS}
    for m in D.MONTHS:
        F = pd.read_parquet(NF.feat_path(m), columns=NF.CATS)
        for c in NF.CATS:
            vals[c] |= set(F[c].astype(str).unique())
    voc = {c: sorted(v) for c, v in vals.items()}
    NO.refuse_existing([VOCAB], False)                 # the vocabulary is never replaced
    with VOCAB.open("x") as fh:                        # and a concurrent creator fails loud
        fh.write(json.dumps(voc, sort_keys=True))
    return voc


def encode(F: pd.DataFrame, cols: list[str], voc: dict) -> pd.DataFrame:
    X = pd.DataFrame(index=F.index)
    for c in cols:
        if c in NF.CATS:
            X[c] = pd.Categorical(F[c].astype(str), categories=voc[c])
        else:
            X[c] = F[c].to_numpy("float64")
    return X


def fit_expert(Xtr: pd.DataFrame, y: np.ndarray, cats: list[str]):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=ROUNDS, n_jobs=THREADS, random_state=SEED, **PARAMS)
    m.fit(Xtr, y, categorical_feature=cats)
    return m


def train_rows(F: pd.DataFrame) -> np.ndarray:
    return ((F.regime.to_numpy() == "nm_bulk") & (F.y.to_numpy("float64") > 0)
            & F.keep_reachable.to_numpy(bool))


def mutate_nm(frame: pd.DataFrame, rng: np.random.Generator, mode: str) -> pd.DataFrame:
    """Every NM input changed on EVERY row (departures and arrivals, i.e. all context rows).

    `shuffle`: datetimes shifted by a random +-6 h and 10 % blanked; other types permuted
    across rows and 10 % blanked.  `null`: every NM input blanked outright.
    """
    out = frame.copy()
    n = len(out)
    for c in NF.NM_INPUTS:
        s = out[c]
        if mode == "null":
            out[c] = s.where(np.zeros(n, dtype=bool))     # every value blanked, dtype kept
            continue
        blank = rng.random(n) < 0.10
        if str(s.dtype).startswith("datetime"):
            shift = pd.to_timedelta(rng.integers(-21600, 21601, size=n), unit="s")
            v = s + shift
            v[blank] = pd.NaT
        else:
            v = pd.Series(s.to_numpy()[rng.permutation(n)], index=out.index).astype(s.dtype)
            v[blank] = None
        out[c] = v
    return out


def _same(a: pd.Series, b: pd.Series) -> bool:
    if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
        x, y = a.to_numpy("float64"), b.to_numpy("float64")
        return bool(np.array_equal(x.view("uint64"), y.view("uint64")))
    return bool(np.array_equal(a.astype(str).to_numpy(), b.astype(str).to_numpy()))


def prove_blind(fold: str, blind_model, voc: dict, eligible_ids: np.ndarray) -> dict:
    """RESULTS §2: freeze the eligible ids, mutate every NM input on all context rows,
    rebuild, and require byte-identical blind features and blind predictions."""
    month = D.FOLDS[fold].valid_month
    masked, _ = NF.month_frame(month)
    B0 = NF.blind_features(masked)
    N0 = NF.nm_features(masked)
    keep0 = np.isin(B0.MVT_ID_mvt.to_numpy("float64"), eligible_ids)
    X0 = encode(B0[keep0], NF.BLIND_FEATURES, voc)
    p0 = blind_model.predict(X0)
    rec = {"month": month, "eligible_rows": int(keep0.sum()),
           "eligible_ids_sha256": sha_arr(np.sort(eligible_ids)),
           "baseline_blind_pred_sha256": sha_arr(p0), "mutations": {}}
    ok_all = True
    for i, mode in enumerate(("shuffle", "null")):
        mut = mutate_nm(masked, np.random.default_rng(SEED + i), mode)
        changed_inputs = {c: int((mut[c].astype(str).to_numpy()
                                  != masked[c].astype(str).to_numpy()).sum())
                          for c in NF.NM_INPUTS}
        B1 = NF.blind_features(mut)
        N1 = NF.nm_features(mut)
        same_order = bool(np.array_equal(B0.MVT_ID_mvt.to_numpy(), B1.MVT_ID_mvt.to_numpy()))
        feat_same = {c: _same(B0[c], B1[c]) for c in ["MVT_ID_mvt"] + NF.BLIND_FEATURES}
        keep1 = np.isin(B1.MVT_ID_mvt.to_numpy("float64"), eligible_ids)
        X1 = encode(B1[keep1], NF.BLIND_FEATURES, voc)
        p1 = blind_model.predict(X1)
        pred_same = bool(np.array_equal(np.asarray(p0).view("uint64"),
                                        np.asarray(p1).view("uint64")))
        nm_rows_changed = int((~np.all(np.isclose(
            N0[NF.NM_NUM].to_numpy("float64"), N1[NF.NM_NUM].to_numpy("float64"),
            equal_nan=True, rtol=0, atol=0), axis=1))[keep0].sum())
        ok = same_order and all(feat_same.values()) and pred_same and nm_rows_changed > 0
        ok_all &= ok
        rec["mutations"][mode] = {
            "nm_input_values_changed_all_rows": changed_inputs,
            "row_order_identical": same_order,
            "blind_features_byte_identical": all(feat_same.values()),
            "blind_feature_columns_differing": [c for c, v in feat_same.items() if not v],
            "blind_pred_sha256": sha_arr(p1), "blind_predictions_byte_identical": pred_same,
            "eligible_rows_whose_nm_features_changed": nm_rows_changed,
            "passes": ok}
    rec["passes"] = bool(ok_all)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fold", choices=["july_loo", "december", "january"])
    ap.add_argument("--overwrite", action="store_true",
                    help="replace this fold's existing expert artifacts")
    args = ap.parse_args()
    fold = args.fold
    out_json = EXP / f"analysis/experts-{fold}.json"
    out_oof = EXP / f"oof/experts-{fold}.parquet"
    out_model = {n: WORK / f"model-{n}-{fold}.txt" for n in ("blind", "aware")}
    NO.refuse_existing([out_json, out_oof, *out_model.values()], args.overwrite)
    fd = D.FOLDS[fold]
    t0 = time.time()
    voc = build_vocab()
    tr = pd.concat([load_month(m) for m in fd.train_months], ignore_index=True)
    tr = tr[train_rows(tr)].reset_index(drop=True)
    va = load_month(fd.valid_month)
    va = va[va.regime.to_numpy() == "nm_bulk"].reset_index(drop=True)
    y = tr.y.to_numpy("float64")
    print(f"{fold}: {len(tr):,} training rows from {len(fd.train_months)} months, "
          f"{len(va):,} nm_bulk validation rows ({time.time() - t0:.0f}s)", flush=True)

    sys.path.insert(0, str(HERE))
    import nb_panel as NP                                             # noqa: E402
    pan = NP.load(fold)
    pop = (pan.regime.astype(str).to_numpy() == "nm_bulk") & ~pan.guard_held.to_numpy(bool)
    eligible = np.sort(pan.MVT_ID_mvt.to_numpy("float64")[pop])

    rec = {"fold": fold, "train_months": list(fd.train_months),
           "valid_month": fd.valid_month, "train_rows": int(len(tr)),
           "valid_nm_bulk_rows": int(len(va)), "params": PARAMS, "rounds": ROUNDS,
           "seed": SEED, "threads": THREADS, "vocab_sha256": sha_file(VOCAB),
           "experts": {}}
    cols_of = {"blind": NF.BLIND_FEATURES, "aware": NF.AWARE_FEATURES}
    models = {}

    def fit(name: str) -> None:
        cols = cols_of[name]
        gate = hold_load(name)
        t1 = time.time()
        m = fit_expert(encode(tr, cols, voc), y, [c for c in cols if c in NF.CATS])
        models[name] = m
        NO.refuse_existing([out_model[name]], args.overwrite)
        m.booster_.save_model(str(out_model[name]))
        rec["experts"][name] = {
            "features": cols, "n_features": len(cols), "seconds": round(time.time() - t1, 1),
            "model": str(out_model[name].relative_to(ROOT)),
            "model_sha256": sha_file(out_model[name]), **gate}
        print(f"  {name}: {len(cols)} features, {time.time() - t1:.0f}s", flush=True)

    # 1. the blind expert, and its proof before any score and before the aware fit
    fit("blind")
    rec["blindness_proof"] = prove_blind(fold, models["blind"], voc, eligible)
    if not rec["blindness_proof"]["passes"]:
        NO.refuse_existing([out_json], args.overwrite)
        out_json.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
        raise SystemExit(f"{fold}: THE BLINDNESS PROOF FAILS -- every score is blocked: "
                         f"{json.dumps(rec['blindness_proof']['mutations'], default=float)[:2000]}")
    print(f"  {utc()} blindness proof PASSES on {rec['blindness_proof']['eligible_rows']:,} "
          f"eligible rows under both mutations", flush=True)
    # 2. the aware expert
    fit("aware")
    # 3. only now the validation month's predictions and labels
    yv = va.y.to_numpy("float64")
    preds = {"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy("float64")}
    for name, cols in cols_of.items():
        p = models[name].predict(encode(va, cols, voc))
        preds[f"p_{name}"] = p
        rec["experts"][name]["valid_rmse_nm_bulk"] = float(np.sqrt(np.mean((yv - p) ** 2)))
        rec["experts"][name]["pred_sha256"] = sha_arr(p)
        print(f"  {utc()} {name}: nm_bulk RMSE {rec['experts'][name]['valid_rmse_nm_bulk']:.2f}",
              flush=True)
    out = pd.DataFrame(preds)
    out["fold"] = fold
    NO.refuse_existing([out_oof], args.overwrite)
    out.to_parquet(out_oof, index=False)
    rec["expert_oof"] = str(out_oof.relative_to(ROOT))
    rec["expert_oof_sha256"] = sha_file(out_oof)
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([out_json], args.overwrite)
    out_json.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"written oof/experts-{fold}.parquet ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
