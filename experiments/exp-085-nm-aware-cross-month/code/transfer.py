"""Exp-085 registered source-only fit, correction freeze, and one-shot scoring."""
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
E50 = ROOT / "experiments/exp-050-nmblind"
E52 = ROOT / "experiments/exp-052-nmv22"
for path in (ROOT, E50 / "code", E52 / "code"):
    sys.path.insert(0, str(path))

import rx_common as C  # noqa: E402

NF, NE = C.NF, C.NE
FOLDS = ("july_loo", "december", "january")
MONTH = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
EXPECTED_P = {"july_loo": 172028, "december": 158212, "january": 147455}
TYPES = ("aware", "sham", "blind")
OUTPUT_TYPES = ("aware", "sham", "blind", "airport_constant", "null")
KGRID = (25, 50, 100, 200, 400)
SEED = 85
SEED_SHAM = 8501
SEED_NULL = 8502
BOOT_SEED = 852026
BOOT = 2000
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621,
          "december": 0.0}
PARAMS = dict(C.PARAMS)
THREADS = 3
LOAD_LIMIT = 10.0
MANIFEST = EXP / "source-manifest.json"
FIT_RECEIPTS = EXP / "fit-receipts.json"
FROZEN = EXP / "correction-manifest.json"
SCORE_SENTINEL = EXP / "analysis/target-label-opened.json"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_values(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def check_sources() -> dict:
    rec = json.loads(MANIFEST.read_text())
    for rel, want in rec["files"].items():
        p = ROOT / rel
        if p.stat().st_size != want["bytes"] or sha_file(p) != want["sha256"]:
            raise SystemExit(f"frozen source changed: {rel}")
    if rec["aware_columns"] != NF.AWARE_FEATURES or rec["blind_columns"] != NF.BLIND_FEATURES:
        raise SystemExit("registered feature lists differ from exp-052")
    return rec


def panel_path(fold: str) -> Path:
    return E50 / f"work/panel-v22-{fold}.parquet"


def feat_path(fold: str) -> Path:
    return E50 / f"work/feat-{MONTH[fold]}.parquet"


def load_label_free(fold: str) -> dict:
    """This projection cannot load y_true or the feature parquet's y column."""
    pan_cols = ["MVT_ID_mvt", "airport", "day", "regime", "guard_held", "clean",
                "nm_missing", "p_v22"]
    pan = pd.read_parquet(panel_path(fold), columns=pan_cols)
    pop = (pan.regime.astype(str).to_numpy() == "nm_bulk") & ~pan.guard_held.to_numpy(bool)
    if int(pop.sum()) != EXPECTED_P[fold]:
        raise SystemExit(f"{fold}: P has {pop.sum()} rows, expected {EXPECTED_P[fold]}")
    P = pan.loc[pop].reset_index(drop=True)
    cols = ["MVT_ID_mvt", "keep_reachable"] + list(NF.AWARE_FEATURES)
    feat = pd.read_parquet(feat_path(fold), columns=cols).set_index("MVT_ID_mvt")
    feat = feat.reindex(P.MVT_ID_mvt.to_numpy()).reset_index()
    if feat.MVT_ID_mvt.isna().any():
        raise SystemExit(f"{fold}: missing feature row in P")
    voc = json.loads((E50 / "work/vocab.json").read_text())
    X = {"aware": NE.encode(feat, NF.AWARE_FEATURES, voc),
         "blind": NE.encode(feat, NF.BLIND_FEATURES, voc)}
    perm = C.airday_perm((P.airport.astype(str) + "|" + P.day.astype(str)).to_numpy(),
                         np.random.default_rng(SEED_SHAM))
    X["sham"] = X["aware"].copy()
    for col in NF.NM_FEATURES:
        X["sham"][col] = pd.Series(X["aware"][col].array.take(perm), index=X["sham"].index)
    return {"pan": pan, "pop": pop, "P": P, "feat": feat, "X": X,
            "blocks": C.blocks(P), "sham_perm_sha256": sha_values(perm)}


def load_source_residual(fold: str, lf: dict) -> tuple[np.ndarray, np.ndarray]:
    """The only label loader used before freeze; callers name this fold as the source."""
    lab = pd.read_parquet(panel_path(fold), columns=["MVT_ID_mvt", "y_true"])
    pids = lf["P"].MVT_ID_mvt.to_numpy()
    y = lab.set_index("MVT_ID_mvt").reindex(pids).y_true.to_numpy("float64")
    r = y - lf["P"].p_v22.to_numpy("float64")
    fitmask = lf["feat"].keep_reachable.to_numpy(bool) & (y > 0) & np.isfinite(r)
    return r, fitmask


def hold_load(label: str) -> dict:
    while os.getloadavg()[0] > LOAD_LIMIT:
        print(f"load above {LOAD_LIMIT:g}; holding {label}", flush=True)
        time.sleep(30)
    return {"launch_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "load_1m": round(float(os.getloadavg()[0]), 2)}


def model(cols: list[str], rounds: int):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=rounds, n_jobs=THREADS, random_state=SEED, **PARAMS), [
        c for c in cols if c in NF.CATS]


def fit_one(X, y, rows, cols, rounds, label):
    launch = hold_load(label)
    m, cats = model(cols, rounds)
    start = time.time()
    m.fit(X.loc[rows, cols], y[rows], categorical_feature=cats)
    return m, {**launch, "fit_rows": int(rows.sum()), "rounds": rounds,
               "seconds": round(time.time() - start, 3)}


def correction_path(source: str, target: str) -> Path:
    return EXP / f"corrections/{source}-to-{target}.parquet"


def fit_source(source: str) -> dict:
    check_sources()
    if FROZEN.exists() or SCORE_SENTINEL.exists():
        raise SystemExit("corrections are frozen or scoring started; fits are forbidden")
    outputs = [correction_path(source, t) for t in FOLDS if t != source]
    if any(p.exists() for p in outputs):
        raise SystemExit(f"{source}: correction output already exists")
    data = {f: load_label_free(f) for f in FOLDS}  # projections exclude labels
    src = data[source]
    r, fitmask = load_source_residual(source, src)
    blocks = src["blocks"]
    receipts, selected, full_pred = [], {}, {}
    for typ in TYPES:
        cols = list(NF.BLIND_FEATURES if typ == "blind" else NF.AWARE_FEATURES)
        oof = {k: np.full(len(r), np.nan) for k in KGRID}
        for b in range(5):
            rows = fitmask & (blocks != b)
            m, receipt = fit_one(src["X"][typ], r, rows, cols, 400,
                                 f"{source} {typ} inner {b}")
            score = blocks == b
            for k in KGRID:
                oof[k][score] = m.predict(src["X"][typ].loc[score, cols], num_iteration=k)
            receipts.append({"source": source, "type": typ, "unit": f"inner-{b}", **receipt})
        mse = {k: float(np.mean((r[fitmask] - oof[k][fitmask]) ** 2)) for k in KGRID}
        kstar = min(KGRID, key=lambda k: (mse[k], k))
        selected[typ] = {"k": kstar, "source_oof_mse": mse}
        m, receipt = fit_one(src["X"][typ], r, fitmask, cols, 400, f"{source} {typ} full")
        receipts.append({"source": source, "type": typ, "unit": "full", **receipt})
        full_pred[typ] = {t: np.asarray(m.predict(data[t]["X"][typ][cols], num_iteration=kstar),
                                         dtype="float64") for t in FOLDS if t != source}
    key = (src["P"].airport.astype(str) + "|" + src["P"].day.astype(str)).to_numpy()
    rn = r.copy()
    idx = np.flatnonzero(fitmask)
    rn[idx] = r[idx][C.airday_perm(key[idx], np.random.default_rng(SEED_NULL))]
    null_model, receipt = fit_one(src["X"]["aware"], rn, fitmask, list(NF.AWARE_FEATURES),
                                  selected["aware"]["k"], f"{source} null full")
    receipts.append({"source": source, "type": "null", "unit": "full", **receipt})
    apt_mean = pd.Series(r[fitmask]).groupby(src["P"].airport.astype(str)[fitmask]).mean()
    global_mean = float(np.mean(r[fitmask]))
    for target in FOLDS:
        if target == source:
            continue
        tgt = data[target]
        vectors = {typ: full_pred[typ][target] for typ in TYPES}
        vectors["null"] = np.asarray(null_model.predict(tgt["X"]["aware"][NF.AWARE_FEATURES]),
                                     dtype="float64")
        vectors["airport_constant"] = tgt["P"].airport.astype(str).map(apt_mean).fillna(global_mean).to_numpy("float64")
        n = len(tgt["P"])
        out = pd.DataFrame({
            "MVT_ID_mvt": np.tile(tgt["P"].MVT_ID_mvt.to_numpy(), len(OUTPUT_TYPES)),
            "target_fold": np.repeat(target, n * len(OUTPUT_TYPES)),
            "source_fold": np.repeat(source, n * len(OUTPUT_TYPES)),
            "expert_type": np.repeat(OUTPUT_TYPES, n),
            "correction": np.concatenate([vectors[x] for x in OUTPUT_TYPES]),
        })
        correction_path(source, target).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(correction_path(source, target), index=False)
    return {"source": source, "selected": selected, "fit_receipts": receipts,
            "fit_count": len(receipts), "source_rows": len(r), "fit_rows": int(fitmask.sum()),
            "sham_perm_sha256": src["sham_perm_sha256"]}


def freeze() -> dict:
    check_sources()
    if FROZEN.exists() or SCORE_SENTINEL.exists():
        raise SystemExit("freeze already exists or scoring started")
    runs = []
    for f in FOLDS:
        p = EXP / f"analysis/fit-{f}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        runs.append(json.loads(p.read_text()))
    if sum(x["fit_count"] for x in runs) != 57:
        raise SystemExit("fit accounting is not exactly 57")
    files = {}
    for source in FOLDS:
        for target in FOLDS:
            if source == target:
                continue
            p = correction_path(source, target)
            x = pd.read_parquet(p)
            want = load_label_free(target)["P"].MVT_ID_mvt.to_numpy()
            for typ in OUTPUT_TYPES:
                got = x.loc[x.expert_type == typ, "MVT_ID_mvt"].to_numpy()
                if not np.array_equal(got, want):
                    raise SystemExit(f"{source}->{target} {typ}: target ID/order mismatch")
            files[str(p.relative_to(ROOT))] = {"bytes": p.stat().st_size, "sha256": sha_file(p),
                                               "rows": len(x)}
    rec = {"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "fit_count": 57, "runs": runs, "files": files,
           "schema": ["MVT_ID_mvt", "target_fold", "source_fold", "expert_type", "correction"]}
    FROZEN.write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def target_data(fold: str) -> dict:
    """The only scoring label loader; it is unreachable until the frozen manifest verifies."""
    if not FROZEN.exists():
        raise RuntimeError("target residual blocked until every correction is frozen")
    frozen = json.loads(FROZEN.read_text())
    for rel, want in frozen["files"].items():
        p = ROOT / rel
        if sha_file(p) != want["sha256"] or p.stat().st_size != want["bytes"]:
            raise RuntimeError("a frozen correction changed")
    lf = load_label_free(fold)
    lab = pd.read_parquet(panel_path(fold), columns=["MVT_ID_mvt", "y_true"])
    yall = lab.set_index("MVT_ID_mvt").reindex(lf["pan"].MVT_ID_mvt).y_true.to_numpy("float64")
    return {**lf, "yall": yall, "rall": yall - lf["pan"].p_v22.to_numpy("float64")}


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None


def score_table(d: dict, corr_p: np.ndarray) -> dict:
    pop, y, r = d["pop"], d["yall"], d["rall"]
    u = np.zeros(len(r)); u[pop] = corr_p
    pred0 = d["pan"].p_v22.to_numpy("float64")
    pred1 = pred0.copy(); pred1[pop] += corr_p
    if not np.array_equal(pred0[~pop].view("uint64"), pred1[~pop].view("uint64")):
        raise RuntimeError("outside-P bytes changed")
    masks = {"overall": np.ones(len(r), bool), "le3600": y <= 3600, "gt3600": y > 3600,
             "no_nm": d["pan"].nm_missing.to_numpy(bool)}
    return {name: {"rows": int(m.sum()), "rmse_v22": rmse(r[m]), "rmse_corrected": rmse((r-u)[m])}
            for name, m in masks.items()}


def point_from_gain(base_mse: float, pfrac: float, gain: float) -> float:
    return float(np.sqrt(base_mse) - np.sqrt(base_mse - pfrac * gain))


def score() -> dict:
    if SCORE_SENTINEL.exists():
        raise SystemExit("target labels were already opened; score is write-once")
    check_sources()
    if not FROZEN.exists():
        raise SystemExit("corrections are not frozen")
    SCORE_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SCORE_SENTINEL.write_text(json.dumps({"opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    data = {f: target_data(f) for f in FOLDS}
    pair, vectors = {}, {f: {} for f in FOLDS}
    for target in FOLDS:
        ids = data[target]["P"].MVT_ID_mvt.to_numpy()
        rt = data[target]["rall"][data[target]["pop"]]
        vectors[target] = {typ: [] for typ in OUTPUT_TYPES}
        for source in FOLDS:
            if source == target: continue
            x = pd.read_parquet(correction_path(source, target))
            for typ in OUTPUT_TYPES:
                z = x[x.expert_type == typ]
                if not np.array_equal(z.MVT_ID_mvt.to_numpy(), ids): raise RuntimeError("ID/order moved")
                u = z.correction.to_numpy("float64")
                vectors[target][typ].append(u)
                pair[f"{source}->{target}:{typ}"] = {
                    "gain_s2_on_P": float(np.mean(rt**2 - (rt-u)**2)),
                    "tables": score_table(data[target], u)}
    rng = np.random.default_rng(BOOT_SEED)
    draws = {f: [] for f in FOLDS}
    for f, d in data.items():
        days = d["P"].day.to_numpy(); uniq = np.unique(days)
        groups = [np.flatnonzero(days == day) for day in uniq]
        draws[f] = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                    for _ in range(BOOT)]
    targets, board = {}, {}
    for typ in OUTPUT_TYPES:
        bm0, bm1, boot_b0, boot_b1 = 0., 0., np.zeros(BOOT), np.zeros(BOOT)
        for target, d in data.items():
            rfull, rt = d["rall"], d["rall"][d["pop"]]
            us = vectors[target][typ]
            gains = [float(np.mean(rt**2-(rt-u)**2)) for u in us]
            mean_gain = float(np.mean(gains)); b = float(np.mean(rfull**2)); frac = len(rt)/len(rfull)
            val = point_from_gain(b, frac, mean_gain)
            boot = np.empty(BOOT)
            for j, ix in enumerate(draws[target]):
                gb = np.mean([np.mean(rt[ix]**2-(rt[ix]-u[ix])**2) for u in us])
                # Resample P effect, retaining the registered full-fold base level.
                boot[j] = point_from_gain(b, frac, float(gb))
            mean_u = np.mean(us, axis=0)
            targets.setdefault(target, {})[typ] = {"pair_gains_s2": gains, "mean_gain_s2_on_P": mean_gain,
                "point_s": val, "ci95_day_s": np.percentile(boot,[2.5,97.5]).tolist(),
                "mean_correction_tables": score_table(d, mean_u)}
            bm0 += SHARES[target]*b; bm1 += SHARES[target]*(b-frac*mean_gain)
            boot_b0 += SHARES[target]*b; boot_b1 += SHARES[target]*(b-frac*np.array([
                np.mean([np.mean(rt[ix]**2-(rt[ix]-u[ix])**2) for u in us]) for ix in draws[target]]))
        bp = float(np.sqrt(bm0)-np.sqrt(bm1)); bdraw=np.sqrt(boot_b0)-np.sqrt(boot_b1)
        board[typ] = {"point_s": bp, "ci95_day_s": np.percentile(bdraw,[2.5,97.5]).tolist()}
    contrasts = {}
    for other in ("sham", "airport_constant"):
        contrasts[f"aware_minus_{other}"] = {}
        for target in FOLDS:
            a, o = targets[target]["aware"], targets[target][other]
            # Paired bootstrap difference is reconstructed from stored target boot logic.
            rt=data[target]["rall"][data[target]["pop"]]; b=float(np.mean(data[target]["rall"]**2)); frac=len(rt)/len(data[target]["rall"])
            vals=[]
            for ix in draws[target]:
                ga=np.mean([np.mean(rt[ix]**2-(rt[ix]-u[ix])**2) for u in vectors[target]["aware"]])
                go=np.mean([np.mean(rt[ix]**2-(rt[ix]-u[ix])**2) for u in vectors[target][other]])
                vals.append(point_from_gain(b,frac,float(ga))-point_from_gain(b,frac,float(go)))
            contrasts[f"aware_minus_{other}"][target]={"point_s":a["point_s"]-o["point_s"],"ci95_day_s":np.percentile(vals,[2.5,97.5]).tolist()}
        contrasts[f"aware_minus_{other}"]["board"]={"point_s":board["aware"]["point_s"]-board[other]["point_s"]}
    aw=board["aware"]["point_s"]
    sham_nonpos=sum(contrasts["aware_minus_sham"][f]["point_s"]<=0 for f in FOLDS)
    advance=(aw>=.05 and targets["july_loo"]["aware"]["ci95_day_s"][0]>0 and targets["january"]["aware"]["ci95_day_s"][0]>0 and targets["december"]["aware"]["point_s"]>=0 and sum(contrasts["aware_minus_sham"][f]["ci95_day_s"][0]>0 for f in FOLDS)>=2 and aw>board["airport_constant"]["point_s"])
    gate="ADVANCE" if advance else ("FAIL" if aw<=0 or sham_nonpos>=2 else "MAYBE")
    return {"gate":gate,"historical_within_month_s":{"point":.19273,"ci95":[.10881,.29912]},
            "pair_results":pair,"targets":targets,"board":board,"contrasts":contrasts,"bootstrap":{"draws":BOOT,"seed":BOOT_SEED}}


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=float))


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("fit-source"); p.add_argument("source",choices=FOLDS)
    sub.add_parser("freeze"); sub.add_parser("score")
    args=ap.parse_args()
    if args.cmd=="fit-source": write_json(EXP/f"analysis/fit-{args.source}.json",fit_source(args.source))
    elif args.cmd=="freeze": print(json.dumps(freeze(),indent=2))
    else: write_json(EXP/"analysis/results.json",score())


if __name__ == "__main__": main()
