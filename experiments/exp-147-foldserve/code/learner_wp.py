"""exp-147: exp-134's proxy learner L under one split, one code path for fold and serve.

    python experiments/exp-147-foldserve/code/learner_wp.py --fit outer-3 \
        --input-root <dir with the Kaggle inputs> --out <work dir> [--cache <dir>]

`--fit` is one of `pilot.FITS` (`outer-k`, `inner-a-b` with a < b, `final`); an inner fit
writes the two artifacts `inner-a-b` and `inner-b-a` (exp-148). Every fit goes through
`fit_learner`, which is exp-134's `final_refit` feature logic
(`fit_exp134.py:434-470`): encodings fitted on the training rows only, the 21 proxies with
the 12-month label-free denominators, exp-130's 149 base features, 1,500 LightGBM rounds on
`y - anchor`, 6 threads, seeds 7/8/9. `final` trains on all 2025 months and scores the two
ranking months, exactly as the served v24 fit did; a week-pair split trains on the split's
label rows (`Split.label_mask`, checked by `assert_training_frame` with `y` as the target)
and scores its held days. The old fold path (`model_split`) differed from the final fit in
its training months (11 of 12), its encodings and its row views; here the only difference
between fold and serve is which rows carry targets and which rows are scored.

Inputs are the exact Kaggle packages of the served fit: the rebuilt exp130 clean-input package,
the rebuilt exp134 input package and the 17 `prc-harvest-*` kernel outputs (points verified
against `points_expected.json` by `proxy_features.build_all`). The code imported is the
packages' own `fit_exp134.py`, `proxy_features.py` and exp-130 `fit.py` (equal to the
committed copies). Output per artifact, through `pilot.write`: `L-<split>.parquet` (id, prediction, day,
airport) and `L-<split>.json` (split manifest, owner, input and code hashes, fit record).
Write-once.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from prc import weekpair as WP      # noqa: E402
sys.path.insert(0, str(HERE))
import pilot as P                   # noqa: E402

ID = "MVT_ID_mvt"
MONTHS = tuple(f"2025-{m:02d}" for m in range(1, 13))
RANK_MONTHS = ("2026-01", "2026-07")
SCORED_MONTHS = ("2025-01", "2025-07")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The columns `weekpair` reads: take-off time from exp-130's `utc_day` (UTC date of
    MVT_TIME_UTC_mvt, exp-130 prepare.py:107); all rows are departures."""
    t = pd.to_datetime(frame["utc_day"].astype(str), format="%Y-%m-%d").dt.tz_localize("UTC")
    return pd.DataFrame({WP.TIME_COLUMN: t + pd.Timedelta(hours=12)}, index=frame.index)


def setup(root: Path):
    d134 = next(p for p in root.rglob("fit_exp134.py")).parent
    sys.path.insert(0, str(d134))
    import proxy_features as PX                                     # noqa: E402
    PX.INPUT = root
    spec = importlib.util.spec_from_file_location("fit_exp134", d134 / "fit_exp134.py")
    F134 = importlib.util.module_from_spec(spec)
    sys.modules["fit_exp134"] = F134
    spec.loader.exec_module(F134)
    base, base_path = F134.base_fit_module()
    base.INPUT = root
    D18, F, M, D = base.setup_modules()
    code = {"fit_exp134.py": sha(d134 / "fit_exp134.py"),
            "proxy_features.py": sha(d134 / "proxy_features.py"),
            "exp130 fit.py": sha(base_path)}
    for mod in (D18, F, M, D):
        code[Path(mod.__file__).name] = sha(Path(mod.__file__))
    return PX, F134, base, D18, F, M, D, code


def raw_months_cached(PX, cache: Path) -> tuple[dict, dict]:
    """PX.build_all is label-free and the same for every split: build it once."""
    names = ("proxy_departures.parquet", "proxy_arrivals.parquet", "points_expected.json",
             "stand_geometry.csv")
    key = hashlib.sha256("|".join(sha(PX.exact_file(n)) for n in names).encode()
                         + sha(Path(PX.__file__)).encode()).hexdigest()[:16]
    path = cache / f"proxy-raw-months-{key}.pkl"
    if path.exists():
        with path.open("rb") as s:
            raw, audit = pickle.load(s)
        return raw, {**audit, "cache": str(path), "cache_key": key}
    t0 = time.monotonic()
    raw, audit = PX.build_all(MONTHS, RANK_MONTHS)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with tmp.open("wb") as s:
        pickle.dump((raw, audit), s, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    return raw, {**audit, "cache": str(path), "cache_key": key,
                 "build_seconds": round(time.monotonic() - t0, 1)}


def add_encodings_month_out(D18, train: pd.DataFrame, targets: list[pd.DataFrame],
                            smooth: float = 30.0):
    """exp-018 `add_encodings`, with ONE change: a scored row's encoding leaves out the
    training rows of its own month, as a training row's encoding already does.

    On the ranking months no training row shares the month, so the served encodings are
    unchanged (checked in `main` for `--split final`). On a week-pair split the held days'
    encodings no longer read the other weeks of their own month, which serving never has.
    """
    tr, targets = D18.add_encodings(train, targets, smooth)
    y = tr["y"].to_numpy(dtype="float64")
    off = y - tr["r_aobt"].to_numpy(dtype="float64")
    prior = {"y": float(np.nanmedian(y)), "off": float(np.nanmedian(off))}
    mcodes, months = pd.factorize(tr["month"].to_numpy())
    nm = len(months)
    changed = 0
    for name, cols in D18.ENC_SPECS:
        codes, uniq = pd.factorize(D18._key(tr, cols).to_numpy(), sort=False)
        K = len(uniq)
        cm = codes * nm + mcodes
        for t in targets:
            tm = pd.Index(months).get_indexer(t["month"].to_numpy())
            if (tm < 0).all():
                continue
            tc = pd.Index(uniq).get_indexer(D18._key(t, cols).to_numpy())
            for tag, v in (("y", y), ("off", off)):
                ok = np.isfinite(v)
                s_all = np.bincount(codes[ok], weights=v[ok], minlength=K)
                n_all = np.bincount(codes[ok], minlength=K).astype("float64")
                s_m = np.bincount(cm[ok], weights=v[ok], minlength=K * nm)
                n_m = np.bincount(cm[ok], minlength=K * nm).astype("float64")
                g = np.maximum(tc, 0)
                own = np.maximum(tc, 0) * nm + np.maximum(tm, 0)
                s = s_all[g] - np.where(tm >= 0, s_m[own], 0.0)
                n = n_all[g] - np.where(tm >= 0, n_m[own], 0.0)
                enc = (s + smooth * prior[tag]) / (n + smooth)
                new = np.where(tc >= 0, enc, prior[tag])
                changed += int((new != t[f"te_{tag}_{name}"].to_numpy("float64")).sum())
                t[f"te_{tag}_{name}"] = new
    return tr, targets, changed


#: exp-137's two row clock inputs (fit_exp137_final.py:54-62); exp-137's proxy module equals
#: exp-134's byte for byte (except trailing blanks), so one build serves both learners.
CLOCK_COLS = ["clock_current250_row", "clock_implied_row"]
LEARNERS = {"L": {"base_features": 149, "clock": False},    # exp-134 (v24)
            "Lp": {"base_features": 151, "clock": True}}    # exp-137 (v32), all rows scored


def with_clock_columns(frame: pd.DataFrame, raw_months: dict) -> pd.DataFrame:
    """exp-137 `add_clock_features` on a frame that mixes months."""
    out = frame.copy()
    src = pd.concat([raw_months[m]["row_features"][[ID, *CLOCK_COLS]] for m in sorted(set(out["month"]))],
                    ignore_index=True)
    src[ID] = pd.to_numeric(src[ID], errors="raise").astype("int64")
    if src[ID].duplicated().any():
        raise RuntimeError("clock inputs repeat an id")
    sel = src.set_index(ID).reindex(pd.to_numeric(out[ID], errors="raise").astype("int64").to_numpy())
    for c in CLOCK_COLS:
        out[c] = pd.to_numeric(sel[c], errors="coerce").to_numpy("float32")
    return out


def fit_learner(env, raw_months: dict, train_raw: pd.DataFrame, score_parts: list[pd.DataFrame],
                score_months: list[str], name: str, learner: str = "L"):
    """ONE path for fold and serve (exp-134 `final_refit` logic, fit_exp134.py:434-470;
    exp-137 `fit_exp137_final.py:103-160` for `Lp`, which adds the two clock inputs and
    trains on the same ordinary rows)."""
    PX, F134, base, D18, F, M, D, _ = env
    spec = LEARNERS[learner]
    if spec["clock"]:
        train_raw = with_clock_columns(train_raw, raw_months)
        score_parts = [with_clock_columns(p, raw_months) for p in score_parts]
    train_encoded, score_encoded, enc_changed = add_encodings_month_out(D18, train_raw, score_parts)
    if not any(m.startswith("2025") for m in score_months) and enc_changed:
        raise RuntimeError(f"{name}: the month-out rule moved {enc_changed} ranking encodings")
    train_ord = train_encoded.loc[train_encoded["ghost"].to_numpy("float64") < 0.5].reset_index(drop=True)
    score_ord = pd.concat([s.loc[s["ghost"].to_numpy("float64") < 0.5] for s in score_encoded],
                          ignore_index=True)
    denoms = PX.fit_denominators(raw_months, list(MONTHS))
    months = sorted(set(MONTHS) | set(score_months))
    z_source = pd.concat([PX.transform_rows(raw_months[m], denoms)[0][[ID, *PX.Z_COLS]]
                          for m in months], ignore_index=True).set_index(ID)
    train_z = F134.aligned_proxy_features(z_source, train_ord, list(MONTHS), f"{name} training")
    score_z = F134.aligned_proxy_features(z_source, score_ord, list(score_months), f"{name} scored")
    x_train, x_score, columns, categorical = base.feature_frames(train_ord, score_ord, F, M)
    if len(columns) != spec["base_features"] or (spec["clock"] and not set(CLOCK_COLS) <= set(columns)):
        raise RuntimeError(f"{name}: base feature count is {len(columns)}, not {spec['base_features']}")
    x_train = pd.concat([x_train.reset_index(drop=True), train_z.reset_index(drop=True)], axis=1)
    x_score = pd.concat([x_score.reset_index(drop=True), score_z.reset_index(drop=True)], axis=1)
    if x_train.shape[1] != spec["base_features"] + 21 or list(x_train.columns) != list(x_score.columns):
        raise RuntimeError(f"{name}: the learner does not have {spec['base_features'] + 21} matching columns")
    anchor_train, anchor_score = base.anchor_for(train_ord), base.anchor_for(score_ord)
    y = train_ord["y"].to_numpy("float64")
    if not (np.isfinite(anchor_train).all() and np.isfinite(y).all() and np.isfinite(anchor_score).all()):
        raise RuntimeError(f"{name}: a label or anchor is not finite")
    pred, rec = base.fit_arm(name, x_train, x_score, y - anchor_train,
                             np.ones(len(train_ord), dtype="float32"), categorical,
                             anchor_score, M, name)
    out = score_ord[[ID, "month", "utc_day", "airport"]].copy()
    out[ID] = out[ID].astype("int64")
    out[f"pred_{learner}"] = pred.astype("float64")
    features = list(x_train.columns)
    return out, {**rec["info"], "encodings_changed_by_month_out": enc_changed,
                 "feature_list_sha256":
                 hashlib.sha256(json.dumps(features).encode()).hexdigest(),
                 "features": len(features)}


FEATURE_MONTHS = ("2025-01", "2025-07", "2026-01", "2026-07")


#: exp-148 SPEC-r2: the level-2 training months without labels of January or July, and January
#: again as the parity check against the committed table.
FEATURE_MONTHS_R2 = ("2025-01", "2025-12", "2025-02", "2025-06", "2025-08")
#: exp-148 SPEC-r4: the three further months, and January again as the parity check.
FEATURE_MONTHS_R4 = ("2025-01", "2025-03", "2025-10", "2025-11")


def export_features(a, months=FEATURE_MONTHS, name="proxy-features") -> None:
    """exp-148 stacker inputs: the 21 proxies (12-month label-free denominators, as in every
    fit) and exp-137's two clock inputs for every row of the scored months. Label-free, so one
    table serves every split. Write-once `<name>.parquet` and `.json`."""
    pq, js = a.out / f"{name}.parquet", a.out / f"{name}.json"
    if pq.exists() or js.exists():
        raise SystemExit(f"{pq} exists; write-once")
    t0 = time.monotonic()
    env = setup(a.input_root)
    PX = env[0]
    raw, audit = raw_months_cached(PX, a.cache or a.out / "cache")
    denoms = PX.fit_denominators(raw, list(MONTHS))
    parts = []
    for m in months:
        z = PX.transform_rows(raw[m], denoms)[0][[ID, *PX.Z_COLS]].copy()
        z[ID] = pd.to_numeric(z[ID], errors="raise").astype("int64")
        c = raw[m]["row_features"][[ID, *CLOCK_COLS]].copy()
        c[ID] = pd.to_numeric(c[ID], errors="raise").astype("int64")
        f = z.merge(c, on=ID, how="outer", validate="one_to_one")
        f.insert(1, "month", m)
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    if out[ID].duplicated().any():
        raise RuntimeError("feature ids repeat")
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pq, index=False)
    js.write_text(json.dumps({"what": "exp-148 proxy and clock-input features (label-free)",
                              "rows": int(len(out)), "columns": list(out.columns),
                              "z_cols": list(PX.Z_COLS), "clock_cols": CLOCK_COLS,
                              "months": list(months), "output_sha256": sha(pq),
                              "code": env[7], "code_learner_wp": sha(Path(__file__)),
                              "proxy_audit": {k: v for k, v in audit.items() if k != "matches_by_month"},
                              "seconds": round(time.monotonic() - t0, 1)},
                             indent=1, sort_keys=True, default=str) + "\n")
    print(f"features: {len(out):,} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="one of pilot.FITS")
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--proxies-only", action="store_true", help="build the proxy cache and stop")
    ap.add_argument("--learners", default="L,Lp", help="comma list of L (exp-134) and Lp (exp-137)")
    a = ap.parse_args()
    learners = a.learners.split(",")
    if not set(learners) <= set(LEARNERS):
        raise SystemExit(f"unknown learner in {learners}")
    if a.fit == "features":
        export_features(a)
        return
    if a.fit == "features-r2":
        export_features(a, FEATURE_MONTHS_R2, "proxy-features-r2")
        return
    if a.fit == "features-r4":
        export_features(a, FEATURE_MONTHS_R4, "proxy-features-r4")
        return
    if a.fit not in P.FITS:
        raise SystemExit(f"unknown fit {a.fit}")
    arts = P.artifacts_of(a.fit)
    split = arts[0][0]
    for s, _ in arts:
        for ln in learners:
            for ext in ("parquet", "json"):
                if not a.proxies_only and (a.out / f"{ln}-{s.name}.{ext}").exists():
                    raise SystemExit(f"{ln}-{s.name}.{ext} exists; write-once")
    t0 = time.monotonic()
    env = setup(a.input_root)
    raw, audit = raw_months_cached(env[0], a.cache or a.out / "cache")
    print(f"proxy sources ready ({time.monotonic() - t0:.0f}s)", flush=True)
    if a.proxies_only:
        return
    base, D = env[2], env[6]
    frames = {m: base.month_frame(m) for m in MONTHS}
    inputs = {f"month_{m}.parquet": sha(base.locate_file(f"month_{m}.parquet")) for m in MONTHS}
    train_raw = pd.concat([frames[m] for m in MONTHS], ignore_index=True)
    train_raw = train_raw.loc[D.keep_reachable(train_raw)]
    keep = split.label_mask(wp_frame(train_raw))
    train_raw = train_raw.loc[keep]
    split.assert_training_frame(wp_frame(train_raw), target=train_raw["y"].to_numpy(),
                                what=f"L {a.fit}")
    if a.fit == "final":
        score_months = list(RANK_MONTHS)
        score_parts = [base.month_frame(m).drop(columns=["y"], errors="ignore") for m in RANK_MONTHS]
        inputs.update({f"month_{m}.parquet": sha(base.locate_file(f"month_{m}.parquet")) for m in RANK_MONTHS})
        rows_of = {"final": None}
    else:
        # the months of the split's held days (January and July for the week-pair and janjul
        # fits, as before; the jjx fits of SPEC-r2 score their own month)
        score_months = sorted({m for s, _ in arts for m in P.scored_months(s)})
        score_parts, rows_of = [], {}
        for m in score_months:
            f = frames[m]
            hit = np.zeros(len(f), dtype=bool)
            for s, _ in arts:
                sm = s.score_mask(wp_frame(f))
                if (hit & sm).any():
                    raise RuntimeError(f"{a.fit}: two artifacts score one row")
                hit |= sm
                rows_of.setdefault(s.name, set()).update(f.loc[sm, ID].astype("int64"))
            score_parts.append(f.loc[hit].drop(columns=["y"]))
    train_raw = train_raw.reset_index(drop=True)
    print(f"{a.fit}: {len(train_raw):,} training rows, "
          f"{sum(len(p) for p in score_parts):,} scored rows", flush=True)
    import lightgbm
    for ln in learners:
        t1 = time.monotonic()
        pred, rec = fit_learner(env, raw, train_raw, score_parts, score_months, f"{ln}_{a.fit}", ln)
        record = {"stage_what": {"L": "exp-134 proxy learner, final_refit logic",
                                 "Lp": "exp-137 with-clock proxy learner, all ordinary rows scored"}[ln],
                  "fit_name": a.fit, "learner": ln,
                  "train_split": split.manifest(), "train_rows": int(len(train_raw)),
                  "fit": rec, "code": env[7], "code_learner_wp": sha(Path(__file__)),
                  "code_pilot": sha(HERE / "pilot.py"), "weekpair_sha256": sha(ROOT / "prc/weekpair.py"),
                  "inputs": inputs,
                  "proxy_audit": {k: v for k, v in audit.items() if k != "matches_by_month"},
                  "seconds_fit": round(time.monotonic() - t1, 1),
                  "seconds_total": round(time.monotonic() - t0, 1),
                  "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                  "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS",)}}
        for s, owner in arts:
            part = pred if rows_of[s.name] is None else pred.loc[pred[ID].isin(rows_of[s.name])]
            path = P.write(a.out, ln, s, owner, part.reset_index(drop=True),
                           {**record, "scored_rows": int(len(part))})
            print(json.dumps({"artifact": str(path), "rows": int(len(part)),
                              "seconds_fit": record["seconds_fit"]}), flush=True)
        del pred
        gc.collect()
    del train_raw, frames
    gc.collect()


if __name__ == "__main__":
    main()
