"""exp-147/148: member F1-fresh (`F1-fresh-s1`, exp-018 tag fold4-final) under one week-pair fit.

    PRC_SOURCE_ROOT=<tree> CUDA_VISIBLE_DEVICES=1 python F1-fresh_wp.py --fit outer-3 \
        --out <dir> --threads 6 --device gpu [--smoke]

`--fit` is a `pilot.FITS` name (`outer-k`, `inner-a-b` with a < b, `final`); `--split NAME` runs
the fit that writes split NAME (`_wp_common.resolve_fit`). One fit trains on the label rows
of the first split of `pilot.artifacts_of(fit)` and writes one artifact per (split, owner):
an inner fit writes two, `final` writes one that scores the 344,841 ranking departures
(the departures of the final file under $PRC_RANKING_MODE=final, exp-186).

The member is exp-018's `model.fit_predict` with the settings of `run.py --tag fold4-final`
(results/fold4-final.json:4-9: 1,200 LightGBM rounds, z clipped at 6,000 s, 300 exact-head
rounds, ghost head `mixbase`, main head LightGBM + CatBoost 50/50, one seed), imported from
`$PRC_SOURCE_ROOT/experiments/exp-018-fresh/code`. Rows come from the member's own loader
(`dataset.load`) over its per-month cache `artifacts/exp-018-fresh/cache/<month>.parquet`
(label-free features, codes from `vocab.json`). The cache has no time column; a row's UTC
take-off day is 2025-01-01 + (doy - 1) (features.py:152-156 takes `doy` from
MVT_TIME_UTC_mvt; checked equal to the raw take-off day on all 2,085,047 rows, 2026-09-30).
The day lives in a side frame only; the model frame gets no new column (every column
outside `model.DROP` is a feature).

    train = the twelve 2025 tables, label rows of the split
    valid = the split's held rows (2025 fits) or the two ranking tables (final), y blanked

Encodings: `add_encodings` (dataset.py:105-139) gives training rows a leave-one-MONTH-out
encoding and scored rows the full one. This driver calls exp-147's
`learner_wp.add_encodings_month_out` instead: a scored row's encoding also leaves out the
training rows of its own month. Ranking months are not training months, so on `final` the
encoding equals the served one (checked: zero cells change). Then `fit_predict(...,
prepared=True)`. Differences from the published member are listed in `DIFFERENCES`.

Output: `MVT_ID_mvt`, `y_pred` (seconds, floored at 30 s, as in exp-018-fresh/oof/*.parquet).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _wp_common as C  # noqa: E402  (stdlib only at import)
sys.path.remove(str(HERE))

STAGE = "F1-fresh"
ID = "MVT_ID_mvt"
#: run.py --tag fold4-final (results/fold4-final.json:4-9); ghost_rounds is fit_predict's
#: default, which run.py does not pass
SETTINGS = dict(rounds=1200, clip_z=6000.0, exact_rounds=300, ghost_rounds=300,
                ghost_mode="mixbase", main_model="both")
SMOKE_SETTINGS = dict(rounds=C.SMOKE_ROUNDS, exact_rounds=C.SMOKE_ROUNDS,
                      ghost_rounds=C.SMOKE_ROUNDS, catboost_iterations=C.SMOKE_ROUNDS)

DIFFERENCES = [
    "rows: training rows are the split's label rows of all twelve 2025 months (old fold: 11 "
    "whole months; served: 12 whole months); scored rows are the split's held departures",
    "encodings: a scored 2025 row's te_y_*/te_off_* leave out the training rows of its own "
    "month (learner_wp.add_encodings_month_out); old fold and served scored rows got the full "
    "encoding, which there never shared a month with training rows; n_* unchanged",
    "seed: seed=None as run.py passes for --seeds 1 (LightGBM seed 7 from MAIN_PARAMS, CatBoost "
    "random_seed 7); served submit.py passed seed=7, which also pins bagging_seed 8 and "
    "feature_fraction_seed 9, so the served LightGBM draws differ",
    "CatBoost: --device gpu keeps the member's task_type GPU, devices '0' (the first visible "
    "CUDA device); --device cpu sets task_type CPU and thread_count --threads (a different "
    "estimator, for smoke tests only); allow_writing_files=False on both (no catboost_info/ "
    "in the working directory; the model is unchanged)",
    "threads: --threads replaces num_threads 6 in MAIN_PARAMS (via params) and GHOST_PARAMS "
    "(module constant set for this process); equal to the member at --threads 6",
    "scored frame: y is blanked (run.py kept the validation labels in the frame; fit_predict "
    "does not read them with curve=0)",
]


def parse(argv=None) -> argparse.Namespace:
    """`_wp_common.parse_args` plus `--device` (the CatBoost device)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fit", help="a pilot.FITS name: outer-k, inner-a-b (a<b) or final")
    g.add_argument("--split", help="alias: the fit that writes this split's artifact")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
                    help="CatBoost device; gpu is the member's own setting")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{C.SMOKE_FRACTION:.0%} of the training rows, {C.SMOKE_ROUNDS} "
                         "rounds, written under <out>/smoke/")
    a = ap.parse_args(argv)
    if a.threads < 1:
        raise SystemExit("--threads must be at least 1")
    for v in C.THREAD_VARS:           # before numpy is imported
        os.environ[v] = str(a.threads)
    a.out = a.out.resolve()
    a.dest = a.out / "smoke" if a.smoke else a.out
    return a


def file_record(path: Path, src: Path) -> tuple[str, dict]:
    p = Path(path).resolve()
    key = str(p.relative_to(src)) if src in p.parents else str(p)
    return key, {"sha256": C.sha(p), "bytes": p.stat().st_size}


def main() -> None:
    a = parse()
    t0 = time.monotonic()
    for v in ("PRC_WINDOW", "PRC_EXTRAS"):   # either one changes the member's feature set
        if os.environ.get(v, "").strip():
            raise SystemExit(f"{v} is set; fold4-final ran without it")
    src, WP, P = C.bootstrap()
    import numpy as np
    import pandas as pd

    a.fit, arts = C.resolve_fit(P, a)
    rank_months, n_rank = C.rank_months(), C.n_ranking()     # exp-186: after bootstrap()
    # `final` scores the ranking months; `jjrank` scores them and its own 2025 rows (`jjrefit`)
    rank_arts = [s for s, _ in arts if s.name in P.RANKING_SPLITS]
    month_arts = [s for s, _ in arts if s.name not in P.RANKING_SPLITS]
    for s, _ in arts:
        C.refuse_existing(a.dest, STAGE, s)
    train_split = arts[0][0]

    code = src / "experiments/exp-018-fresh/code"
    sys.path.insert(0, str(code))
    import dataset as D18                                     # noqa: E402
    import model as M18                                       # noqa: E402
    for mod in (D18, M18, sys.modules["features"]):
        if Path(mod.__file__).resolve().parent != code.resolve():
            raise RuntimeError(f"{mod.__name__} resolved to {mod.__file__}, not {code}")
    n_path = len(sys.path)
    LW = C._load("exp147_learner_wp", C.CODE / "learner_wp.py")
    del sys.path[0:len(sys.path) - n_path]   # learner_wp puts the worktree root first

    # ---- the member's cached month tables; a missing one would be rebuilt and would
    # append to vocab.json, so refuse instead
    months = [f"2025-{m:02d}" for m in range(1, 13)]
    wanted = months + (list(rank_months) if rank_arts else [])
    # exp-186: a 2026 table of the final file has its own cache name (`dataset.final_tag`, "-final"),
    # so the final mode never reads the tables of ranking.parquet; "" in the ranking mode
    tag = getattr(D18, "final_tag", None)
    if tag is None and C.ranking_mode() != "ranking":
        raise SystemExit(f"PRC_RANKING_MODE={C.ranking_mode()}: {D18.__file__} has no final_tag")
    paths = [D18.CACHE / f"{m}{tag(m) if tag else ''}.parquet" for m in wanted]
    lost = [str(p) for p in paths + [D18.VOCAB] if not p.exists()]
    if lost:
        raise SystemExit(f"member cache missing: {lost}")
    inputs = dict(file_record(p, src) for p in paths + [D18.VOCAB])

    full = D18.load(months)
    doy = full["doy"].to_numpy("float64")
    if not (np.isfinite(doy).all() and (doy == np.round(doy)).all()):
        raise RuntimeError("doy is not a whole day number on every row")
    day = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(doy.astype("int64") - 1, unit="D")
    side = pd.DataFrame({WP.TIME_COLUMN: day + pd.Timedelta(hours=12)})
    # a file month holds take-offs of that month, and (2 rows in 2025: 2025-02-28 in the March
    # file, 2025-08-31 in the September file) of the day before it; nothing else is expected
    fm = full["month"].astype(str)
    start = fm.map({m: pd.Timestamp(f"{m}-01") for m in months}).to_numpy("datetime64[ns]")
    end = fm.map({m: pd.Timestamp(f"{m}-01") + pd.offsets.MonthBegin(1) for m in months}
                 ).to_numpy("datetime64[ns]")
    d = day.tz_localize(None).to_numpy("datetime64[ns]")
    n_prev = int((d < start).sum())
    if ((d < start - np.timedelta64(1, "D")) | (d >= end)).any():
        raise RuntimeError("a row's doy falls outside its cache month and the day before it")
    lab = train_split.label_mask(side)
    parts, side_s = [], None
    if month_arts:
        sc = np.zeros(len(side), dtype=bool)
        for s in month_arts:
            sc |= s.score_mask(side)
        parts.append(full.loc[sc].reset_index(drop=True))
        side_s = side.loc[sc].reset_index(drop=True)
    n_month = len(parts[0]) if parts else 0
    if rank_arts:
        parts.append(D18.load(list(rank_months)))
        C.check_month_departures(parts[-1]["month"].astype(str).value_counts(), "F1 ranking tables")
    if len(parts) == 2:     # one fit scores both (per-row predictions); the tables must agree
        if not parts[0].dtypes.equals(parts[1].dtypes):
            raise RuntimeError("the 2025 and ranking month tables differ in columns or types")
        valid = pd.concat(parts, ignore_index=True)
    else:
        valid = parts[0]
    valid["y"] = np.nan
    train = full.loc[lab].reset_index(drop=True)
    side_tr = side.loc[lab].reset_index(drop=True)
    del full
    gc.collect()
    if a.smoke:
        keep = C.smoke_keep(len(train))
        train = train.loc[keep].reset_index(drop=True)
        side_tr = side_tr.loc[keep].reset_index(drop=True)
    train_split.assert_training_frame(side_tr, target=train["y"].to_numpy(),
                                      what=f"{STAGE} {a.fit}")
    t_frames = time.monotonic() - t0
    print(f"[{a.fit}] train {train.shape} scored {valid.shape} loaded in {t_frames:.0f}s",
          flush=True)

    # ---- fit: encodings (month-out rule), then the member's fit_predict ----------------
    t1 = time.monotonic()
    train, (valid,), enc_changed = LW.add_encodings_month_out(D18, train, [valid])
    if rank_arts and enc_changed:
        raise RuntimeError(f"the month-out rule moved {enc_changed} ranking encodings")
    st = dict(SETTINGS)
    # the member's GPU CatBoost; no catboost_info/ written into the working directory (a
    # logging switch; the model is unchanged, checked on the smoke bytes 2026-09-30)
    cat_params = {"allow_writing_files": False}
    if a.device == "cpu":
        cat_params.update({"task_type": "CPU", "devices": None, "thread_count": a.threads})
    if a.smoke:
        st.update({k: v for k, v in SMOKE_SETTINGS.items() if k in st})
        cat_params["iterations"] = SMOKE_SETTINGS["catboost_iterations"]
    params = {"num_threads": a.threads}
    M18.GHOST_PARAMS["num_threads"] = a.threads
    pred, info = M18.fit_predict(train, valid, rounds=st["rounds"], clip_z=st["clip_z"],
                                 params=params, ghost_rounds=st["ghost_rounds"], seed=None,
                                 curve=0, y_valid=None, exact_rounds=st["exact_rounds"],
                                 ghost_mode=st["ghost_mode"], prepared=True,
                                 main_model=st["main_model"], cat_params=cat_params)
    t_fit = time.monotonic() - t1

    out = pd.DataFrame({ID: valid[ID].to_numpy(), "y_pred": np.asarray(pred, "float64")})
    if not np.isfinite(out["y_pred"].to_numpy()).all():
        raise RuntimeError("non-finite prediction")
    if rank_arts and len(out) - n_month != n_rank:
        raise RuntimeError(f"{a.fit} scores {len(out) - n_month} ranking rows, not {n_rank:,}")

    import catboost
    feats = list(info["feats"])
    record = {
        "member": "F1-fresh-s1 (experiments/exp-018-fresh/code model.fit_predict, "
                  "run.py --tag fold4-final settings, one seed)",
        "fit": a.fit, "smoke": a.smoke, "device": a.device, "threads": a.threads,
        "settings": {**st, "seed": None, "prepared": True, "params": params,
                     "ghost_params_num_threads": a.threads, "cat_params": cat_params,
                     "catboost_member_default": "task_type GPU, devices '0', iterations 3000, "
                                                "depth 8, lr 0.06, random_seed 7"},
        "smoke_settings": {"fraction": C.SMOKE_FRACTION, "seed": C.SMOKE_SEED, **SMOKE_SETTINGS}
        if a.smoke else None,
        "train_rows": int(len(train)), "scored_rows_fit": int(len(out)),
        "fit_result": {"main_fit_rows": int(info["n_train"]), "ghost_fit_rows": int(info["n_ghost"]),
                       "encodings_changed_by_month_out": int(enc_changed),
                       "rows_taking_off_the_day_before_their_file_month": n_prev,
                       "features": len(feats),
                       "feature_list_sha256": hashlib.sha256(json.dumps(feats).encode()).hexdigest()},
        "vocab_note": "cache category codes come from vocab.json, which is machine-local; "
                      "the codes match the member only with the same vocab.json sha256",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "differences_from_published_member": DIFFERENCES,
        "inputs": inputs,
        "code": C.code_hashes(src, [Path(__file__), C.CODE / "learner_wp.py"]),
        "environment": {**C.environment(src, a.threads), "catboost": catboost.__version__},
        "seconds": {"frames": round(t_frames, 1), "fit": round(t_fit, 1),
                    "total": round(time.monotonic() - t0, 1)},
        **C.mode_record(ranking_rows=int(len(out) - n_month) if rank_arts else None),
    }
    for s, owner in arts:
        if s.name in P.RANKING_SPLITS:
            rows = out.iloc[n_month:].reset_index(drop=True)
        else:
            rows = out.iloc[:n_month].loc[s.score_mask(side_s)].reset_index(drop=True)
        path = P.write(a.dest, STAGE, s, owner, rows, {**record, "scored_rows": int(len(rows))})
        print(json.dumps({"artifact": str(path), "owner": owner, "rows": int(len(rows)),
                          "nan": int(rows["y_pred"].isna().sum()),
                          "train_rows": record["train_rows"], "seconds": record["seconds"]}),
              flush=True)


if __name__ == "__main__":
    main()
