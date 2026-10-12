"""exp-147/148: member D1-winter (`D1-winter-deice-v1`) under one week-pair fit.

    PRC_SOURCE_ROOT=<tree> python D1-winter_wp.py --fit outer-3 --out <dir> --threads 6 [--smoke]

`--fit` is a `pilot.FITS` name (`outer-k`, `inner-a-b` with a < b, `final`); `--split NAME` runs
the fit that writes split NAME (`_wp_common.resolve_fit`). One fit trains on the label rows
of the first split of `pilot.artifacts_of(fit)` and writes one artifact per (split, owner):
an inner fit writes two, `final` writes one that scores the 344,841 ranking departures.

The member is exp-012's `winter_arm.py --ds-columns deice` (retro-v1 G1cv-s7 plus lirf, phi,
noac head, residual guard and the one day-state column `ds_deice_hr`), imported from
`$PRC_SOURCE_ROOT` with the pinned exp-007 pipeline it imports. Its `run_fold`
(winter_arm.py:161-254) reads `prc.data.FOLDS[fold]` and whole months, so it cannot take a
row selection; this driver calls the same functions in the same order on other rows:

* frames: `R.frame_for_months` over the twelve 2025 months (v2 features, label-free:
  features.py:217-228 blanks every departure label on entry); for `final` the two ranking
  months are built separately with `R.build_features`, as `build_submission` does
  (winter_arm.py:326-330).
* before any fitted step reads a raw frame, every departure label outside the fit's label
  rows is blanked in that frame (checked with `Split.assert_training_frame`);
* `lirf_step`, `phi_step`, `day_state_step.fit_transform` with the twelve months as the
  training months. phi is the only fitted step: `PhiCV` fits one model per month on the
  label rows of the other eleven, and every 2025 row of month m, training or scored, gets
  the model without month m. The ranking rows of `final` get the average of the twelve,
  as served. `ds_deice_hr` reads no fit (day_state_features.py:234-267);
* fit: winter_arm.py:186-216 with seed 7 only (the fold form): early stop on the last 15 %
  of training rows by take-off time, refit at best / 0.85, noac head;
* floor and guard: winter_arm `_postprocess`, with `R._guard_bounds` replaced for the call
  by the range of r = BLOCK - AOBT over the training rows (the member reads whole months
  from disk, guard.py:23-45, which on a week-pair fit includes held days).

Output: `MVT_ID_mvt`, `y_pred` (floored at 60 s and guarded, as in the published OOF and
ranking files), `y_pred_s0` (the seed-7 prediction before the floor and guard of
`_postprocess`). Differences from the published member are listed in `DIFFERENCES`.
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

STAGE = "D1-winter"
SEED = 7
DS_COLUMNS = "deice"
N_COLUMNS = 97                       # MVT_ID_mvt + 96 features (result-D1w-deice.json)
ID = "MVT_ID_mvt"
SMOKE_CFG = {"max_rounds": C.SMOKE_ROUNDS, "es_rounds": 10, "noac_rounds": C.SMOKE_ROUNDS}

DIFFERENCES = [
    "rows: training rows are the split's label rows of all twelve 2025 months (old fold: 11 "
    "whole months; served: 12 whole months); scored rows are the split's held departures",
    "phi: a scored 2025 row gets the PhiCV model fitted without its own month (the one model "
    "that reads no label of that month); the old fold gave the validation month the mean of the "
    "11 models, the served ranking the mean of the 12 (final here does the same)",
    "phi and day-state reference: fitted over twelve months (label rows only for phi); old fold "
    "used the 11 training months",
    "guard bounds: min and max of r over the training rows; old fold and served path read whole "
    "months from disk",
    "seeds: 7 only; served ranking bagged 7, 107, 207 at one frozen count",
    "rounds: early stop on the last 15 % of training rows (two_pass), as the old fold; served "
    "ranking used the three-month budget scan (125 rounds, winter_arm.py:272-310)",
    "threads: --threads (recorded); the published member ran at 6, linear_tree=True is not "
    "thread-count reproducible",
]


def parse(argv=None) -> argparse.Namespace:
    """`_wp_common.parse_args` plus `--device` (D1-winter accepts cpu only)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fit", help="a pilot.FITS name: outer-k, inner-a-b (a<b) or final")
    g.add_argument("--split", help="alias: the fit that writes this split's artifact")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{C.SMOKE_FRACTION:.0%} of the training rows, {C.SMOKE_ROUNDS} "
                         "rounds, written under <out>/smoke/")
    a = ap.parse_args(argv)
    if a.threads < 1:
        raise SystemExit("--threads must be at least 1")
    if a.device != "cpu":
        raise SystemExit("D1-winter is LightGBM on CPU; it has no GPU path")
    for v in C.THREAD_VARS:           # before numpy is imported; PRC_THREADS sets num_threads
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
    src, WP, P = C.bootstrap()
    import numpy as np
    import pandas as pd

    a.fit, arts = C.resolve_fit(P, a)
    final = a.fit == "final"
    for s, _ in arts:
        C.refuse_existing(a.dest, STAGE, s)
    train_split = arts[0][0]

    W = C._load("exp012_winter_arm", src / "experiments/exp-012-blend2/winter_arm.py")
    R, D = W.R, W.D
    pipe = src / "experiments/exp-007-winter/code"
    for mod, want in ((R, pipe / "pipeline/run.py"), (W.guard_mod, pipe / "pipeline/guard.py"),
                      (W.lirf_step, pipe / "pipeline/lirf_step.py"),
                      (W.phi_step, pipe / "pipeline/phi_step.py"),
                      (W.day_state_step, pipe / "day_state_step.py"),
                      (sys.modules["features"], pipe / "pipeline/features.py"),
                      (D, src / "prc/data.py")):
        if Path(mod.__file__).resolve() != want.resolve():
            raise RuntimeError(f"{mod.__name__} resolved to {mod.__file__}, not {want}")
    if int(R.DEFAULT_PARAMS["num_threads"]) != a.threads:
        raise RuntimeError(f"pipeline num_threads {R.DEFAULT_PARAMS['num_threads']} != {a.threads}")

    cfg = W.base_cfg(SEED, STAGE)
    if a.smoke:
        cfg.update(SMOKE_CFG)
    months = list(D.MONTHS)

    # ---- frames: twelve 2025 months, label-free features ------------------------------
    R._reseed(SEED)
    raw: dict = {}
    X_all, y_all, t_all = R.frame_for_months(months, cfg["features"], keep_raw=raw)
    inputs = dict(file_record(D.month_file(m), src) for m in months)
    side = pd.concat([raw[m].loc[raw[m]["PHASE_mvt"].eq("DEP").to_numpy(), [ID, WP.TIME_COLUMN]]
                      for m in months], ignore_index=True)
    if not np.array_equal(side[ID].to_numpy(), X_all[ID].to_numpy()):
        raise RuntimeError("feature rows are not aligned with the raw departure rows")
    lab = train_split.label_mask(side)
    # blank every departure label the fit may not read, before any fitted step sees a frame
    for m in months:
        f = raw[m]
        off = f["PHASE_mvt"].eq("DEP").to_numpy() & ~train_split.label_mask(f)
        f.loc[off, "BLOCK_TIME_UTC_mvt"] = pd.NaT
        f.loc[off, "TAXITIME_SEC_mvt"] = np.nan
        for col in WP.LABEL_COLUMNS:
            train_split.assert_training_frame(f, target=col, what=f"raw {m} {col}")
    y_all = y_all.where(lab)
    r_all = pd.concat([(f.loc[f["PHASE_mvt"].eq("DEP"), "BLOCK_TIME_UTC_mvt"]
                        - f.loc[f["PHASE_mvt"].eq("DEP"), "AOBT_3_flt"]).dt.total_seconds()
                       for f in (raw[m] for m in months)], ignore_index=True).to_numpy("float64")

    others: list = []
    if final:
        rr = D.load_ranking()
        inputs.update([file_record(D.data_dir() / "ranking.parquet", src)])
        r_months = sorted(rr["month"].unique())
        others = [(R.build_features(rr[rr["month"] == m], cfg["features"]).reset_index(drop=True),
                   rr[rr["month"] == m]) for m in r_months]
        del rr

    # ---- member steps, twelve months as the training months ----------------------------
    X_all, outs = W.lirf_step.fit_transform(months, X_all, raw, others, gate=cfg["lirf_gate"])
    others = list(zip(outs, [r for _, r in others]))
    X_all, outs = W.phi_step.fit_transform(months, X_all, raw, others,
                                           mode=cfg["phi_fit"], shrink=cfg["phi_shrink"])
    others = list(zip(outs, [r for _, r in others]))
    X_all, outs = W.day_state_step.fit_transform(months, X_all, raw, others, columns=DS_COLUMNS)
    del raw
    gc.collect()

    # ---- rows ---------------------------------------------------------------------------
    if final:
        Xs = pd.concat(outs, ignore_index=True)
        side_s = None
    else:
        sc = np.zeros(len(side), dtype=bool)
        for s, _ in arts:
            sc |= s.score_mask(side)
        Xs = X_all.loc[sc].reset_index(drop=True)
        side_s = side.loc[sc].reset_index(drop=True)
    missing = [c for c in X_all.columns if c not in Xs.columns]
    if missing:
        raise RuntimeError(f"scored frame is missing {missing}")
    Xs = Xs[list(X_all.columns)]
    Xtr = X_all.loc[lab].reset_index(drop=True)
    ytr = y_all.loc[lab].reset_index(drop=True)
    ttr = t_all[lab]
    r_tr = r_all[lab]
    side_tr = side.loc[lab].reset_index(drop=True)
    del X_all, y_all, outs, others
    gc.collect()
    if a.smoke:
        keep = C.smoke_keep(len(Xtr))
        Xtr, ytr = Xtr.loc[keep].reset_index(drop=True), ytr.loc[keep].reset_index(drop=True)
        ttr, r_tr = ttr[keep], r_tr[keep]
        side_tr = side_tr.loc[keep].reset_index(drop=True)
    train_split.assert_training_frame(side_tr, target=ytr.to_numpy(), what=f"{STAGE} {a.fit}")
    if not np.isfinite(ytr.to_numpy()).all():
        raise RuntimeError("a training row has no label")
    if Xtr.shape[1] != N_COLUMNS:
        raise RuntimeError(f"{Xtr.shape[1]} columns, the published member has {N_COLUMNS}")
    t_frames = time.monotonic() - t0
    print(f"[{a.fit}] train {Xtr.shape} scored {Xs.shape} built in {t_frames:.0f}s", flush=True)

    # ---- fit: winter_arm.py:186-216, one seed --------------------------------------------
    t1 = time.monotonic()
    ds_cols = [c for c in W.day_state_step.SETS[DS_COLUMNS] if c in Xs.columns]
    _, n_outside = W.ds_clip(Xs, W.ds_train_range(Xtr, ds_cols))
    cut = np.quantile(ttr, 1.0 - cfg["es_frac"])
    es = ttr >= cut
    fb = R.airport_fallback(Xtr, ytr, cfg["target"])
    base = R.base_score(Xtr, fb, cfg["target"])
    base_s = R.base_score(Xs, fb, cfg["target"])
    yt = R.forward_target(R.training_target(cfg, ytr.to_numpy()), cfg["target"])
    w = R.sample_weights(cfg, ytr.to_numpy(), Xtr)
    R._reseed(SEED)
    (pv,), best, rounds, _ = R.two_pass(cfg, Xtr, yt, w, base, es, [(Xs, base_s)])
    print(f"[{a.fit}] seed {SEED}: best {best} -> refit {rounds}", flush=True)
    (mm, ph), = R.noac_head(cfg, Xtr, ytr.to_numpy(), [Xs])
    if ph is not None and mm.any():
        pv = pv.copy()
        pv[mm] = ph
    ok = np.isfinite(r_tr)
    lo, hi = float(np.quantile(r_tr[ok], 0.0)), float(np.quantile(r_tr[ok], 1.0))
    member_bounds = R._guard_bounds
    R._guard_bounds = lambda _cfg, _months: (lo, hi)
    try:
        pred, pinfo = W._postprocess(cfg, [pv], Xs, None)
    finally:
        R._guard_bounds = member_bounds
    t_fit = time.monotonic() - t1

    out = pd.DataFrame({ID: Xs[ID].to_numpy(), "y_pred": pred, "y_pred_s0": pv})
    if not np.isfinite(out[["y_pred", "y_pred_s0"]].to_numpy()).all():
        raise RuntimeError("non-finite prediction")
    if final and len(out) != 344_841:
        raise RuntimeError(f"final scores {len(out)} rows, not 344,841")

    features = [c for c in Xtr.columns if c != ID]
    record = {
        "member": "D1-winter-deice-v1 (experiments/exp-012-blend2/winter_arm.py, "
                  "--ds-columns deice, fold form, one seed)",
        "fit": a.fit, "smoke": a.smoke, "device": "cpu", "threads": a.threads,
        "seeds": [SEED], "ds_columns": DS_COLUMNS,
        "settings": {k: v for k, v in cfg.items()},
        "smoke_settings": {"fraction": C.SMOKE_FRACTION, "seed": C.SMOKE_SEED, **SMOKE_CFG}
        if a.smoke else None,
        "train_rows": int(len(Xtr)), "scored_rows_fit": int(len(out)),
        "fit_result": {
            "best_iteration": int(best), "refit_rounds": int(rounds),
            "early_stop_cut_utc": str(pd.Timestamp(cut, unit="s", tz="UTC")),
            "early_stop_rows": int(es.sum()),
            "noac_head_scored_rows": int(mm.sum()) if ph is not None else 0,
            "guard_bounds_source": "min and max of BLOCK - AOBT over the training rows",
            "ds_cells_outside_training_range": int(n_outside),
            "n_columns": int(Xtr.shape[1]),
            "feature_list_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
            **pinfo},
        "num_threads_pipeline": int(R.DEFAULT_PARAMS["num_threads"]),
        "provenance": W.provenance(cfg),
        "differences_from_published_member": DIFFERENCES,
        "inputs": inputs,
        "code": C.code_hashes(src, [Path(__file__)]),
        "environment": C.environment(src, a.threads),
        "seconds": {"frames": round(t_frames, 1), "fit": round(t_fit, 1),
                    "total": round(time.monotonic() - t0, 1)},
    }
    for s, owner in arts:
        rows = out if final else out.loc[s.score_mask(side_s)].reset_index(drop=True)
        path = P.write(a.dest, STAGE, s, owner, rows, {**record, "scored_rows": int(len(rows))})
        print(json.dumps({"artifact": str(path), "owner": owner, "rows": int(len(rows)),
                          "nan": int(rows[["y_pred", "y_pred_s0"]].isna().to_numpy().sum()),
                          "train_rows": record["train_rows"], "seconds": record["seconds"]}),
              flush=True)


if __name__ == "__main__":
    main()
