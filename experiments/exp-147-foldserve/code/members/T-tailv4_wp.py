"""exp-147/148: member T-tailv4 under one week-pair fit, one code path for fold and serve.

    PRC_SOURCE_ROOT=<tree with prc/ and data/raw> python T-tailv4_wp.py \
        --fit outer-3 | inner-1-3 | final  --out <dir> --threads 6 [--smoke]

The member is `prc.tail_model.TailModel` as `experiments/exp-009-noac2/deliver.py` fits it for
the published OOF v3 (`TailModel().fit(train, n_jobs=THREADS, noac_head="v4")`: phi on, no
keep mask, no weights, no placeholder regime, no window block, seed 7 and the five noac seeds
of the module, 500 main rounds, deterministic LightGBM) and as
`experiments/exp-012-blend2/tail_ranking_v4.py` fits it for the served ranking file.

One fit, every split:

* Training frame: the twelve 2025 months, all rows, in month order (`prc.data.load_month`).
  Departure `TAXITIME_SEC_mvt` and `BLOCK_TIME_UTC_mvt` are blanked outside the fit's
  `label_mask`; the rows stay, because the member's features are per-month context (stand
  occupancy, airport-day aggregates) and read no departure label (`prc/tail_features.py`
  docstring). Everything the member fits (both LightGBM heads, the noac heads, the band
  mixture, the phi tables of `prc.phi_features.PhiModel`, the categories) is fitted inside
  `TailModel.fit` from the labelled rows only. `final` (`WP.FINAL`) blanks nothing.
* Scored rows, 2025 fits: each scored month is rebuilt with `prc.weekpair.context_frame`
  (every departure label blank, as in the ranking file), predicted whole, and the rows of
  `score_mask` kept. An inner fit writes two artifacts (owners a and b) from one fit.
* Scored rows, `final`: the ranking file, one month at a time, as the served path does.
* Output: `T-tailv4-<split>.parquet` with `MVT_ID_mvt` (float64, as in the raw file) and
  `y_pred` (seconds, clipped at 60 s by the member), plus the JSON record (`pilot.write`).

`--smoke`: a fixed-seed 2 % of the label rows plus every noac label row (so the noac heads and
the band mixture still run), and 20 rounds for the main and noac heads; under `<out>/smoke/`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _wp_common as W                                           # noqa: E402

STAGE = "T-tailv4"
NOAC_PIN = "v4"


def main() -> None:
    a = W.parse_args()                          # sets the thread variables before numpy loads
    src, WP, P = W.bootstrap()
    fit, arts = W.resolve_fit(P, a)
    for s, _ in arts:
        W.refuse_existing(a.dest, STAGE, s)
    import numpy as np
    import pandas as pd
    import prc.data as D
    import prc.tail_model as TM
    if TM.PhiModel is None:          # tail_model.py:55-58 swallows the import error
        raise SystemExit("prc.phi_features did not import; T-tailv4 without phi is another member")
    t0 = time.monotonic()
    first = arts[0][0]

    # ---- training frame: every 2025 row; departure labels only on the fit's label rows
    inputs = {}
    frames = []
    for m in D.MONTHS:
        p = D.month_file(m)
        inputs[str(p.name)] = W.sha(p)
        frames.append(D.load_month(m))
    train = pd.concat(frames, ignore_index=True)
    del frames
    train["TAXITIME_SEC_mvt"] = train["TAXITIME_SEC_mvt"].astype("float64")
    dep = (train["PHASE_mvt"] == "DEP").to_numpy()
    lab = first.label_mask(train)
    smoke = {}
    if a.smoke:
        noac = train["AIRCRAFT_TYPE_mvt"].isna().to_numpy()
        lab = lab & (W.smoke_keep(len(train)) | noac)
        TM.MAIN_PARAMS = dict(TM.MAIN_PARAMS, n_estimators=W.SMOKE_ROUNDS)
        TM.NOAC_PARAMS = dict(TM.NOAC_PARAMS, n_estimators=W.SMOKE_ROUNDS)
        smoke = {"fraction": W.SMOKE_FRACTION, "seed": W.SMOKE_SEED, "rounds": W.SMOKE_ROUNDS,
                 "plus_every_noac_label_row": True}
    blank = dep & ~lab
    train.loc[blank, "TAXITIME_SEC_mvt"] = np.nan
    train.loc[blank, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    for s, _ in arts:
        s.assert_training_frame(train, target=train["TAXITIME_SEC_mvt"].to_numpy(),
                                what=f"{STAGE} {fit} taxi target")
        s.assert_training_frame(train, target=train["BLOCK_TIME_UTC_mvt"].to_numpy(),
                                what=f"{STAGE} {fit} phi target (block time)")
    train_rows = int((dep & train["TAXITIME_SEC_mvt"].notna().to_numpy()).sum())
    phi_rows = int((dep & train["BLOCK_TIME_UTC_mvt"].notna().to_numpy()).sum())
    print(f"{STAGE} {fit}: {len(train):,} rows, {int(dep.sum()):,} departures, "
          f"{train_rows:,} labelled ({time.monotonic() - t0:.0f}s)", flush=True)

    # ---- the member's own fit (exp-009 deliver.py main(); exp-012 tail_ranking_v4.py main())
    t1 = time.monotonic()
    model = TM.TailModel().fit(train, n_jobs=a.threads, noac_head=NOAC_PIN)
    fit_s = time.monotonic() - t1
    if model.phi is None or model.phi_cols != TM.PHI_COLUMNS:
        raise SystemExit(f"phi did not reach the NM head: {model.phi_cols}")
    del train
    print(f"{STAGE} {fit}: fitted in {fit_s:.0f}s", flush=True)

    # ---- scored rows
    preds = {}
    if fit == "final":
        rk = D.load_ranking()
        inputs["ranking.parquet"] = W.sha(D.data_dir() / "ranking.parquet")
        WP.assert_context_masked(rk, what="ranking file")
        parts = [model.predict(g)[["MVT_ID_mvt", "y_pred"]]
                 for _, g in rk.groupby("month", sort=True)]
        out = pd.concat(parts, ignore_index=True)
        n_dep = int((rk["PHASE_mvt"] == "DEP").sum())
        if len(out) != n_dep:
            raise SystemExit(f"final: {len(out):,} predictions for {n_dep:,} ranking departures")
        preds[first.name] = out
    else:
        by_month = {}
        for m in sorted({m for s, _ in arts for m in P.scored_months(s)}):
            vf = WP.context_frame(D.load_month(m))
            WP.assert_context_masked(vf, what=f"{m} scored frame")
            by_month[m] = (vf, model.predict(vf))
        for s, _ in arts:
            parts = []
            for m in P.scored_months(s):
                vf, p = by_month[m]
                sel = vf.index[s.score_mask(vf)]
                if not sel.isin(p.index).all():
                    raise SystemExit(f"{s.name} {m}: scored departures without a prediction")
                parts.append(p.loc[sel, ["MVT_ID_mvt", "y_pred"]])
            preds[s.name] = pd.concat(parts, ignore_index=True)

    # ---- record and write, one artifact per (split, owner)
    settings = {"member": "T-tailv4 (prc.tail_model.TailModel, noac head v4)",
                "tail_model_VERSION": TM.VERSION, "noac_head": NOAC_PIN,
                "n_jobs": a.threads, "use_phi": True, "phi_weight": TM.PHI_WEIGHT,
                "phi_cols": list(model.phi_cols), "keep": None, "weights": None,
                "placeholder": False, "window": None,
                "main_rounds": TM.MAIN_PARAMS["n_estimators"],
                "main_seed": TM.MAIN_PARAMS["seed"],
                "noac_rounds": TM.NOAC_PARAMS["n_estimators"],
                "noac_seeds": list(TM.NOAC_SEEDS),
                "deterministic": TM.MAIN_PARAMS["deterministic"], "smoke": smoke or None}
    base = {"fit": fit, "settings": settings, "inputs": inputs,
            "code": W.code_hashes(src, [Path(__file__).resolve()]),
            "environment": W.environment(src, a.threads),
            "fit_seconds": round(fit_s, 1), "train_rows": train_rows,
            "phi_rows_with_block_time": phi_rows}
    for s, owner in arts:
        pred = preds[s.name]
        rec = {**base, "scored_rows": int(len(pred)),
               "scored_nan": int(pred["y_pred"].isna().sum()),
               "seconds_total": round(time.monotonic() - t0, 1)}
        path = P.write(a.dest, STAGE, s, owner, pred, rec)
        print(f"{STAGE} {s.name} owner {owner}: {len(pred):,} rows, "
              f"{rec['scored_nan']} NaN -> {path}", flush=True)


if __name__ == "__main__":
    main()
