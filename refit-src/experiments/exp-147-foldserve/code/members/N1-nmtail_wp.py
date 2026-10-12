"""exp-147/148: member N1-nmtail under one week-pair fit, one code path for fold and serve.

    PRC_SOURCE_ROOT=<tree with prc/, experiments/, artifacts/, data/raw> python N1-nmtail_wp.py \
        --fit outer-3 | inner-1-3 | final  --out <dir> --threads 6 [--smoke]

The member is `prc.nmtail.NMTailModel` on exp-010's frame, as
`experiments/exp-010-nmtail/code/run_folds.py` `run()` fits it: `frame(m)` (the exp-003
feature cache `artifacts/exp-003-anatomy/feat/<m>.parquet` joined to the re-occupation
signals and exp-010's derived columns), `FEATURES`, `CAT`, training rows
`has_nm & TAXITIME_SEC_mvt.notna()`, mode `phi`, 900 rounds, lr 0.04, 63 leaves, seed 7,
deterministic LightGBM (prc/nmtail.py VERSION 4).

Choices this driver makes (the lead's assignment, recorded in every JSON):

* Mode `phi`, one seed (7). The published fold files `N1-nmtail-v4-<fold>` are exp-010's
  `nmtail-phi_<fold>` OOF with the non-NM rows nulled afterwards; the month lane ran mode `p`.
* `y_pred` is NaN where the row has no NM join (`~has_nm`), as in the served ranking file
  (`experiments/exp-012-blend2/nmtail_submit.py`). base6's stack drops N1-NaN rows from
  `nm_late` (`experiments/exp-026-rebuild/code/rule.py:42-43`). `run()`'s airport-median
  fallback for those rows is therefore not fitted and not written.
* Re-occupation signals are rebuilt for each run from the raw months by exp-010's own
  `build_reocc.one` (compiled from `build_reocc.py`'s source; importing the module would
  create `artifacts/exp-010-nmtail/`), into a temporary directory that `run_folds.REOCC`
  points at. The builder masks each month like the ranking file first, so the signals read
  no departure label. For `final` the ranking months are built as `build_reocc.main` builds
  them: `one()` per `month` group of `prc.data.load_ranking()`, concatenated.

One fit, every split: 2025 departures outside the fit's `label_mask` get `TAXITIME_SEC_mvt`
and `BLOCK_TIME_UTC_mvt` blank (the frame is per row, so this equals dropping them), then
`run()`'s row selection. Scored rows: `score_mask` rows of each scored month (2025 fits), or
all 344,841 rows of `frame("ranking")` (`final`, the served path's `ranking_frame()`), each
passed through `prc.weekpair.context_frame` before `predict`. An inner fit writes two
artifacts (owners a and b). Output: `N1-nmtail-<split>.parquet` with `MVT_ID_mvt` (float64,
as in the cache), `y_pred` (seconds, NaN off NM) and `has_nm` (int8).

`--smoke`: a fixed-seed 2 % of the label rows and 20 rounds; under `<out>/smoke/`.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _wp_common as W                                           # noqa: E402

STAGE = "N1-nmtail"
MODE = "phi"
SEED = 7
E3 = "experiments/exp-003-anatomy/code"
E10 = "experiments/exp-010-nmtail/code"


def import_member(src: Path):
    """exp-010's run_folds from SRC, with its imports resolved the way its own run resolves
    them (its cwd-relative sys.path inserts put exp-003's `common` and `features` first)."""
    for name in ("common", "features", "run_folds"):
        if name in sys.modules:
            raise RuntimeError(f"a module named {name} is already imported")
    W._load("common", src / E3 / "common.py")
    W._load("features", src / E3 / "features.py")
    RF = W._load("run_folds", src / E10 / "run_folds.py")
    sys.path[:] = [p for p in sys.path if p not in (E3, E10)]    # its cwd-relative entries
    for mod in (sys.modules["common"], sys.modules["features"], RF):
        if src not in Path(mod.__file__).resolve().parents:
            raise RuntimeError(f"{mod.__name__} came from {mod.__file__}, not {src}")
    return RF


def main() -> None:
    a = W.parse_args()                          # sets the thread variables before numpy loads
    src, WP, P = W.bootstrap()
    fit, arts = W.resolve_fit(P, a)
    for s, _ in arts:
        W.refuse_existing(a.dest, STAGE, s)
    import numpy as np
    import pandas as pd
    import prc.data as d
    import prc.nmtail as nt
    RF = import_member(src)
    one = W.member_function(src / E10 / "build_reocc.py", "one",
                            {"pd": pd, "d": d, "reoccupation_signals": nt.reoccupation_signals})
    t0 = time.monotonic()
    first = arts[0][0]
    ranks = any(s.name in P.RANKING_SPLITS for s, _ in arts)     # `final`, `jjrank`
    # exp-186: the ranking frame takes the stem of the mode's ranking file ("ranking", or
    # "final_ranking"), so a final run never reads the board's exp-003 feature cache
    r_path = W.ranking_path(d) if ranks else None
    rank_name = r_path.stem if ranks else None
    names = list(d.MONTHS) + ([rank_name] if ranks else [])

    # ---- frames: exp-010 run_folds.frame over the exp-003 cache and fresh reocc signals
    inputs, reocc_sha = {}, {}
    with tempfile.TemporaryDirectory(prefix="exp147-n1-reocc-") as tmp:
        RF.REOCC = Path(tmp)
        for m in d.MONTHS:
            p = d.month_file(m)
            inputs[f"data/{p.name}"] = W.sha(p)
            one(d.load_month(m)).to_parquet(RF.REOCC / f"reocc_{m}.parquet", index=False)
        if ranks:
            inputs[f"data/{r_path.name}"] = W.sha(r_path)
            rk = d.load_ranking()
            parts = [one(g) for _, g in rk.groupby(rk["month"], observed=True)]
            pd.concat(parts, ignore_index=True).to_parquet(RF.REOCC / f"reocc_{rank_name}.parquet",
                                                          index=False)
            del rk, parts
        for m in names:
            reocc_sha[f"reocc_{m}.parquet"] = W.sha(RF.REOCC / f"reocc_{m}.parquet")
            f = d.ROOT / "artifacts/exp-003-anatomy/feat" / f"{m}.parquet"
            inputs[str(f.relative_to(src))] = W.sha(f)
        frames = {m: RF.frame(m) for m in names}
    reocc_s = time.monotonic() - t0

    # ---- training rows: run()'s selection over the fit's label rows
    tr = pd.concat([frames[m] for m in d.MONTHS], ignore_index=True)
    tr["TAXITIME_SEC_mvt"] = tr["TAXITIME_SEC_mvt"].astype("float64")
    lab = first.label_mask(tr)
    smoke = {}
    if a.smoke:
        lab = lab & W.smoke_keep(len(tr))
        smoke = {"fraction": W.SMOKE_FRACTION, "seed": W.SMOKE_SEED, "rounds": W.SMOKE_ROUNDS}
    tr.loc[~lab, "TAXITIME_SEC_mvt"] = np.nan
    tr.loc[~lab, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    tr_nm = tr[tr.has_nm & tr.TAXITIME_SEC_mvt.notna()]
    for s, _ in arts:
        s.assert_training_frame(tr_nm, target=tr_nm["TAXITIME_SEC_mvt"].to_numpy(),
                                what=f"{STAGE} {fit} taxi target")
    label_rows = int(tr["TAXITIME_SEC_mvt"].notna().sum())
    nm_rows = int(len(tr_nm))
    print(f"{STAGE} {fit}: {len(tr):,} departures, {label_rows:,} labelled, {len(tr_nm):,} "
          f"NM training rows ({time.monotonic() - t0:.0f}s)", flush=True)

    # ---- the member's own fit (run_folds.run)
    model = nt.NMTailModel(mode=MODE, features=RF.FEATURES, categoricals=list(RF.CAT),
                           seed=SEED, num_threads=a.threads)
    if a.smoke:
        model.n_estimators = W.SMOKE_ROUNDS
    t1 = time.monotonic()
    model.fit(tr_nm, tr_nm["TAXITIME_SEC_mvt"].to_numpy(float))
    fit_s = time.monotonic() - t1
    if int(model.model_.params.get("num_threads", -1)) != a.threads:
        raise SystemExit(f"LightGBM holds num_threads={model.model_.params.get('num_threads')}")
    del tr, tr_nm
    print(f"{STAGE} {fit}: fitted in {fit_s:.0f}s", flush=True)

    # ---- scored rows
    def score(va: pd.DataFrame) -> pd.DataFrame:
        va = WP.context_frame(va)
        WP.assert_context_masked(va, what=f"{STAGE} scored rows")
        has_nm = va["has_nm"].to_numpy().astype(bool)
        y = np.where(has_nm, model.predict(va), np.nan)
        return pd.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"].to_numpy(), "y_pred": y,
                             "has_nm": has_nm.astype("int8")})

    preds = {}
    for s, _ in arts:
        if s.name in P.RANKING_SPLITS:
            preds[s.name] = score(frames[rank_name])
        else:
            preds[s.name] = pd.concat([score(frames[m].loc[s.score_mask(frames[m])])
                                       for m in P.scored_months(s)], ignore_index=True)

    # ---- record and write, one artifact per (split, owner)
    settings = {"member": "N1-nmtail (prc.nmtail.NMTailModel on exp-010 run_folds.frame)",
                "nmtail_VERSION": nt.VERSION, "mode": model.mode, "seeds": [model.seed],
                "n_estimators": model.n_estimators, "learning_rate": model.learning_rate,
                "num_leaves": model.num_leaves, "min_data_in_leaf": model.min_data_in_leaf,
                "min_gap": model.min_gap, "num_threads": model.num_threads,
                "deterministic": True, "n_features": len(RF.FEATURES),
                "off_nm": "NaN (served convention; base6 rule.py:42-43 drops N1-NaN rows "
                          "from nm_late); run()'s airport-median fallback not fitted",
                "reocc": "rebuilt per run by build_reocc.one (source-compiled) from the raw "
                         "months; files hashed under reocc_sha256",
                "smoke": smoke or None}
    base = {"fit": fit, "settings": settings, "inputs": inputs, "reocc_sha256": reocc_sha,
            "code": W.code_hashes(src, [Path(__file__).resolve()],
                                  [src / E10 / "build_reocc.py"]),
            "environment": W.environment(src, a.threads),
            "reocc_and_frames_seconds": round(reocc_s, 1), "fit_seconds": round(fit_s, 1),
            "train_rows": label_rows, "train_rows_nm": nm_rows,
            **W.mode_record(ranking_rows=int(len(frames[rank_name])) if ranks else None)}
    for s, owner in arts:
        pred = preds[s.name]
        rec = {**base, "scored_rows": int(len(pred)),
               "scored_nan": int(pred["y_pred"].isna().sum()),
               "scored_off_nm": int((pred["has_nm"] == 0).sum()),
               "seconds_total": round(time.monotonic() - t0, 1)}
        path = P.write(a.dest, STAGE, s, owner, pred, rec)
        print(f"{STAGE} {s.name} owner {owner}: {len(pred):,} rows, {rec['scored_nan']} NaN "
              f"({rec['scored_off_nm']} off NM) -> {path}", flush=True)


if __name__ == "__main__":
    main()
