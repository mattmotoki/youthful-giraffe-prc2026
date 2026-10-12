"""exp-148: the one stacked learner over the member predictions, fitted per split and served.

    python experiments/exp-148-short-route/code/stacker.py --owner 3 --art DIR [--art DIR ...] \
        --features features.parquet --proxy proxy-features.parquet --out DIR --threads 8

`--owner k` fits the stacker of outer fold k: training rows are the held rows of
inner(k, j) for the four j != k, with the member predictions of the inner(k, j) fits; scored
rows are pair k with the member predictions of the outer(k) fits. `--owner final` fits the
served stacker: training rows are the five outer predictions (all 2025 January and July
rows), scored rows are the 344,841 ranking departures with the members' final fits. One
code path for both (SPEC section 3). Every member artifact is read through `pilot.load`,
which checks `admits`, the owner and the hash; the training frame passes
`assert_training_frame` with the target before the fit.

Model (frozen in SPEC section 3): anchor a = row mean of the base members {T-tailv4, C2-cat,
L2-hub8725, D1-winter, F1-fresh}; target y - a; one LightGBM (L2 loss) on the feature list
`feature_list()`; rounds from early stopping over the training pairs (leave one pair out),
then one refit on all training rows; served value max(a + f(x), 60 s).
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
ROOT = HERE.parents[2]
os.environ.setdefault("PRC_DATA_DIR", str(Path(os.environ["PRC_SOURCE_ROOT"]) / "data/raw"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp-147-foldserve/code"))
from prc import weekpair as WP       # noqa: E402
import pilot as P                    # noqa: E402

ID = "MVT_ID_mvt"
FLOOR = 60.0
BASE = ("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh")
MEMBERS = {"T-tailv4": "y_pred", "C2-cat": "y_pred", "L2-hub8725": "y_pred", "D1-winter": "y_pred",
           "F1-fresh": "y_pred", "N1-nmtail": "y_pred", "L": "pred_L", "Lp": "pred_Lp"}
PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 63,
          "min_data_in_leaf": 100, "feature_fraction": 0.7, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 10.0, "max_bin": 255, "seed": 601,
          "deterministic": True, "force_row_wise": True, "verbosity": -1}
MAX_ROUNDS, PATIENCE = 5000, 200
CATEGORICAL = ["ctx_airport", "ctx_cls"]
RES_TAGS = ("cell", "rwy_cell", "apt_30m", "rwy_30m", "rwy_60m", "rwy_minus_apt")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find(arts: list[Path], stage: str, split: str) -> Path:
    hits = [d / f"{stage}-{split}.parquet" for d in arts if (d / f"{stage}-{split}.parquet").exists()]
    if len(hits) != 1:
        raise SystemExit(f"{stage}-{split}: found {len(hits)} artifacts in {arts}")
    return hits[0]


def members_for(arts: list[Path], splits: list[tuple[str, int | str]], group: str,
                drop_ids: frozenset = frozenset()) -> tuple[pd.DataFrame, dict]:
    """One row per id: every member's prediction, the group (pair or 'rank') and the split.
    `drop_ids` (SPEC-r4): recorded ids without row features that two month splits can both hold."""
    rec, frames = {}, []
    for split, owner in splits:
        cols = None
        for stage, col in MEMBERS.items():
            src = MEMBER_SOURCE.get(stage, stage)
            path = find(arts, src, split)
            f = P.load(path, owner, what=f"{src} {split}")[[ID, col]].rename(columns={col: stage})
            f[ID] = f[ID].astype("int64")
            rec[f"{src}-{split}"] = sha(path)
            cols = f if cols is None else cols.merge(f, on=ID, how="outer", validate="one_to_one")
        cols["split"] = split
        if drop_ids:
            cols = cols[~cols[ID].isin(drop_ids)]
        frames.append(cols)
    out = pd.concat(frames, ignore_index=True)
    if out[ID].duplicated().any():
        raise SystemExit(f"{group}: an id has predictions from two splits")
    return out, rec


def queue_residuals(df: pd.DataFrame, anchor: np.ndarray) -> pd.DataFrame:
    """The 12 exp-047 residual columns against our anchor, neighbours on the same UTC day only
    (so each row's neighbours come from the same fit, on fold and serve alike)."""
    sys.path.insert(0, str(Path(os.environ["PRC_SOURCE_ROOT"]) / "experiments/exp-047-queue/code"))
    import state as Q                                             # noqa: E402  (exp-047)
    day = df["ctx_day"].astype(str).to_numpy()
    apt = pd.factorize(np.char.add(np.char.add(df["q_apt"].astype(str).to_numpy(), "|"), day))[0]
    rwy = pd.factorize(np.char.add(np.char.add(df["q_rwy"].astype(str).to_numpy(), "|"), day))[0]
    mvt = df["q_mvt"].to_numpy("float64")
    cell = np.char.add(np.char.add(df["q_apt"].astype(str).to_numpy(), "|"),
                       np.floor(mvt / 600.0).astype("int64").astype(str))
    out = {}
    for tag, col in (("Nf_", "q_taxi_f"), ("No_", "q_taxi_o")):
        taxi = df[col].to_numpy("float64")
        heard = np.isfinite(taxi)
        f = Q.neighbourhood(apt, rwy, cell, mvt, taxi - anchor, np.full(len(df), np.nan), heard, tag)
        for t in RES_TAGS:
            out[f"qr_{tag}n_res_{t}"] = f[f"{tag}n_res_{t}"].to_numpy("float64")
    return pd.DataFrame(out, index=df.index)


V22_MEMBERS = ("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh", "N1-nmtail")
STAGE = {"full": "S", "v22": "S22", "r2": "R2", "r2v22": "R22",
         # SPEC-r4: R2g = R2's owners with CatBoost on the GPU, R4 = nine months, s = season input
         "r2g": "R2g", "r2s": "R2s", "r4": "R4", "r4s": "R4s",
         # SPEC-r4 item N (SPEC-r4n): the grind's L pack as extra members on R2's owners, a matched pair
         "r2Lc": "R2Lc", "r2Lv": "R2Lv",
         # R4 with the 12 queue count columns that `startswith("q_n")` never matched (lead, 10-01)
         "r4q": "R4q",
         # R4 with the grind's bagged L and Lp members (desk, 10-01: one variant against R4)
         "r4Lv": "R4Lv",
         # R4 with D1-winter read from grind's four-seed mean (desk, 10-01: a swap against R4)
         "r4d1m": "R4d1m",
         # R4 with L2-hub8725 read from grind's four-seed mean (desk, 10-01: a swap against R4)
         "r4l2m": "R4l2m",
         # R4Lv plus opdi2's OPDI members; its controls equal Lcontrol and Lpcontrol (desk, 10-01)
         "r4LvO": "R4LvO"}
#: SPEC-r4n: members added to MEMBERS for an input (the anchor stays the mean of BASE)
EXTRA_MEMBERS = {"r2Lc": {"Lcontrol": "pred_Lcontrol"},
                 # SPEC-r4n amendments 1 and 2: Lbag4 replaces Lbag, Lweatherbag replaces Lweather in
                 # the variant arm (both before any pair fit)
                 "r2Lv": {"Lcontrol": "pred_Lcontrol", "Lbag4": "pred_Lbag4", "Lweatherbag": "pred_Lweatherbag",
                          "Lgeometry": "pred_Lgeometry"},
                 # desk 10-01: no L-only R4 pair; one R4 variant with the L and Lp bag members
                 "r4Lv": {"Lcontrol": "pred_Lcontrol", "Lbag4": "pred_Lbag4", "Lweatherbag": "pred_Lweatherbag",
                          "Lpcontrol": "pred_Lpcontrol", "Lpbag": "pred_Lpbag"},
                 # exp-155 Lcontrol155 and Lpcontrol155 equal Lcontrol and Lpcontrol (max |diff| 0),
                 # so the matched OPDI arm is R4Lv plus the two OPDI members
                 "r4LvO": {"Lcontrol": "pred_Lcontrol", "Lbag4": "pred_Lbag4", "Lweatherbag": "pred_Lweatherbag",
                           "Lpcontrol": "pred_Lpcontrol", "Lpbag": "pred_Lpbag",
                           "Lopdi": "pred_Lopdi", "Lpopdi": "pred_Lpopdi"}}
#: Member sources for an input: the member keeps its name (column, `d_` input, BASE and anchor
#: role) and reads the files `<source>-<split>` (desk, 10-01: seed-mean swaps of BASE members).
SOURCE_OF = {"r4d1m": {"D1-winter": "D1-winter-m4"}, "r4l2m": {"L2-hub8725": "L2-hub8725-m4"}}
MEMBER_SOURCE: dict[str, str] = {}
#: Only R2g's CatBoost runs on the GPU. On R2's months it lost 726 s² of local G against R2b (the
#: LightGBM form equal to 0.0; development read `logs/workstation/dev-score-R2g-*.log`), so R4, R4s
#: and R2s stay on the CPU as R2 (09-30 21:50 UTC).
GPU_INPUTS = ("r2g",)
#: SPEC-r2: the level 2 of each scored month and the served level 2, fitted without labels of
#: January or July at level 1 and without labels of the scored month at level 2.
R2_OWNERS = ("r01", "r07", "rfinal")
#: SPEC-r4 item M: the same on nine months (R2's six and March, October, November)
R4_OWNERS = ("q01", "q07", "qfinal")
INPUTS_OF = {**{o: ("r2", "r2v22", "r2g", "r2s", "r2Lc", "r2Lv") for o in R2_OWNERS}, **{o: ("r4", "r4s", "r4q", "r4Lv", "r4d1m", "r4LvO", "r4l2m") for o in R4_OWNERS}}
#: SPEC-r2 training rows without row features: two departures at 23:59:57 UTC on the last day of
#: February and of August that sit in the March and September raw files. The day-based members
#: (C2, L2, D1-winter, F1) score them in jjx-02 and jjx-08; `features_r2.py` reads each month's own raw
#: file, so they have no row features. They leave the level-2 training; any other row without
#: features still stops the fit (found by the first R2 run, 2026-09-30 14:58 UTC).
def r2_no_features() -> set[int]:
    """Read caller-owned boundary-row IDs; never redistribute challenge IDs."""
    path = os.environ.get("PRC_R2_NO_FEATURES")
    if not path:
        raise SystemExit("Set PRC_R2_NO_FEATURES to the boundary-row JSON from scripts/exp148_missing_features.py")
    values = json.loads(Path(path).read_text())["missing_feature_ids"]
    if len(values) != 2 or len(set(values)) != 2 or any(type(v) is not int for v in values):
        raise SystemExit("The recorded boundary exception needs exactly two distinct integer IDs")
    return set(values)


def drop_no_features(mem: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    have = mem[ID].isin(features[ID].astype("int64")).to_numpy()
    lost = sorted(int(i) for i in mem.loc[~have, ID])
    if not set(lost) <= r2_no_features():
        raise SystemExit(f"{len(lost)} member rows have no features, not only the recorded ones: {lost[:5]}")
    return mem[have].reset_index(drop=True), lost


def feature_list(features: pd.DataFrame, proxy: pd.DataFrame, inputs: str = "full") -> list[str]:
    if inputs in ("r2", "r2v22", "r2g", "r4", "r2Lc", "r2Lv", "r4Lv", "r4d1m", "r4LvO", "r4l2m"):   # SPEC-r2 section 3.1: no v23_*, no ctx_month
        return [c for c in feature_list(features, proxy, "v22" if inputs == "r2v22" else "full")
                if not c.startswith("v23_") and c != "ctx_month"]
    if inputs in ("r2s", "r4s"):                  # SPEC-r4 item L: the season input
        return feature_list(features, proxy, "r2") + ["ctx_summer"]
    if inputs == "r4q":                           # the queue count columns q_Nf_*, q_No_* (12)
        return feature_list(features, proxy, "r4") + sorted(
            c for c in features.columns if c.startswith(("q_Nf_", "q_No_")))
    """`full`: the SPEC list. `v22`: the inputs v22's route reads (six base members, the head's
    observation and shipped columns, the 24 queue columns, context), SPEC section 4."""
    if inputs == "v22":
        return ([f"d_{m}" for m in V22_MEMBERS] + ["anchor"]
                + ["ctx_airport", "ctx_month", "ctx_hour", "ctx_weekday", "ctx_cls"]
                + sorted(c for c in features.columns if c.startswith(("obs_", "shp_")))
                + sorted(c for c in features.columns if c.startswith("q_n"))
                + [f"qr_{t}n_res_{r}" for t in ("Nf_", "No_") for r in RES_TAGS]
                + ["q_taxi_f", "q_taxi_o", "x_obs_taxi_minus_anchor", "x_qtaxi_o_minus_anchor",
                   "x_qtaxi_f_minus_anchor"])
    fx = ([f"d_{m}" for m in MEMBERS] + ["anchor"]
          + ["ctx_airport", "ctx_month", "ctx_hour", "ctx_weekday", "ctx_cls"]
          + sorted(c for c in features.columns if c.startswith(("obs_", "shp_", "v23_")))
          # never matches (the columns are q_Nf_*, q_No_*; lead, 10-01); kept so that every
          # earlier stage stays reproducible; input r4q adds the 12 columns
          + sorted(c for c in features.columns if c.startswith("q_n"))
          + [f"qr_{t}n_res_{r}" for t in ("Nf_", "No_") for r in RES_TAGS]
          + ["q_taxi_f", "q_taxi_o"]
          + sorted(c for c in proxy.columns if c not in (ID, "month"))
          + ["x_obs_taxi_minus_anchor", "x_current250_minus_anchor", "x_implied_minus_anchor",
             "x_qtaxi_o_minus_anchor", "x_qtaxi_f_minus_anchor"])
    return fx


def frame(mem: pd.DataFrame, features: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    df = mem.merge(features, on=ID, how="left", validate="one_to_one")
    df = df.merge(proxy.drop(columns=["month"], errors="ignore"), on=ID, how="left", validate="one_to_one")
    if df["ctx_airport"].isna().any():
        raise SystemExit(f"{int(df.ctx_airport.isna().sum())} rows have no features")
    base = df[list(BASE)].to_numpy("float64")
    if (~np.isfinite(base)).all(axis=1).any():
        raise SystemExit("a row has no base member prediction")
    a = np.nanmean(base, axis=1)
    df["anchor"] = a
    for m in MEMBERS:
        df[f"d_{m}"] = df[m].to_numpy("float64") - a
    df = pd.concat([df, queue_residuals(df, a)], axis=1)
    df["x_obs_taxi_minus_anchor"] = df["obs_obs_taxi"] - a
    df["x_current250_minus_anchor"] = df["clock_current250_row"] - a
    df["x_implied_minus_anchor"] = df["clock_implied_row"] - a
    df["x_qtaxi_o_minus_anchor"] = df["q_taxi_o"] - a
    df["x_qtaxi_f_minus_anchor"] = df["q_taxi_f"] - a
    df["ctx_summer"] = df["ctx_month"].astype("int64").between(4, 9).astype("float64")   # SPEC-r4 item L
    for c in CATEGORICAL:
        df[c] = pd.Categorical(df[c].astype(str))
    return df


def labels_2025(ids: np.ndarray) -> np.ndarray:
    """SPEC-r2 section 3.3: departure taxi time of each 2025 id from the twelve raw month files."""
    from prc import data as D
    parts = []
    for m in P.MONTHS:
        f = D.load_month(m, columns=[ID, "PHASE_mvt", "TAXITIME_SEC_mvt"])
        f = f[f.PHASE_mvt == "DEP"]
        parts.append(pd.Series(f["TAXITIME_SEC_mvt"].to_numpy("float64"), index=f[ID].to_numpy("int64")))
    y = pd.concat(parts)
    if y.index.duplicated().any():
        raise SystemExit("a 2025 departure id is in two month files")
    y = y.reindex(pd.Index(ids))
    if y.isna().any():
        raise SystemExit(f"{int(y.isna().sum())} ids have no 2025 departure label")
    return y.to_numpy("float64")


def time_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["q_mvt"].to_numpy("float64"), unit="s", utc=True)
    return pd.DataFrame({WP.TIME_COLUMN: t, "PHASE_mvt": "DEP"}, index=df.index)


CAT_PARAMS = {"loss_function": "RMSE", "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3.0,
              "border_count": 254, "random_seed": 601, "od_type": "Iter", "od_wait": PATIENCE,
              "allow_writing_files": False, "task_type": "CPU", "verbose": False}


def cat_frame(df: pd.DataFrame, fx: list[str]) -> pd.DataFrame:
    x = df[fx].copy()
    for c in CATEGORICAL:
        x[c] = x[c].astype(str)
    return x


def splits_for(owner: int | str) -> tuple[list, list, int | None]:
    """(training splits, scored splits, month kept on the scored side or None)."""
    if owner == "final":
        return [(f"outer-{k}", k) for k in range(1, 6)], [("final", "final")], None
    if owner in ("m01", "m07"):                   # SPEC-r1: whole-month check on the janjul fits
        return [("janjul", "janjul")], [("janjul", "janjul")], int(owner[1:])
    if owner in R2_OWNERS or owner in R4_OWNERS:   # SPEC-r2 section 2, SPEC-r4 item M
        fits = P.FITS_R2 + (P.FITS_R4 if owner in R4_OWNERS else ())
        train = [(f, f) for f in fits] + [("janjul", "janjul")]
        if owner in ("rfinal", "qfinal"):
            return train, [("final", "final")], None
        return train, [("janjul", "janjul")], int(owner[1:])
    return ([(f"inner-{owner}-{j}", owner) for j in WP.inner_rotation(owner)],
            [(f"outer-{owner}", owner)], None)


def main() -> None:
    import catboost as cb
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--art", type=Path, action="append", required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--proxy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--inputs", choices=sorted(STAGE), default="full")
    # seed check (desk, 10-01): another seed for both learners, output stage <stage>seed<seed>;
    # with --rounds-from, the rounds of a finished fit of the same owner and inputs, no CV
    ap.add_argument("--seed", type=int)
    ap.add_argument("--rounds-from", type=Path)
    a = ap.parse_args()
    stage = STAGE[a.inputs] + (f"seed{a.seed}" if a.seed is not None else "")
    if a.rounds_from is not None and a.seed is None:
        raise SystemExit("--rounds-from needs --seed")
    MEMBERS.update(EXTRA_MEMBERS.get(a.inputs, {}))  # SPEC-r4n: before any member read or feature list
    MEMBER_SOURCE.update(SOURCE_OF.get(a.inputs, {}))
    t0 = time.monotonic()
    owner: int | str = a.owner if a.owner in ("final", "m01", "m07", *R2_OWNERS, *R4_OWNERS) else int(a.owner)
    r2 = owner in INPUTS_OF                        # the month-hold level 2 (SPEC-r2, SPEC-r4)
    if a.inputs not in INPUTS_OF.get(owner, ("full", "v22")):
        raise SystemExit(f"owner {owner} does not take inputs {a.inputs}")
    drop_ids = frozenset(r2_no_features()) if owner in R4_OWNERS else frozenset()
    consumer = P.consumer(owner)
    for ext in ("parquet", "json", "model.txt", "cat.cbm"):
        if (a.out / f"{stage}-{consumer.name}.{ext}").exists():
            raise SystemExit(f"{stage}-{consumer.name}.{ext} exists; write-once")
    train_splits, score_splits, month = splits_for(owner)
    feats = pd.read_parquet(a.features)
    proxy = pd.read_parquet(a.proxy)
    proxy[ID] = proxy[ID].astype("int64")
    no_features: list[int] = []
    if month is None:
        mem_tr, rec_tr = members_for(a.art, train_splits, "training", drop_ids)
        mem_sc, rec_sc = members_for(a.art, score_splits, "scored")
        if r2:
            mem_tr, no_features = drop_no_features(mem_tr, feats)
            for s_name, _ in train_splits:
                consumer.assert_admits(P.split(s_name), what=f"{s_name} for {consumer.name}")
    else:
        mem, rec_tr = members_for(a.art, train_splits, "janjul", drop_ids)
        if r2:
            mem, no_features = drop_no_features(mem, feats)
        rec_sc = {}
        for s_name, _ in train_splits:
            consumer.assert_admits(P.split(s_name), what=f"{s_name} for {consumer.name}")
        mon = feats.set_index(ID).ctx_month.reindex(mem[ID].to_numpy()).to_numpy()
        mem_tr, mem_sc = mem[mon != month].reset_index(drop=True), mem[mon == month].reset_index(drop=True)
    if mem_tr[ID].isin(mem_sc[ID]).any():
        raise WP.LabelLeak("a scored id is a training id")
    tr, sc = frame(mem_tr, feats, proxy), frame(mem_sc, feats, proxy)
    dropped = 0
    if r2:          # the consumer's buffer days (e.g. 2025-02-01 for January) train no stage
        keep = consumer.label_mask(time_frame(tr))
        dropped = int((~keep).sum())
        tr = tr.loc[keep].reset_index(drop=True)
    fx = feature_list(feats, proxy, a.inputs)
    fl_sha = hashlib.sha256(json.dumps(fx).encode()).hexdigest()

    # labels: training rows only, after the isolation check
    import prc.gscore as G
    y = labels_2025(tr[ID].to_numpy("int64")) if r2 else G.labels(tr[ID].to_numpy("int64"))
    consumer.assert_training_frame(time_frame(tr), target=y, what=f"S {consumer.name}")
    if not np.isfinite(y).all():
        raise SystemExit("a training label is not finite")
    z = y - tr["anchor"].to_numpy("float64")
    if r2:          # SPEC-r2 section 3.2: leave one training month out
        groups = tr["ctx_month"].to_numpy("int64")
    else:
        groups = WP.pair_of(tr["ctx_day"].astype(str).to_numpy())
        if (groups == 0).any():
            raise SystemExit("a training row is outside every week pair")
    params = {**PARAMS, "num_threads": a.threads}
    cparams = {**CAT_PARAMS, "thread_count": a.threads}
    if a.seed is not None:
        params["seed"], cparams["random_seed"] = a.seed, a.seed
    if a.inputs in GPU_INPUTS:                     # SPEC-r4 item M.4: CatBoost on the GPU
        cparams.update(task_type="GPU", devices="0", gpu_ram_part=0.15)   # several stackers share one GPU
    xc_tr = cat_frame(tr, fx)

    # rounds: leave one training pair out, mean of the best iterations, per family
    best, bestc, cv = [], [], {}
    oof = {"lgb": np.full(len(tr), np.nan), "cat": np.full(len(tr), np.nan)}
    ref = json.loads(a.rounds_from.read_text()) if a.rounds_from is not None else None
    if ref is not None and (ref["inputs"] != a.inputs or ref["feature_list_sha256"] != fl_sha
                            or ref["consumer"] != json.loads(json.dumps(consumer.manifest(), default=str))
                            or ref["train_rows"] != len(tr)):
        raise SystemExit(f"{a.rounds_from} is not a fit of the same owner, inputs, features and rows")
    for g in (sorted(set(groups.tolist())) if ref is None else []):
        m = groups == g
        dtr = lgb.Dataset(tr.loc[~m, fx], z[~m], categorical_feature=CATEGORICAL, free_raw_data=False)
        dva = lgb.Dataset(tr.loc[m, fx], z[m], categorical_feature=CATEGORICAL, reference=dtr)
        bst = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        best.append(bst.best_iteration)
        oof["lgb"][m] = bst.predict(tr.loc[m, fx], num_iteration=bst.best_iteration)
        cm = cb.CatBoostRegressor(iterations=MAX_ROUNDS, **cparams)
        cm.fit(cb.Pool(xc_tr[~m], z[~m], cat_features=CATEGORICAL),
               eval_set=cb.Pool(xc_tr[m], z[m], cat_features=CATEGORICAL), use_best_model=True)
        bestc.append(cm.get_best_iteration() + 1)
        oof["cat"][m] = cm.predict(xc_tr[m])
        mean = (oof["lgb"][m] + oof["cat"][m]) / 2
        cv[str(g)] = {"rows": int(m.sum()), "best_lgb": int(best[-1]), "best_cat": int(bestc[-1]),
                      "mse_anchor": float(np.mean(z[m] ** 2)),
                      "mse_lgb": float(np.mean((z[m] - oof["lgb"][m]) ** 2)),
                      "mse_cat": float(np.mean((z[m] - oof["cat"][m]) ** 2)),
                      "mse_mean": float(np.mean((z[m] - mean) ** 2))}
        print(json.dumps({str(g): cv[str(g)]}), flush=True)
    if ref is None:
        cv_pooled = {k: float(np.mean([v[k] for v in cv.values()])) for k in ("mse_anchor", "mse_lgb", "mse_cat", "mse_mean")}
        rounds = max(1, int(round(float(np.mean(best)))))
        rounds_c = max(1, int(round(float(np.mean(bestc)))))
    else:           # the reference fit's CV (other seed) and rounds
        cv_pooled, rounds, rounds_c = ref["cv_pooled"], ref["rounds"], ref["rounds_cat"]
    bst = lgb.train(params, lgb.Dataset(tr[fx], z, categorical_feature=CATEGORICAL), rounds)
    cm = cb.CatBoostRegressor(iterations=rounds_c, **{k: v for k, v in cparams.items() if not k.startswith("od_")})
    cm.fit(cb.Pool(xc_tr, z, cat_features=CATEGORICAL))
    f_l, f_c = bst.predict(sc[fx]), cm.predict(cat_frame(sc, fx))
    anc = sc["anchor"].to_numpy("float64")
    out = pd.DataFrame({ID: sc[ID].to_numpy("int64"),
                        "pred_S": np.maximum(anc + f_l, FLOOR),
                        "pred_Scat": np.maximum(anc + f_c, FLOOR),
                        "pred_Sb": np.maximum(anc + (f_l + f_c) / 2, FLOOR),
                        "anchor": anc, "anchor_served": np.maximum(anc, FLOOR)})
    if not np.isfinite(out.drop(columns=["anchor"]).to_numpy()).all():
        raise SystemExit("a scored prediction is not finite")
    imp = pd.Series(bst.feature_importance("gain"), index=fx).sort_values(ascending=False)
    a.out.mkdir(parents=True, exist_ok=True)
    model_path = a.out / f"{stage}-{consumer.name}.model.txt"
    cat_path = a.out / f"{stage}-{consumer.name}.cat.cbm"
    bst.save_model(str(model_path))
    cm.save_model(str(cat_path))
    spec = {n: sha(HERE.parent / n) for n in ("SPEC.md", "SPEC-r1.md", "SPEC-r2.md", "SPEC-r3.md", "SPEC-r4.md")
            if (HERE.parent / n).exists()}
    record = {"stage_what": f"exp-148 one stacked learner, inputs {a.inputs}", "inputs": a.inputs,
              "consumer": consumer.manifest(), "model_sha256": sha(model_path),
              "cat_model_sha256": sha(cat_path), "spec_sha256": spec,
              "train_splits": [s for s, _ in train_splits], "score_splits": [s for s, _ in score_splits],
              "scored_month": month, "train_rows": int(len(tr)), "scored_rows": int(len(out)),
              "train_rows_by_month": {str(k): int(v) for k, v in tr["ctx_month"].value_counts().sort_index().items()},
              "buffer_rows_dropped": dropped, "rows_without_features_dropped": no_features,
              "ids_left_out_of_every_split": sorted(drop_ids),
              "features": fx, "feature_list_sha256": fl_sha, "params": params, "cat_params": cparams,
              "rounds": rounds, "rounds_cat": rounds_c, "cv": cv, "cv_pooled": cv_pooled,
              "member_artifacts": {**rec_tr, **rec_sc},
              "features_sha256": sha(a.features), "proxy_sha256": sha(a.proxy),
              "code_sha256": sha(Path(__file__)), "lightgbm": lgb.__version__, "catboost": cb.__version__,
              "gain_top30": {k: float(v) for k, v in imp.head(30).items()},
              **({"seed": a.seed, "rounds_from": str(a.rounds_from) if ref is not None else None,
                  "rounds_from_sha256": sha(a.rounds_from) if ref is not None else None}
                 if a.seed is not None else {}),
              "seconds": round(time.monotonic() - t0, 1)}
    path = P.write(a.out, stage, consumer, owner, out, record)
    print(json.dumps({"artifact": str(path), "rounds": rounds, "rounds_cat": rounds_c, "rows": len(out),
                      "cv_pooled": cv_pooled, "seconds": record["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
