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
                drop_ids: frozenset = frozenset(),
                split_of: dict[str, tuple[str, str]] | None = None) -> tuple[pd.DataFrame, dict]:
    """One row per id: every member's prediction, the group (pair or 'rank') and the split.
    `drop_ids` (SPEC-r4): recorded ids without row features that two month splits can both hold.
    `split_of` (fold form): a source read from another (split, owner) of the same rows, e.g. a
    member whose `jjrefit` vector equals its saved `janjul` one."""
    rec, frames = {}, []
    for split, owner in splits:
        cols = None
        for stage, col in MEMBERS.items():
            src = MEMBER_SOURCE.get(stage, stage)
            sp, ow = (split_of or {}).get(src, (split, owner))
            path = find(arts, src, sp)
            f = P.load(path, ow, what=f"{src} {sp}")[[ID, col]].rename(columns={col: stage})
            f[ID] = f[ID].astype("int64")
            rec[f"{src}-{sp}"] = sha(path)
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
         "r4LvO": "R4LvO",
         # R4 with clockhead's with-clock head (exp-159) as a ninth member, clock rows only (desk,
         # 10-02); A3c is the same head without the pushback-phase inputs
         "r4A3": "R4A3", "r4A3c": "R4A3c",
         # R4 with C2-cat swapped for grind's four-seed C2-cat-m4 (pack a003b343; desk, 10-02)
         "r4c2m": "R4c2m",
         # exp-163 test 3, the level-2 loss mask on R4's inputs (desk, 10-02): see LOSS_MASK
         "r4mk": "R4mk", "r4mks": "R4mks",
         # R4 with clockhead's twelve-month head HB1 (exp-159, 1,800 rounds) as a ninth member, clock
         # rows only, as r4A3 (desk, 10-02: after R4A3, before the loss mask and F5)
         "r4HB1": "R4HB1",
         # R4 with features' F2 turnaround member Lturn160 (exp-160 pack r4-002) as an extra member,
         # read against R4 and in v47's form (desk, 10-02: with F5, after R4A3 and HB1)
         "r4F2": "R4F2",
         # R4HB1 with the level-2 loss mask, rounds from R4HB1's own records (desk, 10-02)
         "r4HB1mk": "R4HB1mk",
         # R4 with features2's F5 runway member (grind's pack Lf5, ae561210; desk, 10-02)
         "r4F5": "R4F5",
         # R4 with diverse's runway sequence and row control (exp-170 SPEC-LEVEL2-015, native
         # export on F5's L ids) as two extra members (desk, 10-02: level-2 admission of the pair)
         "r4SR": "R4SR",
         # R4 with mlpc's two-seed MLP (exp-164 unit 008 pack-level2, on F5's splits and ids) as an
         # extra member (desk, 10-02 13:07: after the R4A3 step)
         "r4Mlp": "R4Mlp",
         # R4Mlp with the level-2 loss mask, rounds from R4Mlp's own records (exp-148, 10-02: the
         # served route R4mk carries the mask, so the stack-relevant arm needs it too)
         "r4Mlpmk": "R4Mlpmk",
         # R4mks with D1-winter read from D1s-winter, D1-winter with its early stop on the eligible rows
         # (desk, 10-02 23:38: stops on served rows; D1s_wp.py)
         "r4mksD1s": "R4mksD1s",
         # R4mks with the round cap raised: rounds from probes of the full CV (cap 15,000) run in
         # parallel (desk, 10-03 00:01: R4mks stopped at the 5,000 cap in five of eight months)
         "r4mksC": "R4mksC",
         # R4mks with features' F2 turnaround member Lturn160 (R4F2's member on the R4mks level 2;
         # desk 10-04 02:08, re-read on the final 0eec38e8)
         "r4mksF2": "R4mksF2",
         # R4mks with C2-cat read from grind's four-seed bag C2-cat-m4 (pack-C2-cat, PACK a003b343;
         # desk 10-04 02:24, re-read on the final 0eec38e8)
         "r4mksc2m": "R4mksc2m",
         # T2b (desk 10-04 05:24): R4mks with the six reception columns of T2-drift.md section F set to
         # 0 on every row (training, held and 2026), R4mks's own rounds (`fixed`)
         "r4mksZ6": "R4mksZ6",
         # seed control of the 10-04 re-reads (SPEC-reread-seed.md): R4mks with seed 602 for both learners,
         # R4mks's own rounds (`fixed`); the level-2 draw noise in the same form
         "r4mksS2": "R4mksS2"}
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
                           "Lopdi": "pred_Lopdi", "Lpopdi": "pred_Lpopdi"},
                 # exp-159 packs HA3 (c0cae43a) and HA3c (efcb8936): missing outside the clock rows
                 "r4A3": {"HA3": "pred_HA3"}, "r4A3c": {"HA3c": "pred_HA3c"},
                 # exp-159 pack HB1 (column checked against its PACK.json before staging)
                 "r4HB1": {"HB1": "pred_HB1"}, "r4HB1mk": {"HB1": "pred_HB1"},
                 # exp-160 PACK r4-002 (9 of 9 files checked), staged in kaggle/F2/out
                 "r4F2": {"Lturn160": "pred_Lturn160"}, "r4mksF2": {"Lturn160": "pred_Lturn160"},
                 # exp-153 pack f5-r4-023 (PACK ae561210) plus exp-161's janjul vector renamed
                 # (stage_f5.py), staged in kaggle/F5/out
                 "r4F5": {"Lf5": "pred_Lf5"},
                 # exp-170 pack-level2-015 (PACK 6f9d9fd8; stage_sr.py checks all 18 vectors)
                 "r4SR": {"Seq170": "pred_Seq170", "Row170": "pred_Row170"},
                 # exp-164 mlp-008 pack-level2 (PACK 41275249; stage_mlp.py checks all 9 vectors)
                 "r4Mlp": {"Mlp164": "pred_Mlp164"}, "r4Mlpmk": {"Mlp164": "pred_Mlp164"}}
#: Member sources for an input: the member keeps its name (column, `d_` input, BASE and anchor
#: role) and reads the files `<source>-<split>` (desk, 10-01: seed-mean swaps of BASE members).
SOURCE_OF = {"r4d1m": {"D1-winter": "D1-winter-m4"}, "r4l2m": {"L2-hub8725": "L2-hub8725-m4"},
             "r4c2m": {"C2-cat": "C2-cat-m4"}, "r4mksD1s": {"D1-winter": "D1s-winter"},
             "r4mksc2m": {"C2-cat": "C2-cat-m4"}}
#: T2b: inputs whose named feature columns are set to 0 on every row in `frame` (fold_serve.py too)
ZERO_OF = {"r4mksZ6": ("obs_n_ground_points", "obs_seg_points", "obs_seg_dt_median_s", "obs_seg_gap_max_s",
                       "obs_arr_gap_reception_s", "obs_seen_before_move_s")}
ZEROED: set[str] = set()
#: the seed of both level-2 learners for an input (default PARAMS/CAT_PARAMS seed 601)
SEED_OF = {"r4mksS2": 602}
MEMBER_SOURCE: dict[str, str] = {}
#: exp-163 test 3 (desk, 10-02): the rows of the served pass-through sets (serve_r2.py: class noac;
#: class ordinary without AOBT_3_flt) get zero label weight in both level-2 fits: they leave the
#: fits and stay in the frame. `fixed`: R4's rounds of the same owner, no CV. `eligible`: the rounds
#: of each held month stop on its eligible rows.
LOSS_MASK = {"r4mk": "fixed", "r4mks": "eligible", "r4HB1mk": "fixed", "r4Mlpmk": "fixed", "r4mksD1s": "eligible",
             "r4mksC": "eligible", "r4mksF2": "eligible", "r4mksc2m": "eligible",
             "r4mksZ6": "fixed", "r4mksS2": "fixed"}
#: the unmasked stage (and its inputs) whose records give a `fixed` mask its rounds
MASK_BASE = {"r4mk": ("R4", "r4"), "r4mks": ("R4", "r4"), "r4HB1mk": ("R4HB1", "r4HB1"),
             "r4Mlpmk": ("R4Mlp", "r4Mlp"), "r4mksD1s": ("R4", "r4"), "r4mksC": ("R4", "r4"),
             "r4mksF2": ("R4", "r4"), "r4mksc2m": ("R4", "r4"), "r4mksZ6": ("R4mks", "r4mks"), "r4mksS2": ("R4mks", "r4mks")}
#: inputs whose final fits take their rounds from development probes of the full CV, and the input the
#: probes ran with (the same members, features and loss rows; desk, 10-03 00:01 UTC)
PROBE_INPUTS = {"r4mksC": "r4mks"}
#: Only R2g's CatBoost runs on the GPU. On R2's months it lost 726 s² of local G against R2b (the
#: LightGBM form equal to 0.0; development read `logs/workstation/dev-score-R2g-*.log`), so R4, R4s
#: and R2s stay on the CPU as R2 (09-30 21:50 UTC).
GPU_INPUTS = ("r2g",)
#: SPEC-r2: the level 2 of each scored month and the served level 2, fitted without labels of
#: January or July at level 1 and without labels of the scored month at level 2.
R2_OWNERS = ("r01", "r07", "rfinal")
#: SPEC-r4 item M: the same on nine months (R2's six and March, October, November)
R4_OWNERS = ("q01", "q07", "qfinal")
INPUTS_OF = {**{o: ("r2", "r2v22", "r2g", "r2s", "r2Lc", "r2Lv") for o in R2_OWNERS}, **{o: ("r4", "r4s", "r4q", "r4Lv", "r4d1m", "r4LvO", "r4l2m", "r4A3", "r4A3c", "r4c2m", "r4mk", "r4mks", "r4HB1", "r4F2", "r4HB1mk", "r4F5", "r4SR", "r4Mlp", "r4Mlpmk", "r4mksD1s", "r4mksC", "r4mksF2", "r4mksc2m", "r4mksZ6", "r4mksS2") for o in R4_OWNERS}}
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
    if inputs in ("r2", "r2v22", "r2g", "r4", "r2Lc", "r2Lv", "r4Lv", "r4d1m", "r4LvO", "r4l2m", "r4A3", "r4A3c", "r4c2m", "r4mk", "r4mks", "r4HB1", "r4F2", "r4HB1mk", "r4F5", "r4SR", "r4Mlp", "r4Mlpmk", "r4mksD1s", "r4mksC", "r4mksF2", "r4mksc2m", "r4mksZ6", "r4mksS2"):   # SPEC-r2 section 3.1: no v23_*, no ctx_month
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
    for c in sorted(ZEROED):                        # T2b: after every derived column (none reads these)
        if c not in df.columns:
            raise SystemExit(f"zeroed column {c} is not a feature")
        df[c] = 0.0
    for c in CATEGORICAL:
        df[c] = pd.Categorical(df[c].astype(str))
    return df


def check_proxy_months(proxy: pd.DataFrame, what: str) -> None:
    """exp-186, final mode ($PRC_RANKING_MODE=final): the proxy table holds rows of every 2026 month of
    the mode. `frame` joins it left, so a table of the ranking mode leaves the February and June rows
    without proxies (NaN) and gives no error. The ranking mode checks nothing."""
    from prc import data as D
    if D.ranking_mode() == "ranking":
        return
    lost = sorted(set(D.rank_months()) - set(proxy["month"].astype(str)))
    if lost:
        raise SystemExit(f"{what}: the proxy table has no rows of {lost} ({D.ranking_mode()} mode)")


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


def pass_through_2025(df: pd.DataFrame) -> np.ndarray:
    """serve_r2.py's keep sets noac and ordna on 2025 rows: class noac, or class ordinary without
    AOBT_3_flt in the row's raw month file."""
    from prc import data as D
    parts = []
    for m in P.MONTHS:
        f = D.load_month(m, columns=[ID, "PHASE_mvt", "AOBT_3_flt"])
        f = f[f.PHASE_mvt == "DEP"]
        parts.append(pd.Series(f["AOBT_3_flt"].isna().to_numpy(), index=f[ID].to_numpy("int64")))
    na = pd.concat(parts)
    if na.index.duplicated().any():
        raise SystemExit("a 2025 departure id is in two month files")
    na = na.reindex(pd.Index(df[ID].to_numpy("int64")))
    if na.isna().any():
        raise SystemExit(f"{int(na.isna().sum())} ids have no 2025 departure row")
    cls = df["ctx_cls"].astype(str).to_numpy()
    return (cls == "noac") | ((cls == "ordinary") & na.to_numpy(bool))


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


def rounds_from_probes(paths, inputs, owner, fl_sha, train_rows, mask_rows, months) -> dict:
    """The rounds of an input in PROBE_INPUTS from probe records (`probe`): each CV month once per learner,
    the same owner, probe input, feature list, training rows and loss-mask rows; rounds = the rounded
    mean of the best iterations, as main's loop. `cap_bound`: months whose best lies within the
    patience of the cap (the cap still cut the search there)."""
    best, mse, anchor, caps, src, bound = {"lgb": {}, "cat": {}}, {"lgb": {}, "cat": {}}, {}, set(), {}, []
    for p in paths:
        r = json.loads(Path(p).read_text())
        if (r["owner"], r["inputs"], r["feature_list_sha256"], r["train_rows"], r["loss_mask_rows"]) != \
                (owner, PROBE_INPUTS[inputs], fl_sha, train_rows, mask_rows):
            raise SystemExit(f"{p} is not a probe of the same owner, inputs, features and rows")
        caps.add(int(r["cap"]))
        src[str(p)] = sha(Path(p))
        for g, mrec in r["months"].items():
            if anchor.setdefault(g, (mrec["rows"], mrec["mse_anchor"])) != (mrec["rows"], mrec["mse_anchor"]):
                raise SystemExit(f"month {g}: the probes disagree on its rows")
            for lrn in ("lgb", "cat"):
                if lrn not in mrec:
                    continue
                if g in best[lrn]:
                    raise SystemExit(f"month {g} ({lrn}) is in two probes")
                v, metric = float(mrec[lrn]["best_value"]), mrec[lrn]["metric"]
                if metric not in ("l2", "rmse", "RMSE"):
                    raise SystemExit(f"{p}: metric {metric}")
                best[lrn][g], mse[lrn][g] = int(mrec[lrn]["best"]), (v if metric == "l2" else v ** 2)
                if mrec[lrn]["best"] > r["cap"] - PATIENCE:
                    bound.append(f"{g}:{lrn}")
    want = {str(g) for g in months}
    for lrn in best:
        if set(best[lrn]) != want:
            raise SystemExit(f"{lrn}: probes cover months {sorted(best[lrn], key=int)}, not {sorted(want, key=int)}")
    # main's CV fields; no mse_mean (the probes keep no out-of-fold vectors)
    cv = {g: {"rows": int(anchor[g][0]), "best_lgb": best["lgb"][g], "best_cat": best["cat"][g],
              "mse_anchor": float(anchor[g][1]), "mse_lgb": mse["lgb"][g], "mse_cat": mse["cat"][g]}
          for g in sorted(want, key=int)}
    return {"cv": cv, "cv_pooled": {k: float(np.mean([v[k] for v in cv.values()]))
                                    for k in ("mse_anchor", "mse_lgb", "mse_cat")},
            "rounds": max(1, int(round(float(np.mean(list(best["lgb"].values())))))),
            "rounds_cat": max(1, int(round(float(np.mean(list(best["cat"].values())))))),
            "caps": sorted(caps), "probes": src, "cap_bound": sorted(bound)}


def probe(a, lgb, cb, tr, fx, z, lm, groups, params, cparams, xc_tr, stage, owner, fl_sha) -> None:
    """Development probe (desk, 10-03 00:01 UTC): the CV of the named held months, as in main's loop
    (training on the other months' loss rows, early stop on the held month's loss rows), with the cap
    --max-rounds; per learner the best iteration, the curve every 250 rounds and its value at 5,000."""
    cap = a.max_rounds or MAX_ROUNDS
    out = {"what": "exp-148 dev probe: level-2 CV curves with another round cap", "stage": stage,
           "owner": owner, "inputs": a.inputs, "cap": cap, "patience": PATIENCE, "learner": a.probe_learner,
           "feature_list_sha256": fl_sha, "train_rows": int(len(tr)), "loss_mask_rows": int((~lm).sum()),
           "months": {}, "code_sha256": sha(Path(__file__))}
    for g in a.probe_months:
        m = groups == g
        if not m.any():
            raise SystemExit(f"{g} is not a CV month of {owner}")
        t, v = ~m & lm, m & lm
        rec = {"rows": int(v.sum()), "mse_anchor": float(np.mean(z[v] ** 2))}
        if a.probe_learner in ("lgb", "both"):
            t1, ev = time.monotonic(), {}
            dtr = lgb.Dataset(tr.loc[t, fx], z[t], categorical_feature=CATEGORICAL, free_raw_data=False)
            dva = lgb.Dataset(tr.loc[v, fx], z[v], categorical_feature=CATEGORICAL, reference=dtr)
            bst = lgb.train(params, dtr, cap, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(PATIENCE, verbose=False), lgb.record_evaluation(ev)])
            name, curve = next(iter(ev["valid_0"].items()))
            rec["lgb"] = {"metric": name, "best": int(bst.best_iteration), "rounds_run": len(curve),
                          "best_value": float(curve[bst.best_iteration - 1]),
                          "at_5000": float(curve[4999]) if len(curve) >= 5000 else None,
                          "curve_every_250": [float(x) for x in curve[249::250]],
                          "seconds": round(time.monotonic() - t1, 1)}
        if a.probe_learner in ("cat", "both"):
            t1 = time.monotonic()
            cm = cb.CatBoostRegressor(iterations=cap, **cparams)
            cm.fit(cb.Pool(xc_tr[t], z[t], cat_features=CATEGORICAL),
                   eval_set=cb.Pool(xc_tr[v], z[v], cat_features=CATEGORICAL), use_best_model=True)
            curve = cm.get_evals_result()["validation"]["RMSE"]
            bi = cm.get_best_iteration()
            rec["cat"] = {"metric": "RMSE", "best": int(bi + 1), "rounds_run": len(curve),
                          "best_value": float(curve[bi]),
                          "at_5000": float(curve[4999]) if len(curve) >= 5000 else None,
                          "curve_every_250": [float(x) for x in curve[249::250]],
                          "seconds": round(time.monotonic() - t1, 1)}
        out["months"][str(g)] = rec
        print(json.dumps({str(g): {k: ({kk: vv for kk, vv in x.items() if kk != "curve_every_250"}
                                      if isinstance(x, dict) else x) for k, x in rec.items()}}), flush=True)
    a.probe_out.parent.mkdir(parents=True, exist_ok=True)
    a.probe_out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


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
    # development probe (desk, 10-03 00:01 UTC): the CV of the named held months only, with another
    # round cap and one or both learners; writes the curves to --probe-out, then stops (no fit, no vector)
    ap.add_argument("--probe-out", type=Path)
    ap.add_argument("--probe-months", type=int, nargs="+")
    ap.add_argument("--probe-learner", choices=("lgb", "cat", "both"), default="both")
    ap.add_argument("--max-rounds", type=int)
    # the rounds of an input in PROBE_INPUTS: the probe records of every CV month and both learners
    ap.add_argument("--rounds-from-probes", type=Path, nargs="+")
    a = ap.parse_args()
    if (a.probe_out is None) != (a.probe_months is None) or (a.max_rounds is not None and a.probe_out is None):
        raise SystemExit("--probe-out, --probe-months and --max-rounds go together (development probe only)")
    if a.probe_out is not None and a.probe_out.exists():
        raise SystemExit(f"{a.probe_out} exists; write-once")
    stage = STAGE[a.inputs] + (f"seed{a.seed}" if a.seed is not None else "")
    if a.rounds_from is not None and a.seed is None:
        raise SystemExit("--rounds-from needs --seed")
    MEMBERS.update(EXTRA_MEMBERS.get(a.inputs, {}))  # SPEC-r4n: before any member read or feature list
    MEMBER_SOURCE.update(SOURCE_OF.get(a.inputs, {}))
    ZEROED.update(ZERO_OF.get(a.inputs, ()))
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
    if any(s in P.RANKING_SPLITS for s, _ in score_splits):
        check_proxy_months(proxy, str(a.proxy))
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
    masked = pass_through_2025(tr) if a.inputs in LOSS_MASK else np.zeros(len(tr), bool)
    lm = ~masked                                    # the rows that enter the level-2 loss
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
    if a.inputs in SEED_OF:
        params["seed"], cparams["random_seed"] = SEED_OF[a.inputs], SEED_OF[a.inputs]
    if a.inputs in GPU_INPUTS:                     # SPEC-r4 item M.4: CatBoost on the GPU
        cparams.update(task_type="GPU", devices="0", gpu_ram_part=0.15)   # several stackers share one GPU
    xc_tr = cat_frame(tr, fx)

    if a.probe_out is not None:
        probe(a, lgb, cb, tr, fx, z, lm, groups, params, cparams, xc_tr, stage, consumer.name, fl_sha)
        return
    # rounds: leave one training pair out, mean of the best iterations, per family
    best, bestc, cv = [], [], {}
    oof = {"lgb": np.full(len(tr), np.nan), "cat": np.full(len(tr), np.nan)}
    rounds_from = (a.out / f"{MASK_BASE[a.inputs][0]}-{consumer.name}.json" if LOSS_MASK.get(a.inputs) == "fixed"
                   else a.rounds_from)
    ref = json.loads(rounds_from.read_text()) if rounds_from is not None else None
    if ref is not None and (ref["inputs"] != (MASK_BASE[a.inputs][1] if a.inputs in LOSS_MASK else a.inputs)
                            or ref["feature_list_sha256"] != fl_sha
                            or ref["consumer"] != json.loads(json.dumps(consumer.manifest(), default=str))
                            or ref["train_rows"] != len(tr)):
        raise SystemExit(f"{rounds_from} is not a fit of the same owner, inputs, features and rows")
    probes = None
    if a.inputs in PROBE_INPUTS:
        if not a.rounds_from_probes or ref is not None or a.probe_out is not None:
            raise SystemExit(f"{a.inputs} takes its rounds from --rounds-from-probes only")
        probes = rounds_from_probes(a.rounds_from_probes, a.inputs, consumer.name, fl_sha, len(tr),
                                    int((~lm).sum()), sorted(set(groups.tolist())))
    elif a.rounds_from_probes:
        raise SystemExit(f"{a.inputs} does not take --rounds-from-probes")
    for g in (sorted(set(groups.tolist())) if ref is None and probes is None else []):
        m = groups == g
        t, v = ~m & lm, m & lm                      # without a mask: ~m and m
        dtr = lgb.Dataset(tr.loc[t, fx], z[t], categorical_feature=CATEGORICAL, free_raw_data=False)
        dva = lgb.Dataset(tr.loc[v, fx], z[v], categorical_feature=CATEGORICAL, reference=dtr)
        bst = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        best.append(bst.best_iteration)
        oof["lgb"][m] = bst.predict(tr.loc[m, fx], num_iteration=bst.best_iteration)
        cm = cb.CatBoostRegressor(iterations=MAX_ROUNDS, **cparams)
        cm.fit(cb.Pool(xc_tr[t], z[t], cat_features=CATEGORICAL),
               eval_set=cb.Pool(xc_tr[v], z[v], cat_features=CATEGORICAL), use_best_model=True)
        bestc.append(cm.get_best_iteration() + 1)
        oof["cat"][m] = cm.predict(xc_tr[m])
        mean = (oof["lgb"][v] + oof["cat"][v]) / 2
        cv[str(g)] = {"rows": int(v.sum()), "best_lgb": int(best[-1]), "best_cat": int(bestc[-1]),
                      "mse_anchor": float(np.mean(z[v] ** 2)),
                      "mse_lgb": float(np.mean((z[v] - oof["lgb"][v]) ** 2)),
                      "mse_cat": float(np.mean((z[v] - oof["cat"][v]) ** 2)),
                      "mse_mean": float(np.mean((z[v] - mean) ** 2))}
        print(json.dumps({str(g): cv[str(g)]}), flush=True)
    if probes is not None:      # the probes' CV (same loop, cap 15,000) and their rounds
        cv, cv_pooled, rounds, rounds_c = probes["cv"], probes["cv_pooled"], probes["rounds"], probes["rounds_cat"]
    elif ref is None:
        cv_pooled = {k: float(np.mean([v[k] for v in cv.values()])) for k in ("mse_anchor", "mse_lgb", "mse_cat", "mse_mean")}
        rounds = max(1, int(round(float(np.mean(best)))))
        rounds_c = max(1, int(round(float(np.mean(bestc)))))
    else:           # the reference fit's CV (other seed) and rounds
        cv_pooled, rounds, rounds_c = ref["cv_pooled"], ref["rounds"], ref["rounds_cat"]
    if a.inputs in LOSS_MASK:
        bst = lgb.train(params, lgb.Dataset(tr.loc[lm, fx], z[lm], categorical_feature=CATEGORICAL), rounds)
    else:
        bst = lgb.train(params, lgb.Dataset(tr[fx], z, categorical_feature=CATEGORICAL), rounds)
    cm = cb.CatBoostRegressor(iterations=rounds_c, **{k: v for k, v in cparams.items() if not k.startswith("od_")})
    cm.fit(cb.Pool(xc_tr[lm], z[lm], cat_features=CATEGORICAL) if a.inputs in LOSS_MASK
           else cb.Pool(xc_tr, z, cat_features=CATEGORICAL))
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
              "features": fx, "feature_list_sha256": fl_sha, "zeroed_columns": sorted(ZEROED), "params": params, "cat_params": cparams,
              "rounds": rounds, "rounds_cat": rounds_c, "cv": cv, "cv_pooled": cv_pooled,
              "member_artifacts": {**rec_tr, **rec_sc},
              "features_sha256": sha(a.features), "proxy_sha256": sha(a.proxy),
              "code_sha256": sha(Path(__file__)), "lightgbm": lgb.__version__, "catboost": cb.__version__,
              "gain_top30": {k: float(v) for k, v in imp.head(30).items()},
              **({"seed": a.seed, "rounds_from": str(a.rounds_from) if ref is not None else None,
                  "rounds_from_sha256": sha(a.rounds_from) if ref is not None else None}
                 if a.seed is not None else {}),
              **({"loss_mask": LOSS_MASK[a.inputs], "loss_mask_sets": ["noac", "ordna"],
                  "loss_mask_rows": int(masked.sum()),
                  "loss_mask_rows_by_month": {str(k): int(v) for k, v in
                                              tr.loc[masked, "ctx_month"].value_counts().sort_index().items()},
                  "rounds_from": str(rounds_from) if ref is not None else None,
                  "rounds_from_sha256": sha(rounds_from) if ref is not None else None}
                 if a.inputs in LOSS_MASK else {}),
              **({"rounds_from_probes": probes["probes"], "probe_caps": probes["caps"],
                  "probe_cap_bound": probes["cap_bound"]} if probes is not None else {}),
              "seconds": round(time.monotonic() - t0, 1)}
    path = P.write(a.out, stage, consumer, owner, out, record)
    print(json.dumps({"artifact": str(path), "rounds": rounds, "rounds_cat": rounds_c, "rows": len(out),
                      "cv_pooled": cv_pooled, "seconds": record["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
