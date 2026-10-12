"""Shared data contract for PRC 2026 taxi-out work.

One definition of the raw files, the frozen forward folds, the ranking-style
masking of a validation month, the metric, and the submission writer. Every
experiment imports from here so that numbers are comparable across workers.

Rule of the game (docs/input-rule.md): every column present in the ranking
file is a legal input. A validation month is therefore the month's file with
departure BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt blanked, nothing else.
Features for a month are computed from that month's frame alone, because the
ranking file holds January and July 2026 as two isolated months.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = "TAXITIME_SEC_mvt"
ID = "MVT_ID_mvt"
TIME_COLUMNS = (
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
)
MONTHS = tuple(f"2025-{m:02d}" for m in range(1, 13))
TEAM = "youthful-giraffe"
S3_ENDPOINT = "https://s3.opensky-network.org"


def data_dir(mode: str | None = None) -> Path:
    """Raw data directory: $PRC_DATA_DIR, else repo data/raw, else the sibling checkout.

    The directory holds the twelve 2025 training files. In the ranking mode it also holds the live
    `ranking.parquet`; in the final mode it holds `final_ranking.parquet` in `final/` or at its top.
    """
    env = os.environ.get("PRC_DATA_DIR")
    candidates = [Path(env)] if env else []
    candidates += [ROOT / "data" / "raw", ROOT.parent / "prc-data-challenge-2026" / "data" / "raw"]
    if ranking_mode(mode) == "final":
        for c in candidates:
            if (c / "training_2025-01-01_2025-02-01.parquet").exists() and _final_path(c, "ranking") is not None:
                return c
        raise FileNotFoundError("no raw data directory with the training files and final_ranking.parquet "
                                "(in final/ or at its top) found; set PRC_DATA_DIR")
    for c in candidates:
        if (c / "training_2025-01-01_2025-02-01.parquet").exists() and _is_live_ranking(c / "ranking.parquet"):
            return c
    raise FileNotFoundError("no raw data directory with the live ranking file found; set PRC_DATA_DIR")


LIVE_RANKING_BYTES = 43_564_605  # bucket version of 2026-09-04, SHA-256 9db75e3a…325f86
LIVE_SUBMITTING_BYTES = 1_677_097


# ------------------------------------------------------------------ ranking modes
# exp-186-final-core (2026-10-08). The organisers' final submission covers four 2026 months in
# `final_ranking.parquet` / `final_submitting.parquet` (bucket version of 2026-10-08 11:04 UTC). The
# mode selects which pair of files is the 2026 population. "ranking" is the board pair and the
# default, so every caller that sets nothing reads what it read before. "final" is chosen with
# $PRC_RANKING_MODE=final or with mode="final"; its files are accepted only by their SHA-256.
# Facts: docs/status/2026-10-08-final-plan/PLAN.md (a); docs/DATA_CARD.md (final data section).

@dataclass(frozen=True)
class RankingSpec:
    mode: str
    ranking: str                       # file name
    submitting: str
    ranking_sha256: str
    submitting_sha256: str
    ranking_bytes: int
    submitting_bytes: int
    months: tuple[str, ...]            # UTC month of MVT_TIME_UTC_mvt, in calendar order
    rows: int                          # all rows of the ranking file
    month_departures: dict             # departures (= template rows) per month

    @property
    def departures(self) -> int:
        return sum(self.month_departures.values())

    @property
    def month_numbers(self) -> tuple[int, ...]:
        return tuple(int(m[5:]) for m in self.months)


RANKING_SPECS = {
    "ranking": RankingSpec(
        "ranking", "ranking.parquet", "submitting.parquet",
        "9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86",
        "0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b",
        LIVE_RANKING_BYTES, LIVE_SUBMITTING_BYTES,
        ("2026-01", "2026-07"), 689_534,
        {"2026-01": 152_719, "2026-07": 192_122}),
    "final": RankingSpec(
        "final", "final_ranking.parquet", "final_submitting.parquet",
        "4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f",
        "b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924",
        85_853_399, 3_003_722,
        ("2026-01", "2026-02", "2026-06", "2026-07"), 1_341_197,
        {"2026-01": 152_719, "2026-02": 144_158, "2026-06": 181_791, "2026-07": 192_122}),
}
RANKING_MODE_ENV = "PRC_RANKING_MODE"


def ranking_mode(mode: str | None = None) -> str:
    """`mode` if given, else $PRC_RANKING_MODE, else "ranking". Raises on an unknown name."""
    m = mode if mode is not None else (os.environ.get(RANKING_MODE_ENV) or "ranking")
    if m not in RANKING_SPECS:
        raise ValueError(f"unknown ranking mode {m!r}; known: {sorted(RANKING_SPECS)}")
    return m


def ranking_spec(mode: str | None = None) -> RankingSpec:
    return RANKING_SPECS[ranking_mode(mode)]


def rank_months(mode: str | None = None) -> tuple[str, ...]:
    """The 2026 months of the ranking population: ('2026-01', '2026-07') or the four final months."""
    return ranking_spec(mode).months


def rank_month_numbers(mode: str | None = None) -> tuple[int, ...]:
    """Month numbers of `rank_months`, for `month.isin([1, 7])`-style selections."""
    return ranking_spec(mode).month_numbers


def n_ranking(mode: str | None = None) -> int:
    """Departures to predict (template rows): 344,841 in the ranking mode, 670,790 in the final mode."""
    return ranking_spec(mode).departures


_VERIFIED: dict = {}


def _final_path(directory: Path, which: str) -> Path | None:
    """The final file `which` ("ranking" or "submitting") in `directory`/final/ or `directory`, by size."""
    spec = RANKING_SPECS["final"]
    name, size = ((spec.ranking, spec.ranking_bytes) if which == "ranking"
                  else (spec.submitting, spec.submitting_bytes))
    for p in (directory / "final" / name, directory / name):
        if p.exists() and p.stat().st_size == size:
            return p
    return None


def _verified(path: Path, want: str) -> Path:
    """`path` after a SHA-256 check against `want`; one hash per (path, size, mtime) and process."""
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if _VERIFIED.get(key) != want:
        got = sha256(path)
        if got != want:
            raise RuntimeError(f"{path}: SHA-256 {got} is not the expected {want}")
        _VERIFIED[key] = want
    return path


def ranking_file(mode: str | None = None) -> Path:
    """Path of the ranking file of the mode. The final file is accepted only by its SHA-256."""
    m = ranking_mode(mode)
    if m == "ranking":
        return data_dir(m) / "ranking.parquet"
    p = _final_path(data_dir(m), "ranking")
    return _verified(p, RANKING_SPECS[m].ranking_sha256)


def submitting_file(mode: str | None = None) -> Path:
    """Path of the submission template of the mode. The final file is accepted only by its SHA-256."""
    m = ranking_mode(mode)
    if m == "ranking":
        return data_dir(m) / "submitting.parquet"
    p = _final_path(data_dir(m), "submitting")
    if p is None:
        raise FileNotFoundError(f"final_submitting.parquet not found next to the final ranking file in {data_dir(m)}")
    return _verified(p, RANKING_SPECS[m].submitting_sha256)


def synthetic_mode() -> bool:
    """`PRC_SYNTHETIC=1` accepts generated stand-in files in place of the real ones.

    exp-025-cloud (2026-09-07) needs the Kaggle port proved before the
    competition files and the S3 key reach this machine, so the two file-size
    guards below can be lifted by an explicit environment variable. Off unless
    a caller sets it, and no run that sets it may put a number in the ledger:
    a synthetic file carries the real schema and none of the real values.
    """
    return os.environ.get("PRC_SYNTHETIC") == "1"


def _is_live_ranking(path: Path) -> bool:
    """True only for the re-issued 2026-09-04 ranking file, never the stale 28 MB one."""
    if synthetic_mode():
        return path.exists()
    return path.exists() and path.stat().st_size == LIVE_RANKING_BYTES


def month_file(month: str) -> Path:
    """month is 'YYYY-MM' for a 2025 training month."""
    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthBegin(1)
    return data_dir() / f"training_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"


def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    for c in TIME_COLUMNS:
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], utc=True)
        elif c in df.columns and df[c].dt.tz is None:
            df[c] = df[c].dt.tz_localize("UTC")
    return df


def load_month(month: str, columns: list[str] | None = None) -> pd.DataFrame:
    """One training month with parsed UTC timestamps and a `month` column."""
    df = pd.read_parquet(month_file(month), columns=columns)
    df = _parse_times(df)
    df["month"] = month
    return df


def load_months(months: tuple[str, ...] | list[str], columns: list[str] | None = None) -> pd.DataFrame:
    return pd.concat([load_month(m, columns) for m in months], ignore_index=True)


def load_ranking(columns: list[str] | None = None, mode: str | None = None) -> pd.DataFrame:
    """The ranking file of the mode, split into a `month` column by take-off time.

    Ranking mode: the live board file, months '2026-01' and '2026-07'. Final mode:
    `final_ranking.parquet`, months '2026-01', '2026-02', '2026-06' and '2026-07'.
    """
    m = ranking_mode(mode)
    df = pd.read_parquet(ranking_file(m), columns=columns)
    df = _parse_times(df)
    df["month"] = df["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    if m == "final":
        spec = RANKING_SPECS[m]
        assert len(df) == spec.rows, (len(df), spec.rows)
        assert tuple(sorted(df["month"].unique())) == spec.months, sorted(df["month"].unique())
    return df


def load_submitting(mode: str | None = None) -> pd.DataFrame:
    """The submission template of the mode (IDs in the order the scorer expects)."""
    m = ranking_mode(mode)
    if m == "final":
        df = pd.read_parquet(submitting_file(m))
        assert len(df) == RANKING_SPECS[m].departures and df[ID].is_unique, len(df)
        return df
    path = data_dir(m) / "submitting.parquet"
    if not synthetic_mode() and path.stat().st_size != LIVE_SUBMITTING_BYTES:
        raise RuntimeError(f"{path} is not the 2026-09-04 submitting template")
    return pd.read_parquet(path)


def mask_like_ranking(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Blank departure block time and taxi time exactly as the ranking file does.

    Returns the masked frame and the hidden labels (indexed like the frame,
    NaN on non-departure rows).
    """
    out = df.copy()
    dep = out["PHASE_mvt"] == "DEP"
    labels = out[TARGET].where(dep)
    out.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    out.loc[dep, TARGET] = np.nan
    return out, labels


@dataclass(frozen=True)
class Fold:
    name: str
    train_months: tuple[str, ...]
    valid_month: str


FOLDS = {
    "may": Fold("may", MONTHS[:4], "2025-05"),
    "july": Fold("july", MONTHS[:6], "2025-07"),
    "december": Fold("december", MONTHS[:11], "2025-12"),
    # Backward fold: December 2025 holds almost no winter weather (0.2 mm snow at
    # EDDF against 27.8 mm in January 2026), so January 2025 is the only month
    # that resembles the January ranking month. Train on Feb-Dec, validate Jan.
    "january": Fold("january", MONTHS[1:], "2025-01"),
    # Leave-one-month-out July: the submission model trains on all twelve 2025
    # months, so July 2026 is predicted with July and August 2025 in training.
    # The forward `july` fold (six months, no same-season data) is pessimistic
    # for that; this fold trains on the other eleven months and is the faithful
    # analogue of the ranking situation, as `january` is for January 2026.
    "july_loo": Fold("july_loo", MONTHS[:6] + MONTHS[7:], "2025-07"),
}
# Leave-one-out folds: the submission trains on all twelve months, so a fold that
# holds one month out of eleven is the analogue of a ranking month and a forward
# fold is not. `november_loo` was added for round-budget selection across seasons
# (retro-v1, 2026-09-05); additive, no existing fold changed.
for _m, _name in (("2025-11", "november_loo"),):
    FOLDS.setdefault(_name, Fold(_name, tuple(m for m in MONTHS if m != _m), _m))

# Nested rotation folds (exp-025-cloud, 2026-09-07, for the clean instrument of
# the nested validation rule). To score fold `g` the blend weights and the stacker are
# fitted on the OTHER folds' member OOF, and those members were trained on month
# sets that CONTAIN g's month -- so g's labels reach the weight vector that
# scores g, one step removed. `<fold>_nx_<mon>` is `<fold>` with one more month
# removed from its training set: ten months instead of eleven, the same
# validation month, the same contract. Additive; no existing fold changes and
# nothing reads these unless a caller names one.
#
# Three distinct training sets serve all six, because a set is the complement of
# a PAIR from {Jul, Dec, Jan} and predicts both of the months it drops.
_NX_SHORT = {"2025-01": "jan", "2025-07": "jul", "2025-12": "dec"}
for _base in ("july_loo", "december", "january"):
    _f = FOLDS[_base]
    for _drop, _tag in _NX_SHORT.items():
        if _drop == _f.valid_month:
            continue
        FOLDS.setdefault(f"{_base}_nx_{_tag}", Fold(
            f"{_base}_nx_{_tag}",
            tuple(m for m in _f.train_months if m != _drop),
            _f.valid_month))

#: The six nested folds, and for each the scored fold whose rotation it serves.
NESTED_FOLDS = {
    "july_loo_nx_dec": ("december", "july_loo"),
    "july_loo_nx_jan": ("january", "july_loo"),
    "december_nx_jul": ("july_loo", "december"),
    "december_nx_jan": ("january", "december"),
    "january_nx_jul": ("july_loo", "january"),
    "january_nx_dec": ("december", "january"),
}

# Label-isolated replay folds (exp-056-q12advocate, 2026-09-11). A rotation holds
# out three months: V (scored), S (correction fit) and I (inner selection). Its
# base trains on the other nine months A and predicts each held month:
# `iso_<r>_v`, `iso_<r>_s`, `iso_<r>_i`. `iso_<r>_x` is the exposure control: A
# with the same-season month M swapped for S, validated on S (in-sample on
# purpose; a diagnostic, never a candidate). Additive; no existing fold changes
# and nothing reads these unless a caller names one.
ISO_ROTATIONS = {            # r: (V, S, I, M)
    "jul": ("2025-07", "2025-08", "2025-09", "2025-06"),
    "jan": ("2025-01", "2025-02", "2025-03", "2025-12"),
    "dec": ("2025-12", "2025-11", "2025-10", "2025-09"),
}
for _r, (_v, _s, _i, _mx) in ISO_ROTATIONS.items():
    _a = tuple(m for m in MONTHS if m not in (_v, _s, _i))
    for _k, _vm in (("v", _v), ("s", _s), ("i", _i)):
        FOLDS.setdefault(f"iso_{_r}_{_k}", Fold(f"iso_{_r}_{_k}", _a, _vm))
    FOLDS.setdefault(f"iso_{_r}_x", Fold(
        f"iso_{_r}_x", tuple(sorted(set(_a) - {_mx} | {_s})), _s))

PRIMARY_FOLDS = ("july", "december")
WINTER_FOLD = "january"
RANKING_ANALOGUE_FOLDS = ("july_loo", "january")
#: 3 — exp-017 adds label_artefacts / clean_training_labels / recency_weights,
#: all additive and off unless a caller asks. Folds, masking and the submission
#: contract are unchanged from 2.
#: 4 — exp-020 adds UNREACHED_ROLLOVER, excluded_ids and rmse_clean: the
#: SCORING-side convention for rows no input can reach. Additive; `rmse` is
#: untouched and every existing number reproduces.
#: 5 — exp-017 adds `protect` to clean_training_labels, so a recipe can drop a
#: row for the heads that call it an artefact while a head that is FITTED on
#: that row keeps it. Additive and off unless a caller passes it.
#: 6 — exp-017 delivers its recipe: `keep_reachable`, the measured arm A1.
#: Additive and off unless a caller uses it.
#: 7 — exp-186 adds the ranking modes (`ranking_mode`, `rank_months`, `n_ranking`,
#: `ranking_file`, `submitting_file`, `mode=` on the loaders and the writer). The
#: default mode is "ranking", so every caller that sets nothing is unchanged.
VERSION = 7

#: Rows whose label is a date-rollover artefact **and which no input can reach**:
#: the block time sits on an earlier calendar day, the label is abnormal, and
#: every timing column (`SCHED`, `AOBT_3`, `LOBT`, `IOBT`, `EOBT_1`) is ordinary.
#: Four exist in the twelve 2025 months.
#:
#: **Why this is a scoring convention and not a cleaning trick.** `exp-007
#: rollover` measured the two january rows at 55.3 % of that whole fold's squared
#: error, and `exp-020` measures them at **116 s of january RMSE and 50 s of board
#: estimate** for every candidate in the table. The class occurs about four times
#: a year in 2.09 M departures, so its Poisson mean in the 344,841 ranking
#: departures is **0.66** and the most likely number of them there is **zero**
#: (P = 0.52). A fold number dominated by two such rows describes 2025's
#: accidents, not skill on 2026. Report both, always: `rmse` is what the board
#: will do to us if the ranking months contain the same accidents, `rmse_clean`
#: is what our modelling is worth.
#:
#: **This is NOT `label_artefacts()["rollover_undetectable"]`, and the difference
#: is measured** (`experiments/exp-020-cleanboard/analysis/`). That classifier is
#: a TRAINING-side tool and asks only whether `AOBT_3_flt` carries the day shift;
#: of its 14 "undetectable" rows, **12 are reached by `MVT − SCHED`** above
#: 10,000 s, which is the path `prc.tail_model` 2.2.0's noac head v4 actually
#: routes them through. Its `|y − 86,400| < 3600` window also misses two
#: rollover-class rows at 58,206 s and 31,268 s. Neither point is a defect for its
#: own purpose — dropping a reachable rollover from TRAINING costs little — but
#: the name does not transfer to scoring, so this list is separate.
# Private checkouts may supply the clean-score exclusions. They contain row-level
# organiser data and are intentionally absent from the public source release.
try:
    from prc._private_clean_score_rows import UNREACHED_ROLLOVER
except ModuleNotFoundError as exc:
    if exc.name != "prc._private_clean_score_rows":
        raise
    UNREACHED_ROLLOVER = {}


def excluded_ids(fold: "Fold | str") -> set:
    """`MVT_ID_mvt` values a clean score excludes for this fold's validation month."""
    if isinstance(fold, str) and fold in FOLDS:
        month = FOLDS[fold].valid_month
    elif isinstance(fold, str):
        month = fold
    else:
        month = fold.valid_month
    return {r["id"] for r in UNREACHED_ROLLOVER.get(month, ())}


def rmse_clean(y_true, y_pred, mvt_ids, fold: "Fold | str") -> float:
    """RMSE with this fold's unreached-rollover rows removed.

    `mvt_ids` must align with `y_true` and `y_pred`. Raises if none of the
    fold's excluded ids is present, because that means the caller passed ids
    from a different fold or a different id space, and a silent no-op here
    would report a clean number that is not clean.
    """
    ex = excluded_ids(fold)
    ids = np.asarray(mvt_ids, dtype="float64")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(ids) != len(y_true) or len(ids) != len(y_pred):
        raise ValueError("mvt_ids, y_true and y_pred must be the same length")
    if ex:
        keep = ~np.isin(ids, list(ex))
        if keep.all():
            raise ValueError(
                f"fold {fold!r} excludes {sorted(int(i) for i in ex)} and none is in "
                f"the ids passed; check the fold or the id space rather than "
                f"reporting this as a clean score")
    else:
        keep = np.ones(len(ids), dtype=bool)
    m = keep & ~np.isnan(y_true)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def rmse_both(y_true, y_pred, mvt_ids, fold: "Fold | str") -> dict:
    """`{"rmse": ..., "rmse_clean": ..., "n_excluded": ...}` -- report both."""
    ex = excluded_ids(fold)
    ids = np.asarray(mvt_ids, dtype="float64")
    return {"rmse": rmse(y_true, y_pred),
            "rmse_clean": rmse_clean(y_true, y_pred, ids, fold),
            "n_excluded": int(np.isin(ids, list(ex)).sum()) if ex else 0}


def fold_frames(fold: Fold | str, columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(train, valid_masked, y_valid) for one fold.

    train: the training months with full labels.
    valid_masked: the validation month's frame with departure block time and
    taxi time blanked. Compute validation features from this frame only.
    y_valid: hidden taxi time for the validation departures, indexed like
    valid_masked (NaN on arrivals). Score on rows where y_valid.notna().
    """
    if isinstance(fold, str):
        fold = FOLDS[fold]
    train = load_months(fold.train_months, columns)
    valid_full = load_month(fold.valid_month, columns)
    valid_masked, y_valid = mask_like_ranking(valid_full)
    return train, valid_masked, y_valid


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~np.isnan(y_true)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_submission(pred: pd.Series, path: Path | str, mode: str | None = None) -> str:
    """Write a valid submission file. `pred` is indexed by MVT_ID_mvt.

    Checks: exact ID set of the template of the mode, no missing or non-finite
    values, float64 dtype, template row order. Returns the SHA-256 of the file.
    """
    template = load_submitting(mode)
    ids = template[ID]
    pred = pd.Series(pred).astype("float64")
    pred.index = pred.index.astype("float64")
    missing = ids[~ids.isin(pred.index)]
    extra = pred.index[~pd.Index(pred.index).isin(ids)]
    if len(missing) or len(extra):
        raise ValueError(f"id mismatch: {len(missing)} missing, {len(extra)} extra")
    values = pred.reindex(ids.values).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("non-finite predictions")
    out = pd.DataFrame({ID: ids.astype("float64").to_numpy(), TARGET: values})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return sha256(path)

# --------------------------------------------------------------- label hygiene
#: exp-017. Bump when the classification or the default policy changes.
LABEL_HYGIENE_VERSION = "1.2.0"   # 1.1.0 `protect`; 1.2.0 `keep_reachable` (arm A1)
_DAY = 86400.0


def label_artefacts(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify TRAINING rows whose target is an artefact, not a taxi time.

    One boolean column per class, indexed like `frame`'s departure rows. Reads
    the label, so this is a training-side tool only: never call it on a
    validation or ranking frame, and never let it touch the scored population.

    Measured over the twelve 2025 months at the ten ranking airports
    (2,085,047 departures, `experiments/exp-017-labels/catalogue.py`):

    =========================  =====  ===========================
    class                       rows  share of training SSE
    =========================  =====  ===========================
    rollover_undetectable         14  **16.4 %**
    rollover_detectable            1  1.2 %
    nonpositive                  388  0.05 %
    duplicate                     40  0.12 %
    =========================  =====  ===========================

    Fifteen rows carry **17.6 %** of the squared error an L2 fit sees. That is
    the whole of this lane: the other 428 rows are rounding.

    `rollover_detectable` means `AOBT_3_flt` carries the same day shift, so the
    row can be corrected rather than dropped. Only **1 of 15** qualifies -- the
    other 14 are no-NM rows with no NM column to carry a shift -- so correction
    is a nearly empty treatment and dropping is the operative one.
    """
    dep = frame[frame["PHASE_mvt"] == "DEP"]
    y = dep[TARGET].astype(float)
    mvt = dep["MVT_TIME_UTC_mvt"]
    roll = (y - _DAY).abs() < 3600
    aobt_shift = ((mvt - dep["AOBT_3_flt"]).dt.total_seconds() - _DAY).abs() < 3600
    out = pd.DataFrame(index=dep.index)
    out["rollover_detectable"] = roll & aobt_shift.fillna(False)
    out["rollover_undetectable"] = roll & ~aobt_shift.fillna(False)
    out["nonpositive"] = y <= 0
    key = (dep["ADEP_mvt"].astype(str) + "|" + dep["STAND_mvt"].astype(str) + "|"
           + dep["FLIGHT_mvt"].astype(str) + "|" + mvt.dt.floor("min").astype(str))
    out["duplicate"] = key.duplicated(keep=False)
    out["any"] = out.any(axis=1)
    return out


def keep_reachable(frame: pd.DataFrame) -> pd.Series:
    """exp-017 arm A1: the training rows to KEEP. `False` only for the four
    rows no input can reach. Off unless a caller uses it.

    Returns a boolean mask on `frame`'s **own index**, so it composes whatever
    the frame holds::

        train = train[D.keep_reachable(train)]

    The four rows are `UNREACHED_ROLLOVER`: the label is a date-rollover
    artefact and every timing column is ordinary, so no model can reach them and
    an L2 fit spends real capacity trying. Dropping them from bulk training is
    the whole of exp-017 — measured **-1.56 s on july_loo** against a control
    seed spread of 0.09, and -3.52 s on july, from **four rows**
    (`experiments/exp-017-labels/work/arms.csv`, three seeds).

    Two warnings the measurement carries:

    * **Training side only.** These rows stay in the scored population; that is
      what `rmse_clean` is for, and the two questions are separate.
    * **Read `rmse_clean` on january.** January validates 2025-01, which holds
      two of the four, so its raw `rmse` moves +0.49 s on this recipe while
      `rmse_clean` moves -0.27 s. The raw number is dominated by the rows
      themselves, not by what the model learned.

    Prefer this over `clean_training_labels(rollover=...)` for bulk heads: the
    window classifier there drops 2 rows and misses two rollover-class rows at
    58,206 s and 31,268 s (exp-020 census), and it must never reach the noac
    band head, which is FITTED on 11 rollover rows.
    """
    ids = {r["id"] for rows in UNREACHED_ROLLOVER.values() for r in rows}
    return ~frame["MVT_ID_mvt"].isin(ids)


def clean_training_labels(frame: pd.DataFrame, *, rollover: str = "drop",
                          nonpositive: str = "drop", duplicates: str = "drop",
                          floor: float | None = None,
                          protect: pd.Series | None = None):
    """Apply the exp-017 recipe to a TRAINING frame. Off by default at the caller.

    Returns `(keep, y)`: a boolean mask over `frame`'s departure rows and the
    training target to use for the kept rows. The caller decides whether to
    subset or to zero-weight; nothing here touches a scored row.

    Policies -- `rollover`: "drop", "correct" (subtract a day where detectable,
    drop the rest) or "keep"; `nonpositive` and `duplicates`: "drop" or "keep";
    `floor`: winsorise the training target up to this value instead of dropping
    (retro-v1's `train_clip_lo` does the same thing and has never been enabled).

    `protect` is a boolean mask of rows the recipe must never drop. It exists
    because the classes are not artefacts for every head: 12 of the 14
    undetectable rollovers are `noac` rows, where a day-long block gap is the
    ordinary structure the noac heads are fitted on, and 11 of them are the
    entire training signal of the high-delay band mixture. Dropping a row that
    another head needs is how exp-017's first screen measured the wrong thing
    (`experiments/exp-017-labels/SPEC.md`).
    """
    a = label_artefacts(frame)
    dep = frame[frame["PHASE_mvt"] == "DEP"]
    y = dep[TARGET].astype(float).copy()
    keep = pd.Series(True, index=dep.index)
    if rollover == "correct":
        y.loc[a["rollover_detectable"]] -= _DAY
        keep &= ~a["rollover_undetectable"]
    elif rollover == "drop":
        keep &= ~(a["rollover_detectable"] | a["rollover_undetectable"])
    if nonpositive == "drop":
        keep &= ~a["nonpositive"]
    if duplicates == "drop":
        keep &= ~a["duplicate"]
    if floor is not None:
        y = y.clip(lower=floor)
        keep |= a["nonpositive"]          # winsorised rather than dropped
    if protect is not None:
        keep |= protect.reindex(keep.index).fillna(False).astype(bool)
    return keep, y


def recency_weights(frame: pd.DataFrame, half_life_days: float = 180.0) -> pd.Series:
    """Exponential recency weight over a training frame's departure rows.

    `w = 0.5 ** (age_in_days / half_life_days)`, age measured from the frame's
    latest take-off. retro-v1 has `gap`, `no_nm` and a winter multiplier but
    **no temporal scheme**, so this is new rather than a re-implementation.
    """
    dep = frame[frame["PHASE_mvt"] == "DEP"]
    age = (dep["MVT_TIME_UTC_mvt"].max() - dep["MVT_TIME_UTC_mvt"]).dt.total_seconds() / 86400.0
    return pd.Series(0.5 ** (age / float(half_life_days)), index=dep.index)
