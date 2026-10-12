"""exp-097 PK-C motion transition: the registered library.

One hypothesis: on a v22-complement departure, the aircraft's observed MOTION-STATE
TRANSITION in the exp-045 ground summaries -- it was heard stationary at a stand and
then heard to move, or it was heard moving and then heard to stop -- carries a physical
off-block / taxi-out signal that v22 throws away.

v22 owns a row only when its matched ground segment carries the 250 m displacement
onset (`arms.SERVED_CLASS` = `obs_has_clock & obs_taxi.notna() & ~noac`).  A row whose
segment WAS identified by the strict take-off rule but whose displacement onset is
missing is `reach_class == "matched_no_clock"`: v22 falls back to the base, which has
no surface observation at all, while the block already carries the segment's motion
state -- `obs_taxi_motion` (MVT minus the first sustained motion), `stand_stop_s` (the
stationary run that preceded it, i.e. the moving->stationary transition) and
`turnaround_anchored`.  Those three columns are the hypothesis.

Nothing here re-derives identity.  `experiments/exp-046-surfacefit/code/block.py` is the
repository's one matcher and it is imported, never copied and never modified: the
segment a row is matched to is exactly the segment v22 matched it to.

**No module in this file has a generic name.**  `imports.load` puts other experiments'
code directories at the FRONT of `sys.path`, and a generic name (`base`, `serve`,
`census`, `vectors`) resolves to another lane's file.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-097-pkc-motion-transition/code/test_pkcmotion.py
"""
from __future__ import annotations

import contextlib
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]

# ------------------------------------------------------------------ registered constants

#: the three v22-equivalent out-of-fold panels and the 2025 month each one is.
FOLDS = ("july_loo", "december", "january")
FOLD_MONTH = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
PANEL = {f: f"experiments/exp-050-nmblind/work/panel-v22-{f}.parquet" for f in FOLDS}
#: the two ranking months and v22's own ownership vector for each.
RANK_MONTHS = ("2026-01", "2026-07")
CLASS_FILE = {m: f"artifacts/exp-046/class/{m}-v22.parquet" for m in RANK_MONTHS}
UPLOADED_V22 = "submissions/uploaded/youthful-giraffe_v22.parquet"

#: board-share currency.  The ranking file holds 192,122 July and 152,719 January
#: departures, 344,841 together; December is reported beside the currency, never in it.
BOARD_ROWS = {"july_loo": 192122, "january": 152719}
BOARD_TOTAL = 344841
BOARD_SHARE = {f: BOARD_ROWS[f] / BOARD_TOTAL for f in BOARD_ROWS}
V22_BOARD = 245.0207               # docs/board, youthful-giraffe v22
BOARD_TARGET = 243.5805            # vigorous-whistle, the recorded rule target
UPLOAD_GATE_S = 0.10               # the standing board-share upload gate

#: the physical taxi window a motion clock may claim.  Both ends are v22's own
#: (`block.WIDE_MIN_TAXI`, `block.WIDE_MAX_TAXI`); a served prediction is clipped here.
TAXI_LO = 60.0
TAXI_HI = 7200.0

#: ridge constant (squared seconds), fixed before any label read.  Standardised design,
#: intercept unpenalised.
RIDGE = 10.0
#: bootstrap
BOOT = 2000
SEED = 97026
FOLD_OFFSET = {"july_loo": 0, "december": 1, "january": 2}
ARM_OFFSET = {"motion": 0, "extent": 100, "placebo": 200}

#: arm M -- the motion-state transition.  `obs_taxi_motion` is MVT minus the first
#: sustained motion after the stand dwell; `stand_stop_s` is the length of that dwell,
#: which is the moving->stationary transition the aircraft made before it moved;
#: `turnaround_anchored` says the dwell was a stand turnaround (>= 600 s).
MOTION_COLS = ("obs_taxi_motion", "stand_stop_log", "turnaround_anchored")
#: arm E -- the falsification control.  Segment EXTENT and identity quality only: the
#: same rows, the same fit, no motion-state variable.  If E scores like M then being
#: matched carries the signal and the motion transition does not.
EXTENT_COLS = ("seg_span_s", "n_ground_points", "dt_takeoff", "stand_dist_km")

#: the label-free support condition that lets the label-bearing phase run at all.
#: The registered gate set, in order.  The producer writes exactly these keys and the
#: watcher's validator permits exactly these keys: one schema, one definition (SPEC §7).
GATES = ("G1_board_gain_ge_0.10", "G2_nonnegative_every_fold",
         "G3_positive_lower_bound_july_january", "G4_beats_extent_control",
         "G5_beats_placebo_control")

MIN_SUPPORT_TOTAL = 300
MIN_SUPPORT_PER_FOLD = 60
#: the candidate phase needs this many changed ranking rows.
MIN_RANKING_CHANGED = 100

#: every column this experiment may project, per source.  A reader compares what it
#: asked for and what came back with these lists and fails closed.
PANEL_COLS_LABEL_FREE = ("MVT_ID_mvt", "airport", "day", "fold", "p_v22",
                         "obs_has_clock", "noac", "nm_missing")
PANEL_COLS_LABEL = ("MVT_ID_mvt", "y_true")
CLASS_COLS = ("MVT_ID_mvt", "obs_has_clock", "served_by_exp_046", "noac")
#: The DEPARTURE projection, for a training month and for the ranking file alike:
#: `block.COHORT_COLS` plus the two columns the turnaround (stand-chain) pass reads off
#: the cohort (`block.py:799`).  Every one is a ranking-file input and **neither
#: forbidden column is in it**, so a departure frame in this experiment cannot carry a
#: block time or a target at all (Sol round 1, blocking 1).  This replaces
#: `rowfeat.departures`, which loads the whole month and materialises the real target
#: vector inside `prc.data.mask_like_ranking` before blanking it.
DEP_COLS = ("MVT_ID_mvt", "PHASE_mvt", "FLIGHT_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
            "CALLSIGN_flt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt",
            "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt")
RANKING_COLS = DEP_COLS
#: The ARRIVAL projection, exactly the columns `block.arrival_keys` and
#: `block.arrival_legs` read.  `BLOCK_TIME_UTC_mvt` is in it because an ARRIVAL's
#: in-block is the stand chain's clock; it is an input on 100 % of ARR rows and
#: `mask_like_ranking` never blanks it.  The loader restricts the frame to ARR rows
#: before it returns, so no departure block time survives into this experiment, and
#: `TAXITIME_SEC_mvt` is never projected from a training month in any phase.
ARR_COLS = ("PHASE_mvt", "ADES_mvt", "CALLSIGN_flt", "STAND_mvt",
            "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt", "BLOCK_TIME_UTC_mvt")
#: The prefix map's projection: three columns of the nine non-fold 2025 months.
PREFIX_COLS = ("PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt")
#: never projected anywhere, in any phase, under any name.
FORBIDDEN = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")
FORBIDDEN_SUBSTRINGS = ("target", "label", "truth", "residual")


def refuse_columns(asked, allowed, where: str) -> None:
    """Fail closed on any projection outside the allow-list."""
    extra = [c for c in asked if c not in set(allowed)]
    if extra:
        raise SystemExit(f"{where}: refused columns outside the registered projection: {extra}")
    for c in asked:
        if c in FORBIDDEN:
            raise SystemExit(f"{where}: refused forbidden column {c}")
        low = str(c).lower()
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in low and c not in ("y_true",):
                raise SystemExit(f"{where}: refused column {c} (contains {bad!r})")


def assert_no_forbidden(df, where: str, allow: tuple[str, ...] = ()) -> None:
    """No frame this experiment holds may carry a forbidden column.

    `BLOCK_TIME_UTC_mvt` is forbidden because it is half the target: the label is
    `MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` **on a departure row**.  On an ARR row the
    same column is the arrival's in-block, which is not any departure's label and which
    the turnaround / stand-chain pass needs.  `_arrivals` therefore passes it in `allow`,
    and only after the frame has been restricted to ARR.  `TAXITIME_SEC_mvt` is the
    target itself and no caller may ever allow it.
    """
    bad = set(allow) & {"TAXITIME_SEC_mvt"}
    if bad:
        raise SystemExit(f"{where}: the target may never be allowed: {sorted(bad)}")
    for c in df.columns:
        if c in FORBIDDEN and c not in allow:
            raise SystemExit(f"{where}: frame carries the forbidden column {c}")


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_ids(ids: np.ndarray, where: str) -> np.ndarray:
    """IDs may be stored as float or int; every value must be an exact int64.

    Exp-096 review finding: a float64 conversion before the exactness check accepts
    both `2**53` boundaries.  The check happens on the stored values, and both signed
    boundaries are rejected before any conversion.
    """
    a = np.asarray(ids)
    if a.dtype.kind == "f":
        if not np.isfinite(a).all():
            raise SystemExit(f"{where}: non-finite movement id")
        if not np.all(a == np.floor(a)):
            raise SystemExit(f"{where}: non-integral movement id")
        if np.any(np.abs(a) >= 2.0 ** 53):
            raise SystemExit(f"{where}: movement id outside the exact binary64 range")
        out = a.astype(np.int64)
    elif a.dtype.kind == "i":
        # the SAME open boundary as the float branch, BEFORE any conversion (Sol round 1,
        # blocking 7): an int64 2**53 used to pass while its float64 twin was refused.
        if np.any(np.abs(a.astype(object)) >= 2 ** 53):
            raise SystemExit(f"{where}: movement id outside the exact binary64 range")
        out = a.astype(np.int64)
    elif a.dtype.kind == "u":
        # an unsigned value at or above 2**63 wraps NEGATIVE under `astype(int64)`; the
        # open 2**53 bound is checked in Python integers, where nothing wraps.
        if np.any(a.astype(object) >= 2 ** 53):
            raise SystemExit(f"{where}: movement id outside the exact binary64 range")
        out = a.astype(np.int64)
    else:
        raise SystemExit(f"{where}: movement id of unusable dtype {a.dtype}")
    if len(np.unique(out)) != len(out):
        raise SystemExit(f"{where}: duplicate movement id")
    return out


# --------------------------------------------------------------------- the matcher side

def prc_data():
    """`prc.data`, resolved from THIS repository and nowhere else.

    `import prc` succeeds only when the repository root is on `sys.path`.  On the Mac it
    was there by accident — `PYTHONPATH` begins with an empty entry, which Python reads as
    the working directory — and on the workstation it was not there at all, so the same
    commit imported on one machine and raised `ModuleNotFoundError` on the other.  Worse,
    an environment that put a DIFFERENT checkout first would have loaded code the manifest
    does not bind.  The root is therefore placed at the FRONT of `sys.path` here, and the
    module's own file is checked to be inside it.
    """
    root = str(ROOT)
    if sys.path[:1] != [root]:
        while root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    import prc.data as D                                               # noqa: PLC0415
    got = Path(D.__file__).resolve()
    if ROOT not in got.parents:
        raise SystemExit(f"prc.data loaded from {got}, outside the bound repository {ROOT}")
    return D


def _lane():
    """exp-046's code directory on `sys.path`, once, and its modules."""
    code = str(ROOT / "experiments/exp-046-surfacefit/code")
    if code not in sys.path:
        sys.path.insert(0, code)
    import arms as A                                                   # noqa: PLC0415
    import block as B                                                  # noqa: PLC0415
    return A, B


def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    """`prc.data._parse_times` on the projected frame, without loading the month."""
    D = prc_data()
    for c in D.TIME_COLUMNS:
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], utc=True)
        elif c in df.columns and df[c].dt.tz is None:
            df[c] = df[c].dt.tz_localize("UTC")
    return df


def _departures(path, month: str | None = None) -> pd.DataFrame:
    """The DEP rows of one file under the registered label-free projection."""
    d = pd.read_parquet(path, columns=list(DEP_COLS))
    if list(d.columns) != list(DEP_COLS):
        raise SystemExit(f"{path}: returned columns {list(d.columns)} != {list(DEP_COLS)}")
    assert_no_forbidden(d, f"departures {path}")
    d = _parse_times(d)
    d = d[d["PHASE_mvt"] == "DEP"].copy()
    if month is not None:
        d = d[d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").eq(month)].copy()
    return d.reset_index(drop=True)


def _arrivals(path, month: str | None = None) -> pd.DataFrame:
    """The ARR rows of one file under the registered arrival projection.

    The frame is restricted to ARR before it is returned, so no DEPARTURE in-block ever
    reaches this experiment, and the target column is never projected at all.
    """
    a = pd.read_parquet(path, columns=list(ARR_COLS))
    if list(a.columns) != list(ARR_COLS):
        raise SystemExit(f"{path}: returned columns {list(a.columns)} != {list(ARR_COLS)}")
    a = _parse_times(a)
    a = a[a["PHASE_mvt"] == "ARR"].copy()
    # The allowance is granted only here, and only to a frame already proven to hold no
    # departure row, so no half of any departure's label can survive this function.
    assert_no_forbidden(a, f"arrivals {path}", allow=("BLOCK_TIME_UTC_mvt",))
    if month is not None:
        a = a[a["BLOCK_TIME_UTC_mvt"].dt.strftime("%Y-%m").eq(month)
              | a["BLOCK_TIME_UTC_mvt"].isna()].copy()
    if (a["PHASE_mvt"] != "ARR").any():
        raise SystemExit(f"{path}: a non-arrival row survived the arrival projection")
    return a.reset_index(drop=True)


def _prefix_frame() -> pd.DataFrame:
    """The prefix map's training frame: the nine non-fold 2025 months, DEP rows.

    `arms.prefix_months_for("")` is v22's own choice of months and is used unchanged, so
    the map is the same function on the fold side and the serving side.
    """
    A, _ = _lane()
    D = prc_data()
    parts = []
    for m in A.prefix_months_for(""):
        t = pd.read_parquet(D.month_file(m), columns=list(PREFIX_COLS))
        assert_no_forbidden(t, f"prefix month {m}")
        parts.append(t[t["PHASE_mvt"] == "DEP"].drop(columns="PHASE_mvt"))
    return pd.concat(parts, ignore_index=True)


def _build(dep: pd.DataFrame, arr: pd.DataFrame, tr: pd.DataFrame,
           month: str, pin: bool) -> pd.DataFrame:
    """v22's own `block.build`, on frames this experiment projected itself.

    Nothing is cached and nothing is written outside exp-097: `arms.month_block` returns
    a stored block when one exists and copies its record into exp-046's `analysis/`
    (Sol round 1, blocking 6).  The builder, the prefix map and the serving rule are
    v22's; only the frames' provenance is exp-097's.
    """
    A, B = _lane()
    rule, _ = A.serving_rule()
    ctx = B.pinned(B.pin_for_month(month)) if pin else contextlib.nullcontext()
    with ctx:
        return B.build(dep, tr, require_rule=rule, arrivals=arr)


def month_block(month: str) -> pd.DataFrame:
    """v22's feature block for a 2025 fold month, rebuilt from label-free projections.

    The stored block parquet is float32 and a fitted read must not use it; the cached
    rebuild is not used either, because its bytes are not bound by this experiment's
    manifest.
    """
    D = prc_data()
    f = D.month_file(month)
    return _build(_departures(f), _arrivals(f), _prefix_frame(), month, pin=True)


def ranking_month_block(month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ranking file's departures for one month and v22's block over them.

    Built per month, as `serve.ranking_block` builds it: passing both ranking months to
    one `build()` call makes `_arrival_side` take a January segment as a July segment's
    predecessor.  Ownership is never taken from here; it is read from v22's own class
    vector.
    """
    D = prc_data()
    path = D.data_dir() / "ranking.parquet"
    assert_ranking_departures_blank(path)
    dep = _departures(path, month)
    arr = _arrivals(path, month)
    if not len(dep):
        raise SystemExit(f"the ranking file carries no departure in {month}")
    blk = _build(dep, arr, _prefix_frame(), month, pin=False)
    return dep, blk


def assert_ranking_departures_blank(path) -> dict:
    """The ranking file's DEPARTURE rows carry neither forbidden column.

    This is the one place either name is projected, it is the RANKING file and never a
    training month, and the values leave the function only as two counts.
    """
    chk = pd.read_parquet(path, columns=["PHASE_mvt", *FORBIDDEN])
    dep = chk["PHASE_mvt"] == "DEP"
    counts = {c: int(chk.loc[dep, c].notna().sum()) for c in FORBIDDEN}
    del chk
    for c, n in counts.items():
        if n:
            raise SystemExit(f"the ranking file carries a departure {c} on {n} rows")
    return counts


# ----------------------------------------------------------------- population and signal

def _bool_fill(s: pd.Series) -> np.ndarray:
    """A left join can leave a bool column as object with NaN; False is the safe fill."""
    a = s.to_numpy(dtype=object, copy=True)
    out = np.zeros(len(a), dtype=bool)
    for i, v in enumerate(a):
        out[i] = bool(v) if isinstance(v, (bool, np.bool_)) else False
    return out


def add_signal(d: pd.DataFrame) -> pd.DataFrame:
    """The derived motion columns.  Pure function of the block; reads no label."""
    d = d.copy()
    d["stand_stop_log"] = np.log1p(np.clip(d["stand_stop_s"].to_numpy("float64"), 0.0, None))
    d["obs_matched"] = _bool_fill(d["obs_matched"])
    d["block_has_clock"] = _bool_fill(d["obs_has_clock_block"])
    d["reach_class"] = d["reach_class"].fillna("unmatched")
    return d


def support_mask(d: pd.DataFrame) -> np.ndarray:
    """The registered support class S.

    A row is supported when NEITHER ownership vector claims it (v22's own
    `obs_has_clock` and the rebuilt block's agree that there is no 250 m clock), the
    `noac` hold does not hold it, the matcher identified its ground segment, and that
    segment carries a motion-state transition whose implied taxi is inside the
    registered physical window.
    """
    own = d["own_has_clock"].to_numpy(bool)
    blk = d["block_has_clock"].to_numpy(bool)
    noac = d["noac_bool"].to_numpy(bool)
    matched = d["obs_matched"].to_numpy(bool)
    tm = d["obs_taxi_motion"].to_numpy("float64")
    ok = np.isfinite(tm) & (tm >= TAXI_LO) & (tm <= TAXI_HI)
    return (~own) & (~blk) & (~noac) & matched & ok


def complement_mask(d: pd.DataFrame) -> np.ndarray:
    """v22's ownership complement, the corrected four-case rule of exp-084.

    `not (obs_has_clock and not noac)`: a row with no clock is in it, and a `noac` row
    is in it whether or not it has a clock.
    """
    return ~(d["own_has_clock"].to_numpy(bool) & ~d["noac_bool"].to_numpy(bool))


def grade_counts(d: pd.DataFrame, sel: np.ndarray) -> dict:
    """Support by motion-transition grade, for the label-free screen."""
    s = d[sel]
    ss = s["stand_stop_s"].to_numpy("float64")
    ta = s["turnaround_anchored"].to_numpy("float64")
    return {
        "rows": int(sel.sum()),
        "no_observed_dwell": int(np.sum(~(ss > 0))),
        "dwell_gt_0s": int(np.sum(ss > 0)),
        "dwell_ge_60s": int(np.sum(ss >= 60)),
        "dwell_ge_600s": int(np.sum(ss >= 600)),
        "turnaround_anchored": int(np.sum(ta > 0)),
    }


def quantiles(x, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)) -> dict:
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    if not len(a):
        return {str(q): None for q in qs}
    return {str(q): float(np.quantile(a, q)) for q in qs}


# ----------------------------------------------------------------------------- the fit

def design(d: pd.DataFrame, cols, medians: dict | None = None) -> tuple[np.ndarray, dict]:
    """`[1, p_v22, *cols]` with non-finite entries replaced by the training median.

    `medians` is the training frame's; passing it is what keeps the target fold out of
    its own design.  The returned dict is the medians actually used.
    """
    n = len(d)
    used = {}
    parts = [np.ones(n), d["p_v22"].to_numpy("float64")]
    for c in cols:
        v = d[c].to_numpy("float64").copy()
        bad = ~np.isfinite(v)
        if medians is not None:
            m = medians[c]
        else:
            good = v[~bad]
            m = float(np.median(good)) if len(good) else 0.0
        used[c] = float(m)
        v[bad] = m
        parts.append(v)
    X = np.column_stack(parts)
    if not np.isfinite(X).all():
        raise SystemExit("design matrix is not finite")
    return X, used


def ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Standardised ridge with an unpenalised intercept.  Column 0 is the intercept."""
    mu = X[:, 1:].mean(axis=0)
    sd = X[:, 1:].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X[:, 1:] - mu) / sd])
    P = np.eye(Z.shape[1]) * ridge
    P[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
    # back to the raw scale so a single vector applies to any frame
    raw = np.zeros(X.shape[1])
    raw[1:] = beta[1:] / sd
    raw[0] = beta[0] - float(np.sum(beta[1:] * mu / sd))
    return raw


def apply_fit(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.clip(X @ beta, TAXI_LO, TAXI_HI)


def permute_within(d: pd.DataFrame, cols, seed: int) -> pd.DataFrame:
    """Placebo: the motion columns move to another row of the same airport and day.

    Deterministic: each group is sorted by its integer movement id first, so no frame
    ordering can change the permutation.
    """
    out = d.copy()
    ids = out["id64"].to_numpy(np.int64)
    key = out["airport"].astype(str).to_numpy() + "|" + out["day"].astype(str).to_numpy()
    order = np.lexsort((ids, key))
    rng = np.random.default_rng(seed)
    src = np.arange(len(out))
    ordered = order.copy()
    start = 0
    for g in range(1, len(ordered) + 1):
        if g == len(ordered) or key[ordered[g]] != key[ordered[start]]:
            block = ordered[start:g]
            src[block] = block[rng.permutation(len(block))]
            start = g
    for c in cols:
        out[c] = out[c].to_numpy()[src]
    return out


# ------------------------------------------------------------------------- the metrics

def fold_scores(y: np.ndarray, p_base: np.ndarray, p_new: np.ndarray) -> dict:
    e0 = y - p_base
    e1 = y - p_new
    return {
        "rows": int(len(y)),
        "mse_base": float(np.mean(e0 ** 2)),
        "mse_new": float(np.mean(e1 ** 2)),
        "rmse_base": float(np.sqrt(np.mean(e0 ** 2))),
        "rmse_new": float(np.sqrt(np.mean(e1 ** 2))),
        "rmse_gain": float(np.sqrt(np.mean(e0 ** 2)) - np.sqrt(np.mean(e1 ** 2))),
        "sse_gain": float(np.sum(e0 ** 2) - np.sum(e1 ** 2)),
        "changed_rows": int(np.sum(p_new != p_base)),
    }


def board_rmse(mse_by_fold: dict) -> float:
    return float(np.sqrt(sum(BOARD_SHARE[f] * mse_by_fold[f] for f in BOARD_SHARE)))


def day_bootstrap_gain(y, p0, p1, days, seed: int, boot: int = BOOT) -> dict:
    """RMSE gain with a UTC-day cluster bootstrap on one fold."""
    days = np.asarray(days)
    uniq, inv = np.unique(days, return_inverse=True)
    idx_by_day = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(boot)
    e0 = (y - p0) ** 2
    e1 = (y - p1) ** 2
    for b in range(boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by_day[k] for k in pick])
        out[b] = float(np.sqrt(e0[sel].mean()) - np.sqrt(e1[sel].mean()))
    lo, hi = np.quantile(out, [0.025, 0.975])
    return {"point": float(np.sqrt(e0.mean()) - np.sqrt(e1.mean())),
            "ci95": [float(lo), float(hi)], "draws": int(boot)}


def board_bootstrap(per_fold: dict, seed: int, boot: int = BOOT) -> dict:
    """Board-share RMSE gain with July and January days resampled together.

    `per_fold[f]` carries `e0`, `e1` (squared errors) and `inv`/`n_days` for the fold.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(boot)
    prep = {}
    for f in BOARD_SHARE:
        d = per_fold[f]
        uniq, inv = np.unique(d["days"], return_inverse=True)
        prep[f] = ([np.flatnonzero(inv == k) for k in range(len(uniq))],
                   d["e0"], d["e1"], len(uniq))
    for b in range(boot):
        m0 = {}
        m1 = {}
        for f, (idx, e0, e1, nd) in prep.items():
            pick = rng.integers(0, nd, nd)
            sel = np.concatenate([idx[k] for k in pick])
            m0[f] = float(e0[sel].mean())
            m1[f] = float(e1[sel].mean())
        out[b] = board_rmse(m0) - board_rmse(m1)
    lo, hi = np.quantile(out, [0.025, 0.975])
    return {"ci95": [float(lo), float(hi)], "draws": int(boot)}
