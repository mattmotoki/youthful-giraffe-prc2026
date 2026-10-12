"""Per-regime blending of member predictions into one taxi-out prediction.

A blend here is three things: a **regime** label computable from the ranking
file's own columns, one **weight vector per regime**, and a set of **member
prediction files**.  Given those, producing a submission candidate is a config
change, which is the point of the module — `build_candidate` takes a JSON
config and writes the parquet, so a later candidate needs no new code.

The four regimes are the populations that behave differently in this task
(exp-001 RESULTS, exp-002 RESULTS), and every one of them is defined from
inputs, never from the label:

===========  ==========================================================
``noac``     ``AIRCRAFT_TYPE_mvt`` is null; the recorded block time is
             often the schedule rather than an actual off-block.
``nonm``     no Network Manager join (``FLIGHT_ID_mvt`` null) and an
             aircraft type is present.
``nm_late``  NM join present and ``AOBT_3 − LOBT`` above ``LATE_CUT``:
             the off-block the network saw is far from the plan.
``nm_bulk``  everything else — the ordinary flight.
===========  ==========================================================

Weights are fitted by least squares on out-of-fold predictions of folds other
than the one being scored (`fit_weights` never sees the fold it is applied to;
the rotation is the caller's job and `rotated` does it).

VERSION history (2026-09-05, exp-008).  Nothing here has a default that another
experiment already depends on.

1. First version: per-regime least squares, NNLS/OLS/flat.
2. Adds `shrink_to_pool` (default off, so VERSION 1 behaviour is reachable).
3. Ridge is relative to the scale of P'P and the OLS solve is singular-safe.
4. `build_candidate` prepares each member with the floor and the residual guard
   before blending, so a served member is the same estimator as the out-of-fold
   member its weight was fitted on. `prepare_members=False` restores VERSION 3.
   Split out of item 3 after review: it is a default change that alters the
   bytes of every candidate, so it earns its own version.
5. A member may be **undefined** (null) on rows where its weight is exactly
   zero. A regime specialist -- exp-010's NM-tail head predicts NM-joined
   departures only -- otherwise cannot be served at all: filling its gap would
   invent a prediction it never made, and dropping the rows would lose them from
   the submission. `build_candidate` now fills such nulls with the floor AFTER
   verifying that every null row carries weight 0.0 for that member under its
   own regime's vector, and raises if any does not. Candidates whose members are
   all total are byte-identical to VERSION 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 5  # 4 + a member may be undefined where its weight is exactly zero

#: AOBT − LOBT above this many seconds puts an NM-joined departure in the
#: `nm_late` regime.  From the assignment; exp-002 measures the same population
#: as the one where take-off minus LOBT beats take-off minus AOBT.
LATE_CUT = 1800.0

REGIMES = ("nm_bulk", "nm_late", "nonm", "noac")

#: Output floor in seconds; a taxi-out time below this is never right and
#: exp-001 measured the floor as a small unconditional gain.
FLOOR = 60.0


# --------------------------------------------------------------- regimes ----
def regime_labels(noac, nm_missing, d_aobt_lobt, late_cut: float = LATE_CUT) -> np.ndarray:
    """Regime per row from three input-readable quantities, in priority order."""
    noac = np.asarray(noac).astype(bool)
    nm_missing = np.asarray(nm_missing).astype(bool)
    d = np.asarray(d_aobt_lobt, dtype="float64")
    out = np.full(len(noac), "nm_bulk", dtype=object)
    out[np.isfinite(d) & (d > late_cut) & ~nm_missing] = "nm_late"
    out[nm_missing & ~noac] = "nonm"
    out[noac] = "noac"
    return out


def regimes_from_features(X: pd.DataFrame, late_cut: float = LATE_CUT) -> np.ndarray:
    """Regimes from an exp-001 v2 feature frame (or an exp-008 OOF file)."""
    return regime_labels(X["noac"], X["nm_missing"], X["d_aobt_lobt"], late_cut)


def regimes_from_raw(df: pd.DataFrame, late_cut: float = LATE_CUT) -> np.ndarray:
    """Regimes from a raw movement frame's departure rows (the ranking file)."""
    d = (df["AOBT_3_flt"] - df["LOBT_flt"]).dt.total_seconds().to_numpy()
    return regime_labels(df["AIRCRAFT_TYPE_mvt"].isna(), df["FLIGHT_ID_mvt"].isna(), d, late_cut)


# --------------------------------------------------------------- weights ----
def fit_weights(P: np.ndarray, y: np.ndarray, mode: str = "nnls", ridge: float = 1e-6) -> np.ndarray:
    """One weight vector for one regime's rows.

    `mode` is `nnls` (non-negative, the default because unconstrained weights on
    members correlating above 0.9 are badly conditioned and transfer poorly
    between folds), `ols` (unconstrained, reported as a comparison), or `flat`.
    No intercept: a blend that shifts the level is a calibration change, not a
    blend, and it would be fitted on months other than the one it is applied to.
    """
    P = np.asarray(P, dtype="float64")
    y = np.asarray(y, dtype="float64")
    k = P.shape[1]
    if mode == "flat" or len(y) == 0:
        return np.full(k, 1.0 / k)
    if mode == "nnls":
        from scipy.optimize import nnls

        scale = float(np.sqrt((P ** 2).mean())) + 1.0
        A = np.vstack([P, np.sqrt(ridge) * scale * np.eye(k)])
        b = np.concatenate([y, np.zeros(k)])
        w, _ = nnls(A, b)
        return w if w.sum() > 0 else np.full(k, 1.0 / k)
    # The ridge is RELATIVE to the scale of P'P, not absolute. Members are
    # predictions in seconds, so P'P entries run to 1e9 on the bulk and a fixed
    # 1e-6 is not a regulariser at all. It also has to survive exact collinearity:
    # every LightGBM arm shares one noac head, so on the 63-to-339-row noac
    # regime four of the columns are bit-identical and the normal matrix is
    # singular. lstsq returns the minimum-norm solution there instead of raising.
    G = P.T @ P
    G = G + ridge * (np.trace(G) / max(k, 1) + 1.0) * np.eye(k)
    try:
        return np.linalg.solve(G, P.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(G, P.T @ y, rcond=None)[0]


def shrink_to_pool(w: np.ndarray, w_all: np.ndarray, n: int, n0: float) -> np.ndarray:
    """Shrink a regime's fitted weights toward the pooled ones by sample size.

    `w_shrunk = n/(n+n0) * w + n0/(n+n0) * w_all`. The regimes are wildly unequal
    -- `nm_bulk` offers hundreds of thousands of fitting rows and `noac` about
    150 -- and a weight vector fitted on 150 heavy-tailed rows measurably does
    not transfer to an unseen month (exp-008 RESULTS, finding 7). `n0` is a
    pseudo-count in rows, declared rather than tuned: a regime with `n0` fitting
    rows gets half its own fit and half the pool.
    """
    if n0 <= 0:
        return w
    a = n / (n + n0)
    return a * np.asarray(w, dtype="float64") + (1.0 - a) * np.asarray(w_all, dtype="float64")


def fit_per_regime(P: np.ndarray, y: np.ndarray, regime: np.ndarray,
                   mode: str = "nnls", ridge: float = 1e-6,
                   min_rows: int = 50, shrink_rows: float = 0.0) -> dict[str, np.ndarray]:
    """Weights per regime, with a pooled fallback where a regime is too small.

    `min_rows` exists because the regimes are wildly unequal: `nm_bulk` is about
    90 % of a month and `noac` is 63 to 339 rows, so a fit set built from two
    validation months can offer as few as 152 noac rows. The default is
    deliberately low — 152 rows still support a non-negative fit of a handful of
    weights, and `noac` is both the smallest regime and the one carrying a fifth
    of the squared error, so falling back to pooled weights there gives up the
    largest thing per-regime blending can win. Whether the fitted noac weights
    actually transfer is a measurement, and the rotation reports it.
    """
    out = {"_all": fit_weights(P, y, mode, ridge)}
    for r in REGIMES:
        m = regime == r
        if m.sum() < min_rows:
            out[r] = out["_all"]
            continue
        w = fit_weights(P[m], y[m], mode, ridge)
        out[r] = shrink_to_pool(w, out["_all"], int(m.sum()), shrink_rows)
    return out


def apply_weights(P: np.ndarray, regime: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    P = np.asarray(P, dtype="float64")
    out = np.empty(len(P), dtype="float64")
    for r in REGIMES:
        m = regime == r
        if m.any():
            out[m] = P[m] @ np.asarray(weights.get(r, weights["_all"]), dtype="float64")
    return out


def apply_residual_guard(y_hat, d_takeoff_aobt, lo: float, hi: float,
                         floor: float | None = FLOOR) -> np.ndarray:
    """Clip the implied r = BLOCK − AOBT into a range observed in training.

    Same object as `experiments/exp-001-retro-v1/guard.py`, restated here so a
    candidate can be built from this module alone; the bounds are passed in
    rather than defaulted, because they are a property of the months fitted.
    """
    y = np.asarray(y_hat, dtype="float64").copy()
    a = np.asarray(d_takeoff_aobt, dtype="float64")
    ok = np.isfinite(a)
    y[ok] = a[ok] - np.clip(a[ok] - y[ok], lo, hi)
    return np.maximum(y, floor) if floor is not None else y


# ------------------------------------------------------------- candidate ----
def stack_members(files: dict[str, str], root: Path, fold: str | None = None,
                  column: str = "y_pred") -> tuple[pd.DataFrame, list[str]]:
    """Join member prediction files on MVT_ID_mvt; inner join, order declared.

    A row a member does not cover is dropped rather than filled, so a blend can
    never be scored on a row one of its members did not actually predict.
    """
    names = list(files)
    base = None
    for n in names:
        p = Path(str(files[n]).format(fold=fold))
        p = p if p.is_absolute() else root / p
        d = pd.read_parquet(p)
        if fold is not None and "fold" in d.columns:
            d = d[d["fold"] == fold]
        keep = ["MVT_ID_mvt", column] + [c for c in ("y_true", "noac", "nm_missing", "d_aobt_lobt",
                                                     "d_takeoff_aobt", "airport") if c in d.columns]
        d = d[keep].rename(columns={column: n})
        base = d if base is None else base.merge(d[["MVT_ID_mvt", n]], on="MVT_ID_mvt", how="inner")
    return base, names


def build_candidate(config: dict | str | Path, out_path: str | Path, root: Path | None = None) -> dict:
    """Write a submission candidate from member ranking predictions and weights.

    `config["ranking"]` maps member name to a parquet with MVT_ID_mvt and
    y_pred over the ranking rows; `config["weights"]` maps regime to a weight
    per member.  Returns the record to paste into RESULTS.md.
    """
    import prc.data as D

    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    root = root or D.ROOT
    frame, names = stack_members(config["ranking"], Path(root))
    late = float(config.get("late_cut", LATE_CUT))
    r = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt",
                                "AOBT_3_flt", "LOBT_flt", "MVT_TIME_UTC_mvt"])
    r = r[r["PHASE_mvt"] == "DEP"].copy()
    r["regime"] = regimes_from_raw(r, late)
    r["d_takeoff_aobt"] = (r["MVT_TIME_UTC_mvt"] - r["AOBT_3_flt"]).dt.total_seconds()
    m = frame.merge(r[["MVT_ID_mvt", "regime", "d_takeoff_aobt"]], on="MVT_ID_mvt", how="inner")
    if len(m) != len(r):
        raise ValueError(f"members cover {len(m)} of {len(r)} ranking departures")
    W = {k: np.array([v[n] for n in names], dtype="float64") for k, v in config["weights"].items()}
    # Prepare each MEMBER exactly as the out-of-fold members the weights were
    # fitted on were prepared: floor, then the residual guard. A member served
    # raw is a different estimator from the one that earned its weight, and the
    # error is asymmetric across members because some floor internally and some
    # do not. Both operations are idempotent, so this is safe on members whose
    # own writer already applied them.
    P = m[names].to_numpy("float64").copy()
    a = m["d_takeoff_aobt"].to_numpy("float64")
    floor = float(config.get("floor", FLOOR))
    guard = config.get("guard_bounds")
    # A member undefined on some rows (a regime specialist) is legal only where
    # its weight is exactly zero. Verify that before filling, and name the
    # member and the regime if it is violated -- silently filling would serve a
    # number the member never produced.
    regime_col = m["regime"].to_numpy()
    for j, n in enumerate(names):
        bad = ~np.isfinite(P[:, j])
        if not bad.any():
            continue
        for rg in np.unique(regime_col[bad]):
            w = np.asarray(W.get(rg, W["_all"]), dtype="float64")[j]
            if w != 0.0:
                k = int((bad & (regime_col == rg)).sum())
                raise ValueError(
                    f"member {n!r} is undefined on {k} {rg} rows but carries "
                    f"weight {w:.4f} there")
        P[bad, j] = floor
    if config.get("prepare_members", True):
        for j in range(P.shape[1]):
            P[:, j] = np.maximum(P[:, j], floor)
            if guard:
                P[:, j] = apply_residual_guard(P[:, j], a, float(guard[0]), float(guard[1]), floor)
    pred = apply_weights(P, m["regime"].to_numpy(), W)
    pred = np.maximum(pred, floor)
    if guard:
        pred = apply_residual_guard(pred, m["d_takeoff_aobt"].to_numpy("float64"),
                                    float(guard[0]), float(guard[1]),
                                    floor)
    sha = D.write_submission(pd.Series(pred, index=m["MVT_ID_mvt"].to_numpy()), out_path)
    return {"path": str(out_path), "sha256": sha, "n": int(len(m)), "members": names,
            "weights": {k: dict(zip(names, map(float, v))) for k, v in W.items()},
            "late_cut": late, "blend_version": VERSION}
