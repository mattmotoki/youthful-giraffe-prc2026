"""exp-098: the Q09 stand-chain serving update priced at the final predictor on v22.

Pure functions.  The controller owns every receipt, every file write and the phase
boundary; this module owns the arithmetic and nothing else.

The object under test is the **served** update the exp-040 chain member produces:

    u = (chain d11-noac) - (v10 replay),   zeroed outside the held regimes

`u` is a frozen difference of two committed prediction vectors.  It carries no
label.  The comparator is the v22-equivalent fold panel `p_v22`.  One arm serves

    d(s) = max(p_v22 + s*u, 60) - p_v22

and is priced against `r = y - p_v22` with exp-038's registered instrument, so the
numbers sit in the same units as every earlier reading of this lane.

Two properties are used and checked rather than assumed:

* **Unchanged-row cancellation.**  A row with `u == 0` has `d == 0` and contributes
  exactly `0.0` to the paired squared-error difference `d*d - 2*r*d`.  The fold gain
  is therefore computable from the changed rows alone, and `check_cancellation`
  requires the two computations to agree.
* **Absorbed alignment is label-free.**  With `A_h = mean(u * (y - p_h))` for any
  head `h`, `A_v10 - A_v22 = mean(u * (p_v22 - p_v10))`.  The whole fall of this
  lane's alignment between the two heads is an inner product of two prediction
  vectors and needs no label at all, so it is computed before the first label read.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT / "experiments/exp-038-packet/code"))
import align as A  # noqa: E402  the registered instrument, imported and not rewritten

FOLDS = ("july_loo", "december", "january")
BOARD_FOLDS = ("july_loo", "january")
HELD_REGIMES = ("nm_bulk", "nm_late")
ALL_REGIMES = ("nm_bulk", "nm_late", "nonm", "noac")
FLOOR = 60.0
BOOT_DRAWS = 2000
BOOT_SEED = 7
BLOCK_DAYS_PRIMARY = 1
BLOCK_DAYS_SENSITIVITY = (3, 7)
SHAM_SEED = 98021
SCALE_BOUNDS = (0.0, 1.0)
AIRPORT_MIN_CHANGED = 2000
GATE_BOARD = 0.10
TAIL_SECONDS = 3600.0

ARMS = ("A0", "A1", "A2")
RECORD_ONLY_ARMS = ("A3",)
DECISION_PRECEDENCE = ("A0", "A1", "A2")


# --------------------------------------------------------------------------- shapes
def sha_values(a: np.ndarray) -> str:
    """SHA-256 of an array's exact bytes, for a receipt that names a vector."""
    b = np.ascontiguousarray(a)
    return hashlib.sha256(b.tobytes()).hexdigest()


def held_mask(regime: np.ndarray) -> np.ndarray:
    return np.isin(regime, list(HELD_REGIMES))


def raw_update(p_arm: np.ndarray, p10: np.ndarray, regime: np.ndarray) -> np.ndarray:
    """`u`: the served arm minus the v10 replay, zero outside the held regimes."""
    u = np.asarray(p_arm, "float64") - np.asarray(p10, "float64")
    return np.where(held_mask(regime), u, 0.0)


def served_delta(p_v22: np.ndarray, u: np.ndarray, s) -> np.ndarray:
    """The delta actually served on top of v22 at amplitude `s`, with the 60 s floor.

    `s` is a scalar or a per-row array.  The floor makes this non-linear in `s`, so
    the amplitude is fitted on `u` and the statistics are read on `d`.
    """
    p = np.asarray(p_v22, "float64")
    return np.maximum(p + np.asarray(s, "float64") * np.asarray(u, "float64"), FLOOR) - p


def sham_permutation(regime: np.ndarray, day: np.ndarray, ids: np.ndarray,
                     seed: int = SHAM_SEED) -> np.ndarray:
    """A deterministic within-(regime, UTC day) permutation of the row order.

    Amplitude, regime and day-block composition survive; the row assignment does
    not.  The order inside a cell is fixed by `ids` before the draw, so the result
    depends only on the seed and the data, never on the file's row order.
    """
    n = len(ids)
    out = np.arange(n)
    order = np.lexsort((np.asarray(ids), np.asarray(day), np.asarray(regime).astype(str)))
    key = np.asarray(regime).astype(str)[order] + "|" + np.asarray(day)[order].astype(str)
    rng = np.random.default_rng(int(seed))
    bounds = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        block = order[lo:hi]
        out[block] = block[rng.permutation(hi - lo)]
    return out


# --------------------------------------------------------------------------- scales
def ab_by_fold(r: dict, u: dict, keep: dict | None = None) -> tuple:
    """`A`, `B` and `n` per fold on the raw (unfloored) update, as align.py defines them."""
    Ad, Bd, nd = {}, {}, {}
    for f in r:
        m = np.ones(len(r[f]), bool) if keep is None else keep[f]
        rr, uu = r[f][m], u[f][m]
        Ad[f] = float(np.mean(rr * uu))
        Bd[f] = float(np.mean(uu * uu))
        nd[f] = int(m.sum())
    return Ad, Bd, nd


def global_loo_scales(r: dict, u: dict) -> dict:
    """A1: one bounded amplitude per fold, fitted on the other two folds only."""
    Ad, Bd, nd = ab_by_fold(r, u)
    return {f: v["s"] for f, v in A.scale_loo(Ad, Bd, nd, SCALE_BOUNDS).items()}


def airport_loo_scales(r: dict, u: dict, airport: dict, fallback: dict) -> tuple:
    """A2: one bounded amplitude per airport per fold, fitted on the other two folds.

    An airport whose two fitting folds together move fewer than
    `AIRPORT_MIN_CHANGED` rows takes that fold's A1 scale instead, so a thin
    airport never carries its own fitted parameter.
    """
    apts = sorted({a for f in u for a in np.unique(airport[f])})
    scales, prov = {}, {}
    for f in u:
        s_row = np.full(len(u[f]), float(fallback[f]))
        prov[f] = {}
        for a in apts:
            num = den = 0.0
            changed = 0
            for g in u:
                if g == f:
                    continue
                m = (airport[g] == a) & (u[g] != 0.0)
                changed += int(m.sum())
                num += float(np.sum(r[g][m] * u[g][m]))
                den += float(np.sum(u[g][m] * u[g][m]))
            own = changed >= AIRPORT_MIN_CHANGED and den > 0.0
            s = float(min(max(num / den, SCALE_BOUNDS[0]), SCALE_BOUNDS[1])) if own \
                else float(fallback[f])
            sel = airport[f] == a
            s_row[sel] = s
            prov[f][str(a)] = {"s": s, "own_fit": bool(own),
                               "changed_rows_in_fitting_folds": changed,
                               "rows_in_this_fold": int(sel.sum())}
        scales[f] = s_row
    return scales, prov


def cv_scales(r: dict, u: dict, day: dict) -> dict:
    """A3, record only: the within-month day-parity cross-fitted amplitude."""
    out = {}
    for f in u:
        blk = A.blocks(day[f], BLOCK_DAYS_PRIMARY)
        out[f] = A.scale_cv(r[f], u[f], blk, np.ones(len(u[f]), bool), SCALE_BOUNDS)
    return out


# --------------------------------------------------------------------------- pricing
def check_cancellation(r: np.ndarray, d: np.ndarray) -> dict:
    """Unchanged rows must cancel exactly in the paired squared-error difference."""
    zero = d == 0.0
    dsse = d * d - 2.0 * r * d
    changed = ~zero
    n = len(d)
    full = float(np.sum(2.0 * r * d - d * d)) / n
    only_changed = float(np.sum(2.0 * r[changed] * d[changed] - d[changed] * d[changed])) / n
    return {"rows": n,
            "rows_unchanged": int(zero.sum()),
            "rows_changed": int(changed.sum()),
            "max_abs_dsse_on_unchanged_rows": float(np.abs(dsse[zero]).max()) if zero.any() else 0.0,
            "gain_mse_all_rows": full,
            "gain_mse_changed_rows_only": only_changed,
            "abs_difference": abs(full - only_changed),
            "cancels": bool(np.all(dsse[zero] == 0.0) and abs(full - only_changed) < 1e-9)}


def slice_masks(y: np.ndarray, regime: np.ndarray, clean: np.ndarray) -> dict:
    return {"all": np.ones(len(y), bool),
            "y_le_3600": y <= TAIL_SECONDS,
            "y_gt_3600": y > TAIL_SECONDS,
            "no_nm": regime == "nonm",
            "clean": np.asarray(clean, bool)}


def price_fold(r: np.ndarray, d: np.ndarray, day: np.ndarray, regime: np.ndarray,
               band: np.ndarray, airport: np.ndarray, y: np.ndarray,
               clean: np.ndarray, proposed: np.ndarray) -> dict:
    """Every registered per-fold number for one arm on one fold.

    `proposed` is `s*u`, the update before the 60 s floor; `d` is what the floor
    leaves.  Their disagreement is the floor-bound population and is counted.
    """
    keep = np.ones(len(r), bool)
    st = A.fold_stats(r, d, keep)
    out = {"rows": st["n"], "rows_changed": st["n_changed"],
           "max_abs_delta": st["max_abs_d"],
           "A": st["A"], "B": st["B"],
           "scale_A_over_B": st["scale"],
           "gain_mse_full_s2": st["gain_mse_full"],
           "gain_mse_fitted_in_sample_s2": st["gain_mse_fitted"],
           "mse_v22_s2": st["mse_v10"], "rmse_v22_s": st["rmse_v10"],
           "rmse_candidate_s": st["rmse_cand"], "gain_rmse_s": st["gain_rmse"],
           "floor_bound_rows": int(np.sum(d != np.asarray(proposed, "float64")))}
    e22, ec = r * r, (r - d) * (r - d)
    blk = A.blocks(day, BLOCK_DAYS_PRIMARY)
    out["ci95_1day"] = A.boot_fold(blk, e22, ec, BOOT_DRAWS, BOOT_SEED)
    out["clear_of_zero_1day"] = bool(out["ci95_1day"][0] > 0.0)
    out["ci95_sensitivity"] = {
        f"{w}day": A.boot_fold(A.blocks(day, w), e22, ec, BOOT_DRAWS, BOOT_SEED)
        for w in BLOCK_DAYS_SENSITIVITY}
    out["by_regime"] = {}
    for g in ALL_REGIMES:
        m = regime == g
        out["by_regime"][g] = {
            "rows": int(m.sum()), "rows_changed": int((m & (d != 0.0)).sum()),
            "sse_gain_s2": float(np.sum(2.0 * r[m] * d[m] - d[m] * d[m])) if m.any() else 0.0,
            "rms_delta": float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else 0.0,
            "max_abs_delta": float(np.abs(d[m]).max()) if m.any() else 0.0}
    out["by_band"] = {}
    for label, m in (("fixed_table_band", np.asarray(band, bool)),
                     ("learned_mixture", ~np.asarray(band, bool))):
        out["by_band"][label] = {
            "rows": int(m.sum()), "rows_changed": int((m & (d != 0.0)).sum()),
            "sse_gain_s2": float(np.sum(2.0 * r[m] * d[m] - d[m] * d[m])) if m.any() else 0.0}
    out["by_airport"] = {}
    for a in sorted(np.unique(airport)):
        m = airport == a
        out["by_airport"][str(a)] = {
            "rows": int(m.sum()), "rows_changed": int((m & (d != 0.0)).sum()),
            "sse_gain_s2": float(np.sum(2.0 * r[m] * d[m] - d[m] * d[m]))}
    out["slices"] = {}
    for name, m in slice_masks(y, regime, clean).items():
        if not m.any():
            out["slices"][name] = {"rows": 0}
            continue
        mse22 = float(np.mean(r[m] ** 2))
        msec = float(np.mean((r[m] - d[m]) ** 2))
        out["slices"][name] = {
            "rows": int(m.sum()), "rows_changed": int((m & (d != 0.0)).sum()),
            "rmse_v22_s": float(np.sqrt(mse22)), "rmse_candidate_s": float(np.sqrt(msec)),
            "gain_rmse_s": float(np.sqrt(mse22) - np.sqrt(msec))}
    out["cancellation"] = check_cancellation(r, d)
    return out


def board(per_fold: dict, parts: dict) -> dict:
    """The board endpoint and its interval, from the same per-fold statistics."""
    stats = {f: {"mse_v10": per_fold[f]["mse_v22_s2"],
                 "gain_mse_full": per_fold[f]["gain_mse_full_s2"],
                 "gain_mse_fitted": per_fold[f]["gain_mse_fitted_in_sample_s2"]}
             for f in BOARD_FOLDS}
    g = A.board_gain(stats)
    ci = A.boot_board(parts, BOOT_DRAWS, BOOT_SEED)
    return {"board_rmse_v22_s": g["board_v10"], "board_rmse_candidate_s": g["board_cand"],
            "board_gain_s": g["gain_board"], "board_gain_ci95_1day": ci,
            "board_gain_in_sample_fitted_ceiling_s": g["gain_board_fitted"],
            "clear_of_zero": bool(ci[0] > 0.0)}


def gates(arm: dict, ctrl_refit: dict, ctrl_perm: dict) -> dict:
    """The registered gates.  Every one is decided before the read; none is edited after."""
    b = arm["board"]["board_gain_s"]
    g = {
        "board_gain_at_least_0_10_s": bool(b >= GATE_BOARD),
        "july_lower_bound_above_zero": bool(arm["folds"]["july_loo"]["ci95_1day"][0] > 0.0),
        "january_lower_bound_above_zero": bool(arm["folds"]["january"]["ci95_1day"][0] > 0.0),
        "december_point_nonnegative": bool(arm["folds"]["december"]["gain_rmse_s"] >= 0.0),
        "beats_sham_refit_control": bool(b > ctrl_refit["board"]["board_gain_s"]),
        "beats_permutation_control": bool(b > ctrl_perm["board"]["board_gain_s"]),
    }
    g["all_pass"] = bool(all(g.values()))
    return g
