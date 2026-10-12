"""Q01's alignment algebra, and the three endpoints the contract asks for.

For one candidate on one fold, with `y` the label, `p10` the served v10 vector
and `p` the candidate's:

    r = y - p10                 the residual v10 leaves
    d = p  - p10                the update the candidate proposes
    A = mean(r*d)               alignment: how much of `r` the update explains
    B = mean(d*d)               the update's own energy

    MSE(p) - MSE(p10) = mean((r-d)^2) - mean(r^2) = B - 2A

so the **full-amplitude MSE gain is 2A - B** and the unrestricted fitted scale
is `A/B` when `B > 0` (the `s` minimising `mean((r - s*d)^2)`), whose gain is
`A^2/B`.  `B = 0` means the candidate is v10 on those rows and the update is
zero; the row contributes nothing and `A/B` is undefined, not infinite.

Three endpoints, each computed on the same vectors:

  * **all-row** January/July MSE at the ranking month shares -- the contract's
    primary.  `board = sqrt(share_july*MSE_july + share_jan*MSE_jan)`, so a gain
    in board seconds is `board(v10) - board(candidate)` on the same rows.
  * **clean** -- the same with each fold's unreached-rollover rows removed
    (`prc.data.excluded_ids`).  This is the convention every LEDGER row from
    stack-5 onward is quoted on, so it is what makes this table comparable with
    the record; it is NOT the primary here.
  * **december**, reported alone: it is scored and gated but carries no board
    share.

Uncertainty is a block bootstrap over CALENDAR DAYS shared across airports
(`assignments/packet.md` and the queue's contract: "date blocks shared across
airports"), resampled independently inside each fold and recombined through the
same board formula, so the interval is on the endpoint and not on one fold.  The
multi-day sensitivity check repeats it on fixed 3-day and 7-day blocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FOLDS = ("july_loo", "december", "january")
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}
BOARD_FOLDS = ("july_loo", "january")


def day_of(t: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(t, "float64") / 86400.0).astype("int64")


def blocks(day: np.ndarray, width: int) -> np.ndarray:
    """Fixed `width`-day blocks, shared across airports, keyed from the fold's first day."""
    return (day - day.min()) // int(width)


def per_row(y: np.ndarray, p10: np.ndarray, p: np.ndarray) -> dict:
    """`r`, `d` and the paired squared-error contribution of every row."""
    y = np.asarray(y, "float64")
    r = y - np.asarray(p10, "float64")
    d = np.asarray(p, "float64") - np.asarray(p10, "float64")
    return {"r": r, "d": d, "dsse": d * d - 2.0 * r * d}   # e_cand - e_v10, row by row


def fold_stats(r: np.ndarray, d: np.ndarray, keep: np.ndarray) -> dict:
    """A, B, the two gains and the fitted scale on the rows `keep` selects."""
    n = int(keep.sum())
    rr, dd = r[keep], d[keep]
    A = float(np.mean(rr * dd))
    B = float(np.mean(dd * dd))
    out = {"n": n, "n_changed": int((dd != 0.0).sum()),
           "max_abs_d": float(np.abs(dd).max()) if n else 0.0,
           "A": A, "B": B, "gain_mse_full": 2.0 * A - B,
           "mse_v10": float(np.mean(rr * rr))}
    out["scale"] = (A / B) if B > 0 else None
    out["gain_mse_fitted"] = (A * A / B) if B > 0 else 0.0
    out["rmse_v10"] = float(np.sqrt(out["mse_v10"]))
    out["rmse_cand"] = float(np.sqrt(max(out["mse_v10"] - out["gain_mse_full"], 0.0)))
    out["gain_rmse"] = out["rmse_v10"] - out["rmse_cand"]
    return out


def board_gain(stats: dict) -> dict:
    """The two-month board endpoint from per-fold `mse_v10` and `gain_mse_full`."""
    b10 = np.sqrt(sum(SHARES[f] * stats[f]["mse_v10"] for f in BOARD_FOLDS))
    bc = np.sqrt(sum(SHARES[f] * (stats[f]["mse_v10"] - stats[f]["gain_mse_full"])
                     for f in BOARD_FOLDS))
    gmse = sum(SHARES[f] * stats[f]["gain_mse_full"] for f in BOARD_FOLDS)
    gfit = sum(SHARES[f] * stats[f]["gain_mse_fitted"] for f in BOARD_FOLDS)
    bfit = np.sqrt(sum(SHARES[f] * (stats[f]["mse_v10"] - stats[f]["gain_mse_fitted"])
                       for f in BOARD_FOLDS))
    return {"board_v10": float(b10), "board_cand": float(bc),
            "gain_board": float(b10 - bc), "gain_mse": float(gmse),
            "gain_board_fitted": float(b10 - bfit), "gain_mse_fitted": float(gfit)}


def _cells(blk: np.ndarray, e10: np.ndarray, ec: np.ndarray) -> tuple:
    """Per-block sums of the two squared errors and the count."""
    k = int(blk.max()) + 1
    order = np.argsort(blk, kind="stable")
    starts = np.searchsorted(blk[order], np.arange(k))
    s10 = np.add.reduceat(e10[order], starts)
    sc = np.add.reduceat(ec[order], starts)
    cnt = (np.searchsorted(blk[order], np.arange(k), side="right") - starts).astype("float64")
    return s10, sc, cnt


def boot_board(parts: dict, draws: int = 2000, seed: int = 7) -> list:
    """95 % block-bootstrap interval on the board-second gain, in the GAIN convention.

    `parts[fold] = (block index, e_v10, e_cand)` on the rows that count for that
    fold.  Blocks are resampled inside each fold; the two board folds are then
    recombined through the same weighted formula on every draw, so the interval
    is on the endpoint the table reports.
    """
    rng = np.random.default_rng(seed)
    tot10 = np.zeros(draws)
    totc = np.zeros(draws)
    for f in BOARD_FOLDS:
        blk, e10, ec = parts[f]
        s10, sc, cnt = _cells(blk, e10, ec)
        k = len(cnt)
        W = rng.multinomial(k, np.full(k, 1.0 / k), size=draws).astype("float64")
        n = W @ cnt
        tot10 += SHARES[f] * (W @ s10) / n
        totc += SHARES[f] * (W @ sc) / n
    out = np.sqrt(tot10) - np.sqrt(totc)
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def boot_fold(blk: np.ndarray, e10: np.ndarray, ec: np.ndarray,
              draws: int = 2000, seed: int = 7) -> list:
    """The same interval on one fold's RMSE gain."""
    rng = np.random.default_rng(seed)
    s10, sc, cnt = _cells(blk, e10, ec)
    k = len(cnt)
    W = rng.multinomial(k, np.full(k, 1.0 / k), size=draws).astype("float64")
    n = W @ cnt
    out = np.sqrt((W @ s10) / n) - np.sqrt((W @ sc) / n)
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def scale_loo(A_by_fold: dict, B_by_fold: dict, n_by_fold: dict,
              bound: tuple = (0.0, 1.0)) -> dict:
    """One bounded amplitude per fold, fitted on the OTHER folds only.

    `assignments/packet.md` / Q01: "Never fit and score an update weight on the
    same rows", and the repair is "fit one bounded amplitude on the development
    side and test it on the challenge side".  The rotation has no challenge side,
    so the strictest analogue available is cross-MONTH: the amplitude a fold is
    scored at is fitted on the two months it is not.  `A` and `B` are means, so
    the pooled numerator and denominator are row-count weighted.
    """
    out = {}
    for f in A_by_fold:
        num = sum(A_by_fold[g] * n_by_fold[g] for g in A_by_fold if g != f)
        den = sum(B_by_fold[g] * n_by_fold[g] for g in B_by_fold if g != f)
        s = (num / den) if den > 0 else 0.0
        s = float(min(max(s, bound[0]), bound[1]))
        out[f] = {"s": s, "gain_mse": 2.0 * s * A_by_fold[f] - s * s * B_by_fold[f]}
    return out


def scale_cv(r: np.ndarray, d: np.ndarray, blk: np.ndarray, keep: np.ndarray,
             bound: tuple = (0.0, 1.0)) -> dict:
    """Two-fold cross-fitted bounded amplitude INSIDE one month, on date blocks.

    Blocks alternate by parity, so each half is a set of whole days spread
    across the month: the amplitude scoring a row is fitted only on days the row
    is not in.  This holds the month fixed, which `scale_loo` does not, and the
    two together separate "the amplitude does not transfer across days" from
    "it does not transfer across months".
    """
    r, d, blk = r[keep], d[keep], blk[keep]
    par = (blk % 2).astype(bool)
    tot, n = 0.0, len(d)
    ss = {}
    for half in (False, True):
        fit = par != half
        num = float(np.sum(r[fit] * d[fit]))
        den = float(np.sum(d[fit] * d[fit]))
        s = (num / den) if den > 0 else 0.0
        s = float(min(max(s, bound[0]), bound[1]))
        ss[int(half)] = s
        sc = par == half
        tot += 2.0 * s * float(np.sum(r[sc] * d[sc])) - s * s * float(np.sum(d[sc] * d[sc]))
    return {"gain_mse": tot / n if n else 0.0, "s": ss}
