"""Step 2 -- Q11's cheap event-order arm, on the channel the anatomy named.

The anatomy (T6) says the shared within-cell error sits on the RUNWAY inside the
window: `Var(runway | window)` is 5,977 / 4,622 / 3,886 s^2 against 1,779 / 935 /
1,483 for the whole airport-window.  Q11's cheap arm is the one screen in the
queue that reads the runway's own event stream, so it runs first.

Every take-off clock in the ranking file is an input (`docs/input-rule.md`), so
the ordered stream of take-offs on a runway is observable at scoring time.  The
question is whether its ORDER carries anything the same numbers unordered do not.

Five arms, one learner, one capacity, one seed, one set of rows:

    h0      v10 + the SHIPPED K2-enabled current feature block  (the matched base)
    ord     h0 + the last four same-runway take-off gaps and the four
            interleaved arrival counts, IN TIME ORDER                     (+8)
    unord   h0 + the SAME EIGHT NUMBERS with the four (gap, count) pairs
            sorted by gap: the identical multiset, no temporal position   (+8)
    unord2  h0 + classic unordered summaries of the same stream: span,
            min/max/median gap, total/min/max/median count                (+8)
    ord_c   h0 + the ordered block read SEVEN DAYS AWAY on the same airport
            and runway -- same shape, no current information              (+8)

`unord` is the contrast the queue asks for and `ord_c` is the placebo Q04 taught
this repository to run: at matched capacity a shifted block that cannot contain
the answer beat the real one there, and an arm that does not beat its own
placebo has added capacity, not information.

Advance clause (assignments/cell.md): `ord` beats `unord` AND the pair beats v10
with intervals clear.

    python order.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import base as B  # noqa: E402

sys.path.insert(0, str(B.S39))
import q04 as Q4      # noqa: E402  (arrival loader)
import rowfeat as R   # noqa: E402
import anatomy as AN  # noqa: E402  (arrival_cells / wide window, one implementation)

EXP = B.EXP
FOLDS = B.FOLDS
D = B.D
PK = B.PK

# ---- predeclared -------------------------------------------------------------
K = 4                       # "the last four same-runway take-off gaps"
PLACEBO_DAYS = 7
GAP_CLIP = 3600.0           # a gap longer than an hour is a quiet runway, not a queue
WIDE_BINS = 4
#: two matched capacities, both applied to every arm.  B is the headline: a
#: depth-2 stump ensemble cannot represent a joint effect of four ordered gaps,
#: which is the whole question.  A is Q04's capacity, kept for comparability.
CAPS = {"A": dict(max_iter=64, max_depth=2, learning_rate=0.05),
        "B": dict(max_iter=300, max_depth=3, learning_rate=0.05)}
HEADLINE_CAP = "B"
ARMS = ("h0", "ord", "unord", "unord2", "ord_c")


def query_times(m: pd.DataFrame, shift_s: float = 0.0) -> np.ndarray:
    """The query clock: the row's own take-off, or a placebo `shift_s` away.

    A single backward shift loses the month's first seven days; the placebo
    therefore steps BACK where the month reaches that far and FORWARD otherwise,
    so it is defined on the same rows the real block is (Q04's round-1 lesson:
    a placebo must be read on matched support, not on a smaller cohort).
    """
    t = m["t"].to_numpy("float64")
    if shift_s == 0.0:
        return t.copy()
    lo, hi = float(t.min()), float(t.max())
    q = t - shift_s
    q = np.where(q < lo, t + shift_s, q)
    return np.where(q > hi, t - shift_s, q)


def stream_blocks(m: pd.DataFrame, q: np.ndarray) -> dict:
    """The last `K` same-runway take-off gaps before `q`, and their boundaries.

    The stream is every departure of the month on that airport and runway,
    whatever its serving regime -- at scoring time the whole take-off stream is
    given (`docs/input-rule.md`).
    """
    apt = m["apt"].to_numpy()
    rwy = m["rwy"].to_numpy()
    t = m["t"].to_numpy("float64")
    n = len(m)
    prev = np.full((n, K + 1), np.nan)
    prev[:, 0] = q
    key = np.char.add(np.char.add(apt.astype(str), "|"), rwy.astype(str))
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1, len(ks)]
    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        T = np.sort(t[idx])
        pos = np.searchsorted(T, q[idx], side="left")   # take-offs strictly before q
        for j in range(1, K + 1):
            p = pos - j
            ok = p >= 0
            v = np.full(len(idx), np.nan)
            v[ok] = T[p[ok]]
            prev[idx, j] = v
    gaps = np.clip(prev[:, :-1] - prev[:, 1:], 0.0, GAP_CLIP)   # g1..gK, in order
    return {"prev": prev, "gaps": gaps}


def arrival_counts(m: pd.DataFrame, prev: np.ndarray, arr_t: dict) -> np.ndarray:
    """Arrivals at the airport landing inside each consecutive take-off interval.

    Same airport, any runway: at most European airports the arrival runway is not
    the departure runway, so a same-runway count would be zero almost everywhere
    and would test the join, not the interleaving.  Stated as a choice, not a
    default.
    """
    apt = m["apt"].to_numpy()
    n = len(m)
    out = np.full((n, K), np.nan)
    for a in np.unique(apt):
        sel = np.flatnonzero(apt == a)
        A = arr_t.get(a)
        if A is None or len(A) == 0:
            continue
        P = prev[sel]
        for j in range(K):
            hi = np.searchsorted(A, P[:, j], side="right")
            lo = np.searchsorted(A, P[:, j + 1], side="right")
            v = (hi - lo).astype("float64")
            v[~np.isfinite(P[:, j + 1])] = np.nan
            out[sel, j] = v
    return out


def order_blocks(gaps: np.ndarray, cnts: np.ndarray) -> dict:
    """The three eight-number representations of one stream."""
    ordered = np.column_stack([gaps, cnts])
    # the SAME multiset of (gap, count) pairs, sorted by gap: only the temporal
    # position is destroyed.  NaNs sort last, so a short stream keeps its shape.
    o = np.argsort(np.where(np.isnan(gaps), np.inf, gaps), axis=1, kind="stable")
    g_s = np.take_along_axis(gaps, o, axis=1)
    c_s = np.take_along_axis(cnts, o, axis=1)
    sorted_pairs = np.column_stack([g_s, c_s])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summ = np.column_stack([
            np.nansum(gaps, axis=1), np.nanmin(gaps, axis=1),
            np.nanmax(gaps, axis=1), np.nanmedian(gaps, axis=1),
            np.nansum(cnts, axis=1), np.nanmin(cnts, axis=1),
            np.nanmax(cnts, axis=1), np.nanmedian(cnts, axis=1)])
    allnan = np.isnan(gaps).all(axis=1)
    summ[allnan] = np.nan
    return {"ordered": ordered, "sorted": sorted_pairs, "summary": summ}


#: Extra ordinal columns a caller can ask for.  Nothing in this experiment does
#: any more: Sol round 1, blocking finding 7 established that the shipped block's
#: own `c_rwy`, `c_oper` and the rest come from ONE persistent vocabulary shared
#: by every month, so the same base transfers between folds unchanged and the
#: cross-month control reads exactly the columns steps 2 and 3 read.
BASE_CATS = ("rwy", "pier", "typ", "wtc", "seg")


def base_features(m: pd.DataFrame, ac: pd.DataFrame, month: str,
                  vocab: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """The matched base: v10 plus the SHIPPED current feature block.

    exp-039's Sol round 2 (blocking finding b1) settled that "all current
    arrival, runway, clock and window summaries" means the shipped block itself,
    not a hand-built substitute, so that is what carries the base here.  Three
    columns sit beside it -- v10's own prediction and the two clock differences
    the repository names -- and six carry the exact 10-minute cell and
    wide-window counts the arms' own features are built on, so the base already
    holds every summary of the cell the arms will add a representation of.

    `vocab` is retained for a caller that wants hand-built ordinals instead of
    the shipped codes; no caller here uses it, because the shipped codes are
    already shared across months (`base.shipped_frame` asserts it).
    """
    apt = m["apt"].to_numpy()
    bins = m["bin"].to_numpy()
    key = pd.MultiIndex.from_arrays([apt, bins], names=["apt", "bin"])
    dep_cnt = pd.Series(1.0, index=key).groupby(level=[0, 1]).sum()
    n_dep = dep_cnt.reindex(key).fillna(0.0).to_numpy()
    n_arr = ac["n_arr"].reindex(key).fillna(0.0).to_numpy()
    dep_wide = AN.wide_count(dep_cnt, WIDE_BINS).reindex(key).fillna(0.0).to_numpy()
    arr_wide = AN.wide_count(ac["n_arr"], WIDE_BINS).reindex(key).fillna(0.0).to_numpy()
    arr_taxi = ac["arr_taxi"].reindex(key).to_numpy()
    taxi_wide = AN.wide(ac["arr_taxi"], ac["n_arr"], WIDE_BINS).reindex(key).to_numpy()
    W, wcols = B.shipped(month, m["MVT_ID_mvt"].to_numpy("float64"),
                         drop_cats=vocab is not None)
    head = np.column_stack([m["p10"].to_numpy("float64"),
                            m["d_takeoff_aobt"].to_numpy("float64"),
                            m["d_aobt_lobt"].to_numpy("float64")])
    cells = np.column_stack([n_dep, n_arr, dep_wide, arr_wide, arr_taxi, taxi_wide])
    names = (["p10", "d_takeoff_aobt", "d_aobt_lobt"] + wcols
             + ["n_dep_cell", "n_arr_cell", "n_dep_wide", "n_arr_wide",
                "arr_taxi_cell", "arr_taxi_wide"])
    blocks = [head.astype("float32"), W, cells.astype("float32")]
    if vocab is not None:
        ords = np.column_stack(
            [np.array([vocab["apt"].get(v, -1) for v in apt], dtype="float64")]
            + [np.array([vocab[c].get(v, -1) for v in m[c].to_numpy()],
                        dtype="float64") for c in BASE_CATS])
        blocks.append(ords.astype("float32"))
        names += ["apt_ord"] + [f"{c}_ord" for c in BASE_CATS]
    return np.column_stack(blocks), names


def fit_arm(X, r, ei, blk, cap) -> np.ndarray:
    h = np.zeros(len(r))
    for b in range(B.BLOCKS):
        fit = ei[blk[ei] != b]
        sc = ei[blk[ei] == b]
        g = HistGradientBoostingRegressor(random_state=B.SEED, **CAPS[cap])
        g.fit(X[fit], r[fit])
        h[sc] = g.predict(X[sc])
    return h


def run_fold(fold: str, pan: pd.DataFrame, rng, preds: dict) -> dict:
    month = R.month_of(fold)
    m = B.panel(fold, pan)
    r = m["r"].to_numpy("float64")
    y = m["y_true"].to_numpy("float64")
    ids = m["MVT_ID_mvt"].to_numpy("float64")
    p10 = m["p10"].to_numpy("float64")
    days = m["day"].to_numpy()
    blk = B.blocks_of(days)
    ei = np.flatnonzero(m["elig"].to_numpy())

    a = Q4.arrivals(month)
    arr_t = {k: np.sort(v.to_numpy()) for k, v in a.groupby("apt")["t"]}
    ac = AN.arrival_cells(month)

    S = stream_blocks(m, query_times(m))
    C = arrival_counts(m, S["prev"], arr_t)
    reps = order_blocks(S["gaps"], C)
    Sc = stream_blocks(m, query_times(m, PLACEBO_DAYS * 86400.0))
    Cc = arrival_counts(m, Sc["prev"], arr_t)
    reps_c = order_blocks(Sc["gaps"], Cc)

    base, base_names = base_features(m, ac, month)
    #  One arm matrix at a time: five copies of a 174-column float64 block is
    #  1.8 GB on a laptop this experiment shares with another worker, so the
    #  blocks are float32 and each arm's matrix is built, fitted at both
    #  capacities and freed before the next one is built.
    BLOCK = {"h0": None, "ord": reps["ordered"], "unord": reps["sorted"],
             "unord2": reps["summary"], "ord_c": reps_c["ordered"]}

    def arm_matrix(name: str) -> np.ndarray:
        b = BLOCK[name]
        return base if b is None else np.column_stack(
            [base, np.asarray(b, "float32")])

    out = {"fold": fold, "month": month, "n_eligible": int(len(ei)),
           "n_base_features": int(base.shape[1]),
           "stream_coverage": {
               f"gap_{j+1}_present_pct": round(
                   100 * float(np.isfinite(S["gaps"][ei, j]).mean()), 2)
               for j in range(K)},
           "placebo_coverage_pct": round(
               100 * float(np.isfinite(Sc["gaps"][ei, 0]).mean()), 2),
           "gap1_median_s": round(float(np.nanmedian(S["gaps"][ei, 0])), 1),
           "arrivals_between_median": round(float(np.nanmedian(C[ei, 0])), 2),
           "order_differs_pct": round(100 * float(np.nanmean(
               (reps["ordered"][ei] != reps["sorted"][ei]).any(axis=1))), 2),
           "arms": {}}

    H = {}
    for name in ARMS:
        Xn = arm_matrix(name)
        for cap in CAPS:
            h = fit_arm(Xn, r, ei, blk, cap)
            H[f"{name}@{cap}"] = h
            p_new = p10 + h
            A_ = float((r[ei] * h[ei]).mean()); Bq = float((h[ei] ** 2).mean())
            v = 2 * r[ei] * h[ei] - h[ei] ** 2
            pt, lo, hi = B.boot_ratio(v, np.ones(len(ei)), days[ei], rng)
            rec = {"cap": cap, "n_features": int(Xn.shape[1]),
                   "A_mean_r_h": round(A_, 3), "B_mean_h2": round(Bq, 3),
                   "mse_gain_s2": round(pt, 3), "ci": [round(lo, 3), round(hi, 3)],
                   "unrestricted_scale": round(A_ / Bq, 4) if Bq > 0 else None,
                   "multiday": PK.boot_blocks(v, days[ei], rng),
                   "rmse_all_v10": round(D.rmse(y, p10), 4),
                   "rmse_all_new": round(D.rmse(y, p_new), 4),
                   "rmse_clean_v10": round(D.rmse_clean(y, p10, ids, fold), 4),
                   "rmse_clean_new": round(D.rmse_clean(y, p_new, ids, fold), 4)}
            rec["gain_all"] = round(rec["rmse_all_v10"] - rec["rmse_all_new"], 4)
            rec["gain_clean"] = round(rec["rmse_clean_v10"] - rec["rmse_clean_new"], 4)
            out["arms"][f"{name}@{cap}"] = rec
            if cap == HEADLINE_CAP:
                preds.setdefault(name, {})[fold] = p_new
                PK.dates("order", name, fold, days, y, p10, p_new)
        del Xn
    for cap in CAPS:
        # The paired contrasts, at this capacity, on the SAME rows and the SAME
        # days: the difference of two arms' per-row `2A - B` is bootstrapped as
        # one quantity, so the interval is on the contrast and not on the
        # difference of two marginal intervals.  This is the advance clause's
        # statistic.
        for a1, a2 in (("ord", "h0"), ("unord", "h0"), ("unord2", "h0"),
                       ("ord_c", "h0"), ("ord", "unord"), ("ord", "unord2"),
                       ("ord", "ord_c"), ("unord", "ord_c")):
            k1, k2 = f"{a1}@{cap}", f"{a2}@{cap}"
            h1, h2 = H[k1][ei], H[k2][ei]
            dv = (2 * r[ei] * h1 - h1 ** 2) - (2 * r[ei] * h2 - h2 ** 2)
            pt, lo, hi = B.boot_ratio(dv, np.ones(len(ei)), days[ei], rng)
            out["arms"][k1].setdefault("vs", {})[a2] = {
                "delta_s2": round(pt, 3), "ci": [round(lo, 3), round(hi, 3)],
                "clear_of_zero": bool(lo > 0 or hi < 0),
                "multiday": PK.boot_blocks(dv, days[ei], rng)}
    return out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(B.SEED)
    pan = B.V.load()
    recs, preds = {}, {}
    for f in FOLDS:
        recs[f] = run_fold(f, pan[f], rng, preds)
        c = recs[f]
        print(f"[{f}] {c['n_eligible']:,} eligible; gap1 median "
              f"{c['gap1_median_s']} s, arrivals between {c['arrivals_between_median']}; "
              f"order differs from the sorted multiset on "
              f"{c['order_differs_pct']} % of rows", flush=True)
        for k, v in c["arms"].items():
            print(f"    {k:12s} {v['n_features']:2d} feats  2A-B "
                  f"{v['mse_gain_s2']:+9.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] "
                  f"s^2   all-row {v['gain_all']:+.4f} s"
                  + ("\n                 vs " + "; ".join(
                      f"{a} {d['delta_s2']:+.1f} [{d['ci'][0]:+.1f}, "
                      f"{d['ci'][1]:+.1f}]{'*' if d['clear_of_zero'] else ''}"
                      for a, d in v["vs"].items()) if "vs" in v else ""),
                  flush=True)
    board = {}
    for name in ARMS:
        board[name] = PK.primary(pan, preds[name], f"arm {name} at capacity "
                                 f"{HEADLINE_CAP}")
        #  an interval on the PRIMARY endpoint itself (Sol round 1, blocking
        #  finding 5), blocks drawn inside each fold
        parts = {f: (B.V.day(pan[f]["t"].to_numpy("float64")),
                     (pan[f]["y_true"].to_numpy("float64")
                      - pan[f]["p10"].to_numpy("float64")) ** 2,
                     (pan[f]["y_true"].to_numpy("float64")
                      - np.asarray(preds[name][f], "float64")) ** 2)
                 for f in FOLDS}
        board[name]["board_interval"] = B.board_boot(parts, rng)
    out = {"screen": "order",
           "predeclared": {"K": K, "placebo_days": PLACEBO_DAYS,
                           "gap_clip_s": GAP_CLIP, "caps": CAPS,
                           "headline_cap": HEADLINE_CAP, "blocks": B.BLOCKS,
                           "seed": B.SEED, "bin_s": B.BIN,
                           "eligible_regime": B.ELIGIBLE_REGIME,
                           "hold_airports": list(B.HOLD_AIRPORTS),
                           "advance_clause": "ord beats unord AND the pair beats "
                                             "v10 with intervals clear"},
           "folds": recs, "primary": board["ord"], "boards": board,
           "wall_s": round(time.time() - t0, 1)}
    p = EXP / "analysis" / "order.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    for name in ARMS:
        print(f"  oof: {B.write_oof(f'order-{name}', pan, preds[name])}")
    B.write_packet("order", out, pan)
    print("\n  PRIMARY all-row january/july board, capacity " + HEADLINE_CAP + ":")
    for name in ARMS:
        b = board[name]["board_all_row"]
        bi = board[name]["board_interval"]
        print(f"    {name:8s} {b['v10']:.4f} -> {b['arm']:.4f} ({b['gain']:+.4f} s)"
              f"   clean {board[name]['board_clean']['gain']:+.4f} s"
              f"   1d {bi['ci_block_1d']} 3d {bi['ci_block_3d']}")
    print(f"  written: {p}  ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
