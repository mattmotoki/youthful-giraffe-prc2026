"""Step 2 -- the fit Q03 earned: a capacity-matched learner on the same objective.

Five arms on Q03's objective and Q03's cell (airport x UTC 10-minute bin x
regime, four-row minimum, `nm_bulk` outside EHAM and EDDF):

    ridge              the Q03 reference, 202 row-only features, alpha 100
    lgb_small          LightGBM, 4 leaves x 64 trees -- 256 leaves against the
                       ridge's ~202 effective parameters: the capacity match
    lgb_wide           LightGBM, 16 leaves x 400 trees -- 30x the ridge, declared
                       only to answer whether the capacity-matched budget bound
    lgb_shipped        lgb_small's budget on the 202 row-only features PLUS the
                       89 numeric columns of the shipped block
    lgb_shipped_wide   lgb_wide's budget on the same 291 columns

Each under two cross-fittings: `dayblock` (Q03's, five contiguous day blocks
inside the month) and `xmonth` (fit on the other two folds' months, score the
third -- the estimator the served candidate actually is, with the categorical
level list frozen on the fitting months).

Both arms of a family are scored at the SAME amplitude: Q03's cross-month
scalar, fitted on the other two folds' DAYBLOCK updates, so it is a function of
months the scored fold is not, under either cross-fitting.  A two-month fit is
less noisy than a within-month one, so that amplitude is conservative for the
`xmonth` arm; the unrestricted scale A/B is reported beside it and says by how
much.

    OMP_NUM_THREADS=3 python learner.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zs                                   # noqa: E402
import rowfeat as R                         # noqa: E402
import packet as PK                         # noqa: E402

EXP = zs.EXP
FOLDS = zs.FOLDS
ARMS = ["ridge", "lgb_small", "lgb_wide", "lgb_shipped", "lgb_shipped_wide"]
BASE = {"ridge": "ridge", "lgb_small": "lgb_small", "lgb_wide": "lgb_wide",
        "lgb_shipped": "lgb_small", "lgb_shipped_wide": "lgb_wide"}
SHIPPED = {"lgb_shipped", "lgb_shipped_wide"}
CROSSFITS = ["dayblock", "xmonth"]


def order_index(d: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    idx = pd.Series(np.arange(len(d)), index=d["MVT_ID_mvt"].astype("float64").to_numpy())
    take = idx.reindex(ids)
    if take.isna().any():
        raise SystemExit("a panel row has no feature row in the month frame")
    return take.to_numpy("int64")


def fold_state(fold: str, panel: pd.DataFrame) -> dict:
    air = panel["airport"].astype(str).to_numpy()
    reg = panel["regime"].astype(str).to_numpy()
    t = panel["t"].to_numpy("float64")
    ids = panel["MVT_ID_mvt"].to_numpy("float64")
    elig = (reg == zs.ELIGIBLE_REGIME) & ~np.isin(air, zs.HOLD_AIRPORTS)
    code, ncell = zs.cellcode(air, reg, t, zs.BIN)
    ei = np.flatnonzero(elig)
    cnt = np.bincount(code[ei], minlength=ncell)
    m0 = elig & (cnt[code] >= zs.MIN_CELL)
    ce, uce = pd.factorize(code[ei])
    ia, ib = zs.pairs_in_cells(ce, ids[ei], len(uce), zs.MIN_CELL, None)
    return {"fold": fold, "month": R.month_of(fold), "air": air, "t": t, "ids": ids,
            "elig": elig, "code": code, "ncell": ncell, "cnt": cnt, "m0": m0,
            "days": zs.V.day(t), "blk": zs.blocks_of(zs.V.day(t)),
            "r": panel["r"].to_numpy("float64"),
            "y": panel["y_true"].to_numpy("float64"),
            "p10": panel["p10"].to_numpy("float64"),
            "ia": ei[ia], "ib": ei[ib]}


def month_matrix(month: str, levels, with_shipped: bool, ids: np.ndarray | None = None):
    """The design for one month, in `ids` order when given."""
    d = zs.dep(month)
    X, cols = zs.design(d, levels)
    if with_shipped:
        W = zs.shipped(month).reindex(d["MVT_ID_mvt"].astype("float64").to_numpy())
        if W.isna().all(axis=1).any():
            raise SystemExit(f"{month}: a departure has no shipped feature row")
        X = np.hstack([X, W.to_numpy("float32")])
        cols = cols + list(W.columns)
    if ids is not None:
        X = X[order_index(d, ids)]
    return X, cols


def month_matrix_frame(frame: pd.DataFrame, levels, with_shipped: bool):
    """The same design built from an arbitrary departure frame (the ranking file).

    The frame must carry a `month` column when `with_shipped` is set, because the
    shipped block is cached per month.
    """
    X, cols = zs.design(frame, levels)
    if with_shipped:
        ids = frame["MVT_ID_mvt"].astype("float64").to_numpy()
        W = pd.concat([zs.shipped(m) for m in sorted(set(frame["month"]))])
        W = W.reindex(ids)
        if W.isna().all(axis=1).any():
            raise SystemExit("a row of the frame has no shipped feature row")
        X = np.hstack([X, W.to_numpy("float32")])
        cols = cols + list(W.columns)
    return X, cols


def fit_h(kind: str, Xfit, r_fit, code_fit, Xscore) -> np.ndarray:
    fc, ufc = pd.factorize(code_fit)
    nfc = len(ufc)
    rc = zs.project(np.asarray(r_fit, "float64"), fc, nfc)
    if kind == "ridge":
        return zs.ridge_h(np.asarray(Xfit, "float64"), rc, fc, nfc,
                          np.asarray(Xscore, "float64"))
    return zs.lgb_h(Xfit, rc, fc, nfc, Xscore, kind)


def h_for_fold(fold: str, S: dict, states: dict) -> dict:
    """Every arm's held-out `h` on one fold, under both cross-fittings."""
    out = {a: {} for a in ARMS}
    ids = S["ids"]
    local = zs.levels_of([zs.dep(S["month"])])
    Xr, _ = month_matrix(S["month"], local, False, ids)
    Xs, _ = month_matrix(S["month"], local, True, ids)
    ei = np.flatnonzero(S["elig"])
    for arm in ARMS:
        X = Xs if arm in SHIPPED else Xr
        h = np.zeros(len(ids))
        t0 = time.time()
        for b in range(zs.BLOCKS):
            fit, sc = ei[S["blk"][ei] != b], ei[S["blk"][ei] == b]
            h[sc] = fit_h(BASE[arm], X[fit], S["r"][fit], S["code"][fit], X[sc])
        out[arm]["dayblock"] = h
        print(f"    [{fold}] {arm:18s} dayblock {time.time() - t0:6.1f}s", flush=True)
    del Xr, Xs

    others = [g for g in FOLDS if g != fold]
    lv = zs.levels_of([zs.dep(states[g]["month"]) for g in others])
    for shipped in (False, True):
        arms = [a for a in ARMS if (a in SHIPPED) == shipped]
        Xf, _ = month_matrix(S["month"], lv, shipped, ids)
        parts, rs, cs = [], [], []
        for g in others:
            G = states[g]
            Xg, _ = month_matrix(G["month"], lv, shipped, G["ids"])
            eg = np.flatnonzero(G["elig"])
            parts.append(Xg[eg])
            rs.append(G["r"][eg])
            cs.append(G["code"][eg].astype("int64") + 1000000 * FOLDS.index(g))
            del Xg
        Xtr = np.vstack(parts)
        del parts
        rtr = np.concatenate(rs)
        ctr = np.concatenate(cs)
        for arm in arms:
            t0 = time.time()
            out[arm]["xmonth"] = fit_h(BASE[arm], Xtr, rtr, ctr, Xf)
            print(f"    [{fold}] {arm:18s} xmonth   {time.time() - t0:6.1f}s "
                  f"(fit rows {len(rtr):,})", flush=True)
        del Xtr, Xf
    return out


def score(S: dict, upd: np.ndarray, h: np.ndarray, amp: float, rng, checks: bool) -> dict:
    m, r, y, p10, ids = S["m0"], S["r"], S["y"], S["p10"], S["ids"]
    p_new = p10 + upd
    A = float((r[m] * upd[m]).mean())
    B = float((upd[m] ** 2).mean())
    v = 2 * r[m] * upd[m] - upd[m] ** 2
    pt, lo, hi = zs.boot_days(v, np.ones(int(m.sum())), S["days"][m], rng)
    ex = {"all_v10": float(zs.D.rmse(y, p10)), "all_new": float(zs.D.rmse(y, p_new)),
          "clean_v10": float(zs.D.rmse_clean(y, p10, ids, S["fold"])),
          "clean_new": float(zs.D.rmse_clean(y, p_new, ids, S["fold"]))}
    ia, ib = S["ia"], S["ib"]
    dr = r[ia] - r[ib]
    dh = amp * (h[ia] - h[ib])
    red = dr ** 2 - (dr - dh) ** 2
    ppt, plo, phi = zs.boot_days(red, np.ones(len(ia)), S["days"][ia], rng)
    rec = {"amplitude": round(amp, 4), "n_updated": int((upd != 0).sum()),
           "rmse_exact": ex,
           "gain_all": round(ex["all_v10"] - ex["all_new"], 4),
           "gain_clean": round(ex["clean_v10"] - ex["clean_new"], 4),
           "A_mean_r_d": round(A, 4), "B_mean_d2": round(B, 4),
           "mse_gain_s2": round(2 * A - B, 3), "ci": [round(lo, 3), round(hi, 3)],
           "unrestricted_scale": round(A / B, 4) if B > 0 else None,
           "multiday": PK.boot_blocks(v, S["days"][m], rng),
           "rms_update": round(float(np.sqrt((upd[m] ** 2).mean())), 4),
           "pair": {"n_pairs": int(len(ia)), "reduction_s2": round(ppt, 3),
                    "ci": [round(plo, 3), round(phi, 3)]}}
    if checks:
        s = np.bincount(S["code"][m], weights=upd[m], minlength=S["ncell"])
        scale = max(1.0, float(np.abs(upd[m]).sum()))
        cm, ucm = pd.factorize(S["code"][m])
        rw = zs.project(r[m], cm, len(ucm))
        full = float(((r[m] - upd[m]) ** 2).sum() - (r[m] ** 2).sum())
        within = float(((rw - upd[m]) ** 2).sum() - (rw ** 2).sum())
        rec.update({"max_abs_cell_sum": float(np.abs(s).max()),
                    "cell_sum_check": bool(np.abs(s).max() / scale < zs.TOL),
                    "dsse_full": round(full, 6), "dsse_within_cell": round(within, 6),
                    "dsse_rel_gap": float(abs(full - within) / max(abs(full), 1e-12))})
        rec["dsse_check"] = bool(rec["dsse_rel_gap"] < zs.TOL)
    return rec


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(zs.SEED)
    pan = zs.V.load()
    S = {f: fold_state(f, pan[f]) for f in FOLDS}
    H = {}
    for f in FOLDS:
        print(f"  [{f}] fitting", flush=True)
        H[f] = h_for_fold(f, S[f], S)
    print(f"  all fits done at {time.time() - t0:.0f}s", flush=True)

    d0 = {a: {c: {} for c in CROSSFITS} for a in ARMS}
    for a in ARMS:
        for c in CROSSFITS:
            for f in FOLDS:
                m0 = S[f]["m0"]
                cm, ucm = pd.factorize(S[f]["code"][m0])
                v = np.zeros(len(m0))
                v[m0] = zs.project(H[f][a][c][m0], cm, len(ucm))
                d0[a][c][f] = v

    amps = {}
    for a in ARMS:
        amps[a] = {}
        for f in FOLDS:
            num = den = 0.0
            for g in FOLDS:
                if g == f:
                    continue
                m = S[g]["m0"]
                num += float((S[g]["r"][m] * d0[a]["dayblock"][g][m]).sum())
                den += float((d0[a]["dayblock"][g][m] ** 2).sum())
            amps[a][f] = num / den if den > 0 else 0.0
        num = den = 0.0
        for g in FOLDS:
            m = S[g]["m0"]
            num += float((S[g]["r"][m] * d0[a]["dayblock"][g][m]).sum())
            den += float((d0[a]["dayblock"][g][m] ** 2).sum())
        amps[a]["pooled"] = num / den if den > 0 else 0.0

    recs, boards = {}, {}
    for a in ARMS:
        for c in CROSSFITS:
            key = f"{a}|{c}"
            recs[key] = {}
            for f in FOLDS:
                upd = amps[a][f] * d0[a][c][f]
                recs[key][f] = score(S[f], upd, H[f][a][c], amps[a][f], rng, True)
            ex = lambda k: {f: recs[key][f]["rmse_exact"][k] for f in FOLDS}
            av, an = zs.V.board(ex("all_v10")), zs.V.board(ex("all_new"))
            cv, cn = zs.V.board(ex("clean_v10")), zs.V.board(ex("clean_new"))
            boards[key] = {"all_row_v10": round(av, 4), "all_row_arm": round(an, 4),
                           "gain_all_row": round(av - an, 4),
                           "gain_all_row_exact": float(av - an),
                           "clean_v10": round(cv, 4), "clean_arm": round(cn, 4),
                           "gain_clean": round(cv - cn, 4)}
            print(f"  {key:28s} board {av:.4f} -> {an:.4f} ({av - an:+.4f} s)  "
                  f"folds " + " ".join(f"{recs[key][f]['gain_all']:+.4f}" for f in FOLDS)
                  + "  2A-B " + " ".join(
                      f"{recs[key][f]['mse_gain_s2']:+.1f}[{recs[key][f]['ci'][0]:+.0f},"
                      f"{recs[key][f]['ci'][1]:+.0f}]" for f in FOLDS), flush=True)

    out = {"step": "2 learner", "arms": ARMS, "crossfits": CROSSFITS,
           "amplitudes": {a: {k: round(v, 4) for k, v in amps[a].items()} for a in ARMS},
           "folds": recs, "board": boards,
           "cohort": {f: {"n_rows": int(len(S[f]["r"])),
                          "n_eligible": int(S[f]["elig"].sum()),
                          "n_scored_cohort": int(S[f]["m0"].sum()),
                          "n_cells": int(S[f]["ncell"]),
                          "n_pairs": int(len(S[f]["ia"]))} for f in FOLDS},
           "wall_s": round(time.time() - t0, 1)}
    (EXP / "analysis" / "learner.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(EXP / "analysis" / "h_vectors.npz",
                        **{f"{f}|{a}|{c}": H[f][a][c].astype("float32")
                           for f in FOLDS for a in ARMS for c in CROSSFITS})
    print(f"  written {EXP / 'analysis' / 'learner.json'} ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
