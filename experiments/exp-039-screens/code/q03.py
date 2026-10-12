"""Q03 -- is there flight-level error structure left inside an airport-window cell?

QUEUE.md's first test, run as written, and repaired against Sol round 1.

Eligibility is the `nm_bulk` ordinary joined regime outside EHAM and EDDF; every
other row gets exactly zero update.  Cells are airport x UTC 10-minute bin x
serving regime.  A fixed ridge is fitted on day blocks with feature and target
centring **inside the fitting split only**; at inference the update is

    d_i = h(X_i) - mean_cell(h(X))          (uniform weights, features only)

so no departure residual or label is read at scoring time.  `d` is added after the
serving mask and the ORIGINAL all-row residuals are scored.

Round 1 repairs:
  3  the pair diagnostic is now also read at the CROSS-MONTH amplitude, which is
     a function of the other two folds only, so the isolated readout and the
     headline arm use the same weight.
  4  the uncentered control now updates exactly the same rows as the centered arm
     (the same sparse-cell mask), and a `projection_off` ablation adds the SAME
     h at the SAME amplitude without the zero-sum projection, which is the
     control that isolates the projection and nothing else.
  2  the primary endpoint is ALL-ROW january/july at the ranking weights
     (`packet.primary`), with clean and december beside it; per-date
     contributions, input hashes and provenance are written by `packet`, and the
     headline carries a 1/2/3-day moving-block sensitivity.

    python q03.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import v10 as V  # noqa: E402
import rowfeat as R  # noqa: E402
import packet as PK  # noqa: E402

ROOT = V.ROOT
FOLDS = V.FOLDS
D = V.D

# ---- predeclared before any number was looked at ----------------------------
BIN = 600           # the cell's UTC window, seconds
MIN_CELL = 4        # QUEUE.md: "Predeclare a minimum of four eligible rows"
ALPHA = 100.0       # fixed ridge penalty on standardised features, every arm
BLOCKS = 5          # contiguous multi-day blocks, shared across airports
BOOT = 2000
SEED = 39
TOL = 1e-8
HOLD_AIRPORTS = ("EHAM", "EDDF")
ELIGIBLE_REGIME = "nm_bulk"


def ridge(X, y, alpha, intercept):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    b0 = y.mean() if intercept else 0.0
    A = Z.T @ Z
    A[np.diag_indices_from(A)] += alpha
    w = np.linalg.solve(A, Z.T @ (y - b0))
    return (lambda Xn: ((Xn - mu) / sd) @ w + b0), w


def blocks_of(days):
    u = np.unique(days)
    idx = {d: b for b, chunk in enumerate(np.array_split(u, BLOCKS)) for d in chunk}
    return np.array([idx[d] for d in days])


def cell_mean(v, code, n):
    s = np.bincount(code, weights=v, minlength=n)
    c = np.bincount(code, minlength=n).astype("float64")
    return (s / np.maximum(c, 1))[code]


def pairs_in_cells(code, ids, n, min_cell):
    """Fixed disjoint within-cell pairs: sort by cell then movement id, pair 1-2, 3-4…"""
    order = np.lexsort((ids, code))
    c = code[order]
    cnt = np.bincount(code, minlength=n)
    keep = cnt[c] >= min_cell
    order, c = order[keep], c[keep]
    start = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
    size = np.diff(np.r_[start, len(c)])
    rank = np.arange(len(c)) - np.repeat(start, size)
    ok = rank < 2 * np.repeat(size // 2, size)
    order, rank = order[ok], rank[ok]
    return order[rank % 2 == 0], order[rank % 2 == 1]


def boot_days(num, den, days, rng):
    codes, uniq = pd.factorize(days)
    k = len(uniq)
    n = np.bincount(codes, weights=num, minlength=k)
    d = np.bincount(codes, weights=den, minlength=k)
    draws = rng.integers(0, k, size=(BOOT, k))
    b = n[draws].sum(1) / d[draws].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(n.sum() / d.sum()), float(lo), float(hi)


def prepare(fold: str, panel: pd.DataFrame) -> dict:
    """Everything one fold contributes, before any amplitude is known."""
    F = R.features(R.departures(R.month_of(fold)))
    X = F.reindex(panel["MVT_ID_mvt"].to_numpy("float64"))
    if X.isna().any().any():
        raise SystemExit(f"{fold}: a panel row has no feature row in the month frame")
    cols = list(X.columns)
    X = X.to_numpy("float64")
    del F

    air = panel["airport"].astype(str).to_numpy()
    t = panel["t"].to_numpy("float64")
    elig = ((panel["regime"].to_numpy() == ELIGIBLE_REGIME)
            & ~np.isin(air, HOLD_AIRPORTS))
    key = pd.Series(air + "|" + panel["regime"].astype(str).to_numpy()
                    + "|" + (np.floor(t / BIN).astype("int64")).astype(str))
    code, uniq = pd.factorize(key)
    ncell = len(uniq)
    days = V.day(t)
    blk = blocks_of(days)
    r = panel["r"].to_numpy("float64")
    ids = panel["MVT_ID_mvt"].to_numpy("float64")
    ei = np.flatnonzero(elig)

    h_cen = np.zeros(len(panel))
    h_unc = np.zeros(len(panel))
    W = []
    for b in range(BLOCKS):
        fit, sc = ei[blk[ei] != b], ei[blk[ei] == b]
        fc, _ = pd.factorize(code[fit])
        nfc = fc.max() + 1
        Xc = X[fit] - np.column_stack([cell_mean(X[fit][:, j], fc, nfc)
                                       for j in range(X.shape[1])])
        rc = r[fit] - cell_mean(r[fit], fc, nfc)
        f_cen, w = ridge(Xc, rc, ALPHA, intercept=False)
        W.append(w)
        h_cen[sc] = f_cen(X[sc])
        f_unc, _ = ridge(X[fit], r[fit], ALPHA, intercept=True)
        h_unc[sc] = f_unc(X[sc])
    del X

    cnt = np.zeros(ncell)
    ca = np.bincount(code[ei], minlength=ncell)
    cnt[:len(ca)] = ca
    m0 = elig & (cnt[code] >= MIN_CELL)          # the scored cohort, every arm
    cm0, _ = pd.factorize(code[m0])
    d0 = np.zeros(len(panel))
    d0[m0] = h_cen[m0] - cell_mean(h_cen[m0], cm0, cm0.max() + 1)

    amp_blk = np.zeros(len(panel))
    amps = {}
    for b in range(BLOCKS):
        other, here = m0 & (blk != b), m0 & (blk == b)
        B_ = float((d0[other] ** 2).sum())
        a = float((r[other] * d0[other]).sum() / B_) if B_ > 0 else 0.0
        amps[str(b)], amp_blk[here] = round(a, 4), a

    a_pair, b_pair = pairs_in_cells(*(lambda s: (pd.factorize(code[s])[0], ids[s]))(ei),
                                    pd.factorize(code[ei])[0].max() + 1, MIN_CELL)
    ia, ib = ei[a_pair], ei[b_pair]

    Wm = np.array(W)
    nrm = np.linalg.norm(Wm, axis=1, keepdims=True)
    cos = (Wm / nrm) @ (Wm / nrm).T
    iu = np.triu_indices(BLOCKS, 1)
    top = np.argsort(-np.abs(Wm.mean(0)))[:8]

    return {"fold": fold, "panel": panel, "elig": elig, "m0": m0, "code": code,
            "ncell": ncell, "cnt": cnt, "days": days, "blk": blk, "r": r,
            "ids": ids, "h_cen": h_cen, "h_unc": h_unc, "d0": d0,
            "amp_blk": amp_blk, "amps": amps, "ia": ia, "ib": ib,
            "y": panel["y_true"].to_numpy("float64"),
            "p10": panel["p10"].to_numpy("float64"),
            "n_features": len(cols),
            "cohort_sha256": hashlib.sha256(np.sort(ids[elig]).tobytes()).hexdigest(),
            "coef_cosine_min": round(float(cos[iu].min()), 4),
            "coef_cosine_mean": round(float(cos[iu].mean()), 4),
            "top_coefficients": {cols[j]: round(float(Wm.mean(0)[j]), 3) for j in top}}


def cell_arm(S: dict, upd: np.ndarray, rng, note: str, checks: bool) -> dict:
    m = S["m0"]
    r, y, p10, ids = S["r"], S["y"], S["p10"], S["ids"]
    p_new = p10 + upd
    A = float((r[m] * upd[m]).mean()); B = float((upd[m] ** 2).mean())
    v = 2 * r[m] * upd[m] - upd[m] ** 2
    pt, lo, hi = boot_days(v, np.ones(int(m.sum())), S["days"][m], rng)
    ex = {"all_v10": float(D.rmse(y, p10)), "all_new": float(D.rmse(y, p_new)),
          "clean_v10": float(D.rmse_clean(y, p10, ids, S["fold"])),
          "clean_new": float(D.rmse_clean(y, p_new, ids, S["fold"]))}
    rec = {"note": note, "n_updated": int((upd != 0).sum()),
           "rmse_exact": ex,
           "rmse_all_v10": round(ex["all_v10"], 4),
           "rmse_all_new": round(ex["all_new"], 4),
           "rmse_clean_v10": round(ex["clean_v10"], 4),
           "rmse_clean_new": round(ex["clean_new"], 4),
           "A_mean_r_d": round(A, 4), "B_mean_d2": round(B, 4),
           "mse_gain_s2": round(2 * A - B, 3), "ci": [round(lo, 3), round(hi, 3)],
           "unrestricted_scale": round(A / B, 4) if B > 0 else None,
           "multiday": PK.boot_blocks(v, S["days"][m], rng)}
    rec["gain_all"] = round(ex["all_v10"] - ex["all_new"], 4)
    rec["gain_clean"] = round(ex["clean_v10"] - ex["clean_new"], 4)
    if checks:
        s = np.bincount(S["code"][m], weights=upd[m], minlength=S["ncell"])
        scale = max(1.0, float(np.abs(upd[m]).sum()))
        cm, _ = pd.factorize(S["code"][m])
        rw = r[m] - cell_mean(r[m], cm, cm.max() + 1)
        full = float(((r[m] - upd[m]) ** 2).sum() - (r[m] ** 2).sum())
        within = float(((rw - upd[m]) ** 2).sum() - (rw ** 2).sum())
        rec.update({"max_abs_cell_sum": float(np.abs(s).max()),
                    "cell_sum_check": bool(np.abs(s).max() / scale < TOL),
                    "dsse_full": round(full, 6), "dsse_within_cell": round(within, 6),
                    "dsse_rel_gap": float(abs(full - within) / max(abs(full), 1e-12))})
        rec["dsse_check"] = bool(rec["dsse_rel_gap"] < TOL)
    return rec


def pair_arm(S: dict, dh: np.ndarray, rng) -> dict:
    ia, ib, r = S["ia"], S["ib"], S["r"]
    dr = r[ia] - r[ib]
    d = dh[ia] - dh[ib]
    before, after = dr ** 2, (dr - d) ** 2
    pt, lo, hi = boot_days(before - after, np.ones(len(ia)), S["days"][ia], rng)
    A = float((dr * d).mean()); B = float((d ** 2).mean())
    return {"n_pairs": int(len(ia)),
            "mse_diff_before": round(float(before.mean()), 2),
            "mse_diff_after": round(float(after.mean()), 2),
            "reduction_s2": round(pt, 3), "ci": [round(lo, 3), round(hi, 3)],
            "A_2mean_dr_dh": round(A, 3), "B_mean_dh2": round(B, 3),
            "unrestricted_scale": round(A / B, 4) if B > 0 else None,
            "multiday": PK.boot_blocks(before - after, S["days"][ia], rng)}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    pan = V.load()
    S = {f: prepare(f, pan[f]) for f in FOLDS}

    # cross-month amplitude: a function of the OTHER TWO folds only
    cross_amp = {}
    for f in FOLDS:
        num = den = 0.0
        for g in FOLDS:
            if g == f:
                continue
            m = S[g]["m0"]
            num += float((S[g]["r"][m] * S[g]["d0"][m]).sum())
            den += float((S[g]["d0"][m] ** 2).sum())
        cross_amp[f] = num / den if den > 0 else 0.0

    recs, preds_headline = {}, {}
    pool = {k: [] for k in ("pair_full", "pair_block", "pair_cross", "pair_uncentered",
                            "cell_full", "cell_block", "cell_cross",
                            "cell_uncentered", "cell_projection_off")}
    for f in FOLDS:
        s = S[f]
        m0, d0 = s["m0"], s["d0"]
        a = cross_amp[f]
        upd = {}
        upd["full"] = d0.copy()
        upd["block"] = s["amp_blk"] * d0
        upd["cross"] = a * d0
        u = np.zeros(len(d0)); u[m0] = s["h_unc"][m0]
        upd["uncentered"] = u
        # the projection ablation: the SAME h at the SAME amplitude, un-projected
        u2 = np.zeros(len(d0)); u2[m0] = a * s["h_cen"][m0]
        upd["projection_off"] = u2
        cells = {
            "full": cell_arm(s, upd["full"], rng, "the zero-sum update at full amplitude", True),
            "block": cell_arm(s, upd["block"], rng, "one amplitude per day block, fitted on the other blocks", True),
            "cross": cell_arm(s, upd["cross"], rng, f"one amplitude {a:.4f} fitted on the OTHER TWO FOLDS", True),
            "uncentered": cell_arm(s, upd["uncentered"], rng, "a separately fitted uncentered ridge on the SAME rows", False),
            "projection_off": cell_arm(s, upd["projection_off"], rng, "the same h at the same cross-month amplitude with the zero-sum projection removed", False)}
        pairs = {
            "full": pair_arm(s, s["h_cen"], rng),
            "block": pair_arm(s, s["amp_blk"] * s["h_cen"], rng),
            "cross": pair_arm(s, a * s["h_cen"], rng),
            "uncentered": pair_arm(s, s["h_unc"], rng)}
        # strata on the cross-amplitude pair statistic
        dr = s["r"][s["ia"]] - s["r"][s["ib"]]
        d = a * (s["h_cen"][s["ia"]] - s["h_cen"][s["ib"]])
        red = dr ** 2 - (dr - d) ** 2
        sizes = s["cnt"][s["code"][s["ia"]]]
        strata = [{"cell_rows": f"{lo}-{hi if hi < 10**9 else '+'}",
                   "n_pairs": int(msk.sum()),
                   "reduction_s2": round(float(red[msk].mean()), 3)}
                  for lo, hi in ((4, 5), (6, 9), (10, 19), (20, 10 ** 9))
                  for msk in [(sizes >= lo) & (sizes <= hi)] if msk.sum()]

        recs[f] = {"fold": f, "n_rows": int(len(s["r"])),
                   "n_eligible": int(s["elig"].sum()),
                   "eligible_pct": round(100 * float(s["elig"].mean()), 2),
                   "n_scored_cohort": int(m0.sum()),
                   "n_cells": s["ncell"], "features": s["n_features"],
                   "cohort_sha256": s["cohort_sha256"],
                   "amplitude_per_block": s["amps"],
                   "amplitude_cross_month": round(a, 4),
                   "coef_cosine_min": s["coef_cosine_min"],
                   "coef_cosine_mean": s["coef_cosine_mean"],
                   "top_coefficients": s["top_coefficients"],
                   "cell": cells, "pair": pairs, "pair_strata_cross": strata}
        preds_headline[f] = s["p10"] + upd["cross"]
        for k, arm in (("cell_full", "full"), ("cell_block", "block"),
                       ("cell_cross", "cross"), ("cell_uncentered", "uncentered"),
                       ("cell_projection_off", "projection_off")):
            uu = upd[arm]
            pool[k].append(pd.DataFrame(
                {"v": 2 * s["r"][m0] * uu[m0] - uu[m0] ** 2,
                 "day": f + "|" + s["days"][m0]}))
        for k, dh in (("pair_full", s["h_cen"]), ("pair_block", s["amp_blk"] * s["h_cen"]),
                      ("pair_cross", a * s["h_cen"]), ("pair_uncentered", s["h_unc"])):
            drr = s["r"][s["ia"]] - s["r"][s["ib"]]
            dd = dh[s["ia"]] - dh[s["ib"]]
            pool[k].append(pd.DataFrame({"v": drr ** 2 - (drr - dd) ** 2,
                                         "day": f + "|" + s["days"][s["ia"]]}))
        for arm in ("cross", "uncentered"):
            PK.dates("q03", arm, f, s["days"], s["y"], s["p10"], s["p10"] + upd[arm])
        PK.rows("q03", "cross", f, s["ids"], s["days"], s["y"], s["p10"],
                s["p10"] + upd["cross"])
        print(f"[{f}] eligible {recs[f]['n_eligible']:,}, scored cohort "
              f"{recs[f]['n_scored_cohort']:,}, {s['ncell']:,} cells, "
              f"cross-month amplitude {a:.4f}", flush=True)
        for k in ("full", "block", "cross", "uncentered"):
            p = pairs[k]
            print(f"    PAIR {k:14s} {p['reduction_s2']:+8.2f} "
                  f"[{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] s^2  scale {p['unrestricted_scale']}")
        for k in ("full", "block", "cross", "uncentered", "projection_off"):
            c = cells[k]
            print(f"    CELL {k:14s} all-row {c['gain_all']:+.4f} s  clean "
                  f"{c['gain_clean']:+.4f} s  2A-B {c['mse_gain_s2']:+8.2f} "
                  f"[{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]  n={c['n_updated']:,}")
        c = cells["cross"]
        print(f"    checks: cell sums zero {c['cell_sum_check']} (max |sum| "
              f"{c['max_abs_cell_sum']:.3g}); full == within-cell {c['dsse_check']} "
              f"(rel gap {c['dsse_rel_gap']:.3g})", flush=True)

    pooled = {}
    for k, frames in pool.items():
        z = pd.concat(frames, ignore_index=True)
        pt, lo, hi = boot_days(z["v"].to_numpy(), np.ones(len(z)), z["day"].to_numpy(), rng)
        pooled[k] = {"n": int(len(z)), "point_s2": round(pt, 3),
                     "ci": [round(lo, 3), round(hi, 3)], "days": int(z["day"].nunique()),
                     "multiday": PK.boot_blocks_pooled(
                         [(d["v"].to_numpy(), d["day"].to_numpy()) for d in frames], rng)}

    boards = {}
    for arm in ("full", "block", "cross", "uncentered", "projection_off"):
        ex = lambda k: {f: recs[f]["cell"][arm]["rmse_exact"][k] for f in FOLDS}
        cl_v10, cl_new = V.board(ex("clean_v10")), V.board(ex("clean_new"))
        ar_v10, ar_new = V.board(ex("all_v10")), V.board(ex("all_new"))
        boards[arm] = {"clean": round(cl_new, 4), "all_row": round(ar_new, 4),
                       "gain_clean": round(cl_v10 - cl_new, 4),
                       "gain_all_row": round(ar_v10 - ar_new, 4)}

    out = {"screen": "Q03",
           "predeclared": {"bin_s": BIN, "min_cell": MIN_CELL, "alpha": ALPHA,
                           "blocks": BLOCKS, "seed": SEED, "boot": BOOT,
                           "eligible_regime": ELIGIBLE_REGIME,
                           "hold_airports": list(HOLD_AIRPORTS)},
           "primary": PK.primary(pan, preds_headline,
                                 "headline arm: the zero-sum update at the "
                                 "cross-month amplitude"),
           "folds": recs, "pooled": pooled, "board": boards,
           "wall_s": round(time.time() - t0, 1)}
    (EXP / "analysis" / "q03.json").write_text(json.dumps(out, indent=2))
    p = PK.write("q03", out, pan)
    pr = out["primary"]
    print("\n  pooled day-clustered bootstrap over all three folds:")
    for k, v in pooled.items():
        print(f"    {k:20s} {v['point_s2']:+9.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
              f"  1/2/3-day {v['multiday']['ci_block_1d']} {v['multiday']['ci_block_2d']} "
              f"{v['multiday']['ci_block_3d']}")
    print(f"\n  PRIMARY (all-row january/july board): {pr['board_all_row']['v10']:.4f} -> "
          f"{pr['board_all_row']['arm']:.4f} ({pr['board_all_row']['gain']:+.4f} s)")
    print(f"  secondary clean board: {pr['board_clean']['v10']:.4f} -> "
          f"{pr['board_clean']['arm']:.4f} ({pr['board_clean']['gain']:+.4f} s)")
    print(f"  december all-row {pr['december_all_row']['v10']:.4f} -> "
          f"{pr['december_all_row']['arm']:.4f} ({pr['december_all_row']['gain']:+.4f} s)")
    print(f"  written: {p}  ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
