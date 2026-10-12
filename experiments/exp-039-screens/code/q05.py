"""Q05 -- observable clock and reporting patterns, audited against v10's residual.

The queue's first test, in its own order.

1  INVENTORY   what the shipping stack already carries.  Written in the report,
               not computed here.
2  IDENTITY    on labelled audit rows, y - (T - AOBT) = -(B - AOBT) row by row.
               This is what makes the rest interpretable: a phase that lives in
               the take-off anchor cannot appear in the anchored error.
3  PHASES      the 5/15/60-second quantisation of every clock, and the effect of
               the OBSERVABLE phase, equality and date-gap flags reported
               SEPARATELY on y, on B - AOBT and on the current residual r.
4  HEADS       phase-only, the existing timing controls at matched capacity, and
               one shrunk fingerprint head, all on blocked day splits, scored as
               2A - B against v10 on held-out blocks.
5  WITNESS     the arrival reporting witness: the airport-day schedule/in-block
               coincidence rate among MISSING-TYPE arrivals, against the shipped
               all-arrival rate `prc.tail_features._arrival_day_rate`.

B (`BLOCK_TIME_UTC_mvt`) and y are read ONLY for the audit tables of stage 2, 3
and 5 and never enter a predictor; every head is fitted on features from
`rowfeat`, which is built on the ranking mask.

    python q05.py
"""
from __future__ import annotations

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
import donor as Dn  # noqa: E402
import packet as PK  # noqa: E402

ROOT = V.ROOT
FOLDS = V.FOLDS
D = V.D

SEED = 39
BOOT = 2000
BLOCKS = 5
SHRINK = 200.0        # empirical-Bayes constant for every shrunk group mean
PH_TOL = 60.0         # `prc.tail_features._arrival_day_rate`'s own coincidence tolerance
HOLD_AIRPORTS = ("EHAM", "EDDF")
CLOCKS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
          "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]


def blocks_of(days: np.ndarray) -> np.ndarray:
    u = np.unique(days)
    idx = {d: b for b, chunk in enumerate(np.array_split(u, BLOCKS)) for d in chunk}
    return np.array([idx[d] for d in days])


def shrunk_oof(g: np.ndarray, r: np.ndarray, blk: np.ndarray) -> np.ndarray:
    """Out-of-block shrunk group mean of `r`: sum/(n + SHRINK), block b excluded."""
    code, uniq = pd.factorize(g)
    k = len(uniq)
    tot = np.bincount(code, weights=r, minlength=k)
    cnt = np.bincount(code, minlength=k).astype("float64")
    out = np.zeros(len(r))
    for b in range(BLOCKS):
        m = blk == b
        t2 = tot - np.bincount(code[m], weights=r[m], minlength=k)
        c2 = cnt - np.bincount(code[m], minlength=k)
        out[m] = (t2 / (c2 + SHRINK))[code[m]]
    return out


def boot_days(v: np.ndarray, days: np.ndarray, rng):
    codes, uniq = pd.factorize(days)
    k = len(uniq)
    num = np.bincount(codes, weights=v, minlength=k)
    cnt = np.bincount(codes, minlength=k).astype("float64")
    draws = rng.integers(0, k, size=(BOOT, k))
    b = num[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(num.sum() / cnt.sum()), float(lo), float(hi)


def audit_frame(fold: str, panel: pd.DataFrame) -> pd.DataFrame:
    """The month's departures with the label-side clocks kept for the audit only."""
    raw = D.load_month(R.month_of(fold))
    dep = raw[raw["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    del raw
    f = pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy("float64")})
    for c in CLOCKS:
        f[c] = Dn.epoch(dep[c])
    f["y_raw"] = dep["TAXITIME_SEC_mvt"].to_numpy("float64")
    f["operator"] = dep["AIRCRAFT_OPERATOR_flt"].astype("string").fillna("<na>").to_numpy()
    f["stand"] = dep["STAND_mvt"].astype("string").fillna("<na>").to_numpy()
    m = panel.merge(f, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if len(m) != len(panel) or m["MVT_TIME_UTC_mvt"].isna().any():
        raise SystemExit(f"{fold}: the audit join lost rows")
    return m


def phase_flags(m: pd.DataFrame) -> pd.DataFrame:
    """Observable phase, equality and date-gap flags -- no B and no y."""
    T, S = m["MVT_TIME_UTC_mvt"], m["SCHED_TIME_UTC_mvt"]
    A, L = m["AOBT_3_flt"], m["LOBT_flt"]
    E, I = m["EOBT_1_flt"], m["IOBT_flt"]
    f = pd.DataFrame(index=m.index)
    for name, x in (("t", T), ("sched", S), ("aobt", A), ("lobt", L),
                    ("eobt", E), ("iobt", I)):
        for k in (5, 15, 60):
            f[f"{name}_mod{k}"] = np.where(np.isfinite(x), np.mod(x, k), -1)
    f["eq_aobt_lobt"] = (np.abs(A - L) < 0.5).fillna(False)
    f["eq_aobt_sched"] = (np.abs(A - S) < 0.5).fillna(False)
    f["eq_aobt_eobt"] = (np.abs(A - E) < 0.5).fillna(False)
    f["eq_lobt_sched"] = (np.abs(L - S) < 0.5).fillna(False)
    f["eq_lobt_iobt"] = (np.abs(L - I) < 0.5).fillna(False)
    f["eq_eobt_lobt"] = (np.abs(E - L) < 0.5).fillna(False)
    day = lambda x: np.floor(x / 86400.0)
    f["dategap_sched_t"] = (day(S) != day(T)).fillna(False)
    f["dategap_aobt_t"] = (day(A) != day(T)).fillna(False)
    f["dategap_lobt_aobt"] = (day(L) != day(A)).fillna(False)
    return f


def run_fold(fold: str, panel: pd.DataFrame, rng, preds_out: dict) -> dict:
    m = audit_frame(fold, panel)
    out = {"fold": fold, "month": R.month_of(fold), "n_rows": int(len(m))}
    T = m["MVT_TIME_UTC_mvt"].to_numpy()
    B = m["BLOCK_TIME_UTC_mvt"].to_numpy()
    A = m["AOBT_3_flt"].to_numpy()
    y = m["y_true"].to_numpy("float64")
    r = m["r"].to_numpy("float64")
    joined = np.isfinite(A)

    # ---- 2. the identity ---------------------------------------------------
    lab = np.isfinite(y) & np.isfinite(B)
    out["identity"] = {
        "n_audit_rows": int(lab.sum()),
        "max_abs_y_minus_T_minus_B": float(np.abs(y[lab] - (T[lab] - B[lab])).max()),
        "n_joined": int((joined & lab).sum()),
        "max_abs_resid_identity": float(np.abs(
            (y[lab & joined] - (T[lab & joined] - A[lab & joined]))
            + (B[lab & joined] - A[lab & joined])).max()),
        "raw_y_equals_panel_y": float(np.nanmax(np.abs(
            m["y_raw"].to_numpy("float64") - y)))}

    # ---- 3. quantisation and the three separate phase readouts -------------
    q = {}
    for c in CLOCKS:
        x = m[c].to_numpy()
        ok = np.isfinite(x)
        q[c] = {f"mod{k}_zero_pct": round(100 * float((np.mod(x[ok], k) == 0).mean()), 2)
                for k in (5, 15, 60)}
        q[c]["present_pct"] = round(100 * float(ok.mean()), 2)
    out["quantisation"] = q

    F = phase_flags(m)
    bmi = B - A                         # B - AOBT, the audit quantity
    anch = y - (T - A)                  # the RAW take-off-anchor error = -(B - AOBT)
    tab = []
    for col in [c for c in F.columns if c.startswith(("eq_", "dategap_"))] + \
               ["aobt_mod5", "aobt_mod15", "aobt_mod60",
                "sched_mod60", "t_mod5", "t_mod15", "t_mod60"]:
        v = F[col].to_numpy()
        if "_mod" in col:
            v = (v == 0)
        g = np.asarray(v, bool) & joined & lab
        h = (~np.asarray(v, bool)) & joined & lab
        if g.sum() < 50 or h.sum() < 50:
            continue
        tab.append({"flag": col, "n_true": int(g.sum()), "share_pct":
                    round(100 * float(g.sum() / (g.sum() + h.sum())), 2),
                    "mean_y_true": round(float(y[g].mean()), 1),
                    "mean_y_false": round(float(y[h].mean()), 1),
                    "mean_BminusAOBT_true": round(float(np.nanmean(bmi[g])), 1),
                    "mean_BminusAOBT_false": round(float(np.nanmean(bmi[h])), 1),
                    "mean_r_true": round(float(r[g].mean()), 2),
                    "mean_r_false": round(float(r[h].mean()), 2),
                    "delta_y": round(float(y[g].mean() - y[h].mean()), 1),
                    "delta_BminusAOBT": round(
                        float(np.nanmean(bmi[g]) - np.nanmean(bmi[h])), 1),
                    # the RAW take-off-anchor error, y - (T - AOBT) = -(B - AOBT).
                    # Sol round 1, blocking finding 9: this is the column that
                    # says whether the ANCHOR absorbs a phase, and it does not.
                    "delta_anchor_error": round(
                        float(np.nanmean(anch[g]) - np.nanmean(anch[h])), 1),
                    "delta_r": round(float(r[g].mean() - r[h].mean()), 2)})
    out["phase_table"] = tab

    # ---- 4. three heads on blocked day splits ------------------------------
    days = V.day(m["t"].to_numpy("float64"))
    blk = blocks_of(days)
    elig = (np.isin(m["regime"].to_numpy(), ("nm_bulk", "nm_late"))
            & ~np.isin(m["airport"].astype(str).to_numpy(), HOLD_AIRPORTS))
    flags = (F["eq_aobt_lobt"].astype(str) + F["eq_aobt_sched"].astype(str)
             + F["eq_lobt_sched"].astype(str) + F["eq_lobt_iobt"].astype(str)
             + F["dategap_sched_t"].astype(str)).to_numpy()
    fp = (pd.Series(flags) + (F["aobt_mod60"] == 0).astype(str)
          + (F["t_mod60"] == 0).astype(str)
          + (F["sched_mod60"] == 0).astype(str)).to_numpy()
    # Sol round 1, blocking finding 8: the whole-minute flag is not the 5/15/60 s
    # phase.  These are the residue classes themselves, crossed between the AOBT
    # clock and the take-off clock.
    phase_k = {k: (F[f"aobt_mod{k}"].astype(int).astype(str) + "|"
                   + F[f"t_mod{k}"].astype(int).astype(str)).to_numpy()
               for k in (5, 15, 60)}
    dta = m["d_takeoff_aobt"].to_numpy("float64")
    dal = m["d_aobt_lobt"].to_numpy("float64")
    ctrl = (pd.qcut(pd.Series(dta), 10, labels=False, duplicates="drop").astype(str)
            + "|" + pd.qcut(pd.Series(dal).fillna(-1e9), 10, labels=False,
                            duplicates="drop").astype(str)).to_numpy()
    apt = m["airport"].astype(str).to_numpy()
    opb = pd.Series(m["operator"].to_numpy())
    top = set(opb.value_counts().index[:40])
    opb = np.where(opb.isin(top), opb, "<other>")

    apt_op = np.char.add(np.char.add(apt, "|"), opb.astype(str))
    # A sham interaction at the fingerprint head's own cardinality: airport x
    # operator crossed with a FIXED RANDOM label chosen so the sham head has as
    # many populated groups as the fingerprint head.  Any gain here is
    # cardinality, not the phase (Sol round 1, blocking finding 8).
    n_fp = int(pd.Series(np.char.add(np.char.add(apt_op[elig], "|"),
                                     fp[elig])).nunique())
    n_ao = int(pd.Series(apt_op[elig]).nunique())
    sham_levels = max(int(round(n_fp / max(n_ao, 1))), 2)
    sham = np.random.default_rng(SEED).integers(
        0, sham_levels, size=len(m)).astype(str)
    groups = {"phase_only": fp,
              "phase_5s": phase_k[5],
              "phase_15s": phase_k[15],
              "phase_60s": phase_k[60],
              "flags_only": flags,
              "timing_controls": ctrl,
              # the control the fingerprint head needs: the same airport x
              # operator lattice with NO phase class in it.  Without this the
              # fingerprint head's gain cannot be attributed to the phase.
              "airport_operator_only": apt_op,
              "fingerprint_head": np.char.add(np.char.add(apt_op, "|"), fp),
              "sham_same_cardinality": np.char.add(np.char.add(apt_op, "|"), sham)}
    ei = np.flatnonzero(elig)
    out["heads"] = {}
    preds = {}
    for name, g in groups.items():
        h = np.zeros(len(m))
        h[ei] = shrunk_oof(g[ei], r[ei], blk[ei])
        preds[name] = h
        out["heads"][name] = score_head(h, r, y, m, ei, days, fold, rng,
                                        n_groups=int(pd.Series(g[ei]).nunique()))
        preds_out.setdefault(name, {})[fold] = m["p10"].to_numpy("float64") + h
        PK.dates("q05", name, fold, days, y, m["p10"].to_numpy("float64"),
                 m["p10"].to_numpy("float64") + h)
    # phase beyond the existing timing controls, at the same shrinkage
    res = r - preds["timing_controls"]
    h2 = np.zeros(len(m))
    h2[ei] = shrunk_oof(fp[ei], res[ei], blk[ei])
    out["heads"]["phase_beyond_controls"] = score_head(
        preds["timing_controls"] + h2, r, y, m, ei, days, fold, rng,
        n_groups=int(pd.Series(fp[ei]).nunique()))
    out["heads"]["phase_beyond_controls"]["increment_over_controls_s2"] = round(
        out["heads"]["phase_beyond_controls"]["mse_gain_s2"]
        - out["heads"]["timing_controls"]["mse_gain_s2"], 3)
    # phase beyond airport x operator, at the same shrinkage: the fingerprint
    # head's own increment, which is the number the advance criterion needs.
    res2 = r - preds["airport_operator_only"]
    h3 = np.zeros(len(m))
    h3[ei] = shrunk_oof(fp[ei], res2[ei], blk[ei])
    out["heads"]["phase_beyond_airport_operator"] = score_head(
        preds["airport_operator_only"] + h3, r, y, m, ei, days, fold, rng,
        n_groups=int(pd.Series(fp[ei]).nunique()))
    out["heads"]["phase_beyond_airport_operator"]["increment_over_apt_op_s2"] = round(
        out["heads"]["phase_beyond_airport_operator"]["mse_gain_s2"]
        - out["heads"]["airport_operator_only"]["mse_gain_s2"], 3)
    out["heads"]["fingerprint_head"]["increment_over_apt_op_s2"] = round(
        out["heads"]["fingerprint_head"]["mse_gain_s2"]
        - out["heads"]["airport_operator_only"]["mse_gain_s2"], 3)

    # ---- 5. the arrival reporting witness ----------------------------------
    out["witness"] = witness(fold, m, rng)
    return out


def score_head(h: np.ndarray, r, y, m, ei, days, fold, rng, n_groups: int) -> dict:
    p_new = m["p10"].to_numpy("float64") + h
    ids = m["MVT_ID_mvt"].to_numpy("float64")
    v = 2 * r[ei] * h[ei] - h[ei] ** 2
    pt, lo, hi = boot_days(v, days[ei], rng)
    A = float((r[ei] * h[ei]).mean()); Bq = float((h[ei] ** 2).mean())
    ex = {"all_v10": float(D.rmse(y, m["p10"])), "all_new": float(D.rmse(y, p_new)),
          "clean_v10": float(D.rmse_clean(y, m["p10"], ids, fold)),
          "clean_new": float(D.rmse_clean(y, p_new, ids, fold))}
    c0, c1 = ex["clean_v10"], ex["clean_new"]
    return {"n_groups": n_groups, "n_scored": int(len(ei)),
            "rmse_exact": ex,
            "rmse_all_v10": round(ex["all_v10"], 4),
            "rmse_all_new": round(ex["all_new"], 4),
            "multiday": PK.boot_blocks(v, days[ei], rng),
            "A_mean_r_h": round(A, 3), "B_mean_h2": round(Bq, 3),
            "mse_gain_s2": round(pt, 3), "ci": [round(lo, 3), round(hi, 3)],
            "unrestricted_scale": round(A / Bq, 4) if Bq > 0 else None,
            "rmse_clean_v10": round(c0, 4), "rmse_clean_new": round(c1, 4),
            "gain_clean": round(c0 - c1, 4)}


def witness(fold: str, m: pd.DataFrame, rng) -> dict:
    """Airport-day in-block/schedule coincidence among MISSING-TYPE arrivals.

    `prc.tail_features._arrival_day_rate` already ships the same rate over ALL
    arrivals.  The queue's conditional extension restricts it to arrivals whose
    `AIRCRAFT_TYPE_mvt` is null, which is an input-only change; both are computed
    here on the same airport-days and compared on the noac departures.
    """
    raw = D.load_month(R.month_of(fold))
    arr = raw[raw["PHASE_mvt"] == "ARR"]
    b = Dn.epoch(arr["BLOCK_TIME_UTC_mvt"])
    s = Dn.epoch(arr["SCHED_TIME_UTC_mvt"])
    ph = (np.abs(b - s) <= PH_TOL)
    key = pd.MultiIndex.from_arrays([arr["ADES_mvt"].astype(str).to_numpy(),
                                     np.floor(Dn.epoch(arr["MVT_TIME_UTC_mvt"]) / 86400)])
    miss = arr["AIRCRAFT_TYPE_mvt"].isna().to_numpy()
    all_rate = pd.Series(ph, index=key).groupby(level=[0, 1]).mean()
    all_n = pd.Series(ph, index=key).groupby(level=[0, 1]).size()
    mr = pd.Series(ph[miss], index=key[miss]).groupby(level=[0, 1]).mean()
    mn = pd.Series(ph[miss], index=key[miss]).groupby(level=[0, 1]).size()
    del raw, arr

    dk = pd.MultiIndex.from_arrays([m["airport"].astype(str).to_numpy(),
                                    np.floor(m["t"].to_numpy("float64") / 86400)])
    a_all = all_rate.reindex(dk).to_numpy()
    n_all = all_n.reindex(dk).fillna(0).to_numpy()
    a_ms = mr.reindex(dk).to_numpy()
    n_ms = mn.reindex(dk).fillna(0).to_numpy()
    gm = float(np.nanmean(ph))
    a_ms_shrunk = np.where(np.isfinite(a_ms),
                           (a_ms * n_ms + gm * 10.0) / (n_ms + 10.0), gm)

    noac = (m["regime"].to_numpy() == "noac")
    B_ = m["BLOCK_TIME_UTC_mvt"].to_numpy()
    S_ = m["SCHED_TIME_UTC_mvt"].to_numpy()
    plc = (np.abs(B_ - S_) <= PH_TOL)
    r = m["r"].to_numpy("float64")
    out = {"n_noac": int(noac.sum()),
           "arrival_days_with_missing_type": int(np.isfinite(mr).sum()),
           "median_missing_type_arrivals_per_day": float(np.nanmedian(mn.to_numpy()))
           if len(mn) else 0.0,
           "noac_rows_with_missing_type_support": int(np.isfinite(a_ms[noac]).sum()),
           "placeholder_rate_noac": round(float(np.nanmean(plc[noac])), 4),
           "all_arrival_rate_mean": round(gm, 4)}
    if noac.sum() >= 20:
        ok = noac & np.isfinite(plc)
        for tag, x in (("all_arrivals", a_all), ("missing_type_shrunk", a_ms_shrunk)):
            z = np.isfinite(x) & ok
            if z.sum() >= 20:
                out[f"corr_{tag}_vs_placeholder"] = round(
                    float(np.corrcoef(x[z], plc[z].astype(float))[0, 1]), 4)
                out[f"corr_{tag}_vs_residual"] = round(
                    float(np.corrcoef(x[z], r[z])[0, 1]), 4)
                out[f"n_{tag}"] = int(z.sum())
        z = ok & np.isfinite(a_all) & np.isfinite(a_ms_shrunk)
        if z.sum() >= 30:
            X = np.column_stack([np.ones(z.sum()), a_all[z], a_ms_shrunk[z]])
            w, *_ = np.linalg.lstsq(X, plc[z].astype(float), rcond=None)
            X0 = X[:, :2]
            w0, *_ = np.linalg.lstsq(X0, plc[z].astype(float), rcond=None)
            e1 = plc[z] - X @ w
            e0 = plc[z] - X0 @ w0
            out["incremental_r2_missing_type"] = round(
                float(1 - (e1 ** 2).sum() / (e0 ** 2).sum()), 5)
            out["coef_missing_type_given_all"] = round(float(w[2]), 4)
    return out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    pan = V.load()
    recs, preds_out = {}, {}
    for f in FOLDS:
        recs[f] = run_fold(f, pan[f], rng, preds_out)
        c = recs[f]
        print(f"[{f}] identity: |y-(T-B)| max {c['identity']['max_abs_y_minus_T_minus_B']:g}, "
              f"|(y-(T-AOBT))+(B-AOBT)| max {c['identity']['max_abs_resid_identity']:g} "
              f"on {c['identity']['n_joined']:,} joined audit rows", flush=True)
        for h, v in c["heads"].items():
            print(f"    {h:24s} groups {v['n_groups']:6,d}  2A-B {v['mse_gain_s2']:+8.2f} "
                  f"[{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] s^2  clean {v['gain_clean']:+.4f} s")
        w = c["witness"]
        print(f"    witness: noac {w['n_noac']}, placeholder rate "
              f"{w['placeholder_rate_noac']}, corr(all-arrival rate) "
              f"{w.get('corr_all_arrivals_vs_placeholder')}, corr(missing-type) "
              f"{w.get('corr_missing_type_shrunk_vs_placeholder')}, incremental R2 "
              f"{w.get('incremental_r2_missing_type')}", flush=True)
    board = {}
    for h in recs[FOLDS[0]]["heads"]:
        ex = lambda k: {f: recs[f]["heads"][h]["rmse_exact"][k] for f in FOLDS}
        cl_v10, cl_new = V.board(ex("clean_v10")), V.board(ex("clean_new"))
        ar_v10, ar_new = V.board(ex("all_v10")), V.board(ex("all_new"))
        board[h] = {"new": round(cl_new, 4), "gain_board": round(cl_v10 - cl_new, 4),
                    "all_row": round(ar_new, 4),
                    "gain_all_row": round(ar_v10 - ar_new, 4)}
    out = {"screen": "Q05", "predeclared": {"shrink": SHRINK, "blocks": BLOCKS,
                                            "ph_tol_s": PH_TOL, "seed": SEED,
                                            "boot": BOOT,
                                            "hold_airports": list(HOLD_AIRPORTS)},
           "primary": PK.primary(pan, preds_out["phase_only"],
                                 "headline arm: the phase-only head"),
           "folds": recs, "board": board, "wall_s": round(time.time() - t0, 1)}
    p = EXP / "analysis" / "q05.json"
    p.write_text(json.dumps(out, indent=2))
    PK.write("q05", out, pan)
    pr = out["primary"]
    print(f"\n  PRIMARY (all-row january/july board), phase only: "
          f"{pr['board_all_row']['v10']:.4f} -> {pr['board_all_row']['arm']:.4f} "
          f"({pr['board_all_row']['gain']:+.4f} s)")
    print("  every head, clean board against v10's 257.0248 and all-row:")
    for h, v in board.items():
        print(f"    {h:28s} clean {v['new']:.4f} ({v['gain_board']:+.4f} s)  "
              f"all-row {v['all_row']:.4f} ({v['gain_all_row']:+.4f} s)")
    print(f"  written: {p}  ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
