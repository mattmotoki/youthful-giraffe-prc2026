"""M25 (unit 11): the served form of the LIRF rows without a plan, v3 = v2 with C43, C44 and C45.

    .venv/bin/python experiments/exp-167-target/code/noplan_form_v3.py --tag 001
    .venv/bin/python experiments/exp-167-target/code/noplan_form_v3.py --tag <new> --base <entry.parquet>

Row order (a row takes the first part that holds it); v1 and v2 are not changed:
    c45   a far row (later UTC date than SCHED, or take-off more than 10.6 h after SCHED) in the
          forced zone: the wrapped block time (take-off - 0.3 h, on the date nearest to SCHED) is
          more than 10.55 h from SCHED.  LIRF keeps no block time that far from SCHED: the
          placeholder point MVT - SCHED (C45, `fresh`).
    c20   the other far rows (open and edge zones; later-date rows under 10.6 h): the base
          clipped between the two points [MVT - SCHED, 86,400 + taxi] (v1).  No bet in the form.
    c30   a decided link state: the plan of the arrival row (v1).
    c39   the airline's arrival default share of the month (v2).
    c32   0 < MVT - SCHED < 3,900 s: placeholder or local-offset shift by airline prefix (v1).
    c43   an r1 row whose destination is held in that hour (share above 0.25 of the joined
          departures of the ten airports to the same destination within 3 h that left more than
          30 min after their last filed time; 4 flights and more), take-off at most 4 h after
          SCHED, class probability 0.7 or less: real point + 0.1 x (MVT - SCHED - real point).
    r1    the other rows without type: max(60, 0.75 base + 0.25 mixture) (v1), with the class
          model's new columns when they pass their lines: turn and local_delay (C44), dest_held
          and dest_n (C43 as an input).
    base  everything else.

The lines of `docs/LEVERS.md` (60b7588d) decide what enters:
    C44         fold G of the form with the two columns, over the form without them, +50 or more
    C43 input   fails only if the leave-one-month-out AUC on the r1 rows rises by less than 0.005
                AND the fold G rises by less than +50
    C43 part    at most 1 cell row in 6 is a placeholder on the ten other months

The bet-arm file holds the open and edge far rows with three weights on the rollover point
(band arm C20 (b), `fresh`'s zone rule, `levers`' advice as a rule).  It is never an entry.
Write-once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C
import inbound_sign as I
import local_offset as LO
import noac_candidate as N
import noac_candidate_r1 as R1
import noplan_form as NF
import noplan_form_v2 as V2
import plan_link as P
import regime_signal as S

sys.path.insert(0, str(C.ROOT))
from prc import gscore as GS  # noqa: E402

FAR_H, LIM_FORCED, LIM_OPEN, BLOCK_BEFORE_H = 10.6, 10.55, 10.2, 0.3
C44, C43 = ["turn", "local_delay"], ["dest_held", "dest_n"]
SETS = {"v2": [], "c44": C44, "c43": C43, "both": C44 + C43}
HELD, CELL_MS, CELL_P, Q = 0.25, 14_400.0, 0.7, 0.10
LINE_G, LINE_AUC, LINE_RATE = 50.0, 0.005, 1.0 / 6.0
PARTS = ("c45", "c20", "c30", "c39", "c32", "c43", "r1", "base")
ARMS = ("band", "zone", "levers")


def zones(ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zone of a row by its lateness (`fresh`, exp-168 `u4_far2.py`), and the wrapped block time in hours."""
    m_h = ms / 3600.0
    tb = m_h - BLOCK_BEFORE_H
    wrapped = tb - 24.0 * np.round(tb / 24.0)
    z = np.where(m_h <= FAR_H, "normal", np.where(np.abs(wrapped) > LIM_FORCED, "forced", np.where(np.abs(wrapped) > LIM_OPEN, "edge", "open")))
    return z, wrapped


def extra_inputs(t: pd.DataFrame) -> pd.DataFrame:
    """turn, local_delay, dest_held, dest_n for the LIRF departures of a frame with `apt` (label-free)."""
    L = t[t.apt == "LIRF"]
    dep = L[L.PHASE_mvt == "DEP"].copy()
    dep[C.ID] = dep[C.ID].astype("int64")
    ds = dep[dep.STAND_mvt.notna()].sort_values("MVT_TIME_UTC_mvt")
    arr = L[(L.PHASE_mvt == "ARR") & L.STAND_mvt.notna() & L.BLOCK_TIME_UTC_mvt.notna()].sort_values("BLOCK_TIME_UTC_mvt")
    a = pd.DataFrame({"STAND_mvt": arr.STAND_mvt.to_numpy(), "a_blk": arr.BLOCK_TIME_UTC_mvt.to_numpy(), "a_pre": arr.FLIGHT_mvt.fillna("").str[:3].to_numpy(),
                      "a_late": C.secs(arr.MVT_TIME_UTC_mvt, arr.SCHED_TIME_UTC_mvt).to_numpy()})
    a["a_blk"] = pd.to_datetime(a.a_blk, utc=True).astype(ds.MVT_TIME_UTC_mvt.dtype)
    m = pd.merge_asof(ds[[C.ID, "STAND_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "FLIGHT_mvt"]], a.assign(key=a.a_blk),
                      left_on="MVT_TIME_UTC_mvt", right_on="key", by="STAND_mvt", direction="backward", allow_exact_matches=False)
    turn = C.secs(m.MVT_TIME_UTC_mvt, m.a_blk)
    linked = ((m.FLIGHT_mvt.fillna("").str[:3] == m.a_pre) & turn.between(1200, 86400)).to_numpy()
    ms = C.secs(m.MVT_TIME_UTC_mvt, m.SCHED_TIME_UTC_mvt).to_numpy()
    lk = pd.DataFrame({C.ID: m[C.ID].to_numpy(), "turn": np.where(linked, turn.to_numpy(), np.nan), "local_delay": np.where(linked, ms - np.clip(m.a_late.to_numpy(), 0, None), np.nan)})
    out = dep[[C.ID]].merge(lk, on=C.ID, how="left")
    X = t[(t.PHASE_mvt == "DEP") & t.LOBT_flt.notna() & t.AOBT_3_flt.notna()]
    xs = pd.DataFrame({"ades": X.ADES_mvt.to_numpy(), "ts": X.MVT_TIME_UTC_mvt.astype("int64").to_numpy() // 10**6, "held": (C.secs(X.AOBT_3_flt, X.LOBT_flt) > 1800).to_numpy().astype("float64")})
    qt = dep.MVT_TIME_UTC_mvt.astype("int64").to_numpy() // 10**6
    qa = dep.ADES_mvt.to_numpy()
    held, cnt = np.full(len(dep), np.nan), np.zeros(len(dep))
    for ades, g in xs.groupby("ades"):
        i = np.flatnonzero(qa == ades)
        if not len(i):
            continue
        g = g.sort_values("ts")
        ts, cs = g.ts.to_numpy(), np.r_[0.0, np.cumsum(g.held.to_numpy())]
        lo, hi = np.searchsorted(ts, qt[i] - 10800), np.searchsorted(ts, qt[i] + 10800, side="right")
        n = hi - lo
        cnt[i] = n
        held[i] = np.where(n >= 4, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    ex = pd.DataFrame({C.ID: dep[C.ID].to_numpy(), "dest_held": held, "dest_n": cnt})
    return out.merge(ex, on=C.ID, how="left")


def practice(t: pd.DataFrame, share: pd.DataFrame, months) -> tuple[pd.Series, float]:
    """Real share of an airline prefix on its LIRF departures 2 to 9.5 h late (labelled rows of `months`), 5 prior rows at the overall share."""
    d = t[(t.PHASE_mvt == "DEP") & (t.ADEP_mvt == "LIRF") & t.MVT_TIME_UTC_mvt.dt.month.isin(list(months))]
    mh = C.secs(d.MVT_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt) / 3600.0
    y = d.TAXITIME_SEC_mvt.astype("float64")
    real = ~((C.secs(d.BLOCK_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt).abs() <= C.PH_TOL) | ((y - 600).abs() <= 12) | (y > 80000))
    mid = pd.DataFrame({"pre": d.FLIGHT_mvt.str[:3].to_numpy(), "real": real.to_numpy().astype("float64")})[((mh > 2) & (mh <= 9.5)).to_numpy()]
    allr = float(mid.real.mean())
    g = mid.groupby("pre").real.agg(["sum", "size"])
    return (g["sum"] + 5 * allr) / (g["size"] + 5), allr


def add_v3(d: pd.DataFrame, ex: pd.DataFrame, pr: pd.Series, allr: float) -> pd.DataFrame:
    d = d.merge(ex, on=C.ID, how="left")
    d["zone"], d["wrapped_h"] = zones(d.ms.to_numpy())
    d["far3"] = d.later.to_numpy() | (d.ms.to_numpy() > FAR_H * 3600)
    shr = d.pre.map(pr).fillna(allr)
    d["practice"] = shr.where(d.FLIGHT_mvt.notna() & (d.pre_n.fillna(0) >= V2.MIN_ARR))
    return d


def form(d: pd.DataFrame, base: np.ndarray, p: pd.Series, m_real: pd.Series, tb: dict, cell: bool) -> tuple[np.ndarray, np.ndarray]:
    """The v3 value and part of every row of `d`; `base` is the base entry on these rows."""
    ms = d.ms.to_numpy()
    lo, hi = np.minimum(ms, tb["r_lo"]), np.maximum(ms, tb["r_hi"])
    c20 = np.clip(base, lo, hi)
    q, _ = LO.weight(d, tb["c32"])
    c32 = q * (d.off.to_numpy() + LO.TAXI) + (1 - q) * ms
    pp, mr = p.reindex(d[C.ID]).to_numpy(), m_real.reindex(d[C.ID]).to_numpy()
    mix = np.maximum(C.FLOOR, pp * ms + (1 - pp) * mr)
    far = d.far3.to_numpy()
    is45 = far & (d.zone == "forced").to_numpy()
    is20 = far & ~is45
    is30 = ~far & d.rule2.notna().to_numpy()
    is39 = ~far & ~is30 & (d.rule39 != "").to_numpy()
    is32 = ~far & ~is30 & ~is39 & d.band.to_numpy()
    isr1 = ~far & ~is30 & ~is39 & ~is32 & d.noac.to_numpy() & np.isfinite(mix)
    is43 = isr1 & cell & (d.dest_held > HELD).to_numpy() & (ms <= CELL_MS) & (pp <= CELL_P)
    conds = [is45, is20, is30, is39, is32, is43, isr1]
    part = np.select(conds, ["c45", "c20", "c30", "c39", "c32", "c43", "r1"], "base")
    pred = np.select(conds, [ms, c20, d.rule2.to_numpy(), ms, c32, mr + Q * (ms - mr), (1 - NF.W) * base + NF.W * mix], base)
    return np.maximum(C.FLOOR, pred), part


def arms(d: pd.DataFrame, v3: np.ndarray, tb: dict) -> pd.DataFrame:
    """The open and edge far rows with the two points and the weight of each arm on the rollover point."""
    m = (d.far3 & d.zone.isin(["open", "edge"])).to_numpy()
    o = d[m].copy()
    o["P"], o["R"], o["v3"] = o.ms, tb["r_mid"], v3[m]
    o["m_h"] = o.ms / 3600.0
    gap = (o.P - o.R).to_numpy()
    o["w_v3"] = np.where(np.abs(gap) > 1.0, (o.P - o.v3) / np.where(np.abs(gap) > 1.0, gap, 1.0), 0.0)
    o["w_band"] = np.where((o.later & o.long).to_numpy(), tb["rate"], o.w_v3)
    rule_rows = (o.m_h > 13.9).to_numpy()
    o["w_zone"] = np.where(rule_rows, np.nan_to_num(0.7 * o.practice.to_numpy(), nan=0.05), o.w_v3)
    o["w_levers"] = np.where((o.m_h >= 16.0).to_numpy(), np.where((o.practice >= 0.8).to_numpy(), 0.7, 0.5), o.w_v3)
    for k in ARMS:
        w = o[f"w_{k}"].to_numpy()
        o[f"value_{k}"] = np.maximum(C.FLOOR, np.clip(w * o.R + (1 - w) * o.P, np.minimum(o.P, o.R), np.maximum(o.P, o.R)))
    o["gap"] = gap ** 2 / C.N_RANKING
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default=str(N.V47))
    a = ap.parse_args()
    out_json, out_md = C.RESULTS / f"noplan-form-v3-{a.tag}.json", C.RESULTS / f"noplan-form-v3-{a.tag}.md"
    out_rows, out_arm = C.EXP / f"candidates/noplan-form-v3-{a.tag}.parquet", C.EXP / f"candidates/noplan-bet-arm-{a.tag}.parquet"
    out_entry = C.SAVED / f"noplan-form-v3-{a.tag}-entry.parquet"
    C.refuse_existing(out_json, out_md, out_rows, out_arm, out_entry)
    base_path = Path(a.base)
    rec: dict = {"what": "exp-167 M25: the served form of the LIRF rows without a plan, v3 (v2 with C43, C44, C45)", "weight_r1": NF.W, "base_entry": str(base_path), "base_sha256": C.sha(base_path),
                 "zones": {"far_h": FAR_H, "forced_beyond_h": LIM_FORCED, "open_inside_h": LIM_OPEN, "block_before_takeoff_h": BLOCK_BEFORE_H},
                 "cell": {"dest_held_above": HELD, "ms_at_most_s": CELL_MS, "p_at_most": CELL_P, "weight_on_placeholder": Q}, "lines": {"G": LINE_G, "auc": LINE_AUC, "cell_rate": LINE_RATE}}
    doc = ["# M25: the served form of the LIRF rows without a plan, v3 (v2 with C43, C44, C45)", ""]

    t25 = C.raw_2025()
    share25 = V2.arrival_share(t25)
    ex25 = extra_inputs(t25)
    pr12, all12 = practice(t25, share25, range(1, 13))
    pr10, all10 = practice(t25, share25, R1.TEN)
    d25 = V2.add_v2(NF.classes(NF.noplan_rows(t25)), share25)
    d25_10, d25_12 = add_v3(d25, ex25, pr10, all10), add_v3(d25, ex25, pr12, all12)
    x, ty = S.build()
    sg = I.inbound_sign(t25).set_index(C.ID)
    x["in_late"], x["in_lead"] = sg.in_late.reindex(x[C.ID]).to_numpy(), sg.in_lead.reindex(x[C.ID]).to_numpy()
    e25 = ex25.set_index(C.ID)
    for c in C44 + C43:
        x[c] = e25[c].reindex(x[C.ID]).to_numpy()
    xid = x[C.ID].to_numpy()

    # ------------------------------------------------------------ (1) class model, each month held in turn
    tb12 = NF.tables(d25)
    P_l = {k: np.full(len(x), np.nan) for k in SETS}
    M_l = np.full(len(x), np.nan)
    for mth in range(1, 13):
        z = pd.concat([x, S.encodings(x, ty, mth)], axis=1)
        tr, te = z[z.month != mth], z[z.month == mth]
        for k, extra in SETS.items():
            p_, m_ = I.fit_score(tr, te, NF.COLS + extra)
            P_l[k][(z.month == mth).to_numpy()] = p_
            if k == "v2":
                M_l[(z.month == mth).to_numpy()] = m_
        print("month", mth, "held", flush=True)
    zero = pd.Series(0.5, index=xid)
    _, part_all = form(d25_12, np.zeros(len(d25_12)), zero, pd.Series(M_l, index=xid), tb12, cell=False)
    r1_ids = set(d25_12[C.ID][part_all == "r1"])
    in_r1 = np.array([i in r1_ids for i in xid])
    ph = x.ph.to_numpy().astype(int)
    auc = {k: {"r1_rows": float(roc_auc_score(ph[in_r1], P_l[k][in_r1])), "all_rows": float(roc_auc_score(ph, P_l[k])),
               "r1_fold_months": float(roc_auc_score(ph[in_r1 & x.month.isin((1, 7)).to_numpy()], P_l[k][in_r1 & x.month.isin((1, 7)).to_numpy()]))} for k in SETS}
    rec["class_model_each_month_held"] = {"r1_rows": int(in_r1.sum()), "r1_placeholders": int(ph[in_r1].sum()), "auc": auc}
    print("LOMO AUC (r1 rows", int(in_r1.sum()), "):", {k: round(v["r1_rows"], 4) for k, v in auc.items()}, "; all rows:", {k: round(v["all_rows"], 4) for k, v in auc.items()}, flush=True)

    # ------------------------------------------------------------ (2) two-fold protocol
    src10 = d25_10[d25_10.month.isin(R1.TEN)]
    tb10, rt10 = NF.tables(src10), V2.rule_tables(src10)
    tsrc = ty[ty.month.isin(R1.TEN)]
    tr = pd.concat([pd.concat([x[x.month == m], S.encodings(x[x.month == m], tsrc, m)], axis=1) for m in R1.TEN])
    te = x[x.month.isin((1, 7))]
    te = pd.concat([te, S.encodings(te, tsrc, 0)], axis=1)
    tid = te[C.ID].to_numpy()
    PF, MF = {}, None
    for k, extra in SETS.items():
        p_, m_ = I.fit_score(tr, te, NF.COLS + extra)
        PF[k] = pd.Series(p_, index=tid)
        if k == "v2":
            MF = pd.Series(m_, index=tid)
    r = C.served_forms(C.fold_rows()).set_index(C.ID)
    ids = r.index.to_numpy("int64")
    y = GS.labels(ids)
    base = np.maximum(C.FLOOR, r.F.to_numpy("float64"))
    f = d25_10[d25_10[C.ID].isin(r.index)].reset_index(drop=True)
    pos = r.index.get_indexer(f[C.ID])
    w26 = r.w26.to_numpy()[pos]
    yf, bf = y[pos], base[pos]

    def read(pred: np.ndarray, ref_rows: np.ndarray | None = None) -> dict:
        b0, cand = base.copy(), base.copy()
        if ref_rows is not None:
            b0[pos] = ref_rows
        cand[pos] = pred
        sc = GS.score(ids, b0, cand, y=y)
        g = ((yf - b0[pos]) ** 2 - (yf - pred) ** 2) * w26
        o = np.argsort(-np.abs(g))
        return {"G": sc["G"], "lo95": sc["interval"]["lo95"], "hi95": sc["interval"]["hi95"], "G_without_top_days": sc["G_without_top_days"], "months": sc["views"]["month"],
                "G_without_top5_rows": float(g.sum() - g[o[:5]].sum()), "rows_moved": int((pred != b0[pos]).sum()), "rows_gain": int((g > 0).sum()), "rows_lose": int((g < 0).sum())}

    def line(name: str, q: dict) -> str:
        return (f"{name}: G {q['G']:+.1f} [{q['lo95']:+.1f}, {q['hi95']:+.1f}] months {({k: round(v) for k, v in q['months'].items()})}; without top days {q['G_without_top_days']:+.0f}; "
                f"without top 5 rows {q['G_without_top5_rows']:+.0f}; moved {q['rows_moved']}, gain {q['rows_gain']}, lose {q['rows_lose']}")

    def mixs(p: pd.Series, m_: pd.Series) -> pd.Series:
        ms_ = te.set_index(C.ID).ms.reindex(p.index).to_numpy()
        return pd.Series(np.maximum(C.FLOOR, p.to_numpy() * ms_ + (1 - p.to_numpy()) * m_.to_numpy()), index=p.index)

    v2 = {k: V2.form(f, bf, mixs(PF[k], MF), tb10, rt10, bet=False)[0] for k in SETS}
    tf: dict = {"rows": int(len(f)), "v2_vs_F": read(v2["v2"]), "columns_over_v2_form": {k: read(v2[k], v2["v2"]) for k in ("c44", "c43", "both")}}
    print(line("v2 against F (parity +1,653.5)", tf["v2_vs_F"]), flush=True)
    for k, q in tf["columns_over_v2_form"].items():
        print(line(f"v2 form with the {k} columns, over the v2 form", q), flush=True)
    g_cols = {k: tf["columns_over_v2_form"][k]["G"] for k in ("c44", "c43", "both")}
    d_auc = {k: auc[k]["r1_rows"] - auc["v2"]["r1_rows"] for k in ("c44", "c43", "both")}
    pass44 = g_cols["c44"] >= LINE_G
    pass43 = not (d_auc["c43"] < LINE_AUC and g_cols["c43"] < LINE_G)
    if pass44 and pass43:
        chosen = "both" if g_cols["both"] >= 0 else ("c44" if g_cols["c44"] >= g_cols["c43"] else "c43")
    else:
        chosen = "c44" if pass44 else ("c43" if pass43 else "v2")
    rec["lines_read"] = {"auc_rise_r1_rows": d_auc, "fold_G_over_v2_form": g_cols, "c44_passes": bool(pass44), "c43_input_passes": bool(pass43), "class_model_of_v3": chosen}
    print("lines: AUC rise", {k: round(v, 4) for k, v in d_auc.items()}, "; G over the v2 form", {k: round(v, 1) for k, v in g_cols.items()}, "; C44 passes", pass44, "; C43 input passes", pass43, "; class model of v3:", chosen, flush=True)

    # ------------------------------------------------------------ (3) the cell
    pl, ml = pd.Series(P_l[chosen], index=xid), pd.Series(M_l, index=xid)
    _, part_l = form(d25_12, np.zeros(len(d25_12)), pl, ml, tb12, cell=True)
    cl = d25_12[part_l == "c43"].copy()
    cl["p"], cl["mr"] = pl.reindex(cl[C.ID]).to_numpy(), ml.reindex(cl[C.ID]).to_numpy()
    cl["rule"] = np.maximum(C.FLOOR, cl.mr + Q * (cl.ms - cl.mr))
    cl["mixv"] = np.maximum(C.FLOOR, cl.p * cl.ms + (1 - cl.p) * cl.mr)
    cl["g"] = ((cl.y - cl.mixv) ** 2 - (cl.y - cl.rule) ** 2) / len(r)
    ten = cl[cl.month.isin(R1.TEN)]
    cell_ten = {"rows": int(len(ten)), "placeholders": int((ten.cls == "P").sum()), "rate": float((ten.cls == "P").mean()) if len(ten) else float("nan"), "mean_p": float(ten.p.mean()) if len(ten) else float("nan"),
                "G_rule_over_mixture": float(ten.g.sum()), "by_month": {int(k): {"rows": int(len(v)), "placeholders": int((v.cls == "P").sum()), "G": float(v.g.sum())} for k, v in ten.groupby("month")},
                "months_at_or_above_0": int(sum(v.g.sum() >= 0 for _, v in ten.groupby("month"))), "G_without_3_largest_gains": float(np.sort(ten.g.to_numpy())[:-3].sum()) if len(ten) > 3 else float("nan"),
                "fold_months_each_held": {"rows": int((~cl.month.isin(R1.TEN)).sum()), "placeholders": int(((cl.cls == "P") & ~cl.month.isin(R1.TEN)).sum())}}
    pass_cell = bool(len(ten) > 0 and cell_ten["rate"] <= LINE_RATE)
    rec["cell_ten_other_months_each_held"] = cell_ten
    rec["lines_read"]["c43_part_passes"] = pass_cell
    print("cell, ten other months (each held):", {k: v for k, v in cell_ten.items() if k != "by_month"}, "\n  by month", cell_ten["by_month"], "; passes", pass_cell, flush=True)

    # ------------------------------------------------------------ (4), (5) the fold form
    pf, mf = PF[chosen], MF
    v3, part3 = form(f, bf, pf, mf, tb10, cell=pass_cell)
    v3_nocell, _ = form(f, bf, pf, mf, tb10, cell=False)
    v3_cell_any, part_any = form(f, bf, pf, mf, tb10, cell=True)
    g3 = ((yf - bf) ** 2 - (yf - v3) ** 2) * w26
    g2 = ((yf - bf) ** 2 - (yf - v2["v2"]) ** 2) * w26
    by_part = {p_: {"rows": int((part3 == p_).sum()), "moved": int(((part3 == p_) & (v3 != bf)).sum()), "gain_vs_F": float(g3[part3 == p_].sum()), "gain_over_v2": float((g3 - g2)[part3 == p_].sum()),
                    "gain_january": float(g3[(part3 == p_) & (f.month == 1).to_numpy()].sum()), "gain_july": float(g3[(part3 == p_) & (f.month == 7).to_numpy()].sum()),
                    "rows_gain": int((g3[part3 == p_] > 0).sum()), "rows_lose": int((g3[part3 == p_] < 0).sum()), "classes": f.cls[part3 == p_].value_counts().to_dict()} for p_ in PARTS}
    cm = part_any == "c43"
    tf.update({"v3_vs_F": read(v3), "v3_over_v2": read(v3, v2["v2"]), "v3_without_cell_over_v2": read(v3_nocell, v2["v2"]), "cell_over_v3_without_cell": read(v3_cell_any, v3_nocell), "by_part": by_part,
               "cell_fold_rows": {"rows": int(cm.sum()), "classes": f.cls[cm].value_counts().to_dict(), "january": int((cm & (f.month == 1).to_numpy()).sum()), "july": int((cm & (f.month == 7).to_numpy()).sum()),
                                  "mean_weight_of_F_on_the_placeholder_point": float(np.clip((bf[cm] - mf.reindex(f[C.ID]).to_numpy()[cm]) / (f.ms.to_numpy()[cm] - mf.reindex(f[C.ID]).to_numpy()[cm]), 0, 1.5).mean()) if cm.any() else float("nan")},
               "forced_fold_rows": {"rows": int((part3 == "c45").sum()), "classes": f.cls[part3 == "c45"].value_counts().to_dict(), "gain_over_v2": float((g3 - g2)[part3 == "c45"].sum())},
               "tables_ten_months": {k: v for k, v in tb10.items() if k != "c32"}})
    for name in ("v3_vs_F", "v3_over_v2", "v3_without_cell_over_v2", "cell_over_v3_without_cell"):
        print(line(name, tf[name]), flush=True)
    tp = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "classes"} for k, v in by_part.items()}).T
    print(tp.round(1).to_string(), "\ncell fold rows:", tf["cell_fold_rows"], "\nforced fold rows:", tf["forced_fold_rows"], flush=True)

    # ------------------------------------------------------------ (6) the arms on the open fold rows
    af = arms(f, v3, tb10)
    arm_read = {}
    for k in ARMS:
        pa = v3.copy()
        pa[af.index.to_numpy()] = af[f"value_{k}"].to_numpy()
        arm_read[k] = read(pa, v3)
        print(line(f"arm {k} over v3 (open and edge fold rows)", arm_read[k]), flush=True)
    af["y"], af["F"] = yf[af.index.to_numpy()], bf[af.index.to_numpy()]
    tf["arms_over_v3"] = arm_read
    tf["open_fold_rows"] = af[["FLIGHT_mvt", "month", "m_h", "wrapped_h", "zone", "cls", "practice", "y", "F", "v3", "w_v3", "w_band", "w_zone", "w_levers"]].to_dict("records")
    print(af[["FLIGHT_mvt", "month", "m_h", "wrapped_h", "zone", "cls", "practice", "w_v3", "w_band", "w_zone", "w_levers"]].round(2).to_string(index=False), flush=True)
    rec["two_fold"] = tf
    doc += ["## Lines", "", f"Class model with each month held, AUC on the {int(in_r1.sum())} r1 rows: " + ", ".join(f"{k} {v['r1_rows']:.4f}" for k, v in auc.items()) + ".",
            "Fold G of the v2 form with the columns, over the v2 form: " + ", ".join(f"{k} {v:+.1f}" for k, v in g_cols.items()) + f". C44 passes: {pass44}; C43 input passes: {pass43}; class model of v3: `{chosen}`.",
            f"Cell on the ten other months (each held): {cell_ten['rows']} rows, {cell_ten['placeholders']} placeholders; G of the rule against the mixture {cell_ten['G_rule_over_mixture']:+.1f}; passes: {pass_cell}.", "",
            "## Two-fold read", "",
            C.md(pd.DataFrame([{"read": n_, "G": q["G"], "lo95": q["lo95"], "hi95": q["hi95"], "January": q["months"].get("1", q["months"].get(1)), "July": q["months"].get("7", q["months"].get(7)),
                                "without top days": q["G_without_top_days"], "without 5 rows": q["G_without_top5_rows"], "rows moved": q["rows_moved"]}
                               for n_, q in (("v2 against F (parity)", tf["v2_vs_F"]), ("v3 against F", tf["v3_vs_F"]), ("v3 over v2", tf["v3_over_v2"]), ("v3 without the cell, over v2", tf["v3_without_cell_over_v2"]),
                                             ("the cell over v3 without it", tf["cell_over_v3_without_cell"]), ("arm band over v3", arm_read["band"]), ("arm zone over v3", arm_read["zone"]), ("arm levers over v3", arm_read["levers"]))]), index=False), "",
            "By part (v3 against F):", "", C.md(tp), ""]

    # ------------------------------------------------------------ (7) 2026 on the base entry
    t26 = pd.read_parquet(C.RAW / "ranking.parquet")
    t26["apt"] = np.where(t26.PHASE_mvt == "DEP", t26.ADEP_mvt, t26.ADES_mvt)
    share26 = V2.arrival_share(t26)
    ex26 = extra_inputs(t26)
    d26 = add_v3(V2.add_v2(NF.noplan_rows(t26), share26), ex26, pr12, all12)
    rt12 = V2.rule_tables(d25_12)
    x26, _ = N.features(t26[t26.apt == "LIRF"].copy())
    s26 = I.inbound_sign(t26).set_index(C.ID)
    x26["in_late"], x26["in_lead"] = s26.in_late.reindex(x26[C.ID]).to_numpy(), s26.in_lead.reindex(x26[C.ID]).to_numpy()
    e26 = ex26.set_index(C.ID)
    for c in C44 + C43:
        x26[c] = e26[c].reindex(x26[C.ID]).to_numpy()
    tr12 = pd.concat([pd.concat([x[x.month == m], S.encodings(x[x.month == m], ty, m)], axis=1) for m in range(1, 13)])
    te26 = pd.concat([x26, S.encodings(x26, ty, 0)], axis=1)
    p2_, m2_ = I.fit_score(tr12, te26, NF.COLS)
    p3_, _ = I.fit_score(tr12, te26, NF.COLS + SETS[chosen]) if chosen != "v2" else (p2_, None)
    i26 = te26[C.ID].to_numpy()
    p26, m26 = pd.Series(p3_, index=i26), pd.Series(m2_, index=i26)
    mix2 = pd.Series(np.maximum(C.FLOOR, p2_ * te26.ms.to_numpy() + (1 - p2_) * m2_), index=i26)
    b = pd.read_parquet(base_path)
    if len(b) != C.N_RANKING:
        raise SystemExit("the base is not a full entry")
    v = pd.Series(b.TAXITIME_SEC_mvt.to_numpy("float64"), index=b.MVT_ID_mvt.to_numpy("float64").astype("int64"))
    d26["base"] = v.reindex(d26[C.ID]).to_numpy()
    if d26.base.isna().any():
        raise SystemExit("a 2026 row without a plan is not in the base entry")
    b26 = d26.base.to_numpy()
    d26["v2"], d26["part_v2"] = V2.form(d26, b26, mix2, tb12, rt12, bet=False)
    d26["candidate"], d26["part"] = form(d26, b26, p26, m26, tb12, cell=pass_cell)
    d26["p"], d26["m_real"] = p26.reindex(d26[C.ID]).to_numpy(), m26.reindex(d26[C.ID]).to_numpy()
    d26["change"], d26["change_from_v2"] = d26.candidate - d26.base, d26.candidate - d26.v2
    g26 = d26.groupby("part")
    t26p = pd.DataFrame({"rows": g26.size(), "changed": g26.change.apply(lambda s_: int((s_ != 0).sum())), "mean change, s": g26.change.mean(),
                         "distance from the base": g26.change.apply(lambda s_: (s_ ** 2).sum() / C.N_RANKING), "distance from v2": g26.change_from_v2.apply(lambda s_: (s_ ** 2).sum() / C.N_RANKING),
                         "rows changed from v2": g26.change_from_v2.apply(lambda s_: int((s_ != 0).sum()))})
    cov = {}
    for name, fr in (("2025", x), ("2025 January and July", x[x.month.isin((1, 7))]), ("2026 January", x26[x26.month == 1]), ("2026 July", x26[x26.month == 7])):
        cov[name] = {"rows": int(len(fr)), **{f"{c} present": float(fr[c].notna().mean()) for c in ("turn", "dest_held")}, "turn median, s": float(fr.turn.median()), "local_delay median, s": float(fr.local_delay.median()),
                     "dest_held mean": float(fr.dest_held.mean()), "dest_held above 0.25": float((fr.dest_held > HELD).mean())}
    # the cell without a label: a row is a placeholder with chance pi, else it sits at the real point of the class model
    cz = d26[d26.part == "c43"] if pass_cell else d26[form(d26, b26, p26, m26, tb12, cell=True)[1] == "c43"]
    rule = np.maximum(C.FLOOR, cz.m_real + Q * (cz.ms - cz.m_real)).to_numpy()
    Pz, Rz = cz.ms.to_numpy(), cz.m_real.to_numpy()

    def exp_gain(old: np.ndarray, pi: float) -> float:
        return float((pi * ((Pz - old) ** 2 - (Pz - rule) ** 2) + (1 - pi) * ((Rz - old) ** 2 - (Rz - rule) ** 2)).sum() / C.N_RANKING)
    cell26 = {"rows": int(len(cz)), "january": int((cz.month == 1).sum()), "july": int((cz.month == 7).sum()), "in_the_served_form": pass_cell,
              "stake": float(((Pz - Rz) ** 2).sum() / C.N_RANKING), "distance_v2_to_rule": float(((cz.v2.to_numpy() - rule) ** 2).sum() / C.N_RANKING), "distance_base_to_rule": float(((cz.base.to_numpy() - rule) ** 2).sum() / C.N_RANKING),
              "three_largest_rows_share_of_distance_from_v2": float(np.sort((cz.v2.to_numpy() - rule) ** 2)[-3:].sum() / max(((cz.v2.to_numpy() - rule) ** 2).sum(), 1e-9)),
              "mean_weight_of_v2_on_the_placeholder_point": float(np.clip((cz.v2.to_numpy() - Rz) / (Pz - Rz), 0, 1.5).mean()) if len(cz) else float("nan"),
              "mean_weight_of_base_on_the_placeholder_point": float(np.clip((cz.base.to_numpy() - Rz) / (Pz - Rz), 0, 1.5).mean()) if len(cz) else float("nan"),
              "expected_gain_over_v2_by_placeholder_share": {str(pi): exp_gain(cz.v2.to_numpy(), pi) for pi in (0.05, 0.10, 0.15, 0.25, 0.35)},
              "expected_gain_over_base_by_placeholder_share": {str(pi): exp_gain(cz.base.to_numpy(), pi) for pi in (0.05, 0.10, 0.15, 0.25, 0.35)}}
    g0, g1 = exp_gain(cz.v2.to_numpy(), 0.0), exp_gain(cz.v2.to_numpy(), 1.0)
    cell26["break_even_share_over_v2"] = float(g0 / (g0 - g1)) if g0 != g1 else float("nan")
    a26 = arms(d26, d26.candidate.to_numpy(), tb12)
    forced26 = d26[d26.part == "c45"]
    rec["rows_2026"] = {"rows": int(len(d26)), "by_part": t26p.to_dict("index"), "tables_twelve_months": {k: v_ for k, v_ in tb12.items() if k != "c32"}, "coverage_of_new_columns": cov, "cell": cell26,
                        "forced_rows": forced26[[C.ID, "FLIGHT_mvt", "ms", "wrapped_h", "base", "v2", "candidate"]].to_dict("records"),
                        "distance_from_base": float((d26.change ** 2).sum() / C.N_RANKING), "distance_from_v2": float((d26.change_from_v2 ** 2).sum() / C.N_RANKING), "rows_changed_from_v2": int((d26.change_from_v2 != 0).sum()),
                        "v2_parts_of_changed_rows": d26.part_v2[d26.change_from_v2 != 0].value_counts().to_dict(), "mean_p_v3_r1_rows": {int(k): float(v_) for k, v_ in d26[d26.part.isin(["r1", "c43"])].groupby("month").p.mean().items()},
                        "mean_p_v2_model_r1_rows": {int(k): float(v_) for k, v_ in pd.Series(p2_, index=i26).reindex(d26[C.ID][d26.part.isin(["r1", "c43"])]).groupby(d26.month[d26.part.isin(["r1", "c43"])].to_numpy()).mean().items()}}
    print("2026:\n", t26p.round(1).to_string(), "\ncoverage:\n", pd.DataFrame(cov).T.round(3).to_string(), "\ncell:", {k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in cell26.items()},
          "\nforced rows:\n", forced26[["FLIGHT_mvt", "ms", "wrapped_h", "base", "v2", "candidate"]].round(1).to_string(index=False),
          "\nopen and edge rows:\n", a26[["FLIGHT_mvt", "ADES_mvt", "m_h", "wrapped_h", "zone", "practice", "pre_def", "pre_n", "P", "R", "base", "v3", "w_v3", "w_band", "w_zone", "w_levers", "gap"]].round(2).to_string(index=False),
          "\ndistance from the base", round(rec["rows_2026"]["distance_from_base"], 1), "; from v2", round(rec["rows_2026"]["distance_from_v2"], 1), "; rows changed from v2", rec["rows_2026"]["rows_changed_from_v2"], rec["rows_2026"]["v2_parts_of_changed_rows"],
          "\nmean p on r1 rows, v3 model", rec["rows_2026"]["mean_p_v3_r1_rows"], "v2 model", rec["rows_2026"]["mean_p_v2_model_r1_rows"], flush=True)
    doc += [f"## 2026 on the base entry `{base_path.name}` (sha256 {rec['base_sha256'][:12]}…)", "", C.md(t26p), "", "Coverage of the new columns (rows without type):", "", C.md(pd.DataFrame(cov).T, digits=3), "",
            f"The cell: {cell26['rows']} rows (January {cell26['january']}, July {cell26['july']}); distance of v2 from the rule {cell26['distance_v2_to_rule']:.1f}; expected gain over v2 by placeholder share: "
            + ", ".join(f"{k}: {v_:+.0f}" for k, v_ in cell26["expected_gain_over_v2_by_placeholder_share"].items()) + f"; break-even share {cell26['break_even_share_over_v2']:.3f}.", "",
            "Open and edge far rows (the bet-arm file):", "", C.md(a26[["FLIGHT_mvt", "m_h", "wrapped_h", "zone", "practice", "base", "v3", "w_v3", "w_band", "w_zone", "w_levers", "gap"]], index=False, digits=2), ""]
    C.refuse_existing(out_json, out_md, out_rows, out_arm, out_entry)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    d26[[C.ID, "month", "FLIGHT_mvt", "ms", "ku", "state", "pre_def", "pre_n", "rule39", "zone", "wrapped_h", "dest_held", "dest_n", "turn", "local_delay", "p", "m_real", "part_v2", "part", "base", "v2", "candidate"]].to_parquet(out_rows, index=False)
    a26[[C.ID, "month", "FLIGHT_mvt", "ADES_mvt", "ms", "m_h", "wrapped_h", "zone", "later", "practice", "pre_def", "pre_n", "P", "R", "gap", "base", "v3", "w_v3"] + [f"w_{k}" for k in ARMS] + [f"value_{k}" for k in ARMS]].to_parquet(out_arm, index=False)
    rec["rows_file"] = {"file": str(out_rows.relative_to(C.ROOT)), "sha256": C.sha(out_rows)}
    rec["bet_arm_file"] = {"file": str(out_arm.relative_to(C.ROOT)), "sha256": C.sha(out_arm), "rows": int(len(a26)), "note": "never an entry of its own; the desk decides the arm",
                           "distance_of_each_arm_from_v3": {k: float(((a26[f"value_{k}"] - a26.v3) ** 2).sum() / C.N_RANKING) for k in ARMS}}
    order = b.MVT_ID_mvt.to_numpy("float64").astype("int64")
    new = v.copy()
    new.loc[d26[C.ID].to_numpy()] = d26.candidate.to_numpy()
    e = pd.DataFrame({"MVT_ID_mvt": b.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": new.reindex(order).to_numpy()})
    if len(e) != C.N_RANKING or not np.isfinite(e.TAXITIME_SEC_mvt).all() or (e.TAXITIME_SEC_mvt < C.FLOOR).any():
        raise SystemExit("the entry is not a full finite vector above the floor")
    e.to_parquet(out_entry, index=False)
    diff = np.abs(e.TAXITIME_SEC_mvt.to_numpy() - b.TAXITIME_SEC_mvt.to_numpy("float64"))
    outside = ~np.isin(order, d26[C.ID].to_numpy())
    if (diff[outside] != 0).any():
        raise SystemExit("the entry differs from the base outside the form's rows")
    rec["entry"] = {"file": str(out_entry.relative_to(C.ROOT)), "sha256": C.sha(out_entry), "rows": int(len(e)), "rows_changed_from_base": int((diff > 0).sum()), "max_abs_change_s": float(diff.max()),
                    "distance_from_base": float((diff ** 2).sum() / C.N_RANKING)}
    print("entry", rec["entry"], "\nbet-arm file", rec["bet_arm_file"], flush=True)
    rec["code_sha256"] = {n_: C.sha(m.__file__) for n_, m in (("noplan_form_v3.py", sys.modules[__name__]), ("noplan_form_v2.py", V2), ("noplan_form.py", NF), ("plan_link.py", P), ("local_offset.py", LO),
                          ("inbound_sign.py", I), ("noac_candidate_r1.py", R1), ("noac_candidate.py", N), ("regime_signal.py", S), ("common.py", C))}
    out_md.write_text("\n".join(doc) + "\n")
    C.write_json(out_json, rec)


if __name__ == "__main__":
    main()
