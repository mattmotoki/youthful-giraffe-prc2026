"""M14: one served form for the LIRF rows without a plan (C20, C30, C32, r1 with the C31 sign).

    .venv/bin/python experiments/exp-167-target/code/noplan_form.py --tag 001
    .venv/bin/python experiments/exp-167-target/code/noplan_form.py --tag <new> --base <entry.parquet>

Row order (a row takes the first part that holds it):
    c20   take-off (- 15 min) on a later UTC date than SCHED: the entry projected onto the two
          points [MVT - SCHED, 86,400 + taxi] (`levers`, exp-165 check 20 (a)).  The bet arm (b)
          also gives the rows with 16 h < MVT - SCHED <= 24 h the rollover weight of 2025.
    c30   a decided link state (`plan_link.py`, rule 2 of `fresh`): the plan of the arrival row.
    c32   0 < MVT - SCHED < 3,900 s: placeholder or local-offset shift by airline prefix
          (`local_offset.py`).
    r1    the other same-date rows without type: max(60, 0.75 base + 0.25 mixture), the class
          model of M8-r1 with the late-inbound sign and lead (`inbound_sign.py`).
    base  everything else.

Part A: the two-fold protocol.  Every table and model is fitted on the ten other 2025 months and
the form is read on January and July 2025 against F (v32 on these rows = v47's form).
Part B: the 2026 vector on a base entry (default: the uploaded v47), tables and models from the
twelve 2025 months.  The base may be any full entry, so that `valid` can apply the form to a new
route.  Write-once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import inbound_sign as I
import local_offset as LO
import noac_candidate as N
import noac_candidate_r1 as R1
import plan_link as P
import regime_signal as S

sys.path.insert(0, str(C.ROOT))
from prc import gscore as GS  # noqa: E402

W = 0.25
COLS = R1.COLS + I.ARMS["sign+lead"]
COLS_R1 = R1.COLS
LONG = (16 * 3600.0, 24 * 3600.0)
PARTS = ("c20", "c30", "c32", "r1", "base")


def noplan_rows(t: pd.DataFrame) -> pd.DataFrame:
    """LIRF departures without a plan, with the inputs of every part (label-free)."""
    d = t[(t.PHASE_mvt == "DEP") & (t.ADEP_mvt == "LIRF") & t.LOBT_flt.isna()].copy()
    d[C.ID] = d[C.ID].astype("int64")
    d["ms"] = C.secs(d.MVT_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
    d["ku"] = ((d.MVT_TIME_UTC_mvt - pd.Timedelta(minutes=15)).dt.normalize() - d.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days
    d["month"] = d.MVT_TIME_UTC_mvt.dt.month
    d["day"] = d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    d["noac"] = d.AIRCRAFT_TYPE_mvt.isna()
    loc = d.MVT_TIME_UTC_mvt.dt.tz_convert("Europe/Rome")
    d["off"] = (loc.dt.tz_localize(None) - d.MVT_TIME_UTC_mvt.dt.tz_localize(None)).dt.total_seconds()
    d["pre"] = d.FLIGHT_mvt.fillna("").str[:3]
    c = P.states(P.link(t)).set_index(C.ID)
    d["state"] = c.state.reindex(d[C.ID]).fillna("no link").to_numpy()
    d["rule2"] = c.rule2.reindex(d[C.ID]).to_numpy()
    d["later"] = d.ku >= 1
    d["band"] = (d.ms > LO.BAND[0]) & (d.ms < LO.BAND[1])
    d["long"] = (d.ms > LONG[0]) & (d.ms <= LONG[1])
    return d.reset_index(drop=True)


def classes(d: pd.DataFrame) -> pd.DataFrame:
    """Label classes of 2025 rows (training side)."""
    d = d.copy()
    d["y"] = d.TAXITIME_SEC_mvt.astype("float64")
    ph = C.secs(d.BLOCK_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt).abs() <= C.PH_TOL
    d["cls"] = np.where(ph, "P", np.where(d.y > 80000, "R", "O"))
    d["kind"] = LO.kinds(d)
    return d


def tables(src: pd.DataFrame) -> dict:
    """C20's rollover constants and C32's prefix table from labelled rows of `src`."""
    lat = src[src.later]
    rt = lat.y[lat.cls == "R"] - C.DAY
    lg = lat[lat.long]
    return {"r_lo": float(C.DAY + rt.min()), "r_hi": float(C.DAY + rt.max()), "r_mid": float(C.DAY + rt.median()),
            "rate": float(((lg.cls == "R").sum() + 0.5) / (len(lg) + 1.0)), "later_rows": int(len(lat)),
            "later_classes": lat.cls.value_counts().to_dict(), "long_rows": int(len(lg)), "long_rollovers": int((lg.cls == "R").sum()),
            "c32": LO.table(src[src.band & ~src.later])}


def mixture(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> pd.Series:
    p, m_real = I.fit_score(tr, te, cols)
    return pd.Series(np.maximum(C.FLOOR, p * te.ms.to_numpy() + (1 - p) * m_real), index=te[C.ID].to_numpy())


def form(d: pd.DataFrame, base: np.ndarray, mix: pd.Series, tb: dict, bet: bool) -> tuple[np.ndarray, np.ndarray]:
    """The served value and the part of every row of `d`; `base` is the base entry on these rows."""
    ms = d.ms.to_numpy()
    lo, hi = np.minimum(ms, tb["r_lo"]), np.maximum(ms, tb["r_hi"])
    c20 = np.clip(base, lo, hi)
    if bet:
        c20 = np.where(d.long, np.clip(tb["rate"] * tb["r_mid"] + (1 - tb["rate"]) * ms, lo, hi), c20)
    q, _ = LO.weight(d, tb["c32"])
    c32 = q * (d.off.to_numpy() + LO.TAXI) + (1 - q) * ms
    mx = mix.reindex(d[C.ID]).to_numpy()
    later = d.later.to_numpy()
    is30 = ~later & d.rule2.notna().to_numpy()
    is32 = ~later & ~is30 & d.band.to_numpy()
    isr1 = ~later & ~is30 & ~is32 & d.noac.to_numpy() & np.isfinite(mx)
    part = np.select([later, is30, is32, isr1], ["c20", "c30", "c32", "r1"], "base")
    pred = np.select([later, is30, is32, isr1], [c20, d.rule2.to_numpy(), c32, (1 - W) * base + W * mx], base)
    return np.maximum(C.FLOOR, pred), part


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default=str(N.V47))
    a = ap.parse_args()
    out_json, out_md = C.RESULTS / f"noplan-form-{a.tag}.json", C.RESULTS / f"noplan-form-{a.tag}.md"
    out_rows = C.EXP / f"candidates/noplan-form-{a.tag}.parquet"
    out_entry, out_entry_b = C.SAVED / f"noplan-form-{a.tag}-entry.parquet", C.SAVED / f"noplan-form-{a.tag}-bet-entry.parquet"
    C.refuse_existing(out_json, out_md, out_rows, out_entry, out_entry_b)
    base_path = Path(a.base)
    rec: dict = {"what": "exp-167 M14: the served form of the LIRF rows without a plan (C20, C30, C32, r1 with the C31 sign)",
                 "weight_r1": W, "class_model_columns": COLS, "base_entry": str(base_path), "base_sha256": C.sha(base_path)}
    doc = ["# M14: one served form for the LIRF rows without a plan", ""]

    t25 = C.raw_2025()
    d25 = classes(noplan_rows(t25))
    x, ty = S.build()
    sg = I.inbound_sign(t25).set_index(C.ID)
    x["in_late"], x["in_lead"] = sg.in_late.reindex(x[C.ID]).to_numpy(), sg.in_lead.reindex(x[C.ID]).to_numpy()

    # ------------------------------------------------------------ part A: two-fold protocol
    tb10 = tables(d25[d25.month.isin(R1.TEN)])
    tsrc = ty[ty.month.isin(R1.TEN)]
    tr = pd.concat([pd.concat([x[x.month == m], S.encodings(x[x.month == m], tsrc, m)], axis=1) for m in R1.TEN])
    te = x[x.month.isin((1, 7))]
    te = pd.concat([te, S.encodings(te, tsrc, 0)], axis=1)
    mix, mix_r1 = mixture(tr, te, COLS), mixture(tr, te, COLS_R1)
    r = C.served_forms(C.fold_rows()).set_index(C.ID)
    ids = r.index.to_numpy("int64")
    y = GS.labels(ids)
    base = np.maximum(C.FLOOR, r.F.to_numpy("float64"))
    f = d25[d25[C.ID].isin(r.index)].reset_index(drop=True)
    pos = r.index.get_indexer(f[C.ID])
    w26 = r.w26.to_numpy()[pos]

    def read(pred: np.ndarray, ref: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
        cand = base.copy()
        cand[pos] = pred
        b0 = base if ref is None else ref
        sc = GS.score(ids, b0, cand, y=y)
        g = ((y[pos] - b0[pos]) ** 2 - (y[pos] - pred) ** 2) * w26
        byd = pd.Series(g).groupby(f.day.to_numpy()).sum().sort_values(ascending=False)
        o = np.argsort(-np.abs(g))
        return {"G": sc["G"], "lo95": sc["interval"]["lo95"], "hi95": sc["interval"]["hi95"], "G_without_top_days": sc["G_without_top_days"],
                "months": sc["views"]["month"], "G_without_top5_rows": float(g.sum() - g[o[:5]].sum()), "rows_moved": int((pred != b0[pos]).sum()),
                "rows_gain": int((g > 0).sum()), "rows_lose": int((g < 0).sum()), "largest_day": str(byd.index[0]),
                "largest_day_share": float(byd.iloc[0] / g.sum()) if g.sum() else float("nan"), "top3_days_share": float(byd.iloc[:3].sum() / g.sum()) if g.sum() else float("nan")}, cand

    pred, part = form(f, base[pos], mix, tb10, bet=False)
    pred_b, _ = form(f, base[pos], mix, tb10, bet=True)
    pred_ns, _ = form(f, base[pos], mix_r1, tb10, bet=False)
    main_read, cand_main = read(pred)
    g = ((y[pos] - base[pos]) ** 2 - (y[pos] - pred) ** 2) * w26
    by_part = {p_: {"rows": int((part == p_).sum()), "moved": int(((part == p_) & (pred != base[pos])).sum()), "gain": float(g[part == p_].sum()),
                    "gain_january": float(g[(part == p_) & (f.month == 1).to_numpy()].sum()), "gain_july": float(g[(part == p_) & (f.month == 7).to_numpy()].sum()),
                    "rows_gain": int((g[part == p_] > 0).sum()), "rows_lose": int((g[part == p_] < 0).sum()),
                    "cell_in_F": float((((y[pos] - base[pos]) ** 2) * w26)[part == p_].sum()), "cell_in_form": float((((y[pos] - pred) ** 2) * w26)[part == p_].sum())}
               for p_ in PARTS}
    bet_read, _ = read(pred_b, ref=cand_main)
    cand_ns = base.copy()
    cand_ns[pos] = pred_ns
    sign_read, _ = read(pred, ref=cand_ns)
    lg = f[f.later & f.long]
    rec["two_fold"] = {"tables_ten_months": {k: v for k, v in tb10.items() if k != "c32"}, "rows": int(len(f)), "form_vs_F": main_read, "by_part": by_part,
                       "form_without_sign_vs_F": read(pred_ns)[0], "sign_inside_form": sign_read, "form_with_bet_vs_F": read(pred_b)[0],
                       "bet_over_form": bet_read, "bet_rows": [{"ms": float(a_), "cls": c_, "y": float(y_), "F": float(b_), "form": float(p_), "bet": float(q_)}
                                                               for a_, c_, y_, b_, p_, q_ in zip(lg.ms, lg.cls, y[pos][lg.index], base[pos][lg.index], pred[lg.index], pred_b[lg.index])]}
    m_ = main_read
    print(f"form vs F: G {m_['G']:+.1f} [{m_['lo95']:+.1f}, {m_['hi95']:+.1f}] months {({k: round(v) for k, v in m_['months'].items()})}; without top days "
          f"{m_['G_without_top_days']:+.0f}; without top 5 rows {m_['G_without_top5_rows']:+.0f}; moved {m_['rows_moved']}, gain {m_['rows_gain']}, lose {m_['rows_lose']}; "
          f"largest day {m_['largest_day']} {m_['largest_day_share']:.3f}; top 3 {m_['top3_days_share']:.3f}", flush=True)
    tp = pd.DataFrame(by_part).T
    print(tp.round(1).to_string(), flush=True)
    for name in ("form_without_sign_vs_F", "sign_inside_form", "form_with_bet_vs_F", "bet_over_form"):
        q = rec["two_fold"][name]
        print(f"{name}: G {q['G']:+.1f} [{q['lo95']:+.1f}, {q['hi95']:+.1f}] months {({k: round(v) for k, v in q['months'].items()})}; without top days {q['G_without_top_days']:+.0f}; "
              f"without top 5 rows {q['G_without_top5_rows']:+.0f}; gain {q['rows_gain']}, lose {q['rows_lose']}", flush=True)
    tbet = pd.DataFrame(rec["two_fold"]["bet_rows"])
    print("bet rows (fold):\n", tbet.round(0).to_string(), "\ntables (ten months):", rec["two_fold"]["tables_ten_months"], flush=True)
    doc += ["## Two-fold read against F (tables and models from the ten other months)", "",
            f"G **{m_['G']:+,.0f} [{m_['lo95']:+,.0f}, {m_['hi95']:+,.0f}]**; months {({k: round(v) for k, v in m_['months'].items()})}; "
            f"{m_['G_without_top_days']:+,.0f} without the top days; {m_['G_without_top5_rows']:+,.0f} without the 5 largest rows; {m_['rows_moved']} rows moved, "
            f"{m_['rows_gain']} gain, {m_['rows_lose']} lose; the largest day ({m_['largest_day']}) holds {100 * m_['largest_day_share']:.0f} %, the top three {100 * m_['top3_days_share']:.0f} %.", "",
            "By part:", "", C.md(tp), "", "The bet arm (b), fold rows with a later date and 16 h < MVT − SCHED ≤ 24 h:", "", C.md(tbet, index=False, digits=0), "",
            f"Bet over the form: {bet_read['G']:+,.0f} [{bet_read['lo95']:+,.0f}, {bet_read['hi95']:+,.0f}]. The sign inside the form: "
            f"{sign_read['G']:+,.0f} [{sign_read['lo95']:+,.0f}, {sign_read['hi95']:+,.0f}].", ""]

    # ------------------------------------------------------------ part B: 2026 on the base entry
    t26 = pd.read_parquet(C.RAW / "ranking.parquet")
    t26["apt"] = np.where(t26.PHASE_mvt == "DEP", t26.ADEP_mvt, t26.ADES_mvt)
    d26 = noplan_rows(t26)
    tb12 = tables(d25)
    x26, _ = N.features(t26[t26.apt == "LIRF"].copy())
    s26 = I.inbound_sign(t26).set_index(C.ID)
    x26["in_late"], x26["in_lead"] = s26.in_late.reindex(x26[C.ID]).to_numpy(), s26.in_lead.reindex(x26[C.ID]).to_numpy()
    tr12 = pd.concat([pd.concat([x[x.month == m], S.encodings(x[x.month == m], ty, m)], axis=1) for m in range(1, 13)])
    te26 = pd.concat([x26, S.encodings(x26, ty, 0)], axis=1)
    mix26 = mixture(tr12, te26, COLS)
    b = pd.read_parquet(base_path)
    if len(b) != C.N_RANKING:
        raise SystemExit("the base is not a full entry")
    v = pd.Series(b.TAXITIME_SEC_mvt.to_numpy("float64"), index=b.MVT_ID_mvt.to_numpy("float64").astype("int64"))
    d26["base"] = v.reindex(d26[C.ID]).to_numpy()
    if d26.base.isna().any():
        raise SystemExit("a 2026 row without a plan is not in the base entry")
    d26["candidate"], d26["part"] = form(d26, d26.base.to_numpy(), mix26, tb12, bet=False)
    d26["candidate_bet"], _ = form(d26, d26.base.to_numpy(), mix26, tb12, bet=True)
    if (d26.later & d26.rule2.notna()).any():
        print("note: a 2026 linked row has a later date; C20 takes it", flush=True)
    d26["change"] = d26.candidate - d26.base
    g26 = d26.groupby("part")
    t26p = pd.DataFrame({"rows": g26.size(), "changed": g26.change.apply(lambda s_: int((s_ != 0).sum())), "mean change, s": g26.change.mean(),
                         "mean absolute change, s": g26.change.apply(lambda s_: s_.abs().mean()),
                         "distance from the base, s2 per ranking row": g26.change.apply(lambda s_: (s_ ** 2).sum() / C.N_RANKING)})
    rec["rows_2026"] = {"rows": int(len(d26)), "by_part": t26p.to_dict("index"), "tables_twelve_months": {k: v_ for k, v_ in tb12.items() if k != "c32"},
                        "bet_rows": d26[d26.later & d26.long][[C.ID, "ms", "base", "candidate", "candidate_bet"]].to_dict("records"),
                        "distance_from_base": float((d26.change ** 2).sum() / C.N_RANKING),
                        "bet_distance_from_form": float(((d26.candidate_bet - d26.candidate) ** 2).sum() / C.N_RANKING)}
    print("2026:\n", t26p.round(1).to_string(), "\nbet rows:", rec["rows_2026"]["bet_rows"], "\ntables (twelve months):", rec["rows_2026"]["tables_twelve_months"], flush=True)
    doc += [f"## 2026 on the base entry `{base_path.name}` (sha256 {rec['base_sha256'][:12]}…)", "", C.md(t26p), ""]
    C.refuse_existing(out_json, out_md, out_rows, out_entry, out_entry_b)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    d26[[C.ID, "month", "FLIGHT_mvt", "ms", "ku", "state", "part", "base", "candidate", "candidate_bet"]].to_parquet(out_rows, index=False)
    rec["rows_file"] = {"file": str(out_rows.relative_to(C.ROOT)), "sha256": C.sha(out_rows)}
    order = b.MVT_ID_mvt.to_numpy("float64").astype("int64")
    for col, path, key in (("candidate", out_entry, "entry"), ("candidate_bet", out_entry_b, "entry_bet")):
        new = v.copy()
        new.loc[d26[C.ID].to_numpy()] = d26[col].to_numpy()
        e = pd.DataFrame({"MVT_ID_mvt": b.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": new.reindex(order).to_numpy()})
        if len(e) != C.N_RANKING or not np.isfinite(e.TAXITIME_SEC_mvt).all() or (e.TAXITIME_SEC_mvt < C.FLOOR).any():
            raise SystemExit("the entry is not a full finite vector above the floor")
        e.to_parquet(path, index=False)
        diff = np.abs(e.TAXITIME_SEC_mvt.to_numpy() - b.TAXITIME_SEC_mvt.to_numpy("float64"))
        rec[key] = {"file": str(path.relative_to(C.ROOT)), "sha256": C.sha(path), "rows": int(len(e)), "rows_changed_from_base": int((diff > 0).sum()),
                    "max_abs_change_s": float(diff.max()), "distance_from_base": float((diff ** 2).sum() / C.N_RANKING)}
        print(key, rec[key], flush=True)
    rec["code_sha256"] = {n_: C.sha(m.__file__) for n_, m in (("noplan_form.py", sys.modules[__name__]), ("plan_link.py", P), ("local_offset.py", LO),
                          ("inbound_sign.py", I), ("noac_candidate_r1.py", R1), ("noac_candidate.py", N), ("regime_signal.py", S), ("common.py", C))}
    out_md.write_text("\n".join(doc) + "\n")
    C.write_json(out_json, rec)


if __name__ == "__main__":
    main()
