"""M21 (unit 10): the served form of the LIRF rows without a plan, v2 = v1 with C41 and C39 (a).

    .venv/bin/python experiments/exp-167-target/code/noplan_form_v2.py --tag 001
    .venv/bin/python experiments/exp-167-target/code/noplan_form_v2.py --tag <new> --base <entry.parquet>

Row order (a row takes the first part that holds it); v1 is `noplan_form.py`, not changed:
    c20   take-off (- 15 min) on a later UTC date than SCHED (v1), or take-off more than 12.5 h
          after SCHED (C41, `fresh`): the base clipped between the two points
          [MVT - SCHED, 86,400 + taxi].  The bet arm keeps v1's rows (later date, 16 to 24 h).
    c30   a decided link state: the plan of the arrival row (v1).
    c39   65 min to 12.5 h after SCHED, and the airline prefix has a default share above 0.6 on
          its LIRF arrivals of the month (10 arrivals and more: rule `share`), or has fewer than
          10 such arrivals or no flight string (rule `few`): the placeholder point MVT - SCHED
          (C39 (a), `fresh`).  A default arrival: BLOCK within 6 s of SCHED, or a taxi-in of
          600 +- 12 s.  Arrivals keep block time and label in the ranking file.
    c32   0 < MVT - SCHED < 3,900 s: placeholder or local-offset shift by airline prefix (v1).
    r1    the other rows without type: max(60, 0.75 base + 0.25 mixture) (v1).
    base  everything else.

Part A: the two-fold protocol (tables and the class model from the ten other 2025 months; the
arrival share from the held month's own arrivals, as in 2026), read against F and against v1.
Part B: the 2026 vector on a base entry (default: the uploaded v47).  `--calibrated` names the
c39 rules that take the calibrated point rate x (MVT - SCHED) + (1 - rate) x real point in
place of the placeholder point.  Write-once.
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
import noplan_form as NF
import plan_link as P
import regime_signal as S

sys.path.insert(0, str(C.ROOT))
from prc import gscore as GS  # noqa: E402

FAR = 45_000.0                      # 12.5 h: no real LIRF block time is this far after SCHED (C41)
BAND39 = (3_900.0, FAR)
SHARE, MIN_ARR = 0.6, 10
RULES = ("share", "few")
PARTS = ("c20", "c30", "c39", "c32", "r1", "base")


def arrival_share(t: pd.DataFrame) -> pd.DataFrame:
    """Default share of the LIRF arrivals by airline prefix and month (index: pre, month)."""
    a = t[(t.PHASE_mvt == "ARR") & (t.ADES_mvt == "LIRF")]
    y = a.TAXITIME_SEC_mvt.astype("float64")
    adef = (C.secs(a.BLOCK_TIME_UTC_mvt, a.SCHED_TIME_UTC_mvt).abs() <= C.PH_TOL) | ((y - 600.0).abs() <= 12.0)
    g = pd.DataFrame({"pre": a.FLIGHT_mvt.str[:3].to_numpy(), "month": a.MVT_TIME_UTC_mvt.dt.month.to_numpy(), "adef": adef.to_numpy()})
    return g.groupby(["pre", "month"]).adef.agg(pre_def="mean", pre_n="size")


def add_v2(d: pd.DataFrame, share: pd.DataFrame) -> pd.DataFrame:
    """The C39 rule of a row and the C41 criterion (label-free)."""
    d = d.copy()
    key = pd.MultiIndex.from_arrays([d.pre.to_numpy(), d.month.to_numpy()])
    d["pre_def"], d["pre_n"] = share.pre_def.reindex(key).to_numpy(), share.pre_n.reindex(key).to_numpy()
    enough = (d.pre_n >= MIN_ARR).to_numpy()
    band = ((d.ms > BAND39[0]) & (d.ms <= BAND39[1])).to_numpy()
    d["rule39"] = np.where(band & enough & (d.pre_def > SHARE).to_numpy(), "share", np.where(band & ~enough, "few", ""))
    d["far"] = d.later.to_numpy() | (d.ms > FAR).to_numpy()
    return d


def rule_tables(src: pd.DataFrame) -> dict:
    """Placeholder rate of each c39 rule and the real point, from labelled rows of `src` that the
    c39 part would hold (not far, no decided link state)."""
    s = src[~src.far & src.rule2.isna()]
    band = s[(s.ms > BAND39[0]) & (s.ms <= BAND39[1])]
    out = {"real_point": float(band.y[band.cls == "O"].median())}
    for k in RULES:
        g = s[s.rule39 == k]
        out[k] = {"rows": int(len(g)), "classes": g.cls.value_counts().to_dict(), "rate": float(((g.cls == "P").sum() + 0.5) / (len(g) + 1.0))}
    return out


def form(d: pd.DataFrame, base: np.ndarray, mix: pd.Series, tb: dict, rt: dict, bet: bool, calibrated: tuple[str, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """The v2 value and part of every row of `d`; `base` is the base entry on these rows."""
    ms = d.ms.to_numpy()
    lo, hi = np.minimum(ms, tb["r_lo"]), np.maximum(ms, tb["r_hi"])
    c20 = np.clip(base, lo, hi)
    if bet:
        c20 = np.where((d.later & d.long).to_numpy(), np.clip(tb["rate"] * tb["r_mid"] + (1 - tb["rate"]) * ms, lo, hi), c20)
    q, _ = LO.weight(d, tb["c32"])
    c32 = q * (d.off.to_numpy() + LO.TAXI) + (1 - q) * ms
    c39 = ms.copy()
    for k in calibrated:
        c39 = np.where((d.rule39 == k).to_numpy(), rt[k]["rate"] * ms + (1 - rt[k]["rate"]) * rt["real_point"], c39)
    mx = mix.reindex(d[C.ID]).to_numpy()
    far = d.far.to_numpy()
    is30 = ~far & d.rule2.notna().to_numpy()
    is39 = ~far & ~is30 & (d.rule39 != "").to_numpy()
    is32 = ~far & ~is30 & ~is39 & d.band.to_numpy()
    isr1 = ~far & ~is30 & ~is39 & ~is32 & d.noac.to_numpy() & np.isfinite(mx)
    part = np.select([far, is30, is39, is32, isr1], ["c20", "c30", "c39", "c32", "r1"], "base")
    pred = np.select([far, is30, is39, is32, isr1], [c20, d.rule2.to_numpy(), c39, c32, (1 - NF.W) * base + NF.W * mx], base)
    return np.maximum(C.FLOOR, pred), part


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default=str(N.V47))
    ap.add_argument("--calibrated", default="", help="comma list of c39 rules (share, few) that take the calibrated point in the served form")
    a = ap.parse_args()
    cal = tuple(k for k in a.calibrated.split(",") if k)
    if not set(cal) <= set(RULES):
        raise SystemExit(f"--calibrated takes {RULES}")
    out_json, out_md = C.RESULTS / f"noplan-form-v2-{a.tag}.json", C.RESULTS / f"noplan-form-v2-{a.tag}.md"
    out_rows = C.EXP / f"candidates/noplan-form-v2-{a.tag}.parquet"
    out_entry, out_entry_b = C.SAVED / f"noplan-form-v2-{a.tag}-entry.parquet", C.SAVED / f"noplan-form-v2-{a.tag}-bet-entry.parquet"
    C.refuse_existing(out_json, out_md, out_rows, out_entry, out_entry_b)
    base_path = Path(a.base)
    rec: dict = {"what": "exp-167 M21: the served form of the LIRF rows without a plan, v2 (v1 with C41 and C39 (a))", "weight_r1": NF.W,
                 "class_model_columns": NF.COLS, "base_entry": str(base_path), "base_sha256": C.sha(base_path),
                 "c39": {"band_s": BAND39, "share_above": SHARE, "min_arrivals": MIN_ARR, "calibrated_rules_in_served_form": list(cal)}, "c41_far_s": FAR}
    doc = ["# M21: the served form of the LIRF rows without a plan, v2 (v1 with C41 and C39 (a))", ""]

    t25 = C.raw_2025()
    share25 = arrival_share(t25)
    d25 = add_v2(NF.classes(NF.noplan_rows(t25)), share25)
    x, ty = S.build()
    sg = I.inbound_sign(t25).set_index(C.ID)
    x["in_late"], x["in_lead"] = sg.in_late.reindex(x[C.ID]).to_numpy(), sg.in_lead.reindex(x[C.ID]).to_numpy()

    # ------------------------------------------------------------ part A: two-fold protocol
    src10 = d25[d25.month.isin(R1.TEN)]
    tb10, rt10 = NF.tables(src10), rule_tables(src10)
    tsrc = ty[ty.month.isin(R1.TEN)]
    tr = pd.concat([pd.concat([x[x.month == m], S.encodings(x[x.month == m], tsrc, m)], axis=1) for m in R1.TEN])
    te = x[x.month.isin((1, 7))]
    te = pd.concat([te, S.encodings(te, tsrc, 0)], axis=1)
    mix = NF.mixture(tr, te, NF.COLS)
    r = C.served_forms(C.fold_rows()).set_index(C.ID)
    ids = r.index.to_numpy("int64")
    y = GS.labels(ids)
    base = np.maximum(C.FLOOR, r.F.to_numpy("float64"))
    f = d25[d25[C.ID].isin(r.index)].reset_index(drop=True)
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
        byd = pd.Series(g).groupby(f.day.to_numpy()).sum().sort_values(ascending=False)
        o = np.argsort(-np.abs(g))
        return {"G": sc["G"], "lo95": sc["interval"]["lo95"], "hi95": sc["interval"]["hi95"], "G_without_top_days": sc["G_without_top_days"],
                "months": sc["views"]["month"], "G_without_top5_rows": float(g.sum() - g[o[:5]].sum()), "rows_moved": int((pred != b0[pos]).sum()),
                "rows_gain": int((g > 0).sum()), "rows_lose": int((g < 0).sum()), "largest_day": str(byd.index[0]),
                "largest_day_share": float(byd.iloc[0] / g.sum()) if g.sum() else float("nan")}

    def line(name: str, q: dict) -> str:
        return (f"{name}: G {q['G']:+.1f} [{q['lo95']:+.1f}, {q['hi95']:+.1f}] months {({k: round(v) for k, v in q['months'].items()})}; without top days "
                f"{q['G_without_top_days']:+.0f}; without top 5 rows {q['G_without_top5_rows']:+.0f}; moved {q['rows_moved']}, gain {q['rows_gain']}, lose {q['rows_lose']}")

    v1, part1 = NF.form(f, bf, mix, tb10, bet=False)
    v2, part2 = form(f, bf, mix, tb10, rt10, bet=False, calibrated=cal)
    v2b, _ = form(f, bf, mix, tb10, rt10, bet=True, calibrated=cal)
    v2rule, _ = form(f, bf, mix, tb10, rt10, bet=False)
    v2cal, _ = form(f, bf, mix, tb10, rt10, bet=False, calibrated=RULES)
    g2 = ((yf - bf) ** 2 - (yf - v2) ** 2) * w26
    g1 = ((yf - bf) ** 2 - (yf - v1) ** 2) * w26
    by_part = {p_: {"rows": int((part2 == p_).sum()), "moved": int(((part2 == p_) & (v2 != bf)).sum()), "gain_vs_F": float(g2[part2 == p_].sum()),
                    "gain_january": float(g2[(part2 == p_) & (f.month == 1).to_numpy()].sum()), "gain_july": float(g2[(part2 == p_) & (f.month == 7).to_numpy()].sum()),
                    "rows_gain": int((g2[part2 == p_] > 0).sum()), "rows_lose": int((g2[part2 == p_] < 0).sum()),
                    "gain_of_v1_on_these_rows": float(g1[part2 == p_].sum()), "v1_parts_of_these_rows": pd.Series(part1[part2 == p_]).value_counts().to_dict()}
               for p_ in PARTS}
    c39_rows = {}
    for k in RULES:
        m = (part2 == "c39") & (f.rule39 == k).to_numpy()
        c39_rows[k] = {"rows": int(m.sum()), "classes": f.cls[m].value_counts().to_dict(), "january": int((m & (f.month == 1).to_numpy()).sum()), "july": int((m & (f.month == 7).to_numpy()).sum()),
                       "gain_vs_F": float(g2[m].sum()), "gain_over_v1": float((g2 - g1)[m].sum()), "v1_parts": pd.Series(part1[m]).value_counts().to_dict(),
                       "mean_weight_of_F_on_the_placeholder_point": float(np.clip((bf[m] - rt10["real_point"]) / (f.ms.to_numpy()[m] - rt10["real_point"]), 0, 1.5).mean()) if m.any() else float("nan")}
    band_all = ((f.ms > BAND39[0]) & (f.ms <= BAND39[1])).to_numpy()
    src_count = {k: {"rows": int((band_all & (f.rule39 == k).to_numpy()).sum()), "placeholders": int((band_all & (f.rule39 == k).to_numpy() & (f.cls == "P").to_numpy()).sum())} for k in RULES}
    new_far = f.far.to_numpy() & ~f.later.to_numpy()
    c39m = part2 == "c39"
    not_ph = int((c39m & (f.cls != "P").to_numpy()).sum())
    rec["two_fold"] = {"rows": int(len(f)), "tables_ten_months": {k: v for k, v in tb10.items() if k != "c32"}, "c39_tables_ten_months": rt10,
                       "v1_vs_F": read(v1), "v2_vs_F": read(v2), "v2_over_v1": read(v2, v1), "v2_rule_point_vs_F": read(v2rule), "v2_calibrated_point_vs_F": read(v2cal),
                       "calibrated_over_rule_point": read(v2cal, v2rule), "v2_with_bet_vs_F": read(v2b), "bet_over_v2": read(v2b, v2), "by_part": by_part, "c39_by_rule": c39_rows,
                       "c39_rows_not_placeholder": not_ph, "c39_rows": int(c39m.sum()), "band_rows_before_c20_c30": src_count,
                       "c41_rows_added_to_c20": [{"FLIGHT_mvt": str(fl), "ms": float(m_), "y": float(y_), "cls": c_, "F": float(b_), "v1": float(p1), "v2": float(p2)}
                                                 for fl, m_, y_, c_, b_, p1, p2 in zip(f.FLIGHT_mvt[new_far], f.ms[new_far], yf[new_far], f.cls[new_far], bf[new_far], v1[new_far], v2[new_far])]}
    tf = rec["two_fold"]
    for name in ("v1_vs_F", "v2_vs_F", "v2_over_v1", "v2_rule_point_vs_F", "v2_calibrated_point_vs_F", "calibrated_over_rule_point", "v2_with_bet_vs_F", "bet_over_v2"):
        print(line(name, tf[name]), flush=True)
    tp = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "v1_parts_of_these_rows"} for k, v in by_part.items()}).T
    print(tp.round(1).to_string(), "\nc39 by rule:", c39_rows, "\nband rows before C20 and C30 (fresh: share 30 of 30, few 15 of 16):", src_count,
          "\nc39 rows that are not placeholders:", not_ph, "of", int(c39m.sum()), "\nC41 rows added to C20 (fold):", tf["c41_rows_added_to_c20"], "\nc39 tables (ten months):", rt10, flush=True)
    doc += ["## Two-fold read (tables and the class model from the ten other months)", "",
            C.md(pd.DataFrame([{"read": n_, "G": q["G"], "lo95": q["lo95"], "hi95": q["hi95"], "January": q["months"].get("1", q["months"].get(1)), "July": q["months"].get("7", q["months"].get(7)),
                                "without top days": q["G_without_top_days"], "without 5 rows": q["G_without_top5_rows"], "rows moved": q["rows_moved"]}
                               for n_, q in (("v1 against F (parity)", tf["v1_vs_F"]), ("v2 against F", tf["v2_vs_F"]), ("v2 over v1", tf["v2_over_v1"]),
                                             ("calibrated point over the rule point", tf["calibrated_over_rule_point"]), ("bet arm over v2", tf["bet_over_v2"]))]), index=False), "",
            "By part (v2 against F):", "", C.md(tp), "",
            f"c39 rows: {int(c39m.sum())}, not a placeholder: {not_ph}. By rule: " + "; ".join(f"`{k}` {v['rows']} rows, classes {v['classes']}, against F {v['gain_vs_F']:+,.0f}, over v1 {v['gain_over_v1']:+,.0f}" for k, v in c39_rows.items()) + ".", ""]

    # ------------------------------------------------------------ part B: 2026 on the base entry
    t26 = pd.read_parquet(C.RAW / "ranking.parquet")
    t26["apt"] = np.where(t26.PHASE_mvt == "DEP", t26.ADEP_mvt, t26.ADES_mvt)
    share26 = arrival_share(t26)
    d26 = add_v2(NF.noplan_rows(t26), share26)
    tb12, rt12 = NF.tables(d25), rule_tables(d25)
    x26, _ = N.features(t26[t26.apt == "LIRF"].copy())
    s26 = I.inbound_sign(t26).set_index(C.ID)
    x26["in_late"], x26["in_lead"] = s26.in_late.reindex(x26[C.ID]).to_numpy(), s26.in_lead.reindex(x26[C.ID]).to_numpy()
    tr12 = pd.concat([pd.concat([x[x.month == m], S.encodings(x[x.month == m], ty, m)], axis=1) for m in range(1, 13)])
    te26 = pd.concat([x26, S.encodings(x26, ty, 0)], axis=1)
    mix26 = NF.mixture(tr12, te26, NF.COLS)
    b = pd.read_parquet(base_path)
    if len(b) != C.N_RANKING:
        raise SystemExit("the base is not a full entry")
    v = pd.Series(b.TAXITIME_SEC_mvt.to_numpy("float64"), index=b.MVT_ID_mvt.to_numpy("float64").astype("int64"))
    d26["base"] = v.reindex(d26[C.ID]).to_numpy()
    if d26.base.isna().any():
        raise SystemExit("a 2026 row without a plan is not in the base entry")
    b26 = d26.base.to_numpy()
    d26["v1"], d26["part_v1"] = NF.form(d26, b26, mix26, tb12, bet=False)
    d26["candidate"], d26["part"] = form(d26, b26, mix26, tb12, rt12, bet=False, calibrated=cal)
    d26["candidate_bet"], _ = form(d26, b26, mix26, tb12, rt12, bet=True, calibrated=cal)
    d26["change"], d26["change_from_v1"] = d26.candidate - d26.base, d26.candidate - d26.v1
    g26 = d26.groupby("part")
    t26p = pd.DataFrame({"rows": g26.size(), "changed": g26.change.apply(lambda s_: int((s_ != 0).sum())), "mean change, s": g26.change.mean(),
                         "mean absolute change, s": g26.change.apply(lambda s_: s_.abs().mean()),
                         "distance from the base, s2 per ranking row": g26.change.apply(lambda s_: (s_ ** 2).sum() / C.N_RANKING),
                         "distance from v1, s2 per ranking row": g26.change_from_v1.apply(lambda s_: (s_ ** 2).sum() / C.N_RANKING)})
    # forecast of the c39 part without a label: a row is a placeholder with the twelve-month rate of its rule, else it sits at the real point
    fc = {}
    ms26, rp = d26.ms.to_numpy(), rt12["real_point"]
    for k in RULES:
        m = ((d26.part == "c39") & (d26.rule39 == k)).to_numpy()
        p = rt12[k]["rate"]
        calp = p * ms26[m] + (1 - p) * rp

        def exp_gain(old: np.ndarray, new: np.ndarray) -> float:
            return float((p * ((ms26[m] - old) ** 2 - (ms26[m] - new) ** 2) + (1 - p) * ((rp - old) ** 2 - (rp - new) ** 2)).sum() / C.N_RANKING)
        fc[k] = {"rows": int(m.sum()), "rate_twelve_months": p, "stake": float(((ms26[m] - rp) ** 2).sum() / C.N_RANKING),
                 "mean_weight_of_the_base_on_the_placeholder_point": float(np.clip((b26[m] - rp) / (ms26[m] - rp), 0, 1.5).mean()) if m.any() else float("nan"),
                 "rule_point_over_base": exp_gain(b26[m], ms26[m]), "rule_point_over_v1": exp_gain(d26.v1.to_numpy()[m], ms26[m]),
                 "calibrated_point_over_base": exp_gain(b26[m], calp), "calibrated_point_over_v1": exp_gain(d26.v1.to_numpy()[m], calp),
                 "distance_base_to_rule_point": float(((b26[m] - ms26[m]) ** 2).sum() / C.N_RANKING), "distance_v1_to_rule_point": float(((d26.v1.to_numpy()[m] - ms26[m]) ** 2).sum() / C.N_RANKING),
                 "prefixes": d26.pre[m].value_counts().to_dict()}
    added = d26[d26.far & ~d26.later]
    rec["rows_2026"] = {"rows": int(len(d26)), "by_part": t26p.to_dict("index"), "tables_twelve_months": {k: v_ for k, v_ in tb12.items() if k != "c32"}, "c39_tables_twelve_months": rt12,
                        "c39_forecast_without_a_label": fc, "v1_parts_of_c39_rows": d26.part_v1[d26.part == "c39"].value_counts().to_dict(),
                        "c41_rows_added_to_c20": added[[C.ID, "FLIGHT_mvt", "ms", "base", "v1", "candidate"]].to_dict("records"),
                        "bet_rows": d26[d26.later & d26.long][[C.ID, "ms", "base", "candidate", "candidate_bet"]].to_dict("records"),
                        "distance_from_base": float((d26.change ** 2).sum() / C.N_RANKING), "distance_from_v1": float((d26.change_from_v1 ** 2).sum() / C.N_RANKING),
                        "rows_changed_from_v1": int((d26.change_from_v1 != 0).sum()),
                        "bet_distance_from_form": float(((d26.candidate_bet - d26.candidate) ** 2).sum() / C.N_RANKING)}
    print("2026:\n", t26p.round(1).to_string(), "\nc39 forecast without a label:", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v_.items()} for k, v_ in fc.items()},
          "\nC41 rows added to C20:", rec["rows_2026"]["c41_rows_added_to_c20"], "\nbet rows:", rec["rows_2026"]["bet_rows"],
          "\ndistance from the base", round(rec["rows_2026"]["distance_from_base"], 1), "; from v1", round(rec["rows_2026"]["distance_from_v1"], 1), "; rows changed from v1", rec["rows_2026"]["rows_changed_from_v1"],
          "\nc39 tables (twelve months):", rt12, flush=True)
    doc += [f"## 2026 on the base entry `{base_path.name}` (sha256 {rec['base_sha256'][:12]}…)", "", C.md(t26p), "",
            "Forecast of the c39 part without a label (twelve-month rate of each rule; s² per ranking row):", "",
            C.md(pd.DataFrame({k: {kk: vv for kk, vv in v_.items() if kk != "prefixes"} for k, v_ in fc.items()}).T, digits=3), ""]
    C.refuse_existing(out_json, out_md, out_rows, out_entry, out_entry_b)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    d26[[C.ID, "month", "FLIGHT_mvt", "ms", "ku", "state", "pre_def", "pre_n", "rule39", "part_v1", "part", "base", "v1", "candidate", "candidate_bet"]].to_parquet(out_rows, index=False)
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
    rec["code_sha256"] = {n_: C.sha(m.__file__) for n_, m in (("noplan_form_v2.py", sys.modules[__name__]), ("noplan_form.py", NF), ("plan_link.py", P), ("local_offset.py", LO),
                          ("inbound_sign.py", I), ("noac_candidate_r1.py", R1), ("noac_candidate.py", N), ("regime_signal.py", S), ("common.py", C))}
    out_md.write_text("\n".join(doc) + "\n")
    C.write_json(out_json, rec)


if __name__ == "__main__":
    main()
