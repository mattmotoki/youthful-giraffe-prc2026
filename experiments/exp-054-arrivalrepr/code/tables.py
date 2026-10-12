"""exp-054-arrivalrepr: render the iteration-1 tables of RESULTS.md from the analysis JSON only.

Reads `analysis/audit-r1.json`, `analysis/rep-<fold>.json`, `analysis/fit-<fold>.json` and
`analysis/iter1.json`; computes nothing that is not a sum, a count or a formatting of their values.
Writes `analysis/iter1-tables.md` (write-once; a correction goes to a new name with --out).

    .venv/bin/python experiments/exp-054-arrivalrepr/code/tables.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e54 as E                                                       # noqa: E402

NO = E.NO
OUT_DEFAULT = "analysis/iter1-tables.md"
CONTROLS = ["const", "gmean", "donor", "shuffle"]
REGIMES = ["nm_bulk", "nm_late", "noac", "nonm"]


def J(p: str) -> dict:
    return json.loads((E.EXP / p).read_text())


def f3(x, d=3, sign=True):
    if x is None:
        return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(c, d=3):
    return f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


def table(head: list[str], rows: list[list], align: str | None = None) -> str:
    al = align or ("|" + "|".join("---" for _ in head) + "|")
    return "\n".join(["| " + " | ".join(head) + " |", al]
                     + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def render() -> str:
    au = J("analysis/audit-r1.json")
    rep = {f: J(f"analysis/rep-{f}.json") for f in E.FOLDS}
    fit = {f: J(f"analysis/fit-{f}.json") for f in E.FOLDS}
    it = J("analysis/iter1.json")
    out = ["# exp-054 iteration 1 — tables rendered by `code/tables.py` from the analysis JSON", ""]

    # ---- T1 run record -----------------------------------------------------------------------
    out += ["## T1. Run record (from the records' own clocks and loads)", ""]
    rows = []
    for f in E.FOLDS:
        r, q = rep[f], fit[f]
        rl = [v["launch_load_1m"] for v in r["fits"].values()]
        fl = [q["fits"][a][b]["launch_load_1m"] for a in q["fits"] for b in q["fits"][a]]
        fs = [q["fits"][a][b]["seconds"] for a in q["fits"] for b in q["fits"][a]]
        rows.append([f, r["held_month"], f"{r['fitting_arrivals']:,}", r["frozen"]["utc"],
                     f"{sum(v['trainable_fits'] for v in r['fits'].values())}",
                     f"{min(rl):.2f}–{max(rl):.2f}", f"{r['seconds']:.1f}",
                     q["matrices"]["frozen_utc"], f"{q['trainable_fits']}",
                     f"{min(fl):.2f}–{max(fl):.2f}", f"{min(fs):.2f}–{max(fs):.2f}", f"{q['seconds']:.1f}"])
    out += [table(["fold", "held month", "fitting arrivals", "representation frozen (UTC)",
                   "rep. fits", "rep. launch load", "rep. job s", "departure matrices frozen (UTC)",
                   "departure fits", "departure launch load", "s per departure fit", "fit job s"], rows), ""]
    tot_rep = sum(sum(v["trainable_fits"] for v in rep[f]["fits"].values()) for f in E.FOLDS)
    tot_dep = sum(fit[f]["trainable_fits"] for f in E.FOLDS)
    lab_ok = all(fit[f]["label_first_read_unix"] > fit[f]["matrices"]["frozen_unix"] for f in E.FOLDS)
    rep_ok = all(rep[f]["held_month_arrival_outcomes_first_read_unix"] > rep[f]["frozen"]["unix_time"]
                 for f in E.FOLDS)
    out += [f"Trainable fits: {tot_rep} representation + {tot_dep} departure = **{tot_rep + tot_dep}** "
            f"(bound 108). Departure label first read after the frozen matrices on every fold: "
            f"**{lab_ok}**. Held-month arrival outcomes first read after the frozen representation on "
            f"every fold: **{rep_ok}**. Constant arm's matrix equals the row-only arm's on every fold: "
            f"**{all(fit[f]['matrices']['const_equals_rowonly_matrix'] for f in E.FOLDS)}**; "
            f"its h equals row-only's: **{all(fit[f]['const_equals_rowonly_h'] for f in E.FOLDS)}**. "
            f"Every reconciliation holds: **{it['all_reconcile']}**. Pricing job: {it['seconds']} s.", ""]

    # ---- T2 representation fits ---------------------------------------------------------------
    out += ["## T2. The representation fits (fitting months only; factor sd over supported values, s)", ""]
    rows = []
    for f in E.FOLDS:
        r = rep[f]
        for a in E.ARMS_REP:
            v = r["fits"][a]
            rows.append([f, a, f"{v['target_mean_s']:.1f}", f"{v['target_sd_s']:.1f}",
                         f"{v['exported_sd_s']['stand']:.1f}", f"{v['exported_sd_s']['rwy']:.1f}",
                         f"{v['exported_sd_s']['op']:.1f}", f"{v['rmse_in_sample']:.1f}",
                         f"{v['max_change_last_sweep_s']:.2f}",
                         ", ".join(f"{k} {x:,}" for k, x in (v.get("control") or {}).items()) or "–"])
    out += [table(["fold", "arm", "target mean", "target sd", "sd f_stand", "sd f_rwy", "sd f_op",
                   "in-sample RMSE", "last-sweep max change", "control detail"], rows), ""]
    rows = []
    for f in E.FOLDS:
        r = rep[f]
        rows.append([f, f"{r['fitting_arrivals']:,}", r["arrivals_without_outcome_dropped"],
                     " / ".join(f"{r['levels'][k]:,}" for k in E.BLOCKS_ARR),
                     " / ".join(f"{r['supported_values'][k]:,}" for k in E.KEYS),
                     f"{r['donor']['with_donor']:,}", f"{r['donor']['donor_shares_runway']:,}",
                     " / ".join(f"{r['donor']['gap_s_quantiles'][p]:.0f}" for p in ("50", "90", "99"))])
    out += [table(["fold", "fitting arrivals", "dropped (no outcome)",
                   "levels date / hour / stand / rwy / op", "supported stand / rwy / op",
                   "arrivals with a donor", "donor on the same runway", "donor gap p50 / p90 / p99 s"],
                  rows), ""]

    # ---- T3 auxiliary accuracy ----------------------------------------------------------------
    out += ["## T3. Auxiliary-task accuracy on the held month's arrivals (after the freeze; a report, "
            "never a gate)", "",
            "gain = mean(e0²) − mean((e0 − f_stand − f_rwy − f_op)²), e0 = taxi-in − the TRUE arm's "
            "dense prior (μ + the airport's mean date effect + g[apt, hour]); s² per arrival; 1-day "
            "interval.", ""]
    rows = []
    for f in E.FOLDS:
        acc = rep[f]["aux_accuracy"]
        for a in E.ARMS_REP:
            v = acc["arms"][a]
            bk = v["by_key"]
            rows.append([f, f"{acc['arrivals']:,}", f"{acc['rmse_prior']:.1f}", a,
                         f"{v['mse_gain_s2']:+,.1f}", ci(v["ci95_day"], 1),
                         f3(v["corr"], 3) if v["corr"] is not None else "–",
                         f"{bk['stand']['mse_gain_s2']:+,.1f}", f"{bk['rwy']['mse_gain_s2']:+,.1f}",
                         f"{bk['op']['mse_gain_s2']:+,.1f}", f"{v['rows_nonzero']:,}"])
    out += [table(["fold", "held arrivals", "prior RMSE s", "arm", "gain s²", "1-day interval",
                   "corr(e0, f)", "stand alone", "runway alone", "operator alone", "rows with a factor"],
                  rows), ""]

    # ---- T4 useful support and learner use ----------------------------------------------------
    out += ["## T4. Useful support on the departure rows and what the learner used", ""]
    rows = []
    for f in E.FOLDS:
        q = fit[f]
        for a in E.ARMS:
            nz = q["matrices"]["arms"][a]["rep_nonzero_rows"]
            spl = {}
            for b in q["fits"][a].values():
                for k, v in b["splits_by_feature"].items():
                    spl[k] = spl.get(k, 0) + v
            tot = sum(spl.values())
            raw = sum(spl.get(c, 0) for c in ("apt_stand", "apt_rwy", "op"))
            repn = sum(spl.get(c, 0) for c in E.REP_COLS)
            rows.append([f, a, f"{nz['stand']:,}", f"{nz['rwy']:,}", f"{nz['op']:,}", tot,
                         f"{raw} ({raw / tot:.0%})", repn,
                         ", ".join(f"{c} {spl[c]}" for c in E.REP_COLS if spl.get(c)) or "–"])
    out += [table(["fold", "arm", "rows with f_stand ≠ 0", "f_rwy ≠ 0", "f_op ≠ 0", "splits (5 block fits)",
                   "on raw apt_stand/apt_rwy/op", "on f_*", "which f_*"], rows), ""]
    out += ["Panel rows: " + ", ".join(f"{f} {fit[f]['rows']:,} (P {fit[f]['P_rows']:,})" for f in E.FOLDS)
            + ". Splits are counted over the 5 block fits of the fold (64 trees × up to 3 splits each).", ""]

    # ---- T5 board -----------------------------------------------------------------------------
    B = it["board"]
    out += ["## T5. Board (july_loo and january at the ranking shares) and december", "",
            "Board RMSE: " + ", ".join(f"{k} {v:.5f}" for k, v in it["board_rmse"].items()) + " s.", ""]
    rows = []
    for name in it["contrasts"]:
        b = B[name]
        dec = it["folds"]["december"]["contrasts"][name]
        rows.append([name.replace("_over_", " over "), f"{b['gain_s']:+.5f} {ci(b['ci_block_1d'], 5)}",
                     ci(b["ci_block_2d"], 5), ci(b["ci_block_3d"], 5), f"{b['gain_clean_s']:+.5f}",
                     f"{dec['rmse_gain_s']:+.5f}", "yes" if b["reconciles"] else "NO"])
    out += [table(["B over A", "board all-row gain s [1-day]", "2-day", "3-day", "board clean s",
                   "december RMSE gain s", "reconciles"], rows), ""]

    # ---- T6 per fold contrasts ----------------------------------------------------------------
    out += ["## T6. Every contrast per fold (s² per row, 1-day interval; rows where A and B differ)", ""]
    rows = []
    for name in it["contrasts"]:
        r = [name.replace("_over_", " over ")]
        for f in E.FOLDS:
            c = it["folds"][f]["contrasts"][name]
            r.append(f"{c['mse_gain_s2_per_row']:+.3f} {ci(c['ci_block_1d'])}")
            r.append(f"{c['rows_differing']:,}")
        rows.append(r)
    out += [table(["B over A", "july_loo", "rows ≠", "december", "rows ≠", "january", "rows ≠"], rows), ""]
    rows = []
    for name in ("true_over_rowonly", "rowonly_over_v22", "true_over_v22"):
        for f in E.FOLDS:
            c = it["folds"][f]["contrasts"][name]
            rows.append([name.replace("_over_", " over "), f, ci(c["ci_block_2d"]), ci(c["ci_block_3d"]),
                         f"{c['sse_gain']:+,.1f}", f"{c['sse_gain_rowsum']:+,.1f}",
                         "yes" if c["reconciles"] else "NO"])
    out += ["2- and 3-day intervals and the SSE reconciliation of the three contrasts that carry the "
            "reading (every contrast's are in `analysis/iter1.json`):", "",
            table(["B over A", "fold", "2-day", "3-day", "SSE_A − SSE_B", "Σ per-row v", "reconciles"], rows), ""]

    # ---- T7 predictors per fold and segment ---------------------------------------------------
    out += ["## T7. Every predictor per fold and segment (RMSE s; rows)", ""]
    segs = ["all", "le3600", "gt3600", "no_nm", "clean", "P", "not_P"] + [f"regime_{g}" for g in REGIMES]
    for f in E.FOLDS:
        T = it["folds"][f]["predictors"]
        n = T["v22"]["segments"]
        out += [f"**{f}** — rows: " + ", ".join(f"{s} {n[s]['rows']:,}" for s in segs), ""]
        rows = []
        for p in ["v22"] + E.ARMS:
            s = T[p]["segments"]
            rows.append([p, f"{T[p]['rows_changed_vs_v22']:,}", T[p]["rows_at_floor"]]
                        + [f3(s[x]["rmse"], 3, False) for x in segs])
        out += [table(["predictor", "rows changed vs v22", "rows at 60 s"] + segs, rows), ""]

    # ---- T8 segment gains of the reading contrasts --------------------------------------------
    out += ["## T8. Segment gains (s² per row in the segment) of the contrasts that carry the reading", ""]
    rows = []
    for name in ("true_over_rowonly", "rowonly_over_v22", "gmean_over_rowonly", "shuffle_over_rowonly",
                 "donor_over_rowonly"):
        for f in E.FOLDS:
            s = it["folds"][f]["contrasts"][name]["segments_mse_gain_s2_per_row"]
            rows.append([name.replace("_over_", " over "), f] + [f3(s[x], 3) for x in segs])
    out += [table(["B over A", "fold"] + segs, rows), ""]

    # ---- T9 airports --------------------------------------------------------------------------
    out += ["## T9. Airport SSE gains (s², whole fold) of the contrasts that carry the reading", ""]
    apts = sorted(it["folds"]["july_loo"]["contrasts"]["true_over_rowonly"]["by_airport_sse_gain"])
    rows = []
    for name in ("true_over_rowonly", "rowonly_over_v22"):
        for f in E.FOLDS:
            a = it["folds"][f]["contrasts"][name]["by_airport_sse_gain"]
            rows.append([name.replace("_over_", " over "), f] + [f"{a[x]:+,.0f}" for x in apts])
    out += [table(["B over A", "fold"] + apts, rows), ""]

    # ---- T10 support audit --------------------------------------------------------------------
    out += ["## T10. Key support on the held month's departures (label-free audit, `analysis/audit-r1.json`)", ""]
    rows = []
    for f in E.FOLDS:
        s = au["folds"][f]["support"]
        rows.append([f, f"{au['folds'][f]['P_rows']:,}"]
                    + [f"{s['keys'][k]['unsupported_mass_P']:.2%}" for k in E.KEYS]
                    + [f"{s['any_key_supported_share_P']:.2%}"])
    out += [table(["fold", "P rows", "unsupported stand", "unsupported runway", "unsupported operator",
                   "any key supported"], rows), ""]

    # ---- T11 readings -------------------------------------------------------------------------
    R = it["readings"]
    out += ["## T11. The registered readings (SPEC §1.11, applied by `price.py`)", ""]
    out += [table(["condition", "holds"], [[k, v] for k, v in R["conditions"].items()]
                  + [["**advances**", R["advances"]],
                     ["stricter reading (no negative fold point on every contrast of true)",
                      R["stricter_reading_no_negative_fold_point_on_every_contrast"]],
                     ["board gain true over rowonly (s)", f"{R['board_gain_true_over_rowonly_s']:+.6f}"],
                     ["board gain of gmean / donor / shuffle over rowonly (s)",
                      " / ".join(f"{R['board_gain_controls_over_rowonly_s'][c]:+.6f}"
                                 for c in ("gmean", "donor", "shuffle"))],
                     ["primary fold points s²/row",
                      ", ".join(f"{f} {R['primary_fold_points_s2'][f]:+.3f}" for f in E.FOLDS)],
                     ["primary fold 1-day lower bounds",
                      ", ".join(f"{f} {R['primary_fold_1d_lower'][f]:+.3f}" for f in E.FOLDS)],
                     ["route", f"{R['route']}: {R['route_text']}"]]), ""]

    # ---- T12 hashes ---------------------------------------------------------------------------
    out += ["## T12. Hashes", ""]
    rows = [["analysis/audit-r1.json", E.sha_file(E.EXP / "analysis/audit-r1.json")],
            ["analysis/iter1.json", E.sha_file(E.EXP / "analysis/iter1.json")]]
    for f in E.FOLDS:
        rows += [[f"analysis/rep-{f}.json", E.sha_file(E.EXP / f"analysis/rep-{f}.json")],
                 [f"work/rep-{f}.parquet (frozen representation)", rep[f]["frozen"]["sha256"]],
                 [f"analysis/fit-{f}.json", E.sha_file(E.EXP / f"analysis/fit-{f}.json")],
                 [f"work/h-{f}.parquet", fit[f]["h_parquet"]["sha256"]],
                 [f"oof/iter1-{f}.parquet", E.sha_file(E.EXP / f"oof/iter1-{f}.parquet")]]
    for f in E.FOLDS:
        for a in E.ARMS:
            rows.append([f"{f} matrix {a}", fit[f]["matrices"]["arms"][a]["matrix_sha256"]])
    code = {}
    for j in [rep[f] for f in E.FOLDS] + [fit[f] for f in E.FOLDS] + [it]:
        code.update(j["code_sha256"])
    rows += [[f"code/{k} (as recorded by the run)", v] for k, v in sorted(code.items())]
    out += [table(["artifact", "sha256"], rows), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT, help="output path, relative to the experiment")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = E.EXP / args.out
    NO.refuse_existing([OUT], args.overwrite)
    text = render()
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(text + "\n")
    print(f"written {OUT} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
