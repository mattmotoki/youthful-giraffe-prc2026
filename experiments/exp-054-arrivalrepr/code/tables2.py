"""exp-054-arrivalrepr: render the iteration-2 tables of RESULTS.md from the analysis JSON only.

Reads `analysis/fit2-<fold>.json` and `analysis/iter2.json`; computes nothing that is not a sum, a
count or a formatting of their values.  Writes `analysis/iter2-tables.md` (write-once; a
correction goes to a new name with --out).

    .venv/bin/python experiments/exp-054-arrivalrepr/code/tables2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e54 as E                                                       # noqa: E402
from tables import ci, f3, table                                      # noqa: E402  (iteration 1's helpers)

NO = E.NO
OUT_DEFAULT = "analysis/iter2-tables.md"
KL = ["true_k", "gmean_k", "donor_k", "shuffle_k", "rowonly_k"]
REGIMES = ["nm_bulk", "nm_late", "noac", "nonm"]
READING = ["true_k_over_rowonly_k", "true_k_over_rowonly", "rowonly_k_over_rowonly",
           "gmean_k_over_rowonly_k", "shuffle_k_over_rowonly_k", "donor_k_over_rowonly_k"]


def J(p: str) -> dict:
    return json.loads((E.EXP / p).read_text())


def nm(name: str) -> str:
    return name.replace("_over_", " over ")


def render() -> str:
    fit = {f: J(f"analysis/fit2-{f}.json") for f in E.FOLDS}
    it = J("analysis/iter2.json")
    out = ["# exp-054 iteration 2 — tables rendered by `code/tables2.py` from the analysis JSON", ""]

    # ---- T1 run record -----------------------------------------------------------------------
    out += ["## T1. Run record (from the records' own clocks and loads)", ""]
    rows = []
    for f in E.FOLDS:
        q = fit[f]
        fl = [q["fits"][a][b]["launch_load_1m"] for a in q["fits"] for b in q["fits"][a]]
        fs = [q["fits"][a][b]["seconds"] for a in q["fits"] for b in q["fits"][a]]
        la = [q["fits"][a][b]["launch_utc"] for a in q["fits"] for b in q["fits"][a]]
        rows.append([f, q["matrices"]["frozen_utc"], f"{min(la)} – {max(la)}", q["trainable_fits"],
                     f"{min(fl):.2f}–{max(fl):.2f}", f"{min(fs):.2f}–{max(fs):.2f}", f"{q['seconds']:.1f}",
                     all(q["matrices"]["iteration1_matrices_rebuilt_equal"].values()),
                     q["label_first_read_unix"] > q["matrices"]["frozen_unix"]])
    out += [table(["fold", "keyless matrices frozen (UTC)", "fit launches (UTC)", "fits", "launch load",
                   "s per fit", "job s", "iteration-1 matrices rebuilt equal", "label read after freeze"],
                  rows), ""]
    tot = sum(fit[f]["trainable_fits"] for f in E.FOLDS)
    out += [f"Trainable fits: 0 representation + {tot} departure = **{tot}** (D36 cap 75). Removed "
            f"columns: {', '.join(fit['july_loo']['matrices']['removed_columns'])}. Keyless features: "
            f"{', '.join(fit['july_loo']['matrices']['features'])}. Constant arm's keyless matrix equals "
            f"row-only's on every fold: "
            f"**{all(fit[f]['matrices']['const_equals_rowonly_matrix'] for f in E.FOLDS)}**. Every "
            f"reconciliation holds: **{it['all_reconcile']}**. Pricing job: {it['seconds']} s.", ""]

    # ---- T2 learner use ----------------------------------------------------------------------
    out += ["## T2. What the keyless learner used (splits over the 5 block fits of the fold)", ""]
    rows = []
    for f in E.FOLDS:
        for a in KL:
            spl = {}
            for b in fit[f]["fits"][a].values():
                for k, v in b["splits_by_feature"].items():
                    spl[k] = spl.get(k, 0) + v
            tot = sum(spl.values())
            h = [fit[f]["fits"][a][b]["h_rms_scored"] for b in fit[f]["fits"][a]]
            rows.append([f, a, tot] + [spl.get(c, 0) for c in ("f_stand", "f_rwy", "f_op", "apt", "ac_type", "hour")]
                        + [sum(v for k, v in spl.items() if k in ("dow", "to_aobt", "to_sched", "to_lobt",
                                                                   "to_eobt1", "aobt_lobt", "wtc")),
                           f"{min(h):.1f}–{max(h):.1f}"])
    out += [table(["fold", "arm", "splits", "f_stand", "f_rwy", "f_op", "apt", "ac_type", "hour",
                   "other row features", "h RMS per block s"], rows), ""]

    # ---- T3 board ----------------------------------------------------------------------------
    B = it["board"]
    out += ["## T3. Board (july_loo and january at the ranking shares) and december", "",
            "Board RMSE: " + ", ".join(f"{k} {v:.5f}" for k, v in it["board_rmse"].items()) + " s.", ""]
    rows = []
    for name in it["contrasts"]:
        b = B[name]
        dec = it["folds"]["december"]["contrasts"][name]
        rows.append([nm(name), f"{b['gain_s']:+.5f} {ci(b['ci_block_1d'], 5)}", ci(b["ci_block_2d"], 5),
                     ci(b["ci_block_3d"], 5), f"{b['gain_clean_s']:+.5f}", f"{dec['rmse_gain_s']:+.5f}",
                     "yes" if b["reconciles"] else "NO"])
    out += [table(["B over A", "board all-row gain s [1-day]", "2-day", "3-day", "board clean s",
                   "december RMSE gain s", "reconciles"], rows), ""]

    # ---- T4 per fold -------------------------------------------------------------------------
    out += ["## T4. Every contrast per fold (s² per row, 1-day interval; rows where A and B differ)", ""]
    rows = []
    for name in it["contrasts"]:
        r = [nm(name)]
        for f in E.FOLDS:
            c = it["folds"][f]["contrasts"][name]
            r += [f"{c['mse_gain_s2_per_row']:+.3f} {ci(c['ci_block_1d'])}", f"{c['rows_differing']:,}"]
        rows.append(r)
    out += [table(["B over A", "july_loo", "rows ≠", "december", "rows ≠", "january", "rows ≠"], rows), ""]
    rows = []
    for name in READING[:3]:
        for f in E.FOLDS:
            c = it["folds"][f]["contrasts"][name]
            rows.append([nm(name), f, ci(c["ci_block_2d"]), ci(c["ci_block_3d"]), f"{c['sse_gain']:+,.1f}",
                         f"{c['sse_gain_rowsum']:+,.1f}", "yes" if c["reconciles"] else "NO"])
    out += ["2- and 3-day intervals and the SSE reconciliation of the contrasts that carry the reading "
            "(every contrast's are in `analysis/iter2.json`):", "",
            table(["B over A", "fold", "2-day", "3-day", "SSE_A − SSE_B", "Σ per-row v", "reconciles"], rows), ""]

    # ---- T5 predictors per fold and segment --------------------------------------------------
    out += ["## T5. Every predictor per fold and segment (RMSE s; rows)", ""]
    segs = ["all", "le3600", "gt3600", "no_nm", "clean", "P", "not_P"] + [f"regime_{g}" for g in REGIMES]
    for f in E.FOLDS:
        T = it["folds"][f]["predictors"]
        n = T["v22"]["segments"]
        out += [f"**{f}** — rows: " + ", ".join(f"{s} {n[s]['rows']:,}" for s in segs), ""]
        rows = []
        for p in ["v22", "rowonly", "true"] + KL:
            s = T[p]["segments"]
            rows.append([p, f"{T[p]['rows_changed_vs_v22']:,}", T[p]["rows_at_floor"]]
                        + [f3(s[x]["rmse"], 3, False) for x in segs])
        out += [table(["predictor", "rows changed vs v22", "rows at 60 s"] + segs, rows), ""]

    # ---- T6 segment gains --------------------------------------------------------------------
    out += ["## T6. Segment gains (s² per row in the segment) of the contrasts that carry the reading", ""]
    rows = []
    for name in READING:
        for f in E.FOLDS:
            s = it["folds"][f]["contrasts"][name]["segments_mse_gain_s2_per_row"]
            rows.append([nm(name), f] + [f3(s[x], 3) for x in segs])
    out += [table(["B over A", "fold"] + segs, rows), ""]

    # ---- T7 airports -------------------------------------------------------------------------
    out += ["## T7. Airport SSE gains (thousand s², whole fold) of the contrasts that carry the reading", ""]
    apts = sorted(it["folds"]["july_loo"]["contrasts"]["true_k_over_rowonly_k"]["by_airport_sse_gain"])
    rows = []
    for name in READING:
        for f in E.FOLDS:
            a = it["folds"][f]["contrasts"][name]["by_airport_sse_gain"]
            rows.append([nm(name), f] + [f"{a[x] / 1e3:+,.0f}" for x in apts])
    out += [table(["B over A", "fold"] + apts, rows), ""]

    # ---- T8 readings -------------------------------------------------------------------------
    R = it["readings"]
    rel, spc, sub = (R["relevance_true_k_over_rowonly_k"], R["specificity_true_k_over_each_control"],
                     R["substitution_true_k_over_iteration1_rowonly"])
    out += ["## T8. The registered readings (SPEC §2.4, §2.6, applied by `price2.py`)", ""]
    out += [table(["condition or reading", "value"], [[k, v] for k, v in R["conditions"].items()]
                  + [["**advances**", R["advances"]],
                     ["primary fold points s²/row", ", ".join(f"{f} {R['primary_fold_points_s2'][f]:+.3f}" for f in E.FOLDS)],
                     ["primary fold 1-day lower bounds", ", ".join(f"{f} {R['primary_fold_1d_lower'][f]:+.3f}" for f in E.FOLDS)],
                     ["D36 fold points s²/row", ", ".join(f"{f} {R['d36_fold_points_s2'][f]:+.3f}" for f in E.FOLDS)],
                     ["D36 fold 1-day lower bounds", ", ".join(f"{f} {R['d36_fold_1d_lower'][f]:+.3f}" for f in E.FOLDS)],
                     ["relevance: true_k over rowonly_k, board s [1-day]", f"{rel['gain_s']:+.5f} {ci(rel['ci_block_1d'], 5)}; holds {rel['holds']}"],
                     ["specificity: true_k over each control, board s [1-day]",
                      "; ".join(f"{c} {v['gain_s']:+.5f} {ci(v['ci_block_1d'], 5)}" for c, v in spc["by_control"].items())
                      + f"; holds {spc['holds']}"],
                     ["substitution: true_k over iteration 1's rowonly, board s [1-day]", f"{sub['gain_s']:+.5f} {ci(sub['ci_block_1d'], 5)}"],
                     ["raw-key cost: rowonly_k over rowonly, board s", f"{R['raw_key_cost_rowonly_k_over_rowonly_s']:+.5f}"],
                     ["recovered share (board MSE)", f"{R['recovered_share_board_mse']:+.4f}"]]), ""]

    # ---- T9 hashes ---------------------------------------------------------------------------
    out += ["## T9. Hashes", ""]
    rows = [["analysis/iter2.json", E.sha_file(E.EXP / "analysis/iter2.json")]]
    for f in E.FOLDS:
        rows += [[f"analysis/fit2-{f}.json", E.sha_file(E.EXP / f"analysis/fit2-{f}.json")],
                 [f"work/h2-{f}.parquet", fit[f]["h_parquet"]["sha256"]],
                 [f"oof/iter2-{f}.parquet", E.sha_file(E.EXP / f"oof/iter2-{f}.parquet")],
                 [f"{f} frozen representation (iteration 1)", fit[f]["matrices"]["representation_sha256"]]]
    for f in E.FOLDS:
        for a in KL:
            rows.append([f"{f} keyless matrix {a}", fit[f]["matrices"]["arms"][a]["matrix_sha256"]])
        rows.append([f"{f} keyless matrix const_k (= rowonly_k)", fit[f]["matrices"]["arms"]["const_k"]["matrix_sha256"]])
    code = {}
    for j in [fit[f] for f in E.FOLDS] + [it]:
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
