"""Generate every table of RESULTS.md from the records in `analysis/`.

The report holds no hand-typed number: each `<!-- TABLES:x -->` block is
replaced from `analysis/*.json`.  Also writes the out-of-fold predictions of the
two ridge arms and the manifest that addresses the large binaries.

    python tables.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zs                                   # noqa: E402
import packet as PK                         # noqa: E402

EXP = zs.EXP
A = EXP / "analysis"
FOLDS = zs.FOLDS
DEN = json.loads((A / "density.json").read_text())
LRN = json.loads((A / "learner.json").read_text())
CTL = json.loads((A / "controls.json").read_text())
SRV = json.loads((A / "serve.json").read_text())
SRV11 = json.loads((A / "serve-v11.json").read_text())
V11 = json.loads((A / "v11base.json").read_text())
RPR = json.loads((A / "repair.json").read_text())
REC = json.loads((A / "records.json").read_text())
VDT = json.loads((A / "verdict.json").read_text())


def tbl(rows: list[list], head: list[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def t_density() -> str:
    s = ["### T1 the density check — the board on three grids\n"]
    rows = []
    for g in ("300", "600", "1200"):
        b = DEN["board"][g]
        n = sum(DEN["folds"][f"{g}|{f}"]["n_scored_cohort"] for f in FOLDS)
        rows.append([f"{g} s", f"{n:,}",
                     " / ".join(f"{DEN['folds'][f'{g}|{f}']['n_cells']:,}" for f in FOLDS),
                     f"{b['all_row_v10']:.4f}", f"{b['all_row_arm']:.4f}",
                     f"**{b['gain_all_row']:+.4f}**", f"{b['gain_clean']:+.4f}"])
    s.append(tbl(rows, ["grid", "scored rows", "cells (jul/dec/jan)", "all-row v10",
                        "all-row arm", "all-row gain", "clean gain"]))
    s.append("\n### T2 pair reduction by cell rows and by traffic rate, matched subsets (s²)\n")
    cb = sorted({k for g in DEN["pooled"] for k in DEN["pooled"][g]["matched"]["by_cell_rows"]},
                key=lambda x: int(x.split("-")[0].rstrip("+")))
    rb = sorted({k for g in DEN["pooled"] for k in DEN["pooled"][g]["matched"]["by_traffic_rate"]},
                key=lambda x: int(x.split("-")[0].rstrip("+")))
    rows = []
    for g in ("300", "600", "1200"):
        P = DEN["pooled"][g]["matched"]
        rows.append([f"{g} s"] + [
            (f"{P['by_cell_rows'][k]['reduction_s2']:+.1f} ({P['by_cell_rows'][k]['n']:,})"
             if k in P["by_cell_rows"] else "–") for k in cb])
    s.append("by **cell rows** (the stratum Q03's T6 used):\n")
    s.append(tbl(rows, ["grid"] + [f"{k} rows" for k in cb]))
    rows = []
    for g in ("300", "600", "1200"):
        P = DEN["pooled"][g]["matched"]
        rows.append([f"{g} s"] + [
            (f"{P['by_traffic_rate'][k]['reduction_s2']:+.1f} ({P['by_traffic_rate'][k]['n']:,})"
             if k in P["by_traffic_rate"] else "–") for k in rb])
    s.append("\nby **traffic rate** (eligible departures at the airport within ±300 s, "
             "the same physical measure on every grid):\n")
    s.append(tbl(rows, ["grid"] + [f"{k} dep/10 min" for k in rb]))
    s.append("\n### T2b the predeclared rule applied to the two-way table\n")
    s.append("Adjacent-band comparisons that **fall** in the declared direction, inside each "
             "level of the other factor (`code/verdict.py` on `code/density.py`'s own "
             "two-way table):\n")
    rows = []
    for g in ("300", "600", "1200"):
        d = VDT["grids"][g]
        a = d["traffic_given_cell_rows"]
        b = d["cell_rows_given_traffic"]
        rows.append([f"{g} s", f"{a['n_falling']} of {a['n_pairs']}",
                     f"{b['n_falling']} of {b['n_pairs']}",
                     VDT["traffic_marginal_monotone"][g]["monotone_decreasing"]])
    pt = VDT["pooled"]["traffic_given_cell_rows"]
    pc = VDT["pooled"]["cell_rows_given_traffic"]
    rows.append(["**all three**", f"**{pt[0]} of {pt[1]}**", f"**{pc[0]} of {pc[1]}**", "-"])
    s.append(tbl(rows, ["grid", "traffic ordering, holding cell rows",
                        "cell-row ordering, holding traffic",
                        "traffic marginal monotone"]))
    s.append(f"\nBoth orderings survive on a majority of comparisons, so the declared rule "
             f"returns **{VDT['verdict']}**. The rule says \"survives\" without saying how to "
             "count, so three honest countings are run; the verdict does not rest on the "
             "choice:\n")
    rows = []
    LAB = (("adjacent_bands", "adjacent bands, unweighted (the headline)"),
           ("pair_weighted", "adjacent bands, weighted by the smaller band's pairs"),
           ("first_band_against_last", "the first band against the last, per level"),
           ("every_comparison_must_fall", "strict: one reversal ends the ordering"))
    for k, lab in LAB:
        v = VDT["other_readings"][k]
        w = VDT.get("other_readings_no_min_pairs", {}).get(k)
        a, b = v["traffic_given_cell_rows"], v["cell_rows_given_traffic"]
        rows.append([lab, f"{a[0]:g} of {a[1]:g}", f"{b[0]:g} of {b[1]:g}",
                     f"**{v['verdict']}**",
                     f"{w['verdict']}" if w else "-"])
    s.append(tbl(rows, ["how \"survives\" is counted", "traffic ordering",
                        "cell-row ordering", "verdict",
                        "verdict without the undeclared 200-pair minimum"]))
    s.append(f"\nNo counting returns DENSITY and none returns MECHANISM. The "
             f"{VDT['two_way_min_pairs'] if 'two_way_min_pairs' in VDT else 200}-pair "
             "minimum on a two-way cell was an implementation choice and was not "
             "predeclared; the last column runs the same countings on every non-empty "
             "cell, and no verdict changes.")
    s.append("")
    s.append("The full two-way table at Q03's own 600 s grid, matched subsets (s², pairs):\n")
    T = DEN["pooled"]["600"]["matched"]["table"]
    cbs = sorted({r["cell_rows"] for r in T}, key=lambda x: int(x.split("-")[0].rstrip("+")))
    rbs = sorted({r["traffic_rate"] for r in T}, key=lambda x: int(x.split("-")[0].rstrip("+")))
    cellv = {(r["cell_rows"], r["traffic_rate"]): r for r in T}
    rows = []
    for c in cbs:
        row = [f"{c} rows"]
        for rb_ in rbs:
            v = cellv.get((c, rb_))
            row.append(f"{v['reduction_s2']:+.1f} ({v['n']:,})" if v else "-")
        rows.append(row)
    s.append(tbl(rows, ["cell rows"] + [f"{k} dep/10 min" for k in rbs]))
    return "\n".join(s)


def t_learner() -> str:
    s = ["### T3 the five arms under both cross-fittings — all-row board gain over v10\n"]
    rows = []
    for a in LRN["arms"]:
        r = []
        for c in ("dayblock", "xmonth"):
            b = LRN["board"][f"{a}|{c}"]
            r.append(f"{b['gain_all_row']:+.4f}")
        f_day = LRN["folds"][f"{a}|dayblock"]
        f_x = LRN["folds"][f"{a}|xmonth"]
        rows.append([f"`{a}`",
                     " / ".join(f"{LRN['amplitudes'][a][f]:.4f}" for f in FOLDS),
                     f"{LRN['amplitudes'][a]['pooled']:.4f}", r[0], r[1],
                     f"{LRN['board'][f'{a}|dayblock']['gain_clean']:+.4f}",
                     f"{LRN['board'][f'{a}|xmonth']['gain_clean']:+.4f}",
                     " / ".join(f"{f_day[f]['gain_all']:+.4f}" for f in FOLDS),
                     " / ".join(f"{f_x[f]['gain_all']:+.4f}" for f in FOLDS)])
    s.append(tbl(rows, ["arm", "cross-month amp (jul/dec/jan)", "pooled amp",
                        "dayblock all-row", "**xmonth all-row**", "dayblock clean",
                        "xmonth clean", "dayblock folds (jul/dec/jan)",
                        "xmonth folds (jul/dec/jan)"]))
    s.append("\n### T4 2A − B per scored row with day-clustered intervals (s²)\n")
    rows = []
    for a in LRN["arms"]:
        for c in ("dayblock", "xmonth"):
            r = LRN["folds"][f"{a}|{c}"]
            rows.append([f"`{a}`", c] + [
                f"{r[f]['mse_gain_s2']:+.1f} [{r[f]['ci'][0]:+.0f}, {r[f]['ci'][1]:+.0f}]"
                for f in FOLDS] + [
                " / ".join(f"{r[f]['unrestricted_scale']:.2f}" for f in FOLDS)])
    s.append(tbl(rows, ["arm", "cross-fit"] + list(FOLDS) + ["A/B (jul/dec/jan)"]))
    s.append("\n### T5 the two algebraic checks, every arm and both cross-fittings\n")
    rows = []
    for a in LRN["arms"]:
        for c in ("dayblock", "xmonth"):
            r = LRN["folds"][f"{a}|{c}"]
            rows.append([f"`{a}`", c,
                         f"{max(r[f]['max_abs_cell_sum'] for f in FOLDS):.2e}",
                         all(r[f]["cell_sum_check"] for f in FOLDS),
                         f"{max(r[f]['dsse_rel_gap'] for f in FOLDS):.2e}",
                         all(r[f]["dsse_check"] for f in FOLDS)])
    s.append(tbl(rows, ["arm", "cross-fit", "max abs cell sum", "sums zero",
                        "max rel gap full vs within-cell", "identity holds"]))
    s.append("\n### T6 the pair diagnostic at the same amplitude (s², positive is better)\n")
    rows = []
    for a in LRN["arms"]:
        for c in ("dayblock", "xmonth"):
            r = LRN["folds"][f"{a}|{c}"]
            rows.append([f"`{a}`", c] + [
                f"{r[f]['pair']['reduction_s2']:+.1f} [{r[f]['pair']['ci'][0]:+.0f}, "
                f"{r[f]['pair']['ci'][1]:+.0f}]" for f in FOLDS])
    s.append(tbl(rows, ["arm", "cross-fit"] + list(FOLDS)))
    return "\n".join(s)


def t_controls() -> str:
    s = ["### T7 the four fitting designs, in order of distance in time (all-row board gain)\n"]
    rows = []
    for a in CTL["arms"]:
        rows.append([f"`{a}`",
                     f"{LRN['board'][f'{a}|dayblock']['gain_all_row']:+.4f}",
                     f"{CTL['board'][f'{a}|dayblock_frozen']['gain_all_row']:+.4f}",
                     f"{CTL['board'][f'{a}|halfmonth']['gain_all_row']:+.4f}",
                     f"**{LRN['board'][f'{a}|xmonth']['gain_all_row']:+.4f}**"])
    s.append(tbl(rows, ["arm", "dayblock (Q03)", "dayblock, frozen vocabulary",
                        "half-month", "**cross-month**"]))
    s.append("\nPer fold, the half-month control:\n")
    rows = []
    for a in CTL["arms"]:
        r = CTL["folds"][f"{a}|halfmonth"]
        rows.append([f"`{a}`"] + [
            f"{r[f]['gain_all']:+.4f} ({r[f]['mse_gain_s2']:+.0f} "
            f"[{r[f]['ci'][0]:+.0f}, {r[f]['ci'][1]:+.0f}])" for f in FOLDS])
    s.append(tbl(rows, ["arm"] + list(FOLDS)))
    return "\n".join(s)


def t_serve() -> str:
    c = SRV["census"]
    s = ["### T8 the served candidate\n"]
    rows = [["base", "v10 (the assignment's)", "v11 (the sibling, §7)"],
            ["file", f"`{SRV['candidate']}`", f"`{SRV11['candidate']}`"],
            ["SHA-256", f"`{SRV['sha256']}`", f"`{SRV11['sha256']}`"],
            ["bytes", f"{SRV['bytes']:,}", f"{SRV11['bytes']:,}"],
            ["rows changed", f"{c['n_changed']:,}", f"{SRV11['census']['n_changed']:,}"],
            ["min prediction", f"{SRV['prediction']['min']:.4f}",
             f"{SRV11['prediction']['min']:.4f}"],
            ["max abs cell sum of d", f"{SRV['checks']['max_abs_cell_sum']:.2e}",
             f"{SRV11['checks']['max_abs_cell_sum']:.2e}"],
            [f"{SRV['identical_to_v10']['noac'][0]} noac rows, max abs difference "
             f"from v10", f"{SRV['identical_to_v10']['noac'][1]:.1f}",
             f"{SRV11['identical_to_v10']['vs_v10_bytes_noac']:.1f}"],
            [f"{SRV['identical_to_v10']['EHAM'][0]:,} EHAM rows, max abs difference "
             f"from v10", f"{SRV['identical_to_v10']['EHAM'][1]:.1f}",
             f"{SRV11['identical_to_v10']['vs_v10_bytes_EHAM']:.1f}"],
            [f"{SRV['identical_to_v10']['EDDF'][0]:,} EDDF rows, max abs update "
             f"(zero by eligibility)", f"{SRV['identical_to_v10']['EDDF'][1]:.1f}",
             f"{SRV11['identical_to_v10']['EDDF'][1]:.1f}"]]
    s.append(tbl(rows, ["item", "v10 base", "v11 base"]))
    s.append("\n### T9 the ranking-side census\n")
    fold_rms = {f: LRN["folds"]["ridge|dayblock"][f]["rms_update"] for f in FOLDS}
    fold_rms_x = {f: LRN["folds"]["ridge|xmonth"][f]["rms_update"] for f in FOLDS}
    rows = [["ranking departures", f"{c['n_ranking_departures']:,}"],
            ["eligible (`nm_bulk` outside EHAM and EDDF)",
             f"{c['n_eligible']:,} ({c['eligible_pct']:.2f} %)"],
            ["cells holding an eligible row", f"{c['n_cells_eligible']:,}"],
            ["scored cohort (cells with ≥ 4 eligible rows)", f"{c['n_scored_cohort']:,}"],
            ["eligible rows in cells under four rows",
             f"{c['share_eligible_in_cells_under_four']:.2f} %"],
            ["rows changed", f"{c['n_changed']:,}"],
            ["RMS of the served update, over the scored cohort",
             f"{c['rms_update_all_scored']:.3f} s"],
            ["RMS over every eligible row (scored or not)",
             f"{c['rms_update_over_eligible']:.3f} s"],
            ["RMS over all 344,841 departures",
             f"{c['rms_update_over_all_departures']:.3f} s"],
            ["max abs update", f"{c['max_abs_update']:.1f} s"],
            ["fold-side RMS, `dayblock`",
             " / ".join(f"{fold_rms[f]:.2f}" for f in FOLDS) + " s"],
            ["fold-side RMS, `xmonth`",
             " / ".join(f"{fold_rms_x[f]:.2f}" for f in FOLDS) + " s"]]
    s.append(tbl(rows, ["item", "value"]))
    s.append("\nBy ranking month and by airport:\n")
    rows = [[m, f"{v['n_departures']:,}", f"{v['n_eligible']:,}", f"{v['n_scored']:,}",
             f"{v['rms_update']:.3f}"] for m, v in c["by_month"].items()]
    rows += [[a, f"{v['n_departures']:,}", f"{v['n_eligible']:,}", f"{v['n_scored']:,}",
              f"{v['rms_update']:.3f}"] for a, v in c["by_airport"].items()]
    s.append(tbl(rows, ["group", "departures", "eligible", "scored", "RMS update (s)"]))
    s.append("\n### T10 the same update on the v11 base, fold side\n")
    s.append("`cross-month` is the per-fold amplitude fitted on the other two folds only - the "
             "like-for-like row. `pooled` is the single scalar the written file carries; it is "
             "fitted on all three folds, including the one it scores, and is shown for that "
             "reason alone.\n")
    rows = []
    for cf in ("dayblock", "xmonth"):
        for ak in ("cross_month", "pooled"):
            v = V11["arms"][f"{cf}|{ak}"]
            g = f"{v['gain_over_v11']:+.4f}"
            rows.append([cf, ak.replace("_", "-"), f"{v['board_v10']:.4f}",
                         f"{v['board_v11']:.4f}", f"{v['board_v11_plus_d']:.4f}",
                         f"**{g}**" if ak == "cross_month" else g,
                         f"{v['gain_over_v10_total']:+.4f}",
                         " / ".join(f"{v['folds'][f]['gain_all_over_v11']:+.4f}" for f in FOLDS)])
    s.append(tbl(rows, ["cross-fit", "amplitude", "v10 board", "v11 board", "v11 + d board",
                        "gain over v11", "total gain over v10", "folds (jul/dec/jan)"]))
    return "\n".join(s)


def t_repair() -> str:
    s = ["### T11 Sol round 1 repairs - the exact Hessian, the shared vocabulary, "
         "and two further controls\n"]
    s.append("Every tree arm refitted with the objective's **exact** Hessian diagonal "
             "`1 - 1/n_c` in place of the unit diagonal, and both shipped arms refitted with "
             "the 13 shared-vocabulary `c_*` columns dropped (76 shipped columns, not 89). "
             "Each repaired arm carries **its own** cross-month amplitude, fitted on the other "
             "two folds' day-block updates of that arm (Sol round 2, blocking 1: the first "
             "version of this table reused the amplitudes of the vocabulary-bearing run, which "
             "made these fixed-amplitude perturbations rather than refits). All-row board gain "
             "over v10:\n")
    rows = []
    for a in ("lgb_small", "lgb_wide", "lgb_shipped", "lgb_shipped_wide"):
        tag = "hess_diag_nocachevocab" if "shipped" in a else "hess_diag"
        r = []
        for c in ("dayblock", "xmonth"):
            r += [f"{LRN['board'][f'{a}|{c}']['gain_all_row']:+.4f}",
                  f"{RPR['board'][f'{a}|{c}|{tag}']['gain_all_row']:+.4f}"]
        amp = RPR["amplitudes"][f"{a}|xmonth|{tag}"]
        rows.append([f"`{a}`"] + r
                    + [" / ".join(f"{amp[f]:.3f}" for f in FOLDS),
                       " / ".join(f"{LRN['amplitudes'][a][f]:.3f}" for f in FOLDS)])
    s.append(tbl(rows, ["arm", "dayblock as reported", "dayblock repaired",
                        "xmonth as reported", "**xmonth repaired**",
                        "repaired amplitude (jul/dec/jan)", "reported amplitude"]))
    s.append("\nThe repaired `xmonth` fold intervals, to show that the gate still fails "
             "(2A - B per scored row, s²):\n")
    rows = []
    for a in ("lgb_small", "lgb_wide", "lgb_shipped", "lgb_shipped_wide"):
        tag = "hess_diag_nocachevocab" if "shipped" in a else "hess_diag"
        r = RPR["folds"][f"{a}|xmonth|{tag}"]
        rows.append([f"`{a}`"] + [
            f"{r[f]['mse_gain_s2']:+.0f} [{r[f]['ci'][0]:+.0f}, {r[f]['ci'][1]:+.0f}]"
            for f in FOLDS])
    s.append(tbl(rows, ["arm"] + list(FOLDS)))
    s.append("\nThe two further controls on the cause of the cross-month fall "
             "(`code/repair.py`):\n")
    rows = []
    for k, what in (("ridge|pathcheck|-",
                     "the **cross-month code path**, run inside the month (fit on four of "
                     "five day blocks, score the fifth, through the `xmonth` fitting and "
                     "scoring functions)"),
                    ("ridge|xmonth_one_1|-",
                     "fit on **one** other month, scored across the month boundary "
                     "(first source month)"),
                    ("ridge|xmonth_one_2|-",
                     "fit on **one** other month, the second source month")):
        b = RPR["board"][k]
        f = RPR["folds"][k]
        rows.append([f"`{k.split('|')[1]}`", what, f"{b['gain_all_row']:+.4f}",
                     " / ".join(f"{f[x]['gain_all']:+.4f}" for x in FOLDS)])
    s.append(tbl(rows, ["control", "what it is", "all-row board gain",
                        "folds (jul/dec/jan)"]))
    src = RPR["single_month_fit_source"]
    s.append("\nThe single fitting month per fold: `xmonth_one_1` "
             + ", ".join(f"{f} <- {src['xmonth_one_1'][f]}" for f in FOLDS)
             + "; `xmonth_one_2` "
             + ", ".join(f"{f} <- {src['xmonth_one_2'][f]}" for f in FOLDS) + ".")
    return "\n".join(s)


def t_records() -> str:
    s = ["### T12 the COMMON.md slice decomposition, v10 against both ridge designs\n"]
    rows = []
    for f in FOLDS:
        R = REC["slices"][f]
        for a in R:
            v = R[a]
            rows.append([f, "`" + a.replace("|", " / ") + "`", f"{v['n']:,}",
                         f"{v['rmse_all']:.4f}",
                         f"{v['rmse_le3600']:.4f}", f"{v['rmse_gt3600']:.2f}",
                         f"{v['rmse_nonm']:.2f}"])
    s.append("Every arm on the record, not only the ridge (Sol round 2, blocking 4). Each arm "
             "is scored at its own cross-month amplitude, so these are the same predictions "
             "the gain tables are computed from.\n")
    s.append(tbl(rows, ["fold", "arm", "rows", "RMSE all", "RMSE <= 3600 s",
                        "RMSE > 3600 s", "RMSE no-NM"]))
    n = REC["slices"]["july_loo"]["v10"]
    s.append(f"\nSlice sizes are the same for every arm inside a fold "
             f"(july_loo {n['n_le3600']:,} at or under 3600 s, {n['n_gt3600']:,} above, "
             f"{n['n_nonm']:,} without a network-manager record). The no-NM slice is "
             "identical to v10 in every arm, because those rows are not `nm_bulk` and "
             "carry no update.")
    return "\n".join(s)


def write_oof() -> None:
    """The out-of-fold predictions COMMON.md asks for, for both ridge designs."""
    import learner as L
    pan = zs.V.load()
    H = np.load(A / "h_vectors.npz")
    amp = LRN["amplitudes"]["ridge"]
    d = EXP / "oof"
    d.mkdir(exist_ok=True)
    for cf in ("dayblock", "xmonth"):
        for f in FOLDS:
            S = L.fold_state(f, pan[f])
            m0 = S["m0"]
            cm, ucm = pd.factorize(S["code"][m0])
            u = np.zeros(len(m0))
            u[m0] = amp[f] * zs.project(H[f"{f}|ridge|{cf}"][m0].astype("float64"),
                                        cm, len(ucm))
            pd.DataFrame({"MVT_ID_mvt": S["ids"], "fold": f, "y_true": S["y"],
                          "y_pred": S["p10"] + u}).to_parquet(
                d / f"exp-044-ridge-{cf}-{f}.parquet", index=False, compression="zstd")


def manifest() -> None:
    d = A
    (d / ".gitignore").write_text("*.npz\n*.npy\n")
    out = []
    for f in sorted(list(d.glob("*.npz")) + list(d.glob("*.npy"))):
        out.append({"file": f"experiments/exp-044-zerosum/analysis/{f.name}",
                    "bytes": f.stat().st_size, "sha256": PK.sha_file(f)})
    for f in sorted((d / "rows").glob("*.parquet")):     # the per-row contributions
        out.append({"file": f"experiments/exp-044-zerosum/analysis/rows/{f.name}",
                    "bytes": f.stat().st_size, "sha256": PK.sha_file(f)})
    for f in sorted((zs.ROOT / "submissions/candidates").glob("exp-044-zerosum*.parquet")):
        out.append({"file": f"submissions/candidates/{f.name}",
                    "bytes": f.stat().st_size, "sha256": PK.sha_file(f)})
    (d / "MANIFEST.json").write_text(json.dumps(
        {"note": "large binaries and the candidate files this record was written from; "
                 "rebuild with code/learner.py and code/serve.py.  `h_vectors.npz` holds "
                 "every arm's held-out h per fold and cross-fitting; `served_update.npy` "
                 "is (MVT_ID_mvt, d) for the 344,841 ranking departures; `rows/` holds the "
                 "per-row contribution tables code/records.py writes.",
         "files": out}, indent=2))


def main() -> None:
    if "--oof" in sys.argv:
        write_oof()
    manifest()
    p = EXP / "RESULTS.md"
    text = p.read_text()
    for name, fn in (("density", t_density), ("learner", t_learner),
                     ("controls", t_controls), ("serve", t_serve),
                     ("repair", t_repair), ("records", t_records)):
        block = (f"<!-- TABLES:{name} -->\n<!-- generated by code/tables.py; "
                 f"do not edit by hand -->\n\n{fn()}\n<!-- /TABLES:{name} -->")
        pat = re.compile(rf"<!-- TABLES:{name} -->.*?<!-- /TABLES:{name} -->", re.S)
        if not pat.search(text):
            raise SystemExit(f"RESULTS.md has no TABLES:{name} block")
        text = pat.sub(lambda _: block, text)
    p.write_text(text)
    print(f"tables written into {p}")


if __name__ == "__main__":
    main()
