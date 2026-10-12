"""Every numeric table in RESULTS.md, generated from the analysis records.

exp-039's Sol round 1 (blocking finding 7) found a report carrying intervals from
a superseded run.  Prose cannot be kept in step with a rerun by hand, so the
tables are generated here into `analysis/tables.md` and injected into RESULTS.md
between markers; `tests/check_reports.py` then asserts the injected block is
byte-identical to the generated file and that every number the prose quotes is a
literal substring of the report.

    python tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
A = EXP / "analysis"
FOLDS = ("july_loo", "december", "january")
BEGIN = "<!-- TABLES:exp041 -->"
END = "<!-- /TABLES -->"
REC = {n: json.loads((A / f"{n}.json").read_text())
       for n in ("anatomy", "order", "twofactor", "crossmonth", "slices")}



REL4 = ("same_rwy_same_op", "same_rwy_diff_op",
        "diff_rwy_same_op", "diff_rwy_diff_op")


def shared_partition(cross: dict) -> dict:
    """Split the pooled cross-product into four contributions that add to it.

    The pooled cross-product is the pair-weighted mean of the four relations.
    Writing each relation in the components (window, runway|window,
    operator|window, interaction) and collecting terms, every component enters
    weighted by the share of pairs that share the attribute.
    """
    p = {k: cross[k]["pairs"] for k in REL4}
    v = {k: cross[k]["value"] for k in REL4}
    w = sum(p.values())
    w_rwy = (p["same_rwy_same_op"] + p["same_rwy_diff_op"]) / w
    w_op = (p["same_rwy_same_op"] + p["diff_rwy_same_op"]) / w
    w_both = p["same_rwy_same_op"] / w
    out = {
        "window": v["diff_rwy_diff_op"],
        "runway": (v["same_rwy_diff_op"] - v["diff_rwy_diff_op"]) * w_rwy,
        "operator": (v["diff_rwy_same_op"] - v["diff_rwy_diff_op"]) * w_op,
        "additivity": (v["same_rwy_same_op"] - v["same_rwy_diff_op"]
                       - v["diff_rwy_same_op"] + v["diff_rwy_diff_op"]) * w_both,
    }
    out["total"] = sum(p[k] * v[k] for k in REL4) / w
    for k in ("window", "runway", "operator", "additivity"):
        out[f"{k}_pct"] = 100 * out[k] / out["total"]
    assert abs(sum(out[k] for k in ("window", "runway", "operator",
                                    "additivity")) - out["total"]) < 1e-6
    return out

def ci(v, nd=1):
    return f"[{v[0]:+.{nd}f}, {v[1]:+.{nd}f}]"


def tbl(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def val(d):
    return "n/a" if d.get("value") is None else f"{d['value']:+.1f} {ci(d['ci'])}"


def anatomy() -> list[str]:
    F = REC["anatomy"]["folds"]
    s = ["### T1 cohort and cells", ""]
    s.append(tbl([[f, f"{F[f]['n_rows']:,}", f"{F[f]['n_eligible']:,}",
                   f"{F[f]['n_cells']:,}", F[f]["median_cell_size"],
                   F[f]["cell_size_p90"],
                   f"{F[f]['runway_grain']['n_cells']:,}",
                   F[f]["runway_grain"]["median_cell_size"]] for f in FOLDS],
                 ["fold", "rows", "eligible", "cells (apt × bin)", "median size",
                  "90th pct", "cells (apt × bin × rwy)", "median size"]))
    s += ["", "### T2 the shared component, self-excluded cross-product (s²)", ""]
    keys = ["cell_ge_2", "cell_ge_4", "cell_ge_6", "cell_ge_10",
            "q04_support_dep6_arr6"]
    lab = {"cell_ge_2": "cells ≥ 2 (all eligible)", "cell_ge_4": "cells ≥ 4",
           "cell_ge_6": "cells ≥ 6", "cell_ge_10": "cells ≥ 10",
           "q04_support_dep6_arr6": "**exp-039 Q04's support (≥ 6 dep and ≥ 6 arr)**"}
    s.append(tbl([[lab[k]] + [val(F[f]["headline"][k]) for f in FOLDS]
                  + [f"{F['july_loo']['headline'][k]['pairs']:,}"] for k in keys],
                 ["support", "july_loo", "december", "january", "july pairs"]))
    s += ["", "### T2b the assignment's own statistic: the variance of the "
          "PLUG-IN cell mean (s²)", ""]
    rows = []
    for grain, get in (("airport × bin", lambda f: F[f]["plugin_cell_mean"]),
                       ("airport × bin × runway",
                        lambda f: F[f]["runway_grain"]["plugin_cell_mean"])):
        for f in FOLDS:
            d = get(f)["cell_ge_2"]
            rows.append([grain, f, f"{d['plugin_var_s2']:+.0f} {ci(d['ci'], 0)}",
                         f"{d['noise_term_s2']:+.0f}", f"{d['debiased_s2']:+.0f}",
                         f"{F[f]['headline']['cell_ge_2']['value']:+.0f}"
                         if grain.endswith("bin") else
                         f"{F[f]['runway_grain']['shared']['cell_ge_2']['value']:+.0f}"])
    s.append(tbl(rows, ["grain", "fold", "Var(plug-in cell mean)",
                        "− noise E[Var(ε)/n_c]", "= debiased",
                        "cross-product (T2)"]))
    s += ["", "The last two columns are two estimates of the same quantity by "
          "different routes, which is the check that the estimator swap changed "
          "nothing but the variance of the estimate.", ""]
    s += ["### T3 the runway × operator crossover (s²) and its components", ""]
    rel = ["same_rwy_same_op", "same_rwy_diff_op", "diff_rwy_same_op",
           "diff_rwy_diff_op"]
    cp = {f: F[f]["crossover_population"] for f in FOLDS}
    s += [f"On the MATCHED population: the "
          f"{cp['july_loo']['rows_with_all_four_relations_non_empty']:,} / "
          f"{cp['december']['rows_with_all_four_relations_non_empty']:,} / "
          f"{cp['january']['rows_with_all_four_relations_non_empty']:,} eligible "
          f"rows for which all four relations are non-empty "
          f"({cp['july_loo']['matched_pct_of_eligible']} / "
          f"{cp['december']['matched_pct_of_eligible']} / "
          f"{cp['january']['matched_pct_of_eligible']} % of eligible), so every "
          f"relation is estimated on the same rows and the same days.  The "
          f"unmatched reading over all eligible rows is beneath it.", ""]
    for tag, title in (("crossover_matched", "matched rows"),
                       ("crossover", "all eligible rows")):
        rows = []
        tot = sum(F["july_loo"][tag][x]["pairs"] for x in rel)
        for k in rel:
            rows.append([k.replace("_", " ")] + [val(F[f][tag][k]) for f in FOLDS]
                        + [f"{100 * F['july_loo'][tag][k]['pairs'] / tot:.1f} %"])
        s.append(f"**{title}**")
        s.append("")
        s.append(tbl(rows, ["relation", "july_loo", "december", "january",
                            "share of july pairs"]))
        s.append("")
    SHARE = {"airport_window_s2": "airport_shared_pct",
             "runway_given_window_s2": "runway_shared_pct",
             "operator_given_window_s2": "operator_shared_pct",
             "additivity_residual_s2": "additivity_pct"}
    rows = []
    for lab_, kk in (("matched rows", "variance_components_matched"),
                     ("all eligible", "variance_components")):
        for n, sh in SHARE.items():
            fs = F["july_loo"][kk].get("fractions_of_the_shared_component", {})
            rows.append([lab_, n.replace("_s2", "").replace("_", " ")]
                        + [f"{F[f][kk][n]:+.1f}" for f in FOLDS]
                        + [f"{F['july_loo'][kk]['fractions_of_the_sum'][sh]:+.1f} %",
                           f"{fs[sh]:+.1f} %" if sh in fs else "n/a"])
    s.append(tbl(rows, ["population", "variance component", "july_loo",
                        "december", "january", "share of the july sum",
                        "share of july's shared component"]))
    s += ["", "The last column is the assignment's own denominator (the shared "
          "component read on the same rows, "
          f"{F['july_loo']['variance_components_matched']['shared_component_on_these_rows_s2']:+.0f} s² "
          "on the matched rows). It does not add to 100 %: the components are "
          "conditional variances whose sum is the same-runway-same-operator "
          "relation, while the shared component is the pair-weighted average "
          "over all four relations.", ""]

    # T3c: the partition the assignment's sentence asks for.  The pooled
    # cross-product IS the pair-weighted mean of the four relations, so each
    # conditional component enters it weighted by the share of pairs that
    # share that attribute.  These four contributions add to the shared
    # component exactly, so they are fractions OF it.
    s += ["### T3c what fraction of the shared component is runway-, "
          "operator- and airport-shared (matched rows, adds to 100 %)", ""]
    part = {f: shared_partition(F[f]["crossover_matched"]) for f in FOLDS}
    #  A derived record, so the reverse scan can verify these numbers and a
    #  later reader does not have to redo the algebra.
    (A / "shared-partition.json").write_text(
        json.dumps({"note": "pair-weighted split of the matched-row cross-product; "
                            "the four contributions add to it exactly",
                    "folds": part}, indent=1) + "\n")
    rows = []
    for key, lab_ in (("window", "airport window (shared by every pair)"),
                      ("runway", "runway, beyond the window"),
                      ("operator", "operator, beyond the window"),
                      ("additivity", "runway × operator interaction")):
        rows.append([lab_] + [f"{part[f][key]:+.1f} ({100 * part[f][key] / part[f]['total']:.1f} %)"
                              for f in FOLDS])
    rows.append(["**shared component (T2 cross-product on these rows)**"]
                + [f"**{part[f]['total']:+.1f} (100.0 %)**" for f in FOLDS])
    s.append(tbl(rows, ["contribution to the shared component (s²)",
                        "july_loo", "december", "january"]))
    s += ["", "T3's last column divides a CONDITIONAL variance by the shared "
          "component, so it can exceed 100 %. This table weights each component "
          "by the share of pairs that share the attribute, which is how the "
          "component enters the pooled cross-product; the four contributions "
          "then add to the shared component exactly. The ordering is the same "
          "one T3 gives, and the runway share is the assignment's answer: "
          f"{100 * part['july_loo']['runway'] / part['july_loo']['total']:.1f} / "
          f"{100 * part['december']['runway'] / part['december']['total']:.1f} / "
          f"{100 * part['january']['runway'] / part['january']['total']:.1f} % "
          "runway against "
          f"{100 * part['july_loo']['operator'] / part['july_loo']['total']:.1f} / "
          f"{100 * part['december']['operator'] / part['december']['total']:.1f} / "
          f"{100 * part['january']['operator'] / part['january']['total']:.1f} % "
          "operator and "
          f"{100 * part['july_loo']['window'] / part['july_loo']['total']:.1f} / "
          f"{100 * part['december']['window'] / part['december']['total']:.1f} / "
          f"{100 * part['january']['window'] / part['january']['total']:.1f} % "
          "airport window.", ""]
    scan = F["july_loo"].get("q10_rule_grain_scan", {})
    if scan:
        s += ["### T3b the queue's PREDECLARED Q10 cell rule "
              "(two runways × two operators, ≥ 3 rows per intersection)", ""]
        rows = []
        for w, info in scan.items():
            for f in FOLDS:
                iw = REC["anatomy"]["folds"][f]["q10_rule_grain_scan"][w]
                rows.append([w, f, f"{iw['cells']:,}", f"{iw['cells_supported']:,}",
                             f"{iw['cells_supported_pct']:.2f} %",
                             f"{iw['rows_in_the_chosen_2x2']:,}",
                             f"{iw['rows_pct_of_eligible']:.2f} %"])
        s.append(tbl(rows, ["window", "fold", "cells", "supported",
                            "supported %", "rows in the 2×2", "rows %"]))
        comp_rows = []
        for w, info in scan.items():
            c = info.get("components")
            if not c:
                continue
            for n, sh in SHARE.items():
                cf = {ff: REC["anatomy"]["folds"][ff]["q10_rule_grain_scan"][w]
                      ["components"] for ff in FOLDS}
                comp_rows.append([w, n.replace("_s2", "").replace("_", " ")]
                                 + [f"{cf[ff][n]:+.1f}" for ff in FOLDS]
                                 + [f"{c['fractions_of_the_sum'][sh]:+.1f} %"])
        if comp_rows:
            s += ["", "and the components on the rows the rule supports:", ""]
            s.append(tbl(comp_rows, ["window", "variance component", "july_loo",
                                     "december", "january",
                                     "share of the july sum"]))
        sup = {w: scan[w].get("airports_supported") for w in scan
               if scan[w].get("airports_supported") is not None}
        if sup:
            s += ["", "Airports meeting the 200-pair/20-date rule on all four "
                  "relations, july_loo: "
                  + "; ".join(f"{w}: {', '.join(v) if v else 'none'}"
                              for w, v in sup.items()) + "."]
    s += ["", "### T4 the queue's coverage rule per airport, on the matched rows "
          "(200 pairs, 20 dates)", ""]
    apts = sorted(F["july_loo"]["crossover_coverage"])
    rows = []
    for a in apts:
        c = F["july_loo"]["crossover_coverage"][a]
        rows.append([a] + [f"{c[k]['pairs']:,} / {c[k]['dates']}"
                           + ("" if c[k]["met"] else " ✗") for k in rel])
    s.append(tbl(rows, ["airport (july_loo)"] + [k.replace("_", " ") for k in rel]))
    s += ["", f"Supported on all four relations, every fold: "
          f"{', '.join(F['july_loo']['crossover_airports_supported'])}."]
    s += ["", "### T5 the shared component by airport, hour, cell size and regime (s²)",
          "", "Each stratum twice: the self-excluded cross-product, and the "
          "assignment's own statistic — the variance of the plug-in cell mean "
          "with its noise term `E[Var(ε)/n_c]` removed. They estimate the same "
          "quantity by two routes.", ""]

    def _pair(cross, plug):
        if cross is None:
            return "n/a"
        if plug is None or plug.get("debiased_s2") is None:
            return f"{cross:+.0f} / n/a"
        return f"{cross:+.0f} / {plug['debiased_s2']:+.0f}"

    for lab_, kx, kp, head in (
            ("airport", "by_airport", "by_airport_plugin", "airport"),
            ("hour", "by_hour", "by_hour_plugin", "UTC hour band"),
            ("size", "by_cell_size", "by_cell_size_plugin", "cell size"),
            ("regime", "by_regime", "by_regime_plugin", "regime (own cells)")):
        keys = list(F["july_loo"][kx])
        rows = []
        for k in keys:
            rows.append([k] + [_pair(F[f][kx][k]["value"],
                                     F[f].get(kp, {}).get(k)) for f in FOLDS])
        s += ["", tbl(rows, [head + " (cross-product / debiased plug-in)",
                             "july_loo", "december", "january"])]
    s += ["", "### T6 persistence: the same statistic at lag L bins (s²)", ""]
    lags = list(F["july_loo"]["persistence"])
    rows = []
    for grain, key in (("airport × bin", "persistence"),
                       ("airport × bin × runway", None)):
        for f in FOLDS:
            src = (F[f]["persistence"] if key else F[f]["runway_grain"]["persistence"])
            rows.append([grain, f] + [f"{src[L]['value']:+.0f}" for L in lags])
    s.append(tbl(rows, ["grain", "fold"] + [L.replace("lag_", "L") for L in lags]))
    s += ["", "and the AUTOCORRELATION the assignment names, computed as a "
          "correlation of CELL MEANS on the cell pairs that exist at both lags "
          "(ρ on the plug-in mean / ρ with the within-cell noise removed from "
          "the denominator):", ""]
    lag_keys = [k for k in F["july_loo"]["autocorrelation"] if k.startswith("lag_")]
    rows = []
    for grain, key in (("airport × bin", "autocorrelation"),
                       ("airport × bin × runway", None)):
        for f in FOLDS:
            src = (F[f]["autocorrelation"] if key
                   else F[f]["runway_grain"]["autocorrelation"])
            rows.append([grain, f] + [
                f"{src[k]['rho_plugin']:.3f} / {src[k]['rho_debiased']:.3f}"
                for k in lag_keys])
    s.append(tbl(rows, ["grain", "fold"]
                 + [k.replace("lag_", "ρ") for k in lag_keys]))
    s += ["", "Cell pairs are weighted by `n_a · n_b`, the number of row pairs "
          "they carry — the same weight the cross-product table above uses. The "
          "record also carries the unweighted, one-vote-per-cell reading under "
          "`per_cell`: at lag 1 it is "
          + ", ".join(f"{f} {F[f]['autocorrelation']['lag_1']['per_cell']['rho_debiased']:.3f}"
                      for f in FOLDS) + ".", ""]
    s += ["", "### T7 what cell-level observables reach, cross-fitted inside the month", ""]
    rows = []
    for grain, get in (("airport × bin", lambda f: F[f]["observables"]),
                       ("airport × bin × runway",
                        lambda f: F[f]["runway_grain"]["observables"])):
        for f in FOLDS:
            o = get(f)
            rows.append([grain, f, o["n_features"],
                         f"{o['mse_gain_s2']:+.1f} {ci(o['ci'])}",
                         f"{o['shared_before']['value']:+.0f}",
                         f"{o['shared_after']['value']:+.0f}",
                         f"{100 * o['fraction_of_shared_explained']:.1f} %",
                         f"{o['cross_fitted_r2_cell_mean']:.4f}",
                         f"{o['cross_fitted_r2_shared_part']:.4f}"])
    s.append(tbl(rows, ["grain", "fold", "features", "2A − B (s²)",
                        "shared before", "shared after", "explained",
                        "R² cell mean", "R² shared part"]))
    return s


def order() -> list[str]:
    R = REC["order"]
    F, Bd = R["folds"], R["boards"]
    arms = ("h0", "ord", "unord", "unord2", "ord_c")
    s = ["### T8 step 2 — the five arms, both matched capacities (2A − B, s²)", ""]
    rows = []
    for cap in ("A", "B"):
        for a in arms:
            k = f"{a}@{cap}"
            head = cap == R["predeclared"]["headline_cap"]
            rows.append([cap, a, F["july_loo"]["arms"][k]["n_features"]]
                        + [f"{F[f]['arms'][k]['mse_gain_s2']:+.1f} "
                           f"{ci(F[f]['arms'][k]['ci'])}" for f in FOLDS]
                        + [f"{Bd[a]['board_all_row']['gain']:+.4f}" if head else "",
                           ci(Bd[a]["board_interval"]["ci_block_1d"], 4) if head else "",
                           ci(Bd[a]["board_interval"]["ci_block_3d"], 4) if head else ""])
    s.append(tbl(rows, ["cap", "arm", "features", "july_loo", "december", "january",
                        "all-row board gain", "board 1-day", "board 3-day"]))
    s += ["", "### T9 step 2 — the paired contrasts (s² per eligible row; * = interval clear of zero)", ""]
    pairs = [("ord", "unord"), ("ord", "unord2"), ("ord", "h0"), ("ord", "ord_c"),
             ("unord", "h0"), ("unord", "ord_c"), ("ord_c", "h0")]
    rows = []
    for cap in ("A", "B"):
        for a1, a2 in pairs:
            k = f"{a1}@{cap}"
            v = F["july_loo"]["arms"][k].get("vs", {})
            if a2 not in v:
                continue
            rows.append([cap, f"{a1} − {a2}"]
                        + [f"{F[f]['arms'][k]['vs'][a2]['delta_s2']:+.1f} "
                           f"{ci(F[f]['arms'][k]['vs'][a2]['ci'])}"
                           + ("*" if F[f]["arms"][k]["vs"][a2]["clear_of_zero"] else "")
                           for f in FOLDS]
                        + [ci(F["july_loo"]["arms"][k]["vs"][a2]["multiday"]
                              ["ci_block_3d"])])
    s.append(tbl(rows, ["cap", "contrast", "july_loo", "december", "january",
                        "july 3-day block"]))
    s += ["", "### T10 step 2 — the stream itself", ""]
    s.append(tbl([[f, F[f]["gap1_median_s"], F[f]["arrivals_between_median"],
                   F[f]["stream_coverage"]["gap_4_present_pct"],
                   F[f]["placebo_coverage_pct"], F[f]["order_differs_pct"]]
                  for f in FOLDS],
                 ["fold", "median gap 1 (s)", "median arrivals between",
                  "gap 4 present %", "placebo present %",
                  "order ≠ sorted multiset %"]))
    return s


def twofactor() -> list[str]:
    R = REC["twofactor"]
    F, Bd = R["folds"], R["boards"]
    arms = ("g0", "one", "two", "two_c", "q04h1", "one_q04", "two_q04")
    s = ["### T11 step 3 — the seven arms, both matched capacities (2A − B, s²)", ""]
    rows = []
    for cap in ("A", "B"):
        for a in arms:
            k = f"{a}@{cap}"
            head = cap == R["predeclared"]["headline_cap"]
            rows.append([cap, a, F["july_loo"]["arms"][k]["n_features"]]
                        + [f"{F[f]['arms'][k]['all_eligible']['mse_gain_s2']:+.1f} "
                           f"{ci(F[f]['arms'][k]['all_eligible']['ci'])}"
                           for f in FOLDS]
                        + [f"{Bd[a]['board_all_row']['gain']:+.4f}" if head else "",
                           ci(Bd[a]["board_interval"]["ci_block_3d"], 4) if head else ""])
    s.append(tbl(rows, ["cap", "arm", "features", "july_loo", "december", "january",
                        "all-row board gain", "board 3-day"]))
    s += ["", "### T12 step 3 — the paired contrasts (s²; * = interval clear of zero)", ""]
    rows = []
    for cap in ("A", "B"):
        for a1, a2 in (("two", "g0"), ("two", "one"), ("two", "two_c"),
                       ("one", "g0"), ("two_c", "g0"), ("q04h1", "g0"),
                       ("one_q04", "q04h1"), ("two_q04", "q04h1"),
                       ("two_q04", "one_q04")):
            k = f"{a1}@{cap}"
            if a2 not in F["july_loo"]["arms"][k].get("vs", {}):
                continue
            rows.append([cap, f"{a1} − {a2}"]
                        + [f"{F[f]['arms'][k]['vs'][a2]['delta_s2']:+.1f} "
                           f"{ci(F[f]['arms'][k]['vs'][a2]['ci'])}"
                           + ("*" if F[f]["arms"][k]["vs"][a2]["clear_of_zero"] else "")
                           for f in FOLDS]
                        + [ci(F["july_loo"]["arms"][k]["vs"][a2]["multiday"]
                              ["ci_block_3d"])])
    s.append(tbl(rows, ["cap", "contrast", "july_loo", "december", "january",
                        "july 3-day block"]))
    s += ["", "### T13 step 3 — the innovation, its coverage and the placebo", ""]
    s.append(tbl([[f, f"{F[f]['nuisance_train_departures']:,}",
                   F[f]["nuisance_months"], F[f]["u_present_pct"], F[f]["u_std_s"],
                   F[f]["feature_coverage_pct"]["f_same_rwy_diff_op"],
                   F[f]["feature_coverage_pct"]["f_diff_rwy_same_op"],
                   F[f]["placebo_group_differs_pct"],
                   ", ".join(F[f]["unsupported_airports"])] for f in FOLDS],
                 ["fold", "nuisance departures", "months", "u present %",
                  "u sd (s)", "same-rwy feature %", "same-op feature %",
                  "placebo group size differs %",
                  "airports failing the pair rule"]))
    return s


def crossmonth() -> list[str]:
    R = REC["crossmonth"]
    Ar, Bd, Dg = R["arms"], R["boards"], R["diagnostics"]
    s = ["### T14 the cross-month control (2A − B per eligible row, s²)", ""]
    rows = []
    for n in ("base_within", "base_cross", "base_cross_amp",
              "two_within", "two_cross", "two_cross_amp"):
        rows.append([n] + [f"{Ar[n][f]['mse_gain_s2']:+.1f} {ci(Ar[n][f]['ci'])}"
                           for f in FOLDS]
                    + [f"{Bd[n]['board_all_row']['gain']:+.4f}",
                       ci(Bd[n]["board_interval"]["ci_block_3d"], 4),
                       f"{Bd[n]['board_clean']['gain']:+.4f}"])
    s.append(tbl(rows, ["arm", "july_loo", "december", "january",
                        "all-row board", "board 3-day", "clean board"]))
    s += ["", "The cross-month amplitudes, each fitted on the other two folds' "
          "predictions from models that exclude BOTH the scored fold and the "
          "calibration fold: "
          + "; ".join(f"`{k}` {v}" for k, v in R["amplitudes"].items()) + ".", ""]
    cf = R.get("clean_family")
    if cf:
        s += ["### T14b the leak-free instrument: the same question with no v10 "
              "vector anywhere", "",
              "The base here is fitted inside this experiment on ONE other "
              "month, with `p10` dropped from the columns, and every base a "
              "scored fold uses excludes that fold — so no model behind this "
              "table has seen a row of the month it is scored on. `board` is the "
              "gain over that clean base, not over v10.", ""]
        rows = []
        for n, v in cf["arms"].items():
            rows.append([n] + [f"{v[f]['mse_gain_s2']:+.1f} {ci(v[f]['ci'])}"
                               for f in FOLDS]
                        + [f"{cf['boards'][n]['gain']:+.4f}",
                           ci(cf["boards"][n]["ci_block_3d"], 4)])
        s.append(tbl(rows, ["arm", "july_loo", "december", "january",
                            "board over the clean base", "board 3-day"]))
        s += ["", "The clean base's own all-row RMSE, july_loo / december / "
              "january: " + " / ".join(
                  f"{cf['arms']['clean_within'][f]['rmse_all_base']:.1f}"
                  for f in FOLDS) + " s.", ""]
    s += ["### T15 is the cross fit broken, or is the residual not portable?", ""]
    s.append(tbl([[k, f"{v['insample_2A_B_s2']:+.0f}", v["insample_scale"],
                   v["sd_h_within"], v["sd_h_cross"], v["corr_within_cross"],
                   v["corr_r_within"], v["corr_r_cross"]]
                  for k, v in Dg.items()],
                 ["family : fold", "in-sample 2A − B (s²)", "in-sample scale",
                  "sd h within", "sd h cross", "corr(h within, h cross)",
                  "corr(r, h within)", "corr(r, h cross)"]))
    return s


def contrasts() -> list[str]:
    p = A / "contrasts.json"
    if not p.exists():
        return []
    R = json.loads(p.read_text())
    s = ["### T17 the PAIRED contrast on the primary endpoint "
         "(gain over v10 of A minus gain of B, in s; * = 3-day interval clear "
         "of zero)", "",
         "The advance clauses are written on a pair of arms, so this is the "
         "interval that decides them: the ranking-weighted all-row january/july "
         "board, both arms rebuilt inside every draw, blocks inside each fold. "
         "Positive means the first arm is the better one.", ""]
    rows = []
    for k, v in R["pairs"].items():
        a, b = k.split(" - ")
        rows.append([a, b, f"{v['gain_a_minus_b']:+.4f}"
                     + ("*" if v["clear_of_zero_3d"] else ""),
                     ci(v["ci_block_1d"], 4), ci(v["ci_block_2d"], 4),
                     ci(v["ci_block_3d"], 4)])
    s.append(tbl(rows, ["arm A", "arm B", "gain(A) − gain(B)",
                        "1-day", "2-day", "3-day"]))
    return s


def climb() -> str:
    """`COMMON.md`'s climb log, generated so it cannot drift from the records.

    One line per measured attempt: the mechanism changed, the all-row july and
    december RMSE it produced, and the fraction of the 2.1154 s gap to the leader
    the fold board gain would close at face value (a face-value ratio, not a
    transfer estimate -- BOARD READING 10 puts transfer at about 0.16x).
    """
    GAP = 2.1154
    out = ["# exp-041-cell climb log -- generated by code/tables.py",
           "# iteration | mechanism | july all-row | december all-row | "
           "board gain | face-value fraction of the 2.1154 s gap"]
    rows = []
    O, T, C = REC["order"], REC["twofactor"], REC["crossmonth"]
    cap = O["predeclared"]["headline_cap"]
    for a in ("h0", "ord", "unord", "unord2", "ord_c"):
        rows.append((f"order:{a}", O["folds"]["july_loo"]["arms"][f"{a}@{cap}"],
                     O["folds"]["december"]["arms"][f"{a}@{cap}"],
                     O["boards"][a]["board_all_row"]["gain"]))
    for a in ("g0", "one", "two", "two_c", "q04h1", "one_q04", "two_q04"):
        rows.append((f"twofactor:{a}", T["folds"]["july_loo"]["arms"][f"{a}@{cap}"],
                     T["folds"]["december"]["arms"][f"{a}@{cap}"],
                     T["boards"][a]["board_all_row"]["gain"]))
    for a in C["arms"]:
        rows.append((f"crossmonth:{a}", C["arms"][a]["july_loo"],
                     C["arms"][a]["december"], C["boards"][a]["board_all_row"]["gain"]))
    for i, (name, j, d, g) in enumerate(rows, 1):
        out.append(f"{i:02d} | {name:26s} | {j['rmse_all_new']:.4f} | "
                   f"{d['rmse_all_new']:.4f} | {g:+.4f} s | {100 * g / GAP:+.2f} %")
    return "\n".join(out) + "\n"


def slices() -> list[str]:
    R = REC["slices"]["folds"]
    s = ["### T16 COMMON.md's four-way RMSE, on the saved out-of-fold vectors", ""]
    arms = ["v10"] + [k for k in R[FOLDS[0]] if k not in ("counts", "v10")]
    rows = []
    for f in FOLDS:
        r = R[f]
        for a in arms:
            rows.append([f, a, f"{r['counts']['all']:,}", f"{r[a]['all']:.4f}",
                         f"{r[a]['le3600']:.4f}", f"{r[a]['gt3600']:.4f}",
                         f"{r[a]['no_nm']:.4f}"])
    s.append(tbl(rows, ["fold", "vector", "rows", "RMSE all", "RMSE y ≤ 3600",
                        "RMSE y > 3600", "RMSE no-NM"]))
    s += ["", f"Row counts per fold: y > 3600 on "
          + " / ".join(f"{R[f]['counts']['gt3600']:,}" for f in FOLDS)
          + " rows, no-NM on "
          + " / ".join(f"{R[f]['counts']['no_nm']:,}" for f in FOLDS) + "."]
    return s


def main() -> None:
    (EXP / "climb.log").write_text(climb())
    print(f"  written: {EXP / 'climb.log'}")
    parts = (["<!-- generated by code/tables.py; do not edit by hand -->", ""]
             + anatomy() + [""] + order() + [""] + twofactor() + [""]
             + crossmonth() + [""] + contrasts() + [""] + slices() + [""])
    p = A / "tables.md"
    p.write_text("\n".join(parts))
    print(f"  written: {p}")
    rep = EXP / "RESULTS.md"
    if rep.exists():
        txt = rep.read_text()
        if BEGIN in txt and END in txt:
            head, rest = txt.split(BEGIN, 1)
            _, tail = rest.split(END, 1)
            rep.write_text(head + BEGIN + "\n" + p.read_text() + END + tail)
            print(f"  injected: {rep}")
        else:
            print("  no markers in RESULTS.md; skipped")


if __name__ == "__main__":
    main()
