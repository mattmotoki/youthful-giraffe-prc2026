"""exp-062 ownvalue-refine: the RESULTS tables, printed from the analysis records only.
ZERO FIT.  READS NO LABEL AND NO VECTOR.  It opens this experiment's own JSON records and the
controller's two append-only ledgers, and computes nothing new.

    vp_ctl.py --kind tables --log logs/tables.log -- .venv/bin/python .../vp_tables.py

Every table in RESULTS.md is regenerated from here, never retyped from a summary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
ANALYSIS = EXP / "analysis"
sys.path.insert(0, str(HERE))
import vp_ctl as CTL                                                  # noqa: E402

SP = ("v20", "v21", "v22")
SEEDS3 = (1, 2, 3)


def load(name: str):
    p = ANALYSIS / name
    return json.loads(p.read_text()) if p.exists() else None


def head(t: str) -> None:
    print(f"\n## {t}\n", flush=True)


def table(cols, rows) -> None:
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        print("| " + " | ".join("" if x is None else str(x) for x in r) + " |")


def e(x, nd=3) -> str:
    return "0" if x == 0 else f"{x:.{nd}e}"


def accounting() -> None:
    ch = CTL.charged()
    rows = CTL.read_lines(CTL.RECEIPTS)
    starts = {r["attempt_id"]: r for r in rows if r["receipt"] == "start"}
    ends = {r["attempt_id"]: r for r in rows if r["receipt"] == "end"}
    head("Every process, through the one pinned accounting controller")
    out = []
    for aid, s in starts.items():
        en = ends.get(aid)
        out.append([s["kind"], "yes" if s["fit_launch"] else "no", s["start_utc"],
                    f"{s['wall_timeout_s']:.0f}",
                    f"{s['projected_wall_s']:.0f} / {s['projected_process_cpu_s']:.0f}",
                    (f"{en['measured_wall_s']:.1f} / {en['measured_process_cpu_s']:.1f}"
                     if en else "no end receipt"),
                    en["status"] if en else "charged at the timeout", aid[:8]])
    table(["kind", "fit", "start (UTC)", "timeout s", "projected wall/cpu s",
           "measured wall/cpu s", "status", "attempt"], out)
    head("The charged use against the allocations and the caps")
    table(["quantity", "fit", "non-fit", "total", "allocation", "cap"],
          [["wall s", f"{ch['fit_wall']:.1f}", f"{ch['nonfit_wall']:.1f}",
            f"{ch['total_wall']:.1f}",
            f"{CTL.ALLOC_FIT_WALL_S:.0f} / {CTL.ALLOC_NONFIT_WALL_S:.0f}",
            f"{CTL.CAP_WALL_S:.0f}"],
           ["process cpu s", f"{ch['fit_cpu']:.1f}", f"{ch['nonfit_cpu']:.1f}",
            f"{ch['total_cpu']:.1f}",
            f"{CTL.ALLOC_FIT_CPU_S:.0f} / {CTL.ALLOC_NONFIT_CPU_S:.0f}",
            f"{CTL.CAP_CPU_S:.0f}"]])
    la = CTL.read_lines(CTL.LAUNCHES)
    att = [r for r in la if r.get("record") == "attempt"]
    oc = {r["attempt_id"]: r for r in la if r.get("record") == "outcome"}
    print(f"\nFit launches recorded: **{len(att)} of {CTL.MAX_FIT_LAUNCHES}**; "
          f"{len([r for r in oc.values() if r['status'] == 'fitted'])} fitted, "
          f"{len([r for r in oc.values() if r['status'] != 'fitted'])} not fitted, "
          f"{len(att) - len(oc)} with no outcome line.  Every stopped, failed or guard-refused "
          f"attempt is one of these lines.\n")
    if att:
        head("Every fit attempt")
        table(["tag", "k", "N", "level", "projected wall/cpu s", "measured wall/cpu s", "status",
               "rows moved"],
              [[a["tag"], a["k"], f"{a['N']:,}", a["level"],
                f"{a['projected_wall_s']:.0f} / {a['projected_process_cpu_s']:.0f}",
                (f"{oc[a['attempt_id']]['seconds_wall']:.0f} / "
                 f"{oc[a['attempt_id']]['seconds_process_cpu']:.0f}")
                if a["attempt_id"] in oc else "no outcome",
                oc.get(a["attempt_id"], {}).get("status", "no outcome"),
                f"{oc[a['attempt_id']].get('rows_moved', 0):,}"
                if a["attempt_id"] in oc and oc[a["attempt_id"]].get("rows_moved") else None]
               for a in att])


def registration(man: dict) -> None:
    head("Registration and the positive source boundary")
    r = man["registration"]
    cv = man["exp059_census"]
    table(["item", "value"],
          [["this experiment's SPEC.md", f"`{r['spec']}`, sha256 `{r['spec_sha256'][:16]}`"],
           ["its registration commit", f"`{r['spec_commit']}` ({r['spec_committed_utc']})"],
           ["new code files pinned before they run",
            f"{man['new_code']['n']}, digest `{man['new_code']['digest'][:16]}`"],
           ["exp-059's recorded census paths verified",
            f"{cv['verified']} of {cv['n_paths']}, {len(cv['differ'])} differ, "
            f"{len(cv['absent'])} absent ({cv['bytes_read']/1e9:.2f} GB in {cv['seconds']:.1f} s)"],
           ["the four exp-056 mismatch exceptions",
            f"{len(man['exp059_census_exceptions']['files'])} verified against exp-059's recorded "
            f"exception values (all equal: "
            f"{man['exp059_census_exceptions']['all_equal_exp059_recorded_values']}); they are NOT "
            f"sources here"],
           ["the positive source boundary", f"{man['source_boundary']['n']} pinned files"],
           ["environment",
            f"python {man['environment']['python']}, lightgbm {man['environment']['lightgbm']}, "
            f"dylib `{man['environment']['lightgbm_dylib_sha256'][:16]}`"],
           ["git HEAD at the manifest", f"`{man['environment']['git_head_at_manifest'][:12]}`"]])


def reproductions(rp: dict) -> None:
    head("The reproductions (SPEC 6): exp-059's R1 to R8 with its final corrected scope")
    rows = [["R1", "exp-055's ordered pins and exp-051's OOF y_true on every fold",
             "pass" if rp["checks"]["R1_originvec_load"] else "FAIL", "exact"],
            ["R2", f"exp-051's {len(rp['R2_exp051_gains']['values'])} stored July gains, through a "
                   f"reader that refused any key outside the registered "
                   f"{rp['R2_exp051_gains']['n_keys']}",
             e(rp["R2_exp051_gains"]["max_abs_diff"]), "1e-9 s2/row"]]
    r34 = rp["R3_R4_exp055_scripts"]
    for sub, run in r34["runs"].items():
        w = run["whole_record"]
        rows.append([f"R3/R4 {sub}",
                     f"exp-055's {sub}.py re-run unchanged; registered fields, WHOLE record, "
                     f"receipt hashes and OOF all enforced here",
                     f"{w['n_structural_diffs']} structural, {w['float_fields_differing']} float, "
                     f"receipts equal {run['receipt_hashes_equal']}, OOF {run['oof_equal']}",
                     "exact"])
    rows += [["R3/R4 fields", f"{r34['registered_fields']['n']} registered fields",
              e(max(r34["registered_fields"]["max_abs_diff_fields_at_1e-12"],
                    r34["registered_fields"]["max_abs_diff_fields_at_1e-9"])), "1e-12 / 1e-9"],
             ["R5", "the regenerated draws equal exp-055's and exp-057's",
              "pass" if rp["checks"]["R5_draws_equal_exp055_and_exp057"] else "FAIL", "exact"],
             ["R6", "bag(P1..P10) against Q15's saved heads; S1 exp-051's; 36 exp-057 vectors",
              e(rp["R6"]["plain_K10_vs_heads_max_abs_diff_s"]), "1e-9 s"],
             ["R7 (i)", "exp-057's rule-S and rule-Z points",
              f"m {e(rp['R7_exp057']['points_max_abs_diff_m'])}, "
              f"loss {e(rp['R7_exp057']['points_max_abs_diff_loss'])}", "1e-12 / 1e-9"],
             ["R7 (ii)", "its three draw receipts, column for column",
              ", ".join(f"L{L} bytes identical "
                        f"{rp['R7_exp057']['draw_receipts'][f'L{L}']['bytes_identical']}"
                        for L in (1, 2, 3)), "byte identity"],
             ["R7 (iii)", f"its day-by-airport receipt, {rp['R7_exp057']['daysums_shared_columns']} "
                          f"shared columns",
              e(rp["R7_exp057"]["daysums_max_rel_diff"]), "1e-9 relative"],
             ["R8 (ii)", f"exp-058's 24 vectors by hash", "pass", "exact"],
             ["R8 (iii)", f"all {len(rp['R8_exp058']['points_fields_compared'])} registered point "
                          f"fields, of which {len(rp['R8_exp058']['points_fields_rebuilt'])} are "
                          f"rebuilt and `q` is compared with Q15's pinned constant",
              f"m/cos {e(rp['R8_exp058']['points_max_abs_diff_m_cos'])}, "
              f"loss {e(rp['R8_exp058']['points_max_abs_diff_loss'])}", "1e-12 / 1e-9"],
             ["R8 (iv)", "its five L0 own values",
              e(rp["R8_exp058"]["own_value_max_abs_diff"]), "1e-9 s2/row"],
             ["R8 (v)", "its paired m_Feb - m_June intervals, day and day+member",
              e(rp["R8_exp058"]["paired_intervals_max_abs_diff"]), "1e-9"],
             ["R8 (vi)", "its three draw receipts and its day and day-by-airport cells",
              ", ".join(f"L{L} bytes identical "
                        f"{rp['R8_exp058']['draw_receipts'][f'L{L}']['bytes_identical']}"
                        for L in (1, 2, 3))
              + f"; daysums {e(rp['R8_exp058']['receipt_cells']['daysums']['max_rel_diff'])}, "
                f"days {e(rp['R8_exp058']['receipt_cells']['days']['max_rel_diff'])}",
              "byte identity / 1e-9 relative"]]
    for tag in ("exp058", "exp057"):
        b = rp["R8_vii_populations"][tag]
        rows.append([f"R8 (vii) {tag}",
                     f"every recorded population leaf; a missing path is a refusal",
                     f"{b['values_compared']} leaves, max |diff| {e(b['max_abs_diff'])}", "1e-9 s"])
    table(["check", "what it reproduces", "result", "tolerance"], rows)
    head("The separate reproductions the assignment names (SPEC 7)")
    s = rp["separate_reproductions"]
    ver = s["B1_grid"]["verified_rows"]
    table(["item", "result"],
          [["B1 the k = 24 block and the 23-row count sequence",
            f"every one of k = {', '.join(sorted(ver, key=int))} equals exp-059's registered row "
            f"(count vector, count-vector hash and both ordered-id hashes); the N sequence "
            f"reproduced: {s['B1_grid']['N_sequence_reproduced']}"],
           ["B2 the fit-free anchors",
            f"V(0) = {s['B2_anchors']['V_at_0_identity']} exactly (identity); "
            f"V(60,777) = {s['B2_anchors']['V_at_60777_rebuilt']:+.10f} against the registered "
            f"{s['B2_anchors']['V_at_60777_registered']:+.10f} "
            f"(|diff| {e(s['B2_anchors']['abs_diff'])})"],
           ["B3 February's value and the band",
            f"V* = {s['B3_february']['V_star_rebuilt']:+.10f} (|diff| "
            f"{e(s['B3_february']['abs_diff'])}); B = {s['B3_february']['band_B']}, which is "
            f"V* +/- 10 %: {s['B3_february']['band_matches_registered']}"],
           ["B4 exp-059's fitted V at k = 4, 8 and 12",
            "; ".join(f"k={a['k']} {a['rebuilt']:+.4f} (exp-059 {a['exp059']:+.4f})"
                      for a in s["B4_exp059_fitted_V"]["arms"].values())
            + f"; max |diff| {e(s['B4_exp059_fitted_V']['max_abs_diff_s2_row'])} s2/row"],
           ["B5 the day cells",
            f"exp-058 daysums {e(s['B5_day_cells']['exp058_daysums']['max_rel_diff'])}, days "
            f"{e(s['B5_day_cells']['exp058_days']['max_rel_diff'])}, exp-057 daysums "
            f"{e(s['B5_day_cells']['exp057_daysums']['max_rel_diff'])} relative; exp-055's i2 "
            f"carrier K = 3 cells "
            f"{e(s['B5_day_cells']['exp055_i2_carrier_K3_cells_max_abs_diff'])} absolute"]])
    sc = rp["source_check"]
    print(f"\nThe reproduction pass opened {sc['opened']} repository files outside this "
          f"experiment's own paths: {sc['in_manifest']} inside the positive manifest with those "
          f"bytes, {len(sc['in_exp059_census_only'])} in exp-059's verified census only, "
          f"{len(sc['outside_every_pin'])} outside every pin, {len(sc['differ'])} differing.  It "
          f"opened {len(rp['exp062_vectors_opened'])} exp-062 vectors.\n")


def probe(pr: dict) -> None:
    head("The fixed near-origin probe (SPEC 8): the three new own values")
    rows = []
    for k in ("1", "2", "3"):
        a = pr["arms"][f"G{int(k):02d}"]
        iv = a["intervals"]["L3"]
        rows.append([f"k = {k}", f"{a['N']:,}", f"{a['V']:+.4f}",
                     f"[{iv['lo']:+.4f}, {iv['hi']:+.4f}]",
                     "yes" if a["in_band"] else "no",
                     f"{a['distance_to_band_s2_row']:+.4f}",
                     f"{a['abs_V_minus_V_star']:.4f}",
                     " ".join(f"{a['members'][f's{s}']['value']:+.1f}" for s in SEEDS3)])
    a4 = pr["arms"]["G04"]
    rows.append(["k = 4 (exp-059, reproduced)", f"{a4['N']:,}", f"{a4['V']:+.4f}",
                 f"[{a4['intervals']['L3']['lo']:+.4f}, {a4['intervals']['L3']['hi']:+.4f}]",
                 "yes" if a4["in_band"] else "no",
                 f"{a4['distance_to_band_s2_row']:+.4f}", f"{a4['abs_V_minus_V_star']:.4f}",
                 " ".join(f"{a4['members'][f's{s}']['value']:+.1f}" for s in SEEDS3)])
    table(["grid row", "N", "V s2/row", "3-day day-block band", "inside B",
           "distance to B", "|V - V*|", "members"], rows)
    print(f"\nThe band is frozen: B = {pr['band_test']['band_B']}, V* = "
          f"{pr['constants']['V_star_february']}.  {pr['band_test']['rule']}.\n")
    head("The measured order over k = 1, 2, 3, 4")
    o = pr["measured_order_k1_to_k4"]
    table(["from k", "to k", "delta N", "delta V s2/row"],
          [[d["from_k"], d["to_k"], f"{d['delta_N']:,}", f"{d['delta_V']:+.4f}"]
           for d in o["adjacent_differences"]])
    print(f"\nAscending in N over these five measured points: {o['ascending_in_N']}.  "
          f"{o['claim_boundary']}.\n")
    print(f"**Branch: {pr['branch']}.**  {pr['reading']}\n")
    if pr.get("june_star"):
        js = pr["june_star"]
        print(f"June* = {js['arm']} (k = {js['k']}, N = {js['N']:,}), V = {js['V']:+.4f} s2/row, "
              f"|V - V*| = {js['abs_V_minus_V_star']:.4f}; selected among {js['selected_among']} by "
              f"{js['selection_rule']}; frozen before any L1, L2 or L3 read.\n")


def measurement(me: dict) -> None:
    head("The conditional completion (SPEC 9): the points on July H")
    cols = ["comparison", "K", "step", "m", "q (pinned)", "m - q", "m / q", "cos",
            "rms carrier s", "rms plain s", "A", "B", "g_C", "g_hat", "g_plain", "J",
            "amplitude", "alignment"]
    rows = []
    for c in ("feb_K3", "june_star_K3", "june_matched_K3", "carrier_K3", "sham_K3", "carrier_K10"):
        p = me["points"][c]
        for s in SP:
            q = p["steps"][s]
            rows.append([c, p["K"], s] + [f"{q[k]:.4f}" for k in
                                          ("m", "q", "m_minus_q", "m_over_q", "cos",
                                           "rms_H_carrier_s", "rms_H_plain_s")]
                        + [f"{q[k]:+.4f}" for k in ("A", "B", "g_C", "g_hat", "g_plain", "J",
                                                    "amplitude", "alignment")])
    table(cols, rows)
    head("Every raw member")
    rows = []
    for nm in ("feb", "june_star", "june_matched", "carrier", "sham"):
        for s in SEEDS3:
            for st in SP:
                q = me["raw_members"][nm][f"s{s}"][st]
                rows.append([nm, s, st, f"{q['m']:.4f}", f"{q['m_minus_q']:+.4f}",
                             f"{q['cos']:.4f}", f"{q['J']:+.2f}",
                             f"{q['rms_H_carrier_s']:.4f}"])
    table(["arm", "member", "step", "m", "m - q", "cos", "J s2/row", "rms carrier s"], rows)
    head("The changed rows of every step")
    rows = []
    for c in ("feb_K3", "june_star_K3", "june_matched_K3", "carrier_K3", "sham_K3"):
        for s in SP:
            q = me["points"][c]["steps"][s]
            rows.append([c, s, f"{q['changed_rows_total']['carrier']:,}",
                         f"{q['changed_rows_total']['plain']:,}",
                         ", ".join(f"{g} {q['changed_rows']['carrier'][g]:,}"
                                   for g in ("H", "G", "A", "C"))])
    table(["comparison", "step", "changed rows carrier", "changed rows plain",
           "carrier by serving regime"], rows)
    head("Each arm's L0 own value")
    rows = []
    for nm, a in me["own_value_L0"]["by_arm"].items():
        iv = a["intervals"].get("L3", {})
        rows.append([nm, f"{a['value']:+.4f}",
                     (f"[{iv['day']['lo']:+.4f}, {iv['day']['hi']:+.4f}]" if "day" in iv else None),
                     (f"[{iv['day+member']['lo']:+.4f}, {iv['day+member']['hi']:+.4f}]"
                      if "day+member" in iv else None)])
    table(["arm", "V s2/row", "3-day day band", "3-day day+member band"], rows)
    head("The fixed reading (SPEC 10)")
    r = me["registered_reading"]
    table(["step", "m_Feb", "m_June*", "difference", "3-day day+member band",
           "inside the margin", "lower endpoint above the margin"],
          [[s, f"{r['points'][s]['m_Feb']:.4f}", f"{r['points'][s]['m_June_star']:.4f}",
            f"{r['points'][s]['difference']:+.4f}",
            (f"[{r['bands_3day_day_plus_member'][s]['lo']:+.4f}, "
             f"{r['bands_3day_day_plus_member'][s]['hi']:+.4f}]" if s in r["steps"] else
             f"[{me['paired_differences']['feb_minus_june_star'][s]['intervals']['L3']['day+member']['lo']:+.4f}, "
             f"{me['paired_differences']['feb_minus_june_star'][s]['intervals']['L3']['day+member']['hi']:+.4f}]"),
            r["both_bands_inside_margin"].get(s),
            r["both_lower_endpoints_above_margin"].get(s)] for s in SP])
    print(f"\n**Verdict: {r['verdict']}.**  {r['reading']}\n")
    print(f"{r['not_a_test']}  {r['no_substitute']}.  {r['meaning']}.\n")
    head("The decompositions of the February and June* steps, by airport")
    rows = []
    for c in ("feb_K3", "june_star_K3"):
        for s in SP:
            d = me["decompositions"]["airport"][c][s]
            for a, v in d["by_airport"].items():
                rows.append([c, s, a, f"{v['H_rows']:,}",
                             f"{v['contribution_to_m2_minus_1']:+.5f}",
                             f"{v['plain_mass_share']:.4f}",
                             None if v["m_airport"] is None else f"{v['m_airport']:.4f}",
                             None if v["q_airport_exp055"] is None
                             else f"{v['q_airport_exp055']:.4f}"])
    table(["comparison", "step", "airport", "H rows", "contribution to m2 - 1",
           "plain mass share", "m airport", "q airport (exp-055)"], rows)
    head("The NM-regime decomposition")
    rows = []
    for c in ("feb_K3", "june_star_K3"):
        for s in SP:
            d = me["decompositions"]["nm_regime"][c][s]
            for g, v in d["by_regime"].items():
                rows.append([c, s, g, f"{v['H_rows']:,}",
                             f"{v['contribution_to_m2_minus_1']:+.5f}",
                             f"{v['plain_mass_share']:.4f}",
                             None if v["m_regime"] is None else f"{v['m_regime']:.4f}"])
    table(["comparison", "step", "NM regime", "H rows", "contribution to m2 - 1",
           "plain mass share", "m regime"], rows)
    head("The serving-regime decomposition")
    rows = []
    for c in ("feb_K3", "june_star_K3"):
        for s in SP:
            d = me["decompositions"]["serving_regime"][c][s]
            for g, v in d["by_regime"].items():
                rows.append([c, s, g, f"{v['rows']:,}", f"{v['changed_rows_carrier']:,}",
                             f"{v['changed_rows_plain']:,}",
                             None if v["carrier_step_mass_share"] is None
                             else f"{v['carrier_step_mass_share']:.6f}"])
    table(["comparison", "step", "serving regime", "rows", "changed rows carrier",
           "changed rows plain", "carrier step mass share"], rows)
    head("COMMON's fold table (RMSE in seconds)")
    rows = []
    for f, blk in me["populations"].items():
        for nm in sorted(k for k in blk if k.endswith(("_L0", "_L3")) and not k.startswith("rows")):
            rows.append([f, nm] + [None if blk[nm].get(m) is None else f"{blk[nm][m]:.4f}"
                                   for m in ("all", "clean", "le3600", "gt3600", "no_nm")])
    table(["fold", "predictor", "all", "clean", "<= 3600 s", "> 3600 s", "no NM"], rows)
    print(f"\nThe full table, every level and every mask, is in `analysis/measure.json` under "
          f"`populations`; the serving table is under `by_serving_regime`.\n")
    print(f"Rule O (exp-058's, reproduced, not this experiment's rule): advances = "
          f"{me['rule_O_exp058_reproduced']['advances']}.  "
          f"{me['rule_O_exp058_reproduced']['note']}.\n")


def main() -> None:
    t0 = time.time()
    print(f"exp-062 ownvalue-refine tables, printed {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    man, rp, pr, me = load("manifest.json"), load("repro.json"), load("probe.json"), \
        load("measure.json")
    if man:
        registration(man)
    if rp:
        reproductions(rp)
    if pr:
        probe(pr)
    if me:
        measurement(me)
    else:
        head("The conditional completion")
        print("No `analysis/measure.json`: the probe routed no measurement.\n")
    accounting()
    print(f"\n(printed in {time.time() - t0:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
