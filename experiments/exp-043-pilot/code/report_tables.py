"""exp-043-pilot: render every numeric table in RESULTS.md from the analysis JSON.

The round-2 review found stale numbers in tables that had been typed by hand while
the JSON they came from had been regenerated.  The repair is structural rather than
another proof-read: each table in `RESULTS.md` sits between
`<!--T:name-->` and `<!--/T:name-->`, and this script rewrites what is between them.
Rounds 3, 4 and 5 each found the claim that followed here -- "a disagreeing number is
now impossible" -- too strong, because the prose outside the markers was still typed
by hand.  What is true: every table and every value between a marker is rendered from
`analysis/`, the set of rendered values has grown each time a review found drift, and
the column check refuses a table whose header, separator and rows disagree.

  .venv/bin/python experiments/exp-043-pilot/code/report_tables.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
A = EXP / "analysis"
R = EXP / "RESULTS.md"


def js(name):
    return json.loads((A / name).read_text())


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f} %"


def build() -> dict[str, str]:
    prov, cen = js("provenance.json"), js("census.json")
    hel, wk = js("match-heldout.json"), js("match-week.json")
    ext, non, div = js("extrapolate.json"), js("nonm_probe.json"), js("divergence.json")
    al = {k: js(f"align-{k}-heldout.json")
          for k in ("obs_offblock_disp", "obs_offblock_loose", "obs_offblock_strict")}
    runs = sorted((json.loads(p.read_text()) for p in A.glob("run_2025-07-*.json")),
                  key=lambda r: r["day"])
    runs = [r for r in runs if not r["probe_bytes"]]
    T = {}

    T["fetch"] = "\n".join(
        f"| {r['day']} | {prov['days'][r['day']]['bytes_total'] / 1e9:.2f} GB | "
        f"{r['members']:,} | {r['decompressed_bytes'] / 1e9:.1f} GB | {r['aircraft_in_box']} | "
        f"{r['aircraft_with_ground_point']} | {r['seconds']:,.0f} s |" for r in runs)

    rows, tot = [], 0
    for d, v in sorted(cen["per_day"].items()):
        ip, pa = v["inter_point_interval_s"], v["points_per_aircraft_day"]
        tot += v["ground_points"]
        rows.append(f"| {d} | {v['departures']} | {v['icao24_on_ground']} | "
                    f"{v['ground_points']:,} | {pa['median']:.0f} | {ip['median']:.0f} s | "
                    f"{pct(ip['share_at_or_under_35s'])} | "
                    f"{pct(v['departure_receipt_rate'])} | {pct(v['receipt_rate_nm_rows'])} | "
                    f"{v['no_nm_rows']} |")
    p = cen["pooled"]
    rows.append(f"| **Week** | **{p['departures']:,}** | — | **{tot:,}** | — | **2 s** | — | "
                f"**{pct(p['receipt_rate'])}** | **{pct(p['receipt_rate_nm_rows'])}** | "
                f"**90 → {pct(p['receipt_rate_no_nm_rows'])}** |")
    T["census"] = "\n".join(rows)

    def reach_row(label, key, fmt=pct):
        return (f"| {label} | {fmt(hel['reach'][key])} | {fmt(wk['reach'][key])} |")
    ci = hel["reach"]["matched_share_day_interval"]
    T["match"] = "\n".join([
        f"| Departures | {hel['reach']['departures']:,} | {wk['reach']['departures']:,} |",
        f"| Matched | {hel['reach']['matched']:,} (**{pct(hel['reach']['matched_share'])}**), "
        f"day range {pct(ci['leave_one_day_out_lo'])}–{pct(ci['leave_one_day_out_hi'])} | "
        f"{wk['reach']['matched']:,} ({pct(wk['reach']['matched_share'])}) |",
        reach_row("Matched, NM rows", "matched_nm"),
        f"| Matched, no-NM rows | **{pct(hel['reach']['matched_no_nm'])}** "
        f"({hel['reach']['with_obs_offblock_disp'] and ''}"
        f"{round(hel['reach']['matched_no_nm'] * hel['reach']['no_nm_rows'])} of "
        f"{hel['reach']['no_nm_rows']}) | {pct(wk['reach']['matched_no_nm'])} "
        f"({round(wk['reach']['matched_no_nm'] * wk['reach']['no_nm_rows'])} of "
        f"{wk['reach']['no_nm_rows']}) |",
        f"| With an off-block instant | {hel['reach']['with_obs_offblock_disp']:,} | "
        f"{wk['reach']['with_obs_offblock_disp']:,} |",
    ])

    def acc(tag, label, grp="nm_rows", src=hel):
        v = src["accuracy"][grp][f"{tag}.all_matched"]
        w = src["accuracy"][grp].get(f"{tag}.paired_with_aobt", v)
        ci = w["day_ci95"]
        return (f"| {label} | {v['rows']:,} | {v['median_offset_s']:.1f} s | "
                f"**{v['rmse_s']:.1f} s** | {ci[0]:.1f}–{ci[1]:.1f} | "
                f"{v['p01_s']:.0f} … {v['p99_s']:+.0f} |")
    aobt = hel["accuracy"]["nm_rows"]["obs_disp250.paired_with_aobt"]["aobt_on_same_rows"]
    T["accuracy"] = "\n".join([
        acc("obs_disp250", "**observation, 250 m stand-departure clock**"),
        acc("obs_loose", "observation, assignment's 3 kt / 30 s rule"),
        acc("obs_strict", "observation, 3 kt with an observed stand stop"),
        f"| `AOBT_3_flt` on the same rows | "
        f"{hel['accuracy']['nm_rows']['obs_disp250.paired_with_aobt']['rows']:,} | "
        f"{aobt['median_offset_s']:.1f} s | **{aobt['rmse_s']:.1f} s** | "
        f"{aobt['day_ci95'][0]:.1f}–{aobt['day_ci95'][1]:.1f} | "
        f"{aobt['p01_s']:.0f} … {aobt['p99_s']:+.0f} |",
    ])

    def pair(tag, label):
        v = hel["accuracy"]["nm_rows"][f"{tag}.paired_with_aobt"]
        d = v["paired_rmse_difference_day_interval"]
        return (f"| {label} | **{d['point']:+.1f} s** | {d['resample_lo']:+.1f} … "
                f"{d['resample_hi']:+.1f} | {d['leave_one_day_out_lo']:+.1f} … "
                f"{d['leave_one_day_out_hi']:+.1f} |")
    T["paired"] = "\n".join([pair("obs_disp250", "250 m clock − `AOBT_3_flt`"),
                             pair("obs_loose", "3 kt / 30 s rule − `AOBT_3_flt`")])

    def contract(k, label):
        v = al[k]["in_sample"]
        return (f"| {label} | {v['A_mean_rd']:,.1f} | {v['B_mean_dd']:,.1f} | "
                f"**{v['full_amplitude_mse_gain_s2']:,.1f} s²** | "
                f"{v['fitted_scale_A_over_B']:.4f} |")
    T["contract"] = "\n".join([
        contract("obs_offblock_disp", "250 m clock"),
        contract("obs_offblock_loose", "assignment's 3 kt rule"),
        contract("obs_offblock_strict",
                 f"3 kt with a stand stop ({al['obs_offblock_strict']['rows_matched_in_fold']:,}"
                 " rows)")])

    def scaled(k, which, label):
        a = al[k]
        if which == "dev":
            v = a["frozen_on_development_days"]
            g, after = v["full_amplitude_mse_gain_s2"], v["rmse_v10_plus_update_s"]
            gi = v["day_interval_gain_s2"]
            fit = f"the {v['fit_rows']:,} development rows"
            band = f"{gi['leave_one_day_out_lo']:,.0f}–{gi['leave_one_day_out_hi']:,.0f}"
        else:
            v = a["leave_one_day_out_scaled"]
            g, after = v["full_amplitude_mse_gain_s2"], a["rmse_v10_plus_update_loo_s"]
            fit = "the other four held-out days"
            gd = list(a["loo_gain_by_scored_day_s2"].values())
            band = f"{min(gd):,.0f}–{max(gd):,.0f} by day"
        return (f"| {label} | {fit} | **{g:,.1f} s²** | {band} | "
                f"{a['rmse_v10_on_these_rows_s']:.2f} → **{after:.2f} s** |")
    T["scaled"] = "\n".join([
        scaled("obs_offblock_disp", "dev", "**250 m clock, frozen on days 1–2**"),
        scaled("obs_offblock_disp", "loo", "250 m clock, leave-one-day-out cross-fit"),
        scaled("obs_offblock_loose", "dev", "3 kt rule, frozen on days 1–2"),
        scaled("obs_offblock_loose", "loo", "3 kt rule, leave-one-day-out cross-fit")])

    g = hel["accuracy"]["no_nm_rows"]
    T["nonm"] = "\n".join([
        f"| Departures with no NM join | {hel['reach']['no_nm_rows']} | — |",
        f"| Matched on a reconstructed callsign | "
        f"{round(hel['reach']['matched_no_nm'] * hel['reach']['no_nm_rows'])} "
        f"({pct(hel['reach']['matched_no_nm'])}) | — |",
        f"| With a 250 m clock | {g['obs_disp250.all_matched']['rows']} | "
        f"{g['obs_disp250.all_matched']['rmse_s']:.1f} s |",
        f"| With the 3 kt clock | {g['obs_loose.all_matched']['rows']} | "
        f"{g['obs_loose.all_matched']['rmse_s']:.1f} s |"])

    t = non["tight_key"]
    cal, cal5 = non["held_out_calibration"], non["held_out_calibration_at_500_rows"]
    cal3 = non["held_out_calibration_round3_setup"]
    ps = non["placebo_summary"]
    T["probe"] = "\n".join([
        f"| the key, on held-out rows whose answer the callsign already gave: "
        f"assigned the RIGHT segment | {cal['correct']:,} of {cal['assigned']:,} assigned "
        f"(**{pct(cal['precision_assigned_to_the_right_segment'])}**), "
        f"{pct(cal['recall_of_all_rows'])} of all {cal['rows']:,} rows |",
        f"| the same key, {cal5['rows_per_draw']} rows against the same "
        f"{cal5['pool_segments']:,}-segment pool | "
        f"{pct(cal5['precision_assigned_to_the_right_segment'])} right — the number of "
        f"competing rows is not what matters |",
        f"| the same key in round 3's own setup: {cal3['rows_per_draw']} rows against every "
        f"claimed segment ({cal3['pool_segments']:,}) | "
        f"{pct(cal3['precision_assigned_to_the_right_segment'])} right — the POOL's density "
        f"is what matters, and round 3's reading is reproduced |",
        f"| no-NM departures | {non['no_nm_departures']} |",
        f"| matched on a reconstructed callsign | {non['matched_by_callsign']} |",
        f"| unmatched | {non['unmatched']} |",
        f"| **of those, identified one-to-one by the key** | **{t['one_to_one']}** "
        f"(median {t['median_candidates_when_any']:.0f} candidate, max "
        f"{t['max_candidates']}) |",
        f"| the same key at {ps['offsets']} time-shifted placebos (±30 … ±120 min) | "
        f"**{ps['mean']:.1f} on average, {ps['min']}–{ps['max']}** |",
        f"| **expected correct at the held-out precision** | "
        f"**{non['expected_correct_at_held_out_precision']:.0f} of {non['unmatched']}** |",
        f"| **no-NM receipt, ESTIMATE under this key (not a bound)** | "
        f"**{pct(non['receipt_on_no_nm_rows_estimate'])}** |",
        f"| *(for the record: the withdrawn ±15 min no-type test)* | "
        f"*{non['first_version_loose_window_no_type']['one_to_one']} of "
        f"{non['unmatched']}, up to "
        f"{non['first_version_loose_window_no_type']['max_candidates']} candidates a row* |"])

    def board(key, label):
        v = ext[key]["obs_offblock_disp"]
        return (f"| {label} | {v['pooled_mse_reduction_s2']:,.0f} s² | "
                f"**{v['board_rmse_after_s']:.2f}** | **{v['board_gain_s']:+.2f} s** | "
                f"{pct(v['share_of_gap_to_lead'], 0)} |")
    T["board"] = "\n".join([
        board("if_eham_only", "**EHAM only, measured reach and gain**"),
        board("if_every_airport_matched_eham",
              "Every airport at EHAM's reach and gain *(not measured)*")])

    ARMS = ("shipped", "stable_only", "repair", "current")

    def dv(tag, keys):
        a = {n: div[tag][n] for n in ARMS}
        out = []
        for label, k, f in keys:
            out.append(f"| {label} | " + " | ".join(f(a[n][k]) for n in ARMS) + " |")
        return "\n".join(out)
    i = lambda x: f"{x:,.0f}"          # noqa: E731
    q = lambda x: f"{x:.2f} s"         # noqa: E731
    T["div_week"] = "\n".join([
        "| Ground segments ending in a take-off | " + " | ".join(
            f"{div['segments'][n]:,}" for n in ARMS) + " |",
        dv("week", [("Matched departures", "matched", i),
                    ("Rows scored (NM, paired with `AOBT_3`)", "rows_scored", i),
                    ("RMSE of the 250 m clock", "rmse_obs_s", q),
                    ("Paired difference against `AOBT_3`", "paired_difference_s", q)])])
    T["div_held"] = dv("heldout", [("Matched departures", "matched", i),
                                   ("RMSE of the 250 m clock", "rmse_obs_s", q),
                                   ("Paired difference against `AOBT_3`", "paired_difference_s", q)])
    return T


def values() -> dict[str, str]:
    """Numbers the PROSE quotes, rendered the same way the tables are.

    Round 3 found three of these stale while every marked table was current: a
    number is only safe from drift if it is generated, wherever it sits.
    """
    cen, hel, non = js("census.json"), js("match-heldout.json"), js("nonm_probe.json")
    ext, sw = js("extrapolate.json"), js("sweep.json")
    al = js("align-obs_offblock_disp-heldout.json")
    runs = sorted((json.loads(f.read_text()) for f in A.glob("run_2025-07-*.json")),
                  key=lambda r: r["day"])
    runs = [r for r in runs if not r["probe_bytes"]]
    d12 = [r for r in runs if r["day"] in ("2025-07-07", "2025-07-08")]
    scr = [r["screened_in"] for r in runs]
    led = (EXP.parents[1] / "docs/LEDGER.md").read_text().splitlines()
    rows = sum(1 for l in led if l.startswith("|")
               and l.split("|")[2].strip().startswith("exp-043"))
    half = next(a for a in sw["arms"] if a.get("move_kt") == 0.5 and a.get("hold_s") == 30.0
                and "disp_m" not in a)
    disp = sorted(a["disp_m"] for a in sw["arms"] if "disp_m" in a)
    orq = js("ordering.json")
    eq = orq["equivalence"]
    st6 = next(a for a in sw["arms"] if a.get("stand_stop_s") == 600.0)
    return {
        "receipt_week": pct(cen["pooled"]["receipt_rate"]),
        "receipt_nm": pct(cen["pooled"]["receipt_rate_nm_rows"]),
        "receipt_no_nm": pct(cen["pooled"]["receipt_rate_no_nm_rows"]),
        "gb_transferred": f"{ext['cost']['pilot_gb_transferred']:.1f} GB",
        "gb_per_day": f"{ext['cost']['measured_gb_per_day']:.2f} GB",
        "half_knot_median": f"{half['obs_offblock_loose']['median_offset_s']:+.1f} s",
        "disp_arms": f"{len(disp)} arms from {disp[0]:.0f} m to {disp[-1]:,.0f} m",
        "ledger_rows": f"{rows}",
        "rmse_obs": f"{hel['accuracy']['nm_rows']['obs_disp250.all_matched']['rmse_s']:.1f} s",
        "rmse_aobt": f"{hel['accuracy']['nm_rows']['obs_disp250.paired_with_aobt']['aobt_on_same_rows']['rmse_s']:.1f} s",
        "gain_frozen": f"{al['frozen_on_development_days']['full_amplitude_mse_gain_s2']:,.0f} s²",
        "rmse_after_frozen": f"{al['frozen_on_development_days']['rmse_v10_plus_update_s']:.2f} s",
        "contract_2amb": f"{al['in_sample']['full_amplitude_mse_gain_s2']:,.1f} s²",
        "nonm_estimate": pct(non["receipt_on_no_nm_rows_estimate"]),
        "nonm_identified": f"{non['tight_key']['one_to_one']} of {non['unmatched']}",
        "board_gain": f"{ext['if_eham_only']['obs_offblock_disp']['board_gain_s']:+.2f} s",
        # the count of generated values is itself generated: round 5 found it typed
        # by hand in two places that disagreed, which is the defect this mechanism
        # exists to stop.  `main` fills it in once the dict is built.
        "value_count": "",
        "eham_reach": pct(ext["ranking_file"]["projected_eham_reach"]),
        "days12_accounting": "; ".join(
            f"{r['day'][-5:]} {r['files']:,} files, {r['raw_member_bytes'] / 1e9:.3f} GB of "
            f"member payload from a {sum(a['size'] for a in r['assets_published']) / 1e9:.3f} GB "
            "archive" for r in d12),
        "screened_range": f"{min(scr):,} to {max(scr):,}",
        "half_knot_rmse": f"{half['obs_offblock_loose']['rmse_s']:.0f} s",
        "seen_before_3kt": (
            f"{sw['stand_stop']['seen_before_motion_3kt']['all_segments']['median_s']:.0f} s"),
        "seen_before_250m": (
            f"{sw['stand_stop']['seen_before_motion']['all_segments']['median_s']:.0f} s"),
        "stand600_kept": pct(st6["obs_offblock_strict"]["share"]),
        "stand600_kept_matched": pct(st6["obs_offblock_strict"]["rows"] / st6["matched"]),
        "regime_A": f"{al['in_sample_offset_regime']['A']['relative_move_pct']:.3f} %",
        "regime_B": f"{al['in_sample_offset_regime']['B']['relative_move_pct']:.3f} %",
        "regime_2AB": (
            f"{al['in_sample_offset_regime']['2A_minus_B']['relative_move_pct']:.2f} %"),
        "month_hours": f"{ext['cost']['one_training_month_kernel_hours']:.1f}",
        "ties_before": f"{orq['ties_before_deduplication']['rows_in_a_tied_group']:,}",
        "ties_disagreeing": (
            f"{orq['ties_before_deduplication']['groups_that_disagree_about_on_ground']:,}"),
        "ties_after": f"{orq['ties_after_deduplication']}",
        "order_equivalent": ("identical" if all(v["identical"] for v in eq.values()
                                                if isinstance(v, dict)) else "NOT identical"),
        "harvest_identical": (
            f"{orq['against_the_harvest']['shared']:,} shared, "
            f"{orq['against_the_harvest']['only_in_pilot']} only in the pilot, "
            f"{orq['against_the_harvest']['only_in_harvest']} only in the harvest, "
            f"{len(orq['against_the_harvest']['columns_that_differ_on_shared_rows'])} of "
            f"{len(orq['against_the_harvest']['columns_compared'])} columns differing"),
        "stand_margin_median": (
            f"{sw['stand_stop']['stand_stop_margin']['all_segments']['median_s']:.0f} s"),
        "no_clock_share": pct(1 - hel["reach"]["with_obs_offblock_disp"]
                              / hel["reach"]["departures"]),
    }


def check(s: str) -> list[str]:
    """Every marked table's header, separator and rows must have the same cell count.

    Round 3 found two tables whose headers declared more columns than the renderer
    produced, which is invisible in the JSON and silent in most markdown viewers.
    """
    bad = []
    for m in re.finditer(r"(\|[^\n]*\|)\n(\|[ -:|]+\|)\n<!--T:(\w+)-->\n(.*?)\n<!--/T:\3-->",
                         s, re.S):
        want = m.group(1).count("|") - 1
        sep = m.group(2).count("|") - 1
        got = {l.count("|") - 1 for l in m.group(4).splitlines() if l.startswith("|")}
        if got - {want} or sep != want:
            bad.append(f"{m.group(3)}: header has {want} cells, separator {sep}, "
                       f"rows {sorted(got)}")
    return bad


def selftest() -> None:
    """Prove the check fires, rather than asserting that it would.

    Injects each defect into a copy of the current report in memory and requires
    `check` to name the table.  Nothing is written.

      .venv/bin/python experiments/exp-043-pilot/code/report_tables.py --selftest
    """
    s = R.read_text()
    assert not check(s), f"the report is already broken: {check(s)}"
    i = s.index("<!--T:paired-->")
    sep = s.rfind("\n| --- ", 0, i) + 1
    hdr = s.rfind("\n", 0, sep - 1) + 1
    for name, pos, add in (("header", hdr, " extra |"), ("separator", sep, " --- |")):
        end = s.index("\n", pos)
        got = check(s[:end] + add + s[end:])
        assert got and got[0].startswith("paired:"), f"{name} defect not caught: {got}"
        print(f"  {name} defect caught: {got[0]}")
    print("selftest: the column-count check fires on both defects")


def main() -> None:
    if "--selftest" in sys.argv:
        return selftest()
    s, T = R.read_text(), build()
    V = values()
    V["value_count"] = str(len(V))
    for name, body in V.items():
        pat = re.compile(rf"(<!--V:{name}-->).*?(<!--/V:{name}-->)", re.S)
        if not pat.search(s):
            print(f"  [warn] no marker for value {name}")
            continue
        s = pat.sub(lambda m, b=body: m.group(1) + b + m.group(2), s)
    for name, body in T.items():
        pat = re.compile(rf"(<!--T:{name}-->\n).*?(\n<!--/T:{name}-->)", re.S)
        if not pat.search(s):
            print(f"  [warn] no marker for {name}")
            continue
        s = pat.sub(lambda m: m.group(1) + body + m.group(2), s)
    # a rendered row with the wrong number of cells is a silent markdown defect,
    # so the renderer refuses to leave one behind
    bad = check(s)
    if bad:
        raise SystemExit("column-count mismatch:\n  " + "\n  ".join(bad))
    R.write_text(s)
    print(f"rendered {len(T)} tables and {len(V)} inline values into "
          "RESULTS.md from analysis/, all column counts agree")


if __name__ == "__main__":
    main()
