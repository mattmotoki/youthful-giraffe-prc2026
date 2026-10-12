"""exp-059 ownvalue: print the RESULTS tables from the analysis records.  STDOUT ONLY.

Writes nothing.  Every table in RESULTS.md comes from this printer, so the prose cannot drift from
the record.  The ladder tables print from analysis/ladder.json and, when it exists,
analysis/ladder2.json.  The measurement tables print only when analysis/measure.json exists; a
ladder that stops without `June*` (SPEC 5.5) produces no measurement, and then this printer stops
after the ladder tables and says so.

    .venv/bin/python experiments/exp-059-ownvalue/code/ov_tables.py [--record PATH] [--ladder-only]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

ap = argparse.ArgumentParser()
ap.add_argument("--record", default=str(EXP / "analysis/measure.json"))
ap.add_argument("--ladder-only", action="store_true")
ARGS = ap.parse_args()

L1 = json.loads((EXP / "analysis/ladder.json").read_text())
L2P = EXP / "analysis/ladder2.json"
L2 = json.loads(L2P.read_text()) if L2P.exists() else None
LAD = L2 or L1
SP = ["v20", "v21", "v22"]


def head(t):
    print(f"\n### {t}\n")


def iv(d, key="m", L=3, mode="day+member"):
    o = d["intervals"][f"L{L}"][mode]
    o = o[key] if key in o else o
    return f"[{o['lo']:+.4f}, {o['hi']:+.4f}]"


def ivd(d, L=3, mode="all"):
    o = d["intervals"][f"L{L}"]
    o = o[mode] if mode in o else o
    return f"[{o['lo']:+.1f}, {o['hi']:+.1f}]"


# ============================================================================ the ladder ==========

print(f"ladder record: analysis/{'ladder2' if L2 else 'ladder'}.json "
      f"(stage {LAD['stage']}, spec commit {LAD['spec_commit'][:7]}, "
      f"{LAD['seconds']} s wall / {LAD['seconds_process_cpu']} s process cpu)")

head("R8 (iv): exp-058's five L0 own values, reproduced before any new vector was opened")
print("| arm | X | Y | this experiment (s²/row) | exp-058's record | |difference| |")
print("|---|---|---|---|---|---|")
for nm, e in sorted(LAD["R8_exp058_own_values"]["by_arm"].items()):
    print(f"| {nm} | {e['X']} | {e['Y']} | {e['value']!r} | {e['exp058']!r} | {e['abs_diff']:.2e} |")
print(f"\nlargest |difference| over the five: {LAD['R8_exp058_own_values']['max_abs_diff_s2_row']:.2e} s²/row "
      f"(the registered tolerance is 1e-9)")

head("The June own-value ladder V(N) (K = 3 bag, L0, every July fold row, n = 190,713)")
print("| N | k | f | source | V (s²/row) | in B = [151.73, 185.45] | 3-day day interval |")
print("|---|---|---|---|---|---|---|")
for p in LAD["points"]:
    k = p.get("k")
    arm = LAD["arms"].get(f"G{k:02d}") if k else None
    print(f"| {p['N']:,} | {k if k else '—'} | {f'{k}/24' if k else '—'} | {p['source']} | "
          f"**{p['V']:+.4f}** | {p['in_band']} | {ivd(arm) if arm else '—'} |")
print(f"\nadjacent differences (s²/row): "
      + ", ".join(f"{d:+.4f}" for d in LAD["adjacent_differences"])
      + f"; monotone at 1e-9: {LAD['monotone']}")

head("Every ladder member beside its bag (own value at L0, s²/row)")
print("| arm | k | N | member 1 | member 2 | member 3 | K = 3 bag |")
print("|---|---|---|---|---|---|---|")
for a in sorted(LAD["arms"]):
    e = LAD["arms"][a]
    print(f"| {a} | {e['k']} | {e['N']:,} | "
          + " | ".join(f"{e['members'][f's{s}']['value']:+.1f}" for s in (1, 2, 3))
          + f" | **{e['V']:+.4f}** |")

head("The stopping rule (SPEC §5.2)")
print(f"branch: **{LAD['branch']}**")
print(f"reading: {LAD['reading']}")
if L1.get("third_point"):
    t = L1["third_point"]
    print(f"\nstep 5: the chord between N = {t['bracketing_pair'][0]['N']:,} and "
          f"N = {t['bracketing_pair'][1]['N']:,} reaches V* at N̂ = {t['N_hat']:,.2f}; the nearest "
          f"unfitted grid row is k = {t['k']} (N = {t['N']:,}), {t['distance_to_N_hat']:,.2f} rows away")
if L2 and L2.get("step6"):
    s6 = L2["step6"]
    print(f"\nstep 6: the third point is inserted at position {s6['inserted_at']} between measured "
          f"neighbours {s6['neighbours'][0]:+.4f} and {s6['neighbours'][1]:+.4f} s²/row; outside its "
          f"neighbours: {s6['outside_neighbours']}; inside B: {s6['in_band']}; residual "
          f"|V − V*| = {s6['residual_abs_V_minus_V_star']:.4f} s²/row")
js = LAD.get("june_star")
PENDING = ("**not yet fixed** -- step 5 selected a third arm and stage 2 has not run; "
           "SPEC 5.4 holds every new m until it does")
STOPPED = ("**none** -- the ladder stopped without a match, so SPEC 5.4 holds every new m and "
           "no curve reading is applied (SPEC 5.5)")
print(f"\nJune*: {json.dumps(js) if js else (PENDING if LAD['branch'] == 'step5_third_point' else STOPPED)}")
print(f"\nanchors: V(0) = {LAD['checks']['anchor_V_at_0_identity']['value']!r} by construction; "
      f"V(60,777) = {LAD['constants']['anchor_V_at_60777']!r} from exp-058, reproduced exactly above")

if ARGS.ladder_only or not Path(ARGS.record).exists():
    print("\n(no analysis/measure.json: the tables of the measurement are not printed)")
    raise SystemExit(0)

# ============================================================================ the measurement =====

R = json.loads(Path(ARGS.record).read_text())
JS = R["rules"]["registered_reading"]["june_star"]["arm"]
JSC = f"ladder_{JS}_K3"
LARMS = sorted(a for a in R["arms"]) if isinstance(R.get("arms"), dict) else []
ARMS = [("feb_K3", "February K = 3 (exp-058)"), (JSC, f"June* {JS} K = 3")]
LADDER_OTHER = [(f"ladder_{a}_K3", f"June ladder {a} K = 3") for a in LARMS if f"ladder_{a}_K3" != JSC]
COMPARATORS = [("june_matched_K3", "matched June K = 3, N = 60,777 (exp-058)"),
               ("carrier_K3", "full June K = 3 (exp-051 / exp-055)"),
               ("sham_K3", "K = 3 sham (exp-057)"), ("carrier_K10", "full June K = 10 (exp-057)")]
ALL = ARMS + LADDER_OTHER + COMPARATORS

print(f"\n\nmeasurement record: {ARGS.record}")
print(f"spec commit {R['spec_commit'][:7]}, standin {R['standin']}, {R['seconds']} s wall / "
      f"{R.get('seconds_process_cpu')} s process cpu")

head("The registered reading (SPEC §8)")
rr = R["rules"]["registered_reading"]
print(f"**verdict: {rr['verdict']}**\n")
print(f"{rr['reading']}\n")
print(f"statement: {rr['statement']}")
print(f"\n| step | m_Feb | m_June* | difference | 3-day day+member band | inside ±0.03 | "
      "lower endpoint above +0.03 |")
print("|---|---|---|---|---|---|---|")
for s in rr["steps"]:
    p = rr["points"][s]
    b = rr["bands_3day_day_plus_member"][s]
    print(f"| {s} | {p['m_Feb']:.4f} | {p['m_June_star']:.4f} | **{p['m_Feb_minus_m_June_star']:+.4f}** | "
          f"[{b['lo']:+.4f}, {b['hi']:+.4f}] | {rr['both_bands_inside_margin'][s]} | "
          f"{rr['both_lower_endpoints_above_margin'][s]} |")
print("\nthe same difference at every block length (day+member):")
print("| step | 1-day | 2-day | 3-day |")
print("|---|---|---|---|")
for s in SP:
    o = rr["bands_all_block_lengths"][s]
    print(f"| {s} | " + " | ".join(f"[{o[f'L{L}']['lo']:+.4f}, {o[f'L{L}']['hi']:+.4f}]"
                                   for L in (1, 2, 3)) + " |")

head("The arms on July H (points, full member set)")
print("| arm | step | m | m − q | m / q | cos | rms_H carrier (s) | rms_H plain (s) | "
      "m 3-day day+member |")
print("|---|---|---|---|---|---|---|---|---|")
for c, nm in ALL:
    for s in SP:
        q = R["points"][c]["steps"][s]
        print(f"| {nm} | {s} | **{q['m']:.4f}** | {q['m_minus_q']:+.4f} | {q['m_over_q']:.3f} | "
              f"{q['cos']:.3f} | {q['rms_H_carrier_s']:.3f} | {q['rms_H_plain_s']:.3f} | {iv(q)} |")

head("Paired differences (the same member triple and the same days in every draw)")
for nm, ttl in (("feb_minus_june_star", "m_Feb − m_June*"),
                ("feb_minus_june_matched", "m_Feb − m_June (matched, N = 60,777; exp-058's pair)")):
    if nm not in R["paired_differences"]:
        continue
    print(f"\n**{ttl}**\n")
    print("| step | m_X | m_Y | difference | 1-day | 2-day | 3-day |")
    print("|---|---|---|---|---|---|---|")
    for s in SP:
        d = R["paired_differences"][nm][s]
        print(f"| {s} | {d['m_X']:.4f} | {d['m_Y']:.4f} | **{d['difference']:+.4f}** | "
              + " | ".join(iv(d, key="all", L=L) for L in (1, 2, 3)) + " |")

head("Every raw member beside its bag (m on July H)")
print("| arm | v20 | v21 | v22 |")
print("|---|---|---|---|")
for pre, nm in ([("feb", "February"), (f"ladder_{JS}", f"June* {JS}")]
                + [(f"ladder_{a}", f"ladder {a}") for a in LARMS if a != JS]
                + [("june_matched", "matched June"), ("carrier", "full June (exp-051)"),
                   ("sham", "sham (exp-057)")]):
    for i in (1, 2, 3):
        key = f"{pre}_s{i}"
        if key not in R["points"]:
            continue
        q = R["points"][key]["steps"]
        print(f"| {nm} member {i} (K = 1) | " + " | ".join(f"{q[s]['m']:.4f}" for s in SP) + " |")
    kb = f"{pre}_K3"
    if kb in R["points"]:
        q = R["points"][kb]["steps"]
        print(f"| **{nm} bag (K = 3)** | " + " | ".join(f"**{q[s]['m']:.4f}**" for s in SP) + " |")
for nm in sorted(R.get("difference_by_member", {})):
    d = R["difference_by_member"][nm]
    print(f"| {nm} by member seed | " + " | ".join(
        "/".join(f"{d[f's{i}'][s]:+.3f}" for i in (1, 2, 3)) for s in SP) + " |")

head("Loss equations (s² per July fold row, n = 190,713, sums over H) — diagnostics only")
print("| arm | step | A | B | m | g_hat | g_plain | g_C | J | amplitude | alignment |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for c, nm in ALL:
    for s in SP:
        q = R["points"][c]["steps"][s]
        print(f"| {nm} | {s} | {q['A']:.2f} | {q['B']:.2f} | {q['m']:.4f} | {q['g_hat']:+.2f} | "
              f"{q['g_plain']:+.2f} | {q['g_C']:+.2f} | {q['J']:+.2f} | {q['amplitude']:+.2f} | "
              f"{q['alignment']:+.2f} |")
for c, nm in ALL:
    print(f"mean J {nm}: {R['points'][c]['mean_J']['point']:+.2f} "
          f"(3-day day+member {iv(R['points'][c]['mean_J'], key='all')})")

head("Own value at L0 ([SSE(plain_L0) − SSE(arm_L0)] / n, s²/row)")
print("| arm | value | 3-day day interval |")
print("|---|---|---|")
for nm, e in R["own_value_L0"]["by_arm"].items():
    print(f"| {nm} ({e['X']} vs {e['Y']}) | {e['value']:+.1f} | {ivd(e)} |")
print(f"exp-051's recorded full June K = 3 value: {R['own_value_L0']['exp051_june_K3_recorded']:+.2f}")
print(f"the ladder arms' own values against analysis/ladder*.json: "
      f"{R['checks']['ladder_own_values_max_abs_diff_s2_row']:.2e} s²/row")

head("Per-airport m_a (July H)")
ap_names = sorted(R["decompositions"]["airport"]["feb_K3"]["v20"]["by_airport"])
print("| arm | step | " + " | ".join(ap_names) + " |")
print("|---|---|" + "---|" * len(ap_names))
for c, nm in ALL:
    for s in SP:
        row = R["decompositions"]["airport"][c][s]["by_airport"]
        print(f"| {nm} | {s} | " + " | ".join(
            ("—" if row[a]["m_airport"] is None else f"{row[a]['m_airport']:.3f}") for a in ap_names) + " |")
print("| q_a (exp-055) | v20 | " + " | ".join(
    ("—" if R["decompositions"]["airport"]["feb_K3"]["v20"]["by_airport"][a]["q_airport_exp055"] is None
     else f"{R['decompositions']['airport']['feb_K3']['v20']['by_airport'][a]['q_airport_exp055']:.3f}")
    for a in ap_names) + " |")
print("H rows: " + ", ".join(
    f"{a} {R['decompositions']['airport']['feb_K3']['v20']['by_airport'][a]['H_rows']:,}" for a in ap_names))

head("NM-regime decomposition inside H (m by regime)")
regs = sorted(R["decompositions"]["nm_regime"]["feb_K3"]["v20"]["by_regime"])
print("| arm | step | " + " | ".join(f"{r} (m / plain mass share)" for r in regs) + " |")
print("|---|---|" + "---|" * len(regs))
for c, nm in ARMS + LADDER_OTHER:
    for s in SP:
        row = R["decompositions"]["nm_regime"][c][s]["by_regime"]
        print(f"| {nm} | {s} | " + " | ".join(
            (f"{row[r]['m_regime']:.3f} / {row[r]['plain_mass_share']:.3f}"
             if row[r]["m_regime"] is not None else "—") for r in regs) + " |")
print("H rows by regime: " + ", ".join(
    f"{r} {R['decompositions']['nm_regime']['feb_K3']['v20']['by_regime'][r]['H_rows']:,}" for r in regs))

head("Serving-regime step mass (every bag step is zero outside H)")
print("| arm | step | H | G | A | C | all carrier mass in H |")
print("|---|---|---|---|---|---|---|")
for c, nm in ALL:
    for s in SP:
        d = R["decompositions"]["serving_regime"][c][s]
        by = d["by_regime"]
        print(f"| {nm} | {s} | " + " | ".join(
            (f"{by[g]['carrier_step_mass_share']:.6f}" if by[g]["carrier_step_mass_share"] is not None else "0")
            for g in ("H", "G", "A", "C")) + f" | {d['all_carrier_mass_in_H']} |")

head("Changed rows by serving regime (the raw members move rows outside H; the bags do not)")
print("| arm | step | carrier changed rows H / G / A / C | plain changed rows H / G / A / C |")
print("|---|---|---|---|")
for c in ("feb_K3", JSC, "feb_s1", f"ladder_{JS}_s1"):
    for s in SP:
        q = R["points"][c]["steps"][s]["changed_rows"]
        print(f"| {c} | {s} | " + " / ".join(f"{q['carrier'][g]:,}" for g in ("H", "G", "A", "C"))
              + " | " + " / ".join(f"{q['plain'][g]:,}" for g in ("H", "G", "A", "C")) + " |")

head("Fit-count record: rms of the step on July H (s)")
for s in SP:
    r = R["fit_count_record_july_H"][s]["rms_H_step_s"]
    print(f"{s}: " + ", ".join(f"{nm} {r[nm]:.3f}" for nm in sorted(r)))
    c_ = R["fit_count_record_july_H"][s]["cos_with_K10_plain_step"]
    print("    cos with the K = 10 plain step: " + ", ".join(f"{nm} {c_[nm]:.3f}" for nm in sorted(c_)))

head("COMMON's populations (RMSE in s)")
for f in ("july_loo", "december", "january"):
    p = R["populations"][f]
    print(f"\n**{f}** rows: " + ", ".join(f"{k} {p['rows'][k]:,}" for k in
                                          ("all", "le3600", "gt3600", "no_nm", "clean")))
    keys = [k for k in p if k.endswith(("_L0", "_L1", "_L2", "_L3"))]
    print("| predictor | all | ≤ 3,600 s | > 3,600 s | no-NM | clean |")
    print("|---|---|---|---|---|---|")
    for k in sorted(keys):
        v = p[k]
        print(f"| {k} | " + " | ".join(
            ("—" if v[m] is None else f"{v[m]:.3f}") for m in ("all", "le3600", "gt3600", "no_nm", "clean")) + " |")

head("COMMON's populations by serving regime (RMSE in s, all rows of the regime)")
for f in ("july_loo", "december", "january"):
    p = R["populations"][f]
    print(f"\n**{f}** rows by regime: " + ", ".join(
        f"{g} {p['rows_by_serving_regime'][g]['all']:,}" for g in ("H", "G", "A", "C")))
    print("| predictor | H | G | A | C |")
    print("|---|---|---|---|---|")
    for k in sorted(p["by_serving_regime"]):
        v = p["by_serving_regime"][k]
        print(f"| {k} | " + " | ".join(
            ("—" if v[g]["all"] is None else f"{v[g]['all']:.3f}") for g in ("H", "G", "A", "C")) + " |")

head("Rule O of exp-058, reproduced")
print(json.dumps(R["rules"]["O_exp058_reproduced"], indent=1, sort_keys=True))

head("Registered predictions (SPEC §10) against the record")
print(json.dumps(R["registered_predictions"], indent=1, sort_keys=True))

head("Checks")
for k, v in sorted(R["checks"].items()):
    print(f"  {k}: {json.dumps(v) if not isinstance(v, float) else f'{v:.3e}'}")
print("\nR7 (exp-057):")
r7 = R["R7_exp057"]
print(f"  points max |diff| m {r7['points_max_abs_diff_m']:.3e}, loss {r7['points_max_abs_diff_loss']:.3e}")
for k, v in sorted(r7["draw_receipts"].items()):
    print(f"  draws-{k}: {v['rows']} rows x {v['columns']} columns, max rel {v['max_rel_diff']:.3e}, "
          f"bytes identical {v['bytes_identical']}")
print(f"  daysums: {r7['daysums_shared_columns']} shared columns, max rel {r7['daysums_max_rel_diff']:.3e}")
print("\nR8 (exp-058):")
r8 = R["R8_exp058"]
print(json.dumps({k: v for k, v in r8.items() if not isinstance(v, dict)}, indent=1, sort_keys=True))
for k, v in sorted(r8.items()):
    if isinstance(v, dict):
        print(f"  {k}: {json.dumps(v, sort_keys=True)[:400]}")

head("New vectors")
for t, v in sorted(R["new_vectors"].items()):
    print(f"  {t}: {v.get('vector', v.get('standin_for'))} {v.get('sha256', '')[:16]} "
          f"moved {v.get('rows_moved', '—')} wall {v.get('seconds_fit_wall', '—')} s "
          f"cpu {v.get('seconds_fit_process_cpu', '—')} s")

head("Receipts and OOF")
for k, v in sorted(R["receipts"].items()):
    print(f"  {k}: {v[:16]}…")
for k, v in sorted(R["oof"].items()):
    print(f"  {k}: {v['sha256'][:16]}… {v['rows']:,} rows, {len(v['columns'])} columns")
