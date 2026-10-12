"""exp-058 origincontrast: print the RESULTS tables from analysis/measure.json.  STDOUT ONLY.

Writes nothing.  Every table in RESULTS.md comes from this printer, so the prose cannot drift from
the record.

    .venv/bin/python experiments/exp-058-origincontrast/code/oc_tables.py [--record PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

ap = argparse.ArgumentParser()
ap.add_argument("--record", default=str(EXP / "analysis/measure.json"))
ARGS = ap.parse_args()
R = json.loads(Path(ARGS.record).read_text())

SP = ["v20", "v21", "v22"]
ARMS = [("feb_K3", "February K = 3"), ("june_matched_K3", "matched June K = 3")]
COMPARATORS = [("carrier_K3", "full June K = 3 (exp-051 / exp-055)"),
               ("sham_K3", "K = 3 sham (exp-057)"), ("carrier_K10", "full June K = 10 (exp-057)")]


def iv(d, key="m", L=3, mode="day+member"):
    o = d["intervals"][f"L{L}"][mode]
    o = o[key] if key in o else o
    return f"[{o['lo']:+.4f}, {o['hi']:+.4f}]"


def head(t):
    print(f"\n### {t}\n")


print(f"record: {ARGS.record}")
print(f"spec commit {R['spec_commit']}, standin {R['standin']}, {R['seconds']} s wall / "
      f"{R.get('seconds_process_cpu')} s process cpu")

head("Rule O (SPEC §8)")
ro = R["rules"]["O"]
print(f"advances: {ro['advances']}")
for k, v in sorted(ro["conditions"].items()):
    print(f"  {k}: {v}")
print(f"reading: {ro['reading']}")

head("The two arms on July H (points, full member set)")
print("| arm | step | m | m − q | m / q | cos | rms_H carrier (s) | rms_H plain (s) | "
      "m 3-day day+member |")
print("|---|---|---|---|---|---|---|---|---|")
for c, nm in ARMS + COMPARATORS:
    for s in SP:
        q = R["points"][c]["steps"][s]
        print(f"| {nm} | {s} | **{q['m']:.4f}** | {q['m_minus_q']:+.4f} | {q['m_over_q']:.3f} | "
              f"{q['cos']:.3f} | {q['rms_H_carrier_s']:.3f} | {q['rms_H_plain_s']:.3f} | "
              f"{iv(q)} |")

head("Paired m_Feb − m_June (the same member triple and the same days in every draw)")
print("| step | m_Feb | m_June | difference | 1-day | 2-day | 3-day |")
print("|---|---|---|---|---|---|---|")
for s in SP:
    d = R["feb_minus_june_matched"][s]
    print(f"| {s} | {d['m_Feb']:.4f} | {d['m_June_matched']:.4f} | "
          f"**{d['m_Feb_minus_m_June']:+.4f}** | "
          + " | ".join(iv(d, key="all", L=L) for L in (1, 2, 3)) + " |")

head("Every raw member beside its bag (m on July H)")
print("| arm | v20 | v21 | v22 |")
print("|---|---|---|---|")
for pre, nm in (("feb", "February"), ("june_matched", "matched June")):
    for i in (1, 2, 3):
        q = R["points"][f"{pre}_s{i}"]["steps"]
        print(f"| {nm} member {i} (K = 1) | " + " | ".join(f"{q[s]['m']:.4f}" for s in SP) + " |")
    q = R["points"][f"{pre}_K3"]["steps"]
    print(f"| **{nm} bag (K = 3)** | " + " | ".join(f"**{q[s]['m']:.4f}**" for s in SP) + " |")
for pre, nm in (("carrier", "full June (exp-051)"), ("sham", "sham (exp-057)")):
    for i in (1, 2, 3):
        q = R["points"][f"{pre}_s{i}"]["steps"]
        print(f"| {nm} member {i} (K = 1) | " + " | ".join(f"{q[s]['m']:.4f}" for s in SP) + " |")
print("| m_Feb − m_June by member seed | "
      + " | ".join("/".join(f"{R['feb_minus_june_by_member'][f's{i}'][s]:+.3f}" for i in (1, 2, 3))
                   for s in SP) + " |")

head("Loss equations (s² per July fold row, n = 190,713, sums over H) — diagnostics only")
print("| arm | step | A | B | m | g_hat | g_plain | g_C | J | amplitude | alignment |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for c, nm in ARMS + COMPARATORS:
    for s in SP:
        q = R["points"][c]["steps"][s]
        print(f"| {nm} | {s} | {q['A']:.2f} | {q['B']:.2f} | {q['m']:.4f} | {q['g_hat']:+.2f} | "
              f"{q['g_plain']:+.2f} | {q['g_C']:+.2f} | {q['J']:+.2f} | {q['amplitude']:+.2f} | "
              f"{q['alignment']:+.2f} |")
for c, nm in ARMS + COMPARATORS:
    print(f"mean J {nm}: {R['points'][c]['mean_J']['point']:+.2f} "
          f"(3-day day+member {iv(R['points'][c]['mean_J'], key='all')})")

head("Own value at L0 ([SSE(plain_L0) − SSE(arm_L0)] / n, s²/row)")
print("| arm | value | 3-day day interval |")
print("|---|---|---|")
for nm, e in R["own_value_L0"]["by_arm"].items():
    print(f"| {nm} ({e['X']} vs {e['Y']}) | {e['value']:+.1f} | {iv(e, key='all', mode='day')} |")
print(f"exp-051's recorded full June K = 3 value: {R['own_value_L0']['exp051_june_K3_recorded']:+.2f}")

head("Per-airport m_a (July H)")
ap_names = sorted(R["decompositions"]["airport"]["feb_K3"]["v20"]["by_airport"])
print("| arm | step | " + " | ".join(ap_names) + " |")
print("|---|---|" + "---|" * len(ap_names))
for c, nm in ARMS + COMPARATORS:
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
for c, nm in ARMS:
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
for c, nm in ARMS + COMPARATORS:
    for s in SP:
        d = R["decompositions"]["serving_regime"][c][s]
        by = d["by_regime"]
        print(f"| {nm} | {s} | " + " | ".join(
            (f"{by[g]['carrier_step_mass_share']:.6f}" if by[g]["carrier_step_mass_share"] is not None else "0")
            for g in ("H", "G", "A", "C")) + f" | {d['all_carrier_mass_in_H']} |")

head("Changed rows by serving regime (the raw members move rows outside H; the bags do not)")
print("| arm | step | carrier changed rows H / G / A / C | plain changed rows H / G / A / C |")
print("|---|---|---|---|")
for c in ("feb_K3", "june_matched_K3", "feb_s1", "june_matched_s1"):
    for s in SP:
        q = R["points"][c]["steps"][s]["changed_rows"]
        print(f"| {c} | {s} | " + " / ".join(f"{q['carrier'][g]:,}" for g in ("H", "G", "A", "C"))
              + " | " + " / ".join(f"{q['plain'][g]:,}" for g in ("H", "G", "A", "C")) + " |")

head("Fit-count record: rms of the step on July H (s)")
for s in SP:
    r = R["fit_count_record_july_H"][s]["rms_H_step_s"]
    print(f"{s}: " + ", ".join(f"{nm} {r[nm]:.3f}" for nm in
                               ("F3bag", "M3bag", "C3bag", "S3bag", "P3bag", "C10bag", "P10bag", "T10")))
    c_ = R["fit_count_record_july_H"][s]["cos_with_K10_plain_step"]
    print(f"    cos with the K = 10 plain step: " + ", ".join(
        f"{nm} {c_[nm]:.3f}" for nm in ("F3bag", "M3bag", "C3bag", "S3bag", "P3bag", "C10bag", "T10")))

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
