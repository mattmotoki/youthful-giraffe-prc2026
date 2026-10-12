"""exp-057 originfit: print the RESULTS.md tables from analysis/measure.json (stdout only, no file).

    .venv/bin/python experiments/exp-057-originfit/code/ofit_tables.py [path/to/measure.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
M = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else EXP / "analysis/measure.json").read_text())
SP = ("v20", "v21", "v22")
P = M["points"]


def iv(d, f="{:.3f}"):
    return f"[{f.format(d['lo'])}, {f.format(d['hi'])}]"


def rule_s():
    print("| step | m_S (K = 3 sham) | 1-day | 2-day | 3-day | m_C (K = 3 carrier) | 1-day | 2-day | 3-day | m_S − m_C | 3-day paired |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in SP:
        a, c = P["sham_K3"]["steps"][s], P["carrier_K3"]["steps"][s]
        dd = M["sham_minus_carrier_K3"][s]
        print(f"| {s} | **{a['m']:.3f}** | " + " | ".join(iv(a["intervals"][f"L{L}"]["day+member"]["m"]) for L in (1, 2, 3))
              + f" | **{c['m']:.3f}** | " + " | ".join(iv(c["intervals"][f"L{L}"]["day+member"]["m"]) for L in (1, 2, 3))
              + f" | {dd['m_S_minus_m_C']:+.3f} | {iv(dd['intervals']['L3']['day+member'], '{:+.3f}')} |")
    print()
    print("conditions:", M["rules"]["S"]["conditions"], "advances:", M["rules"]["S"]["advances"])


def rule_z():
    print("| step | m (K = 10 carrier) | q | q 3-day interval | inside | m − q | m / q | cos | rms_H carrier (s) | rms_H plain (s) | m 1-day | m 2-day | m 3-day |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    z = M["rules"]["Z"]
    for s in SP:
        p = z["points"][s]
        qi = z["q_interval_L3"][s]
        print(f"| {s} | **{p['m']:.3f}** | {p['q']:.3f} | [{qi[0]:.3f}, {qi[1]:.3f}] | {'yes' if z['inside'][s] else 'no'} | "
              f"{p['m_minus_q']:+.3f} | {p['m_over_q']:.3f} | {p['cos']:.3f} | {p['rms_H_carrier_s']:.2f} | "
              f"{p['rms_H_plain_s']:.2f} | " + " | ".join(iv(p["m_day_member_intervals"][f"L{L}"]) for L in (1, 2, 3)) + " |")
    print()
    print("advances:", z["advances"])


def members():
    print("| comparison | K | m v20 | m v21 | m v22 | cos v20 | cos v21 | cos v22 | mean J |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c, e in P.items():
        st = e["steps"]
        print(f"| {c} | {e['K']} | " + " | ".join(f"{st[s]['m']:.3f}" for s in SP) + " | "
              + " | ".join(f"{st[s]['cos']:.3f}" for s in SP) + f" | {e['mean_J']['point']:+.1f} |")


def loss():
    for c in ("sham_K3", "carrier_K3", "carrier_K10"):
        print(f"\n{c}")
        print("| step | m | A | B | g_C | g_hat | g_plain | J | J 3-day day+member | m* = A/B | A²/B | g_C − A²/B | amplitude | alignment | (direction) | (orthogonal) | SSE recon. diff |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in SP:
            q = P[c]["steps"][s]
            j3 = q["intervals"]["L3"]["day+member"]["J"]
            print(f"| {s} | {q['m']:.3f} | {q['A']:.2f} | {q['B']:.2f} | {q['g_C']:+.2f} | {q['g_hat']:+.2f} | {q['g_plain']:+.2f} | "
                  f"**{q['J']:+.2f}** | {iv(j3, '{:+.1f}')} | {q['m_star']:.3f} | {q['best_amplitude_value']:.2f} | "
                  f"{q['g_C_minus_best']:+.2f} | {q['amplitude']:+.2f} | {q['alignment']:+.2f} | {q['alignment_direction']:+.2f} | "
                  f"{q['alignment_orthogonal']:+.2f} | {q['direct_sse_gain_all_rows'] - q['g_C'] - q['outside_H_g_C']:.1e} |")
        mj = P[c]["mean_J"]
        print(f"mean J {mj['point']:+.2f}; " + "; ".join(
            f"L{L} day {iv(mj['intervals'][f'L{L}']['day'], '{:+.1f}')} day+member {iv(mj['intervals'][f'L{L}']['day+member'], '{:+.1f}')}"
            for L in (1, 2, 3)))


def damage():
    d = M["damage"]
    print("| level | D (K = 3 sham) | 1-day d+m | 2-day d+m | 3-day d+m | D s1 | D s2 | D s3 |")
    print("|---|---|---|---|---|---|---|---|")
    for L in range(4):
        k = d["K3"][f"L{L}"]
        print(f"| L{L} | **{k['D']:+.1f}** | " + " | ".join(iv(k["intervals"][f"L{b}"]["day+member"], "{:+.0f}") for b in (1, 2, 3))
              + " | " + " | ".join(f"{d['members'][f's{i}'][f'L{L}']['D']:+.1f}" for i in (1, 2, 3)) + " |")
    for kk in ("K3", "K10"):
        o = d["own_value_L0"][kk]
        print(f"own value L0 {kk}: {o['value']:+.2f}; 3-day d+m {iv(o['intervals']['L3']['day+member'], '{:+.1f}')}; "
              f"3-day day {iv(o['intervals']['L3']['day'], '{:+.1f}')}")


def populations():
    for f, pop in M["populations"].items():
        print(f"\n{f} (rows: " + ", ".join(f"{k} {v:,}" for k, v in pop["rows"].items()) + ")")
        keys = ("all", "le3600", "gt3600", "no_nm", "clean", "own_H", "own_G", "own_A", "own_C")
        print("| predictor | " + " | ".join(keys) + " |")
        print("|---|" + "---|" * len(keys))
        for nm, v in pop.items():
            if nm in ("rows", "rows_by_serving_regime", "by_serving_regime"):
                continue
            print(f"| {nm} | " + " | ".join("–" if v[k] is None else f"{v[k]:.3f}" for k in keys) + " |")


def serving():
    for f, pop in M["populations"].items():
        print(f"\n{f} by serving regime (rows: " + "; ".join(
            f"{g}: " + ", ".join(f"{k} {n:,}" for k, n in r.items()) for g, r in pop["rows_by_serving_regime"].items()) + ")")
        print("| predictor | regime | all | le3600 | gt3600 | no_nm | clean |")
        print("|---|---|---|---|---|---|---|")
        for nm, byg in pop["by_serving_regime"].items():
            if not nm.endswith("L3"):
                continue
            for g, v in byg.items():
                print(f"| {nm} | {g} | " + " | ".join("–" if v[k] is None else f"{v[k]:.3f}" for k in
                                                      ("all", "le3600", "gt3600", "no_nm", "clean")) + " |")


def decomp():
    for c in ("sham_K3", "carrier_K10"):
        print(f"\n{c}: airport contributions to m² − 1 (m_a; exp-055 q_a)")
        a = M["decompositions"]["airport"][c]
        aps = list(a["v20"]["by_airport"])
        print("| airport | H rows | " + " | ".join(f"{s} contrib | {s} m_a | {s} q_a" for s in SP) + " |")
        print("|---|---|" + "---|---|---|" * 3)
        for ap in aps:
            row = f"| {ap} | {a['v20']['by_airport'][ap]['H_rows']:,} | "
            for s in SP:
                b = a[s]["by_airport"][ap]
                qa = "–" if b["q_airport_exp055"] is None else f"{b['q_airport_exp055']:.3f}"
                ma = "–" if b["m_airport"] is None else f"{b['m_airport']:.3f}"
                row += f"{b['contribution_to_m2_minus_1']:+.4f} | {ma} | {qa} | "
            print(row)
        print("| total m² − 1 | | " + " | ".join(f"{a[s]['m2_minus_1']:+.4f} | | " for s in SP) + "|")
        n = M["decompositions"]["nm_regime"][c]
        for s in SP:
            print(f"  {s} NM regimes: " + "; ".join(f"{r} {v['contribution_to_m2_minus_1']:+.4f} (m {v['m_regime']:.3f}, {v['H_rows']:,} rows)"
                                                   for r, v in n[s]["by_regime"].items()))


def fitcount():
    for s in SP:
        fc = M["fit_count_record_july_H"][s]
        r = fc["rms_H_step_s"]
        print(f"{s}: rms plain K1 " + ", ".join(f"{r[f'P{i}']:.2f}" for i in range(1, 11)) + f"; P3 {r['P3bag']:.2f}; P10 {r['P10bag']:.2f}; T10 {r['T10']:.2f}")
        print("     rms carrier K1 " + ", ".join(f"{r[f'C{i}']:.2f}" for i in range(1, 11)) + f"; C3 {r['C3bag']:.2f}; C10 {r['C10bag']:.2f}")
        print("     rms sham K1 " + ", ".join(f"{r[f'S{i}']:.2f}" for i in (1, 2, 3)) + f"; S3 {r['S3bag']:.2f}")
        print("     m K1 " + ", ".join(f"{fc['m']['K1'][f's{i}']:.3f}" for i in range(1, 11)) + f"; K3 {fc['m']['K3']:.3f}; "
              f"K10 {fc['m']['K10']:.3f}; sham K1 " + ", ".join(f"{fc['m']['sham_K1'][f's{i}']:.3f}" for i in (1, 2, 3))
              + f"; sham K3 {fc['m']['sham_K3']:.3f}")
        cw = fc["cos_with_K10_plain_step"]
        order = [f"P{i}" for i in range(1, 11)] + [f"C{i}" for i in range(1, 11)] + ["S1", "S2", "S3", "P3bag", "C3bag",
                                                                                     "S3bag", "C10bag", "T10"]
        print("     cos with P10 step: " + ", ".join(f"{k} {cw[k]:.3f}" for k in order))


def checks():
    print(json.dumps(M["checks"], indent=1))
    print("R2", {k: M["R2_exp051_gains"][k] for k in ("max_abs_diff", "n_keys", "other_keys_read")})
    print("R3/R4", json.dumps(M["R3_R4_exp055_scripts"]["registered_fields"]),
          {k: v["whole_record"] | {"receipts": v["receipt_hashes_equal"], "oof": v["oof_equal"]}
           for k, v in M["R3_R4_exp055_scripts"]["runs"].items()})
    print("R6", M["R6"]["plain_K10_vs_heads_max_abs_diff_s"], M["R6"]["S1_equals_exp051_fit_records"])
    print("census", {k: v for k, v in M["census_check"].items() if k != "files_opened"})


def preds():
    print(json.dumps(M["registered_predictions"], indent=1))


if __name__ == "__main__":
    for fn in (checks, rule_s, rule_z, members, loss, damage, decomp, fitcount, populations, serving, preds):
        print(f"\n===== {fn.__name__} =====")
        fn()
