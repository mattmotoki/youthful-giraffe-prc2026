"""exp-058 origincontrast: the tables the round-1 review found missing.  STDOUT ONLY.

`oc_tables.py` stays as it was reviewed at commit 3034823; this printer adds what it left out,
from the same record:

  * the full loss report per arm and step -- `m* = A/B`, `A^2/B`, `g_C - A^2/B`, the direction and
    orthogonal parts of the alignment term, `err_hat` and `err_plain`, the direct SSE gain over
    every fold row, `g_P` on the arm's own `p0` and `g_P - g_plain`, and the two outside-`H` terms;
  * COMMON's four populations and `clean` *inside* every serving regime, not the regime total only;
  * registered prediction 14 read on the six registered airports (SPEC 10 item 14 says "all 6
    airports of both arms"), beside the nine-airport reading the official run recorded.

Writes nothing.

    .venv/bin/python experiments/exp-058-origincontrast/code/oc_tables_r1.py [--record PATH]
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
# SPEC 3's six shared airports: the block set of both arms.  July `H` also holds LEMD, LFPG, LIRF
# and LTFM, which no arm adds rows at.
SIX = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LSZH")
MASKS = (("all", "all"), ("le3600", "<= 3,600 s"), ("gt3600", "> 3,600 s"), ("no_nm", "no-NM"),
         ("clean", "clean"))


def head(t):
    print(f"\n### {t}\n")


def f2(x, w="+.2f"):
    return "—" if x is None else format(x, w)


print(f"record: {ARGS.record}")

head("Loss report, part 1 (s² per July fold row, n = 190,713, sums over H)")
print("| arm | step | A | B | m | m* = A/B | A²/B | g_C − A²/B | err_hat | err_plain |")
print("|---|---|---|---|---|---|---|---|---|---|")
for c, nm in ARMS + COMPARATORS:
    for s in SP:
        q = R["points"][c]["steps"][s]
        print(f"| {nm} | {s} | {q['A']:.2f} | {q['B']:.2f} | {q['m']:.4f} | {q['m_star']:.4f} | "
              f"{q['best_amplitude_value']:+.2f} | {q['g_C_minus_best']:+.2f} | "
              f"{q['err_hat']:+.2f} | {q['err_plain']:+.2f} |")

head("Loss report, part 2 (the same rows: the alignment split, the direct SSE, g_P, outside H)")
print("| arm | step | alignment | direction part | orthogonal part | direct SSE gain, all rows | "
      "g_P (own p0, all rows) | g_P − g_plain | outside-H g_C | outside-H plain step gain |")
print("|---|---|---|---|---|---|---|---|---|---|")
for c, nm in ARMS + COMPARATORS:
    for s in SP:
        q = R["points"][c]["steps"][s]
        print(f"| {nm} | {s} | {q['alignment']:+.2f} | {q['alignment_direction']:+.2f} | "
              f"{q['alignment_orthogonal']:+.2f} | {q['direct_sse_gain_all_rows']:+.2f} | "
              f"{q['g_P_own_p0_all_rows']:+.2f} | {q['g_P_minus_g_plain']:+.2f} | "
              f"{q['outside_H_g_C']:+.2f} | {q['outside_H_plain_step_gain']:+.2f} |")
mx = max(abs(R["points"][c]["steps"][s]["g_C"] - R["points"][c]["steps"][s]["direct_sse_gain_all_rows"])
         for c, _ in ARMS + COMPARATORS for s in SP)
print(f"\nmax |g_C − direct SSE gain| over the 15 arm-steps: {mx:.3e} s²/row")

head("COMMON's populations inside every serving regime (RMSE in s)")
for f in ("july_loo", "december", "january"):
    p = R["populations"][f]
    print(f"\n**{f}**")
    for g in ("H", "G", "A", "C"):
        rows = p["rows_by_serving_regime"][g]
        print("\nregime " + g + " rows: " + ", ".join(f"{lab} {rows[k]:,}" for k, lab in MASKS))
        print("| predictor | " + " | ".join(lab for _, lab in MASKS) + " |")
        print("|---|" + "---|" * len(MASKS))
        for k in sorted(p["by_serving_regime"]):
            v = p["by_serving_regime"][k][g]
            print(f"| {k} | " + " | ".join(f2(v[m], ".3f") for m, _ in MASKS) + " |")

head("Registered prediction 14 on the six registered airports")
apd = R["decompositions"]["airport"]
for c, nm in ARMS:
    for s in ("v20", "v21"):
        by = apd[c][s]["by_airport"]
        six = {a: by[a]["m_airport"] for a in SIX}
        other = {a: v["m_airport"] for a, v in by.items() if a not in SIX}
        print(f"{nm} {s}: six-airport max {max(six.values()):.4f} "
              f"({max(six, key=six.get)}), all six below 1: {all(v < 1 for v in six.values())}")
        print("   six:   " + ", ".join(f"{a} {six[a]:.4f}" for a in SIX))
        print("   other: " + ", ".join(f"{a} " + ("no H rows" if v is None else f"{v:.4f}")
                                       for a, v in sorted(other.items())))
        print(f"   the official run's nine-airport flag: "
              f"{R['registered_predictions']['14_m_airport_below_1'][c][s]['all_six_below_1']}")
