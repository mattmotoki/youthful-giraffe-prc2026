"""Apply the PREDECLARED density rule to the two-way table, mechanically.

Sol round 1, blocking 7: the report announced "DENSITY" without running the
conditional rule `analysis/predeclared.json` carries, and called the traffic
marginal monotone when it is not.  This script reads the two-way tables
`code/density.py` already wrote, applies the rule as written, and stores the
counts the verdict is read from.  It measures nothing new.

The rule: *"DENSITY if the ordering by traffic rate survives conditioning on
cell rows and the ordering by cell rows does not survive conditioning on traffic
rate; MECHANISM if the reverse; MIXED if both survive."*  An ordering "survives"
inside a conditioning stratum when the pair reduction falls from one band to the
next; the count of adjacent band pairs that fall, over the adjacent band pairs
available, is what the verdict is read from.

    python verdict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zs                                   # noqa: E402

EXP = zs.EXP
A = EXP / "analysis"
GRIDS = ("300", "600", "1200")


def _key(b: str) -> int:
    return int(b.split("-")[0].rstrip("+"))


def falls(table: list[dict], hold: str, vary: str) -> list[dict]:
    """Adjacent-band comparisons of `vary` inside each level of `hold`."""
    out = []
    levels = sorted({r[hold] for r in table}, key=_key)
    for lv in levels:
        rows = sorted([r for r in table if r[hold] == lv], key=lambda r: _key(r[vary]))
        for a, b in zip(rows, rows[1:]):
            out.append({hold: lv, "from": a[vary], "to": b[vary],
                        "n_from": a["n"], "n_to": b["n"],
                        "reduction_from": a["reduction_s2"], "reduction_to": b["reduction_s2"],
                        "falls": bool(b["reduction_s2"] < a["reduction_s2"])})
    return out


def main() -> None:
    den = json.loads((A / "density.json").read_text())
    rule = json.loads((A / "predeclared.json").read_text())
    out = {"note": "Sol round 1, blocking 7. The predeclared rule applied to the two-way "
                   "matched table of code/density.py; no new measurement.",
           "rule": rule["step1_density_check"]["verdict_rule"],
           "grids": {}}
    tot_t = tot_tn = tot_c = tot_cn = 0
    for g in GRIDS:
        t = den["pooled"][g]["matched"]["table"]
        tr = falls(t, "cell_rows", "traffic_rate")     # traffic ordering | cell rows
        cr = falls(t, "traffic_rate", "cell_rows")     # cell-rows ordering | traffic
        out["grids"][g] = {
            "traffic_given_cell_rows": {"n_pairs": len(tr),
                                        "n_falling": sum(r["falls"] for r in tr),
                                        "pairs": tr},
            "cell_rows_given_traffic": {"n_pairs": len(cr),
                                        "n_falling": sum(r["falls"] for r in cr),
                                        "pairs": cr}}
        tot_t += sum(r["falls"] for r in tr); tot_tn += len(tr)
        tot_c += sum(r["falls"] for r in cr); tot_cn += len(cr)
    # the traffic marginal, to check the monotonicity claim the report made
    mono = {}
    for g in GRIDS:
        m = den["pooled"][g]["matched"]["by_traffic_rate"]
        bands = sorted(m, key=_key)
        vals = [m[b]["reduction_s2"] for b in bands]
        mono[g] = {"bands": bands, "values": vals,
                   "monotone_decreasing": all(b < a for a, b in zip(vals, vals[1:]))}
    # Two other honest readings of "survives", so the verdict can be shown to be
    # stable under the choice rather than resting on it.
    def reading(fn, tkey: str = "table") -> dict:
        a = b = an = bn = 0.0
        for g in GRIDS:
            T = den["pooled"][g]["matched"][tkey]
            x, xn = fn(falls(T, "cell_rows", "traffic_rate"))
            y, yn = fn(falls(T, "traffic_rate", "cell_rows"))
            a += x; an += xn; b += y; bn += yn
        v = ("MIXED" if a > an / 2 and b > bn / 2 else
             "DENSITY" if a > an / 2 else "MECHANISM" if b > bn / 2 else "NEITHER")
        return {"traffic_given_cell_rows": [round(a, 1), round(an, 1)],
                "cell_rows_given_traffic": [round(b, 1), round(bn, 1)], "verdict": v}

    def adjacent(pairs):
        return float(sum(r["falls"] for r in pairs)), float(len(pairs))

    def weighted(pairs):
        w = [min(r["n_from"], r["n_to"]) for r in pairs]
        return (float(sum(x for r, x in zip(pairs, w) if r["falls"])), float(sum(w)))

    def rng_test(pairs):
        """First band against last, inside each conditioning level."""
        by: dict = {}
        for r in pairs:
            k = r.get("cell_rows", r.get("traffic_rate"))
            by.setdefault(k, []).append(r)
        n = f = 0
        for rows in by.values():
            n += 1
            f += rows[-1]["reduction_to"] < rows[0]["reduction_from"]
        return float(f), float(n)

    def strict(pairs):
        """EVERY adjacent comparison must fall; one reversal ends the ordering."""
        return (1.0 if pairs and all(r["falls"] for r in pairs) else 0.0), 1.0

    out["other_readings"] = {"adjacent_bands": reading(adjacent),
                             "pair_weighted": reading(weighted),
                             "first_band_against_last": reading(rng_test),
                             "every_comparison_must_fall": reading(strict)}
    # Sol round 2, blocking 3: the same countings on the two-way table WITHOUT the
    # undeclared 200-pair minimum, where density.json carries it.
    if "table_all" in den["pooled"]["600"]["matched"]:
        out["other_readings_no_min_pairs"] = {
            "adjacent_bands": reading(adjacent, "table_all"),
            "pair_weighted": reading(weighted, "table_all"),
            "first_band_against_last": reading(rng_test, "table_all"),
            "every_comparison_must_fall": reading(strict, "table_all")}
    out["verdict_is_reading_dependent"] = len(
        {v["verdict"] for v in out["other_readings"].values()}) > 1
    both = tot_t > tot_tn / 2 and tot_c > tot_cn / 2
    verdict = ("MIXED" if both else
               "DENSITY" if tot_t > tot_tn / 2 else
               "MECHANISM" if tot_c > tot_cn / 2 else "NEITHER")
    out["pooled"] = {"traffic_given_cell_rows": [tot_t, tot_tn],
                     "cell_rows_given_traffic": [tot_c, tot_cn]}
    out["traffic_marginal_monotone"] = mono
    out["verdict"] = verdict
    (A / "verdict.json").write_text(json.dumps(out, indent=2))
    for g in GRIDS:
        d = out["grids"][g]
        print(f"  grid {g:>4} s: traffic|cell_rows "
              f"{d['traffic_given_cell_rows']['n_falling']}/{d['traffic_given_cell_rows']['n_pairs']}"
              f"   cell_rows|traffic "
              f"{d['cell_rows_given_traffic']['n_falling']}/{d['cell_rows_given_traffic']['n_pairs']}"
              f"   traffic marginal monotone: {mono[g]['monotone_decreasing']}")
    print(f"  pooled: traffic|cell_rows {tot_t}/{tot_tn}, cell_rows|traffic {tot_c}/{tot_cn}"
          f"  -> VERDICT {verdict}")
    for k, v in out["other_readings"].items():
        print(f"  reading {k:24s} traffic {v['traffic_given_cell_rows'][0]:g}/"
              f"{v['traffic_given_cell_rows'][1]:g}  cell_rows "
              f"{v['cell_rows_given_traffic'][0]:g}/{v['cell_rows_given_traffic'][1]:g}"
              f"  -> {v['verdict']}")
    print(f"  written {A / 'verdict.json'}")


if __name__ == "__main__":
    main()
