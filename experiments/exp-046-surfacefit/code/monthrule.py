"""Section 47.30/47.31: the cyclic-distance rule priced over the whole month pool.

Fits nothing and reads no data -- it is arithmetic on readings already in the record,
so the registration is reproducible instead of transcribed.  Emits analysis/monthrule.json.
"""
import json, math
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

FOLD_MONTH = {"july_loo": 7, "december": 12, "january": 1}
SERVED = {"2026-01": 1, "2026-07": 7}
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}

# the two measured extension months, at matched share (47.25, 47.26)
MEASURED = {"2025-11": {"july_loo": 312.90, "december": 957.11, "january": 405.94},
            "2025-10": {"july_loo": 493.60, "december": 736.15, "january":  78.75}}
BOARD_MEASURED = {"2025-11": 0.59236, "2025-10": 0.51831}
# held out of every fit below
HELDOUT = {"two-month full weight": ((639.07, 1209.36, 341.82), 0.84920),
           "two-month one-third weight": ((551.31, 994.09, 277.53), 0.71958)}
MISMATCH = {"july_loo": 62.94, "december": -61.49, "january": -98.60}   # 47.19 control
TWO_MONTH_TOTAL = {"july_loo": 639.07, "december": 1209.36, "january": 341.82}


def cyc(a: int, b: int) -> int:
    """Month-of-year distance, wrapping at 12."""
    return min(abs(a - b), 12 - abs(a - b))


def main() -> None:
    # -- the served months share a month-of-year with two folds: assert it rather than assume it
    assert SERVED["2026-01"] == FOLD_MONTH["january"], "2026-01 is not the january fold's month"
    assert SERVED["2026-07"] == FOLD_MONTH["july_loo"], "2026-07 is not the july fold's month"
    assert FOLD_MONTH["december"] not in SERVED.values(), "december has a served counterpart"
    assert set(SHARES) == set(FOLD_MONTH) - {"december"}, \
        "the board-share folds are not exactly the folds with a served counterpart"

    # -- the line through the two measured months, per fold
    line = {}
    for f in FOLD_MONTH:
        (m1, m2) = ("2025-11", "2025-10")
        d1, d2 = cyc(11, FOLD_MONTH[f]), cyc(10, FOLD_MONTH[f])
        if d1 == d2:
            raise SystemExit(f"{f}: the two months sit at the same distance; the line is not "
                             f"determined and the rule cannot be fitted on this fold")
        b = (MEASURED[m2][f] - MEASURED[m1][f]) / (d2 - d1)
        line[f] = {"intercept": MEASURED[m1][f] - b * d1, "slope": b,
                   "zero_crossing": -(MEASURED[m1][f] - b * d1) / b}

    # -- the board proxy, solved on the two single-month reads
    import numpy as np
    A = np.array([[MEASURED[m]["july_loo"], MEASURED[m]["january"]] for m in ("2025-11", "2025-10")])
    cj, ca = np.linalg.solve(A, np.array([BOARD_MEASURED[m] for m in ("2025-11", "2025-10")]))
    board = lambda j, a: float(cj * j + ca * a)
    checks = {k: {"predicted": board(v[0][0], v[0][2]), "measured": v[1],
                  "error": board(v[0][0], v[0][2]) - v[1]} for k, v in HELDOUT.items()}
    worst = max(abs(c["error"]) for c in checks.values())
    if worst > 0.002:
        raise SystemExit(f"the board proxy misses a held-out reading by {worst:.5f} s -- it is not "
                         f"the scoring convention showing through and must not be used to rank months")
    ratio_err = abs(float(cj / ca) - SHARES["july_loo"] / SHARES["january"])
    if ratio_err > 0.01:
        raise SystemExit(f"proxy coefficient ratio is {cj/ca:.4f}, SHARES says "
                         f"{SHARES['july_loo']/SHARES['january']:.4f} -- off by {ratio_err:.4f}")

    # -- every candidate month under both far-field forms
    pool = [m for m in range(1, 13) if m not in FOLD_MONTH.values()]
    months = {}
    for m in pool:
        d = {f: cyc(m, FOLD_MONTH[f]) for f in FOLD_MONTH}
        lin = {f: line[f]["intercept"] + line[f]["slope"] * d[f] for f in FOLD_MONTH}
        flo = {f: max(lin[f], MISMATCH[f]) for f in FOLD_MONTH}
        months[f"2025-{m:02d}"] = {
            "cyclic_distance_to_fold": d,
            "cyclic_distance_to_served": {s: cyc(m, mo) for s, mo in SERVED.items()},
            "predicted_LIN": lin, "predicted_FLOOR": flo,
            "board_LIN": board(lin["july_loo"], lin["january"]),
            "board_FLOOR": board(flo["july_loo"], flo["january"]),
            "spread": board(flo["july_loo"], flo["january"]) - board(lin["july_loo"], lin["january"])}
    disc = max(months, key=lambda k: months[k]["spread"])

    # -- 47.31: the three-month arm, bracketed
    three = ("2025-11", "2025-10", "2025-03")
    tri = {}
    for f in FOLD_MONTH:
        vals = [MEASURED[m][f] if m in MEASURED else months[m]["predicted_LIN"][f] for m in three]
        mean3 = sum(vals) / 3
        m2 = TWO_MONTH_TOTAL[f] / (sum(vals[:2]) / 2)
        p = math.log2(m2)
        tri[f] = {"M2": m2, "p": p, "mean_of_three": mean3,
                  "total_power_law": (3 ** p) * mean3, "total_saturated": m2 * mean3,
                  "increment_power_law": (3 ** p) * mean3 - MEASURED["2025-11"][f],
                  "increment_saturated": m2 * mean3 - MEASURED["2025-11"][f]}
    out = {
        "what": "section 47.30/47.31, registered before any data: the cyclic-distance rule priced "
                "over the whole month pool, and the three-month arm bracketed by two combination forms",
        "served_months_share_a_month_of_year_with": {"january": "2026-01", "july_loo": "2026-07",
                                                     "december": None},
        "line_per_fold": line,
        "board_proxy": {"cj_july": float(cj), "ca_january": float(ca), "december_weight": 0.0,
                        "coefficient_ratio": float(cj / ca),
                        "SHARES_ratio": SHARES["july_loo"] / SHARES["january"],
                        "held_out_checks": checks},
        "months": months,
        "discriminating_month": disc,
        "discriminating_spread": months[disc]["spread"],
        "three_month_arm": {"members": list(three), "by_fold": tri,
                            "board_increment_power_law":
                                board(tri["july_loo"]["increment_power_law"],
                                      tri["january"]["increment_power_law"]),
                            "board_increment_saturated":
                                board(tri["july_loo"]["increment_saturated"],
                                      tri["january"]["increment_saturated"])}}
    path = EXP / "analysis/monthrule.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"board proxy {cj:.8f}*july + {ca:.8f}*january, worst held-out error {worst:.5f} s")
    print(f"ranking under LIN:   {', '.join(sorted(months, key=lambda k: -months[k]['board_LIN']))}")
    print(f"ranking under FLOOR: {', '.join(sorted(months, key=lambda k: -months[k]['board_FLOOR']))}")
    print(f"discriminating month: {disc}, spread {months[disc]['spread']:+.3f} board s")
    print(f"three-month increment over the one-month member: board "
          f"{out['three_month_arm']['board_increment_power_law']:+.4f} (power law) to "
          f"{out['three_month_arm']['board_increment_saturated']:+.4f} (saturated)")
    print(f"written to {path.relative_to(EXP.parents[1])}")


if __name__ == "__main__":
    main()
