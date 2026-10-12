"""RESULTS section 47.57: the FIVE-month July head, registered BEFORE any 2025-06 read.

2025-06 is a SECOND built month at served distance 1 from 2026-07 (Aug is the other), so
section 47.46's "nearest built month, unique" no longer picks a single month for the July
head.  The desk's ruling replaces it: a head's month set is EVERY built month at the MINIMUM
served distance, ties included.  July gets folds + {Nov, Oct, Mar, Aug, Jun}; January keeps
folds + {Nov, Oct, Mar, Feb} because Feb at d = 1 is still unique; december, unserved, reads
with the January head.

Everything here is arithmetic on numbers already in the record.  NOTHING is read from 2025-06:
its alone-value is PREDICTED, twice, and both predictions are registered before the fit.

Its own check is that it reproduces section 47.52's measured four-month july marginal from the
measured Aug alone-value through the same conversion, and it refuses to write if it does not.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))

import monthpred as M                                                 # noqa: E402

FOLD = "july_loo"

#: measured alone-values on july: sections 47.26.5 / 47.41 for the three, 47.51 for Aug
AUG_ALONE_JULY = 628.73
#: 47.52's MEASURED four-month marginal over the three-month member, and the total it makes
AUG_MARGINAL_JULY = 354.35
#: 47.45.2's cross-fitted line at d = 1.  June is at d = 1 as Aug is, so the registered form
#: gives June the SAME point -- the line sees distance, nothing else.
CROSSFIT_D1 = 818.05
#: the companion input: Aug is the only d = 1 month ever MEASURED, and it read 23 % below the
#: line.  Section 47.53's ruling says carry both when they differ by more than the noise floor
#: (july's is 83.22 s2/row; these differ by 189.32).
MEASURED_D1 = AUG_ALONE_JULY

K1_TO_K10_JULY = 387.98 / 372.62      # 47.32 against 47.40, the measured scaling
V11_JULY_READ = 307.41                # 47.55, the seasonal bag's july arm over v10's head
V11_BOARD_OVER_V10 = 0.398005522      # 47.55 / 47.56, the identity-checked figure

#: 47.44.1's law on july, and the row counts the new set carries
DIS = {"c": 12.2202, "p": 0.1477, "d3": 10.4831}
DIS_RESID_S = 0.370
FOLD_ROWS_JULY = 172783
EXT_JUL_FOUR = 386574                 # Nov + Oct + Mar + Aug, the set v11 ships
NOMINAL_JUNE_ROWS = 100000            # a 30-day month; the delivered count replaces it


def totals() -> dict:
    """The july stack totals: three measured, four measured, five predicted twice."""
    a3 = [M.ALONE[m][FOLD] for m in M.FITTED]
    t3 = M.TOTAL3[FOLD]
    t4 = t3 + AUG_MARGINAL_JULY
    p = M.stack_exponent(FOLD)
    # the self-check: the same conversion, fed Aug's MEASURED alone-value, must reproduce
    # 47.52's measured four-month marginal to the digit that section printed (+359.87)
    check = M.marginal(FOLD, a3 + [AUG_ALONE_JULY], t3)
    if abs(check - 359.87) > 0.01:
        raise SystemExit(f"the conversion gives {check:.2f} for Aug's fourth month where "
                         f"section 47.52 recorded +359.87 -- the code path has moved and no "
                         f"prediction may be written from it")
    return {"a4": a3 + [AUG_ALONE_JULY], "total_3_measured": t3, "total_4_measured": t4,
            "p": p, "aug_fourth_month_reproduced": round(check, 2)}


def fifth_month(june_alone: float, t: dict) -> dict:
    """June's marginal as the FIFTH month, over the MEASURED four-month total."""
    a5 = t["a4"] + [june_alone]
    power = 5 ** t["p"] * (sum(a5) / 5) - t["total_4_measured"]
    # the additive form at the efficiency the FOUR-month stack actually reached
    eff4 = t["total_4_measured"] / sum(t["a4"])
    additive = june_alone * eff4
    return {"june_alone_value_used": june_alone,
            "k1_marginal_power_law": power, "k1_marginal_additive": additive,
            "efficiency_at_k4_measured": eff4,
            "k10_july_increment_over_v11": power * K1_TO_K10_JULY,
            "k10_july_increment_over_v11_additive": additive * K1_TO_K10_JULY}


def disagreement(rows_june: int) -> dict:
    mult = (FOLD_ROWS_JULY + EXT_JUL_FOUR + rows_june) / FOLD_ROWS_JULY
    d = DIS["c"] * mult ** -DIS["p"]
    L = math.log(d / DIS["d3"])
    se = DIS_RESID_S / 12
    return {"rows_june_used": rows_june, "fitting_multiple": round(mult, 4),
            "three_month_disagreement_s": DIS["d3"],
            "predicted_disagreement_s": round(d, 3),
            "july_only_L": L, "july_only_se": se,
            "F1_mechanism_falsified_if": "L >= 0",
            "F2_law_falsified_if_outside": [L - 2 * se, L + 2 * se],
            "note": ("47.55 measured L = +0.0119 pooled over three folds and BOTH F1 and F2 "
                     "fired. This is the july-only restatement at the five-month multiple, "
                     "registered so the next read scores the same claim rather than a new "
                     "one; the mechanism is already falsified on the rule as written")}


def main() -> None:
    t = totals()
    coef = json.loads((EXP / "analysis/monthrule.json").read_text())["board_proxy"]
    lead = fifth_month(CROSSFIT_D1, t)
    comp = fifth_month(MEASURED_D1, t)
    # break-even: what June must be worth ALONE for a fifth month to be positive at all
    be = t["total_4_measured"] * 5 / (5 ** t["p"]) - sum(t["a4"])
    rec = {
        "what": __doc__.strip().splitlines()[0],
        "ruling": {
            "registered_in": "RESULTS.md 47.57, desk ruling, before any 2025-06 read",
            "form": ("a head's month set is EVERY built month at the MINIMUM served "
                     "distance, ties included"),
            "july_head": ["2025-11", "2025-10", "2025-03", "2025-08", "2025-06"],
            "january_head": ["2025-11", "2025-10", "2025-03", "2025-02"],
            "december_reads_with": "the January head (unserved, nearest Feb at d = 2)",
            "why_parameter_free": ("served distance is 47.30.1's, fixed by the calendar; the "
                                   "tie is broken by INCLUDING both, which needs no number. "
                                   "No fold reading picks a month")},
        "served_distances": {"2025-06": {"d_2026_01": 5, "d_2026_07": 1},
                             "2025-08": {"d_2026_01": 5, "d_2026_07": 1},
                             "2025-02": {"d_2026_01": 1, "d_2026_07": 5}},
        "july_totals": t,
        "break_even_june_alone_value": be,
        "lead_form_power_law_at_the_crossfit_point": lead,
        "companion_at_the_measured_d1_value": comp,
        "form_choice": ("47.53's ruling names the form per k with the evidence for that k. "
                        "July is the NEAR fold, where the power law was right to 1.6 % at "
                        "k = 4 (+359.87 against a read of +354.35), so the power law leads "
                        "here. The additive form is carried as the companion, not as the "
                        "point, and both June inputs are carried because they differ by "
                        "189.32 s2/row against july's 83.22 noise floor"),
        "board": {
            "v11_board_over_v10_head": V11_BOARD_OVER_V10,
            "increment_from_june_lead": coef["cj_july"] * lead["k10_july_increment_over_v11"],
            "increment_from_june_companion":
                coef["cj_july"] * comp["k10_july_increment_over_v11"],
            "total_over_v10_head_lead":
                V11_BOARD_OVER_V10 + coef["cj_july"] * lead["k10_july_increment_over_v11"],
            "total_over_v10_head_companion":
                V11_BOARD_OVER_V10 + coef["cj_july"] * comp["k10_july_increment_over_v11"],
            "why_january_does_not_move": ("June is at d = 5 from 2026-01, so it enters no "
                                          "January-head set. The January arm and december "
                                          "keep 47.55's reads exactly")},
        "disagreement_falsifier_47_44": disagreement(NOMINAL_JUNE_ROWS),
        "falsifiers": {
            "F_gate": ("rule (a) on the read: july's five-month increment over v11's july arm "
                       "must have an interval clear of zero. A negative or crossing interval "
                       "FAILS the ruling's July head and v11 stands"),
            "F_form": ("if the read sits closer to the additive companion than to the power-law "
                       "point by more than july's 83.22 noise floor, the k = 5 form choice was "
                       "wrong and 47.53's per-k rule takes the additive form at k = 5"),
            "F_input": ("if the read implies a June alone-value below the break-even "
                        f"{be:.1f}, a d = 1 month did not carry its distance and the "
                        "cross-fitted line's d = 1 point is falsified a second time"),
            "F_tie": ("the ruling itself is falsified if the five-month july read is BELOW "
                      "the four-month read v11 ships: including a tied month would then have "
                      "cost the fold it was included for")},
        "queue_observation": {
            "what_queue_says": ("2025-06 is the hardest month built and its members agree "
                                "most closely, a 14.7 s spread"),
            "what_it_means_for_this_rule": ("a served-distance rule CANNOT see noise, by "
                                            "construction. That is exactly why no fold "
                                            "reading can pick a month, and it is also the "
                                            "risk being taken: the rule includes June "
                                            "whatever its spread"),
            "and_what_may_not_happen_next": ("a noise-aware tie-break would be a NEW "
                                             "parameter. If June's read disappoints, that "
                                             "parameter must be registered BEFORE it is "
                                             "used, never chosen after seeing which month "
                                             "it would drop")},
    }
    out = EXP / "analysis/junepred.json"
    out.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"july head -> {'+'.join(rec['ruling']['july_head'])}")
    print(f"Aug's 4th month reproduced at {t['aug_fourth_month_reproduced']:+.2f} "
          f"(47.52 recorded +359.87); measured total(4) {t['total_4_measured']:.2f}")
    print(f"break-even June alone-value {be:+.1f}")
    for name, v in (("lead (cross-fit d=1 +818.05)", lead),
                    ("companion (measured d=1 +628.73)", comp)):
        print(f"  {name:34s} K=1 power {v['k1_marginal_power_law']:+8.2f}  "
              f"additive {v['k1_marginal_additive']:+8.2f}  "
              f"K=10 over v11 {v['k10_july_increment_over_v11']:+8.2f}")
    b = rec["board"]
    print(f"board over v10's head: v11 {b['v11_board_over_v10_head']:+.5f} -> lead "
          f"{b['total_over_v10_head_lead']:+.5f} / companion "
          f"{b['total_over_v10_head_companion']:+.5f}")
    d = rec["disagreement_falsifier_47_44"]
    print(f"47.44 at {d['rows_june_used']:,} June rows: multiple {d['fitting_multiple']}, "
          f"{d['three_month_disagreement_s']} -> {d['predicted_disagreement_s']} s, "
          f"L {d['july_only_L']:+.4f} se {d['july_only_se']:.4f}")
    print(f"written {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
