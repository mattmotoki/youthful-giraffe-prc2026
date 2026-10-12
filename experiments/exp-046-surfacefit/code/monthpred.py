"""Section 47.45: the Aug read-out, registered before 2025-08 (and 2025-02) exist.

Fits nothing and reads no month data.  It is arithmetic on readings already in the record --
the three measured one-month values per fold (47.26, 47.41), the two- and three-month totals
(47.24, 47.32) and the 47.19 mismatch control -- so every figure the registration quotes is
reproducible instead of transcribed.  Emits analysis/monthpred.json.

Three quantities, all under both far-field forms:

  1. the one-month DISCRIMINATOR read for 2025-08, beside 47.39's decision rule;
  2. the marginal FOURTH-month increment of 2025-08 over the three-month head
     -- panel {Nov, Oct, Mar, Aug}, a rule (a) arm (47.38);
  3. the marginal FIFTH-month increment of 2025-08 over the FEB panel
     -- panel {Nov, Oct, Mar, Feb, Aug}, registered order (i).

Validated by reproducing 47.41.3's registered Feb figures (-153.5 / +4.6 LIN-FLOOR on july,
+112.9 december, +279.2 january) from the same code path, so the conversion used for Aug is
the conversion that was used for Feb and not a second one written to match.

    .venv/bin/python experiments/exp-046-surfacefit/code/monthpred.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

FOLD_MONTH = {"july_loo": 7, "december": 12, "january": 1}
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}

#: the three measured one-month reads at the matched 1.5x share (47.26.5, 47.41)
ALONE = {"2025-11": {"july_loo": 312.90, "december": 957.11, "january": 405.94},
         "2025-10": {"july_loo": 493.60, "december": 736.15, "january": 78.75},
         "2025-03": {"july_loo": 229.68, "december": 518.48, "january": 449.64}}
#: 47.19's mismatch control -- the floor under FLOOR
MISMATCH = {"july_loo": 62.94, "december": -61.49, "january": -98.60}
#: measured stack totals on the same scale: 47.24 (two months) and 47.32 (three months)
TOTAL2 = {"july_loo": 639.07, "december": 1209.36, "january": 341.82}
TOTAL3 = {"july_loo": 685.52, "december": 1328.63, "january": 655.40}
FITTED = ("2025-11", "2025-10", "2025-03")

#: 47.37's registered one-month discriminator figures, from each fold's OWN line (47.26.5)
AUG_47_37 = {"july_loo": (855.0, 855.0), "december": (294.2, 294.2),
             "january": (-575.6, -98.6)}

#: 47.41.3's registered Feb figures, reproduced here as the conversion's own check
FEB_REGISTERED = {"july_loo": (-153.5, 4.6), "december": (112.9, 112.9),
                  "january": (279.2, 279.2)}


def cyc(a: int, b: int) -> int:
    """Month-of-year distance, wrapping at 12."""
    return min(abs(a - b), 12 - abs(a - b))


def crossfit(month: int) -> dict:
    """47.38.1's registered pooling form: common slope, per-fold intercepts.

    For each held-out fold the slope and the two intercepts are fitted on the OTHER two folds'
    three measured months; the held-out fold's intercept is the mean of the two kept ones, so
    no reading of the scored fold enters its own prediction.
    """
    out = {}
    for held in FOLD_MONTH:
        keep = [g for g in FOLD_MONTH if g != held]
        num = den = 0.0
        cent = {}
        for g in keep:
            xs = [cyc(int(m[5:]), FOLD_MONTH[g]) for m in FITTED]
            ys = [ALONE[m][g] for m in FITTED]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            cent[g] = (mx, my)
            num += sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den += sum((x - mx) ** 2 for x in xs)
        if den == 0:
            raise SystemExit(f"{held}: the kept folds give no spread in distance; the common "
                             f"slope is not determined and the rule cannot be cross-fitted")
        b = num / den
        a = [cent[g][1] - b * cent[g][0] for g in keep]
        d = cyc(month, FOLD_MONTH[held])
        lin = sum(a) / 2 + b * d
        out[held] = {"slope": b, "intercept": sum(a) / 2, "distance": d,
                     "fitted_on": keep, "LIN": lin, "FLOOR": max(lin, MISMATCH[held])}
    return out


def stack_exponent(fold: str) -> float:
    """47.31's stacking form: total(k) = k**p * mean(the k months), p from the measured pair."""
    m2 = TOTAL2[fold] / ((ALONE["2025-11"][fold] + ALONE["2025-10"][fold]) / 2)
    return math.log2(m2)


def marginal(fold: str, alone_values: list[float], base_total: float) -> float:
    """The increment of the LAST month of a stack, over a stack total on the same scale."""
    k = len(alone_values)
    return (k ** stack_exponent(fold)) * (sum(alone_values) / k) - base_total


def efficiency(fold: str, k: int) -> float:
    """A stack's measured EFFICIENCY at k months: total(k) / the sum of the k alone values.

    The optimistic bracket.  Under it every month contributes its own worth at the stack's
    measured efficiency and a weak month DILUTES nothing, which is the opposite assumption to
    the power law's k**p * mean.
    """
    tot = {2: TOTAL2, 3: TOTAL3}[k][fold]
    return tot / sum(ALONE[m][fold] for m in FITTED[:k])


def marginal_additive(fold: str, alone_value: float) -> float:
    """The month's increment under the efficiency-scaled additive form, calibrated at k = 3."""
    return efficiency(fold, 3) * alone_value


def form_check() -> dict:
    """Both forms predicting the MEASURED three-month total from the two-month calibration.

    Neither form is fitted on total(3), so this is out of sample for both and it is the only
    evidence available for choosing which one the registration leads with.
    """
    out = {}
    for f in FOLD_MONTH:
        a3 = [ALONE[m][f] for m in FITTED]
        pl = (3 ** stack_exponent(f)) * (sum(a3) / 3)
        ad = efficiency(f, 2) * sum(a3)
        out[f] = {"measured_total_3": TOTAL3[f],
                  "power_law_from_the_two_month_calibration": pl,
                  "power_law_error": pl - TOTAL3[f],
                  "power_law_error_pct": 100 * (pl - TOTAL3[f]) / TOTAL3[f],
                  "efficiency_additive_from_the_two_month_calibration": ad,
                  "efficiency_additive_error": ad - TOTAL3[f],
                  "efficiency_additive_error_pct": 100 * (ad - TOTAL3[f]) / TOTAL3[f]}
    return out


def board(july: float, january: float, coef: dict) -> float:
    """47.30.2's board proxy: december carries no share."""
    return coef["cj_july"] * july + coef["ca_january"] * january


def main() -> None:
    coef = json.loads((EXP / "analysis/monthrule.json").read_text())["board_proxy"]
    ratio = abs(coef["cj_july"] / coef["ca_january"]
                - SHARES["july_loo"] / SHARES["january"])
    if ratio > 0.01:
        raise SystemExit(f"the board proxy's coefficient ratio is off SHARES by {ratio:.4f} -- "
                         f"it is not the scoring convention showing through")

    feb, aug = crossfit(2), crossfit(8)
    three = [ALONE[m] for m in FITTED]

    rec = {"what": __doc__.strip().splitlines()[0],
           "registered_before": "2025-02 and 2025-08 exist; neither month's data enters any "
                                "number below, and the rule is cross-fitted so no scored "
                                "fold's own reading enters its own prediction",
           "fitted_on": list(FITTED), "board_proxy": coef, "by_fold": {}}

    for f in FOLD_MONTH:
        p = stack_exponent(f)
        a3 = [v[f] for v in three]
        row = {"stack_exponent_p": p, "measured_total_3": TOTAL3[f],
               "one_month_alone": {}, "fourth_month_over_the_three_month_head": {},
               "fifth_month_over_the_feb_panel": {}, "feb_fourth_month_check": {}}
        for form in ("LIN", "FLOOR"):
            va, vf = aug[f][form], feb[f][form]
            row["one_month_alone"][form] = va
            # 4th month = Aug, over the MEASURED three-month total
            row["fourth_month_over_the_three_month_head"][form] = marginal(
                f, a3 + [va], TOTAL3[f])
            row["fourth_month_over_the_three_month_head"][f"{form}_additive"] = \
                marginal_additive(f, va)
            # Feb's own 4th-month figure, the check against 47.41.3
            row["feb_fourth_month_check"][form] = marginal(f, a3 + [vf], TOTAL3[f])
            # 5th month = Aug over the FEB panel; total(4) is PREDICTED, not measured
            total4_pred = (4 ** p) * (sum(a3 + [vf]) / 4)
            row["fifth_month_over_the_feb_panel"][form] = marginal(
                f, a3 + [vf, va], total4_pred)
            row["fifth_month_over_the_feb_panel"][f"{form}_additive"] = \
                marginal_additive(f, va)
            row["fifth_month_over_the_feb_panel"][f"total4_predicted_{form}"] = total4_pred
        # what a fourth month's ALONE value must reach for its increment to be positive at
        # all under the power law: k**p * mean(k) = total(k-1) solved for the new month
        row["break_even_alone_value_for_a_positive_fourth_month"] = (
            TOTAL3[f] * 4 / (4 ** p) - sum(a3))
        row["one_month_alone"]["LIN_47_37_per_fold_line"] = AUG_47_37[f][0]
        row["one_month_alone"]["FLOOR_47_37_per_fold_line"] = AUG_47_37[f][1]
        row["aug_distance"] = aug[f]["distance"]
        row["feb_distance"] = feb[f]["distance"]
        row["crossfit_aug"] = aug[f]
        row["crossfit_feb"] = feb[f]
        rec["by_fold"][f] = row

    # the conversion is the one 47.41.3 registered -- prove it rather than claim it
    bad = {f: (round(rec["by_fold"][f]["feb_fourth_month_check"]["LIN"], 1),
               round(rec["by_fold"][f]["feb_fourth_month_check"]["FLOOR"], 1))
           for f in FOLD_MONTH
           if (round(rec["by_fold"][f]["feb_fourth_month_check"]["LIN"], 1),
               round(rec["by_fold"][f]["feb_fourth_month_check"]["FLOOR"], 1))
           != FEB_REGISTERED[f]}
    if bad:
        raise SystemExit(f"this code does NOT reproduce 47.41.3's registered Feb figures: {bad} "
                         f"against {FEB_REGISTERED}. The conversion used for Aug must be the "
                         f"conversion used for Feb, or the two registrations are not comparable")
    rec["reproduces_47_41_3"] = FEB_REGISTERED
    rec["which_stacking_form"] = form_check()

    for key in ("one_month_alone", "fourth_month_over_the_three_month_head",
                "fifth_month_over_the_feb_panel"):
        forms = [k2 for k2 in rec["by_fold"]["july_loo"][key]
                 if k2.startswith(("LIN", "FLOOR")) and "47_37" not in k2]
        rec.setdefault("board", {})[key] = {
            form: board(rec["by_fold"]["july_loo"][key][form],
                        rec["by_fold"]["january"][key][form], coef)
            for form in forms}

    path = EXP / "analysis/monthpred.json"
    path.write_text(json.dumps(rec, indent=2, sort_keys=True))
    print("reproduces 47.41.3's Feb figures exactly")
    for key, label in (("one_month_alone", "2025-08 alone (the discriminator)"),
                       ("fourth_month_over_the_three_month_head", "Aug as the 4th month"),
                       ("fifth_month_over_the_feb_panel", "Aug as the 5th month, over Feb")):
        print(f"\n{label}")
        for f in FOLD_MONTH:
            v = rec["by_fold"][f][key]
            extra = (f"   additive {v['LIN_additive']:+9.2f}"
                     if "LIN_additive" in v else "")
            print(f"  {f:9s} d={rec['by_fold'][f]['aug_distance']}  "
                  f"LIN {v['LIN']:+9.2f}   FLOOR {v['FLOOR']:+9.2f}{extra}")
        b = rec["board"][key]
        extra = (f"   additive {b['LIN_additive']:+9.5f}"
                 if "LIN_additive" in b else "")
        print(f"  {'board':9s}      LIN {b['LIN']:+9.5f}   FLOOR {b['FLOOR']:+9.5f}{extra}")
    print("\nbreak-even ALONE value for a positive fourth month (power law)")
    for f in FOLD_MONTH:
        print(f"  {f:9s} {rec['by_fold'][f]['break_even_alone_value_for_a_positive_fourth_month']:+9.1f}"
              f"   against Aug cross-fit {rec['by_fold'][f]['one_month_alone']['LIN']:+9.2f}"
              f" and 47.37's {AUG_47_37[f][0]:+9.1f} / {AUG_47_37[f][1]:+9.1f}")
    print(f"\nwritten to {path.relative_to(EXP.parents[1])}")


if __name__ == "__main__":
    main()
