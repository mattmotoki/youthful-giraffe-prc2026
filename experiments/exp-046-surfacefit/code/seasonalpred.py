"""Section 47.49 (d): the SEASONAL BAG's registration, before 2025-08 exists.

Fits nothing and reads no month data.  Arithmetic on readings already in the record:

  * january and december  -- section 47.48's MEASURED K=1 marginal reads, scaled to K=10 by
    the churn argument (sections 47.20/47.33) at its only measured calibration: the K=1 -> K=10
    ratio on the panel immediately before this one (47.32 against 47.40, the same quantity
    priced both ways);
  * july -- section 47.45.3's cross-fitted Aug fourth-month prediction, scaled the same way,
    with its ROW correction stated beside it because 2025-08's writable count does not exist
    yet (47.46.4 registered that correction's form on Feb before Feb's read);
  * the board by served share (47.30.2's proxy, december weight 0);
  * section 47.44's member-disagreement falsifier at the seasonal row counts.

    .venv/bin/python experiments/exp-046-surfacefit/code/seasonalpred.py
"""
from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

FOLDS = ("july_loo", "december", "january")
FOLD_ROWS = {"july_loo": 172783, "december": 159231, "january": 191536}

#: 47.32's three-month K=1 screen and 47.40's K=10 bag -- the SAME quantity (three months
#: minus one month) priced at K=1 and at K=10, which is the only measured calibration of what
#: bagging does to a marginal read on this lane
K1_PANEL = {"july_loo": 372.62, "december": 371.52, "january": 249.46}
K10_PANEL = {"july_loo": 387.98, "december": 350.75, "january": 264.17}

#: 47.48's measured K=1 marginal reads for the Feb set
FEB_READ = {"december": 69.92, "january": 212.38}
FEB_CI = {"december": (0.97, 156.99), "january": (158.58, 266.14)}
#: 47.45.3's cross-fitted Aug fourth-month prediction (LIN = FLOOR on every fold)
AUG_PRED_JULY = 478.74
#: the mean writable count of the three months the rule was fitted on (47.46.4)
MEAN_MONTH_ROWS = (99709 + 103344 + 86984) / 3

EXT_JAN = 358159                     # Nov + Oct + Mar + Feb
EXT_JUL_BASE = 290037                # Nov + Oct + Mar, before 2025-08's rows

#: 47.44.1's fitted disagreement law, and the three-month values it predicts from
DIS = {"july_loo": {"c": 12.2202, "p": 0.1477, "d3": 10.4831},
       "december": {"c": 15.1027, "p": 0.2525, "d3": 11.7401},
       "january": {"c": 14.1448, "p": 0.2633, "d3": 10.9434}}
DIS_RESID_S = 0.370                  # dof-corrected residual scale
NOMINAL_AUG_ROWS = 100000            # a 31-day month; the real count replaces it


def main() -> None:
    coef = json.loads((EXP / "analysis/monthrule.json").read_text())["board_proxy"]
    ratio = {f: K10_PANEL[f] / K1_PANEL[f] for f in FOLDS}

    pred = {"december": FEB_READ["december"] * ratio["december"],
            "january": FEB_READ["january"] * ratio["january"],
            "july_loo": AUG_PRED_JULY * ratio["july_loo"]}
    ci = {f: tuple(v * ratio[f] for v in FEB_CI[f]) for f in FEB_CI}

    board = coef["cj_july"] * pred["july_loo"] + coef["ca_january"] * pred["january"]

    def july_row_corrected(rows: int) -> float:
        return pred["july_loo"] * rows / MEAN_MONTH_ROWS

    def disagreement(rows_aug: int) -> dict:
        import math
        mult = {"january": (FOLD_ROWS["january"] + EXT_JAN) / FOLD_ROWS["january"],
                "december": (FOLD_ROWS["december"] + EXT_JAN) / FOLD_ROWS["december"],
                "july_loo": (FOLD_ROWS["july_loo"] + EXT_JUL_BASE + rows_aug)
                            / FOLD_ROWS["july_loo"]}
        d = {f: DIS[f]["c"] * mult[f] ** -DIS[f]["p"] for f in FOLDS}
        L = sum(math.log(d[f] / DIS[f]["d3"]) for f in FOLDS) / 3
        se = (DIS_RESID_S / 12) / (3 ** 0.5)
        return {"fitting_multiple": {f: round(mult[f], 4) for f in FOLDS},
                "predicted_disagreement_s": {f: round(d[f], 3) for f in FOLDS},
                "three_month_disagreement_s": {f: DIS[f]["d3"] for f in FOLDS},
                "pooled_L": L, "pooled_se": se,
                "F1_mechanism_falsified_if": "L >= 0",
                "F2_law_falsified_if_outside": [L - 2 * se, L + 2 * se],
                "aug_rows_used": rows_aug}

    rec = {
        "what": __doc__.strip().splitlines()[0],
        "registered_before": "2025-08 exists and before any seasonal fit",
        "the_arm": ("the K = 10 SEASONAL bag as an in-fold increment over v10's head "
                    "(d-threemonth-bag10-255-free.npz), each leave-one-out fit carrying its "
                    "own fold's month set: january and december {Nov,Oct,Mar,Feb}, july "
                    "{Nov,Oct,Mar,Aug}"),
        "gate": ("rule (a), section 47.46.2: every fold's increment interval clear, premium "
                 "0. The month per served month is the nearest BUILT month by served "
                 "distance and no fold reading picks it"),
        "k1_to_k10_scaling": {
            "what": ("the only measured calibration: 47.32's K=1 three-month screen against "
                     "47.40's K=10 bag, the same quantity priced both ways"),
            "K1": K1_PANEL, "K10": K10_PANEL,
            "ratio": {f: round(ratio[f], 4) for f in FOLDS}},
        "per_fold": {
            "january": {"source": "47.48 measured K=1 read +212.38, scaled",
                        "prediction": pred["january"], "scaled_interval": list(ci["january"])},
            "december": {"source": "47.48 measured K=1 read +69.92, scaled",
                         "prediction": pred["december"],
                         "scaled_interval": list(ci["december"]),
                         "watch": ("the scaled lower bound is barely above zero; this is the "
                                   "fold that decides the gate")},
            "july_loo": {"source": "47.45.3 cross-fitted Aug 4th-month +478.74, scaled",
                         "prediction": pred["july_loo"],
                         "row_correction": {
                             "form": "prediction * (2025-08 writable rows / 96679)",
                             "at_a_nominal_100k": july_row_corrected(NOMINAL_AUG_ROWS),
                             "why": ("47.46.4 registered this correction's form on Feb "
                                     "BEFORE Feb's read, and Feb landed between the "
                                     "corrected and the uncorrected figure. The uncorrected "
                                     "number is the registered point; the corrected one is "
                                     "the companion, evaluated at the delivered count")}}},
        "board_by_served_share": {"prediction": board, "coefficients": coef,
                                  "december_weight": 0.0},
        "disagreement_falsifier_47_44": disagreement(NOMINAL_AUG_ROWS),
    }
    path = EXP / "analysis/seasonalpred.json"
    path.write_text(json.dumps(rec, indent=2, sort_keys=True))
    print("K=1 -> K=10 ratio: " + ", ".join(f"{f} {ratio[f]:.4f}" for f in FOLDS))
    print("\nthe seasonal bag over v10's head, registered before the fit:")
    for f in FOLDS:
        v = rec["per_fold"][f]
        extra = (f"   [{v['scaled_interval'][0]:+.2f}, {v['scaled_interval'][1]:+.2f}]"
                 if "scaled_interval" in v else
                 f"   row-corrected at 100k {v['row_correction']['at_a_nominal_100k']:+.2f}")
        print(f"  {f:9s} {v['prediction']:+9.2f}{extra}")
    print(f"  {'board':9s} {board:+9.5f}")
    d = rec["disagreement_falsifier_47_44"]
    print(f"\n47.44 at the seasonal row counts (Aug nominal {NOMINAL_AUG_ROWS:,}):")
    for f in FOLDS:
        print(f"  {f:9s} multiple {d['fitting_multiple'][f]:.3f}  "
              f"{d['three_month_disagreement_s'][f]:.3f} -> "
              f"{d['predicted_disagreement_s'][f]:.3f}")
    print(f"  pooled L {d['pooled_L']:+.4f}, se {d['pooled_se']:.4f}; "
          f"F1 falsified if L >= 0; F2 if outside "
          f"[{d['F2_law_falsified_if_outside'][0]:+.4f}, "
          f"{d['F2_law_falsified_if_outside'][1]:+.4f}]")
    print(f"\nwritten to {path.relative_to(EXP.parents[1])}")


if __name__ == "__main__":
    main()
