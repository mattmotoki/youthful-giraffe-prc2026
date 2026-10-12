"""RESULTS section 47.63: what 47.57's tie rule would select if 2025-04/05/09 were built.

No fit, no data read: served distance is the calendar's and the anchors are K = 1 alone
reads already on the record, pulled from analysis/transferlaw.json rather than typed twice.

47.57's rule, stated operationally as it was APPLIED (the registered sentence is looser than
the set it produced, and the set is what shipped):

    head(served month) = the COMMON BASE {2025-11, 2025-10, 2025-03}
                       + every built month at the MINIMUM served distance to that served
                         month, ties included

The common base is a legacy of base set v24's three-month extension, chosen BEFORE the
seasonal rule existed; only the second term is rule-selected.  Keeping the two apart matters,
because `k` in 47.59/47.60 counts both regimes as one number.

    .venv/bin/python experiments/exp-046-surfacefit/code/tierule.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
OUT = EXP / "analysis/tierule.json"

SERVED = {"2026-01": 1, "2026-07": 7}
FOLD_MONTH = {"july_loo": 7, "december": 12, "january": 1}
FLOOR = {"july_loo": 83.22, "december": 43.70, "january": 43.70}
COMMON_BASE = ["2025-11", "2025-10", "2025-03"]
BUILT = ["2025-11", "2025-10", "2025-03", "2025-02", "2025-08", "2025-06"]
HYPOTHETICAL = ["2025-04", "2025-05", "2025-09"]


def cyc(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)


def head_sets(built: list[str]) -> dict:
    """The rule, applied.  Candidates are the built months outside the common base."""
    out = {}
    for sm, m in SERVED.items():
        cand = {x: cyc(int(x[-2:]), m) for x in built if x not in COMMON_BASE}
        lo = min(cand.values())
        out[sm] = {"minimum_served_distance": lo,
                   "rule_selected": sorted(x for x, d in cand.items() if d == lo),
                   "candidates": dict(sorted(cand.items())),
                   "head": COMMON_BASE + sorted(x for x, d in cand.items() if d == lo)}
    return out


def main() -> None:
    tl = json.loads((EXP / "analysis/transferlaw.json").read_text())
    measured = {(r["month"], r["fold"]): r for r in tl["alone_table"]}

    now = head_sets(BUILT)
    later = head_sets(BUILT + HYPOTHETICAL)
    changed = {sm: now[sm]["head"] != later[sm]["head"] for sm in SERVED}

    # Anchored predictions: a MEASURED cell at the SAME cyclic distance on the SAME fold, per
    # 47.60.3.  Where no cell sits at that distance the honest answer is a BRACKET between the
    # nearest measured distances either side, not a point off a fitted curve.
    pred = {}
    for mth in HYPOTHETICAL:
        pred[mth] = {}
        for f, fm in FOLD_MONTH.items():
            d = cyc(int(mth[-2:]), fm)
            same = [r for (mm, ff), r in measured.items() if ff == f and r["cyclic_d"] == d]
            below = sorted([r for (mm, ff), r in measured.items()
                            if ff == f and r["cyclic_d"] < d], key=lambda r: -r["cyclic_d"])
            above = sorted([r for (mm, ff), r in measured.items()
                            if ff == f and r["cyclic_d"] > d], key=lambda r: r["cyclic_d"])
            e = {"cyclic_d": d, "time_d": abs(int(mth[-2:]) - fm)}
            if same:
                a = max(same, key=lambda r: r["gain_s2"])
                lo = min(r["gain_s2"] for r in same)
                hi = max(r["gain_s2"] for r in same)
                e.update({"kind": "anchored",
                          "anchors": sorted(r["month"] for r in same),
                          "point_s2": (lo + hi) / 2,
                          "band_s2": [lo - FLOOR[f], hi + FLOOR[f]]})
            else:
                e.update({"kind": "bracket",
                          "between": {"nearer": below[0]["month"] if below else None,
                                      "farther": above[0]["month"] if above else None},
                          "band_s2": [above[0]["gain_s2"] if above else 0.0,
                                      below[0]["gain_s2"] if below else None]})
            pred[mth][f] = e

    # Would any hypothetical month give a MATCHED-CYCLIC pair -- 47.59.3's discriminator -- on
    # a fold where one is already measured or where two hypotheticals meet?
    pairs = []
    allm = BUILT + HYPOTHETICAL
    for f, fm in FOLD_MONTH.items():
        for a, b in combinations(allm, 2):
            da, db = cyc(int(a[-2:]), fm), cyc(int(b[-2:]), fm)
            ta, tb = abs(int(a[-2:]) - fm), abs(int(b[-2:]) - fm)
            if da == db and ta != tb and (a in HYPOTHETICAL or b in HYPOTHETICAL):
                pairs.append({"fold": f, "cyclic_d": da, "months": [a, b],
                              "time_d": [ta, tb], "time_gap": abs(ta - tb),
                              "both_hypothetical": a in HYPOTHETICAL and b in HYPOTHETICAL})

    out = {"what": __doc__.strip().splitlines()[0],
           "rule_as_applied": ("common base {2025-11, 2025-10, 2025-03} + every built month "
                               "at the MINIMUM served distance, ties included"),
           "common_base_is_legacy": ("base set v24's three-month extension, chosen before the "
                                     "seasonal rule existed; only the second term is "
                                     "rule-selected"),
           "heads_today": now, "heads_if_04_05_09_built": later,
           "any_head_changes": changed,
           "served_distances_of_the_hypothetical_months": {
               m: {sm: cyc(int(m[-2:]), v) for sm, v in SERVED.items()} for m in HYPOTHETICAL},
           "anchored_alone_predictions": pred,
           "matched_cyclic_pairs_they_would_create": pairs}
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))

    print(f"written {OUT.relative_to(EXP.parents[1])}\n")
    print("served distance of the three unbuilt months:")
    for m in HYPOTHETICAL:
        print(f"   {m}   d(2026-01) {cyc(int(m[-2:]),1)}   d(2026-07) {cyc(int(m[-2:]),7)}")
    print("\nheads TODAY vs heads IF 2025-04/05/09 were built:")
    for sm in SERVED:
        print(f"   {sm}  min d {now[sm]['minimum_served_distance']} -> "
              f"{later[sm]['minimum_served_distance']}   CHANGES: {changed[sm]}")
        print(f"      today {now[sm]['head']}")
        print(f"      later {later[sm]['head']}")
    print("\nanchored alone predictions, registered before data:")
    for m in HYPOTHETICAL:
        for f in FOLD_MONTH:
            e = pred[m][f]
            b = e["band_s2"]
            if e["kind"] == "anchored":
                print(f"   {m} {f:10} cyc {e['cyclic_d']} time {e['time_d']:2}  "
                      f"anchor {'+'.join(e['anchors']):17} point {e['point_s2']:7.2f}  "
                      f"[{b[0]:.2f}, {b[1]:.2f}]")
            else:
                print(f"   {m} {f:10} cyc {e['cyclic_d']} time {e['time_d']:2}  "
                      f"BRACKET between {str(e['between']['farther']):9} and "
                      f"{str(e['between']['nearer']):9} -> "
                      f"({b[0]:.2f}, {b[1] if b[1] is not None else float('nan'):.2f})")
    print("\nmatched-cyclic pairs they would create (47.59.3's discriminator):")
    for p in pairs:
        print(f"   {p['fold']:10} cyc {p['cyclic_d']}  {p['months'][0]} (time {p['time_d'][0]}) "
              f"vs {p['months'][1]} (time {p['time_d'][1]})  gap {p['time_gap']}"
              + ("   BOTH unbuilt" if p["both_hypothetical"] else ""))


if __name__ == "__main__":
    main()
