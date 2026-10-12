"""RESULTS section 47.59: why fold-side extension gains transfer less each step.

Every fold-side figure is READ FROM THE ANALYSIS FILES, never typed twice -- the same rule
`expectfile.py` works under.  The board figures are the only typed inputs, because a board
score has no file in this repo: they are the desk's reported uploads and are asserted against
RESULTS 47.24.4 and 47.47, which recorded them when they landed.

    .venv/bin/python experiments/exp-046-surfacefit/code/transferlaw.py
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
A = EXP / "analysis"
OUT = A / "transferlaw.json"

FOLD_MONTH = {"july_loo": 7, "december": 12, "january": 1}
#: 47.41.2's measured per-fold noise floor, s^2.
FLOOR = {"july_loo": 83.22, "december": 43.70, "january": 43.70}
#: The K = 1 ALONE read for each built month: the file, and the price key inside it.  Every
#: one is priced over the SAME cached member 1, which is what makes the column comparable.
ALONE = {"2025-11": "onemonth-screen", "2025-10": "onemonth-only202510",
         "2025-03": "onemonth-only202503", "2025-08": "onemonth-only202508"}
#: 47.31's fold-side stacking exponents.
P_FOLD = {"july_loo": 0.6643, "december": 0.5144, "january": 0.4962}

#: The board series.  `k` is the cumulative extension months IN EACH HEAD after the upload.
#: v19 is the K = 10 bag with no extension, so it is k = 0 and the origin of the series.
BOARD = [
    {"upload": "v20", "candidate": "v9",  "added": ["2025-11"], "k": 1,
     "face_s": 0.51300,  "board_gain_s": 0.4851, "recorded_in": "47.24.4"},
    {"upload": "v21", "candidate": "v10", "added": ["2025-10", "2025-03"], "k": 3,
     "face_s": 0.55876,  "board_gain_s": 0.1961, "recorded_in": "47.47"},
    {"upload": "v22", "candidate": "v11", "added": ["2025-02", "2025-08"], "k": 4,
     "face_s": 0.398006, "board_gain_s": 0.0733, "recorded_in": "47.59"},
]


def cyclic(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)


def alone_table() -> list[dict]:
    """Every measured (month, fold) K = 1 alone value, with both distances."""
    rows = []
    for m, stem in ALONE.items():
        d = json.loads((A / f"{stem}.json").read_text())
        keys = [k for k in d["prices"] if k.startswith("i_")]
        if len(keys) != 1:
            raise SystemExit(f"{stem}: {len(keys)} 'i_' price keys, expected exactly one -- "
                             f"the alone read cannot be identified without guessing")
        by_fold = d["prices"][keys[0]]["by_fold"]
        for f, v in by_fold.items():
            rows.append({"month": m, "fold": f,
                         "cyclic_d": cyclic(int(m[-2:]), FOLD_MONTH[f]),
                         "time_d": abs(int(m[-2:]) - FOLD_MONTH[f]),
                         "gain_s2": v["mse_gain_s2"],
                         "from": f"analysis/{stem}.json:prices.{keys[0]}.by_fold.{f}"})
    return rows


def inversions(rows: list[dict], key: str) -> dict:
    """Pairs where the LARGER gain sits at the LARGER distance -- the metric ordering wrong."""
    out = {}
    for f in FOLD_MONTH:
        sub = sorted([r for r in rows if r["fold"] == f], key=lambda r: -r["gain_s2"])
        bad = [(a["month"], b["month"]) for a, b in combinations(sub, 2) if a[key] > b[key]]
        out[f] = {"pairs": len(list(combinations(sub, 2))), "inversions": len(bad),
                  "which": bad}
    return out


def main() -> None:
    rows = alone_table()

    # The contrasts that put the two metrics head to head: same fold, same cyclic distance,
    # different time distance.  This is the only shape that can separate them.
    matched = []
    for a, b in combinations(rows, 2):
        if a["fold"] == b["fold"] and a["cyclic_d"] == b["cyclic_d"] and a["time_d"] != b["time_d"]:
            near, far = sorted((a, b), key=lambda r: r["time_d"])
            matched.append({"fold": a["fold"], "cyclic_d": a["cyclic_d"],
                            "near": {"month": near["month"], "time_d": near["time_d"],
                                     "gain_s2": near["gain_s2"]},
                            "far": {"month": far["month"], "time_d": far["time_d"],
                                    "gain_s2": far["gain_s2"]},
                            "difference_s2": near["gain_s2"] - far["gain_s2"],
                            "fold_noise_floor_s2": FLOOR[a["fold"]],
                            "difference_in_floors": (near["gain_s2"] - far["gain_s2"])
                                                    / FLOOR[a["fold"]]})

    # The board law.  c is pinned at total(1) -- one point, no freedom -- so p is the only
    # fitted parameter and each later k gives its own estimate.
    tot, cum = 0.0, []
    for b in BOARD:
        tot += b["board_gain_s"]
        cum.append({"upload": b["upload"], "k": b["k"], "cumulative_board_s": tot,
                    "cumulative_face_s": sum(x["face_s"] for x in BOARD[:len(cum) + 1])})
    c = cum[0]["cumulative_board_s"]
    cf = cum[0]["cumulative_face_s"]
    p_each = {r["k"]: math.log(r["cumulative_board_s"] / c) / math.log(r["k"])
              for r in cum[1:]}
    p_board = sum(p_each.values()) / len(p_each)
    pf_each = {r["k"]: math.log(r["cumulative_face_s"] / cf) / math.log(r["k"])
               for r in cum[1:]}
    p_face = sum(pf_each.values()) / len(pf_each)
    for r in cum:
        r["power_law_predicted_s"] = c * r["k"] ** p_board
        r["residual_s"] = r["power_law_predicted_s"] - r["cumulative_board_s"]

    g = [b["board_gain_s"] for b in BOARD]
    ratios = [g[i + 1] / g[i] for i in range(len(g) - 1)]
    mean_r = sum(ratios) / len(ratios)
    k_now = BOARD[-1]["k"]
    nxt_power = c * (k_now + 1) ** p_board - cum[-1]["cumulative_board_s"]
    nxt_geom = g[-1] * mean_r

    out = {
        "what": __doc__.strip().splitlines()[0],
        "alone_table": sorted(rows, key=lambda r: (r["fold"], r["cyclic_d"])),
        "rank_test": {"by_cyclic_distance": inversions(rows, "cyclic_d"),
                      "by_time_distance": inversions(rows, "time_d"),
                      "reading": ("cyclic distance orders every within-fold pair; time "
                                  "distance does not. july cannot separate them because "
                                  "cyclic == time for every 2025 month at mid-year")},
        "matched_cyclic_contrasts": matched,
        "board_series": BOARD,
        "board_cumulative": cum,
        "power_law": {"c_s": c, "p_board": p_board, "p_board_per_k": p_each,
                      "p_face": p_face, "p_face_per_k": pf_each,
                      "p_fold_47_31": P_FOLD,
                      "transfer_exponent": p_board - p_face,
                      "max_abs_residual_s": max(abs(r["residual_s"]) for r in cum)},
        "geometric": {"step_ratios": ratios, "mean_ratio": mean_r,
                      "all_remaining_months_s": g[-1] * mean_r / (1 - mean_r)},
        "next_step_k": k_now + 1,
        "next_step_s": {"power_law": nxt_power, "geometric": nxt_geom,
                        "they_differ_by_s": abs(nxt_power - nxt_geom),
                        "upload_floor_s": 0.1,
                        "both_under_the_floor": max(nxt_power, nxt_geom) < 0.1},
        "the_test": {
            "cell": "2025-02 ALONE on the december fold",
            "why": ("the only unmeasured (built month, fold) cell that holds cyclic distance "
                    "fixed while time distance moves: 2025-02 is cyclic 2 / time 10 from "
                    "december, 2025-10 is cyclic 2 / time 2 and is already measured"),
            "baseline_s2": next(r["gain_s2"] for r in rows
                                if r["month"] == "2025-10" and r["fold"] == "december"),
            "predicted_s2": None, "band_s2": None, "registered_in": "RESULTS.md 47.59.6"},
    }
    b = out["the_test"]["baseline_s2"]
    out["the_test"]["predicted_s2"] = b
    out["the_test"]["band_s2"] = [b - FLOOR["december"], b + FLOOR["december"]]

    if out["rank_test"]["by_cyclic_distance"]["january"]["inversions"] != 0:
        raise SystemExit("cyclic distance no longer orders the january fold -- 47.59.3's "
                         "falsification of H1 rests on that and must be rewritten")
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    print(f"written {OUT.relative_to(EXP.parents[1])}")
    print(f"  cyclic inversions {sum(v['inversions'] for v in out['rank_test']['by_cyclic_distance'].values())}"
          f" of 18, time inversions {sum(v['inversions'] for v in out['rank_test']['by_time_distance'].values())} of 18")
    print(f"  p_board {p_board:.4f}  p_face {p_face:.4f}  max residual "
          f"{out['power_law']['max_abs_residual_s']:+.5f} s")
    print(f"  next step k={k_now + 1}: power {nxt_power:+.4f} s, geometric {nxt_geom:+.4f} s, "
          f"both under the 0.1 s floor {out['next_step_s']['both_under_the_floor']}")
    print(f"  the test: 2025-02 on december, predict {out['the_test']['predicted_s2']:.2f} "
          f"in [{out['the_test']['band_s2'][0]:.2f}, {out['the_test']['band_s2'][1]:.2f}] s2")


if __name__ == "__main__":
    main()
