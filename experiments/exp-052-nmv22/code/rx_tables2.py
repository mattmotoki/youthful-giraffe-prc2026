"""exp-052-nmv22: iteration 2's RESULTS tables, rendered only from the analysis JSON.

Writes `analysis/iter2-tables-r1.md`, the revision after Sol round 1 (B2).  The first render,
`analysis/iter2-tables.md`, stays as written at 33d89ff (write-once; Sol round 2, R2-3).

The selected-capacity tables reuse iteration 1's renderer (`rx_tables.render`, unchanged); the
truncation checks, the chosen budgets and the fixed grid are rendered here.

    .venv/bin/python experiments/exp-052-nmv22/code/rx_tables2.py > /dev/null   # writes the file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rx_common as C                                                 # noqa: E402
import rx_tables as RT1                                               # noqa: E402

FOLDS = C.FOLDS
ci = RT1.ci


def render(TJ: dict, EX: dict, R: dict, N: dict, RD: dict, NMJ: dict | None = None) -> str:
    L = []
    w = L.append
    w("## Truncation checks (SPEC 2.2)\n")
    w("| run | fold | models checked | k = 400 equals iteration 1 bitwise | max abs diff s | seconds |")
    w("|---|---|---:|---|---:|---:|")
    for run in ("real", "null"):
        for f in FOLDS:
            x = TJ["runs"][run][f]
            w(f"| {run} | {f} | {x['models_checked']} | {x['check_a_k400_bitwise']} | "
              f"{x['check_a_max_abs_diff_s']:.3g} | {x['seconds']:.0f} |")
    w(f"\nExactness refit (b): {EX['fold']} {EX['type']} {EX['unit']} at {EX['rounds']} rounds, "
      f"{EX['fit_rows']:,} fitting rows, {EX['pred_rows']:,} predicted rows, {EX['seconds']} s, "
      f"launch load {EX['launch_load_1m']}: bitwise equal to the truncation "
      f"**{EX['bitwise_equal_to_truncation']}**, max abs diff {EX['max_abs_diff_s']:.3g} s, "
      f"passes **{EX['passes']}**.\n")
    w("## The budget chosen away from each scoring block (SPEC 2.3)\n")
    w("| run | fold | k* by block 0–4 | development loss at k* by block (s²) |")
    w("|---|---|---|---|")
    for tag, RR in (("real", R), ("NULL", N)):
        for f in FOLDS:
            s = RR["selection"][f]
            ks = s["k_by_block"]
            lo = [s["dev_loss_by_block"][str(b)][str(ks[b])] for b in range(C.K)]
            w(f"| {tag} | {f} | {', '.join(map(str, ks))} | "
              + ", ".join(f"{v:,.0f}" for v in lo) + " |")
    w("\nDevelopment loss by budget (s², mean over the five scoring blocks):\n")
    w("| run | fold | " + " | ".join(f"k={k}" for k in R["G"]) + " |")
    w("|---|---|" + "---:|" * len(R["G"]))
    for tag, RR in (("real", R), ("NULL", N)):
        for f in FOLDS:
            s = RR["selection"][f]["dev_loss_by_block"]
            w(f"| {tag} | {f} | " + " | ".join(
                f"{sum(s[str(b)][str(k)] for b in range(C.K)) / C.K:,.1f}" for k in R["G"]) + " |")
    w("\n# The selected capacity: every iteration-1 table at k*\n")
    w(RT1.render(R["selected"], N["selected"], None, NMJ, "iter2"))
    w("\n# The fixed grid (report only; no reading and no choice from it)\n")
    for tag, RR in (("real", R), ("NULL", N)):
        w(f"\n### {tag} run: board, s (july_loo + january at the shares)\n")
        w("| k | q_aware over v22 [1-day] | q_blind over v22 | global over v22 | router over v22 "
          "| S = router − global [1-day] | global − q_aware | q_aware − q_sham | oracle − q_aware | "
          "december q_aware / router RMSE gain |")
        w("|---:|---|---:|---:|---:|---|---:|---:|---:|---|")
        for k in RR["G"]:
            g = RR["grid"][str(k)]
            b, bp = g["board"], g["board_paired"]
            w(f"| {k} | {b['q_aware']['all_row_gain_s']:+.5f} "
              f"{ci(b['q_aware']['all_row_ci']['ci_block_1d'], 5)} | "
              f"{b['q_blind']['all_row_gain_s']:+.5f} | {b['global']['all_row_gain_s']:+.5f} | "
              f"{b['router']['all_row_gain_s']:+.5f} | "
              f"{bp['router_minus_global']['all_row_gain_s']:+.5f} "
              f"{ci(bp['router_minus_global']['all_row_ci']['ci_block_1d'], 5)} | "
              f"{bp['global_minus_q_aware']['all_row_gain_s']:+.5f} | "
              f"{bp['q_aware_minus_q_sham']['all_row_gain_s']:+.5f} | "
              f"{bp['oracle_minus_q_aware']['all_row_gain_s']:+.5f} | "
              f"{b['q_aware']['december_rmse_gain_s']:+.5f} / "
              f"{b['router']['december_rmse_gain_s']:+.5f} |")
        w(f"\n### {tag} run: per fold\n")
        w("| k | fold | u_aware RMS | d22 RMS | corr(d22, r − u_aware) | c_g by block | "
          "router − global s²/row | oracle rank (of 51) |")
        w("|---:|---|---:|---:|---:|---|---:|---:|")
        for k in RR["G"]:
            g = RR["grid"][str(k)]
            for f in FOLDS:
                fr = g["folds"][f]
                w(f"| {k} | {f} | {fr['u_rms']['aware']:.2f} | {fr['d22']['rms']:.2f} | "
                  f"{fr['d22']['corr_with_r_minus_u_aware']:+.4f} | "
                  + ", ".join(f"{x['global']:.2f}" for x in fr["fits"])
                  + f" | {fr['paired']['router_minus_global']['mse_s2_per_row']:+.3f} | "
                  f"{fr['oracle_perm_s2_per_row_over_base']['rank_of_real_among_n_plus_1']} |")
    w("\n## The registered readings (SPEC 2.5)\n")
    w("```json\n" + json.dumps(RD, indent=2, default=float) + "\n```")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = C.EXP / "analysis/iter2-tables-r1.md"
    C.NO.refuse_existing([OUT], args.overwrite)
    A = C.EXP / "analysis"
    TJ = json.loads((A / "trunc.json").read_text())
    EX = json.loads((A / "trunc-exactness.json").read_text())
    R = json.loads((A / "iter2.json").read_text())
    N = json.loads((A / "iter2-null.json").read_text())
    RD = json.loads((A / "iter2-readings.json").read_text())
    pm = A / "nonm.json"
    NMJ = json.loads(pm.read_text()) if pm.exists() else None
    text = render(TJ, EX, R, N, RD, NMJ)
    C.NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
