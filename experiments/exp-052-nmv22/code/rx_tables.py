"""exp-052-nmv22: RESULTS tables, rendered only from the analysis JSON (never retyped).

Writes `analysis/iter1-tables-r1.md`, the revision after Sol round 1 (B2: the no-NM table of
COMMON.md, the old column headed `nonm` regime).  The first render, `analysis/iter1-tables.md`,
stays as written at 1ca5029 (write-once; Sol round 2, R2-3).

    .venv/bin/python experiments/exp-052-nmv22/code/rx_tables.py > /dev/null   # writes the file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rx_common as C                                                 # noqa: E402

FOLDS = C.FOLDS
NAMES = ["q_aware", "q_blind", "q_sham", "global", "router", "oracle",
         "sham_global", "sham_router", "sham_oracle"]


def ci(x, k=3):
    return f"[{x[0]:+.{k}f}, {x[1]:+.{k}f}]"


def f3(x, k=3):
    return "–" if x is None else f"{x:.{k}f}"


def render(R: dict, N: dict | None, RD: dict | None, NMJ: dict | None = None,
           run: str = "iter1") -> str:
    L = []
    w = L.append
    w("## Experts on P (outer, scoring values)\n")
    w("| fold | P rows | v22 RMSE on P | q_aware | q_blind | q_sham | u_aware RMS | u_blind RMS "
      "| u_sham RMS | d22 RMS | d22 mean | share abs(d22) > 10 s | corr(d22, r − u_aware) | d_sham RMS |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for f in FOLDS:
        fr = R["folds"][f]
        e = fr["experts_rmse_on_P"]
        w(f"| {f} | {fr['population_rows']:,} | {e['v22']:.3f} | {e['q_aware']:.3f} | "
          f"{e['q_blind']:.3f} | {e['q_sham']:.3f} | {fr['u_rms']['aware']:.2f} | "
          f"{fr['u_rms']['blind']:.2f} | {fr['u_rms']['sham']:.2f} | {fr['d22']['rms']:.2f} | "
          f"{fr['d22']['mean']:+.2f} | {fr['d22']['share_abs_gt_10s']:.3f} | "
          f"{fr['d22']['corr_with_r_minus_u_aware']:+.4f} | {fr['d_sham']['rms']:.2f} |")
    w("\n## Every predictor against v22, per fold\n")
    w("| fold | predictor | changed | < 60 s | s²/row [1-day 95 %] | RMSE all | ≤ 3600 | > 3600 "
      "| `nonm` regime | clean | RMSE gain s |")
    w("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for f in FOLDS:
        p0 = R["folds"][f]["predictors"]["q_aware"]
        w(f"| {f} | v22 | 0 | – | 0 | {p0['rmse_v22']:.3f} | {p0['rmse_v22_le3600']:.3f} | "
          f"{p0['rmse_v22_gt3600']:.3f} | {p0['segments']['nonm']['rmse_v22']:.3f} | "
          f"{p0['rmse_v22_clean']:.3f} | 0 |")
        for n in NAMES:
            p = R["folds"][f]["predictors"][n]
            w(f"| {f} | {n} | {p['rows_changed']:,} | {p['rows_below_60s']} | "
              f"{p['mse_gain_s2_per_row']:+.3f} {ci(p['ci95_day'])} | {p['rmse_new']:.3f} | "
              f"{p['rmse_new_le3600']:.3f} | {p['rmse_new_gt3600']:.3f} | "
              f"{p['segments']['nonm']['rmse_new']:.3f} | {p['rmse_new_clean']:.3f} | "
              f"{p['rmse_gain_s']:+.4f} |")
    if NMJ is not None:
        w("\n## No-NM rows (COMMON.md: no NM join, every `_flt` column null; "
          "`analysis/nonm.json`)\n")
        w("The `nonm` regime column above keys on `FLIGHT_ID_mvt`; this population also holds every "
          "`noac` row and the few `nm_bulk` rows whose `_flt` columns are all null.\n")
        w("| run | fold | rows | of which nonm / noac / nm_bulk | in P | > 3600 | v22 | q_aware "
          "| q_blind | q_sham | global | router | oracle | changed by router |")
        w("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for rn in (run, f"{run}-null"):
            for f in FOLDS:
                fr = NMJ["folds"][f]
                x = fr["runs"][rn]
                g = fr["rows_by_regime"]
                w(f"| {rn} | {f} | {fr['no_nm_rows']:,} | {g['nonm']:,} / {g['noac']:,} / "
                  f"{g['nm_bulk']:,} | {fr['rows_in_P']} | {x['rows_gt3600']} | "
                  f"{x['v22']['rmse_s']:.3f} | "
                  + " | ".join(f"{x[n]['rmse_s']:.3f}" for n in
                               ("q_aware", "q_blind", "q_sham", "global", "router", "oracle"))
                  + f" | {x['router']['rows_changed']} |")
    w("\n## Board (july_loo + january at the ranking shares) against v22, and december\n")
    w("| predictor | all-row s [1-day] | 2-day | 3-day | clean s | december RMSE gain s |")
    w("|---|---|---|---|---:|---:|")
    for n in NAMES:
        b = R["board"][n]
        c = b["all_row_ci"]
        w(f"| {n} | {b['all_row_gain_s']:+.5f} {ci(c['ci_block_1d'], 5)} | "
          f"{ci(c['ci_block_2d'], 5)} | {ci(c['ci_block_3d'], 5)} | {b['clean_gain_s']:+.5f} | "
          f"{b['december_rmse_gain_s']:+.5f} |")
    w("\n## Paired differences: board and per fold\n")
    w("| B − A | board all-row s [1-day] | 2-day | 3-day | board clean s | july_loo s²/row [1-day] "
      "| december s²/row [1-day] | january s²/row [1-day] |")
    w("|---|---|---|---|---:|---|---|---|")
    for k, b in R["board_paired"].items():
        c = b["all_row_ci"]
        cells = []
        for f in FOLDS:
            x = R["folds"][f]["paired"][k]
            cells.append(f"{x['mse_s2_per_row']:+.3f} {ci(x['ci95_day'])}")
        w(f"| {k} | {b['all_row_gain_s']:+.5f} {ci(c['ci_block_1d'], 5)} | "
          f"{ci(c['ci_block_2d'], 5)} | {ci(c['ci_block_3d'], 5)} | {b['clean_gain_s']:+.5f} | "
          + " | ".join(cells) + " |")
    w("\nMulti-day intervals of router − global per fold (s²/row):\n")
    w("| fold | 1-day | 2-day | 3-day | RMSE s | clean RMSE s |")
    w("|---|---|---|---|---:|---:|")
    for f in FOLDS:
        x = R["folds"][f]["paired"]["router_minus_global"]
        m = x["multi_day"]
        w(f"| {f} | {ci(m['ci_block_1d'])} | {ci(m['ci_block_2d'])} | {ci(m['ci_block_3d'])} | "
          f"{x['rmse_s']:+.5f} | {x['rmse_clean_s']:+.5f} |")
    w("\n## Decomposition (telescopes exactly)\n")
    w("| step | july_loo s²/row | december s²/row | january s²/row | board s |")
    w("|---|---:|---:|---:|---:|")
    for k in ("v22_to_q_aware", "q_aware_to_global", "global_to_router", "v22_to_router"):
        w(f"| {k} | " + " | ".join(f"{R['folds'][f]['decomposition_s2_per_row'][k]:+.3f}"
                                    for f in FOLDS)
          + f" | {R['board']['decomposition_s'][k]:+.5f} |")
    w("\n## Oracle and the selection-noise references (points; never subtracted)\n")
    w("| run | contrast | fold | oracle s²/row over its base | 50 permutations: mean | 97.5 % "
      "| max | rank of real (of 51) |")
    w("|---|---|---|---:|---:|---:|---:|---:|")
    for tag, RR in (("real", R), ("NULL", N)):
        if RR is None:
            continue
        for pre, name in (("", "d22"), ("sham_", "d_sham")):
            for f in FOLDS:
                x = RR["folds"][f][f"{pre}oracle_perm_s2_per_row_over_base"]
                w(f"| {tag} | {name} | {f} | {x['real']:+.3f} | {x['mean']:+.3f} | "
                  f"{x['p97_5']:+.3f} | {x['max']:+.3f} | {x['rank_of_real_among_n_plus_1']} |")
            x = RR["perm"][f"{pre}oracle_over_base_board_s"]
            w(f"| {tag} | {name} | board s | {x['real']:+.5f} | {x['mean']:+.5f} | "
              f"{x['p97_5']:+.5f} | {x['max']:+.5f} | {x['rank_of_real_among_n_plus_1']} |")
    w("\n## Router against its 50 placebo relabellings (router − global; points)\n")
    w("| run | contrast | fold | real | placebo mean | 2.5 % | 97.5 % | max | rank (of 51) |")
    w("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for tag, RR in (("real", R), ("NULL", N)):
        if RR is None:
            continue
        for pre, name in (("", "d22"), ("sham_", "d_sham")):
            for f in FOLDS:
                x = RR["folds"][f][f"{pre}placebo_minus_global_s2_per_row"]
                w(f"| {tag} | {name} | {f} s²/row | {x['real']:+.3f} | {x['mean']:+.3f} | "
                  f"{x['p2_5']:+.3f} | {x['p97_5']:+.3f} | {x['max']:+.3f} | "
                  f"{x['rank_of_real_among_n_plus_1']} |")
            x = RR["perm"][f"{pre}placebo_minus_global_board_s"]
            w(f"| {tag} | {name} | board s | {x['real']:+.5f} | {x['mean']:+.5f} | "
              f"{x['p2_5']:+.5f} | {x['p97_5']:+.5f} | {x['max']:+.5f} | "
              f"{x['rank_of_real_among_n_plus_1']} |")
    w("\n## Coefficients per scoring block (development support in rows)\n")
    w("| run | contrast | fold | block | dev rows | c_g | cell 0 | cell 1 | cell 2 | cell 3 | "
      "dev d RMS |")
    w("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, RR in (("real", R), ("NULL", N)):
        if RR is None:
            continue
        for pre, name in (("", "d22"), ("sham_", "d_sham")):
            for f in FOLDS:
                for x in RR["folds"][f][f"{pre}fits"]:
                    cells = " | ".join(f"{c['coef']:.3f} (raw {c['raw']:+.3f}, n {c['dev_rows']:,})"
                                       for c in x["router"])
                    w(f"| {tag} | {name} | {f} | {x['block']} | {x['dev_rows']:,} | "
                      f"{x['global']:.3f} | {cells} | {x['d_rms_dev']:.2f} |")
    w("\n## SSE gain by cell and by airport (Ms², whole fold)\n")
    w("| fold | predictor | cell 0 | cell 1 | cell 2 | cell 3 | airports (sorted by gain) |")
    w("|---|---|---:|---:|---:|---:|---|")
    for f in FOLDS:
        for n in ("q_aware", "q_blind", "global", "router", "sham_global"):
            p = R["folds"][f]["predictors"][n]
            ap = sorted(p["by_airport"].items(), key=lambda kv: -kv[1])
            w(f"| {f} | {n} | " + " | ".join(f"{p['by_cell'][str(k)]['sse_gain_Ms2']:+.2f}"
                                             if str(k) in p['by_cell'] else
                                             f"{p['by_cell'][k]['sse_gain_Ms2']:+.2f}"
                                             for k in range(4))
              + " | " + ", ".join(f"{a} {v:+.2f}" for a, v in ap) + " |")
    w("\n## Each regime: RMSE, and rows changed (every other regime keeps v22's bytes)\n")
    w("| fold | segment | rows | v22 | q_aware | q_blind | q_sham | global | router | oracle | "
      "changed by router |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for f in FOLDS:
        P = R["folds"][f]["predictors"]
        for s, x in P["q_aware"]["segments"].items():
            vals = " | ".join(f3(P[n]["segments"][s]["rmse_new"])
                              for n in ("q_aware", "q_blind", "q_sham", "global", "router",
                                        "oracle"))
            w(f"| {f} | {s} | {x['rows']:,} | {f3(x['rmse_v22'])} | {vals} | "
              f"{P['router']['segments'][s]['rows_changed']:,} |")
    if "old_d" in R:
        o = R["old_d"]
        w("\n## The old Q08 contrast, reproduced through exp-050's code\n")
        w(f"Board router − global reproduced **{o['reproduced_board_router_minus_global']:+.14f}**"
          f" against the recorded {o['recorded']:+.14f} (tolerance {o['tolerance']:g}).\n")
        w("| quantity | board all-row s [1-day] | 2-day | 3-day | clean s |")
        w("|---|---|---|---|---:|")
        for k, lab in (("board_global", "old global over v22"),
                       ("board_router", "old router over v22"),
                       ("board_router_minus_global", "old router − global")):
            b = o[k]
            c = b["all_row_ci"]
            w(f"| {lab} | {b['all_row_gain_s']:+.5f} {ci(c['ci_block_1d'], 5)} | "
              f"{ci(c['ci_block_2d'], 5)} | {ci(c['ci_block_3d'], 5)} | {b['clean_gain_s']:+.5f} |")
    if N is not None:
        w("\n## The NULL run beside the real run (board, s)\n")
        w("| quantity | real | NULL |")
        w("|---|---:|---:|")
        for n in NAMES:
            w(f"| {n} over v22 | {R['board'][n]['all_row_gain_s']:+.5f} | "
              f"{N['board'][n]['all_row_gain_s']:+.5f} |")
        for k in ("router_minus_global", "global_minus_q_aware", "q_blind_minus_q_aware",
                  "q_aware_minus_q_sham", "oracle_minus_q_aware", "sham_router_minus_sham_global"):
            w(f"| {k} | {R['board_paired'][k]['all_row_gain_s']:+.5f} | "
              f"{N['board_paired'][k]['all_row_gain_s']:+.5f} |")
        w("\nNULL c_g by fold and block: " + "; ".join(
            f"{f} " + ", ".join(f"{x['global']:.3f}" for x in N["folds"][f]["fits"]) for f in FOLDS))
    if RD is not None:
        w("\n## The registered readings (SPEC 1.11 as amended by 1.15)\n")
        w("```json\n" + json.dumps(RD, indent=2, default=float) + "\n```")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = C.EXP / "analysis/iter1-tables-r1.md"
    C.NO.refuse_existing([OUT], args.overwrite)
    R = json.loads((C.EXP / "analysis/iter1.json").read_text())
    pn = C.EXP / "analysis/iter1-null.json"
    N = json.loads(pn.read_text()) if pn.exists() else None
    pr = C.EXP / "analysis/iter1-readings.json"
    RD = json.loads(pr.read_text()) if pr.exists() else None
    pm = C.EXP / "analysis/nonm.json"
    NMJ = json.loads(pm.read_text()) if pm.exists() else None
    text = render(R, N, RD, NMJ, "iter1")
    C.NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
