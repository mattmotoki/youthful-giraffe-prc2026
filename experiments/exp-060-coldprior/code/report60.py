"""exp-060-coldprior: generate every RESULTS table from the committed records.

Reads only `analysis/*.json` (never a scratch value, never a summary), so a table can be rebuilt
after a session ends.  Writes `analysis/report.md`.

    .venv/bin/python experiments/exp-060-coldprior/code/report60.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e60 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402


def f(x, n=4):
    return "-" if x is None else f"{x:+.{n}f}"


def band(x):
    return f"[{x[0]:+.4f}, {x[1]:+.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = E.ANALYSIS / "report.md"
    NO.refuse_existing([out], a.overwrite)
    oos = E.load_json(E.ANALYSIS / "oos.json")
    bnd = E.load_json(E.ANALYSIS / "bound.json")
    cold = {r: E.load_json(E.ANALYSIS / f"cold-{r}.json") for r in E.ORDER}
    fits = {r: E.load_json(E.ANALYSIS / f"fit60-{r}.json") for r in E.ORDER}
    bnds = {r: E.load_json(E.ANALYSIS / f"bound-{r}.json") for r in E.ORDER}
    L = []

    L.append("# exp-060-coldprior — generated tables (`code/report60.py`)\n")

    L.append("\n## 1. The cold factor tables\n")
    L.append("| rotation | fitting months A | arrivals | dates | supported stand / rwy / op | "
             "`true` sd stand / rwy / op, s | table sha256 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in E.ORDER:
        c = cold[r]
        sd = c["fits"]["true"]["exported_sd_s"]
        mm = ", ".join(m[5:] for m in c["fitting_months"])
        L.append(f"| {r} | {mm} | "
                 f"{c['fitting_arrivals']:,} | {c['arrival_dates']} | "
                 f"{c['supported_values']['stand']} / {c['supported_values']['rwy']} / "
                 f"{c['supported_values']['op']} | {sd['stand']:.1f} / {sd['rwy']:.1f} / "
                 f"{sd['op']:.1f} | `{c['table']['sha256'][:16]}` |")

    L.append("\n## 2. The registered picks, chosen on S and I only\n")
    L.append("| rotation | arm | level B lambda | c | beta range over airports | level A lambda | c |")
    L.append("|---|---|---|---|---|---|---|")
    for r in E.ORDER:
        for arm in ("zero", "true", "gmean", "donor", "shuffle"):
            b = fits[r]["arms"][arm]["levels"]["B"]
            aa = fits[r]["arms"][arm]["levels"]["A"]
            vals = [x for v in b["betas_by_airport"].values() for x in v]
            L.append(f"| {r} | {arm} | {b['lam']} | {b['c']} | {min(vals):.1f}–{max(vals):.1f} | "
                     f"{aa['lam']} | {aa['c']} |")

    L.append("\n## 3. The fixed out-of-sample gate (level B; SPEC 4.1)\n")
    L.append("| true over | jul point, s²/row | jan point, s²/row | board gain, s | board 1-day | "
             "board 2-day | board 3-day | band wholly below zero (2 or 3 day) | pass |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in E.GATE_CONTROLS:
        cc = oos["contrasts"][f"true_vs_{c}"]["B"]
        g = oos["gate"][c]
        L.append(f"| {c} | {f(cc['folds']['jul']['mse_gain_s2_per_row'], 3)} | "
                 f"{f(cc['folds']['jan']['mse_gain_s2_per_row'], 3)} | "
                 f"{f(cc['board']['board_gain_s'])} | {band(cc['board']['ci_block_1d'])} | "
                 f"{band(cc['board']['ci_block_2d'])} | {band(cc['board']['ci_block_3d'])} | "
                 f"{g['band_wholly_below_zero_2d_3d']} | **{g['pass']}** |")
    w = oos["gate"]["winner_adjusted"]
    L.append(f"\nWinner-adjusted (level B): point **{w['point_s']:+.4f} s**, one-day "
             f"{band(w['ci_block_1d'])}, pass **{w['pass']}**. "
             f"Gate verdict: **advance = {oos['gate']['advance']}**.\n")
    ww = oos["winner_adjusted"]["B"]
    L.append(f"Two- and three-day winner-adjusted bands: {band(ww['ci_block_2d'])} and "
             f"{band(ww['ci_block_3d'])}; the winning sham on the original rows is "
             f"`{ww['winning_sham_on_the_original_rows']}`.\n")

    L.append("\n## 4. dec, the calm control (zero board weight; cannot pass or fail the gate)\n")
    L.append("| true over | dec point, s²/row | dec RMSE gain, s | dec 1-day | dec 3-day |")
    L.append("|---|---|---|---|---|")
    for c in ("zero", "gmean", "donor", "shuffle"):
        d = oos["contrasts"][f"true_vs_{c}"]["B"]["folds"]["dec"]
        L.append(f"| {c} | {f(d['mse_gain_s2_per_row'], 3)} | {f(d['rmse_gain_s'])} | "
                 f"{band(d['ci_block_1d'])} | {band(d['ci_block_3d'])} |")

    L.append("\n## 5. Level A out of sample (descriptive; it enters no gate condition)\n")
    L.append("| true over | jul point, s²/row | jan point, s²/row | board gain, s | board 1-day |")
    L.append("|---|---|---|---|---|")
    for c in ("zero", "gmean", "donor", "shuffle", "sham01"):
        cc = oos["contrasts"][f"true_vs_{c}"]["A"]
        L.append(f"| {c} | {f(cc['folds']['jul']['mse_gain_s2_per_row'], 3)} | "
                 f"{f(cc['folds']['jan']['mse_gain_s2_per_row'], 3)} | "
                 f"{f(cc['board']['board_gain_s'])} | {band(cc['board']['ci_block_1d'])} |")
    wa = oos["winner_adjusted"]["A"]
    L.append(f"\nLevel-A winner-adjusted: point {wa['point_s']:+.4f} s, one-day "
             f"{band(wa['ci_block_1d'])}.\n")

    L.append("\n## 6. Every arm against the common base (level B)\n")
    L.append("| arm over base | jul, s²/row | jan, s²/row | dec, s²/row | board gain, s | board 1-day |")
    L.append("|---|---|---|---|---|---|")
    for arm in ("zero", "true", "gmean", "donor", "shuffle", "sham_mean"):
        cc = oos["contrasts"][f"{arm}_vs_base"]["B"]
        L.append(f"| {arm} | {f(cc['folds']['jul']['mse_gain_s2_per_row'], 3)} | "
                 f"{f(cc['folds']['jan']['mse_gain_s2_per_row'], 3)} | "
                 f"{f(cc['folds']['dec']['mse_gain_s2_per_row'], 3)} | "
                 f"{f(cc['board']['board_gain_s'])} | {band(cc['board']['ci_block_1d'])} |")

    L.append("\n## 7. COMMON.md serving tables (all V rows; level B)\n")
    for r in E.ORDER:
        L.append(f"\n**{r} (V = {E.ROT[r][0]})**\n")
        L.append("| arm | RMSE | RMSE ≤ 3600 s | RMSE > 3600 s | RMSE no-NM | RMSE clean | rows | "
                 "rows > 3600 | rows no-NM |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        t = oos["tables"][r]
        for name, key in [("base", "base")] + [(arm, f"{arm}|B") for arm in
                                               ("zero", "true", "const", "gmean", "donor", "shuffle")]:
            x = t[key]
            L.append(f"| {name} | {x['rmse']:.4f} | {x['rmse_le3600']:.4f} | "
                     f"{x['rmse_gt3600']:.2f} | {x['rmse_nonm']:.2f} | {x['rmse_clean']:.4f} | "
                     f"{x['rows']:,} | {x['rows_gt3600']:,} | {x['rows_nonm']:,} |")

    L.append("\n## 8. The exact cold bound on V (in-sample, descriptive; reference = the zero arm's "
             "lambda-oracle at the same level and kind)\n")
    L.append("| bound | level | true, board s | true rank of 11 | sham mean | sham max | "
             "true beats every sham | gmean | donor | shuffle |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, x in bnd["levels"].items():
        kind, lev = key.split("|")
        g = x["board_gain_s"]
        L.append(f"| {kind} | {lev} | **{f(g['true'])}** | {x['true_rank_of_11']} | "
                 f"{f(x['sham_mean_s'])} | {f(x['sham_max_s'])} | {x['true_beats_every_sham']} | "
                 f"{f(g['gmean'])} | {f(g['donor'])} | {f(g['shuffle'])} |")
    w2 = bnd["winner_adjusted_three_day_band"]
    L.append(f"\nWinner-adjusted three-day band on `{w2['level']}`: point **{w2['point_s']:+.4f} s**, "
             f"band **{band(w2['ci_block_3d'])}**, share of draws with a positive gain "
             f"{w2['gain_positive_share']:.3f}, winning sham on the original rows "
             f"`{w2['winning_sham_on_the_original_rows']}`. "
             f"Bound condition met: **{bnd['reading_inputs']['bound_condition_met']}**.\n")

    L.append("\n## 9. Exactness and isolation checks\n")
    L.append("| check | result |")
    L.append("|---|---|")
    cf = E.load_json(E.ANALYSIS / "check-fit.json")
    cp = E.load_json(E.ANALYSIS / "check-price.json")
    dry = E.load_json(E.ANALYSIS / "dry-coldrep.json")
    L.append(f"| cold zero correction = exp-056 iteration 1 zero, byte for byte | "
             f"{cf['all_byte_identical']} |")
    L.append(f"| cold zero V per-row loss = exp-056 iteration 1 zero V loss, byte for byte | "
             f"{cp['all_byte_identical']} |")
    L.append(f"| cold-table dry test (isolation, equal rows and support, const exactly zero, "
             f"sham multisets, determinism) | {dry['all_checks_pass']} |")
    for r in E.ORDER:
        ch = bnds[r]["checks"]["direct_recomputation_on_the_one_beta_grid"]
        L.append(f"| {r}: clipped grid SSE against the served prediction vector, "
                 f"{ch['points']} points | max relative {ch['max_relative_difference']:.2e} |")
    for r in E.ORDER:
        v = max(x["inner_sse_relative_difference"] for arm in E.ARMS
                for x in fits[r]["arms"][arm]["levels"].values())
        L.append(f"| {r}: selection SSE against a direct recomputation at every arm's chosen map | "
                 f"max relative {v:.2e} |")
    mx = {r: max(v["mixed"] for v in fits[r]["clip_counts"].values()) for r in E.ORDER}
    al = {r: max(v["always_clipped"] for v in fits[r]["clip_counts"].values()) for r in E.ORDER}
    sc = {r: fits[r]["rows"]["score"]["S"] + fits[r]["rows"]["score"]["I"] for r in E.ORDER}
    L.append(f"| selection: most mixed (partly floored) rows in one cell | {mx} of {sc} "
             f"scored rows (S and I together) |")
    L.append(f"| selection: most rows floored at every grid point in one cell | {al} |")
    bm = {r: bnd["rotations"][r]["clip_rows_max"] for r in E.ORDER}
    L.append(f"| bound: most mixed rows in one cell | {bm} |")

    out.write_text("\n".join(L) + "\n")
    print(f"written {out.relative_to(E.ROOT)} ({len(L)} lines)")


if __name__ == "__main__":
    main()
