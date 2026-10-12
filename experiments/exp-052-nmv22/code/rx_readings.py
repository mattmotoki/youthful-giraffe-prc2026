"""exp-052-nmv22: iteration 1's registered readings, with SPEC 1.15's amended noise test.

Reads `analysis/iter1.json` (the real run) and `analysis/iter1-null.json` (the NULL run) and
applies SPEC 1.11 as amended by 1.15 (c)-(d):

  * reversed            : Q08's confirmation rule on S (unchanged): the pooled 1-day interval
                          of board router - global clear above zero, no fold negative;
  * above_noise[fold]   : the real oracle beats all 50 d22-only permutations on the fold AND
                          the NULL run's oracle does NOT beat all 50 of its own there;
  * useful_raw_ceiling  : board oracle over q_aware >= 0.10 s AND above_noise on every fold;
  * route               : reversed -> witness beside the four cells; useful ceiling -> witness;
                          otherwise -> the smallest matched residual-target or capacity control;
  * guard (1.15 d)      : S beside the SHAM's and the NULL's router - global.

    .venv/bin/python experiments/exp-052-nmv22/code/rx_readings.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rx_common as C                                                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = C.EXP / "analysis/iter1-readings.json"
    C.NO.refuse_existing([OUT], args.overwrite)
    src = {k: C.EXP / f"analysis/{k}.json" for k in ("iter1", "iter1-null")}
    real, null = (json.loads(src[k].read_text()) for k in ("iter1", "iter1-null"))
    if real.get("null") or not null.get("null") or real.get("dry_run") or null.get("dry_run"):
        raise SystemExit("the two inputs are not the real run and the NULL run")
    S = real["board_paired"]["router_minus_global"]
    neg = [f for f in C.FOLDS
           if real["folds"][f]["paired"]["router_minus_global"]["mse_s2_per_row"] < 0]
    orc = real["board_paired"]["oracle_minus_q_aware"]["all_row_gain_s"]
    rank = {f: {"real_beats_all_50": real["folds"][f]["oracle_perm_s2_per_row_over_base"]
                ["real_exceeds_all"],
                "null_beats_all_50": null["folds"][f]["oracle_perm_s2_per_row_over_base"]
                ["real_exceeds_all"]} for f in C.FOLDS}
    above = {f: bool(v["real_beats_all_50"] and not v["null_beats_all_50"])
             for f, v in rank.items()}
    rev = bool(S["all_row_ci"]["ci_block_1d"][0] > 0 and not neg)
    useful = bool(orc >= 0.10 and all(above.values()))
    rec = {"inputs": {k: {"path": str(p.relative_to(C.ROOT)), "sha256": C.sha_file(p)}
                      for k, p in src.items()},
           "Q2": {"S_board_router_minus_global_s": S["all_row_gain_s"],
                  "S_ci": S["all_row_ci"], "S_clean_s": S["clean_gain_s"],
                  "folds_with_negative_point": neg, "reversed": rev},
           "guard_1_15_d": {
               "S_real": S["all_row_gain_s"],
               "S_sham_real_run": real["board_paired"]["sham_router_minus_sham_global"]
               ["all_row_gain_s"],
               "S_null_run": null["board_paired"]["router_minus_global"]["all_row_gain_s"],
               "S_sham_null_run": null["board_paired"]["sham_router_minus_sham_global"]
               ["all_row_gain_s"]},
           "Q1": {"ceiling_board_oracle_over_q_aware_s": orc,
                  "null_run_oracle_over_q_aware_s":
                      null["board_paired"]["oracle_minus_q_aware"]["all_row_gain_s"],
                  "rank_tests": rank},
           "above_noise_by_fold_amended": above,
           "useful_raw_ceiling": useful}
    rec["route"] = ("reversed: witness router beside the four cells" if rev else
                    "useful raw ceiling, four cells fail: rare physical reliability witness"
                    if useful else
                    "no alignment evidence: smallest matched residual-target or capacity control")
    C.NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(json.dumps(rec, indent=2, default=float))


if __name__ == "__main__":
    main()
