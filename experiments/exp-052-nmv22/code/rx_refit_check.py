"""exp-052-nmv22: the re-fit in the repaired order against the scored artifacts (Sol round 1, B1).

`rx_fit.py --out-dir work/refit` re-ran the 270 fits with the label-free prediction-level proof
before any label read.  For each fold and run (real, NULL) this script requires:

1. the label-free proof passed and finished before the first fit that reads a label launched,
   and, where the re-fit records it (the second re-fit, after Sol round 2), before the labels
   were first read (unix times);
2. every one of the 45 model files is byte-identical to the scored one (sha256 of both files on
   disk, and the scored experts record's sha), with the same fitting and predicted rows;
3. the residual parquet is byte-identical to the scored one;
4. the proof record is identical to the scored one except the finish times and the new
   label-free section; the experts record's population, blocks, fitting rows, SHAM and NULL
   permutations and pinned inputs are identical.

It writes `analysis/<dir>-check.json` (`refit-check.json` for `work/refit`), which carries the
six label-free proof sections.

    .venv/bin/python experiments/exp-052-nmv22/code/rx_refit_check.py                        # work/refit
    .venv/bin/python experiments/exp-052-nmv22/code/rx_refit_check.py --refit-dir work/refit2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rx_common as C                                                 # noqa: E402

NO = C.NO
TIMES = {"feature_proof_finished_utc", "prediction_proof_finished_utc",
         "label_free_prediction_proof", "label_free_prediction_proof_finished_utc",
         "label_free_prediction_proof_finished_unix"}
SAME = ["fold", "dry_run", "null", "population_rows", "population_ids_sha256", "block_rows",
        "fitting_rows_all_blocks", "fitting_rows_excluded", "params", "rounds", "seed",
        "threads", "sham", "pinned_inputs", "resid_sha256"]
FIT_SAME = ["type", "unit", "left_out_blocks", "fit_rows", "pred_rows", "model_sha256"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--refit-dir", default="work/refit", help="relative to the experiment")
    args = ap.parse_args()
    REFIT = C.EXP / args.refit_dir
    OUT = C.EXP / "analysis" / f"{REFIT.name}-check.json"
    NO.refuse_existing([OUT], args.overwrite)
    rec = {"what": "re-fit in the repaired order (label-free proof first) against the scored "
                   "artifacts", "code_sha256": C.sha_file(Path(__file__)),
           "refit_dir": str(REFIT.relative_to(C.ROOT)),
           "rx_fit_sha256": C.sha_file(C.EXP / "code/rx_fit.py"), "runs": {}}
    ok_all = True
    for nl in (False, True):
        tag = "null" if nl else "real"
        for f in C.FOLDS:
            ej0 = C.experts_json(f, None, nl)
            ej1 = C.experts_json(f, REFIT, nl)
            E0, E1 = json.loads(ej0.read_text()), json.loads(ej1.read_text())
            pj0 = C.EXP / "analysis" / f"{C.tagged('proof', nl)}-{f}.json"
            pj1 = REFIT / f"{C.tagged('proof', nl)}-{f}.json"
            P0, P1 = json.loads(pj0.read_text()), json.loads(pj1.read_text())
            lf = P1["label_free_prediction_proof"]
            first_launch = min(x["launch_utc"] for x in E1["fits"])
            lab = E1.get("labels_first_read")
            order_ok = bool(lf["passes"]
                            and P1["label_free_prediction_proof_finished_utc"] <= first_launch
                            and (lab is None or P1["label_free_prediction_proof_finished_unix"]
                                 < lab["unix"]))
            models = []
            for a, b in zip(E0["fits"], E1["fits"], strict=True):
                s0, s1 = C.sha_file(C.ROOT / a["model"]), C.sha_file(C.ROOT / b["model"])
                models.append(all(a[k] == b[k] for k in FIT_SAME) and s0 == s1 == a["model_sha256"])
            r0, r1 = C.resid_path(f, None, nl), C.resid_path(f, REFIT, nl)
            resid_ok = C.sha_file(r0) == C.sha_file(r1) == E0["resid_sha256"]
            proof_same = {k: P0[k] == P1[k] for k in P0 if k not in TIMES}
            rec_same = {k: E0.get(k) == E1.get(k) for k in SAME}
            new_keys = {"label_free_prediction_proof", "label_free_prediction_proof_finished_utc",
                        "label_free_prediction_proof_finished_unix"} & set(P1)
            ok = (order_ok and all(models) and len(models) == 45 and resid_ok
                  and all(proof_same.values()) and all(rec_same.values())
                  and set(P1) - set(P0) == new_keys)
            ok_all &= ok
            rec["runs"][f"{tag}-{f}"] = {
                "passes": bool(ok),
                "label_free_proof_passes": bool(lf["passes"]),
                "label_free_proof_finished_utc": P1["label_free_prediction_proof_finished_utc"],
                "first_label_reading_fit_launch_utc": first_launch,
                "labels_first_read": lab,
                "label_free_proof_finished_unix": P1.get("label_free_prediction_proof_finished_unix"),
                "order_ok": order_ok,
                "models_compared": len(models), "models_byte_identical": int(sum(models)),
                "resid_byte_identical": bool(resid_ok), "resid_sha256": E0["resid_sha256"],
                "proof_fields_identical": proof_same, "record_fields_identical": rec_same,
                "refit_proof_sha256": C.sha_file(pj1), "refit_experts_sha256": C.sha_file(ej1),
                "refit_seconds": E1["seconds"], "refit_finished_utc": E1["finished_utc"],
                "refit_launch_loads": [x["launch_load_1m"] for x in E1["fits"]],
                "label_free_prediction_proof": lf}
            print(f"{tag:4s} {f:8s}: label-free proof {lf['passes']} at "
                  f"{P1['label_free_prediction_proof_finished_utc']} (first fit {first_launch}); "
                  f"models {sum(models)}/45 identical; resid {resid_ok}; proof "
                  f"{all(proof_same.values())}; record {all(rec_same.values())}", flush=True)
    rec["passes"] = bool(ok_all)
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"all runs pass: {ok_all}")
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
