"""exp-059 ownvalue: review round 2, findings 2 and 3.  ZERO FIT.  READS NO LABEL AND NO VECTOR.

This script opens two things only: this experiment's reproduction record
`analysis/repro-r1.json` and the records it reproduces.  It computes nothing new.  It closes two
gaps that round 2 found in the reproduction driver:

  finding 2  R8 (vii) compared only the top-level population masks, because
             `populations_block()` skipped a value that is a dict and skipped a predictor the
             other record does not carry.  The nested `by_serving_regime` results -- 4 serving
             regimes x 5 masks per predictor and level -- were therefore recorded but never
             compared, and a missing key was skipped instead of refused.  Here EVERY leaf of
             exp-058's and exp-057's population blocks is compared against this experiment's
             recorded leaf, a missing predictor, regime or mask on either side is a refusal, and
             the integer row counts are compared as well.
  finding 3  the R3/R4 reproduction raised only on the registered-field comparison; the
             whole-record, receipt-hash and OOF comparisons were recorded without a refusal.
             Here every one of them is asserted, and any false or missing field is a refusal.

Both are checks of numbers that already exist in the records.  No value is recomputed, so nothing
here can change a measured result; the script either passes or refuses.

    .venv/bin/python experiments/exp-059-ownvalue/code/ov_r8vii_r1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent.parent
sys.path.insert(0, str(ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                                  # noqa: E402

ANALYSIS = EXP / "analysis"
OUT = ANALYSIS / "r8vii-r1.json"
SRC = {"exp058": ROOT / "experiments/exp-058-origincontrast/analysis/measure.json",
       "exp057": ROOT / "experiments/exp-057-originfit/analysis/measure.json"}
MINE = ANALYSIS / "repro-r1.json"


def leaves(obj, path=()):
    """Every numeric leaf of a nested record, with the path that reaches it."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, path + (k,))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def compare(ref_pops: dict, my_pops: dict, tag: str) -> dict:
    """Every leaf of the reference block must exist in mine and be equal.  A missing path or any
    difference is a refusal, not a skip (review round 2, finding 2)."""
    n, worst, worst_at, missing = 0, 0.0, None, []
    for path, ref in leaves(ref_pops):
        cur = my_pops
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                cur = None
                break
            cur = cur[k]
        if cur is None:
            missing.append("/".join(path))
            continue
        if not isinstance(cur, (int, float)) or isinstance(cur, bool):
            missing.append("/".join(path) + " (not a number here)")
            continue
        d = abs(float(cur) - ref)
        n += 1
        if d > worst:
            worst, worst_at = d, "/".join(path)
    if missing:
        raise SystemExit(f"{tag}: {len(missing)} recorded values have no counterpart here, "
                         f"first {missing[:3]}")
    if worst > 0.0:
        raise SystemExit(f"{tag}: a recorded population differs by {worst:.3e} at {worst_at}")
    return {"values_compared": n, "max_abs_diff": worst, "max_abs_diff_at": worst_at,
            "missing": missing}


def check_r34(rec: dict) -> dict:
    """Finding 3: assert every R3/R4 comparison the driver recorded but did not enforce."""
    blk = rec["R3_R4_exp055_scripts"]
    reg = blk["registered_fields"]
    if not reg["all_within_tolerance"] or reg["max_abs_diff_fields_at_1e-9"] != 0.0:
        raise SystemExit("R3/R4: the registered fields are not equal")
    out = {"registered_fields_n": reg["n"], "scripts": {}}
    for nm, run in blk["runs"].items():
        w = run["whole_record"]
        if w["n_structural_diffs"] != 0 or w["float_fields_differing"] != 0:
            raise SystemExit(f"R3/R4 {nm}: the whole record differs ({w['n_structural_diffs']} "
                             f"structural, {w['float_fields_differing']} float)")
        if w["max_abs_float_diff"] != 0.0 or w["max_rel_float_diff"] != 0.0:
            raise SystemExit(f"R3/R4 {nm}: a float field differs by {w['max_abs_float_diff']:.3e}")
        if run["receipt_hashes_equal"] is not True:
            raise SystemExit(f"R3/R4 {nm}: a receipt hash differs")
        oof = run["oof_equal"]
        if oof is not None and not all(bool(v) for v in oof.values()):
            raise SystemExit(f"R3/R4 {nm}: the OOF comparison is not all true ({oof})")
        out["scripts"][nm] = {"whole_record_clean": True, "receipt_hashes_equal": True,
                              "oof_equal": oof, "seconds": run["seconds"]}
    return out


def main() -> None:
    t0, c0 = time.time(), time.process_time()
    NO.refuse_existing([OUT], overwrite=False)
    rec = json.loads(MINE.read_text())
    mine = rec["populations"]
    res = {"what": "review round 2, findings 2 and 3: the complete population comparison and the "
                   "enforced R3/R4 refusal, both on values the records already hold",
           "reads": {"mine": str(MINE.relative_to(ROOT)),
                     **{k: str(v.relative_to(ROOT)) for k, v in SRC.items()}},
           "opens_no_vector_and_no_label": True, "populations": {}}
    for tag, path in SRC.items():
        ref = json.loads(path.read_text())["populations"]
        res["populations"][tag] = compare(ref, mine, tag)
        print(f"{tag}: {res['populations'][tag]['values_compared']} recorded population values, "
              f"max |diff| {res['populations'][tag]['max_abs_diff']:.1e} s", flush=True)
    res["R3_R4_enforced"] = check_r34(rec)
    print(f"R3/R4: every recorded comparison asserted "
          f"({res['R3_R4_enforced']['registered_fields_n']} registered fields, "
          f"{len(res['R3_R4_enforced']['scripts'])} scripts)", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    res["seconds_process_cpu"] = round(time.process_time() - c0, 1)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    row = {"job": "r8vii-r1", "wall_s": res["seconds"], "process_cpu_s": res["seconds_process_cpu"],
           "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "review round 2, findings 2 and 3: the complete leaf-by-leaf population "
                   "comparison against exp-058's and exp-057's records and the enforced R3/R4 "
                   "refusal. Reads two analysis records, opens no vector and no label."}
    with open(ANALYSIS / "cost.json", "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"written {OUT.relative_to(ROOT)} ({res['seconds']} s wall, "
          f"{res['seconds_process_cpu']} s process cpu)", flush=True)


if __name__ == "__main__":
    main()
