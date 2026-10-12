"""exp-054-arrivalrepr: the comparison of the clean audit with the earlier audit records (Sol round 2,
finding 1).  Run only AFTER `audit3.py` has written `analysis/audit-r3.json`; it computes nothing
that feeds a choice, and it writes only equality flags.

It parses three records in full:
- `analysis/audit-r3.json`, the clean audit (no earlier record read);
- `analysis/audit-r2.json`, the first clean audit, whose figures SPEC section 3.2 quotes: its
  inventory, choice evidence and fold entries must equal audit-r3's exactly;
- `analysis/audit-r1.json`, the registered audit: its nine never-held inventory entries and its
  fold entries must equal audit-r3's.  This file also holds the held months' arrival outcome
  summaries; this job reads them as part of the file and compares none of them.

    .venv/bin/python experiments/exp-054-arrivalrepr/code/audit_compare.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e54 as E                                                       # noqa: E402

NO = E.NO
OUT_DEFAULT = "analysis/audit-compare.json"
NINE = [f"2025-{i:02d}" for i in (2, 3, 4, 5, 6, 8, 9, 10, 11)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = E.EXP / args.out
    NO.refuse_existing([OUT], args.overwrite)
    paths = {n: E.EXP / f"analysis/{n}.json" for n in ("audit-r3", "audit-r2", "audit-r1")}
    r3 = json.loads(paths["audit-r3"].read_text())
    assert sorted(r3["inventory"]) == NINE
    r2 = json.loads(paths["audit-r2"].read_text())
    r1 = json.loads(paths["audit-r1"].read_text())
    eq = {
        "r3_vs_r2": {"inventory": r3["inventory"] == r2["inventory"],
                     "choice_evidence": r3["choice_evidence"] == r2["choice_evidence"],
                     "folds": r3["folds"] == r2["folds"]},
        "r3_vs_r1": {"inventory": {m: r3["inventory"][m] == r1["inventory"][m] for m in NINE},
                     "folds": {f: r3["folds"][f] == r1["folds"][f] for f in E.FOLDS}}}
    eq["all_equal"] = (all(eq["r3_vs_r2"].values()) and all(eq["r3_vs_r1"]["inventory"].values())
                       and all(eq["r3_vs_r1"]["folds"].values()))
    rec = {"what": "exp-054 comparison of the clean audit (audit-r3) with audit-r2 and audit-r1; run after "
                   "audit-r3 was written; equality flags only",
           "code_sha256": {n: E.sha_file(E.HERE / n) for n in ("audit_compare.py", "e54.py")},
           "records_sha256": {n: E.sha_file(p) for n, p in paths.items()}, "equal": eq, "checked_utc": E.utc()}
    print(json.dumps(eq, indent=1), flush=True)
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"written {OUT.relative_to(E.ROOT)}")
    if not eq["all_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
