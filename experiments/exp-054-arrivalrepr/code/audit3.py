"""exp-054-arrivalrepr: the clean audit, with no read of an earlier audit record (Sol round 2, finding 1).

`audit2.py` computed its clean figures first, but then, in the same process, it parsed the whole of
`analysis/audit-r1.json` to compare with it, and that file holds the held months' arrival outcome
summaries.  This job runs steps 1-4 of `audit2.py` with `audit2.py`'s own functions, unchanged
(`inventory_month`, `choice_evidence`, `fold_part`), and writes its record.  It opens no earlier
audit record, no RESULTS or SPEC text, and no held month's arrival row.  The comparison with the
earlier records is a separate job, `audit_compare.py`, run only after this record is written.

What this job reads: the nine never-held months (a PHASE_mvt count and ARR rows only,
`e54r.arrival_rows`); per fold, the arrival KEYS of the fitting months, the held month without
either label column (`e54.departure_frame`) and the panel without `y_true`; and the pinned files that
`e54` checks at import (exp-050's code and records, exp-046's guard).

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-054-arrivalrepr/code/audit3.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e54 as E                                                       # noqa: E402
import audit2 as A2                                                   # noqa: E402  (run code, unchanged)

D, NO = E.D, E.NO
OUT_DEFAULT = "analysis/audit-r3.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT, help="output path, relative to the experiment")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = E.EXP / args.out
    NO.refuse_existing([OUT], args.overwrite)
    print(f"guard: no existing artifact at {OUT.relative_to(E.ROOT)}", flush=True)
    t0 = time.time()
    rec = {"what": "exp-054 clean field and support audit (Sol round 2): audit2.py steps 1-4 only; no earlier "
                   "audit record read, no held month opened by the inventory, no departure label in any frame, "
                   "no fit",
           "code_sha256": {n: E.sha_file(E.HERE / n)
                           for n in ("audit3.py", "audit2.py", "audit.py", "e54.py", "e54r.py")},
           "pinned": E.check_pinned(), "support_rule": {"min_rows": E.MIN_ROWS, "min_days": E.MIN_DAYS},
           "y_clip": list(E.Y_CLIP), "held_months_not_opened_by_inventory": A2.HELD, "started_utc": E.utc()}
    print(f"1. arrival field inventory, nine months {A2.NINE[0]}..{A2.NINE[-1]} (held {A2.HELD} not opened)",
          flush=True)
    inv, extra, pooled = {}, {}, {}
    for m in A2.NINE:
        inv[m], ex = A2.inventory_month(m)
        for k, v in ex.pop("cand").items():
            pooled.setdefault(k, []).append(v)
        extra[m] = ex
        print(f"  {m}: {inv[m]['arrivals']:,} arrivals, identity {inv[m]['identity_taxi_in_equals_block_minus_mvt']}",
              flush=True)
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    rec["inventory"] = inv
    rec["choice_evidence"] = A2.choice_evidence(inv, pooled, extra)
    for k, c in rec["choice_evidence"]["candidates_pooled"].items():
        qq = c["quantiles_s"]
        print(f"  pooled {k:34s} coverage {c['coverage']:.5f}  p1/p50/p99 {qq['1']:.0f} / {qq['50']:.0f} / "
              f"{qq['99']:.0f} s", flush=True)
    rec["folds"] = {}
    for fold in E.FOLDS:
        fr = A2.fold_part(fold)
        rec["folds"][fold] = fr
        s = fr["support"]
        print(f"2-4. {fold}: P {fr['P_rows']:,}; supported share of P: "
              + ", ".join(f"{k} {s['keys'][k]['supported_share_P']:.3f}" for k in E.KEYS)
              + f"; any {s['any_key_supported_share_P']:.3f}; donors {fr['donor']['with_donor']:,} / "
              f"{fr['donor']['arrivals']:,}", flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"written {OUT.relative_to(E.ROOT)} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
