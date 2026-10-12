"""exp-055, Sol round 1 finding 2: the census addendum.  NO VALUE READ (bytes only).

1. The three raw 2025 month files that `pin.py` (take-off dates) and `lossfals.py` (the no-NM
   masks) read were not in `analysis/prehash.json`.  This hashes them and compares each with the
   hash recorded earlier in exp-050's `analysis/features.json` (`months.<m>.month_file_sha256`)
   and in exp-041's `analysis/packet-order.json`
   (`provenance.validation_month_file_sha256.<fold>`).  A mismatch is fatal.
2. The census count: `prehash.json` holds 142 entries, keyed by record, not by file.  This counts
   the distinct paths, lists every path that more than one entry names, and checks that those
   entries carry one hash.

    .venv/bin/python experiments/exp-055-originmove/code/prehash_r1.py
"""
from __future__ import annotations

import collections
import json
import time

import common as C
from prc import data as D                                             # noqa: E402

OUT = C.ANALYSIS / "prehash-r1.json"
E50F = C.ROOT / "experiments/exp-050-nmblind/analysis/features.json"
E41P = C.ROOT / "experiments/exp-041-cell/analysis/packet-order.json"


def main() -> None:
    C.NO.refuse_existing([OUT], overwrite=False)
    t0 = time.time()
    e50 = json.loads(E50F.read_text())["months"]
    e41 = json.loads(E41P.read_text())["provenance"]["validation_month_file_sha256"]
    rec = {"what": __doc__.strip().splitlines()[0],
           "records": {C.rel(E50F): C.sha_file(E50F), C.rel(E41P): C.sha_file(E41P)},
           "raw_months": {}, "mismatches": []}
    for f in C.FOLDS:
        m = C.FOLD_MONTH[f]
        p = D.month_file(m)
        got = C.sha_file(p)
        want50 = e50[m]["month_file_sha256"]
        want41 = e41[f]
        ok = got == want50 == want41
        rec["raw_months"][m] = {"fold": f, "path": C.rel(p), "bytes": p.stat().st_size, "sha256": got,
                                "exp050_features_json": want50, "exp041_packet_order_json": want41,
                                "matches_both_records": ok}
        if not ok:
            rec["mismatches"].append(m)
        print(f"{m} {C.rel(p)} {got[:16]} matches exp-050 and exp-041 records: {ok}", flush=True)
    pre = C.PREHASH["files"]
    by_path = collections.defaultdict(list)
    for k, v in pre.items():
        by_path[v["path"]].append(k)
    dup = {p: ks for p, ks in by_path.items() if len(ks) > 1}
    for p, ks in dup.items():
        if len({pre[k]["sha256"] for k in ks}) != 1:
            rec["mismatches"].append(p)
    rec_paths = {v["path"] for v in pre.values() if "recorded_sha256" in v}
    rec["census"] = {"entries": len(pre), "distinct_paths": len(by_path),
                     "paths_named_by_more_than_one_entry": dup,
                     "distinct_paths_with_an_earlier_recorded_hash": len(rec_paths),
                     "entries_with_an_earlier_recorded_hash": sum("recorded_sha256" in v for v in pre.values()),
                     "entries_checked_against_results_md_prefix": sum("results_md_prefix" in v for v in pre.values()),
                     "distinct_paths_hashed_first": len(set(by_path) - rec_paths)}
    rec["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, rec)
    print(f"census: {len(pre)} entries, {len(by_path)} distinct paths; named twice: {sorted(dup)}; "
          f"mismatches: {rec['mismatches'] or 'none'}; written {C.rel(OUT)}", flush=True)
    if rec["mismatches"]:
        raise SystemExit(f"mismatch: {rec['mismatches']}")


if __name__ == "__main__":
    main()
