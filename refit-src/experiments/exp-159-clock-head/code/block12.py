"""The ungated surface block of a 2025 month that exp-046 never built (April, May, September).

    cd <repo root on the Mac>; .venv/bin/python experiments/exp-159-clock-head/code/block12.py --month 2025-04
    ... --month 2025-03 --parity        (rebuild a month exp-046 holds and compare it)

Runs on the Mac: only the Mac holds the exp-045 harvest summaries. Calls exp-046's own builder
(`chainfold.build_month`, the call every exp-046 script makes: the month's raw rows masked like
the ranking file, the prefix months of `arms.prefix_months_for`, the one serving rule) with its
cache directory pointed at this experiment's `artifacts/`. No exp-046 file is written or changed. The
builder reads no label (`block_ungated.COHORT_COLS`). Write-once: the block, and a record with its
rows, cache key, sha256 and seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E046 = ROOT / "experiments/exp-046-surfacefit"
OUT = HERE.parent / "artifacts"          # ignored by git (.gitignore: artifacts/)
sys.path.insert(0, str(E046 / "code"))
sys.path.insert(0, str(ROOT))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--parity", action="store_true", help="compare with exp-046's own block of the month")
    a = ap.parse_args()
    rec_path = OUT / f"block-{a.month}{'-parity' if a.parity else ''}.json"
    if rec_path.exists():
        raise SystemExit(f"{rec_path} exists; write-once")
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    import arms as A                    # noqa: E402
    import chainfold as CF              # noqa: E402
    own = E046 / "work"
    CF.CACHE = OUT
    rule = A.serving_rule()[0]
    key = A.block_cache_key(a.month, rule)
    p = OUT / f"block-{a.month}-{key}-ungated.parquet"
    if p.exists():
        raise SystemExit(f"{p} exists; write-once")
    prefix = A.prefix_months_for(a.month)
    print(f"{a.month}: rule {rule}, key {key}, prefix months {prefix}, imports {time.time() - t0:.0f}s", flush=True)
    b = CF.build_month(a.month, prefix, rule)
    rec = {"what": "exp-159 ungated surface block by exp-046's builder", "month": a.month, "rule": rule, "key": key,
           "prefix_months": prefix, "rows": int(len(b)), "columns": int(b.shape[1]),
           "obs_has_clock": int(b.obs_has_clock.fillna(False).astype(bool).sum()),
           "match_mode": {str(k): int(v) for k, v in b.match_mode.value_counts(dropna=False).items()},
           "sha256": sha(p), "seconds": round(time.time() - t0, 1),
           "code_sha256": {"block12.py": sha(Path(__file__)), "chainfold.py": sha(E046 / "code/chainfold.py"),
                           "block_ungated.py": sha(E046 / "analysis/chain-check/block_ungated.py")}}
    if a.parity:
        q = own / p.name
        if not q.exists():
            raise SystemExit(f"parity: exp-046 has no {p.name} (its key differs from {key})")
        ref, new = pd.read_parquet(q), pd.read_parquet(p)
        rec["parity"] = {"reference_sha256": sha(q), "bytes_equal": sha(q) == sha(p), "frames_equal": bool(ref.equals(new))}
        print("parity:", rec["parity"], flush=True)
    rec_path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("month", "rows", "obs_has_clock", "match_mode", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
