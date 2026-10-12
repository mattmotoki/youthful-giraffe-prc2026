"""Every record this lane wrote in a time window, parsed, row-counted and hashed.

The desk asked for this on 2026-09-08 16:12 UTC after the laptop volume ran out of
space mid-emit: a full disk truncates a file without an error the writer sees, so
"it wrote" is not evidence that "it is complete".  This reads every record back,
refuses anything that does not parse, checks each fold record's row count against the
panel it names, checks a census's per-airport rows sum to its pooled row, and checks
the manifest's recorded rows and SHA-256 against the parquet actually on disk.

    OMP_NUM_THREADS=1 .venv/bin/python experiments/exp-047-queue/code/verify_writes.py \
        [--since "2026-09-08 12:30"] [--until "2026-09-08 15:00"]

Exit code 1 and a PROBLEMS list if anything fails; nothing is repaired here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
AN = EXP / "analysis"
BLOCKS = ROOT / "artifacts/exp-047/block"
#: the fold panels' row counts, from `base.load()` and unchanged since exp-038
EXPECT = {"july_loo": 190713, "december": 165677, "january": 153706}


def _utc(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M")) - time.timezone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-08 00:00")
    ap.add_argument("--until", default="2099-01-01 00:00")
    ap.add_argument("--out", default="verify-writes.json")
    a = ap.parse_args()
    lo, hi = _utc(a.since), _utc(a.until)
    bad, rows = [], []
    for f in sorted(AN.glob("*.json")) + sorted(BLOCKS.glob("*.parquet")):
        m = f.stat().st_mtime
        if not (lo <= m <= hi) or f.name == a.out:
            continue
        rec = {"file": str(f.relative_to(ROOT)), "utc": time.strftime(
            "%Y-%m-%d %H:%M", time.gmtime(m)), "bytes": f.stat().st_size}
        if f.suffix == ".parquet":
            d = pd.read_parquet(f)
            rec |= {"rows": len(d), "columns": d.shape[1],
                    "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
            rows.append(rec)
            continue
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            rec["error"] = f"unreadable: {e}"
            bad.append(rec); rows.append(rec); continue
        if "rows" in d and "fold" in d:
            want = EXPECT.get(d["fold"])
            n = d["rows"]
            ok = (n <= want) if d.get("cover_only") else (n == want)
            rec |= {"fold": d["fold"], "rows": n, "panel_rows": want,
                    "cover_only": bool(d.get("cover_only")), "ok": ok}
            if not ok:
                bad.append(rec)
            for arm, r in d.get("arms", {}).items():
                for tag in ("all", "matched", "unmatched"):
                    v = r.get(tag) or {}
                    if v.get("rows") and v["rows"] > n:
                        bad.append(dict(rec, error=f"{arm}/{tag} {v['rows']} > {n}"))
        elif "all" in d and "by_airport" in d:
            tot = sum(v["rows"] for v in d["by_airport"].values())
            rec |= {"rows": d["all"]["rows"], "airport_rows_sum": tot,
                    "ok": tot == d["all"]["rows"]}
            if not rec["ok"]:
                bad.append(rec)
        elif "months" in d:
            rec["months"] = {}
            for mth, r in d["months"].items():
                p = Path(r["path"])
                p = p if p.is_absolute() else ROOT / p
                on = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
                nrow = len(pd.read_parquet(p)) if p.exists() else None
                good = (on == r["sha256"] and nrow == r["rows"])
                rec["months"][mth] = {"manifest_rows": r["rows"], "disk_rows": nrow,
                                      "sha_match": on == r["sha256"], "ok": good}
                if not good:
                    bad.append(dict(rec, error=f"{mth}: manifest and disk disagree"))
        rows.append(rec)
    out = {"since": a.since, "until": a.until, "checked": len(rows),
           "problems": bad, "records": rows}
    (AN / a.out).write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    for r in rows:
        print(f"{r['utc']} {r['file']:58s} {r['bytes']:10,d}  "
              f"{r.get('rows', '')} {'' if r.get('ok', True) else 'MISMATCH'}")
    print(f"\n{len(rows)} records checked; problems: {len(bad)}")
    if bad:
        print(json.dumps(bad, indent=1))
        sys.exit(1)


if __name__ == "__main__":
    main()
