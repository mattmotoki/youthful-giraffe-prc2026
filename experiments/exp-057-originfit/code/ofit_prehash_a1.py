"""exp-057 originfit: census amendment a1 (SPEC 2.3).  BYTES ONLY.

The plan stage's fit-input check (analysis/plan-census-fail-a0.json) found 11 repository files
that the process opened and the pre-registration census lacks.  This script hashes each of them
(full SHA-256 and size) and searches every committed record under experiments/*/analysis for the
full hash string, as bytes.  It parses no record and reads no prediction value, label or loss.
An amendment adds paths; it never changes a census hash.

    .venv/bin/python experiments/exp-057-originfit/code/ofit_prehash_a1.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
FAIL = EXP / "analysis/plan-census-fail-a0.json"
OUT = EXP / "analysis/prehash-a1.json"
CLASS = {"artifacts/exp-018-fresh/cache/": "feature-cache", "experiments/exp-035-k2climb/oof/": "base"}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    NO.refuse_existing([OUT], overwrite=False)
    t0 = time.time()
    fail = json.loads(FAIL.read_text())
    missing = fail["census_check"]["missing"]
    if fail["census_check"]["differ"]:
        raise SystemExit("the failed plan also lists changed files; an amendment cannot cover those")
    tracked = set(subprocess.run(["git", "ls-files", "experiments"], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.split())
    records = sorted(p for p in ROOT.glob("experiments/*/analysis/**/*.json")
                     if p.is_file() and "exp-057-originfit" not in p.parts)
    blobs = {str(p.relative_to(ROOT)): p.read_bytes() for p in records}
    files = {}
    for rel in missing:
        p = ROOT / rel
        sha = sha_file(p)
        cls = next((c for pre, c in CLASS.items() if rel.startswith(pre)), "other")
        hits = sorted(r for r, b in blobs.items() if sha.encode() in b)
        files[rel] = {"class": cls, "bytes": p.stat().st_size, "sha256": sha,
                      "earlier_records_holding_this_full_hash": [
                          {"record": r, "tracked_in_git": r in tracked} for r in hits]}
    rec = {"what": __doc__.strip().splitlines()[0],
           "amends": "experiments/exp-057-originfit/analysis/prehash.json",
           "found_by": str(FAIL.relative_to(ROOT)),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "records_searched": len(records), "files": files,
           "every_file_has_a_tracked_earlier_record": all(
               any(h["tracked_in_git"] for h in v["earlier_records_holding_this_full_hash"])
               for v in files.values()),
           "seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(rec, indent=1, sort_keys=True))
    for rel, v in files.items():
        recs = [h["record"] for h in v["earlier_records_holding_this_full_hash"] if h["tracked_in_git"]]
        print(f"{rel}  {v['sha256'][:16]}  {v['bytes']:>11,} B  tracked earlier records: {len(recs)} "
              f"({', '.join(r.split('/analysis/')[0].split('/')[-1] + ':' + r.split('/')[-1] for r in recs)})")
    print(f"{len(files)} files; every file has a tracked earlier record: "
          f"{rec['every_file_has_a_tracked_earlier_record']}; {len(records)} records searched")


if __name__ == "__main__":
    main()
