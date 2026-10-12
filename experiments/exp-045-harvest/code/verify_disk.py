"""exp-045-harvest: did anything this experiment wrote get truncated by the full volume?

The Mac's volume sat at 100 % for some hours on 2026-09-08 from work outside this repo,
and a write that runs out of space leaves a short file, not an error the writer sees.
So every artifact is re-read rather than trusted:

  summaries   each parquet is opened, its rows counted, and its sha256 compared with
              the one manifest.json recorded when it was written
  json        every analysis record is parsed
  code        every source file is compiled

    .venv/bin/python experiments/exp-045-harvest/code/verify_disk.py [YYYY-MM ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(months: list[str]) -> int:
    man = json.loads((EXP / "manifest.json").read_text())
    bad: list[str] = []
    files = rows = 0
    for mo, rec in sorted(man["months"].items()):
        if months and mo not in months:
            continue
        for day, d in sorted(rec["days"].items()):
            for airport, sm in sorted(d.get("summaries", {}).items()):
                p = ROOT / sm["path"]
                if not p.exists():
                    bad.append(f"{day} {airport}: missing {sm['path']}")
                    continue
                files += 1
                if p.stat().st_size != sm["bytes"]:
                    bad.append(f"{day} {airport}: {p.stat().st_size} bytes on disk, "
                               f"manifest says {sm['bytes']}")
                    continue
                if sha256(p) != sm["sha256"]:
                    bad.append(f"{day} {airport}: sha256 differs from the manifest")
                    continue
                try:
                    n = len(pd.read_parquet(p))
                except Exception as exc:                       # noqa: BLE001
                    bad.append(f"{day} {airport}: unreadable ({type(exc).__name__})")
                    continue
                rows += n
                if n != sm["rows"]:
                    bad.append(f"{day} {airport}: {n} rows, manifest says {sm['rows']}")

    jsons = 0
    for p in sorted((EXP / "analysis").rglob("*.json")) + [EXP / "manifest.json"]:
        try:
            json.loads(p.read_text())
            jsons += 1
        except Exception as exc:                               # noqa: BLE001
            bad.append(f"{p.relative_to(ROOT)}: will not parse ({type(exc).__name__})")

    srcs = 0
    for p in sorted((EXP / "code").glob("*.py")) + sorted((EXP / "kernels").rglob("*.py")):
        try:
            compile(p.read_text(), str(p), "exec")
            srcs += 1
        except Exception as exc:                               # noqa: BLE001
            bad.append(f"{p.relative_to(ROOT)}: will not compile ({type(exc).__name__})")

    for line in bad:
        print(line)
    print(f"verified {files} summary files ({rows:,} rows), {jsons} json records, "
          f"{srcs} sources; {len(bad)} bad")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
