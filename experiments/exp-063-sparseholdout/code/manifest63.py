"""Create exp-063's write-once positive source manifest without reading any label value."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "analysis/manifest.json"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"write-once refusal: {OUT}")
    rules = [
        ROOT / "assignments/sparseholdout.md", ROOT / "assignments/COMMON.md",
        ROOT / "assignments/LAPTOP.md", ROOT / "docs/input-rule.md",
        ROOT / "docs/CLOUD-COMPUTE.md", ROOT / "docs/DECISIONS.md",
        ROOT / "docs/pro-reasoning-2026-09-08/QUEUE.md",
    ]
    old_docs = []
    for n in (56, 60, 61):
        d = next((ROOT / "experiments").glob(f"exp-{n:03d}-*"))
        old_docs += [d / "SPEC.md", d / "RESULTS.md"]
        old_docs += sorted((d / "analysis").glob("*report*.md"))
        old_docs += sorted((d / "analysis").glob("*receipt*"))
        old_docs += sorted((d / "review").glob("round[12]*.txt"))
    code = sorted((EXP / "code").glob("*.py"))
    kernels = sorted(p for p in (EXP / "kernels").glob("*/*") if p.is_file())
    raw = sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))
    rec = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True),
        "rotation": {"V": "2025-05", "S": "2025-06", "I": "2025-04",
                     "A": ["2025-01", "2025-02", "2025-03", "2025-07", "2025-08",
                           "2025-09", "2025-10", "2025-11", "2025-12"]},
        "rules": {str(p.relative_to(ROOT)): sha(p) for p in rules},
        "historical_documents_read_not_producing_inputs":
            {str(p.relative_to(ROOT)): sha(p) for p in old_docs},
        "producing_code": {str(p.relative_to(ROOT)): sha(p) for p in code + kernels},
        "raw_months": {str(p.relative_to(ROOT)): sha(p) for p in raw},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "sealed_archive_allowlist": [
            "ids_V", "ids_S", "ids_I", "pop_V", "pop_S", "pop_I", "p_V", "p_S", "p_I",
            "day_V", "nonm_V",
            *[f"code_{s}_{f}" for s in ("V", "S", "I") for f in ("apt", "stand", "rwy", "op")],
            *[f"nA_{f}" for f in ("stand", "rwy", "op")],
            *[f"prior_{a}_{f}" for a in ("true", "const", "gmean", "donor", "shuffle")
              for f in ("stand", "rwy", "op")],
        ],
        "forbidden_producing_roots": ["experiments/exp-056-q12advocate",
                                       "experiments/exp-060-coldprior",
                                       "experiments/exp-061-sparseprior"],
    }
    OUT.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(sha(OUT))


if __name__ == "__main__":
    main()
