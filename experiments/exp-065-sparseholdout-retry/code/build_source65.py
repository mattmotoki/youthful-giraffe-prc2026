"""Build the immutable exp-065 source archive from a recorded positive file list."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "source/exp065-source.tar.gz"
LIST = EXP / "source/source-files.json"

SOURCE_ROOTS = (
    "prc",
    "experiments/exp-001-retro-v1",
    "experiments/exp-008-diversity",
    "experiments/exp-009-noac2",
    "experiments/exp-018-fresh/code",
    "experiments/exp-046-surfacefit/code",
    "experiments/exp-050-nmblind/code",
    "experiments/exp-065-sparseholdout-retry/code",
)
SOURCE_SUFFIXES = {".py", ".json", ".txt", ".md"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists() or LIST.exists():
        raise SystemExit("write-once refusal: source archive or list exists")
    files = []
    for rel_root in SOURCE_ROOTS:
        base = ROOT / rel_root
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in SOURCE_SUFFIXES and not any(
                    x in p.parts for x in ("__pycache__", "logs", "review", "oof", "work")):
                files.append(p.relative_to(ROOT))
    # Exact auxiliary records opened by e54 before import.
    for rel in ("experiments/exp-050-nmblind/analysis/panel.json",
                "experiments/exp-050-nmblind/analysis/router-july_loo-december-january.json"):
        files.append(Path(rel))
    files = sorted(set(files), key=str)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    LIST.write_text(json.dumps({"files": {str(p): sha(ROOT / p) for p in files}},
                               indent=1, sort_keys=True) + "\n")
    with tarfile.open(OUT, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for rel in files:
            tf.add(ROOT / rel, arcname=Path("repo") / rel, recursive=False)
        tf.add(LIST, arcname="repo/experiments/exp-065-sparseholdout-retry/source/source-files.json",
               recursive=False)
    print(json.dumps({"path": str(OUT.relative_to(ROOT)), "bytes": OUT.stat().st_size,
                      "sha256": sha(OUT), "files": len(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
