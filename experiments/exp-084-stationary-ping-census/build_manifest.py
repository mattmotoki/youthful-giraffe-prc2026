import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("source-manifest.json")


def record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


fixed = [
    ROOT / "data/raw/ranking.parquet",
    ROOT / "artifacts/exp-046/class/2026-01-v22.parquet",
    ROOT / "artifacts/exp-046/class/2026-07-v22.parquet",
    ROOT / "submissions/uploaded/youthful-giraffe_v22.parquet",
]
all_summaries = (ROOT / "artifacts/exp-045/summaries").glob("*/*.parquet")
summaries = sorted(path for path in all_summaries
                   if "2025-12-29" <= path.stem <= "2026-01-31"
                   or "2026-06-29" <= path.stem <= "2026-07-31")
manifest = {
    "experiment": "exp-084-stationary-ping-census",
    "population": {"rule": "2026-01/07 ranking departures with not (obs_has_clock and not noac)",
                   "rows": 148585, "arm_n": 69634, "base_v12": 78951, "noac": 388},
    "fixed_sources": [record(path) for path in fixed],
    "summary_collection": {"file_count": len(summaries),
                           "files": [record(path) for path in summaries]},
}
OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
