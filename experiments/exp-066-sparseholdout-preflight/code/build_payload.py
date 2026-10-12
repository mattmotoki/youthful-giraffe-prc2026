"""Build the literal allowlisted payload, internal manifest, and opaque chunks."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--chunk-bytes", type=int, default=262144)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    allow = (EXP / "source_allowlist.txt").read_text().splitlines()
    if allow != sorted(allow) or len(allow) != len(set(allow)):
        raise SystemExit("allowlist must be sorted and unique")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    members = {rel: (ROOT / rel).read_bytes() for rel in allow}
    members["GIT_COMMIT.txt"] = (commit + "\n").encode()
    manifest = {name: {"sha256": sha(data), "bytes": len(data)} for name, data in sorted(members.items())}
    manifest_bytes = (json.dumps({"source_commit": commit, "members": manifest}, sort_keys=True, indent=2) + "\n").encode()
    archive = a.out / "payload.archive"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as tf:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name); info.size = len(data); info.mtime = 0; info.uid = info.gid = 0
            info.uname = info.gname = ""; info.mode = 0o644
            tf.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("SOURCE_MANIFEST.json"); info.size = len(manifest_bytes); info.mtime = 0
        info.uid = info.gid = 0; info.uname = info.gname = ""; info.mode = 0o644
        tf.addfile(info, io.BytesIO(manifest_bytes))
    blob = archive.read_bytes()
    chunks = []
    for index, start in enumerate(range(0, len(blob), a.chunk_bytes)):
        part = blob[start:start+a.chunk_bytes]
        path = a.out / f"payload.part{index:03d}"
        path.write_bytes(part)
        chunks.append({"name": path.name, "index": index, "bytes": len(part), "sha256": sha(part)})
    record = {"source_commit": commit, "internal_manifest_sha256": sha(manifest_bytes),
              "archive_sha256": sha(blob), "archive_bytes": len(blob),
              "manifest_entries": len(manifest), "archive_members": len(manifest)+1,
              "chunk_bytes": a.chunk_bytes, "chunks": chunks}
    (a.out / "transport.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    archive.unlink()


if __name__ == "__main__": main()
