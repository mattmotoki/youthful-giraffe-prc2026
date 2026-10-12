"""Write `release/MANIFEST.tsv`: every file of the public release, sorted,
with its SHA-256, its size and its role.

    .venv/bin/python release/build_manifest.py            # write the manifest
    .venv/bin/python release/build_manifest.py --check    # write nothing; diff
    .venv/bin/python release/build_manifest.py --rev <commit>   # another commit

Every digest is taken from the COMMITTED content of `--rev` (default HEAD), never
from the working tree.  Round 1 finding 1: the first manifest hashed the working
tree, and another session's uncommitted edit to
`experiments/exp-062-ownvalue-refine/analysis/receipts.jsonl` went into it, so the
manifest described a tree that no commit contained and gate C had to fail on the
exact commit.  A release contract is a property of a commit or it is nothing.

The manifest lists every released file except itself: hashing MANIFEST.tsv
is part of writing it, so it cannot carry its own digest.  That single
exception lives in `release_set.manifest_set`.

The manifest is the release's contract.  `release/check_release.py` verifies a
tree against it and fails on a file that is missing, changed or unexpected.
The set itself comes from `release/release_set.py`; this file only hashes it
and names what each path is for.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_set import ROOT, manifest_set                     # noqa: E402

MANIFEST = ROOT / "release/MANIFEST.tsv"
HEADER = ("path", "sha256", "bytes", "role")


def committed(path: str, rev: str) -> bytes:
    """The bytes of `path` as `rev` has them.  Raises if the path is not in it."""
    r = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"{path} is not in {rev}: {r.stderr.decode().strip()}")
    return r.stdout


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def role(path: str) -> str:
    """What this file is for, from its path alone.  One word per class."""
    if path in ("LICENSE",):
        return "licence"
    if path in ("README.md", "requirements.txt", ".env.example", ".gitignore"):
        return "repository"
    if path.startswith("prc/"):
        return "library"
    if path.startswith("release/"):
        return "release-contract"
    if path.startswith(("refit-src/", "refit-pins/")):
        return "refit-source"
    if path.startswith("scripts/"):
        return "entry-point"
    if path.startswith("notebooks/"):
        return "notebook"
    if path.startswith("modeling/"):
        return "first-sprint"
    if path.startswith("references/"):
        return "external-source-lock"
    if path.startswith("docs/"):
        return "documentation"
    if path.startswith("experiments/"):
        parts = path.split("/")
        tail = "/".join(parts[2:]) if len(parts) > 2 else ""
        if tail.startswith("code/") or tail.endswith(".py") or tail.endswith(".sh"):
            return "experiment-code"
        if tail.startswith("kernels/"):
            return "kernel-source"
        if parts[-1] in ("SPEC.md", "RESULTS.md", "REPORT.md", "climb.log", "climb.tsv"):
            return "experiment-record"
        if tail.endswith(".md"):
            return "experiment-record"
        return "experiment-analysis"
    return "other"


def build(rev: str = "HEAD") -> list[tuple[str, str, int, str]]:
    rev = subprocess.check_output(["git", "rev-parse", "--verify", f"{rev}^{{commit}}"],
                                  cwd=ROOT, text=True).strip()
    out = []
    for p in manifest_set(rev=rev):
        b = committed(p, rev)
        out.append((p, sha256_bytes(b), len(b), role(p)))
    return out


def render(rows) -> str:
    lines = ["\t".join(HEADER)]
    lines += ["\t".join((p, s, str(n), r)) for p, s, n, r in rows]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the manifest is out of date")
    ap.add_argument("--rev", default="HEAD",
                    help="the commit whose content is hashed (default HEAD)")
    a = ap.parse_args()
    rows = build(a.rev)
    text = render(rows)
    if a.check:
        old = MANIFEST.read_text() if MANIFEST.exists() else ""
        if old == text:
            print(f"MANIFEST.tsv is current: {len(rows)} files")
            return 0
        print("MANIFEST.tsv is OUT OF DATE; run release/build_manifest.py")
        return 1
    MANIFEST.write_text(text)
    total = sum(n for _, _, n, _ in rows)
    by_role: dict[str, list[int]] = {}
    for _, _, n, r in rows:
        d = by_role.setdefault(r, [0, 0])
        d[0] += 1
        d[1] += n
    print(f"wrote {MANIFEST.relative_to(ROOT)} from {a.rev}: "
          f"{len(rows)} files, {total/2**20:.1f} MiB")
    for r, (n, b) in sorted(by_role.items(), key=lambda kv: -kv[1][1]):
        print(f"  {r:22s} {n:5d} files {b/2**20:8.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
