"""Export one caller-supplied reviewed commit and its separate trusted manifest.

Both outputs come from that immutable commit. No working-tree content or default
HEAD is used. Publish only the verified export, with no private repository history.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = "release/MANIFEST.tsv"


def reviewed_commit(value: str, root: Path = ROOT) -> str:
    """Reject absent, abbreviated, symbolic, moving and non-commit revisions."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ValueError("--rev must be the full 40-hex reviewed commit")
    result = subprocess.run(["git", "rev-parse", "--verify", f"{value}^{{commit}}"],
                            cwd=root, capture_output=True, text=True)
    if result.returncode or result.stdout.strip().lower() != value.lower():
        raise ValueError("--rev must identify a commit object, not a tag or other object")
    return result.stdout.strip()


def export(revision: str, out: Path, contract: Path, root: Path = ROOT) -> dict:
    rev = reviewed_commit(revision, root)  # resolve once, before creating outputs
    out, contract = out.resolve(), contract.resolve()
    if out.exists() or contract.exists():
        raise ValueError("export and trusted-manifest destinations must not exist")
    if contract == out or out in contract.parents:
        raise ValueError("trusted manifest must be outside the export")
    man = subprocess.check_output(["git", "show", f"{rev}:{MANIFEST_PATH}"], cwd=root)
    rows = list(csv.DictReader(io.StringIO(man.decode()), delimiter="\t"))
    paths = [row['path'] for row in rows]
    if paths != sorted(set(paths)) or MANIFEST_PATH in paths:
        raise ValueError("commit manifest is not sorted and unique")
    for rel in paths:
        path = PurePosixPath(rel)
        if path.is_absolute() or '..' in path.parts or '.git' in path.parts or str(path) != rel:
            raise ValueError(f"unsafe manifest path: {rel}")
    out.mkdir(parents=True)
    # One request at a time keeps memory independent of export size.
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=root,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        for rel in paths:
            proc.stdin.write(f"{rev}:{rel}\n".encode())
            proc.stdin.flush()
            header = proc.stdout.readline().decode().split()
            if len(header) != 3 or header[1] != 'blob':
                raise ValueError(f"missing release blob: {rel}")
            remaining = int(header[2])
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                while remaining:
                    block = proc.stdout.read(min(1 << 20, remaining))
                    if not block:
                        raise ValueError(f"incomplete release blob: {rel}")
                    stream.write(block)
                    remaining -= len(block)
            if proc.stdout.read(1) != b'\n':
                raise ValueError("invalid git blob framing")
        proc.stdin.close()
        if proc.wait():
            raise ValueError("git blob export failed")
    (out / MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST_PATH).write_bytes(man)
    contract.parent.mkdir(parents=True, exist_ok=True)
    with contract.open('xb') as stream:
        stream.write(man)
    files = [p for p in out.rglob('*') if p.is_file()]
    return dict(revision=rev, files=len(files), bytes=sum(p.stat().st_size for p in files),
                export=str(out), contract=str(contract))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rev', required=True, help='full 40-hex reviewed commit')
    parser.add_argument('--out', required=True, type=Path, help='new export directory')
    parser.add_argument('--contract-out', required=True, type=Path,
                        help='new trusted manifest outside the export')
    args = parser.parse_args()
    import json
    print(json.dumps(export(args.rev, args.out, args.contract_out), sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
