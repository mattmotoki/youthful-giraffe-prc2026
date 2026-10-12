"""Give every blend member's OOF bytes a durable address, and verify it.

The desk made this the rule for blend inputs after review round 1 found the
0.4438-weight member sourcing an uncommitted file that another lane was still
editing (RESULTS, "Review round 1"). The first attempt at that fix was itself
incomplete: `experiments/exp-012-blend2/.gitignore` carries `oof/*.parquet`, so
`git add -A` silently skipped the pinned copy and the member still had no address
in git. Pins therefore live in `pinned/`, which nothing ignores.

A member's bytes are durable if the file is tracked AND clean -- then its git
blob is the address and copying it would only duplicate 10 MB. Otherwise the
bytes are copied here and committed. Either way the sha256 goes in MANIFEST.md,
so a later session can tell whether what it is reading is what was measured.

  python pin_members.py            # verify only, non-zero exit if anything drifted
  python pin_members.py --write    # copy what needs pinning and rewrite MANIFEST
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = HERE / "pinned"
FOLDS = ("july_loo", "december", "january")
CONFIGS = ("members-cand1.json", "members-2.json", "members-3.json")
MARK = "## Member pins (durable addresses for every blend input)"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a) -> tuple[int, str]:
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def status(rel: str) -> str:
    if git("ls-files", "--error-unmatch", rel)[0] != 0:
        return "untracked"
    return "clean" if not git("status", "--porcelain", rel)[1] else "modified"


def member_files() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in CONFIGS:
        p = HERE / c
        if not p.exists():
            continue
        for name, tmpl in json.loads(p.read_text())["members"].items():
            paths = ([tmpl.format(fold=f) for f in FOLDS] if "{fold}" in tmpl else [tmpl])
            out.setdefault(name, [])
            for q in paths:
                if q not in out[name]:
                    out[name].append(q)
    return out


def main(write: bool) -> int:
    rows, drift = [], 0
    for name, paths in sorted(member_files().items()):
        for rel in paths:
            src = ROOT / rel
            if not src.exists():
                print(f"MISSING {name}: {rel}")
                drift += 1
                continue
            st = status(rel)
            here = rel.startswith("experiments/exp-012-blend2/pinned/")
            if st == "clean":
                # already durable: a tracked, unmodified file IS its own address,
                # whether it is one of these pins or another lane's committed OOF.
                how = "pin, committed" if here else f"git blob, {git('log','-1','--format=%h','--',rel)[1]}"
                rows.append((name, rel if not here else f"pinned/{Path(rel).name}",
                             sha256(src), how))
                continue
            if here:
                print(f"PIN NOT COMMITTED {name}: {rel} is {st}")
                drift += 1
                rows.append((name, f"pinned/{Path(rel).name}", sha256(src), f"pin, {st} -- COMMIT IT"))
                continue
            # not durable: pin a copy here
            dst = PIN / f"{name}--{Path(rel).name}"
            if write:
                PIN.mkdir(exist_ok=True)
                if not dst.exists() or sha256(dst) != sha256(src):
                    shutil.copy2(src, dst)
                    print(f"pinned {name}: {rel} -> pinned/{dst.name}")
            if not dst.exists():
                print(f"NOT PINNED {name}: {rel} (source is {st})")
                drift += 1
                continue
            if sha256(dst) != sha256(src):
                print(f"DRIFT {name}: pinned copy differs from {rel}")
                drift += 1
            rows.append((name, f"pinned/{dst.name}", sha256(dst), f"copy of {rel} ({st} at pin time)"))

    body = [MARK, "",
            "Regenerate and verify with `python pin_members.py [--write]`. A member whose",
            "source is tracked and clean keeps its git blob as the address; anything",
            "untracked or modified is copied into `pinned/` and committed, because a",
            "submission candidate cannot rest on bytes another lane is still editing.",
            "", "| member | address | sha256 | how the address is held |", "|---|---|---|---|"]
    for n, a, h, how in rows:
        body.append(f"| `{n}` | `{a}` | `{h[:16]}…` | {how} |")
    text = "\n".join(body) + "\n"

    mp = HERE / "MANIFEST.md"
    cur = mp.read_text() if mp.exists() else ""
    new = (cur[:cur.index(MARK)] if MARK in cur else cur.rstrip() + "\n\n") + text
    if write:
        mp.write_text(new)
        print(f"MANIFEST.md updated with {len(rows)} member pins")
    print(f"\n{len(rows)} member files, {drift} problem(s)")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main("--write" in sys.argv))
