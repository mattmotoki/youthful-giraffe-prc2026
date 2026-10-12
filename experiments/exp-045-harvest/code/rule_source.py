"""exp-045-harvest: the rule as the repository holds it, not as the working tree does.

Both assemblers inline two files: the pilot's frozen `segments.py` and this
experiment's `allsegments.py`.  Both are read from **HEAD**, through `git show`, and
not from the working copy.

The reason is not tidiness.  At 12:42 UTC on 2026-09-08 an assembly picked up
`exp-043-pilot/code/segments.py` while the pilot had it open and half-edited: the rule
hash moved under a kernel that was about to be pushed, and the manifest would have
cited a repository path whose committed content was something else.  A kernel that
ran must be a kernel a reader can check out.  A dirty working copy is reported and
ignored, so the harvest is never stopped by another worker's unfinished edit.
"""
from __future__ import annotations

import subprocess
import sys

RULE = "experiments/exp-043-pilot/code/segments.py"
EXT = "experiments/exp-045-harvest/code/allsegments.py"


def at_head(root: str, path: str) -> str:
    return subprocess.run(["git", "-C", root, "show", f"HEAD:{path}"],
                          capture_output=True, text=True, check=True).stdout


def rule_sources(root: str) -> tuple[str, str]:
    """`(segments.py, allsegments.py)` at HEAD, with a warning if either is dirty."""
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", RULE, EXT],
                           capture_output=True, text=True, check=True).stdout.split()
    for path in (RULE, EXT):
        if path in dirty:
            print(f"note: {path} differs from HEAD; the kernel inlines the committed "
                  f"version, not the working copy", file=sys.stderr)
    return at_head(root, RULE), at_head(root, EXT)


def strip(src: str, drop: list[str]) -> str:
    """Drop `from __future__` and the imports that bind the two files together."""
    out, skipping = [], False
    for ln in src.splitlines(True):
        if skipping:                      # a parenthesised import spans lines
            skipping = ")" not in ln
            continue
        if ln.startswith("from __future__") or any(ln.startswith(d) for d in drop):
            skipping = ln.count("(") > ln.count(")")
            continue
        out.append(ln)
    return "".join(out)


def version(rule: str, ext: str) -> str:
    """The name `surfacefit` reads out of the manifest.

    `+stable-order` describes the whole inlined rule, so BOTH halves have to order
    their points stably: this experiment's `all_segments` by its sort call, and the
    pilot's `segments` by its `STABLE` flag (added in exp-043's 09998b5; before that
    the pilot had no flag and sorted unstably).
    """
    import re
    req = re.search(r"^REQUIRE_ALT\s*=\s*(True|False)", rule, re.M)
    ver = "repair-2026-09-08" if (req and req.group(1) == "True") else "shipped-2026-09-07"
    stb = re.search(r"^STABLE\s*=\s*(True|False)", rule, re.M)
    if 'sort_values("t", kind="stable")' in ext and stb and stb.group(1) == "True":
        ver += "+stable-order"                     # one answer on every machine
    return ver
