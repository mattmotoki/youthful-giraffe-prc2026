"""Every GPU kernel id another experiment owns, read from its committed metadata.

    python gpu_peers.py [<experiment dir> ...]

`take_gpu_slot.sh` counts Kaggle's two GPU slots before it pushes anything, and
the ids it counts must come from the record rather than from a guess: the desk's
rule is that this experiment may use only a slot k2climb's sessions leave free
and must never stop theirs, so a missed id would mean taking a slot that is not
free. Defaults to exp-035-k2climb, which is the experiment holding them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = (ROOT / "experiments" / "exp-035-k2climb",)


def ids(dirs) -> list[str]:
    out = []
    for d in dirs:
        for f in sorted(Path(d).glob("kernels/*/kernel-metadata.json")):
            try:
                m = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            if m.get("enable_gpu") and m.get("id"):
                out.append(m["id"])
    return sorted(set(out))


if __name__ == "__main__":
    got = ids([Path(a) for a in sys.argv[1:]] or DEFAULT)
    if not got:
        raise SystemExit("no GPU kernel metadata found")
    print("\n".join(got))
