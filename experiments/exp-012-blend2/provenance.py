"""One line naming every module and input a screen depended on.

Adopted from retro-v1's climb-line provenance block (2026-09-06), after
`prc.surface_features.BUCKET` changed under them from 1800 to 1200 with no
signal. This lane's MEMBER inputs are hash-pinned by `pin_members.py`, so a
changed input is detectable — but nothing recorded the version of the CODE doing
the screening, and several screens here ran while other lanes moved modules
underneath them. Three modules moved during this session alone:
`prc.tail_model` 1.1.0 -> 2.4.0, `prc.surface_features` 2 -> 3, and
`prc.nmtail` 2 -> 4 (that last one by me).

A drift then shows up in the row rather than as a coincidence, which is how
retro-v1's 171-vs-175 column confound was eventually caught — by a number
appearing twice when it should not have.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODULES = ("blend", "data", "nmtail", "tail_model", "day_state_features",
           "phi_features", "lirf_features", "surface_features", "stand_features")
#: constants a consumer's result silently depends on, beyond the VERSION
CONSTANTS = {"surface_features": ("BUCKET", "UNIMPEDED_Q"),
             "day_state_features": ("DEICE_THRESHOLD", "MIN_DEPARTURES"),
             "blend": ("LATE_CUT", "FLOOR"),
             }


def stamp(inputs: dict | None = None) -> str:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    parts = []
    for name in MODULES:
        try:
            m = __import__(f"prc.{name}", fromlist=["x"])
        except Exception:
            parts.append(f"{name}[import failed]"); continue
        v = getattr(m, "VERSION", "?")
        extra = ",".join(f"{c}={getattr(m, c)}" for c in CONSTANTS.get(name, ())
                         if hasattr(m, c))
        parts.append(f"{name}[{v}{',' + extra if extra else ''}]")
    line = "mods=" + " ".join(parts) + f" py={platform.python_version()}"
    if inputs:
        h = []
        for k, p in sorted(inputs.items()):
            q = ROOT / p
            d = hashlib.sha256(q.read_bytes()).hexdigest()[:12] if q.exists() else "MISSING"
            h.append(f"{k}={d}")
        line += " in=" + " ".join(h)
    return line


if __name__ == "__main__":
    print(stamp())
