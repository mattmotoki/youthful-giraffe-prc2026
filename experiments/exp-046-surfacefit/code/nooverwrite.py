"""RESULTS section 47.71.6 -- the desk's structural fix: a run REFUSES to write over an
existing analysis/ or work/ artifact.

47.71's launch 1 silently overwrote 47.59.6's artifact because the output name was built
from --only-month alone and a --folds subset of the same month mapped to the same path.
The measurement survived in the RESULTS record; its durable address did not.  A naming
defect is invisible until it destroys something, so this turns it into an ERROR at the
moment it would first do harm: fail loud, and replace only when --overwrite says the
replacement is the intent.

Called TWICE per run: once right after argument parsing, so a refused run costs no
compute, and once immediately before each write, so a file created while the run was
fitting (another session, a second launch) is caught too.
"""
from __future__ import annotations

from pathlib import Path


def refuse_existing(paths, overwrite: bool, flag: str = "--overwrite") -> None:
    """Raise SystemExit naming every path in `paths` that exists, unless `overwrite`."""
    hit = [Path(p) for p in paths if Path(p).exists()]
    if hit and not overwrite:
        raise SystemExit(
            "REFUSING TO OVERWRITE -- these artifacts already exist:\n  "
            + "\n  ".join(str(p) for p in hit)
            + "\nA run that would replace a measured artifact is a naming defect until "
              "shown otherwise (RESULTS 47.71.6).  Pass " + flag + " only if replacing "
              "it is the intent.")
