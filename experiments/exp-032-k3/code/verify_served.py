"""exp-027's served-candidate verification, run for this experiment's files.

The nine checks are exp-027's -- template order, finiteness, the floor, the 388
`noac` and 38,182 EHAM rows bit identical to v7 and to stack-5, no held row
moved, and a rebuild differing by exactly 0.0. Reusing that file rather than
copying it is the point: the K3 candidate is then held to the same bar as v9,
by the same code, and a change to the bar reaches both.

    python code/verify_served.py --name exp-032-k3 --rebuild <path>

`--rebuild` is REQUIRED here (Sol round 1, blocking 4): without it the
imported verifier sets `all_passed` from eight checks, omitting the ninth
-- the rebuild at max abs difference 0.0 that the assignment names.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-027-step/code"))
import verify_served as V  # noqa: E402

if __name__ == "__main__":
    # Sol round 1, blocking 4: exp-027's verifier sets `all_passed` from eight
    # checks when `--rebuild` is absent, so a candidate could read "verified"
    # without the rebuild-at-0.0 check that `assignments/k3.md` requires. This
    # wrapper refuses to run without it rather than edit another experiment's
    # committed code.
    if "--rebuild" not in sys.argv:
        raise SystemExit(
            "refusing to run without --rebuild: the ninth check (rebuild the "
            "candidate from its own record at max abs difference 0.0) is the "
            "one the assignment names, and the imported verifier would "
            "otherwise report all_passed from eight checks")
    V.EXP = EXP          # the record lands in THIS experiment's analysis/
    V.main()
