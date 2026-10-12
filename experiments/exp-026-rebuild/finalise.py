"""Turn every staged prediction vector into a submission candidate.

A staged vector (`candidates/<name>.parquet`, `MVT_ID_mvt` + `y_pred`) is the
blend arithmetic finished; it is not a submission, because a submission must
carry the template's exact id set in the template's order, and the template
lives in `data/raw/submitting.parquet`.  This script closes that last step
through `prc.data.write_submission`, which is the only function allowed to
write a candidate.

For a config-built vector it also rebuilds the same candidate through
`prc.blend.build_candidate` -- the lane's own end-to-end path -- and refuses to
issue anything whose values differ, so the split between `rebuild_blend.py` and
this file cannot silently drift from the code it stands in for.

**Nothing reaches `submissions/candidates/` that `code/verify.py` has not
passed on the work file.**  The candidate is written beside the experiment,
verified there, and moved into place only on success.

**Stated limitation: the last window is not closed here, by decision.** Between
the hash that confirms the verified bytes and the `os.replace` that publishes
them there is an interval in which another process could substitute the work
file. It is narrowed as far as a path-based check can be -- the work name is
unique per run, the hash is taken before the move, and a destination that does
not hash to the verified value is removed and the previous candidate restored --
but it cannot be closed by adding another check of the same kind, because every
such check is itself a read followed by an act on a path. Closing it needs
file-descriptor-based verification (`O_TMPFILE` + `linkat`), which the parquet
readers here do not offer.

**Where the risk is actually closed:** the desk re-verifies every upload by
SHA-256 and a tolerance diff against the base before it leaves the machine. That
check runs on the bytes that are about to be submitted, after any such window has
passed, so it is the binding one. This gate's job is to catch operator error --
a stale staged file, the wrong variant, a config typo, a NaN -- and for that it
is sound. It is not an adversarial boundary and does not claim to be
(desk, 2026-09-07; Sol round 4 finding 1).

Round 1 finding 4 was that the candidate was written first and the check raised
afterwards.  Round 2 found that fix incomplete, and it was: the check ran only
when a config matched the stage name, and the file was moved unconditionally
otherwise -- which is exactly stack-5, the one candidate queued for upload, so
it reached the upload directory with `cross_checked: false`.  A candidate with
no verifier is now REFUSED, not published, and there is no bypass flag:
`--dry-run` stages and verifies into `work/` without publishing, and that is
the only way to skip the move.

    .venv/bin/python experiments/exp-026-rebuild/finalise.py            # all staged
    .venv/bin/python experiments/exp-026-rebuild/finalise.py --name blend-5
"""
from __future__ import annotations

import argparse
import json
import importlib.util
import os
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
import frames as F  # noqa: E402
# The repository root goes at the END: `code/` must stay ahead of it so
# a module at the root cannot shadow one of this experiment's own
# (round 4 -- a root-level `verify.py` would have replaced the gate).
sys.path.append(str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

# The gate is loaded from ITS OWN PATH, not by name. Round 4: the repository
# root was inserted ahead of `code/` before `import verify`, so a `verify.py`
# at the root would shadow the gate and could approve anything. A gate found by
# search is a gate that can be replaced by putting a file somewhere.
_spec = importlib.util.spec_from_file_location(
    "exp026_verify", HERE / "code" / "verify.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

#: A candidate name is used to build a destination path, so it is checked
#: against a whitelist rather than trusted. Round 4: `x/../exp-026-stack-5`
#: dispatched as a generic blend (no reserved match on the literal string) while
#: resolving to the reserved stack-5 destination.
NAME_OK = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

STAGED = HERE / "candidates"
CONFIGS = HERE / "configs"
DEST = F.ROOT / "submissions" / "candidates"
#: Candidates are built here and moved into DEST only after verification.
#: The FILE NAME inside it is unique per run: round 3 found that a single
#: predictable work path lets a second run of the same name replace the file
#: between the check and the move, and the move happened before the re-hash, so
#: the swapped file would already be in the destination when the hash failed.
WORK = HERE / "work"


def finalise(name: str, publish: bool = True) -> dict:
    if not NAME_OK.match(name) or ".." in name:
        raise SystemExit(f"{name!r} is not a candidate name: it must match "
                         f"{NAME_OK.pattern} and carry no path segment")
    src = STAGED / f"{name}.parquet"
    v = pd.read_parquet(src)
    pred = pd.Series(v["y_pred"].to_numpy("float64"), index=v["MVT_ID_mvt"].to_numpy("float64"))
    out = DEST / f"exp-026-{name}.parquet"
    if out.resolve().parent != DEST.resolve():
        raise SystemExit(f"{name!r} resolves outside {DEST}")
    WORK.mkdir(parents=True, exist_ok=True)
    work = WORK / f"exp-026-{name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.parquet"
    # The gate runs on the WORK file, so a failure cannot leave anything in
    # submissions/candidates/ -- there is nothing there to leave. Round 3: the
    # cleanup used to run only for V.Failed, so the claim that a failed run
    # leaves no work file was false for every other exception. Round 4: the
    # write itself sat OUTSIDE the try, so a failure during the write or the
    # record left the partial work file behind and the claim was still false.
    try:
        sha = D.write_submission(pred, work)
        rec = {"staged": str(src.relative_to(F.ROOT)),
               "candidate": str(out.relative_to(F.ROOT)),
               "sha256": sha, "n": int(len(v)),
               "min": float(v["y_pred"].min()), "median": float(v["y_pred"].median()),
               "p99": float(v["y_pred"].quantile(0.99)), "max": float(v["y_pred"].max()),
               "n_below_floor": int((v["y_pred"] < B.FLOOR - 1e-9).sum()),
               "n_nonfinite": int((~np.isfinite(v["y_pred"])).sum())}
        rec.update(V.verify(name, work))
    except V.Failed as e:
        work.unlink(missing_ok=True)
        raise SystemExit(f"{name}: VERIFICATION FAILED -- {e}\n"
                         f"{out.relative_to(F.ROOT)} was NOT written and any earlier "
                         f"file there is untouched.")
    except BaseException:
        work.unlink(missing_ok=True)
        raise
    rec["sha256"] = sha
    if not publish:
        rec["published"] = False
        rec["file"] = str(work.relative_to(F.ROOT))
        return rec
    # Verified. Confirm the bytes are still the ones that were verified BEFORE
    # anything reaches the destination -- the check has to precede the move, not
    # follow it, or a swapped file is already published when the hash fails.
    again = D.sha256(work)
    if again != sha:
        work.unlink(missing_ok=True)
        raise SystemExit(f"{name}: the work file changed between the check and the "
                         f"move ({sha[:12]} -> {again[:12]}); "
                         f"{out.relative_to(F.ROOT)} was NOT written")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Round 4: the post-move hash detected a swapped file but LEFT IT PUBLISHED,
    # which is the one outcome this whole function exists to prevent. The
    # previous bytes are captured first, and a destination that does not hash to
    # the verified value is removed and the previous file put back, so a failure
    # can only ever restore the state before the run.
    prev = out.read_bytes() if out.exists() else None
    work.replace(out)
    final = D.sha256(out)
    if final != sha:
        out.unlink(missing_ok=True)
        if prev is not None:
            out.write_bytes(prev)
        raise SystemExit(
            f"{name}: {out.relative_to(F.ROOT)} hashed {final[:12]}, not the verified "
            f"{sha[:12]}, so the file was REMOVED and "
            f"{'the previous candidate restored' if prev else 'nothing was published'}. "
            f"Do not upload anything from this run.")
    rec["published"] = True
    rec["file"] = str(out.relative_to(F.ROOT))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="one staged vector; default every one")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage and verify into work/ without publishing. This is the "
                         "only way to skip the move; there is no way to skip the check.")
    a = ap.parse_args()
    names = [a.name] if a.name else sorted(p.stem for p in STAGED.glob("*.parquet")
                                           if p.stem != "ranking-gates")
    if not names:
        raise SystemExit("nothing staged in candidates/; run code/rebuild_blend.py first")
    recs = [finalise(n, publish=not a.dry_run) for n in names]
    (HERE / "analysis" / "finalised.json").write_text(json.dumps(recs, indent=2))
    print(json.dumps(recs, indent=2))


if __name__ == "__main__":
    main()
