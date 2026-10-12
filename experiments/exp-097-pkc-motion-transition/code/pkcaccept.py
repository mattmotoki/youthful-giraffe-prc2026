"""exp-097 external desk acceptance: the only door the label-bearing phase can open.

The 2026-09-16 05:30 UTC review policy ends `gpt-5.6-sol` reviews of registrations,
repairs and progress rows in this repository.  exp-097's registered pre-run acceptance was
one fresh Sol xhigh verdict (SPEC.md, assignment item 4).  The round-2 pass was written and
**never sent** -- `review/round2-stopped.md` records the stop -- so no verdict exists and
none can be obtained.  A review the desk stops before a verdict is not a review result and
satisfies no review gate.

In its place phase 2 and phase 3 open only against a receipt the **desk** issues outside
this package, bound to the frozen scientific registration commit and to the policy
continuation commit, and naming the exact run it authorizes.  This replaces the review
requirement and nothing else: every registered population, estimator, control, gate,
source, single-use claim, launcher and write-once check stands exactly as registered.

A run cannot accept itself.  This module only reads: it never writes the receipt, and no
other file in this package names it for writing.  `verify` also proves the receipt path was
**absent** at the policy continuation commit and is tracked at the commit the run executes,
so the worker cannot ship acceptance inside its own commit, and the bytes on disk must equal
the committed bytes.

The schema is exp-100's, deliberately: one desk procedure issues both, and
`experiments/exp-100-pkb-causal-state/acceptance/README.md` is the worked example.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../code
EXP = HERE.parent                               # the experiment
REPO = HERE.parents[2]

#: Registered constants (SPEC.md §12, "Desk acceptance receipt").
SCHEMA = "exp-097-desk-acceptance/1"
REGISTRATION_COMMIT = "ae27ddf9edc5211543dd30505478e434056462aa"
WORKER = "motiontransition"

#: One receipt per phase.  Phase-2 acceptance CANNOT open phase 3: the candidate needs a
#: separate, later desk direction, so it has its own path, its own authorization string and
#: its own run kind (assignment item 2).
PHASES = {
    2: {"receipt": "acceptance/desk-acceptance-phase2.json",
        "authorizes": "exp-097 phase 2 label-bearing run",
        "run_kind": "historical"},
    3: {"receipt": "acceptance/desk-acceptance-phase3.json",
        "authorizes": "exp-097 phase 3 candidate",
        "run_kind": "candidate"},
}
MAX_RECEIPT_BYTES = 8192
REQUIRED_FIELDS = ("schema", "registration_commit", "policy_commit", "authorizes",
                   "run_kind", "single_use", "issuer", "issued_by", "issued_at_utc",
                   "expires_at_utc", "run_id")
HEX = frozenset("0123456789abcdef")


class AcceptanceError(RuntimeError):
    """No valid external desk receipt authorizes this exact label-bearing run."""


def _env() -> dict:
    """Git environment with any inherited repository redirection removed.

    A stray ``GIT_DIR`` or ``GIT_INDEX_FILE`` in the launch environment would otherwise
    point these checks at a different repository than the one being run.
    """
    env = dict(os.environ)
    for var in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY",
                "GIT_COMMON_DIR"):
        env.pop(var, None)
    return env


def _git(repo, *args, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          env=_env())
    if check and proc.returncode != 0:
        raise AcceptanceError(f"exp-097 cannot run `git {' '.join(args)}`: "
                              f"{proc.stderr.strip() or proc.returncode}")
    return proc


def head_commit(repo=None) -> str:
    """The commit the run executes.  Fails closed rather than running unbound."""
    repo = REPO if repo is None else Path(repo)
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    if len(commit) != 40 or not set(commit) <= HEX:
        raise AcceptanceError(f"exp-097 cannot read its source commit; git returned {commit!r}")
    return commit


def _is_commit(repo, sha: str) -> bool:
    return _git(repo, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0


def _is_ancestor(repo, older: str, newer: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", older, newer,
                check=False).returncode == 0


def _tracked_blob(repo, commit: str, rel: str):
    """The blob id of `rel` at `commit`, or None when the path is not tracked there."""
    proc = _git(repo, "rev-parse", f"{commit}:{rel}", check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _parse_utc(value, *, field: str) -> dt.datetime:
    text = str(value).strip()
    if text.endswith("Z"):  # Python 3.10 does not read the military suffix
        text = text[:-1] + "+00:00"
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise AcceptanceError(
            f"exp-097 desk receipt {field} is not an ISO-8601 time: {value!r}") from exc
    if stamp.tzinfo is None:
        raise AcceptanceError(f"exp-097 desk receipt {field} carries no UTC offset: {value!r}")
    return stamp.astimezone(dt.timezone.utc)


def _hex40(value, *, field: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 40 or not set(text) <= HEX:
        raise AcceptanceError(f"exp-097 desk receipt {field} is not a full commit id: {value!r}")
    return text


def verify(phase: int, *, repo=None, here=None, source_commit=None, now=None,
           bound_files=None) -> dict:
    """Return the acceptance record, or raise `AcceptanceError` and open nothing.

    Every check reads the receipt file and Git objects only.  No source, label, residual,
    prediction or ranking value is opened on either path.
    """
    if phase not in PHASES:
        raise AcceptanceError(f"exp-097 has no desk acceptance for phase {phase!r}")
    spec = PHASES[phase]
    authorizes, run_kind = spec["authorizes"], spec["run_kind"]
    repo = REPO if repo is None else Path(repo)
    here = EXP if here is None else Path(here)
    path = here / spec["receipt"]
    rel = path.relative_to(repo).as_posix()

    if not path.is_file():
        raise AcceptanceError(
            f"exp-097 has no desk acceptance receipt at {rel} for phase {phase}; only the "
            f"desk can authorize "
            f"the label-bearing run, and this registration carries no Sol verdict and none can be obtained")
    size = path.stat().st_size
    if size > MAX_RECEIPT_BYTES:
        raise AcceptanceError(f"exp-097 desk receipt is {size} bytes, over the registered "
                              f"cap of {MAX_RECEIPT_BYTES}")
    raw = path.read_bytes()
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"exp-097 desk receipt is not readable JSON: {exc}") from exc
    if not isinstance(receipt, dict):
        raise AcceptanceError("exp-097 desk receipt is not a JSON object")
    missing = [f for f in REQUIRED_FIELDS if f not in receipt]
    if missing:
        raise AcceptanceError(f"exp-097 desk receipt is missing {', '.join(missing)}")

    if receipt["schema"] != SCHEMA:
        raise AcceptanceError(f"exp-097 desk receipt schema is {receipt['schema']!r}, "
                              f"registered {SCHEMA!r}")
    registration = _hex40(receipt["registration_commit"], field="registration_commit")
    if registration != REGISTRATION_COMMIT:
        raise AcceptanceError(
            f"exp-097 desk receipt accepts registration {registration}, but this package is "
            f"the frozen registration {REGISTRATION_COMMIT}")
    if receipt["authorizes"] != authorizes:
        raise AcceptanceError(f"exp-097 desk receipt authorizes {receipt['authorizes']!r}, "
                              f"not {authorizes!r}; phase-2 acceptance cannot open phase 3")
    if receipt["run_kind"] != run_kind:
        raise AcceptanceError(f"exp-097 desk receipt names run kind {receipt['run_kind']!r}, "
                              f"phase {phase} is registered {run_kind!r}")
    if receipt["single_use"] is not True:
        raise AcceptanceError("exp-097 desk receipt is not marked single_use")
    if receipt["issuer"] != "desk":
        raise AcceptanceError(f"exp-097 desk receipt issuer is {receipt['issuer']!r}; only "
                              f"the desk can accept this run")
    issued_by = str(receipt["issued_by"]).strip()
    if not issued_by or issued_by == WORKER:
        raise AcceptanceError(
            f"exp-097 desk receipt is issued by {issued_by!r}; the owning worker cannot "
            f"accept its own run")
    run_id = str(receipt["run_id"]).strip()
    if not run_id:
        raise AcceptanceError("exp-097 desk receipt names no run_id")

    policy = _hex40(receipt["policy_commit"], field="policy_commit")
    source = head_commit(repo) if source_commit is None else _hex40(source_commit,
                                                                    field="source commit")
    for field, sha in (("registration_commit", registration), ("policy_commit", policy),
                       ("source commit", source)):
        if not _is_commit(repo, sha):
            raise AcceptanceError(f"exp-097 desk receipt {field} {sha} is not a commit in "
                                  f"this repository")
    if not _is_ancestor(repo, registration, policy):
        raise AcceptanceError(
            f"exp-097 policy commit {policy[:12]} does not build on the frozen registration "
            f"{registration[:12]}")
    if not _is_ancestor(repo, policy, source):
        raise AcceptanceError(
            f"exp-097 source commit {source[:12]} does not descend from the accepted policy "
            f"commit {policy[:12]}")

    tracked = _tracked_blob(repo, source, rel)
    if tracked is None:
        raise AcceptanceError(f"exp-097 desk receipt is not committed at source commit "
                              f"{source[:12]}; acceptance must be part of the record")
    on_disk = _git(repo, "hash-object", str(path)).stdout.strip()
    if tracked != on_disk:
        raise AcceptanceError("exp-097 desk receipt on disk differs from the receipt "
                              "committed at the source commit")
    if _tracked_blob(repo, policy, rel) is not None:
        raise AcceptanceError(
            "exp-097 desk receipt already exists at the policy continuation commit; the desk "
            "issues acceptance after that commit, so a worker cannot ship its own")

    # A later desk commit may add ONLY the receipt to the accepted package tree
    # (assignment item 2): every bound file must be byte-identical at the policy commit and
    # at the commit the run executes.
    changed = []
    for rel_file in sorted(bound_files or ()):
        at_policy = _tracked_blob(repo, policy, rel_file)
        at_source = _tracked_blob(repo, source, rel_file)
        if at_policy is None:
            changed.append(f"{rel_file} (absent at the policy commit)")
        elif at_policy != at_source:
            changed.append(rel_file)
    if changed:
        raise AcceptanceError(
            "exp-097 package changed after acceptance; the desk may add only the receipt. "
            f"Changed: {', '.join(changed[:6])}"
            + (f" and {len(changed) - 6} more" if len(changed) > 6 else ""))

    now = dt.datetime.now(dt.timezone.utc) if now is None else now
    issued = _parse_utc(receipt["issued_at_utc"], field="issued_at_utc")
    expires = _parse_utc(receipt["expires_at_utc"], field="expires_at_utc")
    if issued > now:
        raise AcceptanceError(f"exp-097 desk receipt is issued in the future "
                              f"({issued.isoformat()})")
    if expires <= issued:
        raise AcceptanceError("exp-097 desk receipt expires at or before it is issued")
    if expires <= now:
        raise AcceptanceError(f"exp-097 desk receipt expired at {expires.isoformat()}")

    return {
        "receipt_path": rel,
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_blob": tracked,
        "registration_commit": registration,
        "policy_commit": policy,
        "source_commit": source,
        "run_id": run_id,
        "phase": phase,
        "run_kind": run_kind,
        "authorizes": authorizes,
        "bound_files_checked": len(bound_files or ()),
        "issuer": receipt["issuer"],
        "issued_by": issued_by,
        "issued_at_utc": issued.isoformat(),
        "expires_at_utc": expires.isoformat(),
        "verified_at_utc": now.isoformat(),
    }
