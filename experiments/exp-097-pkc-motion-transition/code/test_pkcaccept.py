"""exp-097 desk-acceptance tests.  No real receipt, source, label or prediction is read.

Every test builds a throw-away Git repository, writes a receipt into it and drives
`pkcaccept.verify` against it.  The point is that the door fails CLOSED: the owning worker
cannot open it, a receipt that is not committed cannot open it, and a receipt that already
existed at the policy commit cannot open it.

    OMP_NUM_THREADS=2 .venv/bin/python \
        experiments/exp-097-pkc-motion-transition/code/test_pkcaccept.py
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pkcaccept as A                                                  # noqa: E402

FAILED: list[str] = []
REL = "experiments/exp-097-pkc-motion-transition"


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr}")
    return out.stdout.strip()


def commit(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def scaffold(tmp: Path) -> tuple[Path, str, str]:
    """A repository with a registration commit and a policy continuation commit."""
    repo = tmp / "repo"
    (repo / REL / "acceptance").mkdir(parents=True)
    git(repo.parent, "init", "-q", "repo") if False else subprocess.run(
        ["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / REL / "SPEC.md").write_text("registration\n")
    registration = commit(repo, "registration")
    (repo / REL / "SPEC.md").write_text("registration + policy continuation\n")
    policy = commit(repo, "policy continuation")
    return repo, registration, policy


PHASE = 2


def good(policy: str) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    return {
        "schema": A.SCHEMA,
        "registration_commit": A.REGISTRATION_COMMIT,
        "policy_commit": policy,
        "authorizes": A.PHASES[PHASE]["authorizes"],
        "run_kind": A.PHASES[PHASE]["run_kind"],
        "single_use": True,
        "issuer": "desk",
        "issued_by": "desk",
        "run_id": "exp097-label-bearing-1",
        "issued_at_utc": (now - dt.timedelta(hours=1)).isoformat(),
        "expires_at_utc": (now + dt.timedelta(hours=6)).isoformat(),
    }


def write_receipt(repo: Path, receipt: dict | str) -> Path:
    p = repo / REL / A.PHASES[PHASE]["receipt"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(receipt if isinstance(receipt, str) else json.dumps(receipt, indent=2))
    return p


def run(repo: Path, registration: str) -> dict:
    """`verify` against a scaffolded repository, with the real registration substituted."""
    old = A.REGISTRATION_COMMIT
    A.REGISTRATION_COMMIT = registration
    try:
        return A.verify(PHASE, repo=repo, here=repo / REL)
    finally:
        A.REGISTRATION_COMMIT = old


def refuses(name: str, repo: Path, registration: str, why: str = "") -> None:
    try:
        run(repo, registration)
        check(name, False, "verify returned a record")
    except A.AcceptanceError as exc:
        check(name, True, str(exc)[:70])


def test_accepts_a_desk_receipt() -> None:
    print("acceptance: a desk receipt committed after the policy commit opens the door")
    with tempfile.TemporaryDirectory() as t:
        repo, reg, pol = scaffold(Path(t))
        r = good(pol)
        r["registration_commit"] = reg
        write_receipt(repo, r)
        source = commit(repo, "desk acceptance")
        rec = run(repo, reg)
        check("verify returns a record", isinstance(rec, dict))
        check("it binds the source commit", rec["source_commit"] == source)
        check("it binds the policy commit", rec["policy_commit"] == pol)
        check("it binds the registration", rec["registration_commit"] == reg)
        check("it carries the receipt sha256", len(rec["receipt_sha256"]) == 64)
        check("it carries the committed blob id", len(rec["receipt_blob"]) == 40)
        check("it names the run", rec["run_id"] == "exp097-label-bearing-1")


def test_the_worker_cannot_accept_itself() -> None:
    print("acceptance: the owning worker cannot open its own door")
    for field, value, why in [("issued_by", A.WORKER, "issued by the owning worker"),
                              ("issued_by", "", "issued by nobody"),
                              ("issuer", "motiontransition", "issuer is not the desk"),
                              ("issuer", "worker", "issuer is a worker")]:
        with tempfile.TemporaryDirectory() as t:
            repo, reg, pol = scaffold(Path(t))
            r = good(pol)
            r["registration_commit"] = reg
            r[field] = value
            write_receipt(repo, r)
            commit(repo, "receipt")
            refuses(f"refuses a receipt {why}", repo, reg)


def test_it_must_be_committed_after_the_policy_commit() -> None:
    print("acceptance: the receipt must be a desk commit made after the continuation")
    with tempfile.TemporaryDirectory() as t:
        repo, reg, pol = scaffold(Path(t))
        r = good(pol)
        r["registration_commit"] = reg
        write_receipt(repo, r)
        refuses("refuses a receipt that is not committed at all", repo, reg)
    with tempfile.TemporaryDirectory() as t:
        # the worker ships the receipt inside the policy commit itself
        repo = Path(t) / "repo"
        (repo / REL / "acceptance").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
        (repo / REL / "SPEC.md").write_text("registration\n")
        reg = commit(repo, "registration")
        r = good(reg)
        r["registration_commit"] = reg
        write_receipt(repo, r)
        pol = commit(repo, "policy continuation WITH the receipt in it")
        r["policy_commit"] = pol
        write_receipt(repo, r)
        commit(repo, "source")
        refuses("refuses a receipt that already existed at the policy commit", repo, reg)


def test_the_bytes_must_match_the_commit() -> None:
    print("acceptance: the receipt on disk must be the receipt that was committed")
    with tempfile.TemporaryDirectory() as t:
        repo, reg, pol = scaffold(Path(t))
        r = good(pol)
        r["registration_commit"] = reg
        write_receipt(repo, r)
        commit(repo, "desk acceptance")
        tampered = copy.deepcopy(r)
        tampered["run_id"] = "a second run"
        write_receipt(repo, tampered)
        refuses("refuses an edited receipt", repo, reg)


def test_field_and_time_contract() -> None:
    print("acceptance: every registered field and both times are enforced")
    now = dt.datetime.now(dt.timezone.utc)
    cases = [
        ("schema", "exp-100-desk-acceptance/1", "another experiment's schema"),
        ("authorizes", "exp-097 phase 3 candidate", "another phase's authorization"),
        ("run_kind", "label-free", "a different run kind"),
        ("single_use", False, "a receipt that is not single use"),
        ("run_id", "  ", "a receipt with no run id"),
        ("issued_at_utc", (now + dt.timedelta(hours=2)).isoformat(), "a future issue time"),
        ("expires_at_utc", (now - dt.timedelta(hours=1)).isoformat(), "an expired receipt"),
        ("issued_at_utc", "2026-09-16 05:30", "a time with no UTC offset"),
        ("policy_commit", "not-a-commit", "a policy commit that is not a commit id"),
        ("registration_commit", "0" * 40, "a foreign registration"),
    ]
    for field, value, why in cases:
        with tempfile.TemporaryDirectory() as t:
            repo, reg, pol = scaffold(Path(t))
            r = good(pol)
            r["registration_commit"] = reg
            r[field] = value
            write_receipt(repo, r)
            commit(repo, "receipt")
            refuses(f"refuses {why}", repo, reg)
    for missing in A.REQUIRED_FIELDS:
        with tempfile.TemporaryDirectory() as t:
            repo, reg, pol = scaffold(Path(t))
            r = good(pol)
            r["registration_commit"] = reg
            r.pop(missing)
            write_receipt(repo, r)
            commit(repo, "receipt")
            refuses(f"refuses a receipt with no {missing}", repo, reg)


def test_malformed_receipt() -> None:
    print("acceptance: a receipt that is not a JSON object, or is too big, is refused")
    for body, why in [("not json at all", "unreadable JSON"),
                      ("[1, 2, 3]", "a JSON array"),
                      ('{"schema": "' + "x" * 9000 + '"}', "a receipt over the size cap")]:
        with tempfile.TemporaryDirectory() as t:
            repo, reg, pol = scaffold(Path(t))
            write_receipt(repo, body)
            commit(repo, "receipt")
            refuses(f"refuses {why}", repo, reg)


def test_no_file_in_the_package_writes_the_receipt() -> None:
    print("acceptance: nothing in exp-097 can write its own acceptance")
    pkg = HERE.parent
    writers = []
    for f in sorted(pkg.rglob("*.py")):
        if f.name == "test_pkcaccept.py":
            continue
        text = f.read_text()
        if "desk-acceptance.json" in text and (
                "write_text" in text or "to_parquet" in text or "json.dump" in text):
            # a mention is fine; a mention next to a writer is not
            for line in text.splitlines():
                if "desk-acceptance" in line and ("write" in line or "dump" in line):
                    writers.append(f"{f.relative_to(pkg)}: {line.strip()[:60]}")
    check("no module writes the receipt path", not writers, "; ".join(writers))
    for f in sorted(pkg.rglob("*.sh")):
        check(f"{f.name} does not stage or write the receipt",
              "desk-acceptance" not in f.read_text())


def main() -> int:
    for t in (test_accepts_a_desk_receipt, test_the_worker_cannot_accept_itself,
              test_it_must_be_committed_after_the_policy_commit,
              test_the_bytes_must_match_the_commit, test_field_and_time_contract,
              test_malformed_receipt, test_no_file_in_the_package_writes_the_receipt):
        t()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("all desk-acceptance tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
