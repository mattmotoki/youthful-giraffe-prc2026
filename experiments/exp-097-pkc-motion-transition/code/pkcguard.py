"""exp-097 execution boundary: what must be true before a label can be opened.

Three barriers, in this order, none of which opens a source:

1. `predecessor_chain()` — phase 2 rests on the label-free screen of record, which is
   revision **r2** and nothing else.  The unsuffixed phase-1 receipt is the run that reached
   real target values through `mask_like_ranking` (review round 1, blocking 1); it is named
   here so it can be refused by name, and every other revision is refused as well.  The whole
   r2 chain — result, receipt, claim, resource, launch, exit and verify — is required, with
   its digests, and the digests travel into the phase-2 receipts.

2. `require_clean_pushed_branch()` — the run executes on the registered branch, with a clean
   tree and index, at a commit that is already on the remote.  A result whose source is not
   pushed has no durable address.

3. `start_barrier()` — the controller runs only from its launcher.  The launcher mints a
   one-time handoff; the controller checks the token, then checks the things it can observe
   about ITSELF: its CPU affinity, its nice level and its four thread caps.  A direct call has
   no handoff and is refused **before** the single phase-2 claim is taken, so it cannot burn
   the one read.

`durable_failure()` writes the INCOMPLETE receipt that a post-claim failure must leave.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[2]

BRANCH = "exp097-pkc-motion-transition"
HANDOFF_ENV = "EXP097_HANDOFF"
TOKEN_ENV = "EXP097_HANDOFF_TOKEN"
CPU_SET = frozenset(range(63, 72))
NICE = 19
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")
THREADS = "8"

#: The label-free screen of record.  Phase 2 accepts this revision and no other.
PHASE1_REVISION = "r2"
PHASE1_CHAIN = {
    "result": "analysis/screen-r2.json",
    "receipt": "analysis/receipt-phase1-r2.json",
    "claim": "analysis/claim-phase1-r2.json",
    "resource": "analysis/resource-phase1-r2.json",
    "launch": "logs/phase1-r2.launch.json",
    "exit": "logs/phase1-r2.exit.json",
    "verify": "logs/phase1-r2.verify.json",
}
#: The invalid first run, named so it can be refused by name and never by accident.
PHASE1_INVALID = {
    "result": "analysis/screen.json",
    "receipt": "analysis/receipt-phase1.json",
    "resource": "analysis/resource-phase1.json",
}


class BoundaryError(RuntimeError):
    """The execution boundary is not satisfied.  Nothing has been opened."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE",
                        "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR")}
    proc = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          env=env)
    if check and proc.returncode != 0:
        raise BoundaryError(f"exp-097 cannot run `git {' '.join(args)}`: "
                            f"{proc.stderr.strip() or proc.returncode}")
    return proc


def _load(path: Path, what: str) -> dict:
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundaryError(f"exp-097 cannot read the phase-1 {what} at "
                            f"{path.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(obj, dict):
        raise BoundaryError(f"exp-097 phase-1 {what} is not a JSON object")
    return obj


def predecessor_chain(exp=None) -> dict:
    """The phase-1 r2 chain, with digests, or `BoundaryError` and nothing opened."""
    exp = EXP if exp is None else Path(exp)
    files, digests = {}, {}
    for what, rel in PHASE1_CHAIN.items():
        p = exp / rel
        if not p.is_file() or p.stat().st_size == 0:
            raise BoundaryError(
                f"exp-097 phase 2 needs the phase-1 {PHASE1_REVISION} {what} at {rel}; it is "
                f"missing or empty.  The label-free screen of record is revision "
                f"{PHASE1_REVISION} and phase 2 rests on nothing else")
        files[what], digests[what] = p, sha256(p)

    receipt = _load(files["receipt"], "receipt")
    result = _load(files["result"], "result")
    if receipt.get("phase") != 1:
        raise BoundaryError(f"exp-097 phase-1 receipt names phase {receipt.get('phase')!r}")
    if receipt.get("revision") != PHASE1_REVISION:
        raise BoundaryError(
            f"exp-097 phase-1 receipt names revision {receipt.get('revision')!r}; phase 2 "
            f"accepts only {PHASE1_REVISION}")
    verdict = (receipt.get("support_condition") or {}).get("verdict")
    if verdict != "PASS":
        raise BoundaryError(f"exp-097 phase-1 support verdict is {verdict!r}, not PASS")
    if digests["receipt"] != digests["result"]:
        raise BoundaryError(
            "exp-097 phase-1 receipt and result differ; the controller writes the same record "
            "to both, so a difference means one of them was edited")
    if result.get("revision") != PHASE1_REVISION:
        raise BoundaryError(f"exp-097 phase-1 result names revision {result.get('revision')!r}")

    exitrec = _load(files["exit"], "exit receipt")
    if exitrec.get("exit_code") != 0:
        raise BoundaryError(f"exp-097 phase-1 {PHASE1_REVISION} exited "
                            f"{exitrec.get('exit_code')!r}, not 0")
    launch = _load(files["launch"], "launch record")
    for field, want in (("phase", 1), ("cpu_set", "63-71"), ("nice", NICE), ("threads", 8),
                        ("gpu", "none")):
        if launch.get(field) != want:
            raise BoundaryError(f"exp-097 phase-1 launch record has {field}="
                                f"{launch.get(field)!r}, registered {want!r}")
    if f"--revision={PHASE1_REVISION}" not in str(launch.get("command", "")):
        raise BoundaryError("exp-097 phase-1 launch record does not name revision "
                            f"{PHASE1_REVISION}")
    verify = _load(files["verify"], "verify receipt")
    if len(verify.get("checks") or ()) < 2:
        raise BoundaryError("exp-097 phase-1 verify receipt has fewer than two checks")
    resource = _load(files["resource"], "resource receipt")
    for field, want in (("phase", 1), ("cpu_set", "63-71"), ("nice", NICE), ("threads", 8)):
        if resource.get(field) != want:
            raise BoundaryError(f"exp-097 phase-1 resource receipt has {field}="
                                f"{resource.get(field)!r}, registered {want!r}")
    claim = _load(files["claim"], "claim")
    if claim.get("phase") != 1 or str(claim.get("revision", "")).lstrip("-") != PHASE1_REVISION:
        raise BoundaryError("exp-097 phase-1 claim does not name phase 1 revision "
                            f"{PHASE1_REVISION}")

    # Every phase-1 receipt that is NOT r2 is refused by name, with its digest recorded, so
    # the record shows what was rejected and not merely what was used.
    rejected = {}
    for p in sorted((exp / "analysis").glob("receipt-phase1*.json")):
        rel = p.relative_to(exp).as_posix()
        if rel == PHASE1_CHAIN["receipt"]:
            continue
        why = ("the invalid first run: it reached real target values through "
               "mask_like_ranking (review round 1, blocking 1)"
               if rel == PHASE1_INVALID["receipt"] else
               f"not the screen of record, which is revision {PHASE1_REVISION}")
        rejected[rel] = {"sha256": sha256(p), "refused_because": why}
    for rel in PHASE1_INVALID.values():
        p = exp / rel
        if p.is_file() and rel not in rejected:
            rejected[rel] = {"sha256": sha256(p),
                             "refused_because": "part of the invalid first run"}

    return {
        "revision": PHASE1_REVISION,
        "support_verdict": verdict,
        "total_support": (receipt.get("support_condition") or {}).get("total_support"),
        "source_commit": exitrec.get("commit"),
        "files": {k: {"path": PHASE1_CHAIN[k], "sha256": digests[k],
                      "bytes": files[k].stat().st_size} for k in PHASE1_CHAIN},
        "rejected": rejected,
    }


#: What the launcher itself creates for a run, before the controller can look at the tree.
RUN_ARTIFACT_SUFFIXES = (".log", ".pid", ".launch.json", ".exit.json", ".verify.json",
                         ".verify.json.tmp", ".handoff.json")


def run_artifacts(phase: int, revision: str = "") -> set[str]:
    """The repository-relative paths this run's own launcher writes."""
    suf = f"-{revision.lstrip('-')}" if revision else ""
    base = f"experiments/exp-097-pkc-motion-transition/logs/phase{phase}{suf}"
    return {base + ext for ext in RUN_ARTIFACT_SUFFIXES}


def require_clean_pushed_branch(phase=None, revision: str = "") -> dict:
    """The registered branch, an exact tree and index, and a commit already on the remote.

    "Clean" means the source the run executes is exactly the commit it names: **no tracked
    file differs from HEAD**, staged or unstaged.  Untracked files cannot change that
    source, and one class of them is unavoidable — the launcher writes this run's own
    launch record, log and PID file *before* the controller can look at the tree, so a
    blanket `git status --porcelain` makes every launch refuse itself.  That is exactly how
    the first phase-2 attempt refused at 2026-09-16T06:26:25Z.  So untracked paths are
    refused too, except the exact artifacts of THIS phase and revision, which are listed
    and reported rather than ignored silently.
    """
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != BRANCH:
        raise BoundaryError(f"exp-097 runs on {BRANCH}, not {branch!r}")
    tracked = _git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    if tracked:
        raise BoundaryError("exp-097 refuses to run with a modified tree or index: "
                            + "; ".join(tracked.splitlines()[:5]))
    allowed = run_artifacts(phase, revision) if phase is not None else set()
    untracked = [u for u in _git("ls-files", "--others",
                                 "--exclude-standard").stdout.split("\n") if u]
    foreign = sorted(set(untracked) - allowed)
    if foreign:
        raise BoundaryError(
            "exp-097 refuses to run beside untracked files that are not this run's own: "
            + "; ".join(foreign[:5]))
    head = _git("rev-parse", "HEAD").stdout.strip()
    remote = _git("rev-parse", f"refs/remotes/origin/{BRANCH}", check=False)
    if remote.returncode != 0:
        raise BoundaryError(f"exp-097 cannot see origin/{BRANCH}; the run's source must be "
                            f"pushed before it produces a result")
    if _git("merge-base", "--is-ancestor", head, remote.stdout.strip(),
            check=False).returncode != 0:
        raise BoundaryError(f"exp-097 source commit {head[:12]} is not on "
                            f"origin/{BRANCH}; push it before the run")
    return {"branch": branch, "head": head, "origin": remote.stdout.strip(), "clean": True,
            "tracked_changes": 0,
            "own_untracked_artifacts": sorted(set(untracked) & allowed)}


def _ancestors(pid: int) -> list[int]:
    """The process's own id and its parents, as far as the kernel will say."""
    out = [pid]
    for _ in range(8):
        stat = Path(f"/proc/{out[-1]}/stat")
        if not stat.exists():
            break
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            break
        parent = int(fields[1])
        if parent <= 1 or parent in out:
            break
        out.append(parent)
    if len(out) == 1:
        out.append(os.getppid())
    return out


def start_barrier(phase: int, revision: str, env=None, pid=None) -> dict:
    """The launcher's handoff, and this process's own resource contract.

    Read-only: it takes no claim and opens no source, so a direct controller call is refused
    without consuming the one phase-2 read.
    """
    env = os.environ if env is None else env
    pid = os.getpid() if pid is None else pid
    ref = env.get(HANDOFF_ENV)
    token = env.get(TOKEN_ENV)
    if not ref or not token:
        raise BoundaryError(
            f"exp-097 phase {phase} runs only from its launcher: no {HANDOFF_ENV} handoff in "
            f"the environment.  Use `code/ws_run.sh {phase} <eta> [revision]`.  Nothing was "
            f"claimed and nothing was read")
    path = Path(ref)
    if not path.is_file():
        raise BoundaryError(f"exp-097 handoff {ref} does not exist")
    try:
        hand = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundaryError(f"exp-097 handoff {ref} is not readable JSON: {exc}") from exc
    if hand.get("token") != token:
        raise BoundaryError("exp-097 handoff token does not match the environment; this "
                            "process was not started by the launcher that minted it")
    if int(hand.get("phase", -1)) != int(phase):
        raise BoundaryError(f"exp-097 handoff is for phase {hand.get('phase')!r}, not {phase}")
    if str(hand.get("revision") or "") != str(revision or ""):
        raise BoundaryError(f"exp-097 handoff is for revision {hand.get('revision')!r}, not "
                            f"{revision!r}")
    launcher = hand.get("pid")
    if launcher is not None and int(launcher) not in _ancestors(pid):
        raise BoundaryError(f"exp-097 handoff was minted for process {launcher}, which is not "
                            f"an ancestor of this one ({pid})")

    # what this process can observe about ITSELF
    try:
        affinity = set(os.sched_getaffinity(0))
    except AttributeError:                       # macOS has no affinity mask
        affinity = None
    if affinity is not None and affinity != set(CPU_SET):
        raise BoundaryError(f"exp-097 runs on CPUs 63-71 only; this process is pinned to "
                            f"{sorted(affinity)[:12]}")
    nice = os.nice(0)
    if nice != NICE:
        raise BoundaryError(f"exp-097 runs at nice {NICE}; this process is at {nice}")
    caps = {v: env.get(v) for v in THREAD_VARS}
    wrong = {k: v for k, v in caps.items() if v != THREADS}
    if wrong:
        raise BoundaryError(f"exp-097 caps every numerical thread pool at {THREADS}; "
                            f"{wrong} in this environment")
    return {
        "handoff": str(path),
        "handoff_sha256": sha256(path),
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "launcher_pid": launcher,
        "pid": pid,
        "ppid": os.getppid(),
        "cpu_affinity": sorted(affinity) if affinity is not None else None,
        "nice": nice,
        "threads": caps,
        "minted_utc": hand.get("minted_utc"),
        "commit": hand.get("commit"),
    }


def proc_identity(pid: int) -> str | None:
    """A token that identifies THIS process, not merely the number it was given.

    A PID is reused.  The watcher decides whether the run is still alive, and if it settled
    for `kill -0` it would eventually call some unrelated process the run (assignment item
    3).  The token pins the process's start time — field 22 of `/proc/<pid>/stat` on Linux,
    `ps -o lstart=` on macOS — so a reused number reads as a different process, and a dead
    one reads as `None`.
    """
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            return None
        return f"starttime:{fields[19]}"                # field 22, 1-based, after comm
    proc = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True,
                          text=True)
    out = proc.stdout.strip()
    return f"lstart:{out}" if proc.returncode == 0 and out else None


def process_alive(pid, identity=None) -> bool:
    """True only when `pid` is running AND is still the process `identity` described."""
    if pid is None:
        return False
    now = proc_identity(int(pid))
    if now is None:
        return False
    return True if identity in (None, "") else now == identity


LOG_TAIL_CHARS = 4000


def durable_failure(phase: int, revision: str, reason: str, extra=None, analysis=None,
                    log=None) -> Path:
    """The INCOMPLETE receipt a post-claim failure must leave behind.

    First writer wins: the earliest failure is the most proximate one, and a later writer
    must not paper over it.  When a log is named, its size and last lines are copied into
    the receipt, because the log itself is git-ignored and the receipt is the durable
    record of what went wrong.
    """
    import datetime as dt                                              # noqa: PLC0415
    analysis = (EXP / "analysis") if analysis is None else Path(analysis)
    analysis.mkdir(parents=True, exist_ok=True)
    suf = f"-{revision.lstrip('-')}" if revision else ""
    path = analysis / f"incomplete-phase{phase}{suf}.json"
    if path.exists():
        return path
    rec = {"experiment": "exp-097-pkc-motion-transition", "phase": phase,
           "revision": revision.lstrip("-") or None, "status": "INCOMPLETE",
           "reason": reason, "pid": os.getpid(),
           "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if log:
        lp = Path(log)
        if lp.is_file():
            text = lp.read_text(errors="replace")
            rec["log"] = {"path": str(lp), "bytes": lp.stat().st_size,
                          "tail": text[-LOG_TAIL_CHARS:]}
        else:
            rec["log"] = {"path": str(lp), "bytes": 0, "tail": "",
                          "note": "the run produced no log"}
    if extra:
        rec.update(extra)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def launch_process(launch_record) -> tuple[int | None, str | None]:
    """The PID and identity token a launch record names."""
    rec = json.loads(Path(launch_record).read_text())
    pid = rec.get("pid")
    return (int(pid) if pid is not None else None), rec.get("pid_identity")


def main() -> int:
    """The watcher's two questions, as a command.

        pkcguard.py alive --launch <launch.json>
        pkcguard.py identity --pid <n>
        pkcguard.py incomplete --phase N [--revision R] --reason ... [--log ...]
    """
    import argparse                                                    # noqa: PLC0415
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("alive")
    a.add_argument("--launch", required=True)
    c = sub.add_parser("clean")
    c.add_argument("--phase", type=int, required=True)
    c.add_argument("--revision", default="")
    i = sub.add_parser("identity")
    i.add_argument("--pid", type=int, required=True)
    n = sub.add_parser("incomplete")
    n.add_argument("--phase", type=int, required=True)
    n.add_argument("--revision", default="")
    n.add_argument("--reason", required=True)
    n.add_argument("--log", default="")
    n.add_argument("--claim", default="")
    n.add_argument("--analysis", default="")
    args = ap.parse_args()
    if args.cmd == "clean":
        try:
            print(json.dumps(require_clean_pushed_branch(args.phase, args.revision)))
        except BoundaryError as exc:
            print(str(exc))
            return 1
        return 0
    if args.cmd == "identity":
        tok = proc_identity(args.pid)
        print(tok or "")
        return 0 if tok else 1
    if args.cmd == "alive":
        pid, identity = launch_process(args.launch)
        ok = process_alive(pid, identity)
        print(json.dumps({"pid": pid, "identity": identity, "alive": ok}))
        return 0 if ok else 1
    path = durable_failure(args.phase, args.revision, args.reason,
                           extra={"claim": args.claim} if args.claim else None,
                           analysis=args.analysis or None, log=args.log or None)
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
