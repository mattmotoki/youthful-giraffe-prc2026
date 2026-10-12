"""exp-045-harvest: keep the harvest's share of the Kaggle CPU pool full, in order.

The pool is shared with the other workers of this repo and Kaggle caps it at five
batch sessions, so a push is refused rather than queued.  This loop holds at most
`analysis/max_slots` harvest sessions (the desk sets that number; it is read on every
pass, so it can be changed while the loop runs), pushes the next month the moment a
slot frees, pulls a session as soon as it completes, and re-queues the days a session
did not finish.  It never touches another worker's kernels.

A day is done when its `analysis/runs/run_<day>.json` exists.  A day that fails
`ATTEMPTS` times is written to `analysis/missing.json` with the reason and left alone.

    nohup .venv/bin/python experiments/exp-045-harvest/code/dispatch.py \
        > experiments/exp-045-harvest/logs/dispatch.log 2>&1 &
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
from days import ORDER, days as month_days                     # noqa: E402

POLL_S = 300
ATTEMPTS = 2
FLOOR_MB = 500      # the desk's floor, 2026-09-08: land nothing until df shows this
BUDGET_S = 39600
KAGGLE = str(Path.home() / ".local/bin/kaggle")
STATE = EXP / "analysis/dispatch_state.json"
MISSING = EXP / "analysis/missing.json"
SLOTS = EXP / "analysis/max_slots"
FULL = "Maximum batch CPU session count"


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)


def load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save(path: Path, obj) -> None:
    """Write via a temporary file, so a full disk cannot leave a truncated state file.

    The laptop's data volume hit zero at 11:43 UTC on 2026-09-08 from other work and the
    OSError killed this loop while four Kaggle sessions were still running.  State is the
    only thing that makes a dead dispatcher a non-event, so it is never written in place
    and a failed write is logged rather than fatal."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent))
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)
    except OSError as e:                                       # noqa: BLE001
        log(f"could not write {path.name}: {e}")
        try:
            os.unlink(tmp)
        except Exception:                                      # noqa: BLE001
            pass


def done_days() -> set[str]:
    return {p.name[4:14] for p in (EXP / "analysis/runs").glob("run_*.json")}


def max_slots() -> int:
    try:
        return int(SLOTS.read_text().strip())
    except Exception:                                          # noqa: BLE001
        return 3


def status(name: str, previous: str = "RUNNING") -> str:
    """The session's state, or `previous` when Kaggle does not answer.

    Never invent a third answer.  At 14:31 UTC on 2026-09-08 the CLI lost its
    credentials for ten minutes and this returned "UNKNOWN" for four live sessions.
    UNKNOWN is not a live state, so their days left `busy` in next_chunk() and the
    dispatcher pushed 2025-07, 2026-07, 2025-01 and 2025-02 a second time -- four
    duplicate sessions, 121 days re-fetched, and the whole five-session pool held
    away from the other workers for hours.  A status this loop cannot read is a
    session it must assume is still running.
    """
    r = subprocess.run([KAGGLE, "kernels", "status", f"{kaggle_user()}/prc-harvest-{name}"],
                       capture_output=True, text=True, timeout=180)
    out = (r.stdout + r.stderr).strip()
    for s in ("RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCEL"):
        if s in out:
            return s
    log(f"{name}: status unreadable, holding {previous} ({out[:120]})")
    return previous


def push(name: str, day_list: list[str]) -> tuple[bool, str]:
    r = subprocess.run(["bash", str(EXP / "code/push_kernel.sh"), name,
                        json.dumps(day_list), "0", str(BUDGET_S)],
                       capture_output=True, text=True, timeout=600)
    out = (r.stdout + r.stderr).strip()
    ok = r.returncode == 0 and FULL not in out and "error" not in out.lower()
    return ok, out.replace("\n", " | ")[-400:]


def free_mb() -> int:
    st = os.statvfs(EXP)
    return int(st.f_bavail * st.f_frsize / 1e6)


def pull(name: str) -> tuple[bool, str]:
    """`(landed, what happened)`.  A month is about 34 MB of summaries, so it waits
    for the volume rather than half-landing on it."""
    if free_mb() < FLOOR_MB:
        return False, f"{free_mb()} MB free, under the {FLOOR_MB} MB floor"
    r = subprocess.run(["bash", str(EXP / "code/pull_kernel.sh"), name],
                       capture_output=True, text=True, timeout=1800)
    out = (r.stdout + r.stderr).strip().replace("\n", " | ")[-400:]
    return r.returncode == 0, out


def current_rule() -> tuple[str, str]:
    """`(sha256, version)` of the rule the repository holds right now."""
    from rule_source import rule_sources, version
    r, e = rule_sources(str(ROOT))
    return hashlib.sha256((r + e).encode()).hexdigest(), version(r, e)


def rederive(name: str) -> bool:
    """Rewrite one landed month's summaries under the current rule, from its stored
    points.  This belongs to the same step as the pull and not to a loop beside it:
    on 2025-12 the two ran together, the re-derive finished at 13:54:07 and the pull
    overwrote it 41 seconds later with the fetch kernel's own segments, while
    manifest.json went on reporting the re-derived rule.  One writer, in order."""
    r = subprocess.run([sys.executable, str(EXP / "code/rederive_local.py"), name],
                       capture_output=True, text=True)
    line = (r.stdout or r.stderr).strip().splitlines()
    log(f"  {line[-1] if line else '(no output)'}")
    if r.returncode:
        log(f"  {name}: re-derive failed ({r.returncode}); its summaries carry the "
            f"fetch kernel's rule until it is retried")
    return r.returncode == 0


def report(month: str) -> None:
    """Census the month that just landed and write RESULTS.md's tables from it.

    RESULTS.md was edited by hand until 2026-01 landed and refuted the two-day gate at
    three airports; a report that has to be retyped once a month is a stale number
    waiting to happen.  Failures here are logged, never fatal: the harvest is the job
    and the report is downstream of it.
    """
    for args in ([str(EXP / "code/manifest.py")],
                 [str(EXP / "code/check_landed.py")],
                 # only does anything for a month the publisher never signed
                 [str(EXP / "code/check_lengths.py"), month],
                 [str(EXP / "code/census.py"), month],
                 # received <= departure segments observed; see check_receipt.py
                 [str(EXP / "code/check_receipt.py"), month],
                 [str(EXP / "code/tables.py"), "--write"]):
        r = subprocess.run([sys.executable, *args], capture_output=True, text=True)
        line = (r.stdout or r.stderr).strip().splitlines()
        log(f"  {line[-1] if line else '(no output)'}")
        if r.returncode:
            log(f"  {Path(args[0]).name} failed ({r.returncode}); the days are on disk")
            return


def next_chunk(state: dict) -> tuple[str, str, list[str]] | None:
    """`(kernel name, month, days)` for the earliest month with unfinished days."""
    have = done_days()
    miss = {d for d in load(MISSING, {})}
    busy = {d for k in state["kernels"] if k["status"] in ("RUNNING", "QUEUED", "PUSHED")
            for d in k["days"]}
    for month in ORDER:
        todo = [d for d in month_days(month) if d not in have and d not in miss and d not in busy]
        if not todo:
            continue
        used = [k["name"] for k in state["kernels"] if k["month"] == month]
        name = month if not used else f"{month}-{chr(ord('a') + len(used))}"
        return name, month, todo
    return None


def retire(state: dict, k: dict) -> bool:
    """A finished session: pull it, then blame the days it did not deliver.

    Returns False when the pull itself failed.  That distinction is the whole point:
    a day the session never fetched is the session's fault and counts against its
    ATTEMPTS, while a day this laptop could not land is nobody's fault and must not
    be blamed -- two failed pulls would otherwise write a whole delivered month into
    missing.json.  The kernel keeps its finished status and is retired next pass.
    """
    log(f"{k['name']}: {k['status']}, pulling")
    landed, out = pull(k["name"])
    log(f"  {out}")
    if not landed:
        log(f"  {k['name']}: NOT LANDED, the session's output stays on Kaggle; "
            f"retrying next pass")
        return False
    ses = load(EXP / f"analysis/sessions/{k['name']}.json", {})
    sha, ver = current_rule()
    if ses.get("rule_sha256") and ses["rule_sha256"] != sha:
        log(f"  {k['name']}: fetched under {ses['rule_sha256'][:16]}, re-deriving "
            f"to {ver}")
        rederive(k["name"])
    else:
        # This session already carried the current rule, so what it just wrote IS the
        # current rule.  Any re-derive record left over for these days now describes
        # bytes that have been replaced -- the duplicate sessions of 14:31 land over
        # days their originals had re-derived -- and a record that outlives its bytes
        # is how manifest.json came to report a rule the disk did not hold.
        stale = [EXP / f"analysis/rederived/{d}.json" for d in k["days"]]
        gone = [f for f in stale if f.exists()]
        for f in gone:
            f.unlink()
        if gone:
            log(f"  {k['name']}: dropped {len(gone)} re-derive records the pull "
                f"replaced; these days are the fetch kernel's own {ver}")
    have = done_days()
    reasons = {f["day"]: f["reason"] for f in ses.get("failed", [])}
    missing = load(MISSING, {})
    for d in k["days"]:
        if d in have:
            continue
        state["attempts"][d] = state["attempts"].get(d, 0) + 1
        if state["attempts"][d] >= ATTEMPTS:
            missing[d] = {"attempts": state["attempts"][d],
                          "reason": reasons.get(d, f"session {k['name']} ended {k['status']} "
                                                   "without a run record"),
                          "kernel": k["name"]}
    save(MISSING, missing)
    report(k["month"])
    delivered = sum(1 for d in k["days"] if d in have)
    log(f"  {k['name']}: {delivered} of {len(k['days'])} days delivered; "
        f"{len(missing)} days now listed missing")
    return True


def main() -> None:
    state = load(STATE, {"kernels": [], "attempts": {}})
    # the watchdog reads this file to know the dispatcher is alive; writing it here
    # rather than at the launch site is what stops it going stale across a restart,
    # which on 2026-09-08 cost a false "DISPATCHER GONE" while the loop was healthy
    (EXP / "logs/dispatch.pid").write_text(f"{os.getpid()}\n")
    log(f"dispatch up: {len(done_days())} days already on disk, slots={max_slots()}")
    while True:
      try:
        # Re-read the record at the top of every pass.  The dispatcher is the only
        # writer, so this normally returns exactly what it saved last pass -- but it
        # also means an operator can correct `attempts` on disk and have it taken up,
        # which it could not when state lived only in memory.  On 2026-09-08 the four
        # 2025-05 days whose retry budget was reset after the ...-prod-0tmp tag fix
        # were written into missing.json anyway, because this loop saved its stale
        # copy over the reset.
        state = load(STATE, state)
        # a session that finished but could not be landed here: try again, first
        for k in state["kernels"]:
            if k.get("landed") is False and retire(state, k):
                k.pop("landed", None)
        for k in state["kernels"]:
            if k["status"] in ("PUSHED", "RUNNING", "QUEUED"):
                st = status(k["name"], k["status"])
                if st != k["status"]:
                    log(f"{k['name']}: {k['status']} -> {st}")
                k["status"] = st
                if st in ("COMPLETE", "ERROR", "CANCEL"):
                    if not retire(state, k):
                        k["status"] = st            # unretired: try again next pass
                        k["landed"] = False
        live = [k for k in state["kernels"] if k["status"] in ("PUSHED", "RUNNING", "QUEUED")]
        slots = max_slots()
        while len(live) < slots:
            nxt = next_chunk(state)
            if nxt is None:
                break
            name, month, todo = nxt
            ok, out = push(name, todo)
            if not ok:
                log(f"{name}: push refused ({out})")
                break
            rec = {"name": name, "month": month, "days": todo, "status": "PUSHED",
                   "pushed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            state["kernels"].append(rec)
            live.append(rec)
            log(f"{name}: pushed {len(todo)} days ({todo[0]} .. {todo[-1]})")
        save(STATE, state)
        if not live and next_chunk(state) is None:
            log("dispatch: every month is either delivered or listed missing")
            return
      except OSError as e:                                     # noqa: BLE001
        log(f"pass failed ({e}); the sessions keep running, retrying next cycle")
      time.sleep(POLL_S)


if __name__ == "__main__":
    main()
