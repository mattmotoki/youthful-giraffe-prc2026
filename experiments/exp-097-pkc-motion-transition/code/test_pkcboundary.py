"""exp-097 execution-boundary tests: the nine ways the label-bearing door must stay shut.

Assignment `pkc-phase2-desk-execution-fix.md` item 5.  Every test builds a throw-away tree;
no real receipt, source, label, prediction or ranking value is opened.

    OMP_NUM_THREADS=2 .venv/bin/python \
        experiments/exp-097-pkc-motion-transition/code/test_pkcboundary.py
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pkcaccept as A                                                  # noqa: E402
import pkccontrol as C                                                 # noqa: E402
import pkcmotion as M                                                  # noqa: E402
import pkcguard as G                                                   # noqa: E402
import pkcvalidate as V                                                # noqa: E402

FAILED: list[str] = []
REL = "experiments/exp-097-pkc-motion-transition"


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def refuses(name: str, fn, *exc) -> None:
    try:
        fn()
        check(name, False, "it returned instead of refusing")
    except exc or (G.BoundaryError,) as e:                             # noqa: B030
        check(name, True, str(e)[:70])


# --------------------------------------------------------------- the phase-1 r2 chain

def write_chain(exp: Path, *, revision="r2", exit_code=0) -> None:
    (exp / "analysis").mkdir(parents=True, exist_ok=True)
    (exp / "logs").mkdir(parents=True, exist_ok=True)
    rec = {"experiment": "exp-097-pkc-motion-transition", "phase": 1,
           "revision": revision, "phase_kind": "label-free support and identity",
           "support_condition": {"verdict": "PASS", "total_support": 3864}}
    body = json.dumps(rec, indent=2, sort_keys=True) + "\n"
    (exp / f"analysis/screen-{revision}.json").write_text(body)
    (exp / f"analysis/receipt-phase1-{revision}.json").write_text(body)
    (exp / f"analysis/claim-phase1-{revision}.json").write_text(
        json.dumps({"phase": 1, "revision": f"-{revision}", "pid": 1234}))
    (exp / f"analysis/resource-phase1-{revision}.json").write_text(
        json.dumps({"phase": 1, "cpu_set": "63-71", "nice": 19, "threads": 8}))
    (exp / f"logs/phase1-{revision}.launch.json").write_text(json.dumps(
        {"phase": 1, "cpu_set": "63-71", "nice": 19, "threads": 8, "gpu": "none",
         "pid": 1234, "commit": "a" * 40,
         "command": f"taskset -c 63-71 nice -n 19 python pkccontrol.py --phase=1 "
                    f"--revision={revision}"}))
    (exp / f"logs/phase1-{revision}.exit.json").write_text(json.dumps(
        {"phase": 1, "exit_code": exit_code, "commit": "a" * 40}))
    (exp / f"logs/phase1-{revision}.verify.json").write_text(json.dumps(
        {"phase": 1, "pid": 1234,
         "checks": [{"at_s": 20, "log_bytes": 0, "exit_receipt": "no"},
                    {"at_s": 120, "log_bytes": 270, "exit_receipt": "yes"}]}))


def test_1_invalid_unsuffixed_receipt() -> None:
    print("1. the invalid unsuffixed phase-1 receipt can never stand in for r2")
    with tempfile.TemporaryDirectory() as t:
        exp = Path(t) / "exp"
        (exp / "analysis").mkdir(parents=True)
        (exp / "logs").mkdir(parents=True)
        # only the invalid run is present
        body = json.dumps({"phase": 1, "revision": None,
                           "support_condition": {"verdict": "PASS"}})
        (exp / "analysis/screen.json").write_text(body)
        (exp / "analysis/receipt-phase1.json").write_text(body)
        refuses("refuses when only the invalid unsuffixed run is present",
                lambda: G.predecessor_chain(exp))
        # both present: r2 is used, the invalid one is recorded as rejected
        write_chain(exp)
        chain = G.predecessor_chain(exp)
        check("accepts the r2 chain", chain["revision"] == "r2")
        check("records the invalid receipt as rejected",
              "analysis/receipt-phase1.json" in chain["rejected"],
              str(list(chain["rejected"])))
        check("says why it was refused",
              "mask_like_ranking" in
              chain["rejected"]["analysis/receipt-phase1.json"]["refused_because"])
        check("no chain file is the invalid one",
              all(not s["path"].endswith(("screen.json", "receipt-phase1.json"))
                  for s in chain["files"].values()))
    with tempfile.TemporaryDirectory() as t:
        exp = Path(t) / "exp"
        write_chain(exp, revision="r5")                 # another revision, not r2
        refuses("refuses another revision's chain", lambda: G.predecessor_chain(exp))


def test_2_wrong_r2_digest() -> None:
    print("2. an edited r2 artifact breaks the chain")
    with tempfile.TemporaryDirectory() as t:
        exp = Path(t) / "exp"
        write_chain(exp)
        chain = G.predecessor_chain(exp)
        good = copy.deepcopy(chain["files"])
        # the watcher's validator holds the digests; changing a byte must be caught
        p = exp / "analysis/screen-r2.json"
        p.write_text(p.read_text().replace("3864", "3865"))
        check("the recorded digest no longer matches the file",
              G.sha256(p) != good["result"]["sha256"])
        refuses("the chain refuses a result that no longer matches its receipt",
                lambda: G.predecessor_chain(exp))
    with tempfile.TemporaryDirectory() as t:
        exp = Path(t) / "exp"
        write_chain(exp)
        (exp / "analysis/receipt-phase1-r2.json").write_text(
            json.dumps({"phase": 1, "revision": "r2",
                        "support_condition": {"verdict": "FAIL"}}))
        refuses("refuses a phase-1 receipt whose support verdict is not PASS",
                lambda: G.predecessor_chain(exp))
    with tempfile.TemporaryDirectory() as t:
        exp = Path(t) / "exp"
        write_chain(exp, exit_code=1)
        refuses("refuses a phase-1 run that did not exit 0",
                lambda: G.predecessor_chain(exp))
    for missing in G.PHASE1_CHAIN.values():
        with tempfile.TemporaryDirectory() as t:
            exp = Path(t) / "exp"
            write_chain(exp)
            (exp / missing).unlink()
            refuses(f"refuses a chain with no {missing}",
                    lambda e=exp: G.predecessor_chain(e))


def test_3_missing_desk_receipt() -> None:
    print("3. no desk receipt, no label-bearing run")
    with tempfile.TemporaryDirectory() as t:
        exp = Path(t) / "exp"
        (exp / "acceptance").mkdir(parents=True)
        for phase in (2, 3):
            refuses(f"phase {phase} refuses with no desk receipt",
                    lambda p=phase: A.verify(p, repo=Path(t), here=exp),
                    A.AcceptanceError)


def test_4_package_change_after_acceptance() -> None:
    print("4. the desk may add only the receipt to the accepted package")
    with tempfile.TemporaryDirectory() as t:
        repo, reg, pol = scaffold(Path(t))
        bound = [f"{REL}/code/pkcmotion.py"]
        r = desk_receipt(pol, reg, 2)
        write_receipt(repo, r, 2)
        commit(repo, "desk acceptance")
        rec = run_verify(repo, reg, 2, bound)
        check("accepts when only the receipt was added", rec["bound_files_checked"] == 1)
    with tempfile.TemporaryDirectory() as t:
        repo, reg, pol = scaffold(Path(t))
        bound = [f"{REL}/code/pkcmotion.py"]
        (repo / REL / "code/pkcmotion.py").write_text("# changed after acceptance\n")
        r = desk_receipt(pol, reg, 2)
        write_receipt(repo, r, 2)
        commit(repo, "desk acceptance AND a code change")
        refuses("refuses a package that changed after acceptance",
                lambda: run_verify(repo, reg, 2, bound), A.AcceptanceError)


def test_5_direct_phase2_invocation() -> None:
    print("5. a direct controller call refuses and spends no claim")
    env = dict(os.environ)
    env.pop(G.HANDOFF_ENV, None)
    env.pop(G.TOKEN_ENV, None)
    refuses("phase 2 refuses with no launcher handoff",
            lambda: G.start_barrier(2, "", env=env))
    before = sorted((G.EXP / "analysis").glob("claim-phase2*.json"))
    proc = subprocess.run([sys.executable, str(HERE / "pkccontrol.py"), "--phase=2"],
                          capture_output=True, text=True, env=env)
    after = sorted((G.EXP / "analysis").glob("claim-phase2*.json"))
    check("the controller exits non-zero", proc.returncode != 0, str(proc.returncode))
    check("it spends no phase-2 claim", before == after,
          str([p.name for p in after if p not in before]))
    check("it names the boundary it failed",
          "phase-1 r2" in proc.stdout + proc.stderr or
          "launcher" in proc.stdout + proc.stderr, (proc.stdout + proc.stderr)[-120:])


def test_6_start_barrier_ordering() -> None:
    print("6. the handoff must match, and the process must meet its own contract")
    with tempfile.TemporaryDirectory() as t:
        hand = Path(t) / "handoff.json"
        good = {"token": "abc", "phase": 2, "revision": "", "pid": None,
                "commit": "a" * 40, "minted_utc": "2026-09-16T06:00:00Z"}
        caps = {v: "8" for v in G.THREAD_VARS}
        hand.write_text(json.dumps(good))
        env = {G.HANDOFF_ENV: str(hand), G.TOKEN_ENV: "abc", **caps}
        # The resource contract is checked against THIS process, so the test forces the
        # mismatch rather than assuming one: on the workstation the suite itself runs under
        # `taskset -c 63-71 nice -n 19`, where the real contract is satisfied.
        real_nice, real_cpus = G.NICE, G.CPU_SET
        try:
            G.NICE = os.nice(0) + 7
            refuses("refuses a process at the wrong nice level",
                    lambda: G.start_barrier(2, "", env=env))
        finally:
            G.NICE = real_nice
        try:
            G.CPU_SET = frozenset({9991})
            if hasattr(os, "sched_getaffinity"):
                refuses("refuses a process pinned outside the registered CPUs",
                        lambda: G.start_barrier(2, "", env=env))
            else:
                check("no CPU affinity mask on this platform, refusal untestable here", True)
        finally:
            G.CPU_SET = real_cpus
        for tweak, why in [({"token": "different"}, "a token that does not match"),
                           ({"phase": 3}, "a handoff minted for another phase"),
                           ({"revision": "r9"}, "a handoff minted for another revision"),
                           ({"pid": 999999}, "a handoff minted for another process")]:
            hand.write_text(json.dumps({**good, **tweak}))
            refuses(f"refuses {why}", lambda: G.start_barrier(2, "", env=env))
        hand.write_text("not json")
        refuses("refuses an unreadable handoff", lambda: G.start_barrier(2, "", env=env))
        hand.unlink()
        refuses("refuses a handoff that does not exist",
                lambda: G.start_barrier(2, "", env=env))
        hand.write_text(json.dumps(good))
        for var in G.THREAD_VARS:
            bad = dict(env, **{var: "64"})
            refuses(f"refuses an uncapped {var}", lambda b=bad: G.start_barrier(2, "", env=b))


def test_7_post_claim_start_failure() -> None:
    print("7. a failure after the claim leaves a durable INCOMPLETE receipt")
    with tempfile.TemporaryDirectory() as t:
        a = Path(t) / "analysis"
        p = G.durable_failure(2, "-r1", "the start receipt could not be written",
                              extra={"claim": "analysis/claim-phase2-r1.json"}, analysis=a)
        rec = json.loads(p.read_text())
        check("it is written where the watcher looks",
              p.name == "incomplete-phase2-r1.json", p.name)
        check("it says INCOMPLETE", rec["status"] == "INCOMPLETE")
        check("it names the reason", "start receipt" in rec["reason"])
        check("it names the claim it spent", rec["claim"].endswith("claim-phase2-r1.json"))
        check("it carries a UTC stamp", rec["utc"].endswith("Z"))
        p2 = G.durable_failure(2, "", "a second, unsuffixed failure", analysis=a)
        check("an unsuffixed revision gets its own name",
              p2.name == "incomplete-phase2.json", p2.name)


def test_8_inconsistent_result_and_receipt() -> None:
    print("8. the watcher refuses a run whose own files disagree")
    base = build_phase2_tree
    with tempfile.TemporaryDirectory() as t:
        exp = base(Path(t))
        rep = V.validate(2, "", exp=exp)
        check("a consistent run validates", rep["ok"] and rep["decision"] == "FAIL")
    cases = {
        "the result and the phase receipt differ": lambda e: (
            (e / "analysis/receipt-phase2.json").write_text(
                (e / "analysis/receipt-phase2.json").read_text().replace("FAIL", "ADVANCE"))),
        "the exit code is not 0": lambda e: (
            (e / "logs/phase2.exit.json").write_text(
                json.dumps({"phase": 2, "exit_code": 1, "commit": "b" * 40}))),
        "the exit receipt names another commit": lambda e: (
            (e / "logs/phase2.exit.json").write_text(
                json.dumps({"phase": 2, "exit_code": 0, "commit": "c" * 40}))),
        "the gate set is not exactly G1-G5": lambda e: rewrite(
            e / "analysis/score.json", lambda d: d.update(
                {"gates": {k: True for k in list(V.GATES)[:4]}})),
        "the decision contradicts the gates": lambda e: rewrite(
            e / "analysis/score.json", lambda d: d.update(
                {"gates": {k: True for k in V.GATES}, "decision": "FAIL"})),
        "an INCOMPLETE receipt is present": lambda e: (
            (e / "analysis/incomplete-phase2.json").write_text('{"status": "INCOMPLETE"}')),
        "a FAIL left a candidate behind": lambda e: (
            (e / "analysis/candidate.json").write_text('{"rows": 1}')),
        "the start receipt rests on another phase-1 revision": lambda e: rewrite(
            e / "analysis/start-phase2.json",
            lambda d: d["phase1_chain"].update({"revision": "r1"})),
        "a phase-1 artifact changed under the run": lambda e: (
            (e / "analysis/screen-r2.json").write_text('{"tampered": true}')),
        "the nice level was not 19": lambda e: rewrite(
            e / "analysis/resource-phase2.json", lambda d: d.update({"nice": 0})),
        "the verify receipt names another process": lambda e: rewrite(
            e / "logs/phase2.verify.json", lambda d: d.update({"pid": 999})),
    }
    for why, break_it in cases.items():
        with tempfile.TemporaryDirectory() as t:
            exp = base(Path(t))
            break_it(exp)
            refuses(f"refuses when {why}", lambda e=exp: V.validate(2, "", exp=e),
                    V.ValidationError)


def test_9_phase3_needs_its_own_acceptance() -> None:
    print("9. phase-2 acceptance cannot open phase 3")
    with tempfile.TemporaryDirectory() as t:
        repo, reg, pol = scaffold(Path(t))
        write_receipt(repo, desk_receipt(pol, reg, 2), 2)
        commit(repo, "phase-2 acceptance only")
        rec = run_verify(repo, reg, 2, [])
        check("phase 2 opens", rec["phase"] == 2 and rec["run_kind"] == "historical")
        refuses("phase 3 refuses with only the phase-2 receipt",
                lambda: run_verify(repo, reg, 3, []), A.AcceptanceError)
        # a phase-2 receipt copied to the phase-3 path is still refused, by its contents
        (repo / REL / A.PHASES[3]["receipt"]).write_text(
            (repo / REL / A.PHASES[2]["receipt"]).read_text())
        commit(repo, "the phase-2 receipt under the phase-3 name")
        refuses("phase 3 refuses a copied phase-2 receipt",
                lambda: run_verify(repo, reg, 3, []), A.AcceptanceError)
        write_receipt(repo, desk_receipt(pol, reg, 3), 3)
        commit(repo, "phase-3 acceptance")
        rec3 = run_verify(repo, reg, 3, [])
        check("phase 3 opens on its own receipt",
              rec3["phase"] == 3 and rec3["run_kind"] == "candidate")


# ------------------------------------------------------------------------- fixtures

def rewrite(path: Path, mutate) -> None:
    d = json.loads(path.read_text())
    mutate(d)
    path.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")


def build_phase2_tree(tmp: Path) -> Path:
    """A complete, self-consistent phase-2 run, on disk, with a FAIL decision."""
    exp = tmp / "exp"
    write_chain(exp)
    chain = G.predecessor_chain(exp)
    commit = "b" * 40
    result = {"experiment": "exp-097-pkc-motion-transition", "phase": 2, "revision": None,
              "gates": {k: (i != 0) for i, k in enumerate(V.GATES)},
              "decision": "FAIL", "all_gates_pass": False, "phase1_chain": chain,
              "desk_acceptance": {"phase": 2, "source_commit": commit, "run_id": "t"}}
    body = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (exp / "analysis/score.json").write_text(body)
    (exp / "analysis/receipt-phase2.json").write_text(body)
    # the wrapper is 4321 and the controller it started is 4322; the claim is the
    # controller's, never the launcher's
    (exp / "analysis/claim-phase2.json").write_text(
        json.dumps({"phase": 2, "revision": "", "pid": 4322}))
    (exp / "analysis/resource-phase2.json").write_text(
        json.dumps({"phase": 2, "cpu_set": "63-71", "nice": 19, "threads": 8}))
    (exp / "analysis/start-phase2.json").write_text(json.dumps(
        {"phase": 2, "revision": None, "commit": commit, "nice": 19,
         "threads": {v: "8" for v in G.THREAD_VARS},
         "cpu_affinity": sorted(G.CPU_SET), "launcher_pid": 4321,
         "pid": 4322, "claim_pid": 4322, "phase1_chain": chain,
         "desk_acceptance": {"phase": 2, "source_commit": commit, "run_id": "t"}}))
    (exp / "logs/phase2.launch.json").write_text(json.dumps(
        {"phase": 2, "cpu_set": "63-71", "nice": 19, "threads": 8, "gpu": "none",
         "pid": 4321, "pid_identity": "starttime:4321", "commit": commit}))
    (exp / "logs/phase2.exit.json").write_text(
        json.dumps({"phase": 2, "exit_code": 0, "commit": commit}))
    (exp / "logs/phase2.verify.json").write_text(json.dumps(
        {"phase": 2, "pid": 4321,
         "checks": [{"at_s": 20, "log_bytes": 10, "exit_receipt": "no"},
                    {"at_s": 120, "log_bytes": 99, "exit_receipt": "yes"}]}))
    return exp


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
    repo = tmp / "repo"
    (repo / REL / "acceptance").mkdir(parents=True)
    (repo / REL / "code").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / REL / "code/pkcmotion.py").write_text("# the bound code\n")
    registration = commit(repo, "registration")
    (repo / REL / "SPEC.md").write_text("policy continuation\n")
    policy = commit(repo, "policy continuation")
    return repo, registration, policy


def desk_receipt(policy: str, registration: str, phase: int) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    spec = A.PHASES[phase]
    return {"schema": A.SCHEMA, "registration_commit": registration,
            "policy_commit": policy, "authorizes": spec["authorizes"],
            "run_kind": spec["run_kind"], "single_use": True, "issuer": "desk",
            "issued_by": "desk", "run_id": f"exp097-phase{phase}-1",
            "issued_at_utc": (now - dt.timedelta(hours=1)).isoformat(),
            "expires_at_utc": (now + dt.timedelta(hours=6)).isoformat()}


def write_receipt(repo: Path, receipt: dict, phase: int) -> Path:
    p = repo / REL / A.PHASES[phase]["receipt"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(receipt, indent=2))
    return p


def run_verify(repo: Path, registration: str, phase: int, bound) -> dict:
    old = A.REGISTRATION_COMMIT
    A.REGISTRATION_COMMIT = registration
    try:
        return A.verify(phase, repo=repo, here=repo / REL, bound_files=bound)
    finally:
        A.REGISTRATION_COMMIT = old


def test_10_end_to_end_producer_through_validator() -> None:
    """Assignment item 1 and 4: a real `phase2()` record, not a five-gate fixture, has to
    survive the watcher's validator — the defect this repair fixes was exactly a producer
    and a validator that disagreed about the gate map."""
    print("10. a real phase-2 result passes the watcher's validation end to end")
    import test_pkcpath as P                                           # noqa: PLC0415
    with tempfile.TemporaryDirectory() as t:
        exp = Path(t) / "exp"
        write_chain(exp)
        chain = G.predecessor_chain(exp)
        commit_sha = "d" * 40
        acceptance = {"phase": 2, "source_commit": commit_sha, "run_id": "e2e",
                      "issuer": "desk", "issued_by": "desk"}
        boundary = {"phase1_chain": chain,
                    "repo": {"branch": G.BRANCH, "head": commit_sha, "clean": True},
                    "start": {"pid": 222, "nice": 19},
                    "desk_acceptance": acceptance}
        folds = {f: P.synth_fold(f, 700, 180, 0.7, 11 + i)
                 for i, f in enumerate(M.FOLDS)}
        with P.patched(folds, {"support_condition": {"verdict": "PASS"}}):
            result = C.phase2(boundary)
        result["revision"] = None
        check("the producer's gate map is exactly the registered five",
              tuple(result["gates"]) == M.GATES, str(list(result["gates"])))
        check("the aggregate is outside the gate map",
              "all" not in result["gates"] and "all_gates_pass" in result,
              str(list(result["gates"])))
        check("the decision follows from the five gates",
              result["decision"] == ("ADVANCE" if all(result["gates"][g] for g in M.GATES)
                                     else "FAIL"))
        body = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
        (exp / "analysis/score.json").write_text(body)
        (exp / "analysis/receipt-phase2.json").write_text(body)
        (exp / "analysis/claim-phase2.json").write_text(
            json.dumps({"phase": 2, "revision": "", "pid": 222}))
        (exp / "analysis/resource-phase2.json").write_text(
            json.dumps({"phase": 2, "cpu_set": "63-71", "nice": 19, "threads": 8}))
        (exp / "analysis/start-phase2.json").write_text(json.dumps(
            {"phase": 2, "revision": None, "commit": commit_sha, "nice": 19,
             "threads": {v: "8" for v in G.THREAD_VARS},
             "cpu_affinity": sorted(G.CPU_SET), "launcher_pid": 111,
             "pid": 222, "claim_pid": 222, "phase1_chain": chain,
             "desk_acceptance": acceptance}))
        (exp / "logs/phase2.launch.json").write_text(json.dumps(
            {"phase": 2, "cpu_set": "63-71", "nice": 19, "threads": 8, "gpu": "none",
             "pid": 111, "pid_identity": "starttime:1", "commit": commit_sha}))
        (exp / "logs/phase2.exit.json").write_text(
            json.dumps({"phase": 2, "exit_code": 0, "commit": commit_sha}))
        (exp / "logs/phase2.verify.json").write_text(json.dumps(
            {"phase": 2, "pid": 111,
             "checks": [{"at_s": 20, "log_bytes": 10, "exit_receipt": "no"},
                        {"at_s": 120, "log_bytes": 99, "exit_receipt": "yes"}]}))
        rep = V.validate(2, "", exp=exp)
        check("the validator accepts the real producer output", rep["ok"])
        check("it reports the producer's own decision",
              rep["decision"] == result["decision"], str(rep["decision"]))
        # and the identity bindings the repair added must be load-bearing
        for why, break_it in {
            "the launcher pid is missing": lambda e: rewrite(
                e / "analysis/start-phase2.json", lambda d: d.update({"launcher_pid": None})),
            "the cpu affinity is missing": lambda e: rewrite(
                e / "analysis/start-phase2.json", lambda d: d.update({"cpu_affinity": None})),
            "the claim pid is not the controller's": lambda e: rewrite(
                e / "analysis/start-phase2.json", lambda d: d.update({"claim_pid": 999})),
            "the launcher and the controller are one process": lambda e: rewrite(
                e / "analysis/start-phase2.json", lambda d: d.update({"pid": 111,
                                                                     "claim_pid": 111})),
            "the result's acceptance is not the bound one": lambda e: rewrite(
                e / "analysis/score.json",
                lambda d: d["desk_acceptance"].update({"run_id": "another run"})),
            "the verify receipt watched another process": lambda e: rewrite(
                e / "logs/phase2.verify.json", lambda d: d.update({"pid": 777})),
        }.items():
            with tempfile.TemporaryDirectory() as t2:
                copy_tree(exp, Path(t2) / "exp")
                broken = Path(t2) / "exp"
                break_it(broken)
                refuses(f"refuses when {why}", lambda e=broken: V.validate(2, "", exp=e),
                        V.ValidationError)


def test_11_wrapper_death_without_an_exit_receipt() -> None:
    """Assignment item 3: the real wrapper/controller relation, killed, with no synthetic
    exit receipt anywhere."""
    print("11. the wrapper dies after the claim and leaves a durable INCOMPLETE receipt")
    with tempfile.TemporaryDirectory() as t:
        exp = Path(t) / "exp"
        (exp / "analysis").mkdir(parents=True)
        (exp / "logs").mkdir(parents=True)
        log = exp / "logs/phase2.log"
        # a real wrapper shell with a real controller child under it
        wrapper = subprocess.Popen(
            ["bash", "-c", f"echo 'phase 2 starting' >> {log}; "
                           f"exec {sys.executable} -c 'import time; time.sleep(60)'"])
        ident = None
        for _ in range(50):
            ident = G.proc_identity(wrapper.pid)
            if ident:
                break
            time.sleep(0.05)
        check("the wrapper has an identity while it lives", bool(ident), str(ident))
        # Wait for the run to have actually written something before killing it.  On Linux
        # `/proc` answers instantly, so without this the kill lands before the wrapper's
        # first line and the test measures the race instead of the failure record.
        for _ in range(200):
            if log.exists() and log.stat().st_size > 0:
                break
            time.sleep(0.05)
        check("the run wrote to its log before it died", log.exists() and
              log.stat().st_size > 0, f"{log.exists()}")
        (exp / "logs/phase2.launch.json").write_text(json.dumps(
            {"phase": 2, "pid": wrapper.pid, "pid_identity": ident, "cpu_set": "63-71",
             "nice": 19, "threads": 8, "gpu": "none", "commit": "e" * 40}))
        (exp / "analysis/claim-phase2.json").write_text(
            json.dumps({"phase": 2, "revision": "", "pid": wrapper.pid + 1}))
        pid, tok = G.launch_process(exp / "logs/phase2.launch.json")
        check("the watcher sees a live run before the kill", G.process_alive(pid, tok))
        wrapper.kill()
        wrapper.wait()
        for _ in range(60):
            if not G.process_alive(pid, tok):
                break
            time.sleep(0.05)
        check("the watcher sees the death", not G.process_alive(pid, tok))
        check("no exit receipt was invented",
              not (exp / "logs/phase2.exit.json").exists())
        # an UNRELATED live process must never be mistaken for the run
        (exp / "logs/phase2.launch.json").write_text(json.dumps(
            {"phase": 2, "pid": os.getpid(), "pid_identity": ident, "cpu_set": "63-71",
             "nice": 19, "threads": 8, "gpu": "none", "commit": "e" * 40}))
        pid2, tok2 = G.launch_process(exp / "logs/phase2.launch.json")
        check("a live unrelated process with the dead run's identity is still dead",
              not G.process_alive(pid2, tok2))
        check("this process really is alive under its own identity",
              G.process_alive(os.getpid(), G.proc_identity(os.getpid())))
        # the durable record the watcher writes
        out = subprocess.run(
            [sys.executable, str(HERE / "pkcguard.py"), "incomplete", "--phase", "2",
             "--reason", "the launcher process died after the claim without writing an "
                         "exit receipt",
             "--log", str(log), "--claim", str(exp / "analysis/claim-phase2.json"),
             "--analysis", str(exp / "analysis")], capture_output=True, text=True)
        check("the watcher command succeeds", out.returncode == 0, out.stderr[-90:])
        rec_path = exp / "analysis/incomplete-phase2.json"
        check("it wrote the INCOMPLETE receipt", rec_path.is_file())
        rec = json.loads(rec_path.read_text())
        check("it says INCOMPLETE", rec["status"] == "INCOMPLETE")
        check("it names the death", "died after the claim" in rec["reason"])
        check("it preserves a usable failure record",
              "phase 2 starting" in rec["log"]["tail"] and rec["log"]["bytes"] > 0,
              json.dumps(rec.get("log", {}))[:80])
        check("it names the claim it spent", rec["claim"].endswith("claim-phase2.json"))
        # first writer wins: a later failure must not paper over the first
        G.durable_failure(2, "", "a later, less proximate failure",
                          analysis=exp / "analysis")
        again = json.loads(rec_path.read_text())
        check("a later failure does not overwrite the first",
              again["reason"] == rec["reason"])
        # and the validator refuses to commit while it exists
        tree = build_phase2_tree(Path(t) / "ok")
        (tree / "analysis/incomplete-phase2.json").write_text(rec_path.read_text())
        refuses("the watcher refuses to commit while an INCOMPLETE receipt stands",
                lambda: V.validate(2, "", exp=tree), V.ValidationError)


def copy_tree(src: Path, dst: Path) -> None:
    import shutil                                                      # noqa: PLC0415
    shutil.copytree(src, dst)


def test_12_cleanliness_rule_does_not_refuse_the_run_itself() -> None:
    """The first phase-2 attempt refused at 2026-09-16T06:26:25Z because the launcher writes
    this run's launch record before the controller looks at the tree, and the check counted
    untracked files.  The source must still be exact; only the run's OWN artifacts are
    exempt, and they are reported rather than ignored silently."""
    print("12. the tree check is exact, and a run does not refuse its own launch record")
    own = G.run_artifacts(2)
    check("the launch record is one of the run's own artifacts",
          "experiments/exp-097-pkc-motion-transition/logs/phase2.launch.json" in own,
          str(sorted(own)))
    check("so are the log, PID file, exit, verify and handoff",
          {"experiments/exp-097-pkc-motion-transition/logs/phase2.log",
           "experiments/exp-097-pkc-motion-transition/logs/phase2.pid",
           "experiments/exp-097-pkc-motion-transition/logs/phase2.exit.json",
           "experiments/exp-097-pkc-motion-transition/logs/phase2.verify.json",
           "experiments/exp-097-pkc-motion-transition/logs/phase2.handoff.json"} <= own)
    check("a revision gets its own artifact names",
          "experiments/exp-097-pkc-motion-transition/logs/phase2-r1.launch.json"
          in G.run_artifacts(2, "r1"))
    check("another phase's artifacts are NOT this run's",
          not (G.run_artifacts(2) & G.run_artifacts(3)))

    real = G.ROOT, G.BRANCH
    with tempfile.TemporaryDirectory() as t:
        repo = Path(t) / "repo"
        (repo / "experiments/exp-097-pkc-motion-transition/logs").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", "-b", real[1], str(repo)], check=True,
                       capture_output=True)
        (repo / "experiments/exp-097-pkc-motion-transition/SPEC.md").write_text("spec\n")
        commit(repo, "source")
        git(repo, "remote", "add", "origin", str(repo))
        git(repo, "update-ref", f"refs/remotes/origin/{real[1]}", git(repo, "rev-parse", "HEAD"))
        logs = repo / "experiments/exp-097-pkc-motion-transition/logs"
        try:
            G.ROOT = repo
            rec = G.require_clean_pushed_branch(2)
            check("an exact tree passes", rec["clean"] and not rec["own_untracked_artifacts"])
            # the exact situation that refused the real run
            (logs / "phase2.launch.json").write_text("{}")
            (logs / "phase2.verify.json.tmp").write_text("{")
            rec = G.require_clean_pushed_branch(2)
            check("the run's own launch record does not refuse the run",
                  rec["clean"], str(rec))
            check("but it is reported, not ignored silently",
                  len(rec["own_untracked_artifacts"]) == 2,
                  str(rec["own_untracked_artifacts"]))
            # another phase's artifacts are foreign to this one
            refuses("refuses another phase's launch record",
                    lambda: G.require_clean_pushed_branch(3))
            (logs / "phase2.launch.json").unlink()
            (logs / "phase2.verify.json.tmp").unlink()
            # any other untracked file still refuses
            stray = repo / "experiments/exp-097-pkc-motion-transition/logs/notes.txt"
            stray.write_text("x")
            refuses("refuses a foreign untracked file",
                    lambda: G.require_clean_pushed_branch(2))
            stray.unlink()
            # a tracked modification still refuses, staged or not
            spec = repo / "experiments/exp-097-pkc-motion-transition/SPEC.md"
            spec.write_text("edited\n")
            refuses("refuses an unstaged tracked modification",
                    lambda: G.require_clean_pushed_branch(2))
            git(repo, "add", "-A")
            refuses("refuses a staged tracked modification",
                    lambda: G.require_clean_pushed_branch(2))
            git(repo, "checkout", "--", ".")
            # an unpushed commit still refuses
            (logs / ".keep").write_text("")
            git(repo, "add", "-A")
            git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m",
                "unpushed")
            refuses("refuses a commit that is not on the remote",
                    lambda: G.require_clean_pushed_branch(2))
        finally:
            G.ROOT, G.BRANCH = real


def main() -> int:
    for t in (test_1_invalid_unsuffixed_receipt, test_2_wrong_r2_digest,
              test_3_missing_desk_receipt, test_4_package_change_after_acceptance,
              test_5_direct_phase2_invocation, test_6_start_barrier_ordering,
              test_7_post_claim_start_failure, test_8_inconsistent_result_and_receipt,
              test_9_phase3_needs_its_own_acceptance,
              test_10_end_to_end_producer_through_validator,
              test_11_wrapper_death_without_an_exit_receipt,
              test_12_cleanliness_rule_does_not_refuse_the_run_itself):
        t()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("all execution-boundary tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
