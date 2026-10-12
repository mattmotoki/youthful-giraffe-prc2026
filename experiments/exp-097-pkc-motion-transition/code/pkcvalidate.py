"""exp-097 result validation: what the watcher must prove before it commits anything.

The watcher is the last checkpoint before a result becomes the record, so it re-derives the
run's story from the files on disk rather than trusting the run that wrote them: the launch
record, the start receipt, the claim, the process, the exit receipt, the verify receipt, the
resource receipt, the result and the phase receipt must all agree, on one source commit, one
PID lineage, one exit code and one digest.  For phase 2 it also checks the gate set is
exactly G1-G5, that the decision follows from the gates, and that a FAIL left no candidate
behind.

    pkcvalidate.py --phase 2 [--revision r1]

Exit 0 and a JSON report when every check passes; exit 1 and the failures otherwise.  It
opens no source, label, prediction or ranking value.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pkcguard as G                                                   # noqa: E402

EXP = G.EXP
import pkcmotion as M                                                  # noqa: E402

#: One definition, shared with the producer: `pkcmotion.GATES`.
GATES = M.GATES
RESULT_STEM = {1: "screen", 2: "score", 3: "candidate"}


class ValidationError(RuntimeError):
    """The run's own files do not agree.  Nothing is committed."""


def _read(path: Path, what: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValidationError(f"missing or empty {what}: {path.relative_to(EXP)}")
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"unreadable {what}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValidationError(f"{what} is not a JSON object")
    return obj


def validate(phase: int, revision: str = "", exp=None) -> dict:
    exp = EXP if exp is None else Path(exp)
    suf = f"-{revision.lstrip('-')}" if revision else ""
    A, L = exp / "analysis", exp / "logs"
    paths = {
        "launch": L / f"phase{phase}{suf}.launch.json",
        "exit": L / f"phase{phase}{suf}.exit.json",
        "verify": L / f"phase{phase}{suf}.verify.json",
        "claim": A / f"claim-phase{phase}{suf}.json",
        "resource": A / f"resource-phase{phase}{suf}.json",
        "result": A / f"{RESULT_STEM[phase]}{suf}.json",
        "receipt": A / f"receipt-phase{phase}{suf}.json",
    }
    if phase in (2, 3):
        paths["start"] = A / f"start-phase{phase}{suf}.json"
    rec = {k: _read(v, k) for k, v in paths.items()}
    digests = {k: G.sha256(v) for k, v in paths.items()}
    fails: list[str] = []

    def want(cond, why):
        if not cond:
            fails.append(why)

    # the run is one run: one commit, one claim, one exit
    commit = rec["launch"].get("commit")
    want(isinstance(commit, str) and len(commit) == 40, f"launch commit is {commit!r}")
    want(rec["exit"].get("commit") == commit,
         f"exit receipt commit {rec['exit'].get('commit')} != launch {commit}")
    want(rec["exit"].get("exit_code") == 0,
         f"exit code is {rec['exit'].get('exit_code')!r}, not 0")
    want(rec["exit"].get("phase") == phase, "exit receipt names another phase")
    want(rec["launch"].get("phase") == phase, "launch record names another phase")
    want(rec["claim"].get("phase") == phase, "claim names another phase")
    want(str(rec["claim"].get("revision", "")).lstrip("-") == revision.lstrip("-"),
         f"claim revision {rec['claim'].get('revision')!r} != {revision!r}")
    want(rec["result"].get("revision") == (revision.lstrip("-") or None),
         f"result revision {rec['result'].get('revision')!r} != {revision!r}")
    want(digests["result"] == digests["receipt"],
         "the result and the phase receipt are not byte-identical")

    # the resource contract, as recorded by three independent writers
    for name in ("launch", "resource"):
        want(rec[name].get("cpu_set") == "63-71", f"{name} cpu_set is not 63-71")
        want(rec[name].get("nice") == 19, f"{name} nice is not 19")
        want(rec[name].get("threads") == 8, f"{name} threads is not 8")
    want(rec["launch"].get("gpu") == "none", "launch record does not say gpu none")

    # the process: the launcher's PID, the claim's PID and the verify receipt agree
    lpid = rec["launch"].get("pid")
    want(isinstance(lpid, int) and lpid > 0, f"launch pid is {lpid!r}")
    want(rec["verify"].get("pid") == lpid,
         f"verify receipt pid {rec['verify'].get('pid')!r} != launch pid {lpid!r}")
    checks = rec["verify"].get("checks") or []
    want(len(checks) >= 2, "verify receipt has fewer than two checks")
    want(any(c.get("exit_receipt") == "yes" or c.get("log_bytes", 0) > 0 for c in checks),
         "no verify check saw the run produce anything")
    want(isinstance(rec["claim"].get("pid"), int), "claim records no pid")

    if phase in (2, 3):
        start = rec["start"]
        want(start.get("phase") == phase, "start receipt names another phase")
        want(start.get("commit") == commit,
             f"start receipt commit {start.get('commit')} != launch {commit}")
        want(start.get("nice") == 19, "start receipt nice is not 19")
        want(all(v == "8" for v in (start.get("threads") or {}).values()),
             "start receipt thread caps are not all 8")
        # A result comes from the workstation, where the affinity mask exists and the
        # launcher is a real process.  `None` is not an acceptable value for either
        # (assignment item 2): it would mean the run could not see its own contract.
        aff = start.get("cpu_affinity")
        want(aff is not None, "start receipt records no cpu affinity; a workstation run "
                              "must record its own mask")
        want(aff is None or set(aff) == set(G.CPU_SET),
             f"start receipt cpu affinity is {aff}, registered {sorted(G.CPU_SET)}")
        want(start.get("launcher_pid") is not None,
             "start receipt records no launcher pid")
        want(start.get("launcher_pid") == lpid,
             f"start receipt launcher pid {start.get('launcher_pid')!r} != launch pid {lpid!r}")
        want(rec["verify"].get("pid") == start.get("launcher_pid"),
             "the verify receipt watched a different process from the one that started")
        # the claim, the controller and the start receipt are one process
        want(start.get("claim_pid") is not None, "start receipt records no claim pid")
        want(start.get("claim_pid") == rec["claim"].get("pid"),
             f"start receipt claim pid {start.get('claim_pid')!r} != the claim's own "
             f"{rec['claim'].get('pid')!r}")
        want(start.get("pid") == start.get("claim_pid"),
             f"the controller pid {start.get('pid')!r} did not take the claim "
             f"{start.get('claim_pid')!r}")
        want(start.get("pid") != lpid,
             "the controller and its launcher are recorded as the same process")
        want(isinstance(start.get("desk_acceptance"), dict),
             "start receipt carries no desk acceptance")
        want((start.get("desk_acceptance") or {}).get("phase") == phase,
             "the desk acceptance in the start receipt is for another phase")
        want((start.get("desk_acceptance") or {}).get("source_commit") == commit,
             "the desk acceptance names a different source commit")
        # the result's acceptance record is EXACTLY the one the start receipt bound
        want(rec["result"].get("desk_acceptance") == start.get("desk_acceptance"),
             "the result's desk acceptance is not the one the start receipt bound")
        # the phase-1 r2 predecessor chain, by digest
        chain = (start.get("phase1_chain") or {})
        want(chain.get("revision") == G.PHASE1_REVISION,
             f"the start receipt rests on phase-1 revision {chain.get('revision')!r}")
        want(chain.get("support_verdict") == "PASS", "the phase-1 support verdict is not PASS")
        for what, spec in (chain.get("files") or {}).items():
            p = exp / spec["path"]
            want(p.is_file() and G.sha256(p) == spec["sha256"],
                 f"phase-1 {what} at {spec['path']} no longer matches its recorded digest")
        want(set(chain.get("files") or ()) == set(G.PHASE1_CHAIN),
             "the phase-1 chain in the start receipt is incomplete")
        want(G.PHASE1_INVALID["receipt"] not in
             {s.get("path") for s in (chain.get("files") or {}).values()},
             "the invalid unsuffixed phase-1 receipt is inside the accepted chain")
        for k in ("phase1_chain", "desk_acceptance"):
            want(rec["result"].get(k) is not None, f"the result records no {k}")
        want((rec["result"].get("phase1_chain") or {}).get("files") ==
             (chain.get("files") or {}),
             "the result's phase-1 chain differs from the start receipt's")

    if phase == 2:
        gates = rec["result"].get("gates")
        want(isinstance(gates, dict) and set(gates) == set(GATES),
             f"the gate set is {sorted(gates) if isinstance(gates, dict) else gates!r}, "
             f"registered {list(GATES)}")
        decision = rec["result"].get("decision")
        if isinstance(gates, dict) and set(gates) == set(GATES):
            # the aggregate is DERIVED here, never read from inside the gate map
            derived = all(bool(gates[g]) for g in GATES)
            expected = "ADVANCE" if derived else "FAIL"
            want(decision == expected,
                 f"decision is {decision!r} but the gates say {expected}")
            recorded = rec["result"].get("all_gates_pass")
            want(recorded is None or bool(recorded) == derived,
                 f"the recorded aggregate {recorded!r} disagrees with the five gates")
        want(decision in ("ADVANCE", "FAIL"), f"decision is {decision!r}")
        if decision == "FAIL":
            stray = sorted(p.name for p in (exp / "analysis").glob("candidate*.json"))
            want(not stray, f"a FAIL left a phase-3 candidate record behind: {stray}")
            cand = exp / "work" / "candidate.parquet"
            want(not cand.exists(), "a FAIL left a candidate parquet behind")

    stray = sorted(p.name for p in (exp / "analysis").glob(f"incomplete-phase{phase}*.json"))
    want(not stray, f"an INCOMPLETE receipt is present: {stray}")

    if fails:
        raise ValidationError("; ".join(fails))
    return {"phase": phase, "revision": revision.lstrip("-") or None, "commit": commit,
            "ok": True, "checked": sorted(paths),
            "digests": {k: digests[k] for k in sorted(digests)},
            "decision": rec["result"].get("decision"),
            "gate_pass": rec["result"].get("gate_pass")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--revision", default="")
    a = ap.parse_args()
    try:
        print(json.dumps(validate(a.phase, a.revision), indent=2))
    except ValidationError as exc:
        print(json.dumps({"phase": a.phase, "ok": False, "failures": str(exc)}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
