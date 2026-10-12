"""exp-057 originfit: full SHA-256 pins of this experiment's own code files (Sol round 1, finding 4).

    .venv/bin/python experiments/exp-057-originfit/code/ofit_codepins.py

ZERO FIT, no data file, no label.  For each code file: the SHA-256 of the working file, of the blob
in every commit that touched it (`git show <commit>:<path>`), its last-modification time (UTC), and
the run(s) it served with the log that records each run.  A pin is valid for a run when the
committed blob equals the working file and the file's last modification precedes the run.  ofit_fit.py
is also checked against the pin that every fit record already carries.  Writes
analysis/codepins.json once (write-once guard).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
OUT = EXP / "analysis/codepins.json"
_spec = importlib.util.spec_from_file_location("nooverwrite", ROOT / "experiments/exp-046-surfacefit/code/nooverwrite.py")
NO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NO)

# file -> the runs it served: (what, log, run start UTC as recorded in RESULTS section 15)
RUNS = {
    "ofit_prehash.py": [("census", "logs/prehash.log", "02:56:00")],
    "ofit_prehash_a1.py": [("census amendment a1", "logs/prehash-a1.log", "03:23:43")],
    "ofit_fit.py": [("first plan stage", "logs/plan.log", "03:21:10"), ("plan stage", "logs/plan-a1.log", "03:24:19"),
                    ("run-path check", "logs/run-check.log", "03:26:29"), ("36 fits", "logs/fits.log", "03:27:19")],
    "ofit_capacity.py": [("capacity diagnostic", "logs/capacity.log", "03:26:46")],
    "ofit_measure.py": [("dry run", "logs/measure-standin.log", "04:29:47"), ("official measurement", "logs/measure.log", "04:32:22")],
    "ofit_tables.py": [("table printer (stdout only)", None, None)],
    "ofit_arith.py": [("arithmetic printer (stdout only)", None, None)],
    "ofit_codepins.py": [("this record", None, None)],
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git(*a: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *a], check=True, capture_output=True).stdout


def main() -> None:
    NO.refuse_existing([OUT], overwrite=False)
    fit_pin = {json.loads(p.read_text())["code_files"]["experiments/exp-057-originfit/code/ofit_fit.py"]
               for p in sorted((EXP / "analysis/fits").glob("*.json"))}
    files, ok = {}, True
    for name, runs in RUNS.items():
        rel = f"experiments/exp-057-originfit/code/{name}"
        p = ROOT / rel
        work = sha(p.read_bytes())
        mtime = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc)
        commits = []
        for line in git("log", "--format=%H %cI", "--", rel).decode().split("\n"):
            if line:
                c, t = line.split(" ")
                commits.append({"commit": c, "committed": dt.datetime.fromisoformat(t).astimezone(dt.timezone.utc)
                                .strftime("%Y-%m-%d %H:%M:%S UTC"), "sha256": sha(git("show", f"{c}:{rel}"))})
        rec = {"sha256": work, "modified": mtime.strftime("%Y-%m-%d %H:%M:%S UTC"), "commits": commits,
               "working_file_equals_last_commit": bool(commits) and commits[0]["sha256"] == work, "runs": []}
        for what, log, start in runs:
            r = {"run": what, "log": log, "start_utc": start}
            if start is not None:
                t0 = dt.datetime.strptime(f"2026-09-11 {start}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
                r["file_unchanged_since_before_run"] = mtime <= t0
                ok &= r["file_unchanged_since_before_run"]
            rec["runs"].append(r)
        if name == "ofit_fit.py":
            rec["equals_the_pin_in_every_fit_record"] = fit_pin == {work}
            ok &= rec["equals_the_pin_in_every_fit_record"]
        if name != "ofit_codepins.py":
            ok &= rec["working_file_equals_last_commit"]
        files[rel] = rec
        print(f"{name:20s} {work}  modified {rec['modified']}  last commit equal: "
              f"{rec['working_file_equals_last_commit']}  " + "; ".join(
                  f"{r['run']}: {r.get('file_unchanged_since_before_run', '-')}" for r in rec["runs"]))
    out = {"what": "exp-057 originfit: full SHA-256 pins of the experiment's own code files (Sol round 1, finding 4). "
                   "ZERO FIT, no data file read.",
           "files": files, "all_pins_valid": ok,
           "note": "ofit_codepins.py pins itself as the working file at the time it ran; its commit follows this record."}
    NO.refuse_existing([OUT], overwrite=False)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"all pins valid: {ok}; written {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
