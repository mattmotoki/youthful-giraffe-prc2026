"""exp-061-sparseprior: the round-1 review answer to finding 11 --- launch and size receipts.

Every job over two minutes ran detached with its PID written beside its log, but
`.gitignore` holds `experiments/exp-061-sparseprior/logs/*.pid`, so no committed record carried a
PID and the claim in RESULTS could not be checked.  This job writes one committed receipt: every
detached job's PID, its log, the log's size and last write, the `/usr/bin/time` wall and CPU
seconds parsed from that log, and the one-minute load each producing record holds from its own
launch.  It also measures the repository directories, so the artifact and repository-size figures
in RESULTS have a receipt instead of a quoted number.

    .venv/bin/python experiments/exp-061-sparseprior/code/receipts61r1.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e61 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

LOGS = E.EXP / "logs"
SIZED = ("artifacts/exp-061", "experiments/exp-061-sparseprior", "artifacts", "experiments",
         "data", ".git")


def du_mb(rel: str) -> float | None:
    p = E.ROOT / rel
    if not p.exists():
        return None
    out = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True).stdout.split()
    return round(int(out[0]) / 1024, 1) if out else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = E.out_path(E.ANALYSIS / "receipts61-r1.json")
    NO.refuse_existing([out], a.overwrite)

    jobs = {}
    for log in sorted(LOGS.glob("*.log")):
        txt = log.read_text(errors="replace")
        m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", txt)
        pid = LOGS / f"{log.stem}.pid"
        jobs[log.name] = {
            "pid": int(pid.read_text().strip()) if pid.exists() else None,
            "pid_file": str(pid.relative_to(E.ROOT)) if pid.exists() else None,
            "log_bytes": log.stat().st_size,
            "log_last_write_utc": dt.datetime.fromtimestamp(
                log.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_s": float(m.group(1)) if m else None,
            "process_cpu_s": round(float(m.group(2)) + float(m.group(3)), 2) if m else None}
    timed = [v for v in jobs.values() if v["process_cpu_s"] is not None]

    launches = {}
    for p in sorted(E.ANALYSIS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        lm = d.get("launch") or {}
        if lm.get("launch_utc"):
            launches[p.name] = {"launch_utc": lm["launch_utc"],
                                "launch_load_1m": lm.get("launch_load_1m"),
                                "seconds": d.get("seconds")}

    rec = {"what": __doc__.strip().splitlines()[0], "measured_utc": E.E60.utc(),
           "code_sha256": E.code_shas(), "jobs": jobs,
           "detached_jobs_with_a_pid": sorted(k for k, v in jobs.items() if v["pid"]),
           "instrumented_jobs": len(timed),
           "summed_process_cpu_s": round(sum(v["process_cpu_s"] for v in timed), 1),
           "record_launches": launches,
           "directory_megabytes": {rel: du_mb(rel) for rel in SIZED},
           "note": ("the repository was already above the 20 GB limit of assignments/COMMON.md "
                    "before this experiment started; the figure here is measured after it, and "
                    "the space belongs to earlier experiments, so the removal is the desk's call")}
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"written {out.name}: {len(jobs)} logs, "
          f"{len(rec['detached_jobs_with_a_pid'])} with a PID, "
          f"{rec['summed_process_cpu_s']} process-cpu s, "
          f"sizes {rec['directory_megabytes']}")


if __name__ == "__main__":
    main()
