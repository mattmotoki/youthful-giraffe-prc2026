"""Write-once command controller used for local and Kaggle CLI operations."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _utc(): return datetime.now(timezone.utc).isoformat()
def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(name, command, receipt_dir, log, outputs=(), input_files=(), input_hashes=None,
        remote_identity=None, terminal_evidence=None):
    receipt_dir, log = Path(receipt_dir), Path(log)
    start, end = receipt_dir / f"{name}.start.json", receipt_dir / f"{name}.end.json"
    targets = [start, end, log, *map(Path, outputs)]
    if any(p.exists() for p in targets):
        raise FileExistsError("retry requires new receipt, log, and output paths")
    load = os.getloadavg()[0]
    if load > 10:
        raise RuntimeError(f"one-minute load {load:.2f} exceeds 10")
    receipt_dir.mkdir(parents=True, exist_ok=True); log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    threads = {k: "3" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                  "NUMEXPR_NUM_THREADS", "PRC_THREADS")}
    env.update(threads)
    material_inputs = {str(Path(p)): _sha(p) for p in input_files}
    material_inputs.update(input_hashes or {})
    t0, cpu0 = time.monotonic(), resource.getrusage(resource.RUSAGE_CHILDREN)
    gate = receipt_dir / f"{name}.launch-gate"
    if gate.exists():
        raise FileExistsError(gate)
    read_fd, write_fd = os.pipe()
    with log.open("xb") as stream:
        pid = os.fork()
        if pid == 0:
            os.close(write_fd); os.read(read_fd, 1); os.close(read_fd)
            os.dup2(stream.fileno(), 1); os.dup2(stream.fileno(), 2)
            os.execvpe(command[0], list(command), env)
        os.close(read_fd)
        record = {"utc": _utc(), "command": list(command), "log": str(log),
                  "input_hashes": material_inputs, "threads": threads,
                  "one_minute_load": load, "pid_or_remote_identity": remote_identity or pid}
        start.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        gate.write_text(record["utc"] + "\n")
        os.write(write_fd, b"1"); os.close(write_fd)
        _, wait_status = os.waitpid(pid, 0); status = os.waitstatus_to_exitcode(wait_status)
    cpu1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    missing = [str(p) for p in map(Path, outputs) if not p.exists()]
    out_hashes = {str(p): _sha(p) for p in map(Path, outputs) if p.exists()}
    result = {"utc": _utc(), "status": status, "wall_seconds": time.monotonic()-t0,
              "process_cpu_seconds": (cpu1.ru_utime+cpu1.ru_stime)-(cpu0.ru_utime+cpu0.ru_stime),
              "log_sha256": _sha(log), "output_hashes": out_hashes,
              "json_record_hashes": {str(start): _sha(start)},
              "terminal_evidence": terminal_evidence, "disposition": "pass" if status == 0 and not missing else "fail",
              "missing_outputs": missing}
    end.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if result["disposition"] != "pass": raise RuntimeError(result)
    return result
