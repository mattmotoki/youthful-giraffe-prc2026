#!/usr/bin/env python3
"""One-shot controller for the registered exp-091 data process."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent
BINDING = EXP / "exact-binding.json"
STARTED = EXP / "started.json"
EXIT = EXP / "exit.json"
LOCK = EXP / ".launch-lock"
LOG = EXP / "run.log"
PROGRAM = EXP / "screen.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def validate_binding() -> None:
    binding = json.loads(BINDING.read_text())
    expected = {"screen.py": PROGRAM, "controller.py": Path(__file__).resolve()}
    if set(binding) != set(expected):
        raise RuntimeError("binding file list differs")
    for name, path in expected.items():
        if sha256_file(path) != binding[name]:
            raise RuntimeError(f"registered hash differs: {name}")


def main() -> int:
    validate_binding()
    try:
        LOCK.mkdir()
    except FileExistsError:
        raise RuntimeError("launch already claimed")
    if STARTED.exists():
        raise RuntimeError("started receipt already exists")
    load = os.getloadavg()[0]
    if load > 10:
        LOCK.rmdir()
        raise RuntimeError(f"one-minute load exceeds launch limit: {load:.2f}")
    launched = datetime.now(timezone.utc).isoformat()
    with LOG.open("ab", buffering=0) as log:
        process = subprocess.Popen([sys.executable, str(PROGRAM)], cwd=EXP.parents[1],
                                   stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "OMP_NUM_THREADS":"2", "OPENBLAS_NUM_THREADS":"2",
                                        "MKL_NUM_THREADS":"2", "NUMEXPR_NUM_THREADS":"2"})
        atomic_json(STARTED, {"launch_time_utc":launched, "one_minute_load":load, "pid":process.pid})
        code = process.wait()
    atomic_json(EXIT, {"exit_time_utc":datetime.now(timezone.utc).isoformat(), "exit_code":code})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
