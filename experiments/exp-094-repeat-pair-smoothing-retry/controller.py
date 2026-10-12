from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXP = Path(os.environ.get("EXP094_EXPERIMENT_ROOT", Path(__file__).resolve().parent)).resolve()
ROOT = Path(os.environ.get("EXP094_REPO_ROOT", EXP.parents[1])).resolve()
BINDING = EXP / "exact-binding.json"
STARTED = EXP / "started.json"
EXIT = EXP / "exit.json"
LOCK = EXP / ".launch-lock"
STDOUT = EXP / "run.stdout.log"
STDERR = EXP / "run.stderr.log"
PROGRAM = EXP / "score.py"
BOUND = ["SPEC.md", "source-manifest.json", "score.py", "test_score.py", "controller.py", "test_controller.py"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    data = (json.dumps(obj, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        os.fsync(directory)
        os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_binding() -> dict[str, str]:
    binding = json.loads(BINDING.read_text())
    if set(binding) != set(BOUND):
        raise RuntimeError("binding file list differs")
    for name in BOUND:
        if sha(EXP / name) != binding[name]:
            raise RuntimeError("registered hash differs: " + name)
    return binding


def claim_launch() -> None:
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError("launch already claimed") from error
    if STARTED.exists() or EXIT.exists():
        raise RuntimeError("receipt already exists")


def launch_load() -> float:
    forced = os.environ.get("EXP094_TEST_LOAD")
    return float(forced) if forced is not None else os.getloadavg()[0]


def main() -> int:
    binding = validate_binding()
    load = launch_load()
    if load > 10:
        raise RuntimeError(f"one-minute load exceeds launch limit: {load:.2f}")
    claim_launch()
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "EXP094_EXPERIMENT_ROOT": str(EXP),
        "EXP094_REPO_ROOT": str(ROOT),
    }
    with STDOUT.open("ab", buffering=0) as stdout, STDERR.open("ab", buffering=0) as stderr:
        process = subprocess.Popen(
            [sys.executable, str(PROGRAM)], cwd=ROOT, stdout=stdout, stderr=stderr, env=env
        )
        atomic_json(
            STARTED,
            {
                "binding": binding,
                "launch_time_utc": datetime.now(timezone.utc).isoformat(),
                "one_minute_load": load,
                "pid": process.pid,
                "stderr_log": str(STDERR.relative_to(ROOT)),
                "stdout_log": str(STDOUT.relative_to(ROOT)),
            },
        )
        code = process.wait()
    atomic_json(
        EXIT,
        {"exit_code": code, "exit_time_utc": datetime.now(timezone.utc).isoformat(), "pid": process.pid},
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
