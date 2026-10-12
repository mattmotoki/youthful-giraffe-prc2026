from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CONTROLLER = Path(__file__).with_name("controller.py")
BOUND = ["SPEC.md", "source-manifest.json", "score.py", "test_score.py", "controller.py", "test_controller.py"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_tree(directory: str, child_exit: int = 0) -> tuple[Path, Path, dict[str, str]]:
    root = Path(directory) / "repo"
    experiment = root / "experiments" / "synthetic"
    experiment.mkdir(parents=True)
    shutil.copyfile(CONTROLLER, experiment / "controller.py")
    child = (
        "from pathlib import Path\n"
        "import json, os, sys\n"
        "e=Path(os.environ['EXP096_EXPERIMENT_ROOT'])\n"
        "(e/'source-access.marker').write_text('one child access\\n')\n"
        "(e/'phase-receipt.json').write_text(json.dumps({'phase':'synthetic'})+'\\n')\n"
        "print('synthetic stdout', flush=True)\n"
        "print('synthetic stderr', file=sys.stderr, flush=True)\n"
        f"raise SystemExit({child_exit})\n"
    )
    for name in BOUND:
        path = experiment / name
        if name == "controller.py":
            continue
        path.write_text(child if name == "score.py" else name + "\n")
    binding = {name: digest(experiment / name) for name in BOUND}
    (experiment / "exact-binding.json").write_text(json.dumps(binding, sort_keys=True) + "\n")
    return root, experiment, binding


def invoke(root: Path, experiment: Path, load: str = "1") -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "EXP096_EXPERIMENT_ROOT": str(experiment),
        "EXP096_REPO_ROOT": str(root),
        "EXP096_TEST_LOAD": load,
    }
    return subprocess.run(
        [sys.executable, str(experiment / "controller.py")],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_complete_success_and_second_invocation_refusal_before_child_access():
    with tempfile.TemporaryDirectory() as directory:
        root, experiment, binding = synthetic_tree(directory)
        first = invoke(root, experiment)
        assert first.returncode == 0, first.stderr
        assert (experiment / ".launch-lock").is_dir()
        assert (experiment / "source-access.marker").read_text() == "one child access\n"
        marker_stat = (experiment / "source-access.marker").stat().st_mtime_ns
        started = json.loads((experiment / "started.json").read_text())
        phase = json.loads((experiment / "phase-receipt.json").read_text())
        exit_receipt = json.loads((experiment / "exit.json").read_text())
        assert started["binding"] == binding and started["one_minute_load"] == 1.0 and started["pid"] > 0
        assert phase == {"phase": "synthetic"}
        assert exit_receipt["exit_code"] == 0 and exit_receipt["pid"] == started["pid"]
        assert (experiment / "run.stdout.log").read_text() == "synthetic stdout\n"
        assert (experiment / "run.stderr.log").read_text() == "synthetic stderr\n"
        second = invoke(root, experiment)
        assert second.returncode != 0 and "launch already claimed" in second.stderr
        assert (experiment / "source-access.marker").stat().st_mtime_ns == marker_stat


def test_child_failure_has_logs_phase_and_exit_receipt():
    with tempfile.TemporaryDirectory() as directory:
        root, experiment, _ = synthetic_tree(directory, child_exit=7)
        result = invoke(root, experiment)
        assert result.returncode == 7
        assert json.loads((experiment / "exit.json").read_text())["exit_code"] == 7
        assert json.loads((experiment / "phase-receipt.json").read_text()) == {"phase": "synthetic"}
        assert "synthetic stdout" in (experiment / "run.stdout.log").read_text()
        assert "synthetic stderr" in (experiment / "run.stderr.log").read_text()


def test_load_refusal_and_exact_binding_refusal_do_not_start_child():
    with tempfile.TemporaryDirectory() as directory:
        root, experiment, _ = synthetic_tree(directory)
        high_load = invoke(root, experiment, load="11")
        assert high_load.returncode != 0 and "load exceeds" in high_load.stderr
        assert not (experiment / ".launch-lock").exists()
        assert not (experiment / "source-access.marker").exists()
        (experiment / "SPEC.md").write_text("tampered\n")
        bad_binding = invoke(root, experiment)
        assert bad_binding.returncode != 0 and "registered hash differs" in bad_binding.stderr
        assert not (experiment / ".launch-lock").exists()
        assert not (experiment / "source-access.marker").exists()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(len(tests), "tests passed")
