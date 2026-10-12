from __future__ import annotations
import importlib.util
import json
import tempfile
from pathlib import Path

P = Path(__file__).with_name("controller.py")
sp = importlib.util.spec_from_file_location("exp091_controller", P)
c = importlib.util.module_from_spec(sp); sp.loader.exec_module(c)


def test_registered_program_and_controller_hashes():
    c.validate_binding()


def test_atomic_receipt_replaces_standard_json():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"receipt.json"; p.write_text("old")
        c.atomic_json(p,{"exit_code":0})
        assert json.loads(p.read_text()) == {"exit_code":0}
        assert not list(Path(td).glob(".receipt.json.*"))


def test_existing_atomic_lock_refuses_before_process_start():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); (root/"lock").mkdir()
        old=(c.LOCK,c.STARTED)
        c.LOCK,c.STARTED=root/"lock",root/"started.json"
        try:
            try: c.main(); assert False
            except RuntimeError as e: assert "already claimed" in str(e)
        finally: c.LOCK,c.STARTED=old


if __name__ == "__main__":
    tests=[v for k,v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests: t()
    print(f"{len(tests)} tests passed")
