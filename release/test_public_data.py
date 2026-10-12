"""Public-data-module controls that do not read competition data."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PublicDataTests(unittest.TestCase):
    def test_import_and_clean_score_work_without_private_sidecar(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "prc"
            package.mkdir()
            shutil.copy2(root / "prc/__init__.py", package / "__init__.py")
            shutil.copy2(root / "prc/data.py", package / "data.py")
            env = dict(os.environ, PYTHONPATH=td, PYTHONDONTWRITEBYTECODE="1")
            code = (
                "import prc.data as d; "
                "assert d.UNREACHED_ROLLOVER == {}; "
                "assert d.excluded_ids('2099-01') == set()"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], env=env, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
