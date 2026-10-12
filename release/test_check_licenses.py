"""Adversarial tests for the standalone licence checker's exact-byte claim."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_licenses  # noqa: E402


class CanonicalLicenseTests(unittest.TestCase):
    def run_check(self, blob: bytes) -> list[str]:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "LICENSE").write_bytes(blob)
            failures: list[str] = []
            with patch.object(check_licenses, "ROOT", root):
                check_licenses.check_license_text(failures.append)
            return failures

    def test_canonical_gplv3_passes(self):
        self.assertEqual(self.run_check((ROOT / "LICENSE").read_bytes()), [])

    def test_one_byte_mutation_with_same_lines_and_markers_fails(self):
        canonical = (ROOT / "LICENSE").read_bytes()
        marker = b"You may charge any price or no price for each copy that you convey"
        self.assertIn(marker, canonical)
        mutated = canonical.replace(marker, marker.replace(b"price", b"pride", 1), 1)
        self.assertEqual(len(mutated), len(canonical))
        self.assertEqual(mutated.count(b"\n"), canonical.count(b"\n"))
        self.assertTrue(all(text.encode() in mutated for text, _ in check_licenses.GPLV3_STRUCTURE))
        failures = self.run_check(mutated)
        self.assertTrue(any("not the canonical GPLv3 text" in item for item in failures), failures)


if __name__ == "__main__":
    unittest.main()
