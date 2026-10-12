"""Exercise revision pinning in a disposable repository; no network calls."""
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from export_release import export, reviewed_commit


class ExportTests(unittest.TestCase):
    def test_full_commit_and_matching_contract_ignore_working_tree(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'repo'
            root.mkdir()
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
            git('init', '-q')
            git('config', 'user.name', 'Synthetic Release Test')
            git('config', 'user.email', 'fake@example.invalid')
            (root / 'release').mkdir()
            (root / 'file.txt').write_text('original\n')
            manifest = 'path\tsha256\tbytes\trole\nfile.txt\t' + hashlib.sha256(b'original\n').hexdigest() + '\t9\ttest\n'
            (root / 'release/MANIFEST.tsv').write_text(manifest)
            git('add', '.')
            git('commit', '-qm', 'synthetic original')
            revision = git('rev-parse', 'HEAD')
            (root / 'file.txt').write_text('changed\n')
            (root / 'release/MANIFEST.tsv').write_text('working manifest drift')
            for bad in ('', 'HEAD', 'main', revision[:8], 'f' * 40, revision + '^',
                        git('hash-object', 'file.txt')):
                with self.assertRaises(ValueError):
                    reviewed_commit(bad, root)
            with mock.patch('export_release.reviewed_commit', wraps=reviewed_commit) as resolve:
                result = export(revision, Path(td) / 'tree', Path(td) / 'trusted.tsv', root)
                self.assertEqual(resolve.call_count, 1)
            self.assertEqual(result['revision'], revision)
            self.assertEqual((Path(td) / 'tree/file.txt').read_text(), 'original\n')
            self.assertEqual((Path(td) / 'trusted.tsv').read_text(), manifest)
            self.assertEqual((Path(td) / 'tree/release/MANIFEST.tsv').read_text(), manifest)
            self.assertFalse((Path(td) / 'tree/.git').exists())


if __name__ == '__main__':
    unittest.main()
