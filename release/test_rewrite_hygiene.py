"""The prose rewrite must not introduce repeated words or phrases."""
import json
import unittest
from pathlib import Path

from rewrite_hygiene import doubled_phrases, inspect_rewritten

ROOT = Path(__file__).resolve().parents[1]


class RewriteHygieneTests(unittest.TestCase):
    def test_rewritten_lines_have_no_doubled_word_or_phrase(self):
        scope = json.loads((ROOT / 'release/REWRITE-LINES.json').read_text())
        self.assertTrue(scope['files'])
        self.assertEqual(inspect_rewritten(ROOT, scope), [])

    def test_damage_examples_fail_and_clean_or_untouched_text_passes(self):
        for text in ('Frozen frozen gain gate gate',
                     'frozen gain gate gate',
                     'nested validation rule/nested validation rule',
                     'allocation allocation', 'record the model record the model'):
            with self.subTest(text=text):
                self.assertTrue(doubled_phrases(text))
        self.assertFalse(doubled_phrases('The frozen gain gate and nested validation rule.'))
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'note.md').write_text('untouched repeated repeated\nrewritten gain gate\n')
            scope = {'files': [{'path': 'note.md', 'lines': [2], 'expected_new_text': {'2': 'rewritten gain gate'}}]}
            self.assertEqual(inspect_rewritten(root, scope), [])
            (root / 'note.md').write_text('untouched repeated repeated\nrewritten gain gate gate\n')
            self.assertTrue(inspect_rewritten(root, scope))

    def test_all_imported_text_changes_are_covered_by_the_damage_test(self):
        from rewrite_hygiene import inspect_imported
        contract = json.loads((ROOT / 'release/IMPORTED-SOURCES.json').read_text())
        scope = json.loads((ROOT / 'release/REWRITE-LINES.json').read_text())
        self.assertEqual(inspect_imported(ROOT, contract, scope), [])
        self.assertEqual(inspect_rewritten(ROOT, scope), [])

    def test_comments_docstrings_new_lines_and_deletions_need_scope(self):
        from tempfile import TemporaryDirectory
        from rewrite_hygiene import line_hashes, changed_import_lines, inspect_imported
        original = '# member D1\n"""CPU only."""\nx = 1\n'
        variants = ('# member D1-winter\n"""CPU only."""\nx = 1\n',
                    '# member D1\n"""Changed docstring."""\nx = 1\n',
                    original + '# added comment\n',
                    '# member D1\nx = 1\n')
        with TemporaryDirectory() as folder:
            root = Path(folder);path = root / 'prc/foldserve.py';path.parent.mkdir()
            contract = {'files': [{'path': 'prc/foldserve.py', 'original_line_sha256': line_hashes(original)}]}
            for text in variants:
                with self.subTest(text=text):
                    path.write_text(text)
                    changed = sorted(changed_import_lines(line_hashes(original), line_hashes(text)))
                    self.assertTrue(changed)
                    stale = {'files': [{'path': 'prc/foldserve.py', 'lines': []}]}
                    self.assertTrue(inspect_imported(root, contract, stale))
                    covered = {'files': [{'path': 'prc/foldserve.py', 'lines': changed}]}
                    self.assertEqual(inspect_imported(root, contract, covered), [])
                    damaged = text + '# gain gate gate\n'
                    path.write_text(damaged)
                    covered['files'][0]['lines'] = sorted(changed_import_lines(line_hashes(original), line_hashes(damaged)))
                    self.assertEqual(inspect_imported(root, contract, covered), [])
                    self.assertTrue(inspect_rewritten(root, covered))
            path.write_text(original)
            self.assertTrue(inspect_imported(root, {'files': []}, {'files': []}))

    def test_wrong_word_on_a_listed_line_fails_and_missing_expectation_fails(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            root = Path(td);p = root / 'driver.py'
            expected = '# D1-winter accepts cpu only'
            scope = {'files': [{'path': 'driver.py', 'lines': [1], 'expected_new_text': {'1': expected}}]}
            p.write_text(expected + '\n')
            self.assertEqual(inspect_rewritten(root, scope), [])
            p.write_text('# the recorded team rule accepts cpu only\n')
            self.assertTrue(inspect_rewritten(root, scope))
            p.write_text(expected + '\n');scope['files'][0].pop('expected_new_text')
            self.assertTrue(inspect_rewritten(root, scope))
