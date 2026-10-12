"""Top-level path citations use export membership, with history out of scope."""
import unittest

from topdoc_paths import missing_path_citations


class TopDocumentPathTests(unittest.TestCase):
    def test_absent_root_and_relative_paths_fail(self):
        for text, target in (('See `docs/BACKLOG.md`.', 'docs/BACKLOG.md'),
                             ('See `../docs/%42ACKLOG.md#part`.', 'docs/BACKLOG.md'),
                             ('See `RELEASE-CHECKLIST.md`.', 'docs/RELEASE-CHECKLIST.md'),
                             ('See `experiments/exp-027-step/SPEC.md`.', 'experiments/exp-027-step/SPEC.md'),
                             ('See `modeling/missing.py`.', 'modeling/missing.py'),
                             ('See `notebooks/missing.ipynb`.', 'notebooks/missing.ipynb'),
                             ('See `catboost_info/missing.json`.', 'catboost_info/missing.json')):
            with self.subTest(text=text):
                result = missing_path_citations(text, 'docs/notes.md', ['docs/notes.md'])
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]['target'], target)
                self.assertEqual(result[0]['line'], 1)

    def test_present_paths_and_directories_pass(self):
        paths = ['README.md', 'docs/WRITEUP.md', 'prc/data.py', 'release/MANIFEST.tsv']
        text = 'See `WRITEUP.md`, `../prc/data.py`, `prc/`, and `release/MANIFEST.tsv`.'
        self.assertFalse(missing_path_citations(text, 'docs/notes.md', paths))
        self.assertFalse(missing_path_citations('See `docs/WRITEUP.md`.', 'README.md', paths))

    def test_missing_directory_fails_even_when_the_source_policy_permits_it(self):
        result = missing_path_citations('See `experiments/missing/`.', 'README.md', ['README.md'])
        self.assertEqual(result[0]['target'], 'experiments/missing')

    def test_history_is_outside_the_top_level_scope(self):
        for path in ('experiments/old/REPORT.md', 'docs/archive/old.md', 'scripts/build.py'):
            self.assertFalse(missing_path_citations('See `docs/BACKLOG.md`.', path, []))

    def test_commands_urls_and_caller_layouts_are_instructions(self):
        text = '''Run `python scripts/build_v32.py --data-root /your/input`.
Use `https://example.invalid/data/raw/ranking.parquet`.
```text
input-root/data/raw/ranking.parquet
`data/raw/submitting.parquet`
```
~~~bash
python scripts/build_v32.py --out-dir artifacts/new
~~~
See `docs/BACKLOG.md`.
'''
        result = missing_path_citations(text, 'README.md', [])
        self.assertEqual(result, [{'line': 10, 'citation': 'docs/BACKLOG.md', 'target': 'docs/BACKLOG.md'}])

    def test_assignment_naming_convention_is_not_a_record_citation(self):
        self.assertFalse(missing_path_citations('Citations of `assignments/...` name internal task texts.', 'README.md', []))
        self.assertTrue(missing_path_citations('See `assignments/task.md`.', 'README.md', []))


if __name__ == '__main__':
    unittest.main()
