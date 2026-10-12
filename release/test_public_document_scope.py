"""Top-level release documents state team rules and explain the history boundary."""
import ast
import json
import re
import unittest
from pathlib import Path

from label_glossary import LABEL_RULES, glossary_lines
from release_set import is_released
from reference_policy import scan_references

ROOT = Path(__file__).resolve().parents[1]
HISTORY = '''The experiment folders keep their working records of method development.
In those records and in `prc/` and `scripts/` comments, "the desk" is the
team lead agent and Matt is the team's human lead. Citations of
`assignments/...` name internal task texts that are not part of this release.'''
PROCESS = re.compile(r'(?i)\bdesk\b|\bmatt\b|assignments/|internal decision record|\binbox\b|\bworker\s+`|Sol round|GPT-[0-9.]+ Sol reviewed')


class PublicDocumentScopeTests(unittest.TestCase):
    def test_top_level_prose_has_no_process_text_outside_the_history_note(self):
        paths = [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]
        for path in paths:
            rel = path.relative_to(ROOT).as_posix()
            if not is_released(rel):
                continue
            text = path.read_text()
            if rel == 'README.md':
                self.assertEqual(text.count(HISTORY), 1)
                text = text.replace(HISTORY, '')
            with self.subTest(path=rel):
                self.assertIsNone(PROCESS.search(text))
                self.assertFalse(scan_references(path.read_text(), rel)['excluded_links'])
        self.assertFalse(is_released('docs/RELEASE-CHECKLIST.md'))

    def test_glossary_lines_describe_every_exported_rule_label(self):
        readme = (ROOT / 'README.md').read_text()
        for line in glossary_lines():
            self.assertEqual(readme.splitlines().count(line), 1)
        self.assertIn('[WRITEUP.md](docs/WRITEUP.md)', readme)
        self.assertNotIn('[METHOD.md]', readme)
        # Hex digests, model names, stand IDs and lowercase weather prefixes are data.
        # Complete uppercase labels and lowercase decision-key prefixes are checked.
        label = re.compile(r'(?<!\w)D([1-9][0-9]{0,2})(?![\w-])|(?<!\w)d([1-9][0-9]{0,2})(?=_)')
        for path in [*(ROOT / 'experiments').rglob('*.py'), *(ROOT / 'experiments').rglob('*.sh'),
                     *(ROOT / 'release').glob('*.py')]:
            rel = path.relative_to(ROOT).as_posix()
            if not is_released(rel):
                continue
            if path.suffix == '.sh':
                # Shell strings and both inline and whole-line comments name labels.
                for number in re.findall(r'(?<!\w)D([1-9][0-9]{0,2})(?!\w)', path.read_text()):
                    with self.subTest(path=rel):
                        self.assertIn('D' + number, LABEL_RULES)
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            docs = {id(n.body[0].value) for n in ast.walk(tree)
                    if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and n.body and isinstance(n.body[0], ast.Expr)
                    and isinstance(n.body[0].value, ast.Constant)
                    and isinstance(n.body[0].value.value, str)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                    for first, second in label.findall(node.value):
                        with self.subTest(path=rel, line=node.lineno):
                            self.assertIn('D' + (first or second), LABEL_RULES)
