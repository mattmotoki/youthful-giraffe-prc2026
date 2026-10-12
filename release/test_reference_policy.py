"""The source boundary applies to public links and narrative citations."""
import json
import base64
import gzip
import struct
import unittest

from reference_policy import scan_references
from label_glossary import glossary_lines


class ReferencePolicyTests(unittest.TestCase):
    def test_private_status_assignment_and_diary_links_fail(self):
        for target in ('status/restart/message.md', 'assignments/resume.md',
                       'DECISIONS.md', 'archive/old.md', '/docs/status',
                       'docs/status/private.md'):
            with self.subTest(target=target):
                text = '[record](' + target + ')'
                self.assertTrue(scan_references(text, 'docs/input-rule.md')['excluded_links'])

    def test_relative_encoded_reference_html_and_image_links_fail(self):
        targets = ('../docs/status/private.md', '../docs/%73tatus/private.md',
                   '../docs/assignments/private.md#part', '../docs/DECISIONS.md?raw=1')
        for target in targets:
            for text in ('[read](' + target + ')', '[read]: ' + target,
                         '<a href="' + target + '">read</a>', '![' + 'image](' + target + ')'):
                with self.subTest(text=text):
                    self.assertTrue(scan_references(text, 'release/notes.md')['excluded_links'])

    def test_external_public_and_fragment_links_pass(self):
        text = '[rule](input-rule.md) [part](#rule) [external](https://example.invalid/page)'
        self.assertFalse(scan_references(text, 'docs/WRITEUP.md')['excluded_links'])

    def test_decision_citations_fail_in_prose_comments_docstrings_and_json(self):
        label = 'D' + '119'
        examples = ((label, 'docs/notes.md'), ('Use ' + label + '.', 'docs/notes.txt'),
                    ('# Apply ' + label + '\nx = 1\n', 'example.py'),
                    ('"""Apply ' + label + '."""\nx = 1\n', 'example.py'),
                    (json.dumps({'note': 'Apply ' + label + '.'}), 'note.json'),
                    ('| claim | Apply ' + label + '. |', 'docs/notes.md'),
                    ('| decision | ' + label + ' |', 'docs/notes.md'))
        for text, path in examples:
            with self.subTest(path=path):
                self.assertIn(label, scan_references(text, path)['decision_references'])

    def test_valid_model_names_source_identifiers_and_stand_cells_pass(self):
        cases = (('Use `D1-winter`.', 'docs/notes.md'),
                 ('| airport | stand | count |\n| LSZH | D11 | 100 |', 'geometry.md'),
                 ('D18 = 1\nx = D18 + 2\n', 'example.py'))
        for text, path in cases:
            with self.subTest(path=path):
                self.assertFalse(scan_references(text, path)['decision_references'])

    def test_decision_labels_in_executable_strings_and_keys_remain_opaque(self):
        source = 'flag = {"d107_pass": False}\nmessage = "D132 positive-point override"\n'
        self.assertFalse(scan_references(source, 'example.py')['decision_references'])

    def test_only_exact_public_glossary_definitions_are_allowed_in_readme(self):
        for line in glossary_lines():
            self.assertFalse(scan_references(line, 'README.md')['decision_references'])
            self.assertTrue(scan_references(line, 'docs/notes.md')['decision_references'])
            self.assertTrue(scan_references(line + ' (private record)', 'README.md')['decision_references'])

    def test_encoded_array_is_data_and_invalid_encoding_fails_closed(self):
        self.assertFalse(scan_references(json.dumps({'encoding': 'gzip-base64',
                                                    'dtype': '<f8', 'rows': 1,
                                                    'data': base64.b64encode(gzip.compress(struct.pack('<d', 1.0))).decode()}), 'array.json')['decision_references'])
        with self.assertRaises(ValueError):
            scan_references(json.dumps({'encoding': 'gzip-base64', 'data': 'Use ' + 'D119.'}), 'array.json')

    def test_python_code_brackets_are_not_links_but_rendered_strings_are(self):
        source = 'answer = funcs[0](fold, other)\n'
        self.assertFalse(scan_references(source, 'example.py')['excluded_links'])
        text = '[read](' + '../docs/status/private.md' + ')'
        source = 'message = ' + repr(text)
        self.assertTrue(scan_references(source, 'release/example.py')['excluded_links'])

    def test_decision_token_after_a_long_prefix_is_checked(self):
        self.assertTrue(scan_references('x' * 1_100_000 + '\nApply ' + 'D119.', 'notes.md')['decision_references'])


class ShellNarrativeScopeTests(unittest.TestCase):
    def test_documented_whole_line_shell_scope(self):
        label = 'D' + '119'
        self.assertIn(label, scan_references('  # Apply ' + label + '\n', 'script.sh')['decision_references'])
        self.assertFalse(scan_references('value=1 # Apply ' + label + '\n', 'script.sh')['decision_references'])


if __name__ == '__main__':
    unittest.main()
