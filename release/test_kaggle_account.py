"""Account routing is lazy and gives a useful error at use."""
import ast
import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class KaggleAccountTests(unittest.TestCase):
    def test_account_helpers_read_at_use_and_reject_missing_or_blank_accounts(self):
        files = [p for p in (ROOT / 'experiments').rglob('*.py')
                 if 'exp-149' not in str(p) and 'def kaggle_user()' in p.read_text()]
        self.assertTrue(files)
        for path in files:
            with self.subTest(path=str(path.relative_to(ROOT))):
                tree = ast.parse(path.read_text())
                helper = next(n for n in tree.body
                              if isinstance(n, ast.FunctionDef) and n.name == 'kaggle_user')
                scope = {'os': os}
                with patch.dict(os.environ, {}, clear=True):
                    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), scope)
                    for value in ('', '   '):
                        os.environ['PRC_KAGGLE_USER'] = value
                        with self.assertRaisesRegex(RuntimeError, 'Set PRC_KAGGLE_USER'):
                            scope['kaggle_user']()
                    os.environ['PRC_KAGGLE_USER'] = ' first-account '
                    self.assertEqual(scope['kaggle_user'](), 'first-account')
                    os.environ['PRC_KAGGLE_USER'] = 'second-account'
                    self.assertEqual(scope['kaggle_user'](), 'second-account')

    def test_route_tables_import_without_account_and_use_current_account(self):
        examples = (
            ('exp-137-withclock-proxies/code/prepare_kernels.py', 'slugs', 'control'),
            ('exp-137-withclock-proxies/code/combine_exp137_arms.py', 'expected_kernel_metadata_ids', 'control'),
            ('exp-136-wider-proxies/code/combine_exp136_arms.py', 'kernel_url_slugs', 'control'),
            ('exp-136-wider-proxies/code/check_exp136_arm_inputs.py', 'exp136_dataset', None),
            ('exp-127-clock-precision/rederive/prepare_clock_keys.py', 'point_kernel_versions', None),
            ('exp-088-hard-sparse-transport/code/controller.py', 'dataset_ref', None),
        )
        with patch.dict(os.environ, {}, clear=True):
            for rel, function, key in examples:
                with self.subTest(path=rel):
                    scope = runpy.run_path(str(ROOT / 'experiments' / rel))
                    os.environ.pop('PRC_KAGGLE_USER', None)
                    with self.assertRaisesRegex(RuntimeError, 'Set PRC_KAGGLE_USER'):
                        scope[function]()
                    os.environ['PRC_KAGGLE_USER'] = 'test-account'
                    value = scope[function]()
                    value = value[key] if key else next(iter(value)) if isinstance(value, dict) else value
                    self.assertTrue(value.startswith('test-account/'))

    def test_entry_kernels_import_without_account_or_input_mount(self):
        files = [p for p in (ROOT / 'experiments').rglob('kernel.py')
                 if 'def run_kernel()' in p.read_text()]
        with patch.dict(os.environ, {}, clear=True):
            for path in files:
                with self.subTest(path=str(path.relative_to(ROOT))):
                    scope = runpy.run_path(str(path))
                    self.assertTrue(callable(scope['run_kernel']))
                    self.assertNotIn('PRC_KERNEL_SLUG', os.environ)
