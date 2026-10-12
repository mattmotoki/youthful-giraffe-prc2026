"""Required privacy exclusions and the public stage import boundary."""
import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

from release_set import is_released

ROOT = Path(__file__).resolve().parents[1]


class SelectionTests(unittest.TestCase):
    def test_every_required_private_tree_and_diary_is_excluded(self):
        paths = (
            'docs/board/snapshot.json', 'docs/status/restart/worker.md',
            'docs/scout/message.txt', 'docs/assignments/release.md',
            'docs/DECISIONS.md', 'docs/LEDGER.md', 'docs/SUBMISSIONS.md',
            'docs/DESK_CURRENT.md', 'docs/RELEASE-CHECKLIST.md', '.desk/messages.json',
            'docs/archive/old-release.md', 'prc/_private_clean_score_rows.py',
            'experiments/exp-149-release-v35/check_source_parity.py',
            'experiments/exp-148-short-route/results/ranking-B50.parquet',
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertFalse(is_released(path))

    def test_source_arrays_public_records_and_stage_fitters_are_included(self):
        paths = (
            'scripts/build_v32.py', 'release/V32-REPLAY.json',
            'scripts/build_entry.py', 'release/ENTRIES.json',
            'release/saved-vectors/v35.json',
            'experiments/exp-148-short-route/SPEC.md',
            'experiments/exp-148-short-route/code/serve_blend.py',
            'experiments/exp-147-foldserve/code/members/exp008_wp.py',
            'release/V23-FOLD-BASE.json', 'release/saved-vectors/v32.json',
            'docs/PUBLIC-SUBMISSIONS.md', 'docs/PUBLIC-MODEL.json',
            'experiments/exp-134-proxies-in-learner/code/fit_exp134.py',
            'experiments/exp-137-withclock-proxies/kaggle/run-v4/final/kernel.py',
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(is_released(path))

    def test_stage_preparers_have_no_machine_or_moving_ref_default(self):
        paths = (
            'experiments/exp-127-clock-precision/build_stand_geometry.py',
            'experiments/exp-127-clock-precision/prepare_fold_frames.py',
            'experiments/exp-127-clock-precision/prepare_ranking_frame.py',
            'experiments/exp-127-clock-precision/rederive/prepare_clock_keys.py',
            'experiments/exp-130-ranking-exposure/code/prepare.py',
            'experiments/exp-134-proxies-in-learner/code/prepare_inputs.py',
            'experiments/exp-136-wider-proxies/code/prepare_inputs.py',
            'experiments/exp-137-withclock-proxies/code/prepare_inputs.py',
            'experiments/exp-137-withclock-proxies/code/prepare_kernels.py',
        )
        for path in paths:
            with self.subTest(path=path):
                source = (ROOT / path).read_text()
                ast.parse(source)
                self.assertNotIn('/Users/', source)
                self.assertNotIn('/tmp/', source)
                self.assertNotIn('origin/', source)

    def test_exp130_imports_and_packages_without_private_sidecar(self):
        path = ROOT / 'experiments/exp-130-ranking-exposure/code/prepare.py'
        self.assertNotIn('_private_clean_score_rows', path.read_text())
        code = (
            'import runpy,sys; sys.argv=["prepare.py"]; '
            f'runpy.run_path({str(path)!r},run_name="public_import_check")'
        )
        env = dict(os.environ, PRC_ROOT=str(ROOT), PRC_KAGGLE_USER='fake-account',
                   PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
