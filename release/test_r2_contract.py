"""Check R2 month isolation, feature exclusions and the caller-owned row guard."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

class R2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, PRC_SOURCE_ROOT=str(ROOT)):
            spec = importlib.util.spec_from_file_location('release_r2_stacker', ROOT / 'experiments/exp-148-short-route/code/stacker.py')
            cls.st = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.st)

    def test_jjx_ancestors_exclude_scored_months_and_their_own_month(self):
        p = self.st.P
        for month in p.R2_MONTHS:
            split = p.jjx(month)
            excluded = set(split.excluded)
            for held in (*p.SCORED_MONTHS, month):
                self.assertTrue(set(p._days(held)) <= excluded)
                self.assertTrue(set(p.WP.month_hold(held).excluded) <= excluded)
            for owner in ('r01', 'r07', 'rfinal'):
                p.consumer(owner).assert_admits(split)
        with self.assertRaises(ValueError):
            p.jjx('2025-05')
        self.assertEqual(p.artifacts_of('jjx-02')[0][1], 'jjx-02')

    def test_r4_adds_three_months_and_keeps_scored_month_isolation(self):
        p = self.st.P
        self.assertEqual(p.R4_MONTHS, ('2025-03', '2025-10', '2025-11'))
        for month in p.R4_MONTHS:
            split = p.jjx(month)
            for held in (*p.SCORED_MONTHS, month):
                self.assertTrue(set(p.WP.month_hold(held).excluded) <= set(split.excluded))
            for owner in ('q01', 'q07', 'qfinal'):
                p.consumer(owner).assert_admits(split)
        train, scored, month = self.st.splits_for('qfinal')
        self.assertEqual(len(train), 8)
        self.assertEqual(scored, [('final', 'final')])
        self.assertIsNone(month)
        cols = self.st.feature_list(pd.DataFrame(columns=['v23_x']), pd.DataFrame(columns=[self.st.ID, 'month']), 'r4')
        self.assertNotIn('ctx_month', cols)
        self.assertFalse(any(c.startswith('v23_') for c in cols))

    def test_r2_drops_month_and_v23_but_keeps_proxy_clocks(self):
        f = pd.DataFrame(columns=['obs_obs_taxi', 'shp_x', 'v23_implied', 'q_n_count'])
        proxy = pd.DataFrame(columns=[self.st.ID, 'month', 'clock_current250_row', 'clock_implied_row'])
        cols = self.st.feature_list(f, proxy, 'r2')
        self.assertNotIn('ctx_month', cols)
        self.assertFalse(any(c.startswith('v23_') for c in cols))
        self.assertIn('clock_implied_row', cols)
        self.assertIn('d_Lp', cols)

    def test_boundary_exceptions_require_caller_config_and_reject_any_other_loss(self):
        mem = pd.DataFrame({self.st.ID: [101, 102, 103]})
        features = pd.DataFrame({self.st.ID: [103]})
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, 'PRC_R2_NO_FEATURES'):
                self.st.drop_no_features(mem, features)
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / 'exceptions.json'
            with patch.dict(os.environ, PRC_R2_NO_FEATURES=str(config)):
                config.write_text(json.dumps({'missing_feature_ids': [101, 102]}))
                kept, lost = self.st.drop_no_features(mem, features)
                self.assertEqual(lost, [101, 102])
                self.assertEqual(kept[self.st.ID].tolist(), [103])
                with self.assertRaisesRegex(SystemExit, 'not only the recorded ones'):
                    self.st.drop_no_features(mem, pd.DataFrame({self.st.ID: [101]}))
                config.write_text(json.dumps({'missing_feature_ids': [101, 101]}))
                with self.assertRaisesRegex(SystemExit, 'two distinct integer IDs'):
                    self.st.r2_no_features()

if __name__ == '__main__':
    unittest.main()
