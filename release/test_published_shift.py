"""The floor and unsupported-cell rule preserve the original mean correction."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from published_shift import apply
from release_set import is_released

class PublishedShiftTests(unittest.TestCase):
    def test_shrinkage_uses_all_rows_and_does_not_compensate_held_share(self):
        p = dict(mode='ranking_inspection', airport='A', month=1, lambda_value=.5, corrected_mean_s=160.)
        base = np.array([100., 100., 100.])
        out, cells = apply(['A']*3, [1]*3, base, [1, 1, 0], [p])
        np.testing.assert_allclose(out, [130., 130., 100.], rtol=0, atol=1e-12)
        self.assertAlmostEqual(cells[0]['served_mean_s'], 120.)
        # Floor is applied within the bisection, rather than after a mean shift.
        p['corrected_mean_s'] = 60.;p['lambda_value'] = 1.
        out, cells = apply(['A']*3, [1]*3, [60., 140., 100.], [1, 1, 0], [p])
        np.testing.assert_array_equal(out, [60., 60., 100.])
        self.assertEqual(cells[0]['floor_rows'], 2)

    def test_invalid_mask_parameters_and_missing_cells_fail(self):
        p = dict(mode='ranking_inspection', airport='A', month=1, lambda_value=.5, corrected_mean_s=80.)
        for mask, params, airports in [([.5], [p], ['A']), ([1], [p,p], ['A']),
                                      ([1], [dict(p, lambda_value=2.)], ['A']), ([1], [p], ['B'])]:
            with self.subTest(mask=mask, params=params, airports=airports):
                with self.assertRaises(ValueError):apply(airports, [1], [100.], mask, params)

    def test_parameters_ship_and_workbooks_and_worker_evidence_do_not(self):
        self.assertTrue(is_released('experiments/exp-151-published-means/frozen-parameters.json'))
        for path in ('experiments/exp-151-published-means/results/step-d/frozen-parameters.json',
                     'experiments/exp-151-published-means/results/step-d/frozen-cells.csv',
                     'experiments/exp-151-published-means/source/manifest.json',
                     'experiments/exp-151-published-means/source/Taxi-Out_Additional_Time.xlsx',
                     'docs/Taxi-Out_Additional_Time.xlsx', 'experiments/exp-149-release-v37/REPORT.md'):
            self.assertFalse(is_released(path), path)
