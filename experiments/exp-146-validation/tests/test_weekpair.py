"""Tests of prc.weekpair, the week-pair date splitter (exp-146).

    $PRC_SOURCE_ROOT/.venv/bin/python -m unittest discover -s experiments/exp-146-validation/tests -v

Synthetic frames only; no label is read.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from prc import weekpair as WP  # noqa: E402

D = dt.date


def frame(times, phases=None, labels=None):
    n = len(times)
    return pd.DataFrame({
        "MVT_ID_mvt": np.arange(n, dtype="float64") + 1,
        "PHASE_mvt": phases or ["DEP"] * n,
        "MVT_TIME_UTC_mvt": pd.to_datetime(times, utc=True),
        "BLOCK_TIME_UTC_mvt": pd.to_datetime(times, utc=True) - pd.Timedelta(minutes=10),
        "TAXITIME_SEC_mvt": np.asarray(labels if labels is not None else [600.0] * n, dtype="float64"),
    })


class TestBlocks(unittest.TestCase):
    def test_pairs_tile_january_and_july_once(self):
        days = [d for k in WP.PAIRS for d in WP.held_days(k)]
        self.assertEqual(len(days), len(set(days)))
        want = {D(2025, 1, i) for i in range(1, 32)} | {D(2025, 7, i) for i in range(1, 32)}
        self.assertEqual(set(days), want)

    def test_block_bounds_are_the_frozen_ones(self):
        got = {k: (WP.blocks(k)[0][0].day, WP.blocks(k)[0][1].day,
                   WP.blocks(k)[1][0].day, WP.blocks(k)[1][1].day) for k in WP.PAIRS}
        self.assertEqual(got, {1: (1, 5, 1, 6), 2: (6, 12, 7, 13), 3: (13, 19, 14, 20),
                               4: (20, 26, 21, 27), 5: (27, 31, 28, 31)})

    def test_buffers(self):
        self.assertEqual(set(WP.buffer_days(1)), {D(2024, 12, 31), D(2025, 1, 6), D(2025, 6, 30),
                                                  D(2025, 7, 7)})
        self.assertEqual(set(WP.buffer_days(3)), {D(2025, 1, 12), D(2025, 1, 20), D(2025, 7, 13),
                                                  D(2025, 7, 21)})
        self.assertEqual(set(WP.buffer_days(5)), {D(2025, 1, 26), D(2025, 2, 1), D(2025, 7, 27),
                                                  D(2025, 8, 1)})

    def test_pair_of(self):
        days = np.array(["2025-01-05", "2025-01-06", "2025-07-06", "2025-07-07", "2025-07-31",
                         "2025-02-01", "2025-12-31"], dtype="datetime64[D]")
        self.assertEqual(WP.pair_of(days).tolist(), [1, 2, 1, 2, 5, 0, 0])


class TestSplits(unittest.TestCase):
    def test_outer_excludes_held_and_buffers(self):
        for k in WP.PAIRS:
            s = WP.outer(k)
            self.assertEqual(set(s.held), {d.isoformat() for d in WP.held_days(k)})
            self.assertEqual(set(s.excluded), {d.isoformat() for d in WP.held_days(k) + WP.buffer_days(k)})

    def test_inner_excludes_both_pairs(self):
        for k in WP.PAIRS:
            self.assertEqual(len(WP.inner_rotation(k)), 4)
            for j in WP.inner_rotation(k):
                s = WP.inner(k, j)
                self.assertTrue(set(WP.outer(k).excluded) <= set(s.excluded))
                self.assertTrue(set(WP.outer(j).excluded) <= set(s.excluded))
                buf_k = {d.isoformat() for d in WP.buffer_days(k)}
                self.assertEqual(set(s.held), set(WP.outer(j).held) - buf_k)
        with self.assertRaises(ValueError):
            WP.inner(2, 2)

    def test_calibration_rows_are_label_rows_of_the_outer_fold(self):
        """Review 2026-09-30: fold k's calibration must not read fold k's buffer days."""
        cut = 0
        for k in WP.PAIRS:
            for j in WP.inner_rotation(k):
                s = WP.inner(k, j)
                self.assertFalse(set(s.held) & set(WP.outer(k).excluded), (k, j))
                self.assertTrue(set(s.held) <= set(WP.outer(j).held))
                cut += len(WP.outer(j).held) - len(s.held)
                self.assertEqual(s.excluded, WP.inner(j, k).excluded)   # one fit serves both
        self.assertEqual(cut, 16)                                        # 8 rotations x 2 days
        self.assertEqual(set(WP.inner(2, 1).held) & {"2025-01-05", "2025-07-06"}, set())
        f = frame(["2025-01-05 10:00+00:00", "2025-01-04 10:00+00:00", "2025-07-06 10:00+00:00"])
        self.assertEqual(WP.inner(2, 1).score_mask(f).tolist(), [False, True, False])
        self.assertEqual(WP.outer(2).label_mask(f).tolist(), [False, True, False])

    def test_admits_is_the_label_isolation_rule(self):
        o3, i31 = WP.outer(3), WP.inner(3, 1)
        self.assertTrue(o3.admits(i31))           # calibration for fold 3 from inner fits
        self.assertTrue(o3.admits(o3))
        self.assertFalse(i31.admits(o3))          # an outer-3 artifact read pair 1's labels
        self.assertFalse(o3.admits(WP.outer(4)))  # read pair 3's labels
        self.assertFalse(o3.admits(WP.FINAL))     # an all-2025 fit read everything
        self.assertTrue(WP.FINAL.admits(o3))
        with self.assertRaises(WP.LabelLeak):
            o3.assert_admits(WP.FINAL, "base model")
        with self.assertRaises(WP.LabelLeak):
            i31.assert_admits(o3.manifest(), "encoding")

    def test_masks_use_utc_take_off_date(self):
        f = frame(["2025-01-05 23:59:59+00:00",   # pair 1, held
                   "2025-01-06 00:00:00+00:00",   # pair 2 held; pair 1 buffer
                   "2025-01-06 00:30:00+01:00",   # 2025-01-05 23:30 UTC: pair 1
                   "2025-03-10 12:00:00+00:00",   # training day
                   "2025-01-03 10:00:00+00:00"],  # pair 1, but an arrival
                  phases=["DEP", "DEP", "DEP", "DEP", "ARR"])
        s1 = WP.outer(1)
        self.assertEqual(s1.score_mask(f).tolist(), [True, False, True, False, False])
        self.assertEqual(s1.label_mask(f).tolist(), [False, False, False, True, False])
        self.assertEqual(s1.buffer_mask(f).tolist(), [False, True, False, False, False])
        s2 = WP.outer(2)
        self.assertEqual(s2.score_mask(f).tolist(), [False, True, False, False, False])
        self.assertEqual(s2.label_mask(f).tolist(), [False, False, False, True, False])

    def test_assert_training_frame(self):
        f = frame(["2025-01-03 10:00+00:00", "2025-03-10 10:00+00:00"], labels=[np.nan, 500.0])
        with self.assertRaises(WP.LabelLeak):          # no target named: any held departure fails
            WP.outer(1).assert_training_frame(f)
        WP.outer(1).assert_training_frame(f, target="TAXITIME_SEC_mvt")   # held row unlabelled
        WP.outer(1).assert_training_frame(f.iloc[1:])
        f.loc[0, "TAXITIME_SEC_mvt"] = 700.0
        with self.assertRaises(WP.LabelLeak):
            WP.outer(1).assert_training_frame(f, target="TAXITIME_SEC_mvt")

    def test_assert_training_frame_target_in_another_column(self):
        """Review 2026-09-30: a held-day target in column y with TAXITIME blank must fail."""
        f = frame(["2025-01-08 10:00+00:00", "2025-03-10 10:00+00:00"], labels=[np.nan, np.nan])
        f["y"] = [640.0, 500.0]
        for call in (lambda: WP.outer(2).assert_training_frame(f),
                     lambda: WP.outer(2).assert_training_frame(f, target="y"),
                     lambda: WP.outer(2).assert_training_frame(f, target=f["y"].to_numpy())):
            with self.assertRaises(WP.LabelLeak):
                call()
        f.loc[0, "y"] = np.nan
        WP.outer(2).assert_training_frame(f, target="y")
        with self.assertRaises(ValueError):
            WP.outer(2).assert_training_frame(f, target=[1.0])

    def test_manifest_round_trip_and_tamper(self):
        m = WP.inner(2, 5).manifest()
        self.assertEqual(WP.Split.from_manifest(json.loads(json.dumps(m))), WP.inner(2, 5))
        bad = dict(m, excluded=m["excluded"][:-1])
        with self.assertRaises(WP.LabelLeak):
            WP.Split.from_manifest(bad)
        with self.assertRaises(WP.LabelLeak):
            WP.Split.from_manifest(dict(m, protocol="month-folds"))

    def test_month_hold(self):
        s = WP.month_hold("2025-07")
        self.assertEqual(len(s.held), 31)
        self.assertEqual(set(s.excluded) - set(s.held), {"2025-06-30", "2025-08-01"})

    def test_training_days_per_outer_fold(self):
        year = {D(2025, 1, 1) + dt.timedelta(days=i) for i in range(365)}
        got = {k: len(year - {D.fromisoformat(x) for x in WP.outer(k).excluded}) for k in WP.PAIRS}
        self.assertEqual(got, {1: 351, 2: 347, 3: 347, 4: 347, 5: 352})


class TestJoin(unittest.TestCase):
    def setUp(self):
        self.ids = pd.DataFrame({"MVT_ID_mvt": [1, 2, 3, 4, 5, 6], "pair": [1, 2, 3, 4, 5, 5]})
        self.preds = {k: pd.DataFrame({"MVT_ID_mvt": self.ids.MVT_ID_mvt[self.ids.pair == k],
                                       "pred": 100.0 * k}) for k in WP.PAIRS}

    def test_join(self):
        s = WP.join_outer(self.preds, self.ids)
        self.assertEqual(s.sort_index().tolist(), [100, 200, 300, 400, 500, 500])

    def test_wrong_fold_or_missing_row(self):
        p = dict(self.preds)
        p[2] = pd.DataFrame({"MVT_ID_mvt": [3], "pred": [1.0]})   # pair 3's row from fold 2
        with self.assertRaises(WP.LabelLeak):
            WP.join_outer(p, self.ids)
        p = dict(self.preds)
        p[5] = p[5].iloc[:1]
        with self.assertRaises(ValueError):
            WP.join_outer(p, self.ids)


class TestContext(unittest.TestCase):
    def test_context_frame_blanks_departure_labels_only(self):
        f = frame(["2025-01-03 10:00+00:00", "2025-01-03 10:05+00:00"], phases=["DEP", "ARR"],
                  labels=[600.0, 420.0])
        with self.assertRaises(WP.LabelLeak):
            WP.assert_context_masked(f)
        c = WP.context_frame(f)
        WP.assert_context_masked(c)
        self.assertTrue(np.isnan(c.TAXITIME_SEC_mvt[0]))
        self.assertEqual(c.TAXITIME_SEC_mvt[1], 420.0)     # arrival taxi-in stays an input
        self.assertTrue(pd.isna(c.BLOCK_TIME_UTC_mvt[0]))
        self.assertFalse(pd.isna(c.BLOCK_TIME_UTC_mvt[1]))
        self.assertEqual(f.TAXITIME_SEC_mvt[0], 600.0)     # the input frame is untouched


class TestClockLoss(unittest.TestCase):
    def test_dark_days(self):
        self.assertEqual(WP.dark_days("EDDF", "2025-01", 0.0), ())
        self.assertEqual(len(WP.dark_days("LIRF", "2025-07", 1.0)), 31)
        a = WP.dark_days("EDDF", "2025-07", 0.2)
        self.assertEqual(a, WP.dark_days("EDDF", "2025-07", 0.2))
        self.assertEqual(len(a), int(np.floor(0.2 * 31 + 0.5)))
        self.assertTrue(set(a) <= set(WP.dark_days("EDDF", "2025-07", 0.5)))   # nested rates
        self.assertEqual(len(WP.dark_days("EDDF", "2025-02", 0.5)), 14)
        with self.assertRaises(ValueError):
            WP.dark_days("EDDF", "2025-07", 1.5)

    def test_removal_rate_never_adds(self):
        self.assertEqual(WP.removal_rate(0.4, 0.9), 0.0)
        self.assertEqual(WP.removal_rate(0.0, 0.5), 0.0)
        self.assertAlmostEqual(WP.removal_rate(0.8, 0.2), 0.75)
        self.assertEqual(WP.removal_rate(0.9, 0.0), 1.0)

    def test_training_rate_season(self):
        self.assertEqual([WP.training_rate_month(m) for m in range(1, 13)],
                         [1, 1, 1, 7, 7, 7, 7, 7, 7, 1, 1, 1])


class TestFrozen(unittest.TestCase):
    def test_fingerprint_matches_the_frozen_manifest(self):
        man = json.loads((ROOT / "experiments/exp-146-validation/frozen/MANIFEST.json").read_text())
        self.assertEqual(WP.fingerprint(), man["splits_fingerprint"])


if __name__ == "__main__":
    unittest.main()
