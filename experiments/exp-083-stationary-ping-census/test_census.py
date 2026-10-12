import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PATH = Path(__file__).with_name("census.py")
spec = importlib.util.spec_from_file_location("census", PATH)
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


def summary(**changes):
    row = dict(icao="EDDF", callsign_modal="ab-123", callsigns_all="AB123", kind="other",
               seg_start=0.0, seg_end=100.0, first_motion=np.nan, obs_offblock=np.nan,
               ends_stationary=True, stand_lat=50.0, stand_lon=8.0, n_ground_points=2)
    row.update(changes)
    return pd.DataFrame([row], columns=census.ALLOWED["summary"])


def rows(times=(160.0,)):
    frame = pd.DataFrame({"MVT_ID_mvt": [f"private-{i}" for i in range(len(times))],
                          "ADEP_mvt": ["EDDF"] * len(times), "MVT_TIME_UTC_mvt": list(times),
                          "CALLSIGN_flt": ["AB123"] * len(times), "FLIGHT_mvt": ["x"] * len(times),
                          "STAND_mvt": ["A"] * len(times), "RUNWAY_mvt": ["25"] * len(times),
                          "obs_has_clock": [False] * len(times), "noac": [False] * len(times),
                          "prediction": [500.0] * len(times), "_callsign": ["AB123"] * len(times),
                          "_row_token": list(range(len(times))), "month": ["2026-01"] * len(times)})
    return frame


class CensusTests(unittest.TestCase):
    def test_projection_fails_closed(self):
        with self.assertRaises(ValueError):
            census.validate_columns("ranking", ["MVT_ID_mvt", "label"])
        with self.assertRaises(ValueError):
            census.validate_columns("ranking", census.ALLOWED["ranking"], census.ALLOWED["ranking"] + ["x"])
        with self.assertRaises(ValueError):
            census.validate_columns("ranking", ["TAXITIME_SEC_mvt"])

    def test_stationary_and_callsign_ambiguity_exclusions(self):
        self.assertEqual(len(census.eligible_segments(summary())), 1)
        self.assertTrue(census.eligible_segments(summary(first_motion=90.0)).empty)
        self.assertTrue(census.eligible_segments(summary(obs_offblock=95.0)).empty)
        self.assertTrue(census.eligible_segments(summary(callsigns_all="AB123|CD456")).empty)
        self.assertTrue(census.eligible_segments(summary(kind="arrival")).empty)

    def test_exact_window_edges_and_ambiguous_ranking_rows(self):
        seg = census.eligible_segments(summary())
        self.assertEqual(len(census.match(rows((160.0,)), seg, 60, 10_800)), 1)
        self.assertEqual(len(census.match(rows((10_900.0,)), seg, 60, 10_800)), 1)
        self.assertTrue(census.match(rows((159.999,)), seg, 60, 10_800).empty)
        self.assertTrue(census.match(rows((10_900.001,)), seg, 60, 10_800).empty)
        self.assertTrue(census.match(rows((160.0, 160.0)), seg, 60, 10_800).empty)

    def test_both_controls_and_shuffle_are_deterministic(self):
        segments = census.eligible_segments(summary())
        self.assertEqual(len(census.match(rows((160.0,)), segments, 60, 10_800)), 1)
        frame = rows((86_500.0,))
        self.assertEqual(len(census.match(frame, segments, 86_400, 97_200)), 1)
        self.assertTrue(census.shuffled_callsigns(frame).equals(census.shuffled_callsigns(frame)))

    def test_aggregate_formula_and_no_id_output(self):
        frame = rows((160.0,))
        accepted = frame.assign(U=120.0)
        result = census.aggregate(frame, accepted)
        gain = (500.0 - 120.0 - 300.0) ** 2 / census.DENOMINATOR
        self.assertEqual(result["TOTAL"]["margins_s"]["300"]["G_lb_s2"], gain)
        self.assertEqual(result["TOTAL"]["margins_s"]["300"]["above_count"], 1)
        rendered = str(result)
        self.assertNotIn("MVT_ID", rendered)
        self.assertNotIn("private-0", rendered)


if __name__ == "__main__":
    unittest.main()
