"""exp-098 registered-path tests.  Synthetic vectors only; no repository label is read."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import q09v22 as Q  # noqa: E402


def toy(n=600, seed=3):
    rng = np.random.default_rng(seed)
    regime = np.array(["nm_bulk"] * (n // 2) + ["nm_late"] * (n // 4)
                      + ["nonm"] * (n // 8) + ["noac"] * (n - n // 2 - n // 4 - n // 8))
    day = np.repeat(np.arange(n // 20), 20)[:n].astype("int64")
    ids = np.arange(n, dtype="float64")
    p10 = 600.0 + rng.normal(0, 50, n)
    arm = p10 + rng.normal(0, 10, n)
    return regime, day, ids, p10, arm


class Update(unittest.TestCase):
    def test_zero_outside_the_held_regimes(self):
        regime, _, _, p10, arm = toy()
        u = Q.raw_update(arm, p10, regime)
        self.assertTrue(np.all(u[np.isin(regime, ["nonm", "noac"])] == 0.0))
        self.assertTrue(np.any(u[regime == "nm_bulk"] != 0.0))
        self.assertTrue(np.allclose(u[regime == "nm_late"], (arm - p10)[regime == "nm_late"]))

    def test_the_floor_binds_and_zero_amplitude_serves_nothing(self):
        p = np.array([100.0, 65.0, 61.0])
        u = np.array([-50.0, -10.0, 0.0])
        self.assertTrue(np.array_equal(Q.served_delta(p, u, 0.0), np.zeros(3)))
        d = Q.served_delta(p, u, 1.0)
        self.assertTrue(np.array_equal(p + d, np.array([60.0, 60.0, 61.0])))
        self.assertEqual(int(np.sum(d != 1.0 * u)), 2)

    def test_half_amplitude_is_half_the_update_where_the_floor_is_slack(self):
        p = np.array([1000.0, 900.0])
        u = np.array([-20.0, 40.0])
        self.assertTrue(np.allclose(Q.served_delta(p, u, 0.5), 0.5 * u))


class Cancellation(unittest.TestCase):
    def test_unchanged_rows_cancel_exactly(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0, 300, 5000)
        d = np.where(rng.random(5000) < 0.4, rng.normal(0, 8, 5000), 0.0)
        c = Q.check_cancellation(r, d)
        self.assertTrue(c["cancels"])
        self.assertEqual(c["max_abs_dsse_on_unchanged_rows"], 0.0)
        self.assertEqual(c["rows_changed"] + c["rows_unchanged"], 5000)
        self.assertLess(c["abs_difference"], 1e-12)

    def test_an_all_zero_update_gains_exactly_nothing(self):
        r = np.random.default_rng(2).normal(0, 200, 400)
        c = Q.check_cancellation(r, np.zeros(400))
        self.assertEqual(c["gain_mse_all_rows"], 0.0)
        self.assertEqual(c["rows_changed"], 0)


class Sham(unittest.TestCase):
    def test_the_permutation_never_leaves_its_regime_and_day_cell(self):
        regime, day, ids, _, _ = toy()
        p = Q.sham_permutation(regime, day, ids)
        self.assertTrue(np.array_equal(np.sort(p), np.arange(len(ids))))
        self.assertTrue(np.all(regime[p] == regime))
        self.assertTrue(np.all(day[p] == day))

    def test_it_is_deterministic_and_seed_dependent(self):
        regime, day, ids, _, _ = toy()
        a = Q.sham_permutation(regime, day, ids)
        self.assertTrue(np.array_equal(a, Q.sham_permutation(regime, day, ids)))
        self.assertFalse(np.array_equal(a, Q.sham_permutation(regime, day, ids, seed=1)))

    def test_it_does_not_depend_on_the_file_row_order(self):
        regime, day, ids, _, _ = toy()
        u = np.arange(len(ids), dtype="float64")
        order = np.random.default_rng(9).permutation(len(ids))
        a = u[Q.sham_permutation(regime, day, ids)]
        b = u[order][Q.sham_permutation(regime[order], day[order], ids[order])]
        inv = np.empty(len(order), "int64")
        inv[order] = np.arange(len(order))
        self.assertTrue(np.array_equal(a, b[inv]))


class Scales(unittest.TestCase):
    def _three_folds(self):
        rng = np.random.default_rng(5)
        r, u, apt, day = {}, {}, {}, {}
        for i, f in enumerate(Q.FOLDS):
            n = 3000
            u[f] = np.where(rng.random(n) < 0.7, rng.normal(0, 10, n), 0.0)
            # fold i wants amplitude 0.2*(i+1); the other folds must not learn it
            r[f] = 0.2 * (i + 1) * u[f] + rng.normal(0, 200, n)
            apt[f] = np.where(rng.random(n) < 0.5, "AAAA", "BBBB")
            day[f] = np.repeat(np.arange(n // 100), 100).astype("int64")
        return r, u, apt, day

    def test_the_global_scale_never_sees_its_own_fold(self):
        r, u, apt, _ = self._three_folds()
        s = Q.global_loo_scales(r, u)
        for f in Q.FOLDS:
            r2 = dict(r)
            r2[f] = r[f] * 7.0 + 500.0          # ruin only the scoring fold
            self.assertAlmostEqual(s[f], Q.global_loo_scales(r2, u)[f], places=12)

    def test_the_scale_is_bounded(self):
        r, u, _, _ = self._three_folds()
        big = {f: r[f] * 50.0 for f in r}
        for v in Q.global_loo_scales(big, u).values():
            self.assertGreaterEqual(v, Q.SCALE_BOUNDS[0])
            self.assertLessEqual(v, Q.SCALE_BOUNDS[1])

    def test_a_thin_airport_falls_back_and_a_fat_one_fits_itself(self):
        r, u, apt, _ = self._three_folds()
        for f in Q.FOLDS:                       # make CCCC far too thin to fit
            apt[f] = np.where(np.arange(len(apt[f])) < 5, "CCCC", apt[f])
        fb = Q.global_loo_scales(r, u)
        rows, prov = Q.airport_loo_scales(r, u, apt, fb)
        for f in Q.FOLDS:
            self.assertFalse(prov[f]["CCCC"]["own_fit"])
            self.assertAlmostEqual(prov[f]["CCCC"]["s"], fb[f], places=12)
            self.assertTrue(prov[f]["AAAA"]["own_fit"])
            self.assertTrue(np.all(rows[f][apt[f] == "CCCC"] == fb[f]))

    def test_the_airport_scale_never_sees_its_own_fold(self):
        r, u, apt, _ = self._three_folds()
        fb = Q.global_loo_scales(r, u)
        _, prov = Q.airport_loo_scales(r, u, apt, fb)
        f = Q.FOLDS[0]
        r2 = dict(r)
        r2[f] = r[f] * 7.0 + 500.0
        fb2 = dict(fb)                          # hold the fallback fixed to isolate the fit
        _, prov2 = Q.airport_loo_scales(r2, u, apt, fb2)
        self.assertAlmostEqual(prov[f]["AAAA"]["s"], prov2[f]["AAAA"]["s"], places=12)


class Algebra(unittest.TestCase):
    def test_absorbed_alignment_is_label_free(self):
        rng = np.random.default_rng(17)
        n = 4000
        y = rng.normal(900, 300, n)
        p10 = y + rng.normal(0, 200, n)
        p22 = p10 + rng.normal(0, 30, n)
        u = rng.normal(0, 8, n)
        a10 = float(np.mean(u * (y - p10)))
        a22 = float(np.mean(u * (y - p22)))
        self.assertAlmostEqual(a10 - a22, float(np.mean(u * (p22 - p10))), places=9)

    def test_board_currency_is_the_share_weighted_root_mixture(self):
        folds = {"july_loo": {"mse_v22_s2": 73396.0, "gain_mse_full_s2": 40.0,
                              "gain_mse_fitted_in_sample_s2": 45.0},
                 "january": {"mse_v22_s2": 107426.0, "gain_mse_full_s2": 30.0,
                             "gain_mse_fitted_in_sample_s2": 33.0},
                 "december": {"mse_v22_s2": 33220.0, "gain_mse_full_s2": -5.0,
                              "gain_mse_fitted_in_sample_s2": 1.0}}
        parts = {}
        rng = np.random.default_rng(1)
        for f in ("july_loo", "january"):
            blk = np.repeat(np.arange(30), 100)
            parts[f] = (blk, rng.random(3000) * 1e4, rng.random(3000) * 1e4)
        b = Q.board(folds, parts)
        sh = Q.A.SHARES
        want = (np.sqrt(sum(sh[f] * folds[f]["mse_v22_s2"] for f in sh))
                - np.sqrt(sum(sh[f] * (folds[f]["mse_v22_s2"] - folds[f]["gain_mse_full_s2"])
                              for f in sh)))
        self.assertAlmostEqual(b["board_gain_s"], float(want), places=12)
        self.assertNotIn("december", str(sh))


class Gates(unittest.TestCase):
    def _arm(self, board_s, jul_lb, jan_lb, dec_point):
        return {"board": {"board_gain_s": board_s},
                "folds": {"july_loo": {"ci95_1day": [jul_lb, jul_lb + 1.0]},
                          "january": {"ci95_1day": [jan_lb, jan_lb + 1.0]},
                          "december": {"gain_rmse_s": dec_point}}}

    def test_every_condition_is_required(self):
        good = self._arm(0.12, 0.01, 0.02, 0.01)
        ctrl = self._arm(0.0, -1.0, -1.0, 0.0)
        self.assertTrue(Q.gates(good, ctrl, ctrl)["all_pass"])
        self.assertFalse(Q.gates(self._arm(0.09, 0.01, 0.02, 0.01), ctrl, ctrl)["all_pass"])
        self.assertFalse(Q.gates(self._arm(0.12, -0.01, 0.02, 0.01), ctrl, ctrl)["all_pass"])
        self.assertFalse(Q.gates(self._arm(0.12, 0.01, -0.02, 0.01), ctrl, ctrl)["all_pass"])
        self.assertFalse(Q.gates(self._arm(0.12, 0.01, 0.02, -0.01), ctrl, ctrl)["all_pass"])
        self.assertFalse(Q.gates(good, self._arm(0.13, -1, -1, 0), ctrl)["all_pass"])
        self.assertFalse(Q.gates(good, ctrl, self._arm(0.13, -1, -1, 0))["all_pass"])

    def test_equality_with_the_floor_passes_and_equality_with_a_control_does_not(self):
        ctrl = self._arm(0.0, -1.0, -1.0, 0.0)
        self.assertTrue(Q.gates(self._arm(0.10, 0.01, 0.02, 0.0), ctrl, ctrl)["all_pass"])
        self.assertFalse(Q.gates(self._arm(0.10, 0.01, 0.02, 0.0),
                                 self._arm(0.10, -1, -1, 0), ctrl)["all_pass"])


class Pricing(unittest.TestCase):
    def test_price_fold_counts_the_floor_and_separates_band_and_regime(self):
        regime, day, ids, p10, arm = toy(n=400, seed=8)
        u = Q.raw_update(arm, p10, regime)
        p22 = np.full(len(ids), 70.0)
        y = p22 + np.random.default_rng(4).normal(0, 100, len(ids))
        band = np.arange(len(ids)) % 3 == 0
        apt = np.where(np.arange(len(ids)) % 2 == 0, "EDDM", "LFPG")
        d = Q.served_delta(p22, u, 1.0)
        out = Q.price_fold(y - p22, d, day, regime, band, apt, y,
                           np.ones(len(ids), bool), 1.0 * u)
        self.assertEqual(out["by_regime"]["noac"]["rows_changed"], 0)
        self.assertEqual(out["by_regime"]["nonm"]["rows_changed"], 0)
        self.assertGreater(out["floor_bound_rows"], 0)
        self.assertTrue(out["cancellation"]["cancels"])
        self.assertEqual(out["by_band"]["fixed_table_band"]["rows"]
                         + out["by_band"]["learned_mixture"]["rows"], len(ids))
        self.assertEqual(sum(v["rows"] for v in out["by_airport"].values()), len(ids))


if __name__ == "__main__":
    unittest.main(verbosity=2)
