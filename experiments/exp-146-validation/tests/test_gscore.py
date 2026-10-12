"""Tests of prc.gscore, the G statistic and resampling scorer (exp-146).

    $PRC_SOURCE_ROOT/.venv/bin/python -m unittest discover -s experiments/exp-146-validation/tests -v

A synthetic frozen directory carries the real schemas; labels are passed in (`y=`), so no
repository label is read. The last class checks the real frozen tables and skips when they
or the raw data are absent.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from prc import data as D  # noqa: E402
from prc import gscore as GS  # noqa: E402

REAL = ROOT / "experiments/exp-146-validation/frozen"
ROLLOVER_ID = 900                # a synthetic row in the role of an unreached-rollover January row
#: the clean-score exclusion of the synthetic row; the real list is private and absent from the public release
SYNTHETIC_ROLLOVER = {"2025-01": [{"id": ROLLOVER_ID}]}
N = {(1, "EDDF", "clock"): 1000, (1, "EDDF", "ordinary"): 500, (1, "EDDF", "noac"): 50,
     (1, "LIRF", "clock"): 300, (1, "LIRF", "ordinary"): 700,
     (7, "EDDF", "clock"): 800, (7, "EDDF", "ordinary"): 600,
     (7, "LIRF", "clock"): 10, (7, "LIRF", "ordinary"): 900, (7, "LIRF", "noac"): 20}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_fixture(d: Path) -> pd.DataFrame:
    rows, i = [], 1000
    for m in (1, 7):
        for a in ("EDDF", "LIRF"):
            for day in range(1, 32):
                for c, k in (("clock", 3), ("ordinary", 2)):
                    for _ in range(k):
                        rows.append((i, a, m, f"2025-{m:02d}-{day:02d}", c)); i += 1
    rows.append((i, "EDDF", 1, "2025-01-01", "noac")); i += 1           # a singleton cell
    r = pd.DataFrame(rows, columns=["MVT_ID_mvt", "airport", "month", "date", "cls"])
    r.loc[(r.airport == "EDDF") & (r.month == 1) & (r.cls == "ordinary"), "MVT_ID_mvt"] = (
        [ROLLOVER_ID] + list(range(i, i + 61)))
    r["month"] = r.month.astype("int8")
    r["pair"] = np.int8(1)
    r["aobt"] = True
    r["clean"] = r.MVT_ID_mvt != ROLLOVER_ID
    r.to_parquet(d / "rows.parquet", index=False)
    pd.DataFrame([(m, a, c, n) for (m, a, c), n in N.items()],
                 columns=["month", "airport", "cls", "N"]).to_csv(d / "cells.csv", index=False)
    pd.DataFrame({"scheme": "lirf_july", "airport": "LIRF",
                  "date": [f"2025-07-{x:02d}" for x in range(1, 32)]}).to_csv(
        d / "clock_loss_days.csv", index=False)
    man = {"files": {f: sha(d / f) for f in ("rows.parquet", "cells.csv", "clock_loss_days.csv")},
           "draws": {k: hashlib.sha256(GS.draw_uniforms(k).tobytes()).hexdigest()
                     for k in ("airport_day", "common_day", "week_block")}}
    (d / "MANIFEST.json").write_text(json.dumps(man))
    return r


class Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.rows = build_fixture(cls.dir)
        cls.saved = GS.FROZEN
        GS.FROZEN = cls.dir

    @classmethod
    def tearDownClass(cls):
        GS.FROZEN = cls.saved
        cls.tmp.cleanup()

    def arrays(self):
        r = self.rows
        return r.MVT_ID_mvt.to_numpy(), np.full(len(r), 1000.0), np.full(len(r), 900.0)

    def W(self, m, a, c):
        return N[(m, a, c)] / GS.N_RANKING

    def test_identity_is_zero(self):
        ids, y, base = self.arrays()
        out = GS.score(ids, base, base.copy(), y=y)
        self.assertEqual(out["G"], 0.0)
        self.assertEqual(out["interval"]["se"], 0.0)
        self.assertEqual(out["n_changed"], 0)

    def test_hand_computed_G_pools_unchanged_rows_and_zeroes_unsupported(self):
        ids, y, base = self.arrays()
        r = self.rows
        cand = base.copy()
        sel = ((r.airport == "EDDF") & (r.month == 1) & (r.cls == "clock")).to_numpy()
        cand[sel] = 1000.0                               # d = 10,000 on every row of the cell
        ordi = np.flatnonzero(((r.airport == "EDDF") & (r.month == 1) & (r.cls == "ordinary")).to_numpy())
        cand[ordi[::2]] = 1000.0                         # half the cell changed
        cand[(r.cls == "noac").to_numpy()] = 1000.0      # singleton: no claimed gain
        out = GS.score(ids, base, cand, y=y)
        mean_ord = 10000.0 * len(ordi[::2]) / len(ordi)
        want = self.W(1, "EDDF", "clock") * 10000.0 + self.W(1, "EDDF", "ordinary") * mean_ord
        self.assertAlmostEqual(out["G"], want, places=9)
        uns = {(u["month"], u["airport"], u["cls"]): u for u in out["unsupported_cells"]}
        self.assertEqual(uns[(1, "EDDF", "noac")]["changed"], 1)
        self.assertIn((7, "LIRF", "noac"), uns)          # ranking cell with no scored row
        self.assertEqual(out["unsupported_ranking_rows"], 70)
        for view in out["views"].values():               # every partition adds up to G
            self.assertAlmostEqual(sum(view.values()), out["G"], places=9)
        self.assertTrue(out["select_G_positive"])
        # the unit reference is local: the comparator's weighted RMSE (100 s here), no board number
        self.assertTrue(out["R_is_local"])
        self.assertAlmostEqual(out["R_reference"], 100.0, places=9)
        self.assertAlmostEqual(out["gain_s"], 100.0 - np.sqrt(100.0 ** 2 - want))
        self.assertFalse(hasattr(GS, "R_REFERENCE"))

    def test_constant_gain_has_no_resampling_spread(self):
        ids, y, base = self.arrays()
        out = GS.score(ids, base, np.full(len(ids), 1000.0), y=y)
        for kind, res in out["resampling"].items():
            self.assertLess(res["se"], 1e-9, kind)
            self.assertAlmostEqual(res["lo95"], out["G"], places=6)
        self.assertLess(out["se_analytic"], 1e-9)

    def test_analytic_and_bootstrap_se_agree(self):
        ids, y, base = self.arrays()
        rng = np.random.default_rng(3)
        day = pd.to_datetime(self.rows.date).dt.day.to_numpy()
        eff = rng.normal(0, 40, 32)[day] + rng.normal(0, 60, len(ids))
        out = GS.score(ids, base, np.maximum(60.0, 900.0 + eff), y=y)
        ratio = out["resampling"]["airport_day"]["se"] / out["se_analytic"]
        self.assertTrue(0.8 < ratio < 1.25, ratio)

    def test_top_days_and_without(self):
        ids, y, base = self.arrays()
        r = self.rows
        cand = base.copy()
        hit = ((r.airport == "LIRF") & (r.month == 7) & (r.date == "2025-07-09")).to_numpy()
        cand[hit] = 1000.0
        out = GS.score(ids, base, cand, y=y)
        t = out["top_airport_days"][0]
        self.assertEqual((t["airport"], t["month"], t["day"]), ("LIRF", 7, 9))
        self.assertAlmostEqual(out["G_without_top_days"], 0.0, places=9)

    def test_clean_view_drops_the_rollover_row(self):
        ids, y, base = self.arrays()
        cand = base.copy()
        cand[ids == ROLLOVER_ID] = 1000.0
        self.assertGreater(GS.score(ids, base, cand, y=y)["G"], 0.0)
        with mock.patch.object(D, "UNREACHED_ROLLOVER", SYNTHETIC_ROLLOVER):
            out = GS.score(ids, base, cand, y=y, view="clean")
        self.assertEqual(out["G"], 0.0)
        self.assertEqual(out["n_rows"], len(ids) - 1)

    def test_clock_mask_moves_made_dark_rows_to_ordinary(self):
        ids, y, base = self.arrays()
        r = self.rows
        cand = base.copy()
        dark = ((r.airport == "LIRF") & (r.month == 7) & (r.cls == "clock")).to_numpy()
        cand[dark] = 1000.0
        out = GS.score(ids, base, cand, y=y, clock_mask="lirf_july")
        n_ord = int(((r.airport == "LIRF") & (r.month == 7)).sum())
        want = self.W(7, "LIRF", "ordinary") * 10000.0 * dark.sum() / n_ord
        self.assertAlmostEqual(out["G"], want, places=9)
        self.assertAlmostEqual(out["views"]["darkness"]["made_dark"], want, places=9)
        self.assertEqual(out["views"]["darkness"]["natural_dark"], 0.0)
        uns = {(u["month"], u["airport"], u["cls"]) for u in out["unsupported_cells"]}
        self.assertIn((7, "LIRF", "clock"), uns)

    def test_guards(self):
        ids, y, base = self.arrays()
        with self.assertRaises(ValueError):                 # below the served floor
            GS.score(ids, base, np.full(len(ids), 59.0), y=y)
        with self.assertRaises(ValueError):
            GS.score(np.r_[ids[:-1], ids[0]], base, base, y=y)
        with self.assertRaises(ValueError):
            GS.score(ids, base, np.r_[base[:-1], np.nan], y=y)
        with self.assertRaises(ValueError):                 # incomplete population
            GS.score(ids[:-5], base[:-5], base[:-5], y=y[:-5])
        out = GS.score(ids[:-5], base[:-5], base[:-5], y=y[:-5], require_complete=False)
        self.assertEqual(out["n_rows"], len(ids) - 5)
        with self.assertRaises(ValueError):
            GS.score(np.r_[ids[:-1], 5], base, base, y=y)

    def test_tampered_frozen_file_is_refused(self):
        p = self.dir / "cells.csv"
        keep = p.read_bytes()
        try:
            p.write_bytes(keep.replace(b"1000", b"1001"))
            ids, y, base = self.arrays()
            with self.assertRaises(RuntimeError):
                GS.score(ids, base, base, y=y)
        finally:
            p.write_bytes(keep)


class Draws(unittest.TestCase):
    def test_draws_are_deterministic_and_shaped(self):
        a, b = GS.draw_uniforms("airport_day"), GS.draw_uniforms("airport_day")
        self.assertTrue((a == b).all())
        self.assertEqual(a.shape, (2000, 30, 31))
        self.assertFalse(np.allclose(a[:, 0, :], GS.draw_uniforms("common_day")[:, 0, :]))

    def test_counts_resample_present_days_only(self):
        present = np.zeros(31, bool)
        present[[2, 5, 9]] = True
        for kind in ("airport_day", "week_block"):
            u = GS.draw_uniforms(kind if kind != "week_block" else "week_block")[:, 0, :]
            C = GS._counts(u, present, kind)
            self.assertTrue((C.sum(1) == 3).all())
            self.assertTrue((C[:, ~present] == 0).all())


@unittest.skipUnless((REAL / "MANIFEST.json").exists(), "frozen tables absent")
class RealFrozen(unittest.TestCase):
    def test_counts(self):
        rows = GS.load_rows()
        jj = rows[rows.month != 12]
        self.assertEqual(len(jj), 344_419)
        self.assertEqual(len(rows[rows.month == 12]), 165_677)
        self.assertEqual(jj.pair.value_counts().sort_index().index.tolist(), [1, 2, 3, 4, 5])
        cells = GS.load_cells()
        self.assertEqual(int(cells.N.sum()), GS.N_RANKING)
        self.assertEqual(cells.groupby("cls").N.sum().to_dict(),
                         {"clock": 196_256, "noac": 388, "ordinary": 148_197})
        self.assertEqual(int(cells[(cells.n < GS.MIN_SUPPORT) & (cells.N > 0)].N.sum()), 22_721)
        self.assertEqual(int((~rows.clean).sum()), 2)

    def test_real_draw_hashes(self):
        for k in ("airport_day", "common_day", "week_block"):
            GS._uniforms_checked(k)


if __name__ == "__main__":
    unittest.main()
