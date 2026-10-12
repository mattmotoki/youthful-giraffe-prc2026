"""exp-098 registered-path integration tests.

These drive the real `controller.py` end to end over synthetic fixtures that carry the
registered schemas exactly.  They prove the parts a unit test on the arithmetic cannot:
the source gate refuses a changed byte, the label-free block finishes before the first
label is opened, a second invocation refuses, the phase boundary holds a failing run out
of the ranking and uploaded-v22 sources, and a passing run writes a verified candidate
whose `noac` rows are bit-identical.

No repository label, prediction, ranking value or uploaded byte is read here.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
CODE = EXP / "code"
REPO = EXP.parent.parent
FOLDS = ("july_loo", "december", "january")
AIRPORTS = ("EDDM", "EGLL", "LFPG", "LIRF")


def _fold_frame(fold: str, n: int, signal: float, seed: int):
    """Synthetic packet and panel for one fold, in the registered schemas."""
    rng = np.random.default_rng(seed)
    ids = np.arange(n, dtype="float64") + 1_000_000 * (FOLDS.index(fold) + 1)
    regime = np.array(["nm_bulk"] * int(n * 0.8) + ["nm_late"] * int(n * 0.1)
                      + ["nonm"] * int(n * 0.07)
                      + ["noac"] * (n - int(n * 0.8) - int(n * 0.1) - int(n * 0.07)))
    day = (np.arange(n) // max(n // 28, 1)).astype("int64") + 20000
    airport = np.array(AIRPORTS)[np.arange(n) % len(AIRPORTS)]
    p10 = 700.0 + rng.normal(0, 120, n)
    p22 = p10 + rng.normal(0, 40, n)
    u = np.where(np.isin(regime, ["nm_bulk", "nm_late"]) & (rng.random(n) < 0.85),
                 rng.normal(0, 9, n), 0.0)
    y = np.maximum(p22 + signal * u + rng.normal(0, 260, n), 61.0)
    packet = pd.DataFrame({
        "MVT_ID_mvt": ids, "y_true": y, "airport": airport, "regime": regime,
        "band": (np.arange(n) % 9 == 0), "day": day,
        "v10 replay": p10, "none": p10, "none d11-noac": p10 + 0.0,
        "chain": p10 + u, "chain d11-noac": p10 + u,
        "derived": p10 + u, "derived d11-noac": p10 + u})
    panel = pd.DataFrame({
        "MVT_ID_mvt": ids, "airport": airport, "t": day * 86400.0,
        "day": day.astype(str), "regime": regime,
        "nm_missing": np.int8(0), "noac": (regime == "noac").astype("int8"),
        "clean": np.ones(n, bool), "y_true": y, "pbase": p22, "usable": np.ones(n, bool),
        "obs_has_clock": (np.arange(n) % 2 == 0), "reach_class": "reached",
        "d_v22": np.zeros(n), "p_v22": p22, "guard_held": np.zeros(n, bool), "fold": fold})
    return packet, panel


def build_fixture(tmp: Path, signal: float) -> Path:
    """A whole synthetic repository root plus a manifest that binds it."""
    import pyarrow.parquet as pq
    (tmp / "experiments/exp-040-chain/packet").mkdir(parents=True)
    (tmp / "experiments/exp-040-chain/candidates").mkdir(parents=True)
    (tmp / "experiments/exp-050-nmblind/work").mkdir(parents=True)
    (tmp / "experiments/exp-026-rebuild/candidates").mkdir(parents=True)
    (tmp / "submissions/uploaded").mkdir(parents=True)
    for i, f in enumerate(FOLDS):
        pk, pn = _fold_frame(f, 4000, signal, 100 + i)
        pk.to_parquet(tmp / f"experiments/exp-040-chain/packet/q09-{f}.parquet", index=False)
        pn.to_parquet(tmp / f"experiments/exp-050-nmblind/work/panel-v22-{f}.parquet",
                      index=False)
    n = 5000
    rng = np.random.default_rng(77)
    rids = np.arange(n, dtype="float64") + 9_000_000
    rreg = np.array(["nm_bulk"] * int(n * 0.85) + ["nm_late"] * int(n * 0.08)
                    + ["nonm"] * int(n * 0.05)
                    + ["noac"] * (n - int(n * 0.85) - int(n * 0.08) - int(n * 0.05)))
    rapt = np.array(AIRPORTS)[np.arange(n) % len(AIRPORTS)]
    base = 800.0 + rng.normal(0, 150, n)
    pd.DataFrame({"MVT_ID_mvt": rids,
                  "TAXITIME_SEC_mvt": base}).to_parquet(
        tmp / "experiments/exp-040-chain/candidates/head-published.parquet", index=False)
    pd.DataFrame({"MVT_ID_mvt": rids,
                  "TAXITIME_SEC_mvt": base + rng.normal(0, 7, n)}).to_parquet(
        tmp / "experiments/exp-040-chain/candidates/head-chain.parquet", index=False)
    pd.DataFrame({"MVT_ID_mvt": rids, "regime": rreg,
                  "d_takeoff_aobt": rng.normal(0, 60, n), "month": "2026-01",
                  "airport": rapt, "day": "2026-01-01"}).to_parquet(
        tmp / "experiments/exp-026-rebuild/candidates/ranking-gates.parquet", index=False)
    pd.DataFrame({"MVT_ID_mvt": rids,
                  "TAXITIME_SEC_mvt": np.maximum(base, 61.0)}).to_parquet(
        tmp / "submissions/uploaded/youthful-giraffe_v22.parquet", index=False)

    real = json.loads((EXP / "source-manifest.json").read_text())
    import hashlib
    sources = []
    for s in real["sources"]:
        p = tmp / s["path"]
        if not p.exists():                       # align.py and the guard stay real
            p = REPO / s["path"]
        e = {"path": s["path"], "kind": s["kind"], "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        if s["path"].endswith(".parquet"):
            f = pq.ParquetFile(p)
            e["rows"] = f.metadata.num_rows
            e["schema"] = {a: str(b) for a, b in zip(f.schema_arrow.names,
                                                     f.schema_arrow.types)}
        sources.append(e)
    man = dict(real)
    man["sources"] = sources
    exp = tmp / "exp"
    (exp / "analysis").mkdir(parents=True)
    (exp / "source-manifest.json").write_text(json.dumps(man, indent=1))
    for s in real["sources"]:
        if not (tmp / s["path"]).exists():       # the two code sources the manifest binds
            (tmp / s["path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / s["path"], tmp / s["path"])
    return exp


def run_controller(tmp: Path, exp: Path, extra=()) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update({"EXP098_ROOT": str(tmp), "EXP098_EXP": str(exp),
                "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
                "PYTHONDONTWRITEBYTECODE": "1"})
    return subprocess.run([sys.executable, str(CODE / "controller.py"), *extra],
                          capture_output=True, text=True, env=env, timeout=900)


class Registered(unittest.TestCase):
    def test_a_flat_update_fails_and_never_opens_the_ranking_phase(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            exp = build_fixture(tmp, signal=0.0)
            r = run_controller(tmp, exp)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            res = json.loads((exp / "analysis/result.json").read_text())
            rec = json.loads((exp / "analysis/phase-receipt.json").read_text())
            self.assertIn(res["3_gates"]["verdict"], ("FAIL", "MAYBE"))
            self.assertIsNone(res["3_gates"]["decision_arm"])
            self.assertEqual(rec["phase"], "historical_only")
            self.assertFalse(res["4_candidate"]["opened"])
            self.assertFalse((exp / "analysis/candidate.json").exists())
            # the label-free block finished before the first label was opened
            self.assertLess(res["1_label_free"]["label_free_finished_unix"],
                            res["2_priced"]["first_label_read_unix"])
            # both controls ran through every arm
            for d in ("real", "sham_refit", "sham_permutation"):
                self.assertEqual(sorted(res["2_priced"][d]["arms"]), ["A0", "A1", "A2"])
            # unchanged rows cancel on every arm and fold
            for d in ("real", "sham_refit", "sham_permutation"):
                for a in ("A0", "A1", "A2"):
                    for f in FOLDS:
                        c = res["2_priced"][d]["arms"][a]["folds"][f]["cancellation"]
                        self.assertTrue(c["cancels"])
            # the held regimes are the only ones the update touches
            for a in ("A0", "A1", "A2"):
                for f in FOLDS:
                    by = res["2_priced"]["real"]["arms"][a]["folds"][f]["by_regime"]
                    self.assertEqual(by["noac"]["rows_changed"], 0)
                    self.assertEqual(by["nonm"]["rows_changed"], 0)
            self.assertEqual(json.loads((exp / "analysis/exit.json").read_text())["state"],
                             "complete")

    def test_a_strong_update_passes_and_writes_a_verified_candidate(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            exp = build_fixture(tmp, signal=60.0)
            r = run_controller(tmp, exp)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            res = json.loads((exp / "analysis/result.json").read_text())
            self.assertEqual(res["3_gates"]["verdict"], "ADVANCE")
            self.assertIn(res["3_gates"]["decision_arm"], ("A0", "A1", "A2"))
            self.assertEqual(json.loads(
                (exp / "analysis/phase-receipt.json").read_text())["phase"],
                "candidate_authorized")
            cand = json.loads((exp / "analysis/candidate.json").read_text())
            self.assertTrue(cand["opened"])
            self.assertTrue(cand["noac_held_bit_identical"])
            self.assertEqual(cand["by_regime"]["noac"]["rows_changed"], 0)
            self.assertEqual(cand["by_regime"]["nonm"]["rows_changed"], 0)
            self.assertEqual(cand["rows"], 5000)
            self.assertGreaterEqual(cand["min"], 60.0)
            self.assertTrue(Path(cand["path"]).exists())

    def test_a_changed_source_byte_stops_the_run_before_any_value(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            exp = build_fixture(tmp, signal=0.0)
            man = json.loads((exp / "source-manifest.json").read_text())
            for s in man["sources"]:
                if s["kind"] == "v22_equivalent_panel":
                    s["sha256"] = "0" * 64
            (exp / "source-manifest.json").write_text(json.dumps(man, indent=1))
            r = run_controller(tmp, exp)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("source changed since registration", r.stderr)
            self.assertFalse((exp / "analysis/result.json").exists())
            self.assertEqual(json.loads((exp / "analysis/exit.json").read_text())["state"],
                             "INCOMPLETE")

    def test_a_second_invocation_refuses_and_rewrites_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            exp = build_fixture(tmp, signal=0.0)
            self.assertEqual(run_controller(tmp, exp).returncode, 0)
            before = (exp / "analysis/result.json").read_bytes()
            again = run_controller(tmp, exp)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("REFUSING TO OVERWRITE", again.stderr)
            self.assertEqual((exp / "analysis/result.json").read_bytes(), before)

    def test_the_single_use_claim_refuses_even_with_no_artifact_to_guard(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            exp = build_fixture(tmp, signal=0.0)
            (exp / "analysis/launch.claim").write_text("an earlier launch\n")
            r = run_controller(tmp, exp)
            self.assertEqual(r.returncode, 3)
            self.assertIn("already been claimed", r.stderr)
            self.assertFalse((exp / "analysis/started.json").exists())
            self.assertFalse((exp / "analysis/result.json").exists())

    def test_the_write_once_guard_refuses_an_existing_result(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            exp = build_fixture(tmp, signal=0.0)
            (exp / "analysis/result.json").write_text("{}")
            r = run_controller(tmp, exp)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("REFUSING TO OVERWRITE", r.stderr)

    def test_each_fold_scale_is_fitted_only_on_the_other_two_folds(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            exp = build_fixture(tmp, signal=0.4)
            self.assertEqual(run_controller(tmp, exp).returncode, 0)
            res = json.loads((exp / "analysis/result.json").read_text())
            prov = res["2_priced"]["real"]["scales"]["A2_provenance"]
            for f in FOLDS:                       # every airport names its fitting rows
                for a, v in prov[f].items():
                    self.assertIn("changed_rows_in_fitting_folds", v)
                    self.assertGreaterEqual(v["changed_rows_in_fitting_folds"], 0)
            self.assertEqual(set(res["2_priced"]["real"]["scales"]["A0"].values()), {1.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
