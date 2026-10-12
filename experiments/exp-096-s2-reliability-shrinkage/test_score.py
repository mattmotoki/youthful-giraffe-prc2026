from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROGRAM = Path(__file__).with_name("score.py")
spec = importlib.util.spec_from_file_location("score096", PROGRAM)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def frame(times, service, radio, origin=None, ids=None, index=None):
    count = len(times)
    raw = pd.DataFrame(
        {
            "MVT_ID_mvt": ids or list(range(1, count + 1)),
            "PHASE_mvt": ["DEP"] * count,
            "MVT_TIME_UTC_mvt": pd.to_datetime(times, utc=True),
            "FLIGHT_mvt": service,
            "CALLSIGN_flt": radio,
            "ADEP_mvt": origin or ["AAA"] * count,
        },
        index=index,
    )
    return score.prepared(raw)[0]


def test_full_precision_gap_edges():
    cases = [
        (60, []),
        (60.5, [(7, 9)]),
        (86399, [(7, 9)]),
        (86399.5, []),
        (86400, []),
    ]
    for seconds, expected in cases:
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        end = start + pd.Timedelta(seconds=seconds)
        data = frame([start, end], ["A", "A"], ["R", "R"], ids=[7, 9])
        assert score.primary_pairs(data) == expected
    start = pd.Timestamp("2025-01-01T12:00:00Z")
    data = frame([start, start + pd.Timedelta(microseconds=60_000_001)], ["A", "A"], ["R", "R"])
    assert score.primary_pairs(data) == [(1, 2)]


def test_midnight_partition_and_previous_day():
    data = frame(["2025-01-01T23:58:00Z", "2025-01-02T00:02:00Z"], ["A", "A"], ["R", "R"])
    assert score.primary_pairs(data) == []
    assert score.previous_day_pairs(data) == [(1, 2)]


def test_lanes_are_independent_with_missing_other_identity():
    data = frame(
        ["2025-01-01T00:00:00Z", "2025-01-01T00:02:00Z", "2025-01-01T00:04:00Z"],
        ["A", "A", "A"],
        ["R", "R", None],
    )
    assert score.lane_pairs(data, data.service, data.service_ok) == set()
    assert score.lane_pairs(data, data.radio, data.radio_ok) == {(1, 2)}
    assert score.primary_pairs(data) == []
    data = frame(
        ["2025-01-01T00:00:00Z", "2025-01-01T00:02:00Z", "2025-01-01T00:04:00Z"],
        ["A", "A", None],
        ["R", "R", "R"],
    )
    assert score.lane_pairs(data, data.service, data.service_ok) == {(1, 2)}
    assert score.lane_pairs(data, data.radio, data.radio_ok) == set()
    assert score.primary_pairs(data) == []


def test_same_direction_intersection_and_overlap_rejection():
    data = frame(
        ["2025-01-01T00:00:00Z", "2025-01-01T00:02:00Z", "2025-01-01T00:04:00Z", "2025-01-01T00:06:00Z"],
        ["A", "A", "B", "B"],
        ["R", "S", "R", "S"],
    )
    assert score.primary_pairs(data) == []
    assert score.disjoint({(1, 2), (2, 3)}) == []


def test_missing_unicode_collisions_and_origins():
    data = frame(["2025-01-01T00:00Z", "2025-01-01T00:02Z"], ["Ａ", None], ["R", "R"])
    assert score.primary_pairs(data) == []
    data = frame(["2025-01-01T00:00Z", "2025-01-01T00:02Z"], ["Ａ", "A"], ["R", "R"])
    assert score.primary_pairs(data) == []
    data = frame(["2025-01-01T00:00Z", "2025-01-01T00:02Z"], ["A", "A"], ["R", "R"], origin=["AAA", "BBB"])
    assert score.primary_pairs(data) == []


def test_pairs_use_integer_ids_under_index_and_order_shuffle():
    data = frame(
        ["2025-01-01T00:02Z", "2025-01-01T00:00Z"],
        ["A", "A"],
        ["R", "R"],
        ids=[900, 100],
        index=[91, 17],
    )
    assert score.primary_pairs(data) == [(100, 900)]
    base = np.array([30.0, 10.0])
    pred, paired, changed = score.shrink(base, [(100, 900)], data.MVT_ID_mvt, 1.0)
    assert np.array_equal(pred, np.array([20.0, 20.0]))
    assert paired.tolist() == [True, True] and changed.tolist() == [True, True]
    shuffled = data.iloc[::-1].reset_index(drop=True)
    assert score.primary_pairs(shuffled) == [(100, 900)]


def test_rotation_tuple_and_tie_are_deterministic():
    data = frame(
        ["2025-01-01T00:00Z", "2025-01-01T00:02Z", "2025-01-01T00:04Z", "2025-01-01T00:06Z"],
        ["A", "B", "A", "C"],
        ["R", "S", "R", "T"],
        ids=[4, 3, 2, 1],
    )
    assert score.rotation_pairs(data) == [(3, 1)]
    assert score.tie_digest(1) == score.tie_digest(1) != score.tie_digest(2)


def test_unchanged_bits_averaging_and_id_refusals():
    base = np.array([1.0, 3.0, 7.0])
    ids = pd.Series([10, 20, 30], dtype="int64")
    pred, paired, changed = score.shrink(base, [(10, 20)], ids, 1.0)
    assert np.array_equal(pred, np.array([2.0, 2.0, 7.0]))
    assert paired.tolist() == [True, True, False] and changed.tolist() == [True, True, False]
    assert pred[2:].view("uint64")[0] == base[2:].view("uint64")[0]
    largest = 2**53 - 1
    assert np.array_equal(
        score.validate_ids(pd.Series([-largest, largest], dtype="int64"), "integer"),
        np.array([-largest, largest], dtype="int64"),
    )
    assert np.array_equal(
        score.validate_ids(pd.Series([-float(largest), float(largest)]), "floating"),
        np.array([-largest, largest], dtype="int64"),
    )
    assert np.array_equal(score.validate_ids(pd.Series([1.0, 2.0]), "float"), np.array([1, 2]))
    bad_series = [
        pd.Series([1.5]),
        pd.Series([float("nan")]),
        pd.Series([float("inf")]),
        pd.Series([-(2**53)], dtype="int64"),
        pd.Series([2**53], dtype="int64"),
        pd.Series([2**53 + 1], dtype="int64"),
        pd.Series([-float(2**53)]),
        pd.Series([float(2**53)]),
        pd.Series([1, 1], dtype="int64"),
        pd.Series([0.0, -0.0]),
        pd.Series(["1"], dtype="object"),
    ]
    for bad in bad_series:
        try:
            score.validate_ids(bad, "bad")
            assert False
        except ValueError as error:
            assert "IDs" in str(error)


def test_coefficient_fit_bounds_and_shrinkage():
    y = np.array([2.0, 2.0])
    base = np.array([0.0, 4.0])
    direction = np.array([2.0, -2.0])
    mask = np.array([True, True])
    coefficient = score.fit_coefficient([(y, base, direction, mask)])
    assert 0.0 < coefficient < 1.0
    pred, paired, changed = score.shrink(base, [(1, 2)], pd.Series([1.0, 2.0]), coefficient)
    assert paired.all() and changed.all() and np.all(pred > np.array([0.0, 2.0]))
    assert score.fit_coefficient([(np.array([-2.0, 6.0]), base, direction, mask)]) == 0.0
    opposite, _, _ = score.sign_control(base, [(1, 2)], pd.Series([1.0, 2.0]), coefficient)
    assert opposite[0] < base[0] and opposite[1] > base[1]


def synthetic_historical_trace():
    saved = {
        name: getattr(score, name)
        for name in [
            "verify_phase_sources", "raw_path", "prepared", "join_panel", "primary_pairs",
            "rotation_pairs", "previous_day_pairs", "adjustment_direction", "fit_coefficient",
            "shrink", "sign_control", "metrics", "board_gain", "evaluate_gates",
        ]
    }
    saved_read_parquet = score.pd.read_parquet
    fit_calls = []
    seed_calls = []
    fold_number = {fold: offset + 1 for offset, fold in enumerate(score.FOLDS)}

    def read_parquet(path, columns=None):
        path = str(path)
        if "panel-v22-" in path:
            fold = path.split("panel-v22-", 1)[1].removesuffix(".parquet")
            number = fold_number[fold]
            return pd.DataFrame({
                "MVT_ID_mvt": pd.Series([number], dtype="int64"),
                "fold": [fold], "day": ["2025-01-01"], "regime": ["nm"],
                "y_true": [float(number)], "p_v22": [0.0],
            })
        month = next(month for month in score.FOLD_MONTH.values() if month in path)
        fold = next(fold for fold, value in score.FOLD_MONTH.items() if value == month)
        return pd.DataFrame({"synthetic_fold": [fold]})

    def prepared(raw):
        fold = raw.synthetic_fold.iloc[0]
        number = fold_number[fold]
        return pd.DataFrame({
            "MVT_ID_mvt": pd.Series([number], dtype="int64"),
            "date": ["2025-01-01"], "origin": ["AAA"],
        }), {"origin": 0, "service": 0, "radio": 0}

    def fit_coefficient(records):
        fit_calls.append(tuple(int(record[0][0]) for record in records))
        return 0.0

    def metrics(y, base, pred, pair_mask, changed, day, seed, no_nm, airport, pairs):
        seed_calls.append((int(y[0]), seed))
        return {"rmse_gain": 0.0, "interval95": [0.0, 0.0]}

    try:
        score.verify_phase_sources = lambda manifest, phase: None
        score.raw_path = lambda month: Path("training-" + month + ".parquet")
        score.pd.read_parquet = read_parquet
        score.prepared = prepared
        score.join_panel = lambda frame, panel, fold: None
        score.primary_pairs = lambda frame: []
        score.rotation_pairs = lambda frame: []
        score.previous_day_pairs = lambda frame: []
        score.adjustment_direction = lambda base, pairs, ids: (
            np.zeros(len(base)), np.zeros(len(base), dtype=bool)
        )
        score.fit_coefficient = fit_coefficient
        score.shrink = lambda base, pairs, ids, coefficient: (
            base.copy(), np.zeros(len(base), dtype=bool), np.zeros(len(base), dtype=bool)
        )
        score.sign_control = score.shrink
        score.metrics = metrics
        score.board_gain = lambda mses, arm: 0.0
        score.evaluate_gates = lambda output, gains: {}
        score.historical_phase({"projections": {"raw": [], "panel": []}})
        return fit_calls, seed_calls
    finally:
        score.pd.read_parquet = saved_read_parquet
        for name, value in saved.items():
            setattr(score, name, value)


def test_registered_scoring_path_excludes_each_target_fold():
    fit_calls, _ = synthetic_historical_trace()
    assert fit_calls[:6] == [(2, 3), (2, 3), (1, 3), (1, 3), (1, 2), (1, 2)]


def test_registered_scoring_path_deployment_uses_every_fold_once():
    fit_calls, _ = synthetic_historical_trace()
    assert fit_calls[6:] == [(1, 2, 3), (1, 2, 3)]


def test_registered_scoring_path_uses_fold_and_arm_seed_offsets():
    _, seed_calls = synthetic_historical_trace()
    assert seed_calls == [
        (1, score.SEED), (1, score.SEED + 100), (1, score.SEED + 200),
        (2, score.SEED + 1), (2, score.SEED + 101), (2, score.SEED + 201),
        (3, score.SEED + 2), (3, score.SEED + 102), (3, score.SEED + 202),
    ]


def test_join_panel_order_and_duplicate_refusals():
    data = frame(["2025-01-01T00:00Z", "2025-01-01T00:02Z"], ["A", "A"], ["R", "R"], ids=[1, 2])
    panel = pd.DataFrame({"MVT_ID_mvt": pd.Series([2, 1], dtype="int64"), "fold": ["x", "x"], "day": ["2025-01-01"] * 2})
    try:
        score.join_panel(data, panel, "x")
        assert False
    except ValueError as error:
        assert "identity/order" in str(error)
    panel.MVT_ID_mvt = pd.Series([1, 1], dtype="int64")
    try:
        score.join_panel(data, panel, "x")
        assert False
    except ValueError as error:
        assert "invalid IDs" in str(error)


def test_bootstrap_and_atomic_outputs():
    old_draws = score.B
    score.B = 20
    try:
        y = np.array([3.0, 4.0])
        base = np.array([1.0, 1.0])
        pred = np.array([2.0, 2.0])
        day = np.array(["d1", "d2"])
        assert score.bootstrap(y, base, pred, day, 9) == score.bootstrap(y, base, pred, day, 9)
    finally:
        score.B = old_draws
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "x.json"
        score.atomic_json(output, {"x": 1.0})
        assert json.loads(output.read_text()) == {"x": 1.0}
        try:
            score.atomic_json(output, {"x": float("nan")})
            assert False
        except ValueError:
            pass
        parquet = root / "x.parquet"
        expected = pd.DataFrame({"a": [1]})
        score.atomic_parquet(parquet, expected)
        assert pd.read_parquet(parquet).equals(expected)


def test_currency_and_all_historical_gates():
    mses = {
        "base": {"july_loo": 100.0, "january": 100.0},
        "primary": {"july_loo": 81.0, "january": 81.0},
        "rotation": {"july_loo": 100.0, "january": 100.0},
    }
    assert abs(score.board_gain(mses, "primary") - 1.0) < 1e-12
    output = {
        "board_share": {"primary": 0.1, "rotation": 0.0},
        "folds": {fold: {"arms": {"primary": {"interval95": [0.01, 1.0]}}} for fold in score.FOLDS},
    }
    gains = {"primary": {fold: 0.0 for fold in score.FOLDS}, "rotation": {fold: -1.0 for fold in score.FOLDS}}
    assert all(score.evaluate_gates(output, gains).values())
    output["folds"]["january"]["arms"]["primary"]["interval95"][0] = 0.0
    assert not score.evaluate_gates(output, gains)["clear_july_january"]


def test_historical_failure_refuses_candidate_source_access():
    old_historical, old_candidate, old_phase, old_result = score.historical_phase, score.candidate_phase, score.PHASE, score.RESULT
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        marker = root / "candidate-access"
        score.PHASE, score.RESULT = root / "phase.json", root / "result.json"
        score.historical_phase = lambda manifest: ({"historical_gates": {"gate": False}}, {}, {})
        score.candidate_phase = lambda manifest, output: marker.write_text("opened")
        try:
            score.main()
            assert not marker.exists()
            assert json.loads(score.PHASE.read_text())["phase"] == "historical_fail"
            assert json.loads(score.RESULT.read_text())["candidate_phase_opened"] is False
        finally:
            score.historical_phase, score.candidate_phase = old_historical, old_candidate
            score.PHASE, score.RESULT = old_phase, old_result


def test_registered_real_source_schemas_without_rows():
    manifest = json.loads(score.MANIFEST.read_text())
    for relative, source in manifest["sources"].items():
        path = score.ROOT / relative
        parquet = score.pq.ParquetFile(path)
        assert parquet.metadata.num_rows == source[2]
        assert score.schema_fp(path) == manifest["schema_fingerprints"][source[3]]
        projection = manifest["projections"]["raw" if source[3] in {"training", "ranking"} else source[3]]
        assert set(projection) <= set(parquet.schema_arrow.names)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(len(tests), "tests passed")
