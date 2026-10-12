from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROGRAM = Path(__file__).with_name("score.py")
spec = importlib.util.spec_from_file_location("score094", PROGRAM)
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
    pred, paired, changed = score.smooth(base, [(100, 900)], data.MVT_ID_mvt)
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
    pred, paired, changed = score.smooth(base, [(10, 20)], ids)
    assert np.array_equal(pred, np.array([2.0, 2.0, 7.0]))
    assert paired.tolist() == [True, True, False] and changed.tolist() == [True, True, False]
    assert pred[2:].view("uint64")[0] == base[2:].view("uint64")[0]
    try:
        score.validate_ids(pd.Series([1.0, 2.0]), "float")
        assert False
    except ValueError as error:
        assert "exact integers" in str(error)


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


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(len(tests), "tests passed")
