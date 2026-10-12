import importlib.util
from pathlib import Path

import pandas as pd

P = Path(__file__).parents[1] / "code/nmwitness.py"
spec = importlib.util.spec_from_file_location("nmw", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def ts(x):
    return pd.Timestamp(x, tz="UTC") if x is not None else pd.NaT


def test_registered_boundaries_and_bad_rows():
    dep = pd.DataFrame({
        "MVT_ID_mvt": range(1, 8), "ADEP_mvt": ["A"] * 7, "STAND_mvt": ["S"] * 6 + [None],
        "MVT_TIME_UTC_mvt": [ts("2025-01-01 10:00:00")] * 5 + [pd.NaT, ts("2025-01-01 10:00")],
        "AOBT_3_flt": [ts("2025-01-01 10:11"), ts("2025-01-01 10:12"),
                         ts("2025-01-01 10:15"), ts("2025-01-01 10:15:00.000000001"),
                         pd.NaT, ts("2025-01-01 10:20"), ts("2025-01-01 10:20")]})
    arr = pd.DataFrame({
        "ADES_mvt": ["A", "A", "A"], "STAND_mvt": ["S"] * 3,
        "MVT_TIME_UTC_mvt": [ts("2025-01-01 10:05"), ts("2025-01-01 10:05"), ts("2025-01-01 10:40")],
        "BLOCK_TIME_UTC_mvt": [ts("2025-01-01 10:10"), ts("2025-01-01 10:10"), ts("2025-01-01 10:30")]})
    out = m.witness_membership(dep, arr)
    assert out[m.NAMES[0]].tolist() == [True, True, True, True, False, False, False]
    assert out[m.NAMES[1]].tolist() == [False, True, True, True, False, False, False]
    assert out[m.NAMES[2]].tolist() == [False, False, False, True, False, False, False]


def test_tie_at_takeoff_is_not_next_and_impossible_arrival_is_ignored():
    dep = pd.DataFrame({"MVT_ID_mvt": [1], "ADEP_mvt": ["A"], "STAND_mvt": ["S"],
                        "MVT_TIME_UTC_mvt": [ts("2025-01-01 10:00")],
                        "AOBT_3_flt": [ts("2025-01-01 10:20")]})
    arr = pd.DataFrame({"ADES_mvt": ["A", "A", "A"], "STAND_mvt": ["S"] * 3,
                        "MVT_TIME_UTC_mvt": [ts("2025-01-01 09:50"), ts("2025-01-01 10:10"), ts("2025-01-01 10:01")],
                        "BLOCK_TIME_UTC_mvt": [ts("2025-01-01 10:00"), ts("2025-01-01 10:05"), ts("2025-01-01 10:10")]})
    out = m.witness_membership(dep, arr)
    assert out[m.NAMES[0]].tolist() == [True]


def test_labels_or_predictions_cannot_change_membership():
    dep = pd.DataFrame({"MVT_ID_mvt": [1], "ADEP_mvt": ["A"], "STAND_mvt": ["S"],
                        "MVT_TIME_UTC_mvt": [ts("2025-01-01 10:00")], "AOBT_3_flt": [ts("2025-01-01 10:20")],
                        "y_true": [1.0], "prediction": [2.0]})
    arr = pd.DataFrame({"ADES_mvt": ["A"], "STAND_mvt": ["S"],
                        "MVT_TIME_UTC_mvt": [ts("2025-01-01 10:05")],
                        "BLOCK_TIME_UTC_mvt": [ts("2025-01-01 10:10")], "y_true": [3.0]})
    a = m.witness_membership(dep, arr)
    dep[["y_true", "prediction"]] = [[999.0, -999.0]]
    arr["y_true"] = -7.0
    assert a.equals(m.witness_membership(dep, arr))

