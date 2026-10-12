import importlib.util
import json
import tempfile
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


def test_equality_missing_arrival_blank_stand_and_non_utc_conversion():
    dep = pd.DataFrame({"MVT_ID_mvt": [1, 2, 3], "ADEP_mvt": ["A"] * 3,
        "STAND_mvt": ["S", "S", "   "],
        "MVT_TIME_UTC_mvt": [pd.Timestamp("2025-01-01 11:00", tz="Europe/Paris")] * 3,
        "AOBT_3_flt": [ts("2025-01-01 10:10"), ts("2025-01-01 10:11"), ts("2025-01-01 10:30")]})
    arr = pd.DataFrame({"ADES_mvt": ["A", "A"], "STAND_mvt": ["S", "S"],
        "MVT_TIME_UTC_mvt": [ts("2025-01-01 10:05"), pd.NaT],
        "BLOCK_TIME_UTC_mvt": [ts("2025-01-01 10:10"), pd.NaT]})
    out = m.witness_membership(dep, arr)
    assert out[m.NAMES[0]].tolist() == [False, True, False]


def test_every_margin_edge_at_nanosecond_precision():
    base = ts("2025-01-01 10:10")
    dep = pd.DataFrame({"MVT_ID_mvt": range(6), "ADEP_mvt": ["A"] * 6, "STAND_mvt": ["S"] * 6,
        "MVT_TIME_UTC_mvt": [ts("2025-01-01 10:00")] * 6,
        "AOBT_3_flt": [base, base + pd.Timedelta(1, "ns"), base + pd.Timedelta(60, "s"),
            base + pd.Timedelta(60, "s") + pd.Timedelta(1, "ns"), base + pd.Timedelta(300, "s"),
            base + pd.Timedelta(300, "s") + pd.Timedelta(1, "ns")]})
    arr = pd.DataFrame({"ADES_mvt": ["A"], "STAND_mvt": ["S"],
        "MVT_TIME_UTC_mvt": [ts("2025-01-01 10:05")], "BLOCK_TIME_UTC_mvt": [base]})
    out = m.witness_membership(dep, arr)
    assert out[m.NAMES[0]].tolist() == [False, True, True, True, True, True]
    assert out[m.NAMES[1]].tolist() == [False, False, False, True, True, True]
    assert out[m.NAMES[2]].tolist() == [False, False, False, False, False, True]


def test_strict_membership_refuses_schema_duplicate_missing_extra_and_reorder():
    good = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], **{n: [False, True] for n in m.NAMES}})
    assert list(m.strict_membership(pd.Series([1.0, 2.0]).to_numpy(), good).index) == [1.0, 2.0]
    bad = [good.iloc[::-1].reset_index(drop=True), pd.concat([good, good.iloc[[0]]], ignore_index=True),
           good.iloc[[0]].copy(), good.assign(extra=False), good.assign(**{m.NAMES[0]: [0, 1]})]
    for frame in bad:
        try:
            m.strict_membership(pd.Series([1.0, 2.0]).to_numpy(), frame)
        except ValueError:
            pass
        else:
            raise AssertionError("strict membership accepted a mismatch")


def test_source_gate_refuses_changed_missing_and_duplicate_sources():
    old_root, old_manifest = m.ROOT, m.MANIFEST
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "source.py"
        src.write_text("one\n")
        entry = {"path": "source.py", "sha256": m.sha_file(src)}
        manifest = root / "manifest.json"
        base = {"local_sources": [entry], "raw_months": [], "frozen_inputs": []}
        manifest.write_text(json.dumps(base))
        m.ROOT, m.MANIFEST = root, manifest
        assert m.verify_sources() == base
        for changed in ("two\n", None):
            if changed is None:
                src.unlink()
            else:
                src.write_text(changed)
            try:
                m.verify_sources()
            except SystemExit:
                pass
            else:
                raise AssertionError("source gate accepted a changed or missing source")
            src.write_text("one\n")
        base["raw_months"] = [entry]
        manifest.write_text(json.dumps(base))
        try:
            m.verify_sources()
        except SystemExit:
            pass
        else:
            raise AssertionError("source gate accepted a duplicate source")
    m.ROOT, m.MANIFEST = old_root, old_manifest
