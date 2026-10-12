import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
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


def test_exp052_target_contract_is_static_and_exact():
    source = P.read_text()
    assert ".y_true" not in source
    assert 'residual = df.y.to_numpy("float64") - df.p_v22.to_numpy("float64")' in source
    assert 'yy = df.y.to_numpy("float64")' in source


def test_complete_synthetic_score_path_uses_exact_exp052_frame():
    """Execute the real scorer's complete three-fold path without a real row read."""
    class Patch:
        def __init__(self):
            self.old = []
        def setattr(self, obj, name, value):
            self.old.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        def undo(self):
            for obj, name, value in reversed(self.old):
                setattr(obj, name, value)
    monkeypatch = Patch()
    exp052 = m.ROOT / "experiments/exp-052-nmv22/code"
    sys.path.insert(0, str(exp052))
    import rx_common as C
    import rx_iter2 as RI
    import rx_price as RP

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        exp = root / "experiments/exp-081-nmwitness"
        (exp / "analysis").mkdir(parents=True)
        (exp / "work").mkdir()
        e52 = root / "experiments/exp-052-nmv22/analysis"
        e52.mkdir(parents=True)
        raw_dir = root / "data/raw"
        raw_dir.mkdir(parents=True)
        folds = m.FOLDS
        frames = {}
        selected = {}
        for fi, fold in enumerate(folds):
            n = 25
            ids = np.arange(fi * 1000 + 1, fi * 1000 + n + 1, dtype=float)
            block = np.tile(np.arange(5), 5)
            p22 = np.full(n, 100.0)
            # Exact rx_price.load / rx_iter2.with_values contract.
            df = pd.DataFrame({
                "MVT_ID_mvt": ids, "airport": np.repeat(["A", "B", "C", "D", "E"], 5),
                "day": pd.date_range("2025-01-01", periods=n, freq="D"),
                "regime": ["nm_bulk"] * n, "guard_held": [False] * n,
                "clean": [True] * n, "y": p22 + 2.0, "p_v22": p22,
                "r": np.full(n, 2.0), "pop": [True] * n, "block": block,
                "cell": np.tile(np.arange(5) % 4, 5),
            })
            outer = {"aware": np.zeros(n), "blind": np.ones(n), "sham": np.zeros(n)}
            inner = {k: {b: v.copy() for b in range(5)} for k, v in outer.items()}
            frames[fold] = {"df": df, "O": outer, "I": inner}
            rows = pd.DataFrame({"MVT_ID_mvt": ids, "u_router": np.zeros(n),
                                 "u_global": np.full(n, -0.1), "u_q_aware": np.zeros(n)})
            rp = root / f"rows-{fold}.parquet"
            rows.to_parquet(rp, index=False)
            selected[fold] = {"rows_file": str(rp.relative_to(root)), "rows_sha256": m.sha_file(rp)}
            mem = pd.DataFrame({"MVT_ID_mvt": ids, **{name: np.ones(n, dtype=bool) for name in m.NAMES}})
            mp = exp / f"work/membership-{fold}.parquet"
            mem.to_parquet(mp, index=False)
            raw = pd.DataFrame({"MVT_ID_mvt": ids, "PHASE_mvt": ["DEP"] * n,
                                "AOBT_3_flt": [pd.NaT] * n})
            rawp = raw_dir / f"{fold}.parquet"
            raw.to_parquet(rawp, index=False)
        iter2 = {"selection": {f: {"k_by_block": [1] * 5} for f in folds},
                 "selected": {"folds": selected}}
        (e52 / "iter2.json").write_text(json.dumps(iter2))
        (e52 / "trunc.json").write_text(json.dumps({"runs": {"real": {}}}))
        (e52 / "trunc-exactness.json").write_text(json.dumps({"passes": True}))
        memrec = {"schema": "exp081-membership-v1", "receipt": {"exit_status": 0}, "folds": {}}
        for fold in folds:
            mp = exp / f"work/membership-{fold}.parquet"
            memrec["folds"][fold] = {"path": str(mp.relative_to(root)), "sha256": m.sha_file(mp),
                                      "support": {name: 25 for name in m.NAMES}}
        (exp / "analysis/membership.json").write_text(json.dumps(memrec))

        monkeypatch.setattr(m, "ROOT", root)
        monkeypatch.setattr(m, "EXP", exp)
        monkeypatch.setattr(m, "MANIFEST", exp / "source_manifest.json")
        monkeypatch.setattr(m, "RAW", {f: raw_dir / f"{f}.parquet" for f in folds})
        monkeypatch.setattr(m, "verify_sources", lambda: {})
        monkeypatch.setattr(RP, "load", lambda fold, *_: frames[fold])
        monkeypatch.setattr(RI, "load_trunc", lambda *_: pd.DataFrame())
        monkeypatch.setattr(RI, "with_values", lambda frame, *_: frame)
        monkeypatch.setattr(C, "SEEDS_PLACEBO", tuple(range(5300, 5350)))
        monkeypatch.setattr(m.os, "getloadavg", lambda: (0.1, 0.1, 0.1))
        try:
            m.score()
        finally:
            monkeypatch.undo()

        result = json.loads((exp / "analysis/results.json").read_text())
        receipt = json.loads((exp / "analysis/output-receipt.json").read_text())
        assert receipt["all_required_outputs_present"]
        assert (exp / "analysis/tables.md").is_file()
        assert (exp / "RESULTS.md").is_file()
        assert (exp / "climb.log").is_file()
        assert (root / "docs/LEDGER.md").is_file()
        assert set(result["witnesses"]) == set(m.NAMES)
        assert all(len(result["witnesses"][w]["folds"]) == 3 for w in m.NAMES)
        for witness in m.NAMES:
            rec = result["witnesses"][witness]
            assert rec["placebo_partition"]["n"] == 50
            for fold in folds:
                fr = rec["folds"][fold]
                assert set(fr["held_block_gain_s2_per_row"]) == set(map(str, range(5)))
                assert any(v > 0 for v in fr["held_block_gain_s2_per_row"].values())
                assert set(fr["by_block_and_old_cell"]) == set(map(str, range(5)))
                assert set(fr["common_rmse"]) == {"overall", "target_at_most_3600",
                    "target_over_3600", "no_nm_all_flt_null", "clean"}
