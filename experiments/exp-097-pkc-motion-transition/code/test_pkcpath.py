"""exp-097 registered-path integration tests.

The label-bearing phase gets ONE read.  These tests drive the exact registered code path
-- `pkccontrol.phase2` and `pkccontrol.phase3`, unaltered -- on SYNTHETIC frames, so a
defect in the fit, the out-of-fold discipline, the controls, the currency, the gates or
the candidate write is found before that read, not after it.

No real panel, block, label, prediction, ranking row or summary is opened: every loader
is replaced for the duration of a test and restored afterwards.

    OMP_NUM_THREADS=2 .venv/bin/python \
        experiments/exp-097-pkc-motion-transition/code/test_pkcpath.py
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pkcmotion as M                                                  # noqa: E402
import pkccontrol as C                                                 # noqa: E402

FAILED: list[str] = []
SMALL_BOOT = 120


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def synth_fold(fold: str, n: int, n_sup: int, beta_motion: float, seed: int,
               noise: float = 60.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One synthetic fold: `n` rows, `n_sup` of them supported, a planted relation."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1, dtype=np.int64) + 1_000_000 * (seed + 1)
    p = rng.normal(900.0, 150.0, n)
    motion = np.full(n, np.nan)
    sup = np.zeros(n, bool)
    sup[:n_sup] = True
    motion[sup] = rng.uniform(100.0, 2000.0, n_sup)
    y = p + rng.normal(0.0, 120.0, n)
    # on supported rows the truth moves with the motion clock.  `beta_motion == 0` is
    # the NULL: the supported rows are then exactly like every other row, so there is
    # nothing for any arm to learn -- not even a recalibration of `p_v22`.
    if beta_motion:
        y[sup] = (0.6 * p[sup] + beta_motion * motion[sup] + 300.0
                  + rng.normal(0.0, noise, n_sup))
    else:
        y[sup] = p[sup] + rng.normal(0.0, noise, n_sup)
    days = np.array([f"2025-07-{1 + (i % 28):02d}" for i in range(n)])
    d = pd.DataFrame({
        "id64": ids, "MVT_ID_mvt": ids.astype("float64"),
        "airport": np.where(np.arange(n) % 2 == 0, "EHAM", "EDDF"),
        "day": days, "fold": fold, "p_v22": p,
        "obs_has_clock": ~sup, "noac": 0, "nm_missing": 0,
        "own_has_clock": ~sup, "noac_bool": np.zeros(n, bool),
        "obs_has_clock_block": ~sup, "obs_matched": np.ones(n, bool),
        "obs_taxi": np.where(sup, np.nan, 800.0),
        "obs_taxi_motion": motion,
        "stand_stop_s": rng.uniform(0.0, 900.0, n),
        "turnaround_anchored": (rng.random(n) < 0.2).astype("float64"),
        "seg_span_s": rng.uniform(100.0, 3000.0, n),
        "n_ground_points": rng.uniform(5.0, 200.0, n),
        "dt_takeoff": rng.normal(0.0, 100.0, n),
        "stand_dist_km": rng.uniform(0.1, 4.0, n),
        "reach_class": np.where(sup, "matched_no_clock", "strict"),
    })
    return M.add_signal(d), pd.DataFrame({"id64": ids, "y_true": y})


@contextlib.contextmanager
def patched(folds: dict, receipt: dict, boot: int = SMALL_BOOT):
    old = (C.fold_frame, C.panel_labels, C.read_receipt, M.BOOT)
    C.fold_frame = lambda f: folds[f][0].copy()
    C.panel_labels = lambda f: folds[f][1].copy()
    C.read_receipt = lambda n: receipt
    M.BOOT = boot
    try:
        yield
    finally:
        C.fold_frame, C.panel_labels, C.read_receipt, M.BOOT = old


PASS_RECEIPT = {"support_condition": {"verdict": "PASS"}}

#: `phase2` and `phase3` now take the boundary record `preflight` builds: the phase-1 r2
#: chain with its digests, the repository state, the launcher handoff and the desk
#: acceptance.  The boundary itself is tested in `test_pkcboundary.py`; here it is a
#: fixture, so these tests keep measuring the science and not the door.
BOUNDARY = {
    "phase1_chain": {"revision": "r2", "support_verdict": "PASS", "total_support": 3864,
                     "files": {k: {"path": v, "sha256": "0" * 64, "bytes": 1}
                               for k, v in {"result": "analysis/screen-r2.json"}.items()},
                     "rejected": {}},
    "repo": {"branch": "exp097-pkc-motion-transition", "head": "0" * 40, "clean": True},
    "start": {"pid": 0, "nice": 19, "threads": {"OMP_NUM_THREADS": "8"}},
    "desk_acceptance": {"phase": 2, "run_id": "test", "issuer": "desk"},
}


def test_phase2_finds_a_planted_signal() -> None:
    print("phase 2: a planted motion relation is found, and the controls are not")
    folds = {f: synth_fold(f, 6000, 900, 0.7, i) for i, f in enumerate(M.FOLDS)}
    with patched(folds, PASS_RECEIPT):
        rec = C.phase2(BOUNDARY)
    m = rec["arms"]["motion"]
    e = rec["arms"]["extent"]
    p = rec["arms"]["placebo"]
    check("motion gains on every fold",
          all(m["by_fold"][f]["rmse_gain"] > 0 for f in M.FOLDS),
          str({f: round(m["by_fold"][f]["rmse_gain"], 4) for f in M.FOLDS}))
    check("motion board gain is large", m["board_share_gain_s"] > 1.0,
          f"{m['board_share_gain_s']:.4f}")
    check("motion beats extent", m["board_share_gain_s"] > e["board_share_gain_s"],
          f"{m['board_share_gain_s']:.4f} vs {e['board_share_gain_s']:.4f}")
    check("motion beats placebo", m["board_share_gain_s"] > p["board_share_gain_s"],
          f"{m['board_share_gain_s']:.4f} vs {p['board_share_gain_s']:.4f}")
    check("every gate passes", rec["all_gates_pass"], str(rec["gates"]))
    check("the gate map holds exactly the five registered gates",
          tuple(rec["gates"]) == M.GATES, str(list(rec["gates"])))
    check("decision is ADVANCE", rec["decision"] == "ADVANCE")
    check("board after subtracts the gain",
          abs(rec["board_after_if_transferred"]
              - (M.V22_BOARD - m["board_share_gain_s"])) < 1e-9)


def test_phase2_out_of_fold_discipline() -> None:
    print("phase 2: each fold's coefficients are fitted on the OTHER folds only")
    folds = {f: synth_fold(f, 3000, 500, 0.7, 10 + i) for i, f in enumerate(M.FOLDS)}
    with patched(folds, PASS_RECEIPT):
        rec = C.phase2(BOUNDARY)
    co = rec["arms"]["motion"]["coefficients"]
    sup = {f: int(folds[f][0]["obs_taxi_motion"].notna().sum()) for f in M.FOLDS}
    for f in M.FOLDS:
        others = sum(sup[g] for g in M.FOLDS if g != f)
        check(f"{f} trains on exactly the other folds' supported rows",
              co[f]["train_rows"] == others, f"{co[f]['train_rows']} vs {others}")
    check("the three coefficient vectors differ",
          len({tuple(np.round(co[f]["beta"], 9)) for f in M.FOLDS}) == 3)
    # move ONLY the target fold's labels: its own coefficients must not move
    before = [tuple(np.round(co[f]["beta"], 9)) for f in M.FOLDS]
    poisoned = dict(folds)
    lab = folds["january"][1].copy()
    lab["y_true"] = lab["y_true"] + 5000.0
    poisoned["january"] = (folds["january"][0], lab)
    with patched(poisoned, PASS_RECEIPT):
        rec2 = C.phase2(BOUNDARY)
    co2 = rec2["arms"]["motion"]["coefficients"]
    check("january's own coefficients are unchanged by january's labels",
          tuple(np.round(co2["january"]["beta"], 9)) == before[M.FOLDS.index("january")])
    check("the other folds' coefficients DO move (january is in their training set)",
          tuple(np.round(co2["july_loo"]["beta"], 9)) != before[0])


def test_phase2_null_signal_fails() -> None:
    print("phase 2: with no relation the gates FAIL and no arm is promoted")
    folds = {f: synth_fold(f, 4000, 700, 0.0, 20 + i, noise=400.0) for i, f in enumerate(M.FOLDS)}
    with patched(folds, PASS_RECEIPT):
        rec = C.phase2(BOUNDARY)
    check("board gain is under the +0.10 s gate",
          rec["arms"]["motion"]["board_share_gain_s"] < M.UPLOAD_GATE_S,
          f"{rec['arms']['motion']['board_share_gain_s']:.4f}")
    check("G1 fails", not rec["gates"]["G1_board_gain_ge_0.10"])
    check("decision is FAIL", rec["decision"] == "FAIL")


def test_phase2_touches_only_supported_rows() -> None:
    print("phase 2: an unsupported row keeps p_v22 bit for bit")
    folds = {f: synth_fold(f, 2000, 300, 0.7, 30 + i) for i, f in enumerate(M.FOLDS)}
    with patched(folds, PASS_RECEIPT):
        rec = C.phase2(BOUNDARY)
    for f in M.FOLDS:
        s = rec["arms"]["motion"]["by_fold"][f]
        check(f"{f}: changed rows never exceed supported rows",
              s["changed_rows"] <= s["support_rows"],
              f"{s['changed_rows']} > {s['support_rows']}")
        check(f"{f}: support rows are the planted ones", s["support_rows"] == 300,
              str(s["support_rows"]))


def test_phase2_does_not_consult_a_loose_receipt() -> None:
    """Phase 2's predecessor is the phase-1 **r2** chain, checked by `preflight` before the
    claim, not whatever `receipt-phase1*.json` the run's own revision happens to name.  The
    old code read `read_receipt(1)`, which at the default revision is the INVALID unsuffixed
    receipt.  The chain itself is tested in `test_pkcboundary.py`."""
    print("phase 2: the predecessor is the r2 chain, not a loose phase-1 receipt")
    folds = {f: synth_fold(f, 500, 100, 0.7, 40 + i) for i, f in enumerate(M.FOLDS)}
    names = C.phase2.__code__.co_names
    check("phase 2 never calls read_receipt", "read_receipt" not in names,
          f"phase2 still references it: {[n for n in names if 'receipt' in n]}")
    for receipt, why in [(None, "absent"),
                         ({"support_condition": {"verdict": "FAIL"}}, "FAIL")]:
        with patched(folds, receipt):
            rec = C.phase2(BOUNDARY)
        check(f"a {why} loose receipt changes nothing", rec["decision"] in ("ADVANCE", "FAIL"))
        check(f"the {why} case still records the r2 chain",
              rec["phase1_chain"]["revision"] == "r2")


def test_phase3_candidate() -> None:
    print("phase 3: the candidate replaces exactly the supported rows")
    rng = np.random.default_rng(7)
    n = 5000
    ids = np.arange(1, n + 1, dtype=np.int64)
    pred = rng.normal(900.0, 150.0, n)
    sup = np.zeros(n, bool)
    sup[:400] = True
    d = pd.DataFrame({
        "id64": ids, "own_has_clock": ~sup, "noac_bool": np.zeros(n, bool),
        "obs_has_clock_block": ~sup, "obs_matched": np.ones(n, bool),
        "obs_taxi_motion": np.where(sup, rng.uniform(100.0, 2000.0, n), np.nan),
        "stand_stop_s": rng.uniform(0.0, 900.0, n),
        "turnaround_anchored": np.zeros(n),
        "seg_span_s": np.full(n, 900.0), "n_ground_points": np.full(n, 40.0),
        "dt_takeoff": np.zeros(n), "stand_dist_km": np.full(n, 1.0),
        "reach_class": np.where(sup, "matched_no_clock", "strict"),
        "airport": "EHAM", "day": "2026-01-01",
    })
    d = M.add_signal(d)
    src = pd.DataFrame({"MVT_ID_mvt": ids.astype("float64"),
                        "TAXITIME_SEC_mvt": pred})
    receipt = {"decision": "ADVANCE",
               "deployment_fit": {"beta": [300.0, 0.6, 0.2, 0.0, 0.0],
                                  "columns": ["intercept", "p_v22", *M.MOTION_COLS],
                                  "medians": {c: 0.0 for c in M.MOTION_COLS}}}
    old = (C.ranking_frame, C.uploaded_v22, C.uploaded_v22_frame, C.read_receipt,
           C.CANDIDATE_PATH, M.RANK_MONTHS)
    tmp = tempfile.mkdtemp()
    try:
        M.RANK_MONTHS = ("2026-01",)
        C.ranking_frame = lambda m: d.copy()
        C.uploaded_v22 = lambda: pd.DataFrame({"id64": ids, "prediction": pred})
        C.uploaded_v22_frame = lambda: src.copy()
        C.read_receipt = lambda k: receipt
        C.CANDIDATE_PATH = str(Path(tmp) / "cand.parquet")
        rec = C.phase3(dict(BOUNDARY, desk_acceptance={'phase': 3}))
    finally:
        (C.ranking_frame, C.uploaded_v22, C.uploaded_v22_frame, C.read_receipt,
         C.CANDIDATE_PATH, M.RANK_MONTHS) = old
    check("changed rows equal the supported rows", rec["changed_rows"] == 400,
          str(rec["changed_rows"]))
    check("the changed-row gate passes", rec["gate_pass"])
    back = pd.read_parquet(rec["candidate"]["path"])
    check("row count preserved", len(back) == n)
    check("id order preserved",
          np.array_equal(back["MVT_ID_mvt"].to_numpy(), src["MVT_ID_mvt"].to_numpy()))
    check("unsupported rows are bit-identical",
          np.array_equal(back["TAXITIME_SEC_mvt"].to_numpy()[~sup], pred[~sup]))
    check("every value is inside the registered taxi window",
          bool((back["TAXITIME_SEC_mvt"].to_numpy()[sup] >= M.TAXI_LO).all()
               and (back["TAXITIME_SEC_mvt"].to_numpy()[sup] <= M.TAXI_HI).all()))
    check("the candidate record carries a sha256", len(rec["candidate"]["sha256"]) == 64)


def test_phase3_row_gate() -> None:
    print("phase 3: under 100 changed rows writes no candidate")
    n = 500
    ids = np.arange(1, n + 1, dtype=np.int64)
    pred = np.full(n, 900.0)
    sup = np.zeros(n, bool)
    sup[:20] = True
    d = M.add_signal(pd.DataFrame({
        "id64": ids, "own_has_clock": ~sup, "noac_bool": np.zeros(n, bool),
        "obs_has_clock_block": ~sup, "obs_matched": np.ones(n, bool),
        "obs_taxi_motion": np.where(sup, 500.0, np.nan),
        "stand_stop_s": np.zeros(n), "turnaround_anchored": np.zeros(n),
        "seg_span_s": np.full(n, 900.0), "n_ground_points": np.full(n, 40.0),
        "dt_takeoff": np.zeros(n), "stand_dist_km": np.full(n, 1.0),
        "reach_class": "matched_no_clock", "airport": "EHAM", "day": "2026-01-01"}))
    receipt = {"decision": "ADVANCE",
               "deployment_fit": {"beta": [300.0, 0.6, 0.2, 0.0, 0.0],
                                  "columns": ["intercept", "p_v22", *M.MOTION_COLS],
                                  "medians": {c: 0.0 for c in M.MOTION_COLS}}}
    old = (C.ranking_frame, C.uploaded_v22, C.read_receipt, M.RANK_MONTHS)
    try:
        M.RANK_MONTHS = ("2026-01",)
        C.ranking_frame = lambda m: d.copy()
        C.uploaded_v22 = lambda: pd.DataFrame({"id64": ids, "prediction": pred})
        C.read_receipt = lambda k: receipt
        rec = C.phase3(dict(BOUNDARY, desk_acceptance={'phase': 3}))
    finally:
        (C.ranking_frame, C.uploaded_v22, C.read_receipt, M.RANK_MONTHS) = old
    check("gate fails at 20 rows", not rec["gate_pass"] and rec["candidate"] is None)


def test_placebo_seed_is_registered_and_shared() -> None:
    """Sol round 1, blocking 3: the placebo's application used an unregistered `+7`
    offset, so it permuted the rows differently from the fit it was applied with."""
    print("placebo: one registered permutation per arm per fold, used by fit and apply")
    folds = {f: synth_fold(f, 900, 220, 0.0, i) for i, f in enumerate(M.FOLDS)}
    seen = {}
    real_permute = M.permute_within
    calls = []

    def spy(d, cols, seed):
        calls.append((tuple(d["id64"].tolist()), tuple(cols), int(seed)))
        return real_permute(d, cols, seed)

    M.permute_within = spy
    try:
        with patched(folds, PASS_RECEIPT):
            C.phase2(BOUNDARY)
    finally:
        M.permute_within = real_permute
    seeds = sorted({c[2] for c in calls})
    want = sorted({M.SEED + M.ARM_OFFSET["placebo"] + M.FOLD_OFFSET[f]
                   for f in M.FOLDS})
    check("only the placebo arm permutes, once per fold", len(calls) == len(M.FOLDS),
          f"{len(calls)} calls")
    check("every seed is the registered one", seeds == want, f"{seeds} != {want}")
    check("no unregistered offset appears",
          all(c[2] - M.SEED - M.ARM_OFFSET["placebo"] in M.FOLD_OFFSET.values()
              for c in calls))
    check("the permuted columns are the motion columns",
          all(c[1] == tuple(M.MOTION_COLS) for c in calls))
    # the same fold is permuted ONCE, so the training rows and the application rows of a
    # placebo fold are the same permutation, not two draws of it
    for f in M.FOLDS:
        seed = M.SEED + M.ARM_OFFSET["placebo"] + M.FOLD_OFFSET[f]
        sup = folds[f][0][M.support_mask(folds[f][0])].reset_index(drop=True)
        a = real_permute(sup, M.MOTION_COLS, seed)
        b = real_permute(sup, M.MOTION_COLS, seed)
        check(f"{f}: the registered seed is reproducible",
              np.array_equal(a["obs_taxi_motion"].to_numpy(),
                             b["obs_taxi_motion"].to_numpy()))
        check(f"{f}: the permutation really moves the clock",
              not np.array_equal(a["obs_taxi_motion"].to_numpy(),
                                 sup["obs_taxi_motion"].to_numpy()))
        check(f"{f}: the row identity is not permuted with it",
              np.array_equal(a["id64"].to_numpy(), sup["id64"].to_numpy()))


def test_phase3_read_back_is_enforced() -> None:
    """Sol round 1, blocking 5: phase 3 recorded the read-back facts but did not act on
    them, so a corrupted write would have been reported as a clean candidate."""
    print("phase 3: a corrupted candidate read-back RAISES, it is not merely recorded")
    n = 900
    ids = np.arange(1, n + 1, dtype=np.int64)
    pred = np.full(n, 900.0)
    sup = np.zeros(n, bool)
    sup[:300] = True
    d = M.add_signal(pd.DataFrame({
        "id64": ids, "own_has_clock": ~sup, "noac_bool": np.zeros(n, bool),
        "obs_has_clock_block": ~sup, "obs_matched": np.ones(n, bool),
        "obs_taxi_motion": np.where(sup, 500.0, np.nan),
        "stand_stop_s": np.zeros(n), "turnaround_anchored": np.zeros(n),
        "seg_span_s": np.full(n, 900.0), "n_ground_points": np.full(n, 40.0),
        "dt_takeoff": np.zeros(n), "stand_dist_km": np.full(n, 1.0),
        "reach_class": "matched_no_clock", "airport": "EHAM", "day": "2026-01-01"}))
    src = pd.DataFrame({"MVT_ID_mvt": ids.astype("float64"),
                        "TAXITIME_SEC_mvt": pred})
    receipt = {"decision": "ADVANCE",
               "deployment_fit": {"beta": [300.0, 0.6, 0.2, 0.0, 0.0],
                                  "columns": ["intercept", "p_v22", *M.MOTION_COLS],
                                  "medians": {c: 0.0 for c in M.MOTION_COLS}}}
    # Each corruption is applied by a writer that stands in for `to_parquet`.  `WRITE` is
    # the real method, captured here so a writer does not call its own replacement.
    WRITE = pd.DataFrame.to_parquet

    def bend(f, where, value):
        col = f["TAXITIME_SEC_mvt"].to_numpy("float64").copy()
        col[where] = value
        return f.assign(TAXITIME_SEC_mvt=col)

    corruptions = {
        "a dropped row": lambda f, path: WRITE(f.iloc[:-1], path, index=False),
        "a reordered id": lambda f, path: WRITE(f.iloc[::-1], path, index=False),
        "a non-finite value": lambda f, path: WRITE(bend(f, 0, np.nan), path,
                                                    index=False),
        "an altered untouched row": lambda f, path: WRITE(bend(f, len(f) - 1, 1234.0),
                                                          path, index=False),
        "a value outside the taxi window": lambda f, path: WRITE(bend(f, 0, 99999.0),
                                                                 path, index=False),
    }
    tmp = tempfile.mkdtemp()
    for why, writer in corruptions.items():
        old = (C.ranking_frame, C.uploaded_v22, C.uploaded_v22_frame, C.read_receipt,
               C.CANDIDATE_PATH, M.RANK_MONTHS, pd.DataFrame.to_parquet)
        try:
            M.RANK_MONTHS = ("2026-01",)
            C.ranking_frame = lambda m: d.copy()
            C.uploaded_v22 = lambda: pd.DataFrame({"id64": ids, "prediction": pred})
            C.uploaded_v22_frame = lambda: src.copy()
            C.read_receipt = lambda k: receipt
            C.CANDIDATE_PATH = str(Path(tmp) / "cand.parquet")
            pd.DataFrame.to_parquet = lambda self, path, **kw: writer(self, path)
            C.phase3(dict(BOUNDARY, desk_acceptance={'phase': 3}))
            check(f"refuses {why}", False, "phase 3 returned a record")
        except SystemExit as e:
            check(f"refuses {why}", True, str(e))
        finally:
            (C.ranking_frame, C.uploaded_v22, C.uploaded_v22_frame, C.read_receipt,
             C.CANDIDATE_PATH, M.RANK_MONTHS, pd.DataFrame.to_parquet) = old


def main() -> int:
    for t in (test_phase2_finds_a_planted_signal, test_phase2_out_of_fold_discipline,
              test_phase2_null_signal_fails, test_phase2_touches_only_supported_rows,
              test_phase2_does_not_consult_a_loose_receipt,
              test_placebo_seed_is_registered_and_shared, test_phase3_candidate,
              test_phase3_row_gate, test_phase3_read_back_is_enforced):
        t()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("all registered-path integration tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
