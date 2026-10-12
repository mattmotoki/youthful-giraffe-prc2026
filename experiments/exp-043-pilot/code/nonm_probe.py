"""exp-043-pilot: on the no-NM rows, separate MISSING RECEIPT from MISSING IDENTITY.

A callsign census cannot tell the two apart: a no-NM row whose reconstructed
callsign is wrong looks exactly like a no-NM row the network never received.  This
asks the question without the callsign, and it is the THIRD version of the test.

**Version 1** asked only whether some segment the callsign matcher had not used sat
within +/- 15 min of the row's `MVT_TIME_UTC_mvt`, and answered yes for 69 of 72.
Round 2 showed that is not a receipt estimate: 54 of those 69 had between 2 and 16
candidates, the assignment was not one-to-one, and a placebo that shifts the target
time by half an hour still found a candidate for 55 of 72.

**Version 2** narrowed the key to 120 s plus aircraft-type agreement and solved a
maximum one-to-one assignment.  Round 3 showed three faults in it, all real:

*   the key was calibrated on the same rows that then reported the check;
*   the check counted rows the matcher assigned to ANY segment, not rows it
    assigned to the RIGHT one -- an independent audit found 55 of 500 assigned to
    a different aircraft, so 486/500 was not an accuracy;
*   two placebo offsets cannot show a zero false-match rate; other offsets gave
    up to 3.

**This version.**  The key is fixed on the development days alone (07-07/08: the
99th percentile of `|obs_wheels_off - MVT_TIME_UTC_mvt|` is 115.1 s, rounded up to
the next 10 s, and the trace type equals `AIRCRAFT_TYPE_mvt` on 96.9 %).  Its
accuracy is then measured on HELD-OUT callsign-matched rows, in draws the size of
the real problem (72 rows against the free segment pool plus their own segments),
and scored on whether the matcher returned each row's own segment.  The placebo is
swept over ten offsets, and the reported number is an ESTIMATE under this key, with
no bound claimed.

  .venv/bin/python experiments/exp-043-pilot/code/nonm_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "code"))
import census as C          # noqa: E402
import match as M           # noqa: E402
import segments as S        # noqa: E402

DAYS = [f"2025-07-{d:02d}" for d in range(7, 14)]
DEV_DAYS = ("2025-07-07", "2025-07-08")
LOOSE = M.WINDOW     # s, version 1's window, kept for the comparison
DRAWS = 200          # calibration draws, each the size of the real problem
SHIFTS_MIN = (-120, -90, -60, -45, -30, 30, 45, 60, 90, 120)


def _key_from_development(u: pd.DataFrame, day: pd.Series) -> dict:
    """The key's two constants, read off the development days and nothing else."""
    dev = u[day[u.index].isin(DEV_DAYS)]
    p99 = float(np.percentile(np.abs(dev.obs_wheels_off - dev.mvt_s), 99))
    agree = float((dev.type.astype(str) == dev.AIRCRAFT_TYPE_mvt.astype(str)).mean())
    return {"window_s": float(np.ceil(p99 / 10.0) * 10.0),
            "aircraft_type_must_agree": True,
            "development_days": list(DEV_DAYS),
            "development_rows": int(len(dev)),
            "development_p99_abs_wheels_off_minus_mvt_s": round(p99, 1),
            "development_type_agreement": round(agree, 4),
            "rule": ("window = the development days' 99th percentile rounded up to the "
                     "next 10 s; type agreement required because it holds on 96.9 % of "
                     "them.  No held-out row took part in either choice."),
            "assignment": "maximum one-to-one (scipy maximum_bipartite_matching)"}


def assign(rows: pd.DataFrame, seg: pd.DataFrame, tol: float,
           shift: float = 0.0, need_type: bool = True) -> tuple[dict, np.ndarray]:
    """Maximum one-to-one assignment of departures to segments.

    Returns the counts and the assignment itself, so a caller that knows the
    answer can ask whether the matcher returned the RIGHT segment, not just one.
    """
    wo = seg.obs_wheels_off.to_numpy("float64")
    st = seg.type.astype(str).to_numpy()
    ri, ci = [], []
    for i, (_, r) in enumerate(rows.iterrows()):
        ok = np.abs(wo - (r.mvt_s + shift)) <= tol
        if need_type:
            ok &= st == str(r.AIRCRAFT_TYPE_mvt)
        for j in np.flatnonzero(ok):
            ri.append(i); ci.append(j)
    if not ri:
        return ({"rows": int(len(rows)), "with_any_candidate": 0, "one_to_one": 0},
                np.full(len(rows), -1))
    n_cand = pd.Series(ri).value_counts()
    g = csr_matrix((np.ones(len(ri), bool), (ri, ci)), shape=(len(rows), len(seg)))
    m = maximum_bipartite_matching(g, perm_type="column")
    return ({"rows": int(len(rows)),
             "with_any_candidate": int(len(n_cand)),
             "one_to_one": int((m >= 0).sum()),
             "median_candidates_when_any": float(n_cand.median()),
             "max_candidates": int(n_cand.max())}, m)


def calibrate(known: pd.DataFrame, free: pd.DataFrame, seg: pd.DataFrame,
              tol: float, n_rows: int, draws: int = DRAWS,
              add_truth: bool = True) -> dict:
    """How often does the key return the RIGHT segment, on rows where it is known?

    Each draw is the real problem's shape: `n_rows` held-out callsign-matched
    departures against the free pool plus their own segments.  A draw is scored
    per row: assigned to its own segment, assigned to another aircraft's, or not
    assigned at all.
    """
    rng = np.random.default_rng(0)
    idx = np.arange(len(known))
    tot = {"assigned": 0, "correct": 0, "wrong": 0, "unassigned": 0, "rows": 0}
    per_draw = []
    for _ in range(draws):
        take = known.iloc[rng.choice(idx, size=min(n_rows, len(idx)), replace=False)]
        truth = S.seg_key(take).to_numpy()
        pool = (pd.concat([free, seg[seg.key.isin(set(truth))]]).reset_index(drop=True)
                if add_truth else free)
        _, m = assign(take, pool, tol)
        got = np.where(m >= 0, pool.key.to_numpy()[np.clip(m, 0, None)], "")
        ok = (m >= 0) & (got == truth)
        bad = (m >= 0) & (got != truth)
        tot["rows"] += len(take); tot["assigned"] += int((m >= 0).sum())
        tot["correct"] += int(ok.sum()); tot["wrong"] += int(bad.sum())
        tot["unassigned"] += int((m < 0).sum())
        per_draw.append(float(ok.sum()) / max(int((m >= 0).sum()), 1))
    tot["draws"] = draws
    tot["pool_segments"] = int(len(free) + (n_rows if add_truth else 0))
    tot["rows_per_draw"] = int(min(n_rows, len(idx)))
    tot["precision_assigned_to_the_right_segment"] = round(tot["correct"] / max(tot["assigned"], 1), 4)
    tot["recall_of_all_rows"] = round(tot["correct"] / max(tot["rows"], 1), 4)
    tot["precision_draw_range"] = [round(float(np.min(per_draw)), 4),
                                   round(float(np.max(per_draw)), 4)]
    return tot


def main() -> None:
    pts, _ = C.load_slices(DAYS)
    seg = S.segments(pts)
    seg["key"] = S.seg_key(seg)
    m = pd.read_parquet(EXP / "work/matched-week.parquet")
    day = m.day.dt.strftime("%Y-%m-%d")
    used = m[m.obs_wheels_off.notna()]
    claimed = set(S.seg_key(used))
    free = seg[~seg.key.isin(claimed)].reset_index(drop=True)

    no = m[~m.has_nm]
    miss = no[no.obs_wheels_off.isna()]
    key = _key_from_development(used[used.has_nm], day)
    tol = key["window_s"]

    tight, _ = assign(miss, free, tol)
    known = used[used.has_nm & ~day[used.index].isin(DEV_DAYS)]
    cal = calibrate(known, free, seg, tol, len(miss))
    # Round 3 audited the key at 500 rows against the whole CLAIMED pool -- version
    # 2's own setup -- and found 55 rows pushed onto another aircraft's segment.
    # That is real and it is reproduced here.  The two arms below separate the two
    # things that differ from the pilot's actual question: how many rows compete
    # (500 against 72) and how dense the pool is (every claimed segment against the
    # free ones).  The pilot's question has 72 rows and the free pool, so the arm
    # above is the fair calibration and these two say why the audit read lower.
    cal500 = calibrate(known, free, seg, tol, 500)
    claimed_pool = seg[seg.key.isin(claimed)].reset_index(drop=True)
    cal_r3 = calibrate(known, claimed_pool, seg, tol, 500, draws=20, add_truth=False)

    placebo = {}
    for s in SHIFTS_MIN:
        c, _ = assign(miss, free, tol, shift=float(s) * 60.0)
        placebo[f"{s:+d}_min"] = c["one_to_one"]
    pv = list(placebo.values())

    p = cal["precision_assigned_to_the_right_segment"]
    ident = tight["one_to_one"]
    est = ident * p
    res = {
        "days": DAYS,
        "key": key,
        "held_out_calibration": cal,
        "held_out_calibration_at_500_rows": cal500,
        "held_out_calibration_round3_setup": cal_r3,
        "no_nm_departures": int(len(no)),
        "matched_by_callsign": int(no.obs_wheels_off.notna().sum()),
        "unmatched": int(len(miss)),
        "tight_key": tight,
        "placebo_one_to_one_by_shift": placebo,
        "placebo_summary": {"min": int(min(pv)), "max": int(max(pv)),
                            "mean": round(float(np.mean(pv)), 2), "offsets": len(pv)},
        "first_version_loose_window_no_type": assign(miss, free, LOOSE, need_type=False)[0],
        "identified": ident,
        "expected_correct_at_held_out_precision": round(est, 1),
        "receipt_on_no_nm_rows_estimate": round(
            (int(no.obs_wheels_off.notna().sum()) + est) / max(len(no), 1), 4),
        "receipt_on_no_nm_rows_if_every_identification_were_right": round(
            (int(no.obs_wheels_off.notna().sum()) + ident) / max(len(no), 1), 4),
    }
    res["reading"] = (
        f"The key identifies {ident} of the {len(miss)} unmatched no-NM rows, against "
        f"{res['placebo_summary']['mean']} for the same key at ten time-shifted "
        f"placebos (max {res['placebo_summary']['max']}), so most identifications are "
        "not chance.  On held-out rows where the answer is known, the same key returns "
        f"the RIGHT segment for {p:.1%} of the rows it assigns, so about "
        f"{est:.0f} of the {ident} are expected to be correct and the no-NM receipt "
        f"estimate is {res['receipt_on_no_nm_rows_estimate']:.1%}.  This is an ESTIMATE "
        "UNDER THIS KEY, not a bound in either direction: the key can miss a received "
        "aircraft whose type string or clock differs, and it can name the wrong one. "
        "It supports 'the no-NM tail is mostly an identity problem'; it does not "
        "support 'receipt is not a problem there'.")
    (EXP / "analysis/nonm_probe.json").write_text(json.dumps(res, indent=2, sort_keys=True,
                                                             default=float))
    print(json.dumps(res, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
