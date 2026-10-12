"""The repair the desk ordered after the time-placement pass failed its placebo.

Desk, 2026-09-09:

    the inbound callsign already identifies the AIRCRAFT's arrival as a ranking ARRIVAL
    row; use the turnaround chain -- the ranking departure from the SAME STAND by the same
    hashed operator and AIRCRAFT_TYPE whose off-block follows that arrival's in-block,
    uniqueness inside that set, the +/-60 s take-off agreement as a check, not the key.

**Why this can work where time alone cannot.**  RESULTS 21 refuted placement by time: at
EDDF and LFPG a take-off every two to three minutes makes a +/-60 s window a coin, and the
one-day placebo placed MORE than the real pass.  The gate tested departure density and
never tested that this segment belonged to that departure.  The chain here is a different
kind of key: `STAND_mvt` is on 100 % of both phases with 1,697 distinct values, arrivals
carry `BLOCK_TIME_UTC_mvt` (their in-block) on 100 % of rows, and one airframe turning
round at one stand is a physical fact rather than a coincidence of traffic.

**What is a label and what is not.**  A departure's `BLOCK_TIME_UTC_mvt` is blanked by the
ranking mask and is never read here.  An ARRIVAL's in-block is not masked and is an input.
The departure side contributes only `MVT_TIME_UTC_mvt`, its stand, its operator and its
type -- all inputs.

**The check is not the key.**  +/-60 s take-off agreement is measured and reported, never
used to select, so it stays an independent test of the chain: if the chain identifies the
right departure, the segment's own take-off must agree with that departure's; if the chain
is coincidence, agreement sits at the rate the shuffles show.

Two placebos, both per airport:
  * `day`   -- every departure's take-off moved 24 h, the chain unchanged;
  * `stand` -- `STAND_mvt` permuted among the airport's departures, which breaks the key
    itself and is the sharper of the two.

    OMP_NUM_THREADS=1 .venv/bin/python experiments/exp-046-surfacefit/code/turnaround.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arms as A                                                      # noqa: E402
import block as B                                                     # noqa: E402
import prc.data as D                                                  # noqa: E402

OUT = EXP / "analysis/turnaround.json"
MONTHS = ["2026-01", "2026-07"]
MAX_TURNAROUND_S = 24 * 3600.0
CHECK_S = 60.0
# The claimed set.  A departure is claimed when a pass MATCHED it, which is not the
# same as "it has a clock": a strict match onto a segment with a take-off but no
# off-block is claimed and clockless (6,063 rows in 2026-01, 5,472 in 2026-07).  The
# first version of this file used the clock test, let those departures compete as
# free candidates, and two-way uniqueness then declined 78 real placements they
# contended for -- `analysis/chain-reconcile.json`, the disagreement with
# `block.build` that `analysis/chain-spec.md` section 8 required be resolved before
# any commit.  The approximation is kept behind a flag so the reconciliation stays
# reproducible, and it defaults OFF.
FREE_IS_CLOCKLESS = False
SEED = 20260909


def secs(t: pd.Series) -> np.ndarray:
    return (t - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy("float64")


def chain(arr_ex: pd.DataFrame, dep_free: pd.DataFrame) -> pd.DataFrame:
    """Segment -> the one unclaimed departure its arrival's stand/operator/type implies."""
    if not len(arr_ex) or not len(dep_free):
        return pd.DataFrame(columns=["seg_key", "MVT_ID_mvt", "dep_t", "wheels_off"])
    m = arr_ex.merge(dep_free, on=["stand", "op", "type"], how="inner",
                     suffixes=("_a", "_d"))
    m = m[(m.dep_t > m.in_block) & (m.dep_t - m.in_block <= MAX_TURNAROUND_S)]
    if not len(m):
        return pd.DataFrame(columns=["seg_key", "MVT_ID_mvt", "dep_t", "wheels_off"])
    n = m.groupby("seg_key").MVT_ID_mvt.transform("size")
    m = m[n == 1]                                    # uniqueness inside the chained set
    m = m[~m.MVT_ID_mvt.duplicated(keep=False)]      # and one-to-one the other way
    return m[["seg_key", "MVT_ID_mvt", "dep_t", "wheels_off"]]


def main() -> None:
    rk = D.load_ranking()
    rk["_m"] = rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    tr = pd.concat([pd.read_parquet(D.month_file(x),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for x in A.prefix_months_for("")], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
    rule, _ = A.serving_rule()
    rng = np.random.default_rng(SEED)

    out = {"what": ("the turnaround chain: an unclaimed segment's inbound callsign names a "
                    "ranking ARRIVAL, and the departure is the one from that arrival's "
                    "STAND by the same operator and type after its in-block"),
           "ordered_by": "the desk, 2026-09-09, the repair after RESULTS 21",
           "key": ["STAND_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt",
                   "the arrival's in-block precedes the departure's take-off"],
           "check_not_key_s": CHECK_S, "max_turnaround_s": MAX_TURNAROUND_S,
           "no_departure_block_time_is_read": True,
           "matcher_sha256": A.MATCHER_SHA, "by_airport_month": {}}

    for month in MONTHS:
        dep_all = rk[rk.PHASE_mvt.eq("DEP") & rk._m.eq(month)].drop(columns="_m").copy()
        arr_all = rk[rk.PHASE_mvt.eq("ARR") & rk._m.eq(month)]
        blk = B.build(dep_all, tr, require_rule=rule)
        coh = B.cohort(dep_all, B.prefix_map(tr))
        mm = coh[["MVT_ID_mvt", "ADEP_mvt", "callsign", "mvt_s"]].merge(
            blk[["MVT_ID_mvt", "obs_icao", "obs_taxi", "obs_matched"]],
            on="MVT_ID_mvt", how="left")
        mm["offblock"] = mm.mvt_s - mm.obs_taxi
        days = sorted(dep_all.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").unique())

        for a in sorted(set(dep_all.ADEP_mvt.dropna()) & set(B.AIRPORTS)):
            r = mm[mm.ADEP_mvt.eq(a)]
            cl = r[["obs_icao", "offblock"]].dropna()
            claimed = set(zip(cl.obs_icao.astype(str),
                              cl.offblock.round(0).astype("int64")))
            dep_keys = set(B.norm_callsign(r.callsign).dropna()) - {""}

            ar = arr_all[arr_all.ADES_mvt.eq(a)].copy()
            ar["key"] = B.norm_callsign(ar.CALLSIGN_flt)
            ar = ar[ar.key.ne("") & ar.BLOCK_TIME_UTC_mvt.notna()]
            arr_keys = set(ar.key)

            seg = B.load_summaries([a], days, require_rule=rule)
            if not len(seg):
                continue
            seg = seg[seg.kind.eq("departure") & seg.obs_offblock.notna()
                      & seg.obs_wheels_off.notna()].copy()
            seg["seg_key"] = range(len(seg))
            seg = seg[pd.to_datetime(seg.seg_start, unit="s", utc=True)
                      .dt.strftime("%Y-%m").eq(month)]
            if not len(seg):
                continue
            is_cl = np.array([(i, o) in claimed for i, o in
                              zip(seg.icao.astype(str),
                                  seg.obs_offblock.round(0).astype("int64"))], dtype=bool)
            un = seg[~is_cl]
            if not len(un):
                continue
            c3 = B.broadcast_forms(un)
            c3["_dep"] = c3.join_key.isin(dep_keys)
            c3["_arr"] = c3.join_key.isin(arr_keys)
            per = c3.groupby("seg_key")[["_dep", "_arr"]].any()
            elig = per.index[(~per._dep) & per._arr]
            if not len(elig):
                continue

            # the inbound leg: the LATEST arrival of that callsign whose in-block precedes
            # the ground run's own off-block
            ex = c3[c3._arr & c3.seg_key.isin(elig)][["seg_key", "join_key"]]
            sg = un[un.seg_key.isin(elig)][["seg_key", "obs_offblock", "obs_wheels_off"]]
            ex = ex.merge(sg, on="seg_key", how="inner")
            av = ar[["key", "STAND_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt",
                     "BLOCK_TIME_UTC_mvt"]].copy()
            av["in_block"] = secs(av.BLOCK_TIME_UTC_mvt)
            j = ex.merge(av, left_on="join_key", right_on="key", how="inner")
            j = j[j.in_block <= j.obs_offblock]
            if not len(j):
                continue
            j = j.sort_values("in_block").drop_duplicates("seg_key", keep="last")
            arr_ex = pd.DataFrame({
                "seg_key": j.seg_key.to_numpy(),
                "stand": j.STAND_mvt.to_numpy(),
                "op": j.AIRCRAFT_OPERATOR_flt.to_numpy(),
                "type": j.AIRCRAFT_TYPE_mvt.to_numpy(),
                "in_block": j.in_block.to_numpy(),
                "wheels_off": j.obs_wheels_off.to_numpy()}).dropna()

            # `obs_matched` is "a pass matched this departure", which is the block's own
            # record of the claimed set and the one `block.build` uses; `obs_taxi.notna()`
            # is the narrower "and the match carries an off-block".  See FREE_IS_CLOCKLESS.
            _taken = (r.obs_taxi.notna() if FREE_IS_CLOCKLESS
                      else r.obs_matched.fillna(False).astype(bool))
            dfree = dep_all[dep_all.ADEP_mvt.eq(a)
                            & ~dep_all.MVT_ID_mvt.isin(
                                r.loc[_taken, "MVT_ID_mvt"])].copy()
            base = pd.DataFrame({
                "MVT_ID_mvt": dfree.MVT_ID_mvt.to_numpy(),
                "stand": dfree.STAND_mvt.to_numpy(),
                "op": dfree.AIRCRAFT_OPERATOR_flt.to_numpy(),
                "type": dfree.AIRCRAFT_TYPE_mvt.to_numpy(),
                "dep_t": secs(dfree.MVT_TIME_UTC_mvt)}).dropna()

            rec = {"eligible_segments": int(len(arr_ex)),
                   "unclaimed_departures": int(len(base))}
            cells = {}
            for label, d in (("real", base),
                             ("placebo_day", base.assign(dep_t=base.dep_t + 86400.0)),
                             ("placebo_stand",
                              base.assign(stand=rng.permutation(base.stand.to_numpy())))):
                p = chain(arr_ex, d)
                agree = (np.abs(p.wheels_off - p.dep_t) <= CHECK_S) if len(p) else []
                rec[label] = {
                    "placed": int(len(p)),
                    "take_off_agrees_within_60s": int(np.sum(agree)) if len(p) else 0,
                    "agreement_share": (round(float(np.mean(agree)), 4) if len(p) else None)}
                # the GATED pass: the chain AND the take-off agreement.  The check becomes
                # part of the key here, which is only legitimate because the placebos below
                # are re-run against the same conjunction -- a key that a shuffle satisfies
                # as often as the truth is not a key.
                g = p[agree] if len(p) else p
                rec[label]["gated"] = int(len(g))
                # queue's addition: its arm aggregates over 10-minute CELLS, so a bias that
                # concentrates matters more than a per-row rate.  Absolute cells are what
                # the arm actually reads; time-of-day cells test the daily-repeat concern
                # queue raised, that a stand chain repeating each day lands in the same
                # cells every day.
                if len(g):
                    cells[label] = {
                        "abs": pd.Series((g.dep_t // 600).astype("int64")).value_counts(),
                        "tod": pd.Series(((g.dep_t % 86400) // 600).astype("int64"))
                        .value_counts()}
                else:
                    cells[label] = {"abs": pd.Series(dtype="int64"),
                                    "tod": pd.Series(dtype="int64")}
            rec["cells"] = {}
            for kind in ("abs", "tod"):
                real = cells["real"][kind]
                for label in ("placebo_day", "placebo_stand"):
                    pl = cells[label][kind]
                    idx = real.index.union(pl.index)
                    rv = real.reindex(idx, fill_value=0)
                    pv = pl.reindex(idx, fill_value=0)
                    # two different things, and queue's gate turns on the difference: a
                    # CONTESTED cell holds real placements the placebo matches or beats,
                    # and a PLACEBO-ONLY cell holds none of ours at all.  The first is
                    # what pollutes an aggregate a row of ours reads; the second pollutes
                    # a cell we put nothing in.
                    has_real = rv > 0
                    contested = has_real & (pv >= rv)
                    rec["cells"][f"{kind}_{label}"] = {
                        "cells_with_a_real_placement": int(has_real.sum()),
                        "contested_cells_placebo_ge_real": int(contested.sum()),
                        "contested_share_of_cells": (
                            round(float(contested.sum() / has_real.sum()), 4)
                            if has_real.sum() else None),
                        "real_rows_in_contested_cells": int(rv[contested].sum()),
                        "share_of_real_rows_contested": (
                            round(float(rv[contested].sum() / rv.sum()), 4)
                            if rv.sum() else None),
                        "placebo_only_cells": int(((~has_real) & (pv > 0)).sum()),
                        "worst_cell_placebo_over_real": (
                            round(float((pv[has_real] / rv[has_real]).max()), 3)
                            if has_real.any() else None)}
            out["by_airport_month"][f"{month} {a}"] = rec
            print(f"{month} {a}: eligible {rec['eligible_segments']:5d}  "
                  f"real {rec['real']['placed']:5d} agree "
                  f"{rec['real']['agreement_share']}  | day "
                  f"{rec['placebo_day']['placed']:5d}/"
                  f"{rec['placebo_day']['agreement_share']}  | stand "
                  f"{rec['placebo_stand']['placed']:5d}/"
                  f"{rec['placebo_stand']['agreement_share']}", flush=True)

    b = out["by_airport_month"]
    tot = {k: sum(v[k]["placed"] for v in b.values())
           for k in ("real", "placebo_day", "placebo_stand")}
    gat = {k: sum(v[k]["gated"] for v in b.values())
           for k in ("real", "placebo_day", "placebo_stand")}
    agr = {k: sum(v[k]["take_off_agrees_within_60s"] for v in b.values())
           for k in ("real", "placebo_day", "placebo_stand")}
    out["totals"] = {
        "gated_placed": gat,
        "gated_placebo_share_of_real": {
            k: (round(gat[k] / gat["real"], 4) if gat["real"] else None)
            for k in ("placebo_day", "placebo_stand")},
        "cells": {kind + "_" + lab: {
            k: sum(v["cells"][f"{kind}_{lab}"][k] for v in b.values())
            for k in ("cells_with_a_real_placement", "contested_cells_placebo_ge_real",
                      "real_rows_in_contested_cells", "placebo_only_cells")}
            for kind in ("abs", "tod") for lab in ("placebo_day", "placebo_stand")},
        "placed": tot, "agreeing": agr,
        "agreement_share": {k: (round(agr[k] / tot[k], 4) if tot[k] else None)
                            for k in tot},
        "placebo_share_of_real": {k: (round(tot[k] / tot["real"], 4) if tot["real"] else None)
                                  for k in ("placebo_day", "placebo_stand")}}
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\ntotals:", json.dumps(out["totals"]))
    print("written", OUT)


if __name__ == "__main__":
    main()
