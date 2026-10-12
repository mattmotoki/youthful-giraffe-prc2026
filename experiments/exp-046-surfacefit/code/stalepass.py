"""The third pass's evidence: it changes nothing it must not, and its gates.

`analysis/reach-spec.md` §3 registered four gates before the build.  This file measures the
three that are not enforced inside `block.build` itself, and the one property the whole
handshake rests on.

  0. **No existing value changes.**  Every month is built twice, `stale=False` and
     `stale=True`, and all 34 emitted columns are compared on the rows both have.  This is
     the sweep §14.1 records getting wrong the first time -- a five-column diff that missed
     three spillover rows -- so it is the full 34 here, not a sample.
  (a) uniqueness is inside the pass and is reported from its own funnel.
  (c) **The one-day placebo**, per airport, as `exp-047-queue` asked: its arm reads the
     NEIGHBOURS' clocks, so a false placement pollutes a whole 10-minute cell rather than
     one row.  The departures are moved a day and the segments are not; a real placement
     cannot survive that and a coincidental one will.
  (d) **Aircraft consistency**, per airport.  The file carries no registration and its
     operator is hashed, but `AIRCRAFT_TYPE_mvt` is on 99.9 % of rows.  A stale inbound
     identity means one airframe turning round, so the arrival that explains a placed
     segment and the departure it was placed on should be the same type.  A read, not a
     gate -- a code-share or an equipment swap breaks it honestly -- but a disagreement
     rate far above base would refute the mechanism.

Gate (b), §12.3's density-scaled per-airport false-match bound, is re-asserted by
`code/finding7.py` with the pass on; it is not duplicated here.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/stalepass.py
"""
from __future__ import annotations

import importlib.util
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
import prc.data as D                                                  # noqa: E402

# NOT `code/block.py`.  The third pass is not in the shared matcher and never was: it
# failed gate (c) and the matcher's sha never moved.  This file measures the CANDIDATE,
# which lives as a branch copy so the refutation stays reproducible.
_spec = importlib.util.spec_from_file_location(
    "block_with_third_pass", EXP / "analysis/stale-pass/block_with_third_pass.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["block_with_third_pass"] = B
_spec.loader.exec_module(B)

OUT = EXP / "analysis/stale-pass.json"
RANK = ["2026-01", "2026-07"]
PLACEBO_SHIFT_H = 24.0


def prefix_train() -> pd.DataFrame:
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in A.prefix_months_for("")], ignore_index=True)
    return tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")


def sweep(off: pd.DataFrame, on: pd.DataFrame) -> dict:
    """All 34 emitted columns, on the rows both blocks have and the pass did NOT place.

    The placed rows are excluded because on them a change is the POINT: `obs_matched`,
    `obs_has_clock` and `reach_class` are never null, so a legitimate placement reads as a
    changed value rather than a gained one.  The question this sweep exists to answer is
    the other one -- whether the pass disturbs a row it did not place -- and it can only
    be asked of the rows it did not place.
    """
    placed = set(on.loc[on.match_mode.eq(float(B.MODES["stale_inbound"])), "MVT_ID_mvt"])
    off = off[~off.MVT_ID_mvt.isin(placed)]
    on = on[~on.MVT_ID_mvt.isin(placed)]
    a = off.set_index("MVT_ID_mvt").sort_index()
    b = on.set_index("MVT_ID_mvt").sort_index()
    assert a.index.equals(b.index), "the two builds do not carry the same rows"
    rep, changed = {}, 0
    for c in a.columns:
        x, y = a[c], b[c]
        both = x.notna() & y.notna()
        if x.dtype.kind in "fc":
            diff = int((np.abs(x[both].to_numpy("float64")
                               - y[both].to_numpy("float64")) > 1e-9).sum())
        else:
            diff = int((x[both].astype(str) != y[both].astype(str)).sum())
        gained, lost = int((~x.notna() & y.notna()).sum()), int((x.notna() & ~y.notna()).sum())
        rep[c] = {"changed_where_both_present": diff, "gained": gained, "lost": lost}
        changed += diff
    return {"columns": rep, "rows_compared": int(len(a)),
            "total_values_changed_on_rows_the_pass_did_not_place": changed}


def main() -> None:
    rk = D.load_ranking()
    rk["_m"] = rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    tr = prefix_train()
    rule, _ = A.serving_rule()
    out = {"what": "the third pass's evidence: what it does not change, and its gates",
           "ordered_by": "the desk, 2026-09-09, GO on analysis/reach-spec.md",
           "window_s": B.STALE_WINDOW, "placebo_shift_h": PLACEBO_SHIFT_H,
           "gate_b_lives_in": "code/finding7.py, re-asserted with the pass on",
           "months": {}}

    for m in RANK:
        dep = rk[rk.PHASE_mvt.eq("DEP") & rk._m.eq(m)].drop(columns="_m").copy()
        arr = rk[rk.PHASE_mvt.eq("ARR") & rk._m.eq(m)]
        off = B.build(dep, tr, require_rule=rule, stale=False)
        on = B.build(dep, tr, require_rule=rule, arrivals=arr)
        rec = {"stale_report": on.attrs["stale_inbound"], "sweep": sweep(off, on)}

        # ---- gate (c): move the departures a day, leave the segments where they are ----
        sh = dep.copy()
        sh["MVT_TIME_UTC_mvt"] = sh.MVT_TIME_UTC_mvt + pd.Timedelta(hours=PLACEBO_SHIFT_H)
        pl = B.build(sh, tr, require_rule=rule, arrivals=arr)
        rec["placebo"] = {"placed": pl.attrs["stale_inbound"]["placed"],
                          "by_airport": pl.attrs["stale_inbound"].get("by_airport", {}),
                          "real_placed": on.attrs["stale_inbound"]["placed"],
                          "real_by_airport": on.attrs["stale_inbound"].get("by_airport", {})}

        # ---- gate (d): the arrival that explains the segment, and the departure ---------
        got = dep[["MVT_ID_mvt", "ADEP_mvt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"]].merge(
            on[["MVT_ID_mvt", "obs_icao", "obs_taxi", "match_mode"]],
            on="MVT_ID_mvt", how="inner")
        got = got[got.match_mode.eq(float(B.MODES["stale_inbound"]))].copy()
        got["mvt_s"] = (got.MVT_TIME_UTC_mvt - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
        got["offblock"] = got.mvt_s - got.obs_taxi
        days = sorted(dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").unique())
        arr_t = arr[["ADES_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"]].copy()
        arr_t["key"] = B.norm_callsign(arr_t.CALLSIGN_flt)
        arr_t["t"] = (arr_t.MVT_TIME_UTC_mvt - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
        types = {}
        for ap, g in got.groupby("ADEP_mvt"):
            seg = B.load_summaries([str(ap)], days, require_rule=rule)
            if not len(seg):
                continue
            seg = seg[seg.kind.eq("departure")].copy()
            seg["ob"] = seg.obs_offblock.round(0)
            look = seg.set_index(["icao", "ob"])
            ak = arr_t[arr_t.ADES_mvt.eq(str(ap))]
            by_key = {k: v.sort_values("t") for k, v in ak.groupby("key")}
            agree = disagree = unknown = 0
            for icao, ob, dt, t in zip(g.obs_icao, g.offblock.round(0),
                                       g.AIRCRAFT_TYPE_mvt, g.mvt_s):
                try:
                    row = look.loc[(icao, ob)]
                except KeyError:
                    unknown += 1
                    continue
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                forms = [B.norm_callsign(pd.Series([f]))[0]
                         for f in str(row.callsigns_all or row.callsign_modal or "").split("|")]
                cand = [by_key[f] for f in forms if f in by_key]
                if not cand or pd.isna(dt):
                    unknown += 1
                    continue
                c = pd.concat(cand)
                c = c[c.t <= t]                    # the inbound leg, not a later one
                if not len(c):
                    unknown += 1
                    continue
                at = c.sort_values("t").AIRCRAFT_TYPE_mvt.iloc[-1]
                if pd.isna(at):
                    unknown += 1
                elif str(at) == str(dt):
                    agree += 1
                else:
                    disagree += 1
            n = agree + disagree
            types[str(ap)] = {"agree": agree, "disagree": disagree, "unknown": unknown,
                              "agree_share": round(agree / n, 4) if n else None}
        rec["aircraft_type"] = types
        out["months"][m] = rec

        st = rec["stale_report"]
        print(f"=== {m}: placed {st['placed']} "
              f"(of {st['of_them_an_arrival_identity_and_not_a_departure_one']} eligible; "
              f"{st['declined_not_unique']} not unique, "
              f"{st['declined_contended_one_to_one']} contended)", flush=True)
        print(f"    sweep: {rec['sweep']['total_values_changed_on_rows_the_pass_did_not_place']} "
              f"values changed across all 34 columns on rows it did not place", flush=True)
        print(f"    placebo (+{PLACEBO_SHIFT_H:.0f} h): placed {rec['placebo']['placed']} "
              f"against {rec['placebo']['real_placed']} real", flush=True)
        for ap, v in sorted(types.items()):
            print(f"      type {ap}: agree {v['agree']} disagree {v['disagree']} "
                  f"unknown {v['unknown']} ({v['agree_share']})", flush=True)

    tot = sum(v["sweep"]["total_values_changed_on_rows_the_pass_did_not_place"] for v in out["months"].values())
    pl = sum(v["placebo"]["placed"] for v in out["months"].values())
    re_ = sum(v["placebo"]["real_placed"] for v in out["months"].values())
    out["verdict"] = {
        "values_changed_on_rows_the_pass_did_not_place": tot,
        "leaves_every_other_row_alone": tot == 0,
        "placed_real": re_, "placed_under_the_one_day_shift": pl,
        "placebo_share_of_real": round(pl / re_, 5) if re_ else None}
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\nverdict:", json.dumps(out["verdict"]))
    print("written", OUT)


if __name__ == "__main__":
    main()
