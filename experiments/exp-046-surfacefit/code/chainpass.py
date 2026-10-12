"""The turnaround pass's gates, run inside `block.build` before any commit.

`analysis/chain-spec.md` §3 registered the gates before the build.  This file runs the
three that are not enforced by the key itself, and the property the whole handshake rests
on:

  (a) **No existing value changes.**  Every month is built twice, `chain=False` and
      `chain=True`, and all 34 emitted columns are compared on the rows the pass did NOT
      place.  A pass that disturbs a row it did not place cannot ship whatever it reaches.
  (c) **The day placebo**, per airport: departure take-offs moved +24 h, the segments left
      where they are.
  (d) **The stand placebo**, per airport: `STAND_mvt` permuted among the airport's own
      departures, so the chain's shape survives and only the stand is wrong.

§8 of the spec asks one more thing of this file: the placebos must be re-run **inside
`block.py`** and agree with `code/turnaround.py`, which measured them outside it.  Two
implementations of one key that disagree is a defect in one of them, not a result, so the
comparison is written into the output rather than left to a reader.

Gate (b), §12.3's per-airport false-match bound, lives in `code/finding7.py` and is
re-asserted there with the pass on; it is named here and not duplicated.
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

# NOT `code/block.py`.  The pass is not in the shared matcher and does not enter it until
# the desk relays queue's GO on `analysis/chain-spec.md`.
_spec = importlib.util.spec_from_file_location(
    "block_with_chain", EXP / "analysis/chain-pass/block_with_chain.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["block_with_chain"] = B
_spec.loader.exec_module(B)

OUT = EXP / "analysis/chain-pass.json"
RANK = ["2026-01", "2026-07"]
PLACEBO_SHIFT_H = 24.0
SEED = 20260909
CELL_S = 600.0


def prefix_train() -> pd.DataFrame:
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in A.prefix_months_for("")], ignore_index=True)
    return tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")


def sweep(off: pd.DataFrame, on: pd.DataFrame) -> dict:
    """All 34 emitted columns, on the rows both blocks have and the pass did NOT place.

    The placed rows are excluded because on them a change is the POINT: `obs_matched`,
    `obs_has_clock` and `reach_class` are never null, so a legitimate placement reads as a
    changed value rather than a gained one.  The first version of this sweep did not
    exclude them and reported 23,595 "changed" values that were exactly the placements.
    """
    placed = set(on.loc[on.match_mode.eq(float(B.MODES["turnaround"])), "MVT_ID_mvt"])
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
            "columns_swept": int(len(a.columns)),
            "total_values_changed_on_rows_the_pass_did_not_place": changed}


def placed_frame(blk: pd.DataFrame, coh: pd.DataFrame) -> pd.DataFrame:
    """The rows the pass placed, with the airport and the 10-minute cell of the take-off."""
    p = blk[blk.match_mode.eq(float(B.MODES["turnaround"]))][["MVT_ID_mvt", "obs_wheels_off"]]
    p = p.merge(coh[["MVT_ID_mvt", "ADEP_mvt", "mvt_s"]], on="MVT_ID_mvt", how="left")
    p["abs_cell"] = np.floor(p.mvt_s / CELL_S)
    p["tod_cell"] = np.floor((p.mvt_s % 86400.0) / CELL_S)
    return p


def cells(real: pd.DataFrame, pl: pd.DataFrame, col: str) -> dict:
    """Queue's bar: no cell class where the placebo matches the real count.

    A false placement is noise for one of this lane's rows and BIAS for a whole cell of
    queue's arm, which aggregates over 10-minute cells.  So the per-row rate is not
    enough and the distribution over cells is read beside it.
    """
    r = real.groupby(["ADEP_mvt", col]).size()
    q = pl.groupby(["ADEP_mvt", col]).size()
    both = r.index.union(q.index)
    r = r.reindex(both, fill_value=0)
    q = q.reindex(both, fill_value=0)
    contested = (r > 0) & (q >= r)
    return {"cells_with_a_real_placement": int((r > 0).sum()),
            "contested_cells_placebo_ge_real": int(contested.sum()),
            "real_rows_in_contested_cells": int(r[contested].sum()),
            "placebo_only_cells": int(((r == 0) & (q > 0)).sum())}


def main() -> None:
    rk = D.load_ranking()
    rk["_m"] = rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    tr = prefix_train()
    rule, _ = A.serving_rule()
    rng = np.random.default_rng(SEED)
    out = {"what": "the turnaround pass's gates, run inside block.build",
           "ordered_by": "the desk, 2026-09-09, GO on analysis/chain-spec.md",
           "spec": "experiments/exp-046-surfacefit/analysis/chain-spec.md",
           "branch_copy": "analysis/chain-pass/block_with_chain.py",
           "shipped_matcher_sha256_unchanged": A.MATCHER_SHA,
           "gate_b_lives_in": "code/finding7.py, re-asserted with the pass on",
           "placebo_shift_h": PLACEBO_SHIFT_H, "seed": SEED, "cell_s": CELL_S,
           "months": {}}

    for m in RANK:
        dep = rk[rk.PHASE_mvt.eq("DEP") & rk._m.eq(m)].drop(columns="_m").copy()
        arr = rk[rk.PHASE_mvt.eq("ARR") & rk._m.eq(m)]
        coh = B.cohort(dep, B.prefix_map(tr))

        off = B.build(dep, tr, require_rule=rule, chain=False)
        on = B.build(dep, tr, require_rule=rule, arrivals=arr)
        rec = {"report": on.attrs["turnaround"], "sweep": sweep(off, on)}
        real = placed_frame(on, coh)

        # ---- gate (c): move the departures a day, leave the segments where they are ----
        sh = dep.copy()
        sh["MVT_TIME_UTC_mvt"] = sh.MVT_TIME_UTC_mvt + pd.Timedelta(hours=PLACEBO_SHIFT_H)
        pd_blk = B.build(sh, tr, require_rule=rule, arrivals=arr)
        pd_coh = B.cohort(sh, B.prefix_map(tr))

        # ---- gate (d): permute the stand among the airport's own departures ------------
        st = dep.copy()
        st["STAND_mvt"] = (st.groupby("ADEP_mvt", group_keys=False).STAND_mvt
                             .apply(lambda s: pd.Series(rng.permutation(s.to_numpy()),
                                                        index=s.index)))
        st_blk = B.build(st, tr, require_rule=rule, arrivals=arr)
        st_coh = B.cohort(st, B.prefix_map(tr))

        rec["placebo"] = {}
        for label, blk, ch in (("day", pd_blk, pd_coh), ("stand", st_blk, st_coh)):
            p = placed_frame(blk, ch)
            rec["placebo"][label] = {
                "placed": int(blk.attrs["turnaround"]["placed"]),
                "real_placed": int(on.attrs["turnaround"]["placed"]),
                "share_of_real": (round(len(p) / max(len(real), 1), 4)),
                "by_airport": {str(a): int(n) for a, n in
                               p.ADEP_mvt.value_counts().sort_index().items()},
                "real_by_airport": {str(a): int(n) for a, n in
                                    real.ADEP_mvt.value_counts().sort_index().items()},
                "cells_abs": cells(real, p, "abs_cell"),
                "cells_tod": cells(real, p, "tod_cell")}
        out["months"][m] = rec

        st_r = on.attrs["turnaround"]
        print(f"=== {m}: placed {st_r['placed']} "
              f"(of {st_r['of_them_an_arrival_identity_and_not_a_departure_one']} eligible, "
              f"{st_r['with_an_inbound_leg_before_their_own_off_block']} with a leg, "
              f"{st_r['after_uniqueness_both_ways']} unique)")
        print(f"      sweep: {rec['sweep']['total_values_changed_on_rows_the_pass_did_not_place']}"
              f" values changed on rows the pass did not place, over "
              f"{rec['sweep']['columns_swept']} columns and "
              f"{rec['sweep']['rows_compared']} rows")
        for label in ("day", "stand"):
            b = rec["placebo"][label]
            print(f"      placebo {label:5s}: {b['placed']:5d} against {b['real_placed']:5d}"
                  f"  ({100 * b['share_of_real']:.2f} %)   contested cells "
                  f"abs {b['cells_abs']['contested_cells_placebo_ge_real']}"
                  f"/{b['cells_abs']['cells_with_a_real_placement']}"
                  f"  tod {b['cells_tod']['contested_cells_placebo_ge_real']}"
                  f"/{b['cells_tod']['cells_with_a_real_placement']}")

    tot = {"placed": sum(out["months"][m]["report"]["placed"] for m in RANK),
           "values_changed_on_rows_the_pass_did_not_place":
               sum(out["months"][m]["sweep"]
                   ["total_values_changed_on_rows_the_pass_did_not_place"] for m in RANK)}
    for label in ("day", "stand"):
        tot[f"placebo_{label}"] = sum(
            out["months"][m]["placebo"][label]["placed"] for m in RANK)
    # `code/turnaround.py` measured the same key OUTSIDE `block.build`.  The two must
    # agree; a disagreement is a defect in one implementation, not a result (spec §8).
    t = json.loads((EXP / "analysis/turnaround.json").read_text())["totals"]
    tot["turnaround_py"] = {"placed": t["gated_placed"]["real"],
                            "placebo_day": t["gated_placed"]["placebo_day"],
                            "placebo_stand": t["gated_placed"]["placebo_stand"]}
    tot["agrees_with_turnaround_py"] = bool(
        tot["placed"] == t["gated_placed"]["real"])
    out["totals"] = tot
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(f"\ntotal placed {tot['placed']} (turnaround.py: {tot['turnaround_py']['placed']}, "
          f"agree {tot['agrees_with_turnaround_py']}), "
          f"placebos {tot['placebo_day']} / {tot['placebo_stand']}, "
          f"{tot['values_changed_on_rows_the_pass_did_not_place']} values changed on rows "
          f"the pass did not place")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
