"""Gate (b): §12.3's per-airport false-match bound, re-asserted with the turnaround pass on.

`analysis/chain-spec.md` §3(b) requires every airport to stay under
`test_block.month_false_match_bound(density)` with the pass running.  `code/finding7.py`
asserts that bound for the two shipped passes and imports `code/block.py`; the turnaround
pass is not in the shared matcher yet, so this file runs the same measurement against the
branch copy instead of moving `code/block.py` to take a reading.

The measurement is `finding7.py`'s: shift a whole ranking month's departures by each of
`test_block.PLACEBO_SHIFTS_H`, rebuild, and count the rows that still match.  A shifted
departure cannot have a true ground run, so every match it takes is false.  The one change
is that a **mode-3** row now counts in that false share beside strict and wide -- the pass
must be bounded by the same rule it joins, not exempt from it.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/chainboundb.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arms as A                                                      # noqa: E402
import prc.data as D                                                  # noqa: E402
import test_block as T                                                # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "block_with_chain", EXP / "analysis/chain-pass/block_with_chain.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["block_with_chain"] = B
_spec.loader.exec_module(B)

MONTH = "2026-01"          # `finding7.py`'s month, so the two tables are comparable
OUT = EXP / "analysis/chain-bound-b.json"


def density(dep: pd.DataFrame, rule: str) -> dict:
    """All-kind ground segments per departure, per airport: `finding7.segment_density`.

    A property of the population and the rule, not of the pass, so the pass cannot move it
    and the bound it sets is the same number `finding7.json` records.
    """
    days = sorted(dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").unique())
    seg = B.load_summaries(B.AIRPORTS, days, require_rule=rule)
    n_all = seg.groupby(seg.airport.astype(str)).size()
    n_dep = dep.groupby(dep.ADEP_mvt.astype(str)).size()
    return {a: round(float(n_all.get(a, 0) / max(n_dep.get(a, 0), 1)), 4)
            for a in sorted(n_all.index)}


def by_airport(b: pd.DataFrame, coh: pd.DataFrame) -> dict:
    m = b.merge(coh[["MVT_ID_mvt", "ADEP_mvt"]], on="MVT_ID_mvt", how="left",
                validate="one_to_one")
    return {str(a): {"rows": int(len(g)),
                     "strict": int((g.match_mode == 0).sum()),
                     "wide": int(g.match_mode.isin([1, 2]).sum()),
                     "turnaround": int((g.match_mode == 3).sum())}
            for a, g in m.groupby("ADEP_mvt", observed=True)}


def main() -> None:
    rk = D.load_ranking()
    dep = rk[rk.PHASE_mvt.eq("DEP")
             & rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").eq(MONTH)].copy()
    arr = rk[rk.PHASE_mvt.eq("ARR")
             & rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").eq(MONTH)]
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in A.prefix_months_for("")], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
    rule, _ = A.serving_rule()
    coh = B.cohort(dep, B.prefix_map(tr))

    out = {"what": "gate (b): the per-airport false-match bound with the turnaround pass on",
           "spec": "analysis/chain-spec.md section 3(b)",
           "month": MONTH, "shifts_h": T.PLACEBO_SHIFTS_H,
           "measured_against": "code/finding7.py's method, on the branch copy",
           "block_py_unchanged": "6fdf39519eb0f500c06230e2972d3c8300aa6222ed8d9b003998ddf6e286093b",
           "floor": T.MONTH_FALSE_MATCH_FLOOR,
           "per_density": T.MONTH_FALSE_MATCH_PER_DENSITY}

    t0 = time.time()
    real = B.build(dep.copy(), tr, require_rule=rule, arrivals=arr)
    out["real"] = {"by_airport": by_airport(real, coh),
                   "turnaround_placed": real.attrs["turnaround"]["placed"]}
    print(f"real: {out['real']['turnaround_placed']} placed by the pass  "
          f"{time.time() - t0:.0f}s", flush=True)

    out["by_shift"] = {}
    for h in T.PLACEBO_SHIFTS_H:
        t0 = time.time()
        sh = dep.copy()
        sh["MVT_TIME_UTC_mvt"] = sh.MVT_TIME_UTC_mvt + pd.Timedelta(hours=h)
        pl = B.build(sh, tr, require_rule=rule, arrivals=arr)
        ba = by_airport(pl, B.cohort(sh, B.prefix_map(tr)))
        for a, v in ba.items():
            v["false"] = v["strict"] + v["wide"] + v["turnaround"]
            v["false_share"] = round(v["false"] / max(v["rows"], 1), 6)
        out["by_shift"][f"{h:+.1f}h"] = {
            "by_airport": ba,
            "turnaround_placed": pl.attrs["turnaround"]["placed"],
            "seconds": round(time.time() - t0, 1)}
        print(f"{h:+.1f}h: pass placed {pl.attrs['turnaround']['placed']}, "
              f"worst airport false share "
              f"{max(v['false_share'] for v in ba.values()):.6f}  "
              f"{time.time() - t0:.0f}s", flush=True)
        OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")

    dens = density(dep, rule)
    out["density_by_airport"] = dens
    out["by_airport"] = {}
    for a in sorted(out["real"]["by_airport"]):
        worst = max(out["by_shift"][k]["by_airport"].get(a, {}).get("false_share", 0.0)
                    for k in out["by_shift"])
        worst_ta = max(out["by_shift"][k]["by_airport"].get(a, {}).get("turnaround", 0)
                       for k in out["by_shift"])
        d = dens.get(a)
        if d is None:
            continue
        bound = T.month_false_match_bound(d)
        out["by_airport"][a] = {
            "density": d, "bound": round(bound, 6),
            "worst_false_share_across_shifts": round(worst, 6),
            "worst_turnaround_rows_in_a_shift": int(worst_ta),
            "headroom": (round(bound / worst, 2) if worst else None),
            "within_its_bound": bool(worst <= bound)}
    out["within_its_own_bound_everywhere"] = all(
        v["within_its_bound"] for v in out["by_airport"].values())
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")

    print("\nairport  density   false     bound    headroom  mode-3 in a shift  within")
    for a, v in sorted(out["by_airport"].items()):
        print(f"  {a}   {v['density']:6.3f}  {v['worst_false_share_across_shifts']:.6f}  "
              f"{v['bound']:.6f}  x{str(v['headroom']):>6s}  "
              f"{v['worst_turnaround_rows_in_a_shift']:6d}             "
              f"{v['within_its_bound']}")
    print("\nwithin its own bound everywhere:", out["within_its_own_bound_everywhere"])
    print(f"written {OUT}")
    assert out["within_its_own_bound_everywhere"], (
        "an airport is over the bound its own segment density allows; the pass does not ship")


if __name__ == "__main__":
    main()
