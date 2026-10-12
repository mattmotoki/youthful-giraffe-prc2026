"""The mis-timed match, as a rule on the row's own observation rather than on an airport.

`analysis/lfpg-january.json` found the cause of LFPG's 2026-01 doubling: the airport's wide
rows are the rows the block ALREADY FLAGS, `clock_ok == 0` -- the observed wheels-off is
more than a minute from the row's own take-off, so the ground run attached to the row is
not the run that produced that take-off.  Their share at LFPG runs 4.3 / 5.3 / 3.8 % in
2025-07, 2025-12 and 2026-07 and **23.1 %** in 2026-01, and their `dt_takeoff` median there
is -906 s.  The observation is not noisier in January; six times as many rows are matched to
the wrong minute.

That is a per-ROW property of the row's own observation, computable on every fold row and
every ranking row with no label, which is what the assignment's repair clause asks for --
"exclude it by a rule fitted off the scored rows, not by hand" -- and it is a better
instrument than an airport name: it names the mechanism and it generalises.

This prices it, on the fold side, with NO REFIT: the saved update vector of the arm that
ships, with the flagged rows zeroed, priced on the same contract as the arm itself
(`heldprice.py`'s operation with a different mask).  Four rules are compared on one table:

  none          the arm as priced
  lfpg          the airport hold that ships today
  clock_ok      every row the block flags, at every airport
  clock_ok_lfpg the flagged rows at LFPG only

and the ranking-side census says how many rows each would return to the base.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/clockok.py
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
ROOT = EXP.parent.parent

import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402

ARM = "c_lgb_shipped@leaves_127x300_lr050"
OUT = EXP / "analysis/clock-ok.json"


def masks(p: pd.DataFrame) -> dict[str, np.ndarray]:
    """The four rules as row masks: True means "return this row to the base"."""
    ap = p.airport.to_numpy().astype(str)
    bad = ~(pd.to_numeric(p.clock_ok, errors="coerce").to_numpy("float64") > 0.5)
    return {"none": np.zeros(len(p), bool),
            "lfpg": ap == "LFPG",
            "clock_ok": bad,
            "clock_ok_lfpg": bad & (ap == "LFPG")}


def main() -> None:
    rng = np.random.default_rng(A.SEED)
    pan = A.panel_with_block(BA.load())
    z = np.load(S.npz_path(ARM))
    full = {f: z[f].astype("float64") for f in A.FOLDS}
    out = {"what": __doc__.strip().splitlines()[0], "arm": ARM,
           "clock_ok": "block.py:598 -- |dt_takeoff| <= 60 s",
           "no_refit": ("the same saved update vector with a mask; every rule is read on "
                        "one and the same fit, so the comparison is of RULES and not of "
                        "models"),
           "rules": {}}

    # --- the census, folds and ranking, on the with-clock class --------------
    cen = {}
    for f in A.FOLDS:
        p = pan[f][pan[f].usable.to_numpy(bool)]
        bad = ~(pd.to_numeric(p.clock_ok, errors="coerce").to_numpy("float64") > 0.5)
        cen[A.FOLD_MONTH[f]] = {
            "with_clock_rows": int(len(p)), "flagged": int(bad.sum()),
            "flagged_share": round(float(bad.mean()), 4),
            "by_airport": {str(a): round(float(bad[p.airport.to_numpy().astype(str) == a]
                                              .mean()), 4)
                           for a in sorted(set(p.airport.astype(str)))}}
    dep, _ = S.ranking_block()
    for m, g0 in dep.groupby("rank_month"):
        g = g0[g0.usable.to_numpy(bool) & ~g0.noac.to_numpy(bool)]
        bad = ~(pd.to_numeric(g.clock_ok, errors="coerce").to_numpy("float64") > 0.5)
        cen[str(m)] = {
            "with_clock_rows": int(len(g)), "flagged": int(bad.sum()),
            "flagged_share": round(float(bad.mean()), 4),
            "by_airport": {str(a): round(float(bad[g.airport.to_numpy().astype(str) == a]
                                              .mean()), 4)
                           for a in sorted(set(g.airport.astype(str)))}}
    out["flagged_share_by_month"] = cen

    # --- the four rules on one table ----------------------------------------
    for name in ("none", "lfpg", "clock_ok", "clock_ok_lfpg"):
        kept, zeroed = {}, {}
        for f in A.FOLDS:
            d = full[f].copy()
            m = masks(pan[f])[name]
            zeroed[f] = int((d[m] != 0).sum())
            d[m] = 0.0
            kept[f] = d
        rec = BA.price(pan, kept, rng)
        rec = {k: rec[k] for k in ("board_all_row", "board_clean", "board_december",
                                   "every_fold_interval_clear") if k in rec}
        rec["rows_returned_to_the_base_by_fold"] = zeroed
        rec["tail_clauses"] = {f: A.tail_clauses(pan[f], kept[f])
                               ["registered_clause_y_over_3600s"] for f in A.FOLDS}
        out["rules"][name] = rec
        g = rec["board_all_row"]["gain_s"]
        ci = rec["board_all_row"]["interval"]["ci95"]
        t = {f: round(rec["tail_clauses"][f]["rmse_gain_s"], 3) for f in A.FOLDS}
        print(f"{name:15s} all-row {g:+8.4f} [{ci[0]:+7.3f},{ci[1]:+7.3f}] "
              f"clean {rec['board_clean']['gain_s']:+8.4f}  zeroed {zeroed}  tail {t}",
              flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # --- what each rule would return to the base on the ranking side --------
    served = dep.usable.to_numpy(bool) & ~dep.noac.to_numpy(bool)
    ap = dep.airport.to_numpy().astype(str)
    bad = ~(pd.to_numeric(dep.clock_ok, errors="coerce").to_numpy("float64") > 0.5)
    rank = {}
    for name, m in {"lfpg": ap == "LFPG", "clock_ok": bad,
                    "clock_ok_lfpg": bad & (ap == "LFPG")}.items():
        held = served & m
        rank[name] = {
            "rows_returned_to_the_base": int(held.sum()),
            "share_of_the_served_class": round(float(held.sum() / served.sum()), 4),
            "by_airport": {str(a): int(((ap == a) & held).sum())
                           for a in sorted(set(ap[held]))},
            "by_month": {str(mm): int((held & (dep.rank_month.to_numpy() == mm)).sum())
                         for mm in sorted(set(dep.rank_month.to_numpy()))}}
    out["ranking_side"] = rank
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print(json.dumps(rank, indent=1, default=float)[:1200])
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
