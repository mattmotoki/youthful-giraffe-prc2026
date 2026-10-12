"""Is LFPG's 2026-01 the only served month whose observation runs outside its own fit?

`analysis/lfpg-january.json` measured the doubling: 23.1 % of LFPG's 2026-01 with-clock
rows carry an `obs_taxi` above the p95 of LFPG's own FITTING rows, against 4.3 / 5.3 % in
the two fitting months and 3.8 % in 2026-07, and those rows are mis-timed matches -- their
`dt_takeoff` median is -906 s and their `clock_ok` median is 0.

`analysis/clock-ok.json` then refuted the obvious rule: holding every row the block flags
costs +8.8729 -> +6.1028 s and fails the july tail at -5.222, because the flagged rows PAY
on the folds.  So the flag is not the rule; the mismatch between the fitted mix and the
served mix is.

This asks the question generally, so the answer is a rule and not a name: **for every
airport, how far outside its own fitting months does its served month run?**  One
threshold per airport, learned on the fitting months only, applied to every month.  If
LFPG is alone, the hold has a reason; if it is not, the hold is arbitrary and the rule has
to be either widened or dropped.  The per-row form is priced on the fold side with no
refit, so its cost is a number and not an argument.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/widerule.py
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
Q = 95
OUT = EXP / "analysis/wide-rule.json"


def main() -> None:
    rng = np.random.default_rng(A.SEED)
    pan = A.panel_with_block(BA.load())
    dep, _ = S.ranking_block()
    out = {"what": __doc__.strip().splitlines()[0], "arm": ARM,
           "threshold": f"each airport's own p{Q} of obs_taxi, learned on FITTING rows only",
           "no_refit": "the saved update vector with a mask; one fit, four rules"}

    # --- one threshold per airport per rotation, learned on the fitting rows -
    fold_use = {f: [g for g in A.FOLDS if g != f] for f in A.FOLDS}
    def cut_from(folds: list[str]) -> dict[str, float]:
        d = pd.concat([pan[g][pan[g].usable.to_numpy(bool)] for g in folds],
                      ignore_index=True)
        return {str(a): float(np.nanpercentile(g.obs_taxi.to_numpy("float64"), Q))
                for a, g in d.groupby("airport", observed=True)
                if np.isfinite(g.obs_taxi.to_numpy("float64")).sum() >= 50}

    cuts_rank = cut_from(list(A.FOLDS))          # for the ranking months: all three folds
    share = {}
    for f in A.FOLDS:
        cu = cut_from(fold_use[f])
        p = pan[f][pan[f].usable.to_numpy(bool)]
        rec = {}
        for a, g in p.groupby("airport", observed=True):
            c = cu.get(str(a))
            if c is None:
                continue
            v = g.obs_taxi.to_numpy("float64")
            rec[str(a)] = {"rows": int(len(v)), "cut_s": round(c, 1),
                           "wide_share": round(float(np.nanmean(v > c)), 4)}
        share[A.FOLD_MONTH[f]] = rec
    for m, g0 in dep.groupby("rank_month"):
        g0 = g0[g0.usable.to_numpy(bool) & ~g0.noac.to_numpy(bool)]
        rec = {}
        for a, g in g0.groupby("airport", observed=True):
            c = cuts_rank.get(str(a))
            if c is None:
                continue
            v = g.obs_taxi.to_numpy("float64")
            rec[str(a)] = {"rows": int(len(v)), "cut_s": round(c, 1),
                           "wide_share": round(float(np.nanmean(v > c)), 4)}
        share[str(m)] = rec
    out["wide_share_by_airport_month"] = share

    folds_m = [A.FOLD_MONTH[f] for f in A.FOLDS]
    ranks_m = [m for m in share if m not in folds_m]
    table = {}
    for a in sorted({a for m in share for a in share[m]}):
        fv = [share[m][a]["wide_share"] for m in folds_m if a in share[m]
              and share[m][a]["rows"] >= 50]
        for m in ranks_m:
            if a not in share[m]:
                continue
            rv = share[m][a]["wide_share"]
            table.setdefault(a, {})[m] = {
                "served_wide_share": rv,
                "fitted_wide_share": round(float(np.mean(fv)), 4) if fv else None,
                "ratio": (round(float(rv / np.mean(fv)), 2)
                          if fv and np.mean(fv) > 0 else None),
                "served_rows": share[m][a]["rows"]}
    out["served_over_fitted"] = table
    worst = sorted(((v["ratio"], a, m) for a, d in table.items() for m, v in d.items()
                    if v["ratio"] is not None and v["served_rows"] >= 200), reverse=True)
    out["ranked_airport_months"] = [{"airport": a, "month": m, "ratio": r,
                                     "rows": table[a][m]["served_rows"]}
                                    for r, a, m in worst]

    # --- the per-row form, priced on the folds with no refit -----------------
    z = np.load(S.npz_path(ARM))
    full = {f: z[f].astype("float64") for f in A.FOLDS}
    def price(mask_of) -> dict:
        kept, zeroed = {}, {}
        for f in A.FOLDS:
            d = full[f].copy()
            m = mask_of(f)
            zeroed[f] = int((d[m] != 0).sum())
            d[m] = 0.0
            kept[f] = d
        rec = BA.price(pan, kept, rng)
        return {"board_all_row": rec["board_all_row"], "board_clean": rec["board_clean"],
                "board_december": rec.get("board_december"),
                "every_fold_interval_clear": rec["every_fold_interval_clear"],
                "rows_returned_to_the_base_by_fold": zeroed,
                "tail_clauses": {f: A.tail_clauses(pan[f], kept[f])
                                 ["registered_clause_y_over_3600s"] for f in A.FOLDS}}

    def wide_mask(f, lfpg_only=False):
        cu = cut_from(fold_use[f])
        p = pan[f]
        ap = p.airport.to_numpy().astype(str)
        v = p.obs_taxi.to_numpy("float64")
        c = np.array([cu.get(a, np.inf) for a in ap], dtype="float64")
        m = np.isfinite(v) & (v > c)
        return m & (ap == "LFPG") if lfpg_only else m

    for name, fn in (("none", lambda f: np.zeros(len(pan[f]), bool)),
                     ("lfpg_airport", lambda f: pan[f].airport.to_numpy().astype(str) == "LFPG"),
                     ("wide_everywhere", lambda f: wide_mask(f)),
                     ("wide_at_lfpg", lambda f: wide_mask(f, True))):
        rec = price(fn)
        out.setdefault("rules", {})[name] = rec
        ci = rec["board_all_row"]["interval"]["ci95"]
        t = {f: round(rec["tail_clauses"][f]["rmse_gain_s"], 3) for f in A.FOLDS}
        print(f"{name:16s} all-row {rec['board_all_row']['gain_s']:+8.4f} "
              f"[{ci[0]:+7.3f},{ci[1]:+7.3f}] clean {rec['board_clean']['gain_s']:+8.4f} "
              f"zeroed {rec['rows_returned_to_the_base_by_fold']} tail {t}", flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # --- what the per-row rule holds on the ranking side --------------------
    served = dep.usable.to_numpy(bool) & ~dep.noac.to_numpy(bool)
    ap = dep.airport.to_numpy().astype(str)
    v = dep.obs_taxi.to_numpy("float64")
    c = np.array([cuts_rank.get(a, np.inf) for a in ap], dtype="float64")
    wide = np.isfinite(v) & (v > c) & served
    out["ranking_side"] = {
        "wide_everywhere": {"rows": int(wide.sum()),
                            "by_airport": {str(a): int(((ap == a) & wide).sum())
                                           for a in sorted(set(ap[wide]))},
                            "by_month": {str(m): int((wide & (dep.rank_month.to_numpy() == m)).sum())
                                         for m in sorted(set(dep.rank_month.to_numpy()))}},
        "wide_at_lfpg": {"rows": int((wide & (ap == "LFPG")).sum())},
        "lfpg_airport": {"rows": int((served & (ap == "LFPG")).sum())}}
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print(json.dumps(out["ranked_airport_months"][:8], indent=1))
    print(json.dumps(out["ranking_side"], indent=1)[:700])
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
