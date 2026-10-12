"""exp-054-arrivalrepr: the clean field and support audit (SPEC amendment A1; Sol round 1, finding 1).

The registered audit (`audit.py` -> `analysis/audit.json`, `analysis/audit-r1.json`) loaded EVERY
column of all twelve 2025 months before the registration: it read the held months' arrival
outcomes, and a frame held their departure labels.  This audit repeats it without either read.

1. Arrival field inventory on the NINE months that are fitting months in every fold (2025-02..06,
   2025-08..11).  The three held months (2025-01, 2025-07, 2025-12) are not opened by this part.
   Rows are counted from `PHASE_mvt` alone; every other column is read on ARR rows only
   (`e54r.arrival_rows`), so no frame holds a departure label.  Per month: the quantities of
   `audit.py` (its `q()`).  Pooled over the nine months: the evidence that SPEC 1.2 cites for the
   choice of taxi-in, and SPEC 1.3 for the clip and the identity.
2-4. Per fold: the key support, donor feasibility and row-feature census of `audit.py`, repeated
   unchanged except that the arrival keys are read for the fold's fitting months only (`audit.py`
   read the keys of all twelve months; keys are not outcomes).  The held month is read only as
   `e54.departure_frame` (no label column) and the panel without `y_true`, as before.
5. Comparison with `audit-r1.json`: only its nine-month inventory entries and its fold entries are
   read, never a held month's inventory entry.  Each must equal this audit's value.

No model is fitted.  No departure label and no held month's arrival outcome is read.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-054-arrivalrepr/code/audit2.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e54 as E                                                       # noqa: E402
import e54r as R                                                      # noqa: E402
import audit as AU                                                    # noqa: E402  (registered code, unchanged)

D, NO = E.D, E.NO
OUT_DEFAULT = "analysis/audit-r2.json"
HELD = sorted(E.fold_month(f) for f in E.FOLDS)
NINE = [m for m in D.MONTHS if all(m in E.fitting_months(f) for f in E.FOLDS)]
assert HELD == ["2025-01", "2025-07", "2025-12"]
assert NINE == [f"2025-{i:02d}" for i in (2, 3, 4, 5, 6, 8, 9, 10, 11)]
TAXI = "taxi_in = B - T (TAXITIME_SEC_mvt)"


def jround(x):
    """The value as the JSON record stores it."""
    return json.loads(json.dumps(x, sort_keys=True, default=float))


def inventory_month(m: str) -> tuple[dict, dict]:
    """`audit.inventory()` for one month, from a PHASE_mvt count and an ARR-only read."""
    assert m in NINE
    cnt = R.phase_counts(m)
    a = R.arrival_rows(m)
    B, T = a["BLOCK_TIME_UTC_mvt"], a["MVT_TIME_UTC_mvt"]
    tin = a["TAXITIME_SEC_mvt"].to_numpy("float64")
    alg = (B - T).dt.total_seconds().to_numpy("float64")
    cand = {
        TAXI: tin,
        "inblock_delay = B - SCHED": (B - a["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy("float64"),
        "landing_delay = T - SCHED": (T - a["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy("float64"),
        "nm_landing_offset = T - ARVT_3": (T - a["ARVT_3_flt"]).dt.total_seconds().to_numpy("float64")}
    if cnt["arrivals"] != len(a):
        raise SystemExit(f"{m}: {len(a)} ARR rows read, {cnt['arrivals']} counted")
    out = {
        "rows": cnt["rows"], "arrivals": int(len(a)), "departures": cnt["departures"],
        "arr_nonnull_share": {c: round(float(a[c].notna().mean()), 5) for c in a.columns if c != "month"},
        "identity_taxi_in_equals_block_minus_mvt": {"rows_equal": int((tin == alg).sum()), "rows": int(len(a))},
        "candidates": {k: {"coverage": round(float(np.isfinite(v).mean()), 5), "quantiles_s": AU.q(v)}
                       for k, v in cand.items()},
        "taxi_in_outside_clip": {"below_0": int((tin < E.Y_CLIP[0]).sum()),
                                 "above_3600": int((tin > E.Y_CLIP[1]).sum())}}
    extra = {"identity_rows_neither_value": int((np.isnan(tin) & np.isnan(alg)).sum()),
             "identity_rows_one_value": int((np.isnan(tin) ^ np.isnan(alg)).sum())}
    return out, {"cand": cand, **extra}


def choice_evidence(inv: dict, pooled: dict, extra: dict) -> dict:
    """The nine-month figures behind the registered choices (SPEC 1.2, 1.3)."""
    cands = {}
    for k, v in pooled.items():
        qq = AU.q(v)
        cands[k] = {"coverage": round(float(np.isfinite(v).mean()), 5), "quantiles_s": qq,
                    "p1_to_p99_spread_s": qq["99"] - qq["1"] if qq else None,
                    "missing_rows": int((~np.isfinite(v)).sum())}
    tin = pooled[TAXI]
    return {
        "months": NINE, "arrivals": int(sum(inv[m]["arrivals"] for m in NINE)),
        "candidates_pooled": cands,
        "sched_on_arrivals": {"median_B_minus_SCHED_s": cands["inblock_delay = B - SCHED"]["quantiles_s"]["50"],
                              "median_T_minus_SCHED_s": cands["landing_delay = T - SCHED"]["quantiles_s"]["50"]},
        "identity": {"rows": int(sum(inv[m]["identity_taxi_in_equals_block_minus_mvt"]["rows"] for m in NINE)),
                     "rows_equal": int(sum(inv[m]["identity_taxi_in_equals_block_minus_mvt"]["rows_equal"] for m in NINE)),
                     "rows_neither_value": {m: extra[m]["identity_rows_neither_value"] for m in NINE
                                            if extra[m]["identity_rows_neither_value"]},
                     "rows_one_value_only": int(sum(extra[m]["identity_rows_one_value"] for m in NINE))},
        "clip": {"below_0_per_month": [min(inv[m]["taxi_in_outside_clip"]["below_0"] for m in NINE),
                                       max(inv[m]["taxi_in_outside_clip"]["below_0"] for m in NINE)],
                 "above_3600_per_month": [min(inv[m]["taxi_in_outside_clip"]["above_3600"] for m in NINE),
                                          max(inv[m]["taxi_in_outside_clip"]["above_3600"] for m in NINE)],
                 "share_outside": float(((tin < E.Y_CLIP[0]) | (tin > E.Y_CLIP[1])).sum() / np.isfinite(tin).sum())}}


def fold_part(fold: str) -> dict:
    """`audit.py` main(), the per-fold part, with the arrival keys of the fitting months only."""
    month = E.fold_month(fold)
    pan = E.panel(fold, ["MVT_ID_mvt", "regime", "guard_held", "p_v22", "day"])
    pop = E.population(pan, fold)
    frame = E.departure_frame(month)
    F = E.row_features(frame)
    dep = frame[frame["PHASE_mvt"].to_numpy() == "DEP"].reset_index(drop=True)
    dk = E.departure_keys(dep).set_index("MVT_ID_mvt").reindex(
        pan.MVT_ID_mvt.to_numpy("float64")).reset_index()
    if dk["apt"].isna().any() or len(F) != len(pan):
        raise SystemExit(f"{fold}: departure rows and panel rows do not match one to one")
    fm = E.fitting_months(fold)
    assert month not in fm
    A = pd.concat([E.arrival_keys(m) for m in fm], ignore_index=True)
    fr = {"month": month, "fitting_months": fm, "panel_rows": int(len(pan)), "P_rows": int(pop.sum())}
    fr["support"] = AU.support(fold, A, pan, pop, dk)
    dn, st = E.donor_index(A)
    by_apt = {}
    apt_a = A["apt"].to_numpy(object)
    for a in sorted(set(apt_a.tolist())):
        m = apt_a == a
        by_apt[a] = {"arrivals": int(m.sum()), "without_donor": int((dn[m] < 0).sum())}
    st["by_airport"] = by_apt
    fr["donor"] = st
    dn3, st3 = E.donor_index(A, differ=("stand", "rwy", "op"))
    g3 = pd.Series(np.abs(A["t"].to_numpy("float64")[np.maximum(dn3, 0)]
                          - A["t"].to_numpy("float64"))).where(dn3 >= 0)
    st3["by_airport"] = {a: {"arrivals": int((apt_a == a).sum()),
                             "without_donor": int((dn3[apt_a == a] < 0).sum()),
                             "gap_s_p50": float(g3[apt_a == a].median()),
                             "gap_s_p90": float(g3[apt_a == a].quantile(0.9))}
                         for a in sorted(set(apt_a.tolist()))}
    fr["donor_all_keys_differ"] = st3
    Fi = F.set_index("MVT_ID_mvt").reindex(pan.MVT_ID_mvt.to_numpy("float64"))
    fr["row_feature_nonnull_share_all"] = {c: round(float(Fi[c].notna().mean()), 4) for c in E.ROW_FEATURES}
    fr["row_feature_nonnull_share_P"] = {c: round(float(Fi.loc[pop, c].notna().mean()), 4) for c in E.ROW_FEATURES}
    fr["no_nm_rows"] = int(E.no_nm_mask(frame, pan.MVT_ID_mvt.to_numpy("float64")).sum())
    fr["p_v22_below_60"] = int((pan.p_v22.to_numpy("float64") < E.MIN_PRED).sum())
    fr["days"] = int(pan.day.nunique())
    fr["block_days"] = {int(b): int(pd.Series(pan.day)[E.blocks(pan.day.to_numpy()) == b].nunique())
                        for b in range(E.K)}
    return fr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT, help="output path, relative to the experiment")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = E.EXP / args.out
    NO.refuse_existing([OUT], args.overwrite)
    print(f"guard: no existing artifact at {OUT.relative_to(E.ROOT)}", flush=True)
    t0 = time.time()
    rec = {"what": "exp-054 clean field and support audit (SPEC amendment A1): no held month opened by the "
                   "inventory, no departure label in any frame, no fit",
           "code_sha256": {n: E.sha_file(E.HERE / n) for n in ("audit2.py", "audit.py", "e54.py", "e54r.py")},
           "pinned": E.check_pinned(), "support_rule": {"min_rows": E.MIN_ROWS, "min_days": E.MIN_DAYS},
           "y_clip": list(E.Y_CLIP), "held_months_not_opened_by_inventory": HELD, "started_utc": E.utc()}
    print(f"1. arrival field inventory, nine months {NINE[0]}..{NINE[-1]} (held {HELD} not opened)", flush=True)
    inv, extra, pooled = {}, {}, {}
    for m in NINE:
        inv[m], ex = inventory_month(m)
        for k, v in ex.pop("cand").items():
            pooled.setdefault(k, []).append(v)
        extra[m] = ex
        print(f"  {m}: {inv[m]['arrivals']:,} arrivals, identity {inv[m]['identity_taxi_in_equals_block_minus_mvt']}",
              flush=True)
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    rec["inventory"] = inv
    rec["choice_evidence"] = choice_evidence(inv, pooled, extra)
    for k, c in rec["choice_evidence"]["candidates_pooled"].items():
        qq = c["quantiles_s"]
        print(f"  pooled {k:34s} coverage {c['coverage']:.5f}  p1/p50/p99 {qq['1']:.0f} / {qq['50']:.0f} / "
              f"{qq['99']:.0f} s", flush=True)
    rec["folds"] = {}
    for fold in E.FOLDS:
        fr = fold_part(fold)
        rec["folds"][fold] = fr
        s = fr["support"]
        print(f"2-4. {fold}: P {fr['P_rows']:,}; supported share of P: "
              + ", ".join(f"{k} {s['keys'][k]['supported_share_P']:.3f}" for k in E.KEYS)
              + f"; any {s['any_key_supported_share_P']:.3f}; donors {fr['donor']['with_donor']:,} / "
              f"{fr['donor']['arrivals']:,}", flush=True)
    # ---- 5. equality with audit-r1, nine-month inventory entries and fold entries only -----------
    r1 = json.loads((E.EXP / "analysis/audit-r1.json").read_text())
    eq = {"inventory": {m: jround(inv[m]) == r1["inventory"][m] for m in NINE},
          "folds": {f: jround(rec["folds"][f]) == r1["folds"][f] for f in E.FOLDS}}
    eq["all_equal"] = all(eq["inventory"].values()) and all(eq["folds"].values())
    rec["equal_to_audit_r1"] = eq
    rec["audit_r1_sha256"] = E.sha_file(E.EXP / "analysis/audit-r1.json")
    print(f"5. equal to audit-r1: {eq}", flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"written {OUT.relative_to(E.ROOT)} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
