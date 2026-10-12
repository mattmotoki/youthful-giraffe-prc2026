"""exp-054-arrivalrepr: the field and support audit (SPEC section 1.2), run before any fit.

No departure label is read here: departure rows are loaded without either label column
(`e54.departure_frame`), and the panel is read by column without `y_true`.  No model is fitted.

1. Arrival field inventory, all twelve 2025 months: row counts, the non-null share of every
   column on ARR rows, the exact identity TAXITIME_SEC_mvt = BLOCK - MVT on ARR rows, and four
   candidate auxiliary outcomes with their coverage and quantiles (taxi-in, in-block delay,
   landing delay, NM landing offset).  Arrival outcomes are legal inputs, not validation labels;
   this inventory is descriptive and feeds no representation.
2. Key support per fold: for stand, runway and operator, the fitting-month arrival support
   table under the registered rule (MIN_ROWS arrivals over MIN_DAYS dates per airport x value),
   and the share of the held month's departure rows (all rows and P) whose key is supported,
   per airport.  Unsupported mass is the complement.
3. Donor feasibility per fold: the time-matched donor search on the fitting-month arrival KEYS
   and landing clocks only (no outcome): rows with a donor, fallbacks, the time gap.
4. Held-month departure row features: non-null share of each, and p_v22 >= 60 on every row.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-054-arrivalrepr/code/audit.py
    # the re-run at the registered code, under a new name (write-once):
    ... audit.py --out analysis/audit-r1.json
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

D, NO = E.D, E.NO
OUT_DEFAULT = "analysis/audit.json"


def q(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    return {str(p): float(np.percentile(x, p)) for p in (0.1, 1, 10, 50, 90, 99, 99.9)} if len(x) else {}


def inventory() -> dict:
    out = {}
    for m in D.MONTHS:
        raw = D.load_month(m)
        a = raw[raw["PHASE_mvt"].to_numpy() == "ARR"]
        B, T = a["BLOCK_TIME_UTC_mvt"], a["MVT_TIME_UTC_mvt"]
        tin = a["TAXITIME_SEC_mvt"].to_numpy("float64")
        alg = (B - T).dt.total_seconds().to_numpy("float64")
        cand = {
            "taxi_in = B - T (TAXITIME_SEC_mvt)": tin,
            "inblock_delay = B - SCHED": (B - a["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy("float64"),
            "landing_delay = T - SCHED": (T - a["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy("float64"),
            "nm_landing_offset = T - ARVT_3": (T - a["ARVT_3_flt"]).dt.total_seconds().to_numpy("float64")}
        out[m] = {
            "rows": int(len(raw)), "arrivals": int(len(a)),
            "departures": int((raw["PHASE_mvt"].to_numpy() == "DEP").sum()),
            "arr_nonnull_share": {c: round(float(a[c].notna().mean()), 5) for c in a.columns
                                  if c != "month"},
            "identity_taxi_in_equals_block_minus_mvt": {
                "rows_equal": int((tin == alg).sum()), "rows": int(len(a))},
            "candidates": {k: {"coverage": round(float(np.isfinite(v).mean()), 5),
                               "quantiles_s": q(v)} for k, v in cand.items()},
            "taxi_in_outside_clip": {"below_0": int((tin < E.Y_CLIP[0]).sum()),
                                     "above_3600": int((tin > E.Y_CLIP[1]).sum())}}
        print(f"  {m}: {len(a):,} arrivals, identity {out[m]['identity_taxi_in_equals_block_minus_mvt']}",
              flush=True)
        del raw, a
    return out


def support(fold: str, A: pd.DataFrame, pan: pd.DataFrame, pop: np.ndarray, dk: pd.DataFrame) -> dict:
    apt = dk["apt"].to_numpy(object)
    rec = {"fitting_arrivals": int(len(A)), "keys": {}}
    any_sup = np.zeros(len(dk), bool)
    for k in E.KEYS:
        t = E.support_table(A, k)
        sup = set(t.index[t["supported"]])
        v = dk[k].to_numpy(object)
        present = np.array([x is not None for x in v])
        is_sup = np.array([x in sup for x in v])
        any_sup |= is_sup
        by_apt = {}
        for a in sorted(set(apt.tolist())):
            m = apt == a
            by_apt[a] = {"rows": int(m.sum()),
                         "supported_share_all": round(float(is_sup[m].mean()), 4),
                         "supported_share_P": round(float(is_sup[m & pop].mean()), 4)
                         if (m & pop).any() else None}
        rec["keys"][k] = {
            "arrival_values": int(len(t)), "arrival_values_supported": int(t["supported"].sum()),
            "arrival_rows_on_supported_values": int(t.loc[t["supported"], "rows"].sum()),
            "departure_rows": int(len(v)), "departure_key_present_share": round(float(present.mean()), 4),
            "supported_share_all": round(float(is_sup.mean()), 4),
            "supported_share_P": round(float(is_sup[pop].mean()), 4),
            "unsupported_mass_all": round(float(1 - is_sup.mean()), 4),
            "unsupported_mass_P": round(float(1 - is_sup[pop].mean()), 4),
            "by_airport": by_apt}
    rec["any_key_supported_share_all"] = round(float(any_sup.mean()), 4)
    rec["any_key_supported_share_P"] = round(float(any_sup[pop].mean()), 4)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT, help="output path, relative to the experiment")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = E.EXP / args.out
    NO.refuse_existing([OUT], args.overwrite)
    print(f"guard: no existing artifact at {OUT.relative_to(E.ROOT)}", flush=True)
    t0 = time.time()
    rec = {"what": "exp-054 field and support audit; no departure label read, no fit",
           "code_sha256": {"audit.py": E.sha_file(Path(__file__)),
                           "e54.py": E.sha_file(E.HERE / "e54.py")},
           "pinned": E.check_pinned(),
           "support_rule": {"min_rows": E.MIN_ROWS, "min_days": E.MIN_DAYS},
           "y_clip": list(E.Y_CLIP), "started_utc": E.utc()}
    print("1. arrival field inventory", flush=True)
    rec["inventory"] = inventory()
    rec["folds"] = {}
    cache = {m: E.arrival_keys(m) for m in D.MONTHS}
    for fold in E.FOLDS:
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
        A = pd.concat([cache[m] for m in E.fitting_months(fold)], ignore_index=True)
        fr = {"month": month, "fitting_months": E.fitting_months(fold),
              "panel_rows": int(len(pan)), "P_rows": int(pop.sum())}
        fr["support"] = support(fold, A, pan, pop, dk)
        dn, st = E.donor_index(A)
        by_apt = {}
        apt_a = A["apt"].to_numpy(object)
        for a in sorted(set(apt_a.tolist())):
            m = apt_a == a
            by_apt[a] = {"arrivals": int(m.sum()), "without_donor": int((dn[m] < 0).sum())}
        st["by_airport"] = by_apt
        fr["donor"] = st
        # the rejected rule (SPEC rejected reading e): stand, runway AND operator must differ
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
        fr["row_feature_nonnull_share_all"] = {c: round(float(Fi[c].notna().mean()), 4)
                                               for c in E.ROW_FEATURES}
        fr["row_feature_nonnull_share_P"] = {c: round(float(Fi.loc[pop, c].notna().mean()), 4)
                                             for c in E.ROW_FEATURES}
        fr["no_nm_rows"] = int(E.no_nm_mask(frame, pan.MVT_ID_mvt.to_numpy("float64")).sum())
        fr["p_v22_below_60"] = int((pan.p_v22.to_numpy("float64") < E.MIN_PRED).sum())
        fr["days"] = int(pan.day.nunique())
        fr["block_days"] = {int(b): int(pd.Series(pan.day)[E.blocks(pan.day.to_numpy()) == b].nunique())
                            for b in range(E.K)}
        rec["folds"][fold] = fr
        s = fr["support"]
        print(f"2-4. {fold}: P {fr['P_rows']:,}; supported share of P: "
              + ", ".join(f"{k} {s['keys'][k]['supported_share_P']:.3f}" for k in E.KEYS)
              + f"; any {s['any_key_supported_share_P']:.3f}; donors {st['with_donor']:,} / "
              f"{st['arrivals']:,}, gap {st['gap_s_quantiles']}", flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"written {OUT.relative_to(E.ROOT)} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
