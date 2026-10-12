"""exp-054-arrivalrepr: the frozen arrival representations of one or more folds (SPEC 1.4, 1.8).

Per fold, in this order:

  1. load the arrival KEYS of the eleven fitting months (the held month is never loaded), and
     their auxiliary target, taxi-in, clipped to [0, 3600] s (`e54.arrival_outcomes`, the only
     reader of the arrival in-block clock); arrivals with no outcome are dropped;
  2. find each arrival's time-matched donor from keys and landing clocks only;
  3. for each representation arm (true, const, gmean, donor, shuffle): form the arm's auxiliary
     target from the fitting months only, fit the block-ridge factor model once (one trainable
     fit), and export the stand, runway and operator factors, zero where the key value fails the
     registered support rule;
  4. FREEZE: write `work/rep-<fold>.parquet` and record its sha256 and the unix time;
  5. only after the freeze, read the held month's arrival outcomes and score each frozen
     representation on them (auxiliary-task accuracy: a report, never an input or a choice).

No departure row, departure label or v22 vector is read here.  Outputs are write-once.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-054-arrivalrepr/code/represent.py july_loo december january
    # mechanics only (synthetic arrival target, scratch outputs, no outcome fitted):
    ... represent.py july_loo --dry-run --out-dir <scratch>
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


def exported(fit: dict, block: str, sup: pd.DataFrame) -> pd.Series:
    """The block's effects on supported key values; 0.0 on every other value."""
    u, e = fit["effects"][block]
    s = pd.Series(e, index=pd.Index(u, dtype=object))
    ok = sup.index[sup["supported"]]
    return s.reindex(ok).fillna(0.0)


def dense_prior(fit: dict, A_apt: np.ndarray) -> dict:
    """The true arm's dense part, for the held-month accuracy report: mu, the mean date effect
    of each airport over its fitting dates, and the airport x hour effects."""
    ud, ed = fit["effects"]["date"]
    apt_of = np.array([x.split("|")[0] for x in ud], dtype=object)
    date_mean = pd.Series(ed).groupby(apt_of).mean().to_dict()
    uh, eh = fit["effects"]["hour"]
    return {"mu": fit["mu"], "date_mean": date_mean,
            "hour": dict(zip(uh.tolist(), eh.tolist()))}


def accuracy(fold: str, rep: dict, prior: dict, dry: bool) -> dict:
    """Score each frozen representation on the held month's arrivals (read after the freeze).
    gain = mean (y - prior)^2 - mean (y - prior - sparse)^2, sparse = f_stand + f_rwy + f_op."""
    month = E.fold_month(fold)
    H = E.arrival_keys(month)
    if dry:
        y = np.random.default_rng(E.SEED + 1).normal(500.0, 250.0, len(H))
    else:
        y = E.arrival_outcomes(month, H.MVT_ID_mvt.to_numpy("float64"))
        keep = np.isfinite(y)
        H, y = H[keep].reset_index(drop=True), y[keep]
    apt = H["apt"].to_numpy(object)
    hk = E._join(apt, H["hour"].astype(str).to_numpy(object))
    base = (prior["mu"] + np.array([prior["date_mean"].get(a, 0.0) for a in apt])
            + np.array([prior["hour"].get(k, 0.0) for k in hk]))
    e0 = y - base
    days = H["date"].to_numpy(object)
    rng = np.random.default_rng(E.SEED)
    out = {"month": month, "arrivals": int(len(H)), "rmse_prior": float(np.sqrt(np.mean(e0 ** 2))),
           "arms": {}}
    for arm, tabs in rep.items():
        parts = {k: pd.Series(tabs[k]).reindex(H[k].to_numpy(object)).fillna(0.0).to_numpy("float64")
                 for k in E.KEYS}
        sp = sum(parts.values())
        v = e0 ** 2 - (e0 - sp) ** 2
        r = {"mse_gain_s2": float(v.mean()), "ci95_day": NR_day_ci(v, days, rng),
             "corr": float(np.corrcoef(e0, sp)[0, 1]) if np.std(sp) > 0 else None,
             "rows_nonzero": int((sp != 0).sum()), "by_key": {}}
        for k in E.KEYS:
            vk = e0 ** 2 - (e0 - parts[k]) ** 2
            r["by_key"][k] = {"mse_gain_s2": float(vk.mean()),
                              "rows_nonzero": int((parts[k] != 0).sum())}
        out["arms"][arm] = r
    return out


def NR_day_ci(v: np.ndarray, days: np.ndarray, rng) -> list[float]:
    return E.NR.day_ci(v, days, rng)


def run_fold(fold: str, out_dir: Path, dry: bool, overwrite: bool) -> dict:
    t0 = time.time()
    held = E.fold_month(fold)
    months = E.fitting_months(fold)
    if held in months:
        raise SystemExit(f"{fold}: the held month {held} is among the fitting months")
    A = pd.concat([E.arrival_keys(m) for m in months], ignore_index=True)
    if dry:
        y = np.random.default_rng(E.SEED).normal(500.0, 250.0, len(A)).clip(*E.Y_CLIP)
        missing = 0
    else:
        y = np.concatenate([E.arrival_outcomes(m, A.loc[A.month == m, "MVT_ID_mvt"].to_numpy("float64"))
                            for m in months])
        missing = int(np.isnan(y).sum())
        keep = np.isfinite(y)
        A, y = A[keep].reset_index(drop=True), y[keep]
    rec = {"fold": fold, "held_month": held, "fitting_months": months, "dry_run": dry,
           "month_files_sha256": {m: E.sha_file(D.month_file(m)) for m in months},
           "fitting_arrivals": int(len(A)), "arrivals_without_outcome_dropped": missing,
           "target": "taxi-in B_arr - T_arr, clipped to [0, 3600] s" if not dry
                     else "SYNTHETIC N(500, 250) clipped (dry run)",
           "target_mean_s": float(y.mean()), "target_sd_s": float(y.std()),
           "lambda": E.LAMBDA, "sweeps": E.SWEEPS, "blocks": E.BLOCKS_ARR,
           "support_rule": {"min_rows": E.MIN_ROWS, "min_days": E.MIN_DAYS}}
    design = E.arrival_design(A)
    rec["levels"] = {b: int(len(design[b][1])) for b in E.BLOCKS_ARR}
    sup = {k: E.support_table(A, k) for k in E.KEYS}
    rec["supported_values"] = {k: int(sup[k]["supported"].sum()) for k in E.KEYS}
    donor, dstats = E.donor_index(A)
    rec["donor"] = dstats
    rec["fits"] = {}
    rep, prior = {}, None
    for arm in E.ARMS_REP:
        yt, cstats = E.control_target(arm, y, A, donor)
        lo = E.hold_load(f"{fold} representation {arm}")
        t1 = time.time()
        fit = E.fit_factors(yt, design)
        rep[arm] = {k: exported(fit, k, sup[k]) for k in E.KEYS}
        if arm == "true":
            prior = dense_prior(fit, A["apt"].to_numpy(object))
        rec["fits"][arm] = {**lo, "seconds": round(time.time() - t1, 2), "trainable_fits": 1,
                            "control": cstats, "target_mean_s": float(yt.mean()),
                            "target_sd_s": float(yt.std()),
                            "max_change_last_sweep_s": fit["max_change_last_sweep_s"],
                            "rmse_in_sample": fit["rmse_in_sample"],
                            "exported_sd_s": {k: float(rep[arm][k].std()) for k in E.KEYS},
                            "exported_nonzero": {k: int((rep[arm][k] != 0).sum()) for k in E.KEYS}}
        print(f"  [{fold}] {arm:8s} fit {rec['fits'][arm]['seconds']:.1f}s, last-sweep change "
              f"{fit['max_change_last_sweep_s']:.2e} s, factor sd "
              + ", ".join(f"{k} {rec['fits'][arm]['exported_sd_s'][k]:.1f}" for k in E.KEYS),
              flush=True)
    if not all(float(np.max(np.abs(rep["const"][k].to_numpy()))) == 0.0 for k in E.KEYS):
        raise SystemExit(f"{fold}: the constant-target representation is not exactly zero")
    # ---- freeze -------------------------------------------------------------------------
    rows = []
    for k in E.KEYS:
        t = sup[k][sup[k]["supported"]]
        for v in t.index:
            rows.append({"key": k, "value": v, "rows": int(t.at[v, "rows"]),
                         "days": int(t.at[v, "days"]),
                         **{arm: float(rep[arm][k].get(v, 0.0)) for arm in E.ARMS_REP}})
    frz = pd.DataFrame(rows)
    p = out_dir / f"work/rep-{fold}.parquet"
    NO.refuse_existing([p], overwrite)
    frz.to_parquet(p, index=False)
    rec["frozen"] = {"path": str(p.relative_to(E.ROOT)) if p.is_relative_to(E.ROOT) else str(p),
                     "sha256": E.sha_file(p), "rows": int(len(frz)), "unix_time": time.time(),
                     "utc": E.utc()}
    rec["dense_prior_true"] = {"mu": prior["mu"], "airports": sorted(prior["date_mean"])}
    # ---- after the freeze: auxiliary-task accuracy on the held month's arrivals ------------
    rec["held_month_arrival_outcomes_first_read_unix"] = time.time()
    rec["aux_accuracy"] = accuracy(fold, rep, prior, dry)
    rec["seconds"] = round(time.time() - t0, 1)
    for arm, r in rec["aux_accuracy"]["arms"].items():
        print(f"  [{fold}] aux accuracy {arm:8s} gain {r['mse_gain_s2']:+9.1f} s^2 "
              f"{[round(x, 1) for x in r['ci95_day']]} corr {r['corr']}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folds", nargs="+", choices=E.FOLDS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else E.EXP
    if args.dry_run and out_dir == E.EXP:
        raise SystemExit("a dry run writes to --out-dir only")
    (out_dir / "work").mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis").mkdir(parents=True, exist_ok=True)
    paths = []
    for f in args.folds:
        paths += [out_dir / f"work/rep-{f}.parquet", out_dir / f"analysis/rep-{f}.json"]
    NO.refuse_existing(paths, args.overwrite)
    print("guard: no existing artifact at " + ", ".join(str(p) for p in paths), flush=True)
    code = {n: E.sha_file(E.HERE / n) for n in ("e54.py", "represent.py")}
    for f in args.folds:
        rec = run_fold(f, out_dir, args.dry_run, args.overwrite)
        rec["code_sha256"] = code
        j = out_dir / f"analysis/rep-{f}.json"
        NO.refuse_existing([j], args.overwrite)
        j.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
        print(f"written {j} ({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
