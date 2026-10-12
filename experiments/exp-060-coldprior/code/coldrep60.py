"""exp-060-coldprior: the COLD arrival factor tables of one rotation (SPEC 2).

exp-054's `code/e54.py` is imported READ-ONLY after a SHA-256 pin and supplies every mechanic
unchanged: the arrival key construction, the auxiliary target (taxi-in, clipped to [0, 3600] s),
the matched donor, the block-ridge fit (lambda 5 for date and hour, 50 for stand, runway and
operator; 60 sweeps; seed 54), the support rule (>= 100 arrivals on >= 20 distinct dates) and the
stand, runway and operator export.  Two things differ from exp-054's warm table:

  * the fitting months are A = the nine months that are neither V nor S nor I of the rotation, so
    no arrival of the correction month, the inner month or the scored month enters a cold table;
  * there is NO auxiliary-accuracy step: no arrival outcome of S, I or V is read anywhere.

Five arms are fitted per rotation (true, const, gmean, donor, shuffle) -- five trainable fits,
one at a time, fifteen in all.  The ten shams are permutations of the cold `true` values among the
supported keys of the same family and airport and are not fits.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-060-coldprior/code/coldrep60.py jul
    # mechanics only (synthetic arrival target, scratch output, no outcome fitted):
    ... coldrep60.py jul --dry-run --out-dir <scratch>
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
import e60 as E                                                       # noqa: E402

#: exp-054's registered mechanics, pinned by bytes
E54_PINS = {"code/e54.py": "7e4523841be96e687b03915f4ca5d71cc12cde193bd3d2e4dd604d22de96df94",
            "code/represent.py": "36aae29d62d9ffb8b3c35a79bc5903c63ce280179157b25a29c5a19950534afa"}
for _rel, _want in E54_PINS.items():
    _got = E.sha_file(E.E54 / _rel)
    if _got != _want:
        raise SystemExit(f"exp-054 {_rel} is {_got[:16]}, not the pinned {_want[:16]}")
sys.path.insert(0, str(E.E54 / "code"))
import e54 as R                                                       # noqa: E402
sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

#: every month whose arrival outcome this module loads, for the isolation proof
OUTCOME_MONTHS: list = []


def exported(fit: dict, block: str, sup: pd.DataFrame) -> pd.Series:
    """exp-054 `represent.exported`: the block's effects on supported key values, 0.0 elsewhere."""
    u, e = fit["effects"][block]
    s = pd.Series(e, index=pd.Index(u, dtype=object))
    ok = sup.index[sup["supported"]]
    return s.reindex(ok).fillna(0.0)


def sham_checks(tab: pd.DataFrame) -> dict:
    """Proof that every sham preserves each family-airport value multiset exactly."""
    apt = tab["value"].astype(str).str.split("|").str[0].to_numpy()
    fam = tab["key"].astype(str).to_numpy()
    grp = pd.Series(np.arange(len(tab))).groupby([fam, apt]).indices
    out = {"groups": len(grp), "seeds": {}}
    for s in E.SHAM_SEEDS:
        v = E.sham_table(tab, s).to_numpy("float64")
        ok, moved = True, 0
        for key in sorted(grp):
            idx = np.sort(grp[key])
            a = np.sort(tab["true"].to_numpy("float64")[idx])
            b = np.sort(v[idx])
            ok &= np.array_equal(a, b)
            moved += int((v[idx] != tab["true"].to_numpy("float64")[idx]).sum())
        out["seeds"][f"sham{s:02d}"] = {"multiset_preserved_in_every_group": bool(ok),
                                        "values_moved": moved,
                                        "vector_sha256": E.sha_arr(v)}
    out["all_multisets_preserved"] = all(x["multiset_preserved_in_every_group"]
                                         for x in out["seeds"].values())
    return out


def run(r: str, out_dir: Path, dry: bool, overwrite: bool) -> dict:
    t0 = time.time()
    V, S, I, _ = E.ROT[r]
    A = E.cold_months(r)
    tab_p = out_dir / f"work/cold-{r}.parquet"
    rec_p = out_dir / (f"analysis/dry-coldrep-{r}.json" if dry else f"analysis/cold-{r}.json")
    NO.refuse_existing([tab_p, rec_p], overwrite)
    launch = E.hold_load(f"cold tables {r}")

    keys = pd.concat([R.arrival_keys(m) for m in A], ignore_index=True)
    if dry:
        y = np.random.default_rng(R.SEED).normal(500.0, 250.0, len(keys)).clip(*R.Y_CLIP)
        missing = 0
    else:
        parts = []
        for m in A:
            OUTCOME_MONTHS.append(m)
            parts.append(R.arrival_outcomes(m, keys.loc[keys.month == m, "MVT_ID_mvt"].to_numpy("float64")))
        y = np.concatenate(parts)
        missing = int(np.isnan(y).sum())
        keep = np.isfinite(y)
        keys, y = keys[keep].reset_index(drop=True), y[keep]

    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r, "dry_run": dry,
           "excluded_months": {"V": V, "S": S, "I": I},
           "fitting_months": A, "launch": launch, "code_sha256": E.code_shas(),
           "exp054_pins": E54_PINS,
           "month_files_sha256": {m: E.sha_file(R.D.month_file(m)) for m in A},
           "fitting_arrivals": int(len(keys)),
           "arrivals_without_outcome_dropped": missing,
           "arrival_dates": int(pd.Series(keys["date"]).nunique()),
           "rows_per_month": {m: int((keys["month"] == m).sum()) for m in A},
           "target": ("taxi-in B_arr - T_arr, clipped to [0, 3600] s" if not dry
                      else "SYNTHETIC N(500, 250) clipped (dry run)"),
           "target_mean_s": float(y.mean()), "target_sd_s": float(y.std()),
           "lambda": R.LAMBDA, "sweeps": R.SWEEPS, "blocks": R.BLOCKS_ARR,
           "support_rule": {"min_rows": R.MIN_ROWS, "min_days": R.MIN_DAYS},
           "start_utc": E.utc(), "start_unix": time.time()}

    design = R.arrival_design(keys)
    rec["levels"] = {b: int(len(design[b][1])) for b in R.BLOCKS_ARR}
    sup = {k: R.support_table(keys, k) for k in R.KEYS}
    rec["supported_values"] = {k: int(sup[k]["supported"].sum()) for k in R.KEYS}
    rec["supported_values_sha256"] = {k: E.sha_text(sorted(sup[k].index[sup[k]["supported"]].tolist()))
                                      for k in R.KEYS}
    donor, dstats = R.donor_index(keys)
    rec["donor"] = dstats
    rec["fits"] = {}
    rep = {}
    for arm in R.ARMS_REP:
        yt, cstats = R.control_target(arm, y, keys, donor)
        lo = E.hold_load(f"{r} cold factor fit {arm}")
        t1 = time.time()
        fit = R.fit_factors(yt, design)                                # one trainable fit
        rep[arm] = {k: exported(fit, k, sup[k]) for k in R.KEYS}
        rec["fits"][arm] = {**lo, "seconds": round(time.time() - t1, 2), "trainable_fits": 1,
                            "control": cstats, "target_mean_s": float(yt.mean()),
                            "target_sd_s": float(yt.std()),
                            "max_change_last_sweep_s": fit["max_change_last_sweep_s"],
                            "rmse_in_sample": fit["rmse_in_sample"],
                            "exported_sd_s": {k: float(rep[arm][k].std()) for k in R.KEYS},
                            "exported_nonzero": {k: int((rep[arm][k] != 0).sum()) for k in R.KEYS},
                            "exported_rows": {k: int(len(rep[arm][k])) for k in R.KEYS}}
        print(f"  [{r}] {arm:8s} fit {rec['fits'][arm]['seconds']:.1f}s, last-sweep change "
              f"{fit['max_change_last_sweep_s']:.2e} s, factor sd "
              + ", ".join(f"{k} {rec['fits'][arm]['exported_sd_s'][k]:.1f}" for k in R.KEYS),
              flush=True)
    const_max = {k: float(np.max(np.abs(rep["const"][k].to_numpy()))) for k in R.KEYS}
    rec["const_is_exactly_zero"] = {"max_abs": const_max, "ok": all(v == 0.0 for v in const_max.values())}
    if not rec["const_is_exactly_zero"]["ok"]:
        raise SystemExit(f"{r}: the constant-target cold table is not exactly zero")
    same_rows = len({tuple(sorted(rep[a][k].index.tolist())) == tuple(sorted(rep["true"][k].index.tolist()))
                     for a in R.ARMS_REP for k in R.KEYS}) == 1
    rec["every_arm_same_rows_and_support"] = bool(same_rows)

    rows = []
    for k in R.KEYS:
        t = sup[k][sup[k]["supported"]]
        for v in t.index:
            rows.append({"key": k, "value": v, "rows": int(t.at[v, "rows"]), "days": int(t.at[v, "days"]),
                         **{arm: float(rep[arm][k].get(v, 0.0)) for arm in R.ARMS_REP}})
    frz = pd.DataFrame(rows)
    rec["shams"] = sham_checks(frz)
    if not rec["shams"]["all_multisets_preserved"]:
        raise SystemExit(f"{r}: a sham does not preserve its family-airport multiset")

    NO.refuse_existing([tab_p, rec_p], overwrite)
    tab_p.parent.mkdir(parents=True, exist_ok=True)
    rec_p.parent.mkdir(parents=True, exist_ok=True)
    frz.to_parquet(tab_p, index=False)
    rec["table"] = {"path": str(tab_p) if dry else str(tab_p.relative_to(E.ROOT)),
                    "sha256": E.sha_file(tab_p), "rows": int(len(frz))}
    rec["arrival_outcome_months_loaded"] = list(OUTCOME_MONTHS)
    rec["no_S_I_V_arrival_outcome_loaded"] = not ({V, S, I} & set(OUTCOME_MONTHS))
    if not rec["no_S_I_V_arrival_outcome_loaded"]:
        raise SystemExit(f"{r}: an arrival outcome of V, S or I was loaded")
    rec["finish_utc"] = E.utc()
    rec["finish_unix"] = time.time()
    rec["seconds"] = round(time.time() - t0, 1)
    rec_p.write_text(json.dumps(rec, indent=1, sort_keys=True, default=float))
    print(f"written {rec_p.name} and {tab_p.name} {rec['table']['sha256'][:16]} "
          f"({rec['seconds']:.0f}s)", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rotations", nargs="+", choices=E.ORDER)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir) if a.out_dir else E.EXP
    if a.dry_run and out_dir == E.EXP:
        raise SystemExit("a dry run writes to --out-dir only")
    (out_dir / "work").mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis").mkdir(parents=True, exist_ok=True)
    paths = []
    for r in a.rotations:
        paths += [out_dir / f"work/cold-{r}.parquet",
                  out_dir / (f"analysis/dry-coldrep-{r}.json" if a.dry_run else f"analysis/cold-{r}.json")]
    NO.refuse_existing(paths, a.overwrite)
    print("guard: no existing artifact at " + ", ".join(str(p) for p in paths), flush=True)
    for r in a.rotations:
        OUTCOME_MONTHS.clear()
        run(r, out_dir, a.dry_run, a.overwrite)


if __name__ == "__main__":
    main()
