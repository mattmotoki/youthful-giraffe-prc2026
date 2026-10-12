"""exp-057 originfit: the zero-fit capacity diagnostic (SPEC 6.4).  No model is fitted.

Question: can zero-weight rows or duplicated rows be a capacity-neutral control for this exact
learner (`chainfold.fit_arm` -> `arms.arm_lgb` -> LightGBM 4.7.0 `LGBMRegressor`, predeclared
parameters, 255 leaves, seed 46)?

Evidence, with no `train`, no `fit` and no booster:
  (a) the code path, located in the imported source;
  (b) the installed LightGBM's own parameter documentation and the parameters that
      `LGBMRegressor._process_params("fit")` hands to `Dataset` (the call `LGBMRegressor.fit`
      makes before it trains);
  (c) `Dataset.construct()` (binning only) on the exact member-1 L0 June-carrier fitting matrix
      (captured from `fit_arm` by a stub in place of `arm_lgb`, as in `ofit_fit.py --plan`), for
      (i) member 1's 172,783 fold rows alone, (ii) plus June's 98,396 rows at weight 0,
      (ii') the same rows with no weight, (ii'') the same with every row in the bin sample, and
      (iii) plus the first 98,396 fold rows duplicated.  Label 0 (binning does not read it).
      For each: the bin count of every feature and a hash of the fold rows' binned values (the
      Dataset's text dump, written to a temporary file and removed).

    .venv/bin/python experiments/exp-057-originfit/code/ofit_capacity.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ofit_fit as OF                                                 # noqa: E402

import lightgbm as lgb                                                # noqa: E402
import numpy as np                                                    # noqa: E402

OUT = OF.ANALYSIS / "capacity.json"
N_FOLD, N_JUNE = OF.PANEL_ROWS, 98_396


class _Capture(BaseException):
    def __init__(self, a, k):
        super().__init__("capture")
        self.a, self.k = a, k


def find(obj, needle: str) -> dict:
    """The file and line of `needle` in the source of `obj` (must occur exactly once)."""
    lines, start = inspect.getsourcelines(obj)
    hits = [i for i, t in enumerate(lines) if needle in t]
    if len(hits) != 1:
        raise SystemExit(f"{needle!r} occurs {len(hits)} times in {obj.__name__}")
    f = Path(inspect.getsourcefile(obj)).resolve().relative_to(OF.ROOT)
    return {"file": str(f), "line": start + hits[0], "text": lines[hits[0]].strip()}


def code_path() -> dict:
    A, CF = OF.A, OF.CF
    return {
        "extra_share_open_interval": find(CF.fit_arm, "if not 0.0 < extra_share < 1.0"),
        "wmap_is_per_airport": find(CF.fit_arm, "w = fit.airport.astype(str).map(wmap)"),
        "weights_normalised_to_mean_1": find(CF.fit_arm, "w = w / w.mean()"),
        "constant_weight_is_fatal": find(CF.fit_arm, "if float(w.max() - w.min()) < 1e-9:"),
        "arm_lgb_target_is_rbase": find(A.arm_lgb, 'rf = fit.rbase.to_numpy("float64")'),
        "crossfit_day_blocks": find(A.arm_lgb, "blk = _dayblocks(fit.day)"),
        "day_blocks_per_calendar_month": find(A._dayblocks, 'dd.groupby(dd.dt.strftime("%Y-%m"))'),
        "crossfit_models_get_the_weight": find(A.arm_lgb, "_lgb_fit(Xf[tr], rf[tr], seed, params, wb)"),
        "amplitude_takes_no_weight": find(A.arm_lgb, 'amp, n = _per_airport_amplitude(fitc, "d_raw")'),
        "amplitude_is_an_unweighted_mean": find(A._amplitude, "np.mean(r * d)"),
        "amplitude_reads_each_rows_rbase": find(A._per_cell_amplitude, "_amplitude(g.rbase.to_numpy"),
        "amplitude_row_count_floor": find(A._per_cell_amplitude, "if len(g) >= MIN_AIRPORT_ROWS"),
        "lgb_fit_passes_sample_weight": find(A._lgb_fit, "m.fit(X, y, sample_weight=np.asarray(w"),
    }


def docs() -> dict:
    d = lgb.LGBMModel.__init__.__doc__ or ""
    out = {}
    for key in ("min_child_weight :", "min_child_samples :", "subsample :", "subsample_freq :"):
        i = d.find(key)
        out[key.rstrip(" :")] = " ".join(d[i:d.find("\n", d.find("\n", i) + 1)].split())
    d2 = lgb.Dataset.__init__.__doc__ or ""
    i = d2.find("weight :")
    out["Dataset.weight"] = " ".join(d2[i:d2.find("\n", d2.find("\n", i) + 1)].split())
    return out


def learner_params() -> dict:
    p = dict(OF.A.LGB)
    p.update({"num_leaves": OF.C.LEAVES})
    p["seed"] = OF.A.SEED
    m = lgb.LGBMRegressor(**{k: v for k, v in p.items() if k != "objective"}, objective=p["objective"])
    return m._process_params(stage="fit")


def fitting_matrix() -> tuple[np.ndarray, dict]:
    shared: dict = {}
    ctx = OF.context(1, shared)
    extra, rows = OF.carrier.fitting_extra(ctx, 1, "C", 0, None)

    def stub(*a, **k):
        raise _Capture(a, k)
    prov: dict = {}
    try:
        OF.call_fit_arm(ctx, 1, extra, stub, prov)
        raise SystemExit("the stub was not reached")
    except _Capture as c:
        fit, _sco, cols = c.a
    OF.restored()
    X = np.ascontiguousarray(fit[list(cols)].to_numpy("float32"))
    info = {"rows": OF.jround(rows), "provenance": OF.jround(prov[OF.F]), "n_features": len(cols),
            "fit_rows": int(len(fit)),
            "fold_rows_first": bool(prov[OF.F]["panel_rows"] == N_FOLD and len(fit) == N_FOLD + N_JUNE)}
    if not info["fold_rows_first"]:
        raise SystemExit(f"the captured frame is not {N_FOLD:,} fold rows then {N_JUNE:,} June rows")
    return X, info


def binned(X: np.ndarray, w, params: dict, n_head: int) -> dict:
    t0 = time.time()
    ds = lgb.Dataset(X, label=np.zeros(len(X)), weight=w, params=params).construct()
    nb = [int(ds.feature_num_bin(i)) for i in range(ds.num_feature())]
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "dump.txt"
        ds._dump_text(str(f))
        lines = f.read_text().splitlines()
    n = int(next(t for t in lines if t.startswith("num_data:")).split(":")[1])
    data = lines[-n:]
    if n != len(X) or len(data) != n:
        raise SystemExit(f"the dump holds {n} rows for {len(X)}")
    head = hashlib.sha256("\n".join(data[:n_head]).encode()).hexdigest()
    return {"rows": int(len(X)), "weighted": w is not None, "num_bin_by_feature": nb,
            "total_bins": int(sum(nb)), "features_with_one_bin": int(sum(b <= 1 for b in nb)),
            "fold_rows_binned_sha256": head, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    OF.NO.refuse_existing([OUT], overwrite=False)
    t0 = time.time()
    X, info = fitting_matrix()
    params = learner_params()
    fold = X[:N_FOLD]
    wz = np.r_[np.ones(N_FOLD), np.zeros(N_JUNE)]
    dup = np.vstack([fold, fold[:N_JUNE]])
    full_sample = dict(params, bin_construct_sample_cnt=10 ** 7)
    v = {"i_fold_only": binned(fold, None, params, N_FOLD),
         "ii_june_weight_0": binned(X, wz, params, N_FOLD),
         "ii_june_no_weight": binned(X, None, params, N_FOLD),
         "ii_june_every_row_in_bin_sample": binned(X, wz, full_sample, N_FOLD),
         "iii_fold_plus_duplicates": binned(dup, None, params, N_FOLD)}
    ref = v["i_fold_only"]
    cmp = {}
    for k, r in v.items():
        if k == "i_fold_only":
            continue
        diff = [j for j, (a, b) in enumerate(zip(ref["num_bin_by_feature"], r["num_bin_by_feature"])) if a != b]
        cmp[k] = {"fold_rows_binned_as_in_i": r["fold_rows_binned_sha256"] == ref["fold_rows_binned_sha256"],
                  "features_whose_bin_count_differs_from_i": len(diff),
                  "total_bins_minus_i": r["total_bins"] - ref["total_bins"]}
    weight_ignored_by_bins = (v["ii_june_weight_0"]["fold_rows_binned_sha256"]
                              == v["ii_june_no_weight"]["fold_rows_binned_sha256"]
                              and v["ii_june_weight_0"]["num_bin_by_feature"]
                              == v["ii_june_no_weight"]["num_bin_by_feature"])
    zero_bins_neutral = cmp["ii_june_weight_0"]["fold_rows_binned_as_in_i"]
    dup_bins_neutral = cmp["iii_fold_plus_duplicates"]["fold_rows_binned_as_in_i"]
    findings = {
        "zero_weight": {
            "neutral": False,
            "reasons": [
                "fit_arm cannot give the extra rows weight 0: extra_share must lie in the open "
                "interval (0, 1), and wmap is one weight per airport, shared by fold and extra rows "
                "(code_path.extra_share_open_interval, wmap_is_per_airport)",
                "arm_lgb's per-airport amplitude takes no weight: it is an unweighted mean over "
                "every fitting row, so zero-weight June rows still enter it through their true "
                "rbase, a label channel (code_path.amplitude_*)",
                "min_child_samples counts rows, not weight, and bagging samples rows "
                "(docs.min_child_samples, docs.subsample, min_child_weight is the weight floor)",
                ("the bins ignore the weight and are built from every row in the bin sample, so the "
                 "fold rows are binned differently with zero-weight June rows present"
                 if (weight_ignored_by_bins and not zero_bins_neutral) else
                 "bins: see binned; the zero-weight rows " + ("do" if not zero_bins_neutral else "do not")
                 + " change the fold rows' binned values"),
            ]},
        "duplicated_rows": {
            "neutral": False,
            "reasons": [
                "a duplicated row doubles its weight in every gradient, hessian and amplitude sum, "
                "so the control reweights 98,396 of the 172,783 fold rows instead of adding capacity "
                "alone",
                "min_child_samples counts the copies, so a leaf may hold 50 distinct rows twice: the "
                "effective leaf floor halves where copies sit (docs.min_child_samples)",
                "a copy shares its original's day and so its cross-fit day block; the copies add no "
                "cross-block leak but count twice in the per-airport amplitude",
                ("the bin sample holds the copies, so the fold rows are binned differently"
                 if not dup_bins_neutral else "the fold rows' binned values do not change"),
            ]},
    }
    rec = {"what": __doc__.strip().splitlines()[0], "spec_commit": OF.SPEC_COMMIT,
           "lightgbm_version": lgb.__version__, "dylib_sha256": OF.sha_file(OF.DYLIB),
           "fitting_matrix": info, "dataset_params": {k: (v2 if isinstance(v2, (int, float, str, bool, type(None), list))
                                                          else str(v2)) for k, v2 in params.items()},
           "code_path": code_path(), "docs": docs(), "binned": v, "compared_with_i": cmp,
           "bins_ignore_the_weight": weight_ignored_by_bins, "findings": findings,
           "limit": ("Neither zero-weight rows nor duplicated rows is a capacity-neutral control for "
                     "this learner. The sham stays a specificity control that damages the base; its "
                     "damage is measured (SPEC 6.3) and nothing is corrected with it."),
           "census_check": OF.census_check(OF.OPENED), "seconds": round(time.time() - t0, 1)}
    cc = rec["census_check"]
    if cc["missing"] or cc["differ"]:
        raise SystemExit(f"a file this diagnostic opened fails the census: {cc}")
    OF.write_json(OUT, rec)
    for k, r in v.items():
        c = cmp.get(k, {})
        print(f"{k:34} rows {r['rows']:>7,} total bins {r['total_bins']:>6,} "
              f"one-bin features {r['features_with_one_bin']:>3}  fold rows binned as (i): "
              f"{c.get('fold_rows_binned_as_in_i', True)}  features with another bin count: "
              f"{c.get('features_whose_bin_count_differs_from_i', 0)}")
    print(f"bins ignore the weight: {weight_ignored_by_bins}; census {cc['checked']} files match; "
          f"{rec['seconds']}s")


if __name__ == "__main__":
    main()
