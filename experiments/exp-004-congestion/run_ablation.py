"""exp-004 ablation: base model vs base + each surface-state family.

Features for a month are always computed from that month's frame alone
(`prc.surface_features` blanks departure labels internally), so a training
month is treated exactly like a ranking month.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import prc.data as D
import prc.surface_features as S
from common import (base_features, base_features_v2, metrics, fmt, fit_predict,
                    BASE_CAT, EXTRA_NUM, TARGET, ID)

CACHE = HERE.parents[1] / "artifacts" / "exp-004-congestion"
CACHE.mkdir(parents=True, exist_ok=True)


def _code_fingerprint() -> str:
    """Hash of the code that produces a cached feature frame.

    A cache keyed only by filename silently survives a change to the feature
    code or to the fitted state, which would mix feature generations inside one
    comparison. Any edit to either module invalidates every cache entry.
    """
    h = hashlib.sha256()
    for f in (HERE.parents[1] / "prc" / "surface_features.py", HERE / "common.py"):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _meta_path(p: Path) -> Path:
    return p.with_suffix(".meta.json")


def _cache_valid(p: Path, fingerprint: str) -> bool:
    m = _meta_path(p)
    if not (p.exists() and m.exists()):
        return False
    try:
        meta = json.loads(m.read_text())
    except Exception:
        return False
    return meta.get("code") == fingerprint


BASEV = {"v1": ("feat", base_features), "v2": ("featB", base_features_v2)}


def month_features(fold: str, month: str, sf: S.SurfaceFeatures, base: str = "v1") -> pd.DataFrame:
    pre, bfun = BASEV[base]
    p = CACHE / f"{pre}_{fold}_{month}.parquet"
    fp = _code_fingerprint()
    if _cache_valid(p, fp):
        return pd.read_parquet(p)
    df = D.load_month(month)
    masked, _ = D.mask_like_ranking(df)
    B = bfun(masked)
    F = sf.transform(masked)
    out = pd.concat([B, F.loc[B.index]], axis=1)
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype("float32")
    out[ID] = masked.loc[B.index, ID].to_numpy()
    out["y"] = df.loc[B.index, TARGET].to_numpy(float)
    # segment keys the scoring needs, kept out of the model's column list
    out["seg_noac"] = masked.loc[B.index, "AIRCRAFT_TYPE_mvt"].isna().to_numpy()
    out["seg_apt"] = masked.loc[B.index, "ADEP_mvt"].astype(str).to_numpy()
    out["seg_day"] = masked.loc[B.index, "MVT_TIME_UTC_mvt"].dt.floor("D").astype("int64").to_numpy()
    out.reset_index(drop=True).to_parquet(p, index=False)
    _meta_path(p).write_text(json.dumps(
        {"code": fp, "fold": fold, "month": month, "base": base,
         "columns": list(out.columns), "rows": int(len(out))}, indent=2))
    return pd.read_parquet(p)


def build_fold(fold_name: str, base: str = "v1"):
    fold = D.FOLDS[fold_name]
    t0 = time.time()
    fitp = CACHE / f"sf_{fold_name}.pkl"
    pre = BASEV[base][0]
    fp = _code_fingerprint()
    need = (not _cache_valid(fitp, fp)) or not all(
        _cache_valid(CACHE / f"{pre}_{fold_name}_{m}.parquet", fp)
        for m in fold.train_months + (fold.valid_month,))
    train_raw = D.load_months(fold.train_months) if need else None
    import pickle
    if _cache_valid(fitp, fp):
        sf = pickle.loads(fitp.read_bytes())
    else:
        sf = S.SurfaceFeatures().fit(train_raw)
        fitp.write_bytes(pickle.dumps(sf))
        _meta_path(fitp).write_text(json.dumps({"code": fp, "fold": fold_name}, indent=2))
    tr = pd.concat([derive_relative(month_features(fold_name, m, sf, base))
                    for m in fold.train_months], ignore_index=True)
    va = derive_relative(month_features(fold_name, fold.valid_month, sf, base))
    print(f"[{fold_name}] features train {tr.shape} valid {va.shape} in {time.time()-t0:.0f}s", flush=True)
    return tr, va


REL_SOURCES = ["nbr_span_apt", "nbr_span_rwy", "nbr_taxiin_apt", "nbr_taxiin_rwy",
               "nbr_excess_apt", "nbr_excess_rwy"]

# repair families: column subsets that are not one of prc.surface_features' six
EXTRA_FAMILIES = {
    # config without the two high-cardinality categoricals
    "config_num": ["n_dep_rwy", "n_arr_rwy", "cfg_changed", "rwy_share"],
    # month-relative forms of the flow level features (see RESULTS section 6)
    "flow_rel": ["dep_rate_hr", "mvt_rate_hr", "sat_dep", "sat_mvt"] + [c + "_rel" for c in REL_SOURCES],
}


def derive_relative(df: pd.DataFrame) -> pd.DataFrame:
    """Month-relative forms of the neighbour level features.

    A rolling median of `take-off - AOBT` is an absolute level; its level moves
    from month to month, so a tree that splits on it does not transfer. The
    airport's own median inside the same month frame is observable, so
    subtracting it turns the feature into a within-month congestion index.
    """
    for c in REL_SOURCES:
        if c in df.columns:
            df[c + "_rel"] = df[c] - df.groupby("apt")[c].transform("median")
    return df


BASE_NUM = ["dt_to_aobt", "dt_to_lobt", "dt_to_eobt", "dt_to_sched", "dt_aobt_sched",
            "dt_aobt_lobt", "flight_time_est", "hour", "dow", "has_nm"]


def columns_for(families, base: str = "v1"):
    cols = list(BASE_NUM) + (list(EXTRA_NUM) if base == "v2" else []) + list(BASE_CAT)
    cats = list(BASE_CAT)
    for f in families:
        cols_f = S.FAMILY_COLUMNS.get(f) or S.EXTRA_FAMILY_COLUMNS.get(f) or EXTRA_FAMILIES.get(f)
        if cols_f is None:
            raise KeyError(f"unknown family {f!r}")
        for c in cols_f:
            cols.append(c)
            if c in S.CAT_FEATURES:
                cats.append(c)
    return cols, cats


def run(fold_name: str, configs: dict, params=None, save_oof=False, tag="", base="v1"):
    tr, va = build_fold(fold_name, base)
    ytr = tr["y"].to_numpy(float)
    yva = va["y"].to_numpy(float)
    has_nm = va["has_nm"].to_numpy()
    noac = va["seg_noac"].to_numpy()
    out = {}
    for name, fams in configs.items():
        cols, cats = columns_for(fams, base)
        t0 = time.time()
        model, p = fit_predict(tr[cols], ytr, va[cols], cats, params)
        m = metrics(yva, p, has_nm, noac)
        m["secs"] = round(time.time() - t0)
        m["n_features"] = len(cols)
        out[name] = m
        print(f"[{fold_name}] {name:22s} {fmt(m)} ({m['secs']}s)", flush=True)
        res_path = HERE / f"ablation_{fold_name}{tag}.json"
        prev = json.loads(res_path.read_text()) if res_path.exists() else {}
        prev[name] = m
        res_path.write_text(json.dumps(prev, indent=2))
        if save_oof:
            oof = pd.DataFrame({ID: va[ID].to_numpy(), "fold": fold_name,
                                "y_true": yva, "y_pred": p,
                                "seg_noac": noac, "seg_apt": va["seg_apt"].to_numpy(),
                                "seg_day": va["seg_day"].to_numpy(),
                                "seg_has_nm": has_nm})
            (HERE / "oof").mkdir(exist_ok=True)
            oof.to_parquet(HERE / "oof" / f"{name}_{fold_name}.parquet", index=False)
            imp = pd.Series(model.get_feature_importance(), index=cols).sort_values(ascending=False)
            imp.to_csv(HERE / f"importance_{name}_{fold_name}.csv", header=["importance"])
    return out


SINGLES = {
    "base": (),
    "base+queue": ("queue",),
    "base+pressure": ("pressure",),
    "base+occupancy": ("occupancy",),
    "base+config": ("config",),
    "base+flow": ("flow",),
    "base+geometry": ("geometry",),
    "base+stand": ("stand",),
    "base+all": tuple(S.FAMILIES),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["july", "december"])
    ap.add_argument("--configs", nargs="+", default=list(SINGLES))
    ap.add_argument("--oof", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--base", default="v1", choices=["v1", "v2"])
    ap.add_argument("--seed", type=int, default=None,
                    help="override the CatBoost seed (for a run-to-run noise estimate)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="ad-hoc config as name=fam1,fam2")
    a = ap.parse_args()
    cfg = {k: SINGLES[k] for k in a.configs if k in SINGLES}
    for e in a.extra:
        name, fams = e.split("=", 1)
        cfg[name] = tuple(x for x in fams.split(",") if x)
    params = {"random_seed": a.seed} if a.seed is not None else None
    for f in a.folds:
        run(f, cfg, params=params, save_oof=a.oof, tag=a.tag, base=a.base)
