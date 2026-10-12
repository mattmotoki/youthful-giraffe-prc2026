"""exp-018b step 2: the repo's modules, attached to this lineage's frames.

Four things exp-018 lacked and the repo measured. Each is built once per fold
and cached, so the ablation that follows changes only which columns a fit sees
and never recomputes a module.

  phi      prc.phi_features.PhiCV(PhiModel(weight="rows")), cross-fitted:
           a training row in month m sees the fit that excluded m, a serving
           row sees the average of the K fits. This is exp-006's answer to the
           construction question and the reason its july number moved 26.5 s.
  surface  prc.surface_features VERSION 2, the seven families, minus the
           columns this lineage already covers (see SKIP below).
  deice    prc.day_state_features.DayState, the DEICE column only.
  drop     prc.data.UNREACHED_ROLLOVER: four training rows whose label no input
           can reach. Not a column -- a training mask -- so it lives in the
           model, not here.

  PYTHONPATH=. .venv/bin/python experiments/exp-018b-climb/code/extras.py <fold> [...]
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp-018-fresh" / "code"))
from prc import data                       # noqa: E402
from prc import phi_features as PF         # noqa: E402
from prc import surface_features as SF     # noqa: E402
from prc import day_state_features as DS   # noqa: E402

CACHE = data.ROOT / "artifacts" / "exp-018b" / "extras"

#: Surface columns this lineage already has an equivalent of, so that "what the
#: repo measured and you lack" means what it says. Left column is theirs, the
#: parenthesis is mine (`experiments/exp-018-fresh/code/features.py`).
SKIP = {
    "gap_prev_rwy", "gap_next_rwy", "gap_prev_apt", "gap_next_apt",   # rwy/apt_gap_*
    "nbr_taxiin_apt", "nbr_taxiin_rwy",                               # arr_taxiin_last*
    "stand_nxt_arr_dt", "stand_prv_arr_dt",                           # stand_*_arr_inblock_dt
    "stand_nxt_dep_off_dt", "stand_prv_dep_off_dt",                   # stand_*_dep_dt
    "stand_turnaround",                                               # stand_prev_arr_inblock_dt
    "stand_rwy_med",                                                  # stand_taxiin_med / te_y_standrwy
    "rwy_share",                                                      # rwy_share_1h
    "dep_rate_hr",                                                    # apt_dep_1h_before
    "cfg_dep", "cfg_arr",                                             # not covered but categorical-coded
}
SURFACE_FAMILIES = ("queue", "pressure", "occupancy", "config", "flow", "geometry", "stand")


#: exp-035-k2climb arm 3. Which cached column belongs to which module, derived
#: from the same three sources `build` uses rather than guessed from a prefix:
#: `surface` is the kept families, `deice` is `DS.DEICE`, and `phi` is the rest.
#: On the committed cache these partition it exactly, 11 + 80 + 1 = 92.
KINDS = ("phi", "surface", "deice")


def partition(available) -> dict:
    """{kind: [column, ...]} over the columns actually present in a cache."""
    have = [c for c in available if c != "MVT_ID_mvt"]
    surface = [c for fam in SURFACE_FAMILIES for c in SF.FAMILY_COLUMNS[fam]
               if c not in SKIP]
    surface = [c for c in surface if c in have]
    deice = [c for c in DS.DEICE if c in have]
    taken = set(surface) | set(deice)
    return {"phi": [c for c in have if c not in taken],
            "surface": surface, "deice": deice}


def kinds_from_env(env_value: str) -> list:
    """Parse and validate `PRC_EXTRAS`. Empty means "no module columns"."""
    v = (env_value or "").strip()
    if not v:
        return []
    ks = [k.strip() for k in v.replace(" ", ",").split(",") if k.strip()]
    bad = [k for k in ks if k not in KINDS]
    if bad:
        raise SystemExit(f"PRC_EXTRAS names unknown kinds {bad}; known are {list(KINDS)}")
    return ks


def attach(df: pd.DataFrame, extra: pd.DataFrame, kinds, label: str) -> pd.DataFrame:
    """Left-join the selected module columns onto `df` on MVT_ID_mvt, or fail.

    The join is fold-keyed and never month-keyed: `phi` is cross-fitted, so a
    training month's columns come from the fit that excluded it and the
    validation month's from the average of the K fits. A silent partial join
    here would read as a clean negative -- LightGBM accepts missing features --
    so a changed row count and an all-missing match are both fatal.
    """
    part = partition(extra.columns)
    keep = [c for k in kinds for c in part[k]]
    if not keep:
        raise SystemExit(f"{label}: PRC_EXTRAS={kinds} selects no column of this cache")
    n = len(df)
    out = df.merge(extra[["MVT_ID_mvt"] + keep], on="MVT_ID_mvt", how="left")
    if len(out) != n:
        raise SystemExit(f"{label}: the extras join changed the row count {n} -> "
                         f"{len(out)}; the module cache has duplicate ids")
    miss = int(out[keep].isna().all(axis=1).sum())
    print(f"[extras] {label}: +{len(keep)} columns "
          + " ".join(f"{k}={len(part[k])}" for k in kinds)
          + f", {miss} rows with none ({100 * miss / max(n, 1):.2f} %)", flush=True)
    if miss == n:
        raise SystemExit(f"{label}: the extras join matched no row at all")
    return out


def _join(parts: list[pd.DataFrame]) -> pd.DataFrame:
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="MVT_ID_mvt", how="outer")
    return out.drop_duplicates("MVT_ID_mvt")


def _masked_valid(month: str) -> pd.DataFrame:
    f = data.load_month(month)
    masked, _ = data.mask_like_ranking(f)
    return masked


def build(fold_name: str, rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (train, valid) extra columns for one fold in `prc.data.FOLDS`.

    Serving note: this builds folds only. If a module arm is adopted, the
    candidate needs the same columns for the ranking file (train = all twelve
    2025 months, valid = the two 2026 months, each transformed in isolation);
    `build_rank` below is that path.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    ptr, pva = CACHE / f"{fold_name}-train.parquet", CACHE / f"{fold_name}-valid.parquet"
    if ptr.exists() and pva.exists() and not rebuild:
        return pd.read_parquet(ptr), pd.read_parquet(pva)
    fold = data.FOLDS[fold_name]
    t0 = time.time()

    # ---- raw frames. Training months keep labels (phi and the surface geometry
    # families are fitted on them); the validation month is masked exactly as
    # the ranking file is, and every transform reads only that masked frame.
    train_by_month = {m: data.load_month(m) for m in fold.train_months}
    valid_raw = _masked_valid(fold.valid_month)
    print(f"[{fold_name}] raw loaded {time.time()-t0:.0f}s", flush=True)

    tr_parts, va_parts = [], []

    # ---- phi (cross-fitted) -------------------------------------------------
    cv = PF.PhiCV(PF.PhiModel(weight="rows")).fit(train_by_month)
    ptr_list = [cv.transform_train(m, train_by_month[m]) for m in fold.train_months]
    tr_phi = pd.concat(ptr_list, ignore_index=True)
    va_phi = cv.transform_serve(valid_raw, mode="avg")
    tr_parts.append(tr_phi); va_parts.append(va_phi)
    print(f"[{fold_name}] phi {tr_phi.shape[1]-1} cols {time.time()-t0:.0f}s", flush=True)

    # ---- surface (VERSION 2, seven families, minus what we already have) ----
    sf = SF.SurfaceFeatures.v2().fit(pd.concat(train_by_month.values(), ignore_index=True))
    keep = [c for fam in SURFACE_FAMILIES for c in SF.FAMILY_COLUMNS[fam] if c not in SKIP]
    tr_sur = []
    for m in fold.train_months:
        f = train_by_month[m].assign(month=m)
        x = SF.build_features(f, sf)
        x.insert(0, "MVT_ID_mvt", f.loc[x.index, "MVT_ID_mvt"].to_numpy())
        tr_sur.append(x[["MVT_ID_mvt"] + [c for c in keep if c in x.columns]])
    tr_sur = pd.concat(tr_sur, ignore_index=True)
    xv = SF.build_features(valid_raw.assign(month=fold.valid_month), sf)
    xv.insert(0, "MVT_ID_mvt", valid_raw.loc[xv.index, "MVT_ID_mvt"].to_numpy())
    va_sur = xv[["MVT_ID_mvt"] + [c for c in keep if c in xv.columns]]
    tr_parts.append(tr_sur); va_parts.append(va_sur)
    print(f"[{fold_name}] surface {tr_sur.shape[1]-1} cols {time.time()-t0:.0f}s", flush=True)

    # ---- deice --------------------------------------------------------------
    ds = DS.DayState().fit(pd.concat(train_by_month.values(), ignore_index=True))
    tr_de = []
    for m in fold.train_months:
        f = train_by_month[m]
        x = ds.transform(f)
        cols = [c for c in DS.DEICE if c in x.columns]
        x = x[cols].copy(); x.insert(0, "MVT_ID_mvt", f.loc[x.index, "MVT_ID_mvt"].to_numpy())
        tr_de.append(x)
    tr_de = pd.concat(tr_de, ignore_index=True)
    xv = ds.transform(valid_raw)
    cols = [c for c in DS.DEICE if c in xv.columns]
    va_de = xv[cols].copy()
    va_de.insert(0, "MVT_ID_mvt", valid_raw.loc[xv.index, "MVT_ID_mvt"].to_numpy())
    tr_parts.append(tr_de); va_parts.append(va_de)
    print(f"[{fold_name}] deice {tr_de.shape[1]-1} cols {time.time()-t0:.0f}s", flush=True)

    tr, va = _join(tr_parts), _join(va_parts)
    tr.to_parquet(ptr, index=False); va.to_parquet(pva, index=False)
    print(f"[{fold_name}] cached train={tr.shape} valid={va.shape} in {time.time()-t0:.0f}s", flush=True)
    return tr, va


def build_rank(rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The serving half of `build`: train = all twelve 2025 months, valid = the
    live ranking file.

    The ranking file holds January and July 2026 as two isolated months, so
    every per-month transform runs on one month at a time and no neighbour
    feature reaches across the six-month gap (`prc.surface_features
    .transform_by_month` states the same rule and measures its effect).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    ptr, pva = CACHE / "rank-train.parquet", CACHE / "rank-valid.parquet"
    if ptr.exists() and pva.exists() and not rebuild:
        return pd.read_parquet(ptr), pd.read_parquet(pva)
    t0 = time.time()
    train_by_month = {m: data.load_month(m) for m in data.MONTHS}
    rk = data.load_ranking()
    rank_months = [m for m in ("2026-01", "2026-07") if (rk["month"] == m).any()]
    rank_by_month = {m: rk[rk["month"] == m] for m in rank_months}
    print(f"[rank] raw loaded {time.time()-t0:.0f}s", flush=True)

    tr_parts, va_parts = [], []

    cv = PF.PhiCV(PF.PhiModel(weight="rows")).fit(train_by_month)
    tr_parts.append(pd.concat([cv.transform_train(m, train_by_month[m]) for m in data.MONTHS],
                              ignore_index=True))
    va_parts.append(pd.concat([cv.transform_serve(rank_by_month[m], mode="avg")
                               for m in rank_months], ignore_index=True))
    print(f"[rank] phi {tr_parts[-1].shape[1]-1} cols {time.time()-t0:.0f}s", flush=True)

    sf = SF.SurfaceFeatures.v2().fit(pd.concat(train_by_month.values(), ignore_index=True))
    keep = [c for fam in SURFACE_FAMILIES for c in SF.FAMILY_COLUMNS[fam] if c not in SKIP]

    def _surface(by_month: dict) -> pd.DataFrame:
        out = []
        for m, f in by_month.items():
            x = SF.build_features(f.assign(month=m), sf)
            x.insert(0, "MVT_ID_mvt", f.loc[x.index, "MVT_ID_mvt"].to_numpy())
            out.append(x[["MVT_ID_mvt"] + [c for c in keep if c in x.columns]])
        return pd.concat(out, ignore_index=True)

    tr_parts.append(_surface(train_by_month)); va_parts.append(_surface(rank_by_month))
    print(f"[rank] surface {tr_parts[-1].shape[1]-1} cols {time.time()-t0:.0f}s", flush=True)

    ds = DS.DayState().fit(pd.concat(train_by_month.values(), ignore_index=True))

    def _deice(by_month: dict) -> pd.DataFrame:
        out = []
        for m, f in by_month.items():
            x = ds.transform(f)
            cols = [c for c in DS.DEICE if c in x.columns]
            y = x[cols].copy()
            y.insert(0, "MVT_ID_mvt", f.loc[x.index, "MVT_ID_mvt"].to_numpy())
            out.append(y)
        return pd.concat(out, ignore_index=True)

    tr_parts.append(_deice(train_by_month)); va_parts.append(_deice(rank_by_month))
    print(f"[rank] deice {time.time()-t0:.0f}s", flush=True)

    tr, va = _join(tr_parts), _join(va_parts)
    tr.to_parquet(ptr, index=False); va.to_parquet(pva, index=False)
    print(f"[rank] cached train={tr.shape} valid={va.shape} in {time.time()-t0:.0f}s", flush=True)
    return tr, va


PREFIX = {"phi": None, "surface": None, "deice": None}


def columns_of(kind: str, frame_cols) -> list[str]:
    """Which cached extra columns belong to which module."""
    if kind == "deice":
        return [c for c in DS.DEICE if c in frame_cols]
    if kind == "surface":
        keep = [c for fam in SURFACE_FAMILIES for c in SF.FAMILY_COLUMNS[fam] if c not in SKIP]
        return [c for c in keep if c in frame_cols]
    if kind == "phi":
        known = set(DS.DEICE) | {c for fam in SURFACE_FAMILIES for c in SF.FAMILY_COLUMNS[fam]}
        return [c for c in frame_cols if c != "MVT_ID_mvt" and c not in known]
    raise ValueError(kind)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for f in (args or ["july", "july_loo", "december", "january"]):
        if f == "rank":
            build_rank(rebuild="--rebuild" in sys.argv)
        else:
            build(f, rebuild="--rebuild" in sys.argv)
