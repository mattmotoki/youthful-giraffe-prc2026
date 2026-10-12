"""exp-189: the final route feature tables, `features-r4c.parquet` and `proxy-features-r4c.parquet` of
the final file: the saved tables (exp-148 combine_r4, the v73 inputs) with their 2026 rows (January and
July of `ranking.parquet`) replaced by the 2026 rows of the final file (four months).

    PRC_RANKING_MODE=final python combine_final.py --features-saved <features-r4c.parquet> \
        --features-2026 <features-2026.parquet> --proxy-saved <proxy-features-r4c.parquet> \
        --proxy-new <proxy-features.parquet of learner_wp --fit features, final mode> --out DIR

exp-148 combine_r2/combine_r4 build the saved tables by appending whole months (no step reads two
months together), so a 2026 row of the full chain is the row that `features_2026.py` and the
learner's `--fit features` write. The new rows go where the old 2026 rows began; every 2025 row stays,
in its order and with its bytes. Checks (stop on failure): the new 2026 ids are the departures of the
mode's submitting template, once each; the new tables have the saved columns and dtypes; the 2025 rows
that the new proxy table also holds equal the saved ones (as combine_r2 checked January), except the columns
named in `--allow-2025-diff` (their differing rows, NaN changes and largest difference go into the record).
Reason for the exception (finalroute, 10-08 18:40 UTC, `results/proxy-2025-diff.txt`): the final-mode stand
table `stand_geometry-final.csv` holds 39 stand names more than the ranking one, so 2025 departures at those
stands get an implied clock, and the window lags of their airport-days move. Write-once:
`features-r4c.parquet`, `proxy-features-r4c.parquet`, `combine.json` (hashes, rows by month, checks).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from prc import data as D            # noqa: E402

ID = "MVT_ID_mvt"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def replace_2026(saved: pd.DataFrame, new: pd.DataFrame, is26: np.ndarray, what: str) -> pd.DataFrame:
    """The saved rows with every 2026 row taken out and `new` put where the first one stood."""
    if set(new.columns) != set(saved.columns):
        raise SystemExit(f"{what}: columns differ: {sorted(set(new.columns) ^ set(saved.columns))}")
    new = new.reindex(columns=saved.columns)
    for c in saved.columns:
        if new[c].dtype != saved[c].dtype:
            new[c] = new[c].astype(saved[c].dtype)
    at = int(np.flatnonzero(is26)[0])
    keep = saved[~is26]
    before = int((~is26[:at]).sum())
    out = pd.concat([keep.iloc[:before], new, keep.iloc[before:]], ignore_index=True)
    if out[ID].duplicated().any():
        raise SystemExit(f"{what}: an id repeats")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-saved", type=Path, required=True)
    ap.add_argument("--features-2026", type=Path, required=True)
    ap.add_argument("--proxy-saved", type=Path, required=True)
    ap.add_argument("--proxy-new", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-2025-diff", default="", help="comma list of proxy columns whose 2025 rows may differ")
    a = ap.parse_args()
    outs = {"features": a.out / "features-r4c.parquet", "proxy": a.out / "proxy-features-r4c.parquet"}
    js = a.out / "combine.json"
    if js.exists() or any(p.exists() for p in outs.values()):
        raise SystemExit(f"{a.out}: combined tables exist; write-once")
    t0 = time.monotonic()
    src = {"features_saved": a.features_saved, "features_2026": a.features_2026,
           "proxy_saved": a.proxy_saved, "proxy_new": a.proxy_new}
    rec = {"what": "exp-189 final route feature tables (saved 2025 rows, final-file 2026 rows)",
           "ranking_mode": D.ranking_mode(), "sources": {k: str(p) for k, p in src.items()},
           "sources_sha256": {k: sha(p) for k, p in src.items()}, "checks": {}}
    t = D.load_submitting()
    tid = pd.Index(t[ID].to_numpy("float64").astype("int64"))
    assert tid.is_unique and len(tid) == D.n_ranking()
    months26 = list(D.rank_months())

    # ---- features
    fs, fn = pd.read_parquet(a.features_saved), pd.read_parquet(a.features_2026)
    is26 = (fs.ctx_year == 2026).to_numpy()
    if not (fn.ctx_year == 2026).all():
        raise SystemExit("features-2026 holds a row of another year")
    nid = pd.Index(fn[ID].astype("int64"))
    if not nid.is_unique or len(nid) != len(tid) or not nid.isin(tid).all():
        raise SystemExit(f"features-2026: {len(nid):,} ids, template {len(tid):,}")
    f = replace_2026(fs, fn, is26, "features")
    old = fs[~is26].reset_index(drop=True)
    kept = f[(f.ctx_year != 2026).to_numpy()].reset_index(drop=True)
    if not kept.equals(old):
        raise SystemExit("features: the 2025 rows changed")
    rec["checks"]["features"] = {
        "rows_saved": int(len(fs)), "rows_saved_2026": int(is26.sum()), "rows_new_2026": int(len(fn)),
        "rows_out": int(len(f)), "rows_2025_equal": True,
        "rows_2026_by_month": {str(k): int(v) for k, v in fn.ctx_day.str[:7].value_counts().sort_index().items()}}

    # ---- proxy
    ps, pn = pd.read_parquet(a.proxy_saved), pd.read_parquet(a.proxy_new)
    p26s = ps.month.astype(str).str.startswith("2026").to_numpy()
    p26n = pn.month.astype(str).str.startswith("2026").to_numpy()
    if sorted(pn.month[p26n].astype(str).unique()) != months26:
        raise SystemExit(f"proxy-new 2026 months {sorted(pn.month[p26n].unique())} != {months26}")
    pid = pd.Index(pn.loc[p26n, ID].astype("int64"))
    if not pid.is_unique or not pid.isin(tid).all():
        raise SystemExit("proxy-new: 2026 ids repeat or lie outside the template")
    parity = {}
    for m in sorted(pn.month[~p26n].astype(str).unique()):
        j0 = ps[ps.month.astype(str) == m].set_index(ID).sort_index()
        j1 = pn[pn.month.astype(str) == m].set_index(ID).sort_index()
        if not j0.index.equals(j1.index):
            raise SystemExit(f"proxy {m}: ids differ from the saved table")
        allowed = [c for c in a.allow_2025_diff.split(",") if c]
        if not set(allowed) <= set(j0.columns):
            raise SystemExit(f"--allow-2025-diff names columns outside the proxy table: {sorted(set(allowed) - set(j0.columns))}")
        known = {}
        for c in allowed:
            x, y = j0[c].to_numpy("float64"), j1[c].to_numpy("float64")
            nmc = np.isnan(x) != np.isnan(y)
            dc = np.where(np.isnan(x) | np.isnan(y), 0.0, np.abs(x - y))
            known[c] = {"rows_differ": int(((dc > 0) | nmc).sum()), "nan_mismatch": int(nmc.sum()),
                        "max_abs_diff": float(dc.max()) if len(dc) else 0.0}
        num = [c for c in j0.columns if c != "month" and c not in allowed]
        diff = float(np.nanmax(np.abs(j0[num].to_numpy("float64") - j1[num].to_numpy("float64"))))
        nan_mis = int((j0[num].isna().to_numpy() != j1[num].isna().to_numpy()).sum())
        if diff != 0.0 or nan_mis:
            raise SystemExit(f"proxy {m}: differs from the saved table: max {diff}, NaN pattern {nan_mis}")
        parity[m] = {"rows": int(len(j0)), "max_abs_diff": diff, "nan_mismatch": nan_mis,
                     "columns_equal": len(num), "allowed_differences": known}
    p = replace_2026(ps, pn[p26n], p26s, "proxy")
    rec["checks"]["proxy"] = {
        "rows_saved": int(len(ps)), "rows_saved_2026": int(p26s.sum()), "rows_new_2026": int(p26n.sum()),
        "rows_out": int(len(p)), "parity_2025_months": parity,
        "rows_2026_by_month": {str(k): int(v) for k, v in pn.month[p26n].value_counts().sort_index().items()},
        "feature_ids_without_proxy_2026": int((~nid.isin(pid)).sum()),
        "proxy_ids_without_feature_2026": int((~pid.isin(nid)).sum())}

    a.out.mkdir(parents=True, exist_ok=True)
    f.to_parquet(outs["features"], index=False)
    p.to_parquet(outs["proxy"], index=False)
    rec["outputs_sha256"] = {k: sha(v) for k, v in outs.items()}
    rec["code_sha256"] = sha(Path(__file__))
    rec["seconds"] = round(time.monotonic() - t0, 1)
    js.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(rec["checks"], indent=1, default=str))


if __name__ == "__main__":
    main()
