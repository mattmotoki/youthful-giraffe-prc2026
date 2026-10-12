"""exp-189: the route parts of the final file under the part contract (COMMON).

    PRC_RANKING_MODE=final python make_parts.py member --name T-tailv4 --src <members dir> [--root R]
    PRC_RANKING_MODE=final python make_parts.py level2 --src <R4mksP-final.parquet> [--root R]
    PRC_RANKING_MODE=final python make_parts.py route --r4mks <parts/r4mks.parquet> [--gbdt <parquet>] [--root R]

A part is `<root>/finalroute/parts/<part>.parquet` (root: `--root`, else `$PRC_FINAL_ROOT`, else
$PRC_FINAL_ROOT): one row per departure of `final_submitting.parquet` (670,790) in template
order, key `MVT_ID_mvt` (float64, as the template), value columns float64 (NaN where the part has no
value), plus the columns a consumer needs. A JSON record `<part>.json` beside it holds the SHA-256, rows,
columns, inputs with SHA-256, the code SHA-256 and the time. Write-once.

Parts:
* `member`: one part per level-1 member, named as the member except L -> `l` and Lp -> `lp`; the
  member's own value column (`y_pred`, `pred_L`, `pred_Lp`; N1 also `has_nm`).
* `level2`: part `r4mks`, the saved R4mks level 2 on the final members (`fold_serve.py --split final`):
  `pred_Sb`, `pred_S`, `pred_Scat`, `anchor`, `anchor_served`.
* `route`: parts `route` (`TAXITIME_SEC_mvt`, `route_source`: 0 R4mks `pred_Sb`, 1 official-columns
  GBDT where `pred_Sb` is not finite), `F` (column `F`, the same values: the v32 replacement of the rule
  frame) and `np_base` (column `np_base`, the same values; the no-plan mask is the consumer's).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from prc import data as D            # noqa: E402

ID = "MVT_ID_mvt"
MEMBER_COLS = {"T-tailv4": ["y_pred"], "C2-cat": ["y_pred"], "L2-hub8725": ["y_pred"],
               "D1-winter": ["y_pred"], "F1-fresh": ["y_pred"], "N1-nmtail": ["y_pred", "has_nm"],
               "L": ["pred_L"], "Lp": ["pred_Lp"]}
PART_NAME = {"L": "l", "Lp": "lp"}
LEVEL2_COLS = ["pred_Sb", "pred_S", "pred_Scat", "anchor", "anchor_served"]
FLOOR = 60.0


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def template() -> np.ndarray:
    if D.ranking_mode() != "final":
        raise SystemExit("make_parts.py runs in the final mode (PRC_RANKING_MODE=final)")
    t = D.load_submitting()
    ids = t[ID].to_numpy("float64")
    assert len(ids) == D.n_ranking() == 670_790 and not pd.Index(ids).duplicated().any()
    assert np.all(ids == np.round(ids))
    return ids


def aligned(path: Path, cols: list[str], ids: np.ndarray, full: bool) -> pd.DataFrame:
    v = pd.read_parquet(path, columns=[ID, *cols])
    k = v[ID].to_numpy("float64")
    assert not pd.Index(k).duplicated().any(), f"{path}: duplicate ids"
    extra = set(k) - set(ids)
    assert not extra, f"{path}: {len(extra)} ids outside the template"
    if full:
        assert len(k) == len(ids), f"{path}: {len(k):,} rows, template {len(ids):,}"
    v = v.set_index(k).reindex(ids)
    out = pd.DataFrame({ID: ids})
    for c in cols:
        out[c] = v[c].to_numpy("float64") if c != "has_nm" else v[c].fillna(0).to_numpy("int8")
    return out


def write(root: Path, part: str, df: pd.DataFrame, inputs: dict, extra: dict) -> None:
    d = root / "finalroute" / "parts"
    pq, js = d / f"{part}.parquet", d / f"{part}.json"
    if pq.exists() or js.exists():
        raise SystemExit(f"{pq} exists; write-once")
    assert list(df.columns)[0] == ID and len(df) == 670_790
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq, index=False)
    vals = [c for c in df.columns if c != ID and df[c].dtype == "float64"]
    rec = {"part": part, "path": str(pq), "sha256": sha(pq), "rows": int(len(df)), "columns": list(df.columns),
           "nan": {c: int(df[c].isna().sum()) for c in vals},
           "below_floor": {c: int((df[c] < FLOOR).sum()) for c in vals},
           "by_month_mean": {c: {str(m): float(x) for m, x in df.groupby(month_of(df))[c].mean().items()}
                             for c in vals},
           "inputs": inputs, "code_sha256": sha(Path(__file__)), "ranking_mode": D.ranking_mode(),
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    js.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rec, indent=1, sort_keys=True), flush=True)


_MONTHS = None


def month_of(df: pd.DataFrame) -> np.ndarray:
    global _MONTHS
    if _MONTHS is None:
        rk = D.load_ranking(columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"])
        rk = rk[rk.PHASE_mvt == "DEP"]
        _MONTHS = pd.Series(rk.MVT_TIME_UTC_mvt.dt.month.to_numpy(), index=rk[ID].to_numpy("float64"))
    return _MONTHS.reindex(df[ID].to_numpy("float64")).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("member", "level2", "route"))
    ap.add_argument("--name")
    ap.add_argument("--src", type=Path)
    ap.add_argument("--r4mks", type=Path)
    ap.add_argument("--gbdt", type=Path, help="official-columns GBDT, MVT_ID_mvt + TAXITIME_SEC_mvt")
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("PRC_FINAL_ROOT", os.environ.get("PRC_FINAL_ROOT", "/path/to/final"))))
    a = ap.parse_args()
    ids = template()
    if a.kind == "member":
        cols = MEMBER_COLS[a.name]
        p = a.src / f"{a.name}-final.parquet" if a.src.is_dir() else a.src
        rec = json.loads(p.with_suffix(".json").read_text())
        df = aligned(p, cols, ids, full=a.name not in ("L", "Lp"))
        # the fit's thread count (desk 10-08 18:10 UTC: the released counts, recorded per member in MANIFEST)
        threads = rec.get("threads") or (rec.get("environment") or {}).get("threads") or \
            (rec.get("settings") or {}).get("threads")
        write(a.root, PART_NAME.get(a.name, a.name), df, {str(p): sha(p)},
              {"member": a.name, "member_threads": threads,
               "member_record": {k: rec.get(k) for k in ("fit", "owner", "rows", "scored_nan",
                                                         "output_sha256", "ranking_mode", "written_utc")}})
    elif a.kind == "level2":
        df = aligned(a.src, LEVEL2_COLS, ids, full=True)
        assert np.isfinite(df[["pred_Sb", "anchor_served"]].to_numpy()).all()
        write(a.root, "r4mks", df, {str(a.src): sha(a.src)}, {"stage": "R4mks", "owner": ("qfinal")})
    else:
        r = pd.read_parquet(a.r4mks)
        assert np.array_equal(r[ID].to_numpy("float64"), ids), "r4mks part is not in template order"
        sb = r["pred_Sb"].to_numpy("float64")
        bad = ~np.isfinite(sb)
        inputs = {str(a.r4mks): sha(a.r4mks)}
        val, src = sb.copy(), np.zeros(len(ids), "int8")
        if bad.any():
            if a.gbdt is None:
                raise SystemExit(f"{int(bad.sum())} rows without a finite R4mks value and no --gbdt")
            g = aligned(a.gbdt, ["TAXITIME_SEC_mvt"], ids, full=True)["TAXITIME_SEC_mvt"].to_numpy()
            val[bad], src[bad] = g[bad], 1
            inputs[str(a.gbdt)] = sha(a.gbdt)
        assert np.isfinite(val).all() and val.min() >= FLOOR - 1e-9
        extra = {"rows_by_source": {"R4mks": int((src == 0).sum()), "gbdt": int((src == 1).sum())},
                 "design": "exp-189 SPEC design row 1: v32 replaced by R4mks pred_Sb on every row"}
        write(a.root, "route", pd.DataFrame({ID: ids, "TAXITIME_SEC_mvt": val, "route_source": src}), inputs, extra)
        write(a.root, "F", pd.DataFrame({ID: ids, "F": val}), inputs, extra)
        write(a.root, "np_base", pd.DataFrame({ID: ids, "np_base": val}), inputs, extra)


if __name__ == "__main__":
    main()
