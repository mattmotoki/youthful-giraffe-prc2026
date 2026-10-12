"""exp-148 SPEC-r3 section 3: the ranking vector of the chosen R2 variant.

    PRC_SOURCE_ROOT=... python serve_r2.py --final DIR/R2-final.parquet --score DIR/score-r2.json \
        --features features.parquet --v32 v32-exp137-proxy.parquet [--compare NAME=PATH ...] --out DIR

Reads the choice of `score-r2.json` (form, w, keep sets), R2-final through `pilot.load` (owner
`rfinal`), and serves w * form + (1 - w) * anchor_served, with v32's served value on the rows
of the chosen keep sets (noac: class noac; ordna: class ordinary and `AOBT_3_flt` missing in
`ranking.parquet`) and on the unsupported cells (`frozen/cells.csv`, as `rankvec.py`). Template
order, 344,841 rows, finite, at least 60 s. Write-once `<out>/ranking-<variant>.parquet` and
`.json` (row counts of every set, change against v32 and against each `--compare` vector).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("PRC_DATA_DIR", str(Path(os.environ["PRC_SOURCE_ROOT"]) / "data/raw"))
import stacker as ST                 # noqa: E402  (sets sys.path for prc and pilot)
from prc import data as D            # noqa: E402
from prc import gscore as GS         # noqa: E402
import pilot as P                    # noqa: E402

ID = ST.ID
N_RANKING = 344_841


def vector(path: Path) -> pd.Series:
    v = pd.read_parquet(path)
    return pd.Series(v["TAXITIME_SEC_mvt"].to_numpy("float64"), index=v[ID].astype("int64").to_numpy())


def change(d: np.ndarray) -> dict:
    return {"rms_s": float(np.sqrt(np.mean(d ** 2))), "mean_s": float(d.mean()),
            "max_abs_s": float(np.abs(d).max()), "rows_changed": int((d != 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--score", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--v32", type=Path, required=True)
    ap.add_argument("--compare", action="append", default=[], help="NAME=PATH of a ranking vector")
    ap.add_argument("--owner", default="rfinal", help="owner of --final (SPEC-r4: qfinal for R4)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ch = json.loads(a.score.read_text())["choice"]
    pq, js = a.out / f"ranking-{ch['variant']}.parquet", a.out / f"ranking-{ch['variant']}.json"
    if pq.exists() or js.exists():
        raise SystemExit(f"{pq} exists; write-once")
    tpl = pd.read_parquet(Path(os.environ["PRC_DATA_DIR"]) / "submitting.parquet", columns=[ID])
    ids = tpl[ID].to_numpy("float64")
    if len(ids) != N_RANKING or not np.all(ids == np.round(ids)) or pd.Index(ids).duplicated().any():
        raise SystemExit("template ids are not 344,841 unique integers")
    ids = ids.astype("int64")

    s = P.load(a.final, a.owner, what=a.final.stem).set_index(ID)
    s.index = s.index.astype("int64")
    v32 = vector(a.v32)
    if set(s.index) != set(ids) or set(v32.index) != set(ids):
        raise SystemExit("R2-final or v32 does not cover the template ids exactly")
    form = s[ch["form_column"]].reindex(ids).to_numpy("float64")
    anc = s["anchor_served"].reindex(ids).to_numpy("float64")
    blend = ch["w"] * form + (1 - ch["w"]) * anc

    f = pd.read_parquet(a.features, columns=[ID, "ctx_year", "ctx_month", "ctx_airport", "ctx_cls"])
    f = f[f.ctx_year == 2026].set_index(ID).reindex(ids)
    if f.ctx_cls.isna().any():
        raise SystemExit("a ranking row has no class")
    rk = D.load_ranking()
    rk = rk[rk.PHASE_mvt == "DEP"][[ID, "AOBT_3_flt"]]
    rk = rk.assign(**{ID: pd.to_numeric(rk[ID], errors="raise").astype("int64")}).set_index(ID).reindex(ids)
    cls = f.ctx_cls.astype(str).to_numpy()
    sets = {"noac": cls == "noac", "ordna": (cls == "ordinary") & rk.AOBT_3_flt.isna().to_numpy()}
    cells = GS.load_cells()
    unsup = {(int(m), str(p), str(c)) for m, p, c, sup in
             cells[["month", "airport", "cls", "supported"]].itertuples(index=False) if not sup}
    sets["unsupported"] = np.array([(int(m), str(p), str(c)) in unsup
                                    for m, p, c in zip(f.ctx_month, f.ctx_airport, cls)])
    keep = sets["unsupported"].copy()
    for k in ch["keep_sets"]:
        keep |= sets[k]
    old = v32.reindex(ids).to_numpy()
    out = np.where(keep, old, blend)
    if not np.isfinite(out).all() or out.min() < ST.FLOOR - 1e-9:
        raise SystemExit(f"ranking vector not finite or below {ST.FLOOR} s")
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: tpl[ID].to_numpy(), "TAXITIME_SEC_mvt": out}).to_parquet(pq, index=False)
    rec = {"what": f"exp-148 SPEC-r3 ranking vector {ch['variant']}", "choice": ch,
           "rows": int(len(out)), "rows_by_set": {k: int(v.sum()) for k, v in sets.items()},
           "rows_from_v32": int(keep.sum()), "min": float(out.min()), "max": float(out.max()),
           "mean": float(out.mean()), "vs_v32": change(out - old), "vs": {},
           "inputs_sha256": {"final": ST.sha(a.final), "score": ST.sha(a.score), "features": ST.sha(a.features),
                             "v32": ST.sha(a.v32)},
           "code_sha256": {"serve_r2.py": ST.sha(Path(__file__)), "stacker.py": ST.sha(HERE / "stacker.py")}}
    for c in a.compare:
        name, path = c.split("=", 1)
        v = vector(Path(path))
        if set(v.index) != set(ids):
            raise SystemExit(f"{name} does not cover the template ids")
        rec["vs"][name] = {"sha256": ST.sha(Path(path)), **change(out - v.reindex(ids).to_numpy())}
    rec["output_sha256"] = ST.sha(pq)
    js.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    print(js.read_text())


if __name__ == "__main__":
    main()
