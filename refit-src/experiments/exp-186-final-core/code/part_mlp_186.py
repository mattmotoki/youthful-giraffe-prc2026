"""exp-186: part `mlp` (finalplan's form) from the APPLY outputs of the MLP008 and mix12 models.

Columns: `MVT_ID_mvt` (float64, as the template), `mlp008` (MLP008 `final`, mean of seeds 701/702;
exp-164 apply_final186.py) and `mix12` (mlp-010 e80 x6 + drop0 x6, 0.5/0.5; apply_mix12_186.py), float64,
in the order of load_submitting() of the mode. The assembly takes mix12 on member-scope clock rows and mlp008
on every other row (assemble192.py:160-167). The mlp has no value on its ghost rows (AIRCRAFT_TYPE_mvt null;
the `noac` class of the bundle), so both columns must be finite on every departure with an aircraft type and
NaN on the others. The member scope needs a type, so this covers it; finalplan checks the exact scope
(assemble192.py:166). Final mode reads out/mlp-*-final186/final, ranking mode out/mlp-*-replay186/final.
Writes <out>/mlp.parquet and <out>/mlp.json (write-once); the JSON carries the MANIFEST row.

    PRC_RANKING_MODE=final EXP164_WORK=<work root> python experiments/exp-186-final-core/code/part_mlp_186.py \
        --out $PRC_FINAL_ROOT/finalcore/parts --commit <snapshot commit>
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
from prc import data as D  # noqa: E402

ID = "MVT_ID_mvt"
SOURCES = {"mlp008": "out/mlp-008-{tag}/final", "mix12": "out/mlp-012-{tag}/final"}


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def vector(work: Path, col: str, mode: str):
    d = work / SOURCES[col].format(tag="replay186" if mode == "ranking" else "final186")
    rec = json.loads((d / "APPLY.json").read_text())
    assert rec["fit"] == "final" and rec["ranking_mode"] == mode, (col, rec["fit"], rec["ranking_mode"])
    art = rec["prediction_ranking"]
    p = d / "mean-ranking.parquet"
    assert Path(art["path"]).resolve() == p.resolve() and sha(p) == art["sha256"], f"{col}: APPLY prediction pin"
    v = pd.read_parquet(p, columns=[ID, "prediction"])
    ids = v[ID].to_numpy("int64")
    assert pd.Index(ids).is_unique, f"{col}: repeated ids"
    x = v.prediction.to_numpy("float64")
    assert np.isfinite(x).all(), f"{col}: nonfinite prediction"
    src = {"apply": {"path": str(d / "APPLY.json"), "sha256": sha(d / "APPLY.json")},
           "vector": {"path": str(p), "sha256": art["sha256"]}, "rows": int(len(v))}
    return pd.Series(x, index=ids), src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="parts folder ($PRC_FINAL_ROOT/finalcore/parts)")
    ap.add_argument("--commit", required=True, help="code commit of the snapshot")
    ap.add_argument("--work", type=Path, default=os.environ.get("EXP164_WORK") or None, help="default $EXP164_WORK")
    ap.add_argument("--rebuild-tolerance", default="", help="MANIFEST rebuild_tolerance text")
    a = ap.parse_args()
    assert a.work is not None and a.work.is_dir(), "give --work or EXP164_WORK"
    pq, js = a.out / "mlp.parquet", a.out / "mlp.json"
    assert not pq.exists() and not js.exists(), f"{pq}: write-once"
    mode = D.ranking_mode()
    t0 = time.time()

    tmpl = D.load_submitting()[ID].to_numpy("float64")
    ids = tmpl.astype("int64")
    assert len(tmpl) == D.n_ranking() and np.array_equal(ids, tmpl) and pd.Index(ids).is_unique
    rk = D.load_ranking(columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt"])
    dep = rk[rk.PHASE_mvt == "DEP"].copy()
    dep[ID] = dep[ID].astype("int64")
    assert dep[ID].is_unique and np.isin(ids, dep[ID].to_numpy()).all(), "template ids outside the departures"
    dep = dep.set_index(ID).reindex(ids)
    typed = dep.AIRCRAFT_TYPE_mvt.notna().to_numpy()
    month = dep.month.to_numpy()

    part = pd.DataFrame({ID: tmpl})
    inputs = {}
    for col in SOURCES:
        s, src = vector(a.work, col, mode)
        assert set(s.index) <= set(ids), f"{col}: ids outside the template"
        x = s.reindex(ids).to_numpy("float64")
        fin = np.isfinite(x)
        assert fin[typed].all(), f"{col}: {int((~fin & typed).sum())} typed rows without a value"
        assert not fin[~typed].any(), f"{col}: {int((fin & ~typed).sum())} untyped rows with a value"
        part[col] = x
        inputs[col] = src
    assert list(part.columns) == [ID, "mlp008", "mix12"] and (part.dtypes == "float64").all()
    # upstream parts: the inputs both APPLY records name (Lp part, frame12/track12, 2026 month frames)
    inp = a.work / ("inputs/mlp-008/final" if mode == "ranking" else "inputs/mlp-008-final/final") / "INPUTS.json"
    applied = {json.loads(Path(inputs[c]["apply"]["path"]).read_text())["inputs_sha256"] for c in SOURCES}
    assert applied == {sha(inp)}, f"APPLY records name other inputs than {inp}"
    m = json.loads(inp.read_text())
    upstream = {"inputs": {"path": str(inp), "sha256": sha(inp)},
                "lp_part": m.get("lp_part") or m.get("id_contracts", {}).get("jjrank"),
                "A3": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in m.get("A3_sources", {}).items()},
                "months_2026": {p: d for p, d in m.get("source_files", {}).items() if "2026-" in Path(p).name}}
    a.out.mkdir(parents=True, exist_ok=True)
    part.to_parquet(pq, index=False)
    digest = sha(pq)
    by_month = {m: {"rows": int((month == m).sum()), "typed": int((typed & (month == m)).sum())} for m in sorted(set(month))}
    command = " ".join(["python", str(Path(__file__).resolve().relative_to(ROOT)), *sys.argv[1:]])
    row = {"part": "mlp", "remote_path": str(pq), "sha256": digest, "rows": int(len(part)),
           "columns": ",".join(part.columns), "code_commit": a.commit,
           "input_parts": ";".join([f"{c}:{inputs[c]['vector']['path']}:{inputs[c]['vector']['sha256']}" for c in SOURCES]
                                   + ([f"lp:{upstream['lp_part']['path']}:{upstream['lp_part']['sha256']}"] if upstream["lp_part"] else [])
                                   + [f"{k}:{v['path']}:{v['sha256']}" for k, v in upstream["A3"].items()]
                                   + [f"month:{p}:{d}" for p, d in upstream["months_2026"].items()]),
           "command": command, "rebuild_tolerance": a.rebuild_tolerance,
           "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rec = {"part": "mlp", "path": str(pq), "sha256": digest, "rows": int(len(part)), "columns": list(part.columns),
           "ranking_mode": mode, "template": {"path": str(D.submitting_file()), "sha256": sha(D.submitting_file())},
           "ranking_file": {"path": str(D.ranking_file()), "sha256": sha(D.ranking_file())},
           "typed_rows": int(typed.sum()), "untyped_rows": int((~typed).sum()), "rows_by_month": by_month,
           "rule": "both columns finite on every departure with AIRCRAFT_TYPE_mvt, NaN on the others",
           "inputs": inputs, "upstream": upstream, "work": str(a.work), "code_sha256": sha(__file__), "manifest_row": row,
           "seconds": round(time.time() - t0, 1)}
    js.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"event": "part-ready", "part": "mlp", "sha256": digest, "rows": len(part),
                      "typed": int(typed.sum()), "untyped": int((~typed).sum())}))
    print("\t".join(str(row[k]) for k in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
