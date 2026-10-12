"""What the blend has to pass before its hash is quoted.

The desk's order: template order against `data/raw/submitting.parquet`, all
finite, min >= 60, the 388 `noac` and 38,182 EHAM rows bit-identical to v10,
every other row equal to `v10 + s*(stack-5 - v10)` at 1e-9, the changed-row
count and the SHA-256.

`rebuild()` does not import `serve.py`: it reads the record's two source files
by their recorded SHA-256, takes the row order from `submitting.parquet` and
does the arithmetic again (exp-030 Sol round 1, blocker 2 -- a rebuild that
shares its serving functions checks nothing).

    python verify.py --name exp-038-v10-stack5-blend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import regime_table as RT  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

DEST = ROOT / "submissions" / "candidates"
TOL = 1e-9


def column(rel: str, sha: str) -> np.ndarray:
    p = ROOT / rel
    got = D.sha256(p)
    if got != sha:
        raise SystemExit(f"{p.name}: sha256 {got[:8]} against the record's {sha[:8]}")
    d = pd.read_parquet(p)
    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    ids = d["MVT_ID_mvt"].to_numpy("float64")
    if np.array_equal(ids, tmpl):
        return d[D.TARGET].to_numpy("float64")
    v = d.set_index("MVT_ID_mvt")[D.TARGET].reindex(tmpl)
    if v.isna().any():
        raise SystemExit(f"{p.name}: {int(v.isna().sum())} template rows are missing")
    return v.to_numpy("float64")


def rebuild(rec: dict) -> np.ndarray:
    return (column(rec["base_file"], rec["base_sha256"])
            + rec["s"] * (column(rec["source_file"], rec["source_sha256"])
                          - column(rec["base_file"], rec["base_sha256"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp-038-v10-stack5-blend")
    a = ap.parse_args()
    rec = json.loads((EXP / "analysis" / f"served-{a.name}.json").read_text())
    path = DEST / f"{a.name}.parquet"
    d = pd.read_parquet(path)
    ids = d["MVT_ID_mvt"].to_numpy("float64")
    v = d[D.TARGET].to_numpy("float64")
    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    base = column(rec["base_file"], rec["base_sha256"])
    src = column(rec["source_file"], rec["source_sha256"])
    g = RT.load().set_index("MVT_ID_mvt")
    reg = g["regime"].astype(str).reindex(ids).to_numpy()
    apt = g["airport"].astype(str).reindex(ids).to_numpy()
    noac, eham = reg == "noac", apt == "EHAM"

    out, checks = {"name": a.name, "path": str(path.relative_to(ROOT)),
                   "s": rec["s"], "sha256": D.sha256(path)}, []

    def chk(label, ok, got):
        checks.append((label, bool(ok), got))
        out[label] = got

    chk("template_order", np.array_equal(ids, tmpl), int(len(ids)))
    chk("dtype_float64", d[D.TARGET].dtype == np.float64, str(d[D.TARGET].dtype))
    chk("all_finite", np.isfinite(v).all(), int((~np.isfinite(v)).sum()))
    chk("none_below_floor", (v >= B.FLOOR).all(), float(v.min()))
    chk("sha256_equals_record", out["sha256"] == rec["sha256"], out["sha256"][:8])
    chk("noac_identical_to_v10", np.abs(v[noac] - base[noac]).max() == 0.0,
        float(np.abs(v[noac] - base[noac]).max()))
    chk("eham_identical_to_v10", np.abs(v[eham] - base[eham]).max() == 0.0,
        float(np.abs(v[eham] - base[eham]).max()))
    # the two sources are themselves bit-identical on those rows, which is WHY
    # the blend cannot move them at any amplitude
    chk("sources_agree_on_noac", np.abs(src[noac] - base[noac]).max() == 0.0,
        float(np.abs(src[noac] - base[noac]).max()))
    chk("sources_agree_on_eham", np.abs(src[eham] - base[eham]).max() == 0.0,
        float(np.abs(src[eham] - base[eham]).max()))
    r = rebuild(rec)
    chk("rebuild_within_1e-9", np.abs(v - r).max() <= TOL, float(np.abs(v - r).max()))
    chk("every_row_is_the_rule", np.abs(v - (base + rec["s"] * (src - base))).max() <= TOL,
        float(np.abs(v - (base + rec["s"] * (src - base))).max()))
    chk("blend_is_between_its_sources",
        (np.minimum(base, src) - TOL <= v).all() and (v <= np.maximum(base, src) + TOL).all(),
        float(np.max(np.maximum(v - np.maximum(base, src), np.minimum(base, src) - v))))

    out.update({"n_noac": int(noac.sum()), "n_eham": int(eham.sum()),
                "n_moved_vs_v10": int((v != base).sum()),
                "n_unmoved_vs_v10": int((v == base).sum()),
                "rms_move_vs_v10": round(float(np.sqrt(((v - base) ** 2).mean())), 4),
                "max_abs_move_vs_v10": round(float(np.abs(v - base).max()), 4),
                "min": float(v.min()), "median": float(np.median(v)),
                "p99": float(np.quantile(v, 0.99)), "max": float(v.max()),
                "n_above_3600": int((v > 3600).sum())})
    for label, ok, got in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:28s} {got}")
    print(f"  ---- {out['n_moved_vs_v10']} of {len(v)} rows moved against v10 at "
          f"{out['rms_move_vs_v10']} s RMS, max |move| {out['max_abs_move_vs_v10']} s")
    print(f"  ---- sha256 {out['sha256']}")
    out["all_passed"] = all(ok for _, ok, _ in checks)
    p = EXP / "analysis" / f"verify-{a.name}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"  written: {p.relative_to(ROOT)}")
    if not out["all_passed"]:
        raise SystemExit("verification FAILED")


if __name__ == "__main__":
    main()
