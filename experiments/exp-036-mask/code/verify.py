"""What a masked candidate has to pass before its hash is quoted.

`assignments/mask.md`: template order, finite, none below the floor, the 388
`noac` and 38,182 EHAM rows bit identical to v7, **every row equal to one of its
two sources**, and a rebuild from its own record at a maximum absolute difference
of 0.0.  All eleven checks are here and each one raises.

The rebuild is genuinely independent of the serving path in the one way that
matters for a derived candidate: `rebuild()` reads the record's two source files
by their recorded SHA-256, reads the airport column from the recorded gates
table, and takes the row order from `submitting.parquet` -- it does not import
`serve.py` and there is no model, no frame and no block between the bytes and the
answer (exp-030 Sol round 1, blocker 2, which caught a rebuild sharing three
functions with its serve).  What is still shared is `prc.data`'s parquet reading.

    python verify.py --name exp-036-refitbulk-mask
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
S5 = DEST / "exp-026-stack-5.parquet"


def column(p: Path, sha: str | None = None) -> np.ndarray:
    if sha is not None:
        got = D.sha256(p)
        if got != sha:
            raise SystemExit(f"{p.name}: sha256 {got[:8]} against the record's {sha[:8]}")
    return pd.read_parquet(p)[D.TARGET].to_numpy("float64")


def rebuild(rec: dict) -> np.ndarray:
    """The candidate from its record alone: two hashed files and an airport list."""
    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    gt = ROOT / rec["gates_table"]
    if D.sha256(gt) != rec["gates_table_sha256"]:
        raise SystemExit("the gates table has changed since the serve")
    apt = pd.read_parquet(gt).set_index("MVT_ID_mvt")["airport"].astype(str) \
        .reindex(tmpl).to_numpy()
    src = column(ROOT / rec["source_file"], rec["source_sha256"])
    base = column(ROOT / rec["base_file"], rec["base_sha256"])
    return np.where(np.isin(apt, rec["keep_airports"]), src, base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    recp = EXP / "analysis" / f"served-{a.name}.json"
    rec = json.loads(recp.read_text())
    path = DEST / f"{a.name}.parquet"
    d = pd.read_parquet(path)
    ids = d["MVT_ID_mvt"].to_numpy("float64")
    v = d[D.TARGET].to_numpy("float64")
    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    base = column(ROOT / rec["base_file"], rec["base_sha256"])
    src = column(ROOT / rec["source_file"], rec["source_sha256"])
    s5 = column(S5)
    g = RT.load().set_index("MVT_ID_mvt")
    reg = g["regime"].astype(str).reindex(ids).to_numpy()
    apt = g["airport"].astype(str).reindex(ids).to_numpy()
    noac, eham = reg == "noac", apt == "EHAM"
    on = np.isin(apt, rec["keep_airports"])

    out, checks = {"name": a.name, "path": str(path.relative_to(ROOT)),
                   "sha256": D.sha256(path)}, []

    def chk(label, ok, got):
        checks.append((label, bool(ok), got))
        out[label] = got

    chk("template_order", np.array_equal(ids, tmpl), int(len(ids)))
    chk("dtype_float64", d[D.TARGET].dtype == np.float64, str(d[D.TARGET].dtype))
    chk("all_finite", np.isfinite(v).all(), int((~np.isfinite(v)).sum()))
    chk("none_below_floor", (v >= B.FLOOR).all(), int((v < B.FLOOR).sum()))
    chk("sha256_equals_record", out["sha256"] == rec["sha256"], out["sha256"][:8])
    chk("noac_identical_to_v7", np.abs(v[noac] - base[noac]).max() == 0.0,
        float(np.abs(v[noac] - base[noac]).max()))
    chk("noac_identical_to_stack5", np.abs(v[noac] - s5[noac]).max() == 0.0,
        float(np.abs(v[noac] - s5[noac]).max()))
    chk("eham_identical_to_v7", np.abs(v[eham] - base[eham]).max() == 0.0,
        float(np.abs(v[eham] - base[eham]).max()))
    chk("eham_identical_to_stack5", np.abs(v[eham] - s5[eham]).max() == 0.0,
        float(np.abs(v[eham] - s5[eham]).max()))
    # every row is one of the two sources, and it is the one the mask names
    near = np.minimum(np.abs(v - src), np.abs(v - base))
    chk("every_row_from_a_source", near.max() == 0.0, float(near.max()))
    chk("held_rows_are_the_base", np.abs(v[~on] - base[~on]).max() == 0.0,
        float(np.abs(v[~on] - base[~on]).max()))
    chk("served_rows_are_the_source", np.abs(v[on] - src[on]).max() == 0.0,
        float(np.abs(v[on] - src[on]).max()))
    r = rebuild(rec)
    chk("rebuild_max_abs_diff_zero", np.abs(v - r).max() == 0.0, float(np.abs(v - r).max()))

    out.update({"n_noac": int(noac.sum()), "n_eham": int(eham.sum()),
                "n_from_source": int(on.sum()), "n_from_base": int((~on).sum()),
                "n_moved_vs_v7": int((v != base).sum()),
                "n_moved_vs_source": int((v != src).sum()),
                "rms_move_vs_v7": round(float(np.sqrt(((v - base) ** 2).mean())), 4),
                "rms_move_vs_source": round(float(np.sqrt(((v - src) ** 2).mean())), 4),
                "min": float(v.min()), "median": float(np.median(v)),
                "p99": float(np.quantile(v, 0.99)), "max": float(v.max()),
                "n_above_3600": int((v > 3600).sum())})
    for label, ok, got in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:28s} {got}")
    print(f"  ---- {out['n_from_source']} from {rec['source']}, {out['n_from_base']} "
          f"from v7; {out['n_moved_vs_v7']} moved against v7 at "
          f"{out['rms_move_vs_v7']} s RMS")
    print(f"  ---- sha256 {out['sha256']}")
    out["all_passed"] = all(ok for _, ok, _ in checks)
    p = EXP / "analysis" / f"verify-{a.name}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"  written: {p.relative_to(ROOT)}")
    if not out["all_passed"]:
        raise SystemExit("verification FAILED")


if __name__ == "__main__":
    main()
