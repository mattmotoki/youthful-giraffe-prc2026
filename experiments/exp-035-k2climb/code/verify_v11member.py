"""What the tail rule candidate has to pass before its hash is quoted.

The desk's list, 13:50 UTC: template order, finite, `min >= 60`, the 388 `noac`,
the EHAM and the EDDF rows bit-identical to **v11**, rows moved and RMS against
v11, the fold-side/served update RMS ratio, the SHA-256, and the rebuild from the
record's hashes.  Two more are added because they are free and they are what a
composition claim means: every input's hash re-checked, and the standing gate
that K2's own head through the same two lines returns v10 and v11 at 0.

This file reads the RECORD and the bytes, never the builder's memory.

    python verify_v11member.py --name exp-035-k2climb-v11member
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


def column(rel: str, sha: str, tmpl: np.ndarray) -> np.ndarray:
    p = ROOT / rel
    got = D.sha256(p)
    if got != sha:
        raise SystemExit(f"{p.name}: sha256 {got[:8]} against the record's {sha[:8]}")
    d = pd.read_parquet(p)
    ids = d["MVT_ID_mvt"].to_numpy("float64")
    if np.array_equal(ids, tmpl):
        return d[D.TARGET].to_numpy("float64")
    v = d.set_index("MVT_ID_mvt")[D.TARGET].reindex(tmpl)
    if v.isna().any():
        raise SystemExit(f"{p.name}: {int(v.isna().sum())} template rows are missing")
    return v.to_numpy("float64")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp-035-k2climb-v11member")
    a = ap.parse_args()
    rec = json.loads((EXP / "analysis" / f"served-{a.name}.json").read_text())
    path = DEST / f"{a.name}.parquet"
    d = pd.read_parquet(path)
    ids = d["MVT_ID_mvt"].to_numpy("float64")
    v = d[D.TARGET].to_numpy("float64")
    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")

    if D.sha256(ROOT / rec["gates_table"]) != rec["gates_table_sha256"]:
        raise SystemExit("the gates table has changed since the serve")
    g = pd.read_parquet(ROOT / rec["gates_table"]).set_index("MVT_ID_mvt")
    apt = g["airport"].astype(str).reindex(tmpl).to_numpy()
    reg = g["regime"].astype(str).reindex(tmpl).to_numpy()
    on = np.isin(apt, rec["keep_airports"])
    noac, eham, eddf = reg == "noac", apt == "EHAM", apt == "EDDF"

    v7 = column(rec["v7_file"], rec["v7_sha256"], tmpl)
    s5 = column(rec["stack5_file"], rec["stack5_sha256"], tmpl)
    v11 = column(rec["v11_file"], rec["v11_sha256"], tmpl)
    head = column(rec["head_file"], rec["head_sha256"], tmpl)
    k2 = column(rec["k2_head_file"], rec["k2_head_sha256"], tmpl)
    v10 = column(rec["v10_file"], rec["v10_sha256"], tmpl)

    def compose(h):
        m = np.where(on, h, v7)
        return m, m + rec["s"] * (s5 - m)

    _, rebuilt = compose(head)
    m_k2, b_k2 = compose(k2)

    out, checks = {"name": a.name, "path": str(path.relative_to(ROOT)),
                   "sha256": D.sha256(path)}, []

    def chk(label, ok, got):
        checks.append((label, bool(ok), got))
        out[label] = got

    chk("template_order", np.array_equal(ids, tmpl), int(len(ids)))
    chk("dtype_float64", d[D.TARGET].dtype == np.float64, str(d[D.TARGET].dtype))
    chk("all_finite", np.isfinite(v).all(), int((~np.isfinite(v)).sum()))
    chk("none_below_floor", (v >= B.FLOOR).all(), int((v < B.FLOOR).sum()))
    chk("min_at_least_60", v.min() >= 60.0, float(v.min()))
    chk("sha256_equals_record", out["sha256"] == rec["sha256"], out["sha256"][:8])
    chk("noac_identical_to_v11", np.abs(v[noac] - v11[noac]).max() == 0.0,
        float(np.abs(v[noac] - v11[noac]).max()))
    chk("eham_identical_to_v11", np.abs(v[eham] - v11[eham]).max() == 0.0,
        float(np.abs(v[eham] - v11[eham]).max()))
    chk("eddf_identical_to_v11", np.abs(v[eddf] - v11[eddf]).max() == 0.0,
        float(np.abs(v[eddf] - v11[eddf]).max()))
    chk("held_rows_are_v11", np.abs(v[~on] - v11[~on]).max() == 0.0,
        float(np.abs(v[~on] - v11[~on]).max()))
    chk("k2_head_returns_v10", np.abs(m_k2 - v10).max() == 0.0,
        float(np.abs(m_k2 - v10).max()))
    chk("k2_head_returns_v11", np.abs(b_k2 - v11).max() == 0.0,
        float(np.abs(b_k2 - v11).max()))
    chk("rebuild_max_abs_diff_zero", np.abs(v - rebuilt).max() == 0.0,
        float(np.abs(v - rebuilt).max()))
    # the served update must be the size the fold side measured (exp-040's rule)
    served_rms = float(np.sqrt(((v - v11) ** 2).mean()))
    fold_mean = float(np.mean(list(rec["fold_update_rms"].values())))
    ratio = served_rms / fold_mean
    chk("update_rms_ratio_within_one_third", abs(ratio - 1.0) <= 1 / 3, round(ratio, 4))

    # a COUNT, not a check: exact ties on served rows are legitimate (the update
    # can be zero to the last bit), and an earlier version of this file made it a
    # check whose PASS meant "some rows did not move", which is backwards.
    out.update({"n_served_rows_identical_to_v11": int((v[on] == v11[on]).sum()),
                "n_noac": int(noac.sum()), "n_eham": int(eham.sum()),
                "n_eddf": int(eddf.sum()), "n_from_head": int(on.sum()),
                "n_from_v7": int((~on).sum()),
                "n_moved_vs_v11": int((v != v11).sum()),
                "rms_move_vs_v11": round(served_rms, 4),
                "max_abs_move_vs_v11": round(float(np.abs(v - v11).max()), 4),
                "fold_update_rms": rec["fold_update_rms"],
                "served_over_fold_rms_ratio": round(ratio, 4),
                "min": float(v.min()), "median": float(np.median(v)),
                "p99": float(np.quantile(v, 0.99)), "max": float(v.max()),
                "v11_min": float(v11.min()), "v11_median": float(np.median(v11)),
                "v11_p99": float(np.quantile(v11, 0.99)), "v11_max": float(v11.max()),
                "n_above_3600": int((v > 3600).sum()),
                "v11_n_above_3600": int((v11 > 3600).sum())})
    for label, ok, got in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:34s} {got}")
    print(f"  ---- {out['n_served_rows_identical_to_v11']} served rows are exact ties "
          f"with v11 (the update is zero to the last bit there)")
    print(f"  ---- {out['n_from_head']} from the climbed head, {out['n_from_v7']} from v7; "
          f"{out['n_moved_vs_v11']} moved against v11 at {out['rms_move_vs_v11']} s RMS "
          f"(max {out['max_abs_move_vs_v11']})")
    print(f"  ---- min {out['min']:.4f} (v11 {out['v11_min']:.4f})  median {out['median']:.2f} "
          f"({out['v11_median']:.2f})  p99 {out['p99']:.1f} ({out['v11_p99']:.1f})  "
          f"max {out['max']:.1f} ({out['v11_max']:.1f})")
    print(f"  ---- rows above 3600 s: {out['n_above_3600']} (v11 {out['v11_n_above_3600']})")
    print(f"  ---- sha256 {out['sha256']}")
    out["all_passed"] = all(ok for _, ok, _ in checks)
    p = EXP / "analysis" / f"verify-{a.name}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"  {'ALL CHECKS PASS' if out['all_passed'] else 'SOME CHECKS FAILED'}; "
          f"written {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
