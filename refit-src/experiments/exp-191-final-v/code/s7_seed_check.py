"""PORT-S7X9 checks 3-4 on new s7 seed files in an s7 tree, and a row-order read by content.

The fit directory of the new seeds is outside my read scope (lead rule), so its ids.npy cannot be compared. Two
reads stand in for check 1: the NaN pattern of each file against seed 1 of the same mode, and the per-row distance
of each file to the seeds 1-3 mean (a file in another row order is hundreds of seconds away). Rows: held fits on the
2025 January/July rows (the rows their fold vector serves), full fits on the 2026 rows by month. Seed 1 against the
mean of seeds 2-3 is the scale. No pass line; the asserts fail closed after the record is written.

    python s7_seed_check.py <s7 tree> <out.json> <seed> [<seed> ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

tree, out, seeds = Path(sys.argv[1]), Path(sys.argv[2]), [int(s) for s in sys.argv[3:]]
assert not out.exists(), f"{out} exists (write once)"
D = tree / "work/seq7"
is26 = np.load(D / "is2026.npy"); month = np.load(D / "month.npy")
N = len(is26)
assert N == 2755837 and month.shape == (N,)
rows = {"held": {m: ~is26 & (month == m) for m in (1, 7)},
        "full": {m: is26 & (month == m) for m in (1, 2, 6, 7)}}
for m in (1, 2, 6, 7):
    assert rows["full"][m].sum() > 0
fails = []


def dist(a, ref, sel):
    ok = sel & np.isfinite(a) & np.isfinite(ref)
    e = a[ok].astype(float) - ref[ok]
    return {"rows": int(ok.sum()), "rms": round(float(np.sqrt((e ** 2).mean())), 2), "bias": round(float(e.mean()), 2),
            "corr": round(float(np.corrcoef(a[ok], ref[ok])[0, 1]), 5)}


rec = {"tree": str(tree), "rows": N, "modes": {}}
for mode in ("held", "full"):
    ref = [np.load(D / f"s7-{mode}-{s}.npy") for s in (1, 2, 3)]
    mean3 = np.mean([r.astype(float) for r in ref], 0)
    mean23 = np.mean([r.astype(float) for r in ref[1:]], 0)
    r = {"scale_seed1_vs_mean23": {str(m): dist(ref[0], mean23, sel) for m, sel in rows[mode].items()}, "seeds": {}}
    for s in seeds:
        p = D / f"s7-{mode}-{s}.npy"
        a = np.load(p)
        fin = np.isfinite(a)
        x = {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "dtype": str(a.dtype),
             "shape": list(a.shape), "finite": int(fin.sum()), "nan_pattern_equals_seed1": bool(np.array_equal(~fin, np.isnan(ref[0]))),
             "finite_min": float(a[fin].min()), "vs_mean123": {str(m): dist(a, mean3, sel) for m, sel in rows[mode].items()}}
        r["seeds"][str(s)] = x
        if x["dtype"] != "float32" or x["shape"] != [N] or not x["nan_pattern_equals_seed1"] or x["finite_min"] < 60:
            fails.append(f"{mode} {s}")
    rec["modes"][mode] = r
rec["fails"] = fails
out.write_text(json.dumps(rec, indent=1))
for mode, r in rec["modes"].items():
    print(mode, "scale (seed 1 vs mean 2-3) rms", {m: v["rms"] for m, v in r["scale_seed1_vs_mean23"].items()})
    for s, x in r["seeds"].items():
        print(f"  {mode} {s}: {x['dtype']} {x['shape']} finite {x['finite']} nan=seed1 {x['nan_pattern_equals_seed1']} "
              f"min {x['finite_min']:.1f} vs mean 1-3 rms {({m: v['rms'] for m, v in x['vs_mean123'].items()})} "
              f"corr {({m: v['corr'] for m, v in x['vs_mean123'].items()})}")
assert not fails, fails
