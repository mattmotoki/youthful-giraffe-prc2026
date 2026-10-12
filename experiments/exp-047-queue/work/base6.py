"""§7bk: the six-member base — the three folds and 2025-11 — without V3-allin.

Registered in RESULTS.md §7bk before this ran.  The vector is **base6**: not v11,
not v12, a different estimator from the served head because a member is missing
AND the remaining six carry refitted weights.

The composition is `exp-026-rebuild/code/stack5.py`'s, called rather than
paraphrased: `rule.fit_hybrid` / `rule.apply_hybrid` for the per-regime hybrid
NNLS anchor, `stack5.arm` for the stacker over `stack5.SEEDS`, `keep-noac`, then
`prc.blend`'s floor and residual guard.  Nothing in exp-026 is edited or written.

**Step 0 validates the instrument on the SEVEN-member set** and refuses to go on
unless the rotated anchor reproduces `stack5.PUBLISHED` on all three folds to
1e-3 — the same gate `stack5.py` puts on itself.  Only then is V3-allin dropped.
A six-member run that skipped this could not tell a real effect from a wiring bug.

**The fitting is different for a fold and for 2025-11, by construction** (§7bk):

* a fold `f` is **rotated** — anchor and stacker fitted on the other two folds,
  exactly as `stack5.py` does it, so `base6 − pbase_true` prices the missing
  member and the refit rather than a rotation change;
* **2025-11** is outside every fold, so it takes the serve-side form of
  `serve_stack5.build`: weights fitted on all three folds, the stacker fitted on
  the three rotated anchors.

    OMP_NUM_THREADS=2 PRC_THREADS=2 .venv/bin/python work/base6.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
sys.path.insert(0, str(ROOT))
import frames as F                                           # noqa: E402
import rule as R                                             # noqa: E402
import stack5 as S                                           # noqa: E402
import prc.blend as B                                        # noqa: E402
import prc.data as D                                         # noqa: E402

OOF = EXP / "oof"
sys.path.insert(0, str(HERE))
import lane                                                  # noqa: E402
lane.install()
MONTH = lane.MONTH
SUF = "" if MONTH == "2025-11" else f"-{MONTH}"   # §7bk keeps its names
OUT = EXP / f"analysis/base6{SUF}.json"
BLOCK = ROOT / f"artifacts/exp-047/block/{MONTH}.parquet"
CONTRACT = ["MVT_ID_mvt", "fold", "y_true", "y_pred"]
NAMES7 = S.B5
NAMES6 = tuple(n for n in S.B5 if n != "V3-allin")
#: this lane's november_loo vectors, by the B5 lineage name.
NOV = {"T-tailv4": "T-tailv2", "C2-cat": "C2-cat", "L2-hub8725": "L2-hub",
       "D1-winter": "D1-winter", "F1-fresh": "F1-fresh", "N1-nmtail": "N1-nmtail"}


def rotation(data: dict, names) -> dict:
    """`stack5.py` step 1: for each fold, the anchor fitted on the other two."""
    return {g: R.apply_hybrid(data[g], names,
                              R.fit_hybrid(pd.concat([data[h] for h in F.FOLDS if h != g],
                                                     ignore_index=True), names))
            for g in F.FOLDS}


def november_frame(names) -> pd.DataFrame:
    """The six members over 2025-11, prepared exactly as `F.fold_frame` prepares a fold."""
    base = None
    for n in names:
        p = OOF / f"{NOV[n]}--{lane.FOLD}.parquet"
        d = pd.read_parquet(p)[["MVT_ID_mvt", "y_true", "y_pred"]].rename(
            columns={"y_pred": n})
        d["MVT_ID_mvt"] = d["MVT_ID_mvt"].astype("float64")
        how = "left" if n in F.PARTIAL else "inner"
        base = d if base is None else base.merge(d[["MVT_ID_mvt", n]],
                                                 on="MVT_ID_mvt", how=how)
    raw = D.load_month(MONTH)
    masked, _ = D.mask_like_ranking(raw)
    r = masked[masked["PHASE_mvt"] == "DEP"].copy()
    r["regime"] = B.regimes_from_raw(r, B.LATE_CUT)
    r["d_takeoff_aobt"] = (r["MVT_TIME_UTC_mvt"] - r["AOBT_3_flt"]).dt.total_seconds()
    r["airport"] = r["ADEP_mvt"].astype(str)
    r["MVT_ID_mvt"] = r["MVT_ID_mvt"].astype("float64")
    g = r[["MVT_ID_mvt", "regime", "d_takeoff_aobt", "airport"]]
    m = base.merge(g, on="MVT_ID_mvt", how="inner")
    if len(m) != len(base):
        raise SystemExit(f"{MONTH}: gates cover {len(m)} of {len(base)} member rows")
    a = m["d_takeoff_aobt"].to_numpy("float64")
    for n in names:                       # floor, then the residual guard
        v = m[n].to_numpy("float64")
        ok = np.isfinite(v)
        v[ok] = np.maximum(v[ok], B.FLOOR)
        v[ok] = B.apply_residual_guard(v[ok], a[ok], *F.GUARD, B.FLOOR)
        m[n] = v
    return m


def compose(fit: pd.DataFrame, ap: pd.DataFrame, a_fit, a_ap, names) -> np.ndarray:
    """The stacker bag and the keep-noac variant, `stack5.py` step 2."""
    bag = np.maximum(np.mean([S.arm(fit, ap, s, a_fit, a_ap, names) for s in S.SEEDS],
                             axis=0), B.FLOOR)
    ref = np.maximum(a_ap, B.FLOOR)
    keep = np.where(ap["regime"].to_numpy() == "noac", ref, bag)
    a = ap["d_takeoff_aobt"].to_numpy("float64")
    return np.maximum(B.apply_residual_guard(np.maximum(keep, B.FLOOR), a, *F.GUARD,
                                             B.FLOOR), B.FLOOR)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, "float64") - np.asarray(p, "float64")) ** 2)))


def write(ids, y, pred, fold: str, tag: str) -> dict:
    d = pd.DataFrame({"MVT_ID_mvt": np.asarray(ids, "float64"), "fold": fold,
                      "y_true": np.asarray(y, "float64"),
                      "y_pred": np.asarray(pred, "float64")})[CONTRACT]
    p = OOF / f"base6--{tag}.parquet"
    d.to_parquet(p, index=False)
    return {"path": str(p.relative_to(ROOT)), "rows": int(len(d)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def main() -> None:
    t0 = time.time()
    rec = {"what": "§7bk: the six-member base, three folds and 2025-11",
           "names7": list(NAMES7), "names6": list(NAMES6),
           "excluded": "V3-allin", "seeds": list(S.SEEDS), "rounds": S.ROUNDS}

    print("=== step 0: the SEVEN-member instrument must reproduce stack5.PUBLISHED",
          flush=True)
    d7 = {f: F.fold_frame(f, NAMES7) for f in F.FOLDS}
    a7 = rotation(d7, NAMES7)
    rec["instrument"] = {}
    for f in F.FOLDS:
        got = rmse(d7[f]["y_true"].to_numpy("float64"), a7[f])
        delta = got - S.PUBLISHED[f]
        rec["instrument"][f] = {"anchor_rmse": round(got, 4),
                                "published": S.PUBLISHED[f], "delta": round(delta, 6)}
        print(f"    {f:9s} {got:9.4f}  published {S.PUBLISHED[f]:.4f}  "
              f"delta {delta:+.6f}", flush=True)
        if abs(delta) > 1e-3:
            raise SystemExit(f"the instrument does not reproduce on {f}: {delta:+.4f}; "
                             "nothing is composed")
    print("    instrument OK", flush=True)

    print("\n=== step 1: the six-member rotation", flush=True)
    d6 = {f: F.fold_frame(f, NAMES6) for f in F.FOLDS}
    a6 = rotation(d6, NAMES6)
    rec["anchor6"] = {}
    for f in F.FOLDS:
        g6 = rmse(d6[f]["y_true"].to_numpy("float64"), a6[f])
        g7 = rmse(d7[f]["y_true"].to_numpy("float64"), a7[f])
        rec["anchor6"][f] = {"six": round(g6, 4), "seven": round(g7, 4),
                             "six_minus_seven": round(g6 - g7, 4)}
        print(f"    anchor {f:9s} six {g6:9.4f}  seven {g7:9.4f}  "
              f"{g6-g7:+.4f}", flush=True)

    print("\n=== step 2: base6 on the three folds (rotated)", flush=True)
    rec["folds"] = {}
    for f in F.FOLDS:
        ap = d6[f]
        fit = pd.concat([d6[g] for g in F.FOLDS if g != f], ignore_index=True)
        a_fit = np.concatenate([np.maximum(a6[g], B.FLOOR) for g in F.FOLDS if g != f])
        pred = compose(fit, ap, a_fit, a6[f], NAMES6)
        y = ap["y_true"].to_numpy("float64")
        v = write(ap["MVT_ID_mvt"].to_numpy("float64"), y, pred, f, f)
        v["rmse_base6"] = round(rmse(y, pred), 4)
        v["rmse_anchor6"] = round(rmse(y, np.maximum(a6[f], B.FLOOR)), 4)
        rec["folds"][f] = v
        print(f"    {f:9s} rmse {v['rmse_base6']:9.4f}  rows {v['rows']:,}  "
              f"sha {v['sha256'][:16]}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== step 3: base6 on {MONTH} (serve-side: fitted on all three folds)",
          flush=True)
    nov = november_frame(NAMES6)
    fit_all = pd.concat([d6[g] for g in F.FOLDS], ignore_index=True)
    a_fit_all = np.concatenate([np.maximum(a6[g], B.FLOOR) for g in F.FOLDS])
    W6 = R.fit_hybrid(fit_all, NAMES6)
    a_nov = R.apply_hybrid(nov, NAMES6, W6)
    pred = compose(fit_all, nov, a_fit_all, a_nov, NAMES6)
    y = nov["y_true"].to_numpy("float64")
    v = write(nov["MVT_ID_mvt"].to_numpy("float64"), y, pred, MONTH, MONTH)
    v["rmse_base6"] = round(rmse(y, pred), 4)
    v["rmse_anchor6"] = round(rmse(y, np.maximum(a_nov, B.FLOOR)), 4)
    v["weights"] = {k: {n: round(x, 6) for n, x in w.items()} for k, w in W6.items()}
    rec["november"] = v
    print(f"    {MONTH} rmse {v['rmse_base6']:9.4f}  rows {v['rows']:,}  "
          f"sha {v['sha256'][:16]}", flush=True)

    blk = pd.read_parquet(BLOCK, columns=["MVT_ID_mvt"])
    got = pd.read_parquet(OOF / f"base6--{MONTH}.parquet")
    ids_new = set(got["MVT_ID_mvt"].astype("float64"))
    ids_blk = set(blk["MVT_ID_mvt"].astype("float64"))
    rec["november"]["checks"] = {
        "rows_equal_block": len(got) == len(blk),
        "ids_identical_to_the_block": ids_new == ids_blk,
        "in_vector_not_in_block": len(ids_new - ids_blk),
        "in_block_not_in_vector": len(ids_blk - ids_new),
        "y_pred_nulls": int(got["y_pred"].isna().sum()),
        "duplicate_ids": int(len(got) - got["MVT_ID_mvt"].nunique())}
    print("    checks", json.dumps(rec["november"]["checks"]), flush=True)

    rec["wall_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
