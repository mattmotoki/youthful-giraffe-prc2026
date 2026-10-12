"""Step 3 and step 4 -- the serving path and the ranking-side census.

`h` is fitted on the three 2025 months whose v10 residual exists (2025-01,
2025-07, 2025-12; SPEC R3), on Q03's objective and with the categorical level
list frozen on those three months.  The ranking months' cells are formed from
the RANKING FILE's own rows -- airport x UTC 10-minute bin x serving regime,
which `docs/input-rule.md` makes legal: every column of every row of that file
is an input.  `d = h(X) - mean_cell(h(X))` uses features only; cells under four
eligible rows are left at zero, and the update is added AFTER v10's serving
masks, so the ranking `noac` and EHAM rows keep v10's bytes exactly and EDDF's
are unchanged by eligibility.

The output floor is a bound on `p`, and a bound needs a new algebra check
(QUEUE.md, Q03): where `p_v10 + d` would fall under 60 s the cell's update is
re-solved as the projection of `d0` onto {sum zero} INTERSECT {p >= 60}, which
keeps the zero sum exactly.  Zero update is always feasible, so the fallback is
safe.

    OMP_NUM_THREADS=3 python serve.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zs                                   # noqa: E402
import learner as L                         # noqa: E402
import packet as PK                         # noqa: E402
import prc.blend as B                       # noqa: E402

EXP = zs.EXP
ROOT = zs.ROOT
FOLDS = zs.FOLDS
FLOOR = 60.0
BASES = {"v10": ROOT / "submissions/uploaded/youthful-giraffe_v10.parquet",
         "v11": ROOT / "submissions/uploaded/youthful-giraffe_v11.parquet"}
V10_FILE = BASES["v10"]


def bounded_project(d0: np.ndarray, code: np.ndarray, ncell: int,
                    lower: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Project `d0` onto {cell sums zero} INTERSECT {d >= lower}, per cell.

    Active-set water filling: centre, fix the rows that violate their bound at
    it, re-centre the rest, repeat.  A cell that does not settle in 20 passes is
    given a zero update, which is always feasible because `p_v10 >= 60`.
    """
    d = zs.project(d0, code, ncell)
    fixed = np.zeros(len(d0), bool)
    n_iter = 0
    for _ in range(20):
        bad = (d < lower - 1e-12) & ~fixed
        if not bad.any():
            break
        n_iter += 1
        fixed |= bad
        d[fixed] = np.maximum(d[fixed], lower[fixed])
        free = ~fixed
        sfix = np.bincount(code[fixed], weights=d[fixed], minlength=ncell)
        sfree = np.bincount(code[free], weights=d0[free], minlength=ncell)
        nfree = np.bincount(code[free], minlength=ncell).astype("float64")
        lam = np.where(nfree > 0, -(sfix + sfree) / np.maximum(nfree, 1), 0.0)
        d[free] = d0[free] + lam[code[free]]
    bad_cells = np.unique(code[(d < lower - 1e-9)])
    n_zeroed = 0
    if len(bad_cells):
        m = np.isin(code, bad_cells)
        d[m] = 0.0
        n_zeroed = int(len(bad_cells))
    return d, n_iter, n_zeroed


def main() -> None:
    t0 = time.time()
    pan = zs.V.load()
    S = {f: L.fold_state(f, pan[f]) for f in FOLDS}
    amps = json.load(open(EXP / "analysis" / "learner.json"))["amplitudes"]
    arm = sys.argv[1] if len(sys.argv) > 1 else "ridge"
    base = sys.argv[2] if len(sys.argv) > 2 else "v10"
    base_file = BASES[base]
    tag = "" if base == "v10" else f"-{base}"
    amp = amps[arm]["pooled"]
    print(f"  arm {arm}; served amplitude (pooled cross-month) {amp:.4f}", flush=True)

    months = [S[f]["month"] for f in FOLDS]
    lv = zs.levels_of([zs.dep(m) for m in months])
    shipped = arm in L.SHIPPED
    parts, rs, cs = [], [], []
    for f in FOLDS:
        G = S[f]
        X, cols = L.month_matrix(G["month"], lv, shipped, G["ids"])
        e = np.flatnonzero(G["elig"])
        parts.append(X[e]); rs.append(G["r"][e])
        cs.append(G["code"][e].astype("int64") + 1000000 * FOLDS.index(f))
        del X
    Xtr = np.vstack(parts); del parts
    rtr = np.concatenate(rs); ctr = np.concatenate(cs)
    print(f"  fit rows {len(rtr):,}, {Xtr.shape[1]} columns", flush=True)

    rk = zs.ranking_dep()
    air = rk["ADEP_mvt"].astype(str).to_numpy()
    t = zs.epoch_seconds(rk["MVT_TIME_UTC_mvt"])
    reg = B.regimes_from_raw(rk).astype(str)
    elig = (reg == zs.ELIGIBLE_REGIME) & ~np.isin(air, zs.HOLD_AIRPORTS)
    code, ncell = zs.cellcode(air, reg, t, zs.BIN)
    ei = np.flatnonzero(elig)
    cnt = np.bincount(code[ei], minlength=ncell)
    m0 = elig & (cnt[code] >= zs.MIN_CELL)
    # A resolution or timezone slip in the take-off clock turns the 10-minute cell
    # into a multi-day one and the projection into a no-op; the first version of
    # this file did exactly that.  Gate the cell, do not trust it.
    n_days = len(np.unique(np.floor(t / 86400.0)))
    med = float(np.median(cnt[code[ei]]))
    fold_med = float(np.median(np.concatenate(
        [S[f]["cnt"][S[f]["code"][S[f]["m0"]]] for f in FOLDS])))
    if not (55 <= n_days <= 70):
        raise SystemExit(f"the ranking take-off clock covers {n_days} UTC dates, "
                         f"not the two ranking months' 62")
    if not (0.5 * fold_med < med < 2.0 * fold_med):
        raise SystemExit(f"median eligible rows per cell is {med:.1f} against the "
                         f"fold side's {fold_med:.1f}: the cell grid is not Q03's")
    Xrk, cols_rk = L.month_matrix_frame(rk, lv, shipped)
    if cols_rk != cols:
        raise SystemExit("the ranking design does not match the fitting design")
    h_rk = L.fit_h(L.BASE[arm], Xtr, rtr, ctr, Xrk)
    del Xtr, Xrk
    print(f"  fitted and scored at {time.time() - t0:.0f}s", flush=True)

    cm, ucm = pd.factorize(code[m0])
    d0 = np.zeros(len(rk))
    d0[m0] = amp * zs.project(h_rk[m0], cm, len(ucm))

    v10 = pd.read_parquet(base_file)
    p10 = pd.Series(v10[zs.D.TARGET].to_numpy("float64"),
                    index=v10[zs.D.ID].to_numpy("float64")).reindex(
        rk["MVT_ID_mvt"].astype("float64").to_numpy()).to_numpy()
    if not np.isfinite(p10).all():
        raise SystemExit(f"a ranking departure has no {base} prediction")
    lower = np.where(m0, FLOOR - p10, 0.0)
    d = np.zeros(len(rk))
    d[m0], n_iter, n_zeroed = bounded_project(d0[m0], cm, len(ucm), lower[m0])

    s = np.bincount(cm, weights=d[m0], minlength=len(ucm))
    scale = max(1.0, float(np.abs(d[m0]).sum()))
    checks = {"max_abs_cell_sum": float(np.abs(s).max()),
              "cell_sum_check": bool(np.abs(s).max() / scale < zs.TOL),
              "floor_passes": bool((p10 + d).min() >= FLOOR - 1e-9),
              "bound_iterations": n_iter, "cells_zeroed_by_bound": n_zeroed}
    p_new = p10 + d
    noac = reg == "noac"
    eham = air == "EHAM"
    eddf = air == "EDDF"
    ref10 = pd.read_parquet(V10_FILE)
    p_v10_ref = pd.Series(ref10[zs.D.TARGET].to_numpy("float64"),
                          index=ref10[zs.D.ID].to_numpy("float64")).reindex(
        rk["MVT_ID_mvt"].astype("float64").to_numpy()).to_numpy()
    identical = {"noac": [int(noac.sum()), float(np.abs(d[noac]).max())],
                 "EHAM": [int(eham.sum()), float(np.abs(d[eham]).max())],
                 "EDDF": [int(eddf.sum()), float(np.abs(d[eddf]).max())],
                 "vs_v10_bytes_noac": float(np.abs(p_new[noac] - p_v10_ref[noac]).max()),
                 "vs_v10_bytes_EHAM": float(np.abs(p_new[eham] - p_v10_ref[eham]).max())}

    # Sol round 1, non-blocking: the first version recorded a failed check and
    # wrote the file anyway.  A failed structural check now stops the write.
    if not (checks["cell_sum_check"] and checks["floor_passes"]):
        raise SystemExit(f"a structural check failed, no candidate written: {checks}")

    ids = rk["MVT_ID_mvt"].astype("float64").to_numpy()
    path = ROOT / f"submissions/candidates/exp-044-zerosum{tag}.parquet"
    sha = zs.D.write_submission(pd.Series(p_new, index=ids), path)

    mon = rk["month"].to_numpy()
    census = {"n_ranking_departures": int(len(rk)),
              "n_eligible": int(elig.sum()),
              "eligible_pct": round(100 * float(elig.mean()), 2),
              "n_cells_eligible": int((cnt >= 1).sum()),
              "n_scored_cohort": int(m0.sum()),
              "share_eligible_in_cells_under_four": round(
                  float((elig & ~m0).sum()) / max(int(elig.sum()), 1) * 100, 2),
              "n_changed": int((d != 0).sum()),
              "rms_update_all_scored": round(float(np.sqrt((d[m0] ** 2).mean())), 4),
              "rms_update_over_eligible": round(float(np.sqrt((d[elig] ** 2).mean())), 4),
              "rms_update_over_all_departures": round(float(np.sqrt((d ** 2).mean())), 4),
              "max_abs_update": round(float(np.abs(d).max()), 3),
              "by_month": {}, "by_airport": {}}
    for m in sorted(set(mon)):
        k = m0 & (mon == m)
        census["by_month"][m] = {
            "n_departures": int((mon == m).sum()), "n_eligible": int((elig & (mon == m)).sum()),
            "n_scored": int(k.sum()),
            "rms_update": round(float(np.sqrt((d[k] ** 2).mean())), 4) if k.sum() else 0.0}
    for a in sorted(set(air)):
        k = m0 & (air == a)
        census["by_airport"][a] = {
            "n_departures": int((air == a).sum()), "n_eligible": int((elig & (air == a)).sum()),
            "n_scored": int(k.sum()),
            "rms_update": round(float(np.sqrt((d[k] ** 2).mean())), 4) if k.sum() else 0.0}

    out = {"step": "3 serving path and 4 census", "arm": arm, "base": base,
           "served_amplitude": round(amp, 4),
           "fit_months": months, "fit_rows": int(len(rtr)), "n_features": int(len(cols)),
           "candidate": str(path.relative_to(ROOT)), "sha256": sha,
           "bytes": path.stat().st_size,
           "prediction": {"min": round(float(p_new.min()), 4),
                          "max": round(float(p_new.max()), 4),
                          "finite": bool(np.isfinite(p_new).all())},
           "checks": checks, "identical_to_v10": identical, "census": census,
           "provenance": {"v10_file_sha256": PK.sha_file(V10_FILE),
                          "ranking_file_sha256": PK.sha_file(zs.D.data_dir() / "ranking.parquet"),
                          "git_commit": PK.git_commit(),
                          "code_sha256_exp044": {q.name: PK.sha_file(q)
                                                 for q in sorted(HERE.glob("*.py"))},
                          "code_sha256_exp039_imported": {q.name: PK.sha_file(q)
                                                          for q in sorted(PK.HERE.glob("*.py"))}},
           "wall_s": round(time.time() - t0, 1)}
    (EXP / "analysis" / f"serve{tag}.json").write_text(json.dumps(out, indent=2))
    np.save(EXP / "analysis" / f"served_update{tag}.npy",
            np.column_stack([ids, d]).astype("float64"))
    print(json.dumps({k: out[k] for k in ("served_amplitude", "sha256", "prediction",
                                          "checks", "identical_to_v10")}, indent=2))
    print(f"  census: eligible {census['n_eligible']:,} of {census['n_ranking_departures']:,}, "
          f"scored {census['n_scored_cohort']:,}, changed {census['n_changed']:,}, "
          f"RMS {census['rms_update_all_scored']:.3f} s")
    print(f"  written {path} sha {sha[:16]}...  ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
