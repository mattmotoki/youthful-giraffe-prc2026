"""exp-148 SPEC-r2 section 4: the stacker's label-free row features for the level-2 training
months without labels of January or July (2025-12, 2025-02, 2025-06, 2025-08).

    S=/path/to/complete-source-tree
    PRC_SOURCE_ROOT=$S $S/.venv/bin/python experiments/exp-148-short-route/code/features_r2.py --out DIR

The same groups and code as `features.py` builds for January and July (`ctx_*`, `obs_*`,
`shp_*`, `q_*`), less exp-127's `v23_*` columns (SPEC-r2 reading (c)). Rows: every departure
of the month's raw file (each one's UTC day lies in the month; checked). Class by exp-146's rule
(`build_frozen.rows_2025`): exp-046's ungated block of the month, mode-4 rows kept only through
the served gate, the non-mode-4 clock equal to exp-047's block, noac = no aircraft type.
December must equal the frozen class of `frozen/rows.parquet`. Reads no label. Write-once
`features-r2.parquet` and `features-r2.json`.

SPEC-r4 item M: `--months 2025-03,2025-10,2025-11 --name features-r4` builds the three further
months with the same code and checks (December's frozen-class check applies when December is
built). `--cross-check-only` prints each month's exp-047 cross-check count and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as FE                # noqa: E402  (binds prc.data to the source root)
from prc import weekpair as WP       # noqa: E402

ID = FE.ID
D = FE.D
MONTHS = ("2025-12", "2025-02", "2025-06", "2025-08")
UNGATED = {"2025-12": "block-2025-12-d35b7171d7062341-ungated.parquet",   # exp-146's December block
           "2025-02": "block-2025-02-644b0fb0c4473dd9-ungated.parquet",
           "2025-06": "block-2025-06-eb3b7395b2f93dd0-ungated.parquet",
           "2025-08": "block-2025-08-eeee6c43881b166c-ungated.parquet",
           # SPEC-r4 item M: exp-046's ungated blocks of the three further months
           "2025-03": "block-2025-03-4753d248e9f99186-ungated.parquet",
           "2025-10": "block-2025-10-6b59a0f123e265ca-ungated.parquet",
           "2025-11": "block-2025-11-5d47a6d8f64f0cff-ungated.parquet"}
#: exp-046 analysis/chain-fold.json census_by_month (it holds no census for the other months)
KEPT_MODE4 = {"2025-12": 858}
#: rows where exp-047's block has a clock and the month's ungated block, without mode 4, has
#: none (all with no match mode; measured 2026-09-30 before this rule was written). The class
#: follows exp-146's rule (the ungated block); any other count stops the build.
CROSS_CHECK_EXCEPTIONS = {"2025-06": 1, "2025-08": 3,
                          # SPEC-r4: measured 2026-09-30 19:20 UTC with --cross-check-only, before
                          # this line was written
                          "2025-03": 1, "2025-10": 3, "2025-11": 6}


def month_rows(m: str, CF, OBS: list[str], inputs: dict, checks: dict) -> pd.DataFrame:
    raw = D.load_month(m, columns=[ID, "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt",
                                   "AIRCRAFT_TYPE_mvt", "AOBT_3_flt"])
    inputs[str(D.month_file(m).relative_to(FE.SRC))] = FE.sha(D.month_file(m))
    dep = raw[raw.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt").reset_index(drop=True)
    dep[ID] = pd.to_numeric(dep[ID], errors="raise").astype("int64")
    days = pd.Series(WP.day_of(dep).astype(str))
    if not days.str.startswith(m).all():
        if not checks.get("_drop_outside_days"):
            raise SystemExit(f"{m}: {int((~days.str.startswith(m)).sum())} departures fall outside the month")
        # SPEC-r4: a departure of the previous month's last UTC day in this file has no row here
        # (the stacker drops the recorded ids)
        out_ids = sorted(int(i) for i in dep.loc[~days.str.startswith(m).to_numpy(), ID])
        checks.setdefault("_dropped_outside_days", {})[m] = out_ids
        dep = dep[days.str.startswith(m).to_numpy()].reset_index(drop=True)

    # ---- class, exp-146's rule
    p = FE.E046 / "work" / UNGATED[m]
    inputs[str(p.relative_to(FE.SRC))] = FE.sha(p)
    b = pd.read_parquet(p)
    b[ID] = pd.to_numeric(b[ID], errors="raise").astype("int64")
    m4 = (b.match_mode.fillna(-1) == 4).to_numpy()
    keep = (m4 & (b.chain_gap_s > FE.GATE[1]).to_numpy() & (b.chain_abs_dt_offblock <= FE.GATE[0]).to_numpy()
            & (b.chain_runway_agrees == 1.0).to_numpy())
    kept = int(keep.sum())
    if m in KEPT_MODE4 and kept != KEPT_MODE4[m]:
        raise SystemExit(f"{m}: {kept} kept mode-4 rows, recorded {KEPT_MODE4[m]}")
    clock = b.obs_has_clock.fillna(False).to_numpy(bool) & ~(m4 & ~keep)
    p47 = Path(str(FE.QBLOCK).format(m=m))
    inputs[str(p47.relative_to(FE.SRC))] = FE.sha(p47)
    b47 = pd.read_parquet(p47, columns=[ID, "obs_has_clock"])
    b47[ID] = pd.to_numeric(b47[ID], errors="raise").astype("int64")
    base_clock = pd.Series(clock & ~(m4 & keep), index=b[ID].to_numpy())
    n_diff = int((base_clock.reindex(b47[ID].to_numpy()).to_numpy() != b47.obs_has_clock.to_numpy()).sum())
    if checks.get("_cross_check_only"):
        checks[m] = {"exp047_cross_check_rows_differ": n_diff}
        return pd.DataFrame()
    if n_diff != CROSS_CHECK_EXCEPTIONS.get(m, 0):
        raise SystemExit(f"{m}: the non-mode-4 clock differs from exp-047's block on {n_diff} rows")
    ck = pd.Series(clock, index=b[ID].to_numpy()).reindex(dep[ID].to_numpy())
    if ck.isna().any():
        raise SystemExit(f"{m}: {int(ck.isna().sum())} departures have no block row")
    noac = dep.AIRCRAFT_TYPE_mvt.isna().to_numpy()
    cls = np.where(noac, "noac", np.where(ck.to_numpy(bool), "clock", "ordinary"))
    checks[m] = {"departures": int(len(dep)), "kept_mode4": kept, "exp047_cross_check_rows_differ": n_diff,
                 "class": pd.Series(cls).value_counts().to_dict()}

    # ---- context, surface block, shipped columns, queue block (as features.py)
    c = FE.context(dep)
    c["ctx_cls"] = cls
    c["ctx_year"] = 2025
    r = CF.restrict(b, *FE.GATE)[[ID, "obs_has_clock", *OBS]]
    c = FE.join(c, r.rename(columns={o: f"obs_{o}" for o in OBS}), f"obs {m}")
    ck2 = c.obs_has_clock.fillna(False).to_numpy(bool) & (c.ctx_cls != "noac").to_numpy()
    checks[m]["clock_vs_restricted_block_mismatch"] = int((ck2 != (c.ctx_cls == "clock").to_numpy()).sum())
    if checks[m]["clock_vs_restricted_block_mismatch"]:
        raise SystemExit(f"{m}: the restricted block's clock differs from the class on "
                         f"{checks[m]['clock_vs_restricted_block_mismatch']} rows")
    import zs                                                      # noqa: E402  (exp-044)
    w = zs.shipped(m, drop_cache_vocab=True)
    inputs[f"artifacts/exp-018-fresh/cache/{m}.parquet"] = FE.sha(FE.SRC / f"artifacts/exp-018-fresh/cache/{m}.parquet")
    w.columns = [f"shp_{x}" for x in w.columns]
    w.insert(0, ID, w.index.to_numpy("float64"))
    if len(w.columns) != 77:
        raise SystemExit(f"{m}: {len(w.columns) - 1} shipped columns, not 76")
    c = FE.join(c, w.reset_index(drop=True), f"shipped {m}")
    q = pd.read_parquet(p47, columns=[ID, "own_obs_taxi", *FE.QCOLS])
    q = q.rename(columns={"own_obs_taxi": "q_taxi_o", **{x: f"q_{x}" for x in FE.QCOLS}})
    c = FE.join(c, q, f"queue {m}")
    return c.drop(columns=["obs_has_clock"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True, help="the committed features.parquet")
    ap.add_argument("--months", default=",".join(MONTHS))
    ap.add_argument("--name", default="features-r2")
    ap.add_argument("--cross-check-only", action="store_true")
    ap.add_argument("--drop-outside-days", action="store_true",
                    help="drop (and record) departures whose UTC day is outside the file's month")
    a = ap.parse_args()
    months = tuple(a.months.split(","))
    pq, js = a.out / f"{a.name}.parquet", a.out / f"{a.name}.json"
    if pq.exists() or js.exists():
        raise SystemExit(f"{pq} exists; write-once")
    t0 = time.monotonic()
    inputs, checks = {}, {"_cross_check_only": a.cross_check_only, "_drop_outside_days": a.drop_outside_days}
    _, CF, OBS = FE.e046()
    sys.path.insert(0, str(FE.SRC / "experiments/exp-044-zerosum/code"))
    parts = [month_rows(m, CF, OBS, inputs, checks) for m in months]
    if a.cross_check_only:
        print(json.dumps({k: v for k, v in checks.items() if k != "_cross_check_only"}, indent=1))
        return
    checks["dropped_outside_days"] = checks.pop("_dropped_outside_days", {})
    checks.pop("_cross_check_only"), checks.pop("_drop_outside_days")
    out = pd.concat(parts, ignore_index=True)
    if out[ID].duplicated().any():
        raise SystemExit("repeated ids")

    # December against the frozen class; columns against the reference table less v23_*
    fr = pd.read_parquet(FE.F1ROWS)
    inputs[str(FE.F1ROWS.relative_to(FE.SRC))] = FE.sha(FE.F1ROWS)
    if "2025-12" in months:
        fz = fr[fr.month == 12].set_index(ID).cls
        dec = out[out.ctx_month == 12].set_index(ID).ctx_cls
        if len(dec) != len(fz) or not (fz.reindex(dec.index) == dec).all():
            raise SystemExit("December class differs from the frozen rows")
        checks["december_vs_frozen_class"] = "equal"
    ref_cols = [c for c in pd.read_parquet(a.reference).columns if not c.startswith("v23_")]
    inputs["reference features.parquet"] = FE.sha(a.reference)
    if set(out.columns) != set(ref_cols):
        raise SystemExit(f"columns differ from the reference: {sorted(set(out.columns) ^ set(ref_cols))}")
    out = out[ref_cols]
    for c in out.columns:
        if c.startswith(("obs_", "shp_", "q_n")) or c in ("q_taxi_o", "q_taxi_f", "q_mvt"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    checks["coverage"] = {g: {str(mm): float(out.loc[out.ctx_month == mm, [c for c in out.columns if c.startswith(g)]]
                                            .notna().mean().mean()) for mm in (int(x[-2:]) for x in months)}
                          for g in ("obs_", "shp_", "q_")}
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pq, index=False)
    js.write_text(json.dumps({"what": f"exp-148 stacker row features of {'/'.join(months)} (label-free)",
                              "rows": int(len(out)), "columns": list(out.columns), "checks": checks,
                              "inputs": inputs, "code_sha256": {"features_r2.py": FE.sha(Path(__file__)),
                                                                "features.py": FE.sha(HERE / "features.py")},
                              "output_sha256": FE.sha(pq), "seconds": round(time.monotonic() - t0, 1)},
                             indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(checks, indent=1, default=str))


if __name__ == "__main__":
    main()
