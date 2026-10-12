"""exp-148: the stacker's label-free row features (one table for 2025 Jan/Jul and 2026 ranking).

    S=/path/to/complete-source-tree
    PRC_SOURCE_ROOT=$S $S/.venv/bin/python experiments/exp-148-short-route/code/features.py --out DIR

Groups (every column is an input the served route already read; none reads a departure
label or a fitted prediction):

* `ctx_*`   airport, month of year, UTC hour, weekday, class (2025: the frozen class of
            exp-146; 2026: the v25 class export, served = clock).
* `obs_*`   the exp-046 head's 27 surface-block columns (`block.FEATURES` less stand lat/lon).
            2025: the pinned ungated blocks of the v25 serve, then `chainfold.restrict` with
            the served gate; 2026: `serve.ranking_block()` in ungated mode (it restricts).
* `shp_*`   the head's 76 shipped exp-018 columns (`zs.shipped(month, drop_cache_vocab=True)`).
* `q_*`     queue inputs: the 12 label-free neighbour columns of the exp-047 block
            (`_n_anom_`, `_n_n_`, prefixes Nf_ and No_), and the raw inputs of the 12
            residual columns, which the stacker recomputes from its own out-of-fold anchor:
            airport, runway key, take-off epoch, file taxi MVT - AOBT_3 and observed taxi
            `own_obs_taxi` (both kept only on 0 < taxi < 20,000 s, `block.py:925-926`).
* `v23_*`   exp-127's HGB inputs: the three survey columns and
            `implied_offblock_to_current250_s` (Kaggle output of prc-exp127b v5). The implied
            clock exists on with-clock rows only, in both years. The survey columns do not
            have the same construction in the two years: 2025 takes them from exp-127's fold
            frames, which hold its with-clock rows only (NaN on ordinary and noac rows); 2026
            takes them from its ranking frame, which holds every ranking row (values on 63 %
            of the ordinary and 94 % of the noac rows). This table keeps both as built; the
            stacker was fitted on the 2025 construction, and `reserve_r1.py` serves it with the
            2026 survey columns NaN where the implied clock is NaN (review 1, finding e).

The 21 proxies and exp-137's two row clocks come from the learner kernel's
`proxy-features.parquet`; the stacker joins them. Write-once output:
`features.parquet` and `features.json` (checks, input hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = Path(os.environ["PRC_SOURCE_ROOT"])
os.environ.setdefault("PRC_DATA_DIR", str(SRC / "data/raw"))
sys.path.insert(0, str(ROOT))
from prc import data as D            # noqa: E402

ID = "MVT_ID_mvt"
E046 = SRC / "experiments/exp-046-surfacefit"
BLOCK_2025 = {"2025-01": E046 / "work/block-2025-01-cb4d6289cb421aa2-ungated.parquet",
              "2025-07": E046 / "work/block-2025-07-8179f74417c8cb99-ungated.parquet"}
QBLOCK = SRC / "artifacts/exp-047/block/{m}.parquet"
CLASS26 = SRC / "artifacts/exp-046/class/{m}-v25.parquet"
F1ROWS = SRC / "experiments/exp-146-validation/frozen/rows.parquet"
E127 = Path(os.environ.get("PRC_EXP127_ROOT", str(SRC)))\
    / "experiments/exp-127-clock-precision"
V23_FOLD = E127 / "kaggle/dataset/fold_frames.parquet"
V23_RANK = E127 / "kaggle/dataset/ranking_frame.parquet"
QCOLS = [f"{p}n_{w}_{t}" for p in ("Nf_", "No_") for w, ts in (("anom", ("cell", "rwy_cell", "rwy_30m")),
                                                              ("n", ("cell", "rwy_cell", "rwy_30m")))
         for t in ts]
V23_SURVEY = ["survey_to_obs_stand_m", "survey_to_runway_km", "survey_is_osm"]
IMPLIED = "implied_offblock_to_current250_s"
GATE = (1299.0, 62.405, True)     # serve.py UNGATED_GATE, checked below


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def e046():
    sys.path.insert(0, str(E046 / "code"))
    import serve as S          # noqa: E402  (exp-046; imports its own block, arms, chainfold)
    import chainfold as CF     # noqa: E402
    import block as BLK        # noqa: E402
    g = S.UNGATED_GATE
    if (g["check_i_abs_dt_offblock_max_s"], g["step_0_gap_cut_s"], g["check_ii_runway_agreement"]) != GATE:
        raise SystemExit(f"served ungated gate {g} differs from {GATE}")
    obs = [c for c in BLK.FEATURES if c not in ("stand_lat", "stand_lon")]
    if len(obs) != 27:
        raise SystemExit(f"{len(obs)} observation columns, not 27")
    return S, CF, obs


def dep_frame(months: list[str]) -> pd.DataFrame:
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
    parts = [pd.read_parquet(D.month_file(m), columns=cols) for m in months]
    f = pd.concat(parts, ignore_index=True)
    return f[f.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")


def context(dep: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True)
    mvt = (t - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy("float64")
    aobt = (pd.to_datetime(dep["AOBT_3_flt"], utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    taxi_f = mvt - aobt.to_numpy("float64")
    taxi_f = np.where((taxi_f > 0) & (taxi_f < 20000), taxi_f, np.nan)
    return pd.DataFrame({
        ID: dep[ID].to_numpy("int64"),
        "ctx_airport": dep["ADEP_mvt"].astype(str).to_numpy(),
        "ctx_month": t.dt.month.to_numpy("int8"),
        "ctx_hour": t.dt.hour.to_numpy("int8"),
        "ctx_weekday": t.dt.weekday.to_numpy("int8"),
        "ctx_day": t.dt.strftime("%Y-%m-%d").to_numpy(),
        "q_apt": dep["ADEP_mvt"].astype(str).to_numpy(),
        "q_rwy": (dep["ADEP_mvt"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str)).to_numpy(),
        "q_mvt": mvt,
        "q_taxi_f": taxi_f})


def join(base: pd.DataFrame, part: pd.DataFrame, what: str) -> pd.DataFrame:
    part = part.copy()
    part[ID] = pd.to_numeric(part[ID], errors="raise").astype("int64")
    if part[ID].duplicated().any():
        raise SystemExit(f"{what}: repeated ids")
    return base.merge(part, on=ID, how="left", validate="one_to_one")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--implied", type=Path, required=True, help="prc-exp127b clock_features.parquet")
    a = ap.parse_args()
    pq, js = a.out / "features.parquet", a.out / "features.json"
    if pq.exists() or js.exists():
        raise SystemExit(f"{pq} exists; write-once")
    t0 = time.monotonic()
    inputs, checks = {}, {}
    S, CF, OBS = e046()
    sys.path.insert(0, str(SRC / "experiments/exp-044-zerosum/code"))
    import zs                                                     # noqa: E402

    # ---- rows: 2025 Jan/Jul (frozen class) and 2026 ranking (v25 class)
    fr = pd.read_parquet(F1ROWS)
    inputs[str(F1ROWS.relative_to(SRC))] = sha(F1ROWS)
    fr = fr[fr.month.isin([1, 7])]
    dep25 = dep_frame([f"2025-{m:02d}" for m in range(1, 13)])
    dep25 = dep25[dep25[ID].astype("int64").isin(fr[ID])]
    if len(dep25) != len(fr) or dep25[ID].duplicated().any():
        raise SystemExit(f"2025 departures {len(dep25)} != frozen rows {len(fr)}")
    rk = D.load_ranking()
    dep26 = rk[rk.PHASE_mvt == "DEP"][[ID, "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]]
    c25 = context(dep25).merge(fr[[ID, "cls"]].rename(columns={"cls": "ctx_cls"}), on=ID,
                               validate="one_to_one")
    c25["ctx_year"] = 2025
    cls26 = pd.concat([pd.read_parquet(Path(str(CLASS26).format(m=m))) for m in ("2026-01", "2026-07")])
    for m in ("2026-01", "2026-07"):
        p = Path(str(CLASS26).format(m=m))
        inputs[str(p.relative_to(SRC))] = sha(p)
    cls26[ID] = cls26[ID].astype("int64")
    cls26 = cls26.set_index(ID)
    c26 = context(dep26)
    k = cls26.reindex(c26[ID].to_numpy())
    c26["ctx_cls"] = np.where(k.noac.to_numpy(bool), "noac",
                              np.where(k.served_by_exp_046.to_numpy(bool), "clock", "ordinary"))
    c26["ctx_year"] = 2026
    checks["rows_2025"] = int(len(c25))
    checks["rows_2026"] = int(len(c26))
    checks["class_2026"] = c26.ctx_cls.value_counts().to_dict()
    if checks["class_2026"] != {"clock": 196256, "ordinary": 148197, "noac": 388}:
        raise SystemExit(f"2026 class counts {checks['class_2026']}")

    # ---- exp-046 surface block
    obs25 = []
    for m, p in BLOCK_2025.items():
        inputs[str(p.relative_to(SRC))] = sha(p)
        b = CF.restrict(pd.read_parquet(p), *GATE)
        obs25.append(b[[ID, "obs_has_clock", *OBS]])
    obs25 = pd.concat(obs25, ignore_index=True)
    with tempfile.TemporaryDirectory() as tmp:          # ranking_block writes a reach report
        (Path(tmp) / "analysis").mkdir()
        S.EXP = Path(tmp)
        S._UNGATED = True
        rdep, _ = S.ranking_block()
        reach = json.loads((Path(tmp) / "analysis/ranking-reach.json").read_text())
    checks["ranking_block_rows"] = int(len(rdep))
    usable26 = pd.Series(rdep["usable"].to_numpy(bool), index=rdep[ID].astype("int64").to_numpy())
    served = k.served_by_exp_046.to_numpy(bool)
    checks["usable_2026_vs_served_mismatch"] = int((usable26.reindex(c26[ID].to_numpy()).to_numpy(bool)
                                                    != served).sum())
    if checks["usable_2026_vs_served_mismatch"]:
        raise SystemExit(f"rebuilt 2026 block: {checks['usable_2026_vs_served_mismatch']} rows differ "
                         "from served_by_exp_046")
    checks["reach_rows"] = reach["rows"]
    obs26 = rdep[[ID, "obs_has_clock", *OBS]]
    c25 = join(c25, obs25.rename(columns={c: f"obs_{c}" for c in OBS}), "obs 2025")
    c26 = join(c26, obs26.rename(columns={c: f"obs_{c}" for c in OBS}), "obs 2026")
    clock25 = c25.obs_has_clock.fillna(False).to_numpy(bool) & (c25.ctx_cls != "noac").to_numpy()
    checks["clock_2025_vs_frozen_mismatch"] = int((clock25 != (c25.ctx_cls == "clock").to_numpy()).sum())
    if checks["clock_2025_vs_frozen_mismatch"]:
        raise SystemExit(f"2025 block clock differs from the frozen class on "
                         f"{checks['clock_2025_vs_frozen_mismatch']} rows")

    # ---- shipped exp-018 columns (76)
    shp = []
    for m in ("2025-01", "2025-07", "2026-01", "2026-07"):
        w = zs.shipped(m, drop_cache_vocab=True)
        inputs[f"artifacts/exp-018-fresh/cache/{m}.parquet"] = sha(SRC / f"artifacts/exp-018-fresh/cache/{m}.parquet")
        w.columns = [f"shp_{c}" for c in w.columns]
        w.insert(0, ID, w.index.to_numpy("float64"))
        shp.append(w.reset_index(drop=True))
    cols = set(shp[0].columns)
    if any(set(s.columns) != cols for s in shp) or len(cols) != 77:
        raise SystemExit(f"shipped columns differ by month or are not 76: {len(cols) - 1}")
    shp = pd.concat(shp, ignore_index=True)
    c25, c26 = join(c25, shp, "shipped"), join(c26, shp, "shipped")

    # ---- exp-047 queue block (label-free columns and the observed taxi)
    q = []
    for m in ("2025-01", "2025-07", "2026-01", "2026-07"):
        p = Path(str(QBLOCK).format(m=m))
        inputs[str(p.relative_to(SRC))] = sha(p)
        b = pd.read_parquet(p, columns=[ID, "own_obs_taxi", *QCOLS])
        q.append(b.rename(columns={"own_obs_taxi": "q_taxi_o", **{c: f"q_{c}" for c in QCOLS}}))
    q = pd.concat(q, ignore_index=True)
    c25, c26 = join(c25, q, "queue"), join(c26, q, "queue")

    # ---- exp-127 (v23) inputs
    inputs[str(V23_FOLD)] = sha(V23_FOLD)
    inputs[str(V23_RANK)] = sha(V23_RANK)
    inputs[str(a.implied)] = sha(a.implied)
    ff = pd.read_parquet(V23_FOLD, columns=[ID, "fold", *V23_SURVEY])
    ff = ff[ff.fold.isin(["january", "july_loo"])].drop(columns="fold")
    rf = pd.read_parquet(V23_RANK, columns=[ID, *V23_SURVEY])
    imp = pd.read_parquet(a.implied)
    checks["implied_columns"] = list(imp.columns)
    if IMPLIED not in imp.columns:
        raise SystemExit(f"{a.implied} has no {IMPLIED}")
    icol = [ID] + [c for c in ("scope",) if c in imp.columns] + [IMPLIED]
    imp = imp[icol].drop_duplicates(subset=[ID])
    imp = imp.drop(columns=[c for c in ("scope",) if c in imp.columns])
    v25 = ff.merge(imp, on=ID, how="left").rename(columns={c: f"v23_{c}" for c in [*V23_SURVEY, IMPLIED]})
    v26 = rf.merge(imp, on=ID, how="left").rename(columns={c: f"v23_{c}" for c in [*V23_SURVEY, IMPLIED]})
    c25, c26 = join(c25, v25, "v23 2025"), join(c26, v26, "v23 2026")

    out = pd.concat([c25, c26], ignore_index=True)
    out = out.drop(columns=["obs_has_clock"])
    if out[ID].duplicated().any():
        raise SystemExit("an id is both a 2025 and a 2026 row")
    for c in out.columns:
        if c.startswith(("obs_", "shp_", "q_n", "v23_")) or c in ("q_taxi_o", "q_taxi_f", "q_mvt"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    checks["coverage"] = {g: {str(y): float(out.loc[out.ctx_year == y, [c for c in out.columns
                                                                        if c.startswith(g)]].notna().mean().mean())
                              for y in (2025, 2026)}
                          for g in ("obs_", "shp_", "q_", "v23_")}
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pq, index=False)
    js.write_text(json.dumps({"what": "exp-148 stacker row features (label-free)", "rows": int(len(out)),
                              "columns": list(out.columns), "checks": checks, "inputs": inputs,
                              "code_sha256": sha(Path(__file__)), "output_sha256": sha(pq),
                              "seconds": round(time.monotonic() - t0, 1)},
                             indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(checks, indent=1, default=str))


if __name__ == "__main__":
    main()
