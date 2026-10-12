"""exp-189 copy of exp-148 `features.py` (main 5e936513f, sha 4e7df021): the 2026 rows only.

    PRC_SOURCE_ROOT=<tree> [PRC_RANKING_MODE=final] python features_2026.py --out DIR --implied <clock_features> \
        --v23-rank <exp-127 ranking frame>

Change (PROVENANCE.md): the 2025 January/July rows and their reads (frozen rows, the pinned 2025
exp-046 blocks, the 2025 exp-018 and exp-047 months, exp-127 `fold_frames.parquet`) are left out;
every 2026 row is built by the same code from the same 2026 inputs, row by row (joins on
`MVT_ID_mvt`), so the 2026 rows equal those of the full run. `combine_final.py` puts them in place
of the 2026 rows of the saved `features-r4c.parquet` (combine_r2/r4 only append rows). Output:
`features-2026.parquet` and `features-2026.json`. The ranking mode is the parity check against the
saved 2026 rows.

The exp-148 text follows.

exp-148: the stacker's label-free row features (one table for 2025 Jan/Jul and 2026 ranking).

    S=$PRC_COMP_ROOT/prc-data-challenge-2026
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

Final mode (exp-186, `PRC_RANKING_MODE=final`, prc/data.py): the 2026 rows are the four months of
`final_ranking.parquet` (`D.rank_months()`). Their class comes from `serve.ranking_block()` itself
(`usable` = exp-046 `served_by_exp_046`, `noac`), because the v25 class export needs a served exp-046
candidate; the ranking mode checks the two agree (0 mismatches). The 2026 shipped tables are the
exp-018 cache files `<month>-final.parquet` (`zs.final_tag`), the 2026 queue blocks come from
`artifacts/exp-047/block-final/` (exp-047 emit.py in the final mode), `--v23-rank` names the exp-127
ranking frame of the final file, and every 2026 departure must have a row there. The ranking mode is
unchanged.
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
QBLOCK_FINAL = SRC / "artifacts/exp-047/block-final/{m}.parquet"     # exp-186: 2026 months, final mode
CLASS26 = SRC / "artifacts/exp-046/class/{m}-v25.parquet"
F1ROWS = SRC / "experiments/exp-146-validation/frozen/rows.parquet"
E127 = Path(os.environ.get("PRC_EXP127_ROOT", str(SRC.parent / "prc-data-challenge-2026-exp127"))) \
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


def ranking_block(S) -> tuple[pd.DataFrame, dict, dict]:
    """exp-046 `serve.ranking_block()` in ungated mode, with its reach report kept out of the tree.

    Also returns the block's per-month build report (`built_per_month`: the harvest days read)."""
    with tempfile.TemporaryDirectory() as tmp:          # ranking_block writes a reach report
        (Path(tmp) / "analysis").mkdir()
        S.EXP = Path(tmp)
        S._UNGATED = True
        rdep, blk = S.ranking_block()
        reach = json.loads((Path(tmp) / "analysis/ranking-reach.json").read_text())
    return rdep, reach, dict(blk.attrs.get("built_per_month", {}))


def summary_days(built: dict, months: tuple[str, ...]) -> dict:
    """exp-186: harvest airport-days present and absent inside each month, from `built_per_month`.

    exp-046 block.py reads an absent harvest day as "no observation" and does not stop; in the
    ranking mode the exact 2026 class counts catch a missing month, in the final mode this does."""
    out = {}
    for m in months:
        at = built.get(m, {})
        pres = [tuple(x) for x in at.get("days_present_list", []) if str(x[1]).startswith(m)]
        miss = [tuple(x) for x in at.get("days_missing", []) if str(x[1]).startswith(m)]
        out[m] = {"airport_days_present": len(pres), "airport_days_missing": len(miss),
                  "missing_by_airport": pd.Series([a for a, _ in miss], dtype=object)
                  .value_counts().sort_index().astype(int).to_dict()}
    return out


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
    ap.add_argument("--v23-rank", type=Path, default=V23_RANK,
                    help="exp-127 ranking frame (the final mode needs the one built on the final file)")
    a = ap.parse_args()
    pq, js = a.out / "features-2026.parquet", a.out / "features-2026.json"
    if pq.exists() or js.exists():
        raise SystemExit(f"{pq} exists; write-once")
    t0 = time.monotonic()
    inputs, checks = {}, {}
    final = D.ranking_mode() == "final"                           # exp-186
    rank_months = tuple(D.rank_months())
    if final:
        checks["ranking_mode"] = "final"
    S, CF, OBS = e046()
    sys.path.insert(0, str(SRC / "experiments/exp-044-zerosum/code"))
    import zs                                                     # noqa: E402

    # ---- rows: 2026 ranking (v25 class); exp-189: no 2025 rows
    rk = D.load_ranking()
    dep26 = rk[rk.PHASE_mvt == "DEP"][[ID, "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]]
    if final:
        # exp-186: the class of the v25 export, from the block it was written from (see the docstring)
        rdep, reach, built = ranking_block(S)
        checks["summaries_2026_by_month"] = summary_days(built, rank_months)
        empty = [m for m, v in checks["summaries_2026_by_month"].items() if not v["airport_days_present"]]
        if empty:
            raise SystemExit(f"no exp-046 harvest summary inside {empty}: the block would serve these "
                             f"months with no clock; build the summaries first")
        c26 = context(dep26)
        rb = pd.DataFrame({"noac": rdep["noac"].to_numpy(bool),
                           "served_by_exp_046": rdep["usable"].to_numpy(bool)},
                          index=rdep[ID].astype("int64").to_numpy())
        if rb.index.duplicated().any():
            raise SystemExit("the ranking block repeats an id")
        k = rb.reindex(c26[ID].to_numpy())
        if k.isna().any(axis=None):
            raise SystemExit(f"{int(k.isna().any(axis=1).sum())} final departures have no ranking-block row")
    else:
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
    checks["rows_2026"] = int(len(c26))
    checks["class_2026"] = c26.ctx_cls.value_counts().to_dict()
    if final:
        # exp-186: the class counts of the final file are new; the rows must be the template's
        checks["rows_2026_by_month"] = {m: int(n) for m, n in
                                        c26.ctx_day.str[:7].value_counts().sort_index().items()}
        checks["class_2026_by_month"] = {f"{m} {c}": int(n) for (m, c), n in
                                         c26.groupby([c26.ctx_day.str[:7], "ctx_cls"]).size().items()}
        if len(c26) != D.n_ranking() or checks["rows_2026_by_month"] != D.ranking_spec().month_departures:
            raise SystemExit(f"final 2026 rows {checks['rows_2026_by_month']} are not the template's "
                             f"{D.ranking_spec().month_departures}")
    elif checks["class_2026"] != {"clock": 196256, "ordinary": 148197, "noac": 388}:
        raise SystemExit(f"2026 class counts {checks['class_2026']}")

    # ---- exp-046 surface block (exp-189: 2026 only)
    if not final:
        rdep, reach, _ = ranking_block(S)
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
    c26 = join(c26, obs26.rename(columns={c: f"obs_{c}" for c in OBS}), "obs 2026")

    # ---- shipped exp-018 columns (76)
    shp = []
    for m in rank_months:
        w = zs.shipped(m, drop_cache_vocab=True)
        cp = f"artifacts/exp-018-fresh/cache/{m}{zs.final_tag(m)}.parquet"     # exp-186: "-final" in the final mode
        inputs[cp] = sha(SRC / cp)
        w.columns = [f"shp_{c}" for c in w.columns]
        w.insert(0, ID, w.index.to_numpy("float64"))
        shp.append(w.reset_index(drop=True))
    cols = set(shp[0].columns)
    if any(set(s.columns) != cols for s in shp) or len(cols) != 77:
        raise SystemExit(f"shipped columns differ by month or are not 76: {len(cols) - 1}")
    shp = pd.concat(shp, ignore_index=True)
    c26 = join(c26, shp, "shipped")

    # ---- exp-047 queue block (label-free columns and the observed taxi)
    q = []
    for m in rank_months:
        p = Path(str(QBLOCK_FINAL if final and m.startswith("2026") else QBLOCK).format(m=m))
        inputs[str(p.relative_to(SRC))] = sha(p)
        b = pd.read_parquet(p, columns=[ID, "own_obs_taxi", *QCOLS])
        q.append(b.rename(columns={"own_obs_taxi": "q_taxi_o", **{c: f"q_{c}" for c in QCOLS}}))
    q = pd.concat(q, ignore_index=True)
    c26 = join(c26, q, "queue")

    # ---- exp-127 (v23) inputs (exp-189: the ranking frame only)
    inputs[str(a.v23_rank)] = sha(a.v23_rank)
    inputs[str(a.implied)] = sha(a.implied)
    rf = pd.read_parquet(a.v23_rank, columns=[ID, *V23_SURVEY])
    if final:
        miss = ~c26[ID].isin(pd.to_numeric(rf[ID], errors="raise").astype("int64"))
        checks["v23_rank_missing_2026_rows"] = int(miss.sum())
        if miss.any():
            raise SystemExit(f"{int(miss.sum())} final departures have no row in {a.v23_rank}")
    imp = pd.read_parquet(a.implied)
    checks["implied_columns"] = list(imp.columns)
    if IMPLIED not in imp.columns:
        raise SystemExit(f"{a.implied} has no {IMPLIED}")
    icol = [ID] + [c for c in ("scope",) if c in imp.columns] + [IMPLIED]
    imp = imp[icol].drop_duplicates(subset=[ID])
    imp = imp.drop(columns=[c for c in ("scope",) if c in imp.columns])
    v26 = rf.merge(imp, on=ID, how="left").rename(columns={c: f"v23_{c}" for c in [*V23_SURVEY, IMPLIED]})
    c26 = join(c26, v26, "v23 2026")

    out = c26.reset_index(drop=True)
    out = out.drop(columns=["obs_has_clock"])
    if out[ID].duplicated().any():
        raise SystemExit("a 2026 id repeats")
    for c in out.columns:
        if c.startswith(("obs_", "shp_", "q_n", "v23_")) or c in ("q_taxi_o", "q_taxi_f", "q_mvt"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    checks["coverage"] = {g: {str(y): float(out.loc[out.ctx_year == y, [c for c in out.columns
                                                                        if c.startswith(g)]].notna().mean().mean())
                              for y in (2026,)}
                          for g in ("obs_", "shp_", "q_", "v23_")}
    if final:
        mon = out.ctx_day.str[:7]
        checks["coverage_2026_by_month"] = {
            g: {m: float(out.loc[(out.ctx_year == 2026) & (mon == m),
                                 [c for c in out.columns if c.startswith(g)]].notna().mean().mean())
                for m in rank_months}
            for g in ("obs_", "shp_", "q_", "v23_")}
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pq, index=False)
    js.write_text(json.dumps({"what": "exp-189: the 2026 rows of the exp-148 stacker row features (label-free)",
                              "rows": int(len(out)),
                              "columns": list(out.columns), "checks": checks, "inputs": inputs,
                              "code_sha256": sha(Path(__file__)), "output_sha256": sha(pq),
                              "seconds": round(time.monotonic() - t0, 1)},
                             indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(checks, indent=1, default=str))


if __name__ == "__main__":
    main()
