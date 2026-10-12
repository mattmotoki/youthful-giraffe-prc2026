"""exp-194 lever (b), DH: the reduced head for rows with no head value (SPEC step 3).

A LightGBM bag on the 2025 clock rows (the rows HB3u trains on), with every adsb.lol input removed: inputs from the
flight list, NM times, schedule and official neighbour counts only (`shp_*`, `q_taxi_f`, `q_Nf_*`), the context
codes and three frequent-level categories. Target y. Scores every 2025 departure of the split's held days (clock rows
as made-dark rows, the other classes as they are), or with `jjrank` the January/July 2025 rows (`jjrefit`).

    EXP159_SET=12 OMP_NUM_THREADS=2 $PY -I dh_fit.py --c159 <prc-exp159-work/code> --split jjx-02 \
        --seeds 601-603 --threads 2 --out <dir> [--save-models]

Writes (write-once) <dir>/DH-<split>.parquet (MVT_ID_mvt, month, utc_day, airport, cls, pred_DH),
DH-<split>.seeds.npz, DH-<split>.json (inputs, code tables, training rows, settings, hashes), and with
--save-models the boosters under DH-<split>.models/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LGB = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 255, "min_child_samples": 100,
       "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
       "deterministic": True, "force_row_wise": True, "verbosity": -1}   # HB3u's HU18 shape (exp-159 fit.py)
ROUNDS = 1800
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
MIN_ROWS = 5000                     # exp-159 ACTYPE_MIN_ROWS: a level is coded only with this many clock rows
Q_KEEP = ["q_taxi_f"] + [f"q_Nf_n_{k}" for k in ("anom_cell", "anom_rwy_cell", "anom_rwy_30m", "n_cell",
                                                  "n_rwy_cell", "n_rwy_30m")]
CONTEXT = ["x_airport_code", "x_hour_utc", "x_weekday"]
CATS = ["x_actype_common", "x_oper_common", "x_rwy_common"]
#: removed: every adsb.lol input
REMOVED_PREFIX = ("obs_", "v23_", "q_No_", "clock_", "trk_")
#: OPDI surface events (exp-155 opdi-first; finaldata's final table for 2026), thinned to the 2026 coverage
EVENTS = ("exit_parking_position", "exit_apron", "entry_taxiway")
OPDI_COLS = [f"opdi_{e}_{k}" for e in EVENTS for k in ("lead_s", "dist_m")]
TRAIN_CELL = "2-EDDF"               # training rows are thinned with this cell's ratios (desk 04:34 UTC 10-09)


def uniform_of(ids: np.ndarray) -> np.ndarray:
    """A fixed uniform draw per departure id (Knuth multiplicative hash), independent of row order."""
    x = (np.asarray(ids, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(194)) % np.uint64(2 ** 32)
    return x.astype(np.float64) / 2.0 ** 32


def join_opdi(f: pd.DataFrame, path: Path) -> pd.DataFrame:
    """The six OPDI columns, duplicate identities blanked (exp-159 frame12)."""
    op = pd.read_parquet(path)
    lead = [c for c in op.columns if c.endswith(("_lead_s", "_dist_m"))]
    op.loc[op.duplicate_opdi_identity.astype(bool), lead] = np.nan
    op = op[["MVT_ID_mvt"] + lead].rename(columns={c: f"opdi_{c}" for c in lead})
    assert set(OPDI_COLS) <= set(op.columns), sorted(op.columns)
    return f.merge(op[["MVT_ID_mvt"] + OPDI_COLS], on="MVT_ID_mvt", how="left", validate="one_to_one")


def thin_opdi(f: pd.DataFrame, train: np.ndarray, cov26: dict, cell: np.ndarray, clock: np.ndarray) -> dict:
    """Blank OPDI events so that each matches its 2026 coverage on no-head rows: keep event e of a row when
    u(id) < r_e, r_e = min(1, p26_e / p25_e); p25 on the cell's 2025 clock rows. Scored rows use their cell's ratio,
    training rows the ratio of TRAIN_CELL. One draw per row, so the events stay nested."""
    u = uniform_of(f["MVT_ID_mvt"].to_numpy("int64"))
    info = {"p25_clock": {}, "p26_no_head": {}, "ratio": {}}
    for c in np.unique(cell):
        m = (cell == c) & clock
        info["p25_clock"][c] = {e: float(np.isfinite(f[f"opdi_{e}_lead_s"].to_numpy(float)[m]).mean()) if m.any()
                                else 0.0 for e in EVENTS}
        p26 = cov26.get(c)
        info["p26_no_head"][c] = None if p26 is None else {e: p26[e] for e in EVENTS}
        info["ratio"][c] = {e: (1.0 if p26 is None or info["p25_clock"][c][e] == 0 else
                                min(1.0, p26[e] / info["p25_clock"][c][e])) for e in EVENTS}
    if TRAIN_CELL not in info["ratio"]:
        p25 = {e: float(np.isfinite(f[f"opdi_{e}_lead_s"].to_numpy(float)[clock & (f["airport"].astype(str).to_numpy()
                                                                                  == "EDDF") & (f["month"].to_numpy() == "2025-02")]).mean()) for e in EVENTS}
        info["ratio"][TRAIN_CELL] = {e: min(1.0, cov26[TRAIN_CELL][e] / p25[e]) if p25[e] > 0 else 1.0 for e in EVENTS}
        info["p25_clock"][TRAIN_CELL] = p25
    info["train_ratio"] = info["ratio"][TRAIN_CELL]
    before = {e: int(np.isfinite(f[f"opdi_{e}_lead_s"].to_numpy(float)).sum()) for e in EVENTS}
    for e in EVENTS:
        r = np.array([info["ratio"][c][e] for c in cell])
        r = np.where(train, info["train_ratio"][e], r)
        drop = u >= r
        for k in ("lead_s", "dist_m"):
            col = f"opdi_{e}_{k}"
            f.loc[drop, col] = np.nan
    info["rows_with_event_before"] = before
    info["rows_with_event_after"] = {e: int(np.isfinite(f[f"opdi_{e}_lead_s"].to_numpy(float)).sum()) for e in EVENTS}
    info["train_rows_with_event_after"] = {e: int(np.isfinite(f[f"opdi_{e}_lead_s"].to_numpy(float)[train]).sum())
                                           for e in EVENTS}
    return info


def sha(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def levels(f: pd.DataFrame) -> dict[str, pd.Series]:
    """The raw levels of the three categories (exp-159 `derived`: type; operator = ICAO prefix of the callsign, else
    of the flight number; runway = airport|runway)."""
    apt = f["airport"].astype("string")
    typ = f["AIRCRAFT_TYPE_mvt"].astype("string")
    oper = f["CALLSIGN_flt"].astype("string").str.extract(r"^([A-Z]{3})\d", expand=False)
    oper = oper.fillna(f["FLIGHT_mvt"].astype("string").str.extract(r"^([A-Z]{3})\d", expand=False))
    rwy = (apt + "|" + f["RUNWAY_mvt"].astype("string")).where(f["RUNWAY_mvt"].notna())
    return {"x_actype_common": typ, "x_oper_common": oper, "x_rwy_common": rwy}


def code_tables(f: pd.DataFrame, rows: np.ndarray) -> dict[str, list[str]]:
    """Per category, the sorted levels with MIN_ROWS rows or more among `rows` (label-free counts)."""
    out = {}
    for name, v in levels(f).items():
        n = v[rows].value_counts()
        out[name] = sorted(str(x) for x in n[n >= MIN_ROWS].index)
    return out


def encode(f: pd.DataFrame, tables: dict[str, list[str]]) -> pd.DataFrame:
    """Context codes and the three categories as float codes into the saved tables (missing outside them)."""
    out = pd.DataFrame(index=f.index)
    apt = f["airport"].astype(str)
    if not apt.isin(AIRPORTS).all():
        raise ValueError("an airport outside the code table")
    out["x_airport_code"] = apt.map({a: i for i, a in enumerate(AIRPORTS)}).astype("float64")
    t = pd.to_datetime(f["MVT_TIME_UTC_mvt"], utc=True)
    out["x_hour_utc"] = t.dt.hour + t.dt.minute / 60.0
    out["x_weekday"] = t.dt.weekday.astype("float64")
    for name, v in levels(f).items():
        code = {x: i for i, x in enumerate(tables[name])}
        out[name] = v.map(code).astype("float64")
    return out


def load(C, FR, months: set[str]) -> pd.DataFrame:
    """2025 departures of `months`: exp-148 feature rows (features-r4c, and features-r5 for April, May, September)
    joined to the raw departures. Only the kept inputs are read."""
    parts = []
    for path in (C.FEATURES, C.FEATURES5):
        import pyarrow.parquet as pq
        names = pq.ParquetFile(path).schema_arrow.names
        cols = [C.ID, "ctx_airport", "ctx_month", "ctx_year", "ctx_cls"] + \
            [c for c in names if c.startswith("shp_")] + Q_KEEP
        d = pd.read_parquet(path, columns=cols)
        d = d[(d.ctx_year.astype(int) == 2025)]
        parts.append(d)
    feat = pd.concat(parts, ignore_index=True)
    feat["month"] = "2025-" + feat.ctx_month.astype(int).map("{:02d}".format)
    feat = feat[feat.month.isin(months)]
    if feat[C.ID].duplicated().any():
        raise ValueError("feature rows: repeated ids")
    raw = FR.raw_departures()
    f = feat.merge(raw, on=C.ID, how="left", validate="one_to_one")
    if f[C.TIME].isna().any():
        raise ValueError("a feature row has no raw departure")
    t = pd.to_datetime(f[C.TIME], utc=True)
    f["utc_day"] = t.dt.strftime("%Y-%m-%d")
    if not (f.utc_day.str[:7] == f.month).all():
        raise ValueError("a row's UTC day is outside its feature month")
    f = f.rename(columns={"ctx_airport": "airport", "TAXITIME_SEC_mvt": "y", "ctx_cls": "cls"})
    if not (f.airport.astype(str) == f.ADEP_mvt.astype(str)).all():
        raise ValueError("airport of the feature table and ADEP_mvt differ")
    return f.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c159", type=Path, required=True, help="exp-159 code folder (common.py, frame.py)")
    ap.add_argument("--split", required=True, choices=["jjx-02", "jjx-06", "jjrank"])
    ap.add_argument("--seeds", default="601-603")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="development only: N training rows, 20 rounds, one seed")
    ap.add_argument("--opdi-cov", type=Path, help="dh_opdi_cov.py output: adds the six OPDI columns, thinned to the "
                    "2026 coverage on no-head rows (arm DHo); without it, no OPDI input (arm DH)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get("EXP159_SET") != "12":
        raise SystemExit("set EXP159_SET=12 (the twelve-month frame and features-r5)")
    sys.path.insert(0, str(a.c159.resolve()))
    import common as C                    # noqa: E402  (sets the paths of prc and pilot)
    import frame as FR                    # noqa: E402
    import pilot as P                     # noqa: E402
    import lightgbm as lgb                # noqa: E402
    code_sha = {"dh_fit.py": sha(__file__), "common.py": sha(a.c159 / "common.py"), "frame.py": sha(a.c159 / "frame.py")}
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    stage = ("DHo" if a.opdi_cov else "DH") + ("smoke" if a.smoke else "")
    s = P.split(a.split)
    if s.token != C.TOKENS[a.split]:
        raise SystemExit(f"split {a.split}: token {s.token} is not the contract's {C.TOKENS[a.split]}")
    name = "jjrefit" if a.split == "jjrank" else a.split
    for p in (a.out / f"{stage}-{name}.parquet", a.out / f"{stage}-{name}.json"):
        C.refuse_existing(p)
    a.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    f = load(C, FR, set(C.MONTHS_2025))
    wp = f[[C.TIME]]
    y = f["y"].to_numpy("float64")
    clock = f["cls"].astype(str).eq("clock").to_numpy()
    train = clock & s.label_mask(wp) & np.isfinite(y)
    sc = P.split("jjrefit") if a.split == "jjrank" else s
    score = sc.score_mask(wp)
    if (score & train).any():
        raise SystemExit("a scored row is a training row")
    if a.smoke:
        idx = np.flatnonzero(train)[: a.smoke]
        train = np.zeros(len(f), bool); train[idx] = True
        seeds, a.rounds = seeds[:1], 20
    s.assert_training_frame(wp[train], target=y[train], what=f"{stage} {a.split}")
    tables = code_tables(f, train)
    opdi_info = None
    if a.opdi_cov:
        f = join_opdi(f, C.OPDI)
        cov26 = json.loads(a.opdi_cov.read_text())["cells"]
        cell = (f["month"].str[5:].astype(int).astype(str) + "-" + f["airport"].astype(str)).to_numpy()
        opdi_info = thin_opdi(f, train, cov26, cell, clock)
        opdi_info["cov_sha256"] = sha(a.opdi_cov)
        opdi_info["opdi_sha256"] = sha(C.OPDI)
    X = pd.concat([f[[c for c in f.columns if c.startswith("shp_")] + Q_KEEP +
                     (OPDI_COLS if a.opdi_cov else [])], encode(f, tables)], axis=1)
    feats = sorted(c for c in X.columns if c.startswith("shp_")) + Q_KEEP + CONTEXT + CATS + \
        (OPDI_COLS if a.opdi_cov else [])
    if len([c for c in feats if c.startswith("shp_")]) != 76:
        raise SystemExit("expected 76 shp_ inputs")
    if any(c.startswith(REMOVED_PREFIX) or c == "q_taxi_o" for c in feats):
        raise SystemExit("an adsb.lol input is in the list")
    Xn = X[feats].to_numpy("float32")
    print(f"{stage} {a.split}: {int(train.sum())} training rows, {len(feats)} inputs, {int(score.sum())} scored rows "
          f"({int((score & clock).sum())} clock), load {time.time() - t0:.0f}s", flush=True)
    dtrain = lgb.Dataset(Xn[train], y[train], feature_name=feats, categorical_feature=CATS, free_raw_data=False)
    zs = np.zeros((len(seeds), int(score.sum())), "float32")
    gain = np.zeros(len(feats))
    secs = []
    for k, seed in enumerate(seeds):
        t1 = time.time()
        booster = lgb.train({**LGB, "seed": seed, "num_threads": a.threads}, dtrain, num_boost_round=a.rounds,
                            keep_training_booster=True)
        if booster.num_trees() != a.rounds:
            raise SystemExit(f"seed {seed}: {booster.num_trees()} trees, expected {a.rounds}")
        zs[k] = booster.predict(Xn[score], num_threads=a.threads)
        gain += booster.feature_importance("gain")
        if a.save_models:
            mdir = a.out / f"{stage}-{a.split}.models"
            mdir.mkdir(exist_ok=True)
            booster.save_model(str(mdir / f"seed-{seed}.txt"))
        secs.append(round(time.time() - t1, 1))
        print(f"  seed {seed}: {secs[-1]}s", flush=True)
    pred = f.loc[score, [C.ID, "month", "utc_day", "airport", "cls"]].reset_index(drop=True)
    pred["cls"] = pred["cls"].astype(str)
    pred["pred_DH"] = zs.mean(0).astype("float64")
    pred.to_parquet(a.out / f"{stage}-{name}.parquet", index=False)
    np.savez_compressed(a.out / f"{stage}-{name}.seeds.npz", ids=pred[C.ID].to_numpy("int64"), z=zs,
                        seeds=np.array(seeds))
    imp = pd.Series(gain / gain.sum(), index=feats).sort_values(ascending=False)
    tr = f.loc[train]
    rec = {"stage": stage, "split": a.split, "scored": name, "split_token": s.token, "target": "y",
           "class_trained": "cls == clock (2025)", "lightgbm": lgb.__version__, "numpy": np.__version__,
           "pandas": pd.__version__, "python": platform.python_version(),
           "fit": {"params": LGB, "rounds": a.rounds, "seeds": seeds, "num_threads": a.threads,
                   "seconds_per_seed": secs, "categorical": CATS, "smoke": a.smoke},
           "features": feats, "code_tables": tables, "train_rows": int(train.sum()), "opdi": opdi_info,
           "train_rows_by_month": tr.groupby("month").size().to_dict(),
           "train_rows_by_airport": tr.groupby("airport").size().to_dict(),
           "scored_rows": int(score.sum()), "scored_by_cls": pred.groupby("cls").size().to_dict(),
           "pred_sha256": sha(a.out / f"{stage}-{name}.parquet"),
           "inputs": {"features_r4c": sha(C.FEATURES), "features_r5": sha(C.FEATURES5)},
           "importance_gain_top25": imp.head(25).round(5).to_dict(), "code_sha256": code_sha,
           "source_commit": (C.SOURCE / "SOURCE_COMMIT").read_text().strip(), "seconds": round(time.time() - t0)}
    (a.out / f"{stage}-{name}.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {stage}-{name}: {len(pred)} rows; done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
