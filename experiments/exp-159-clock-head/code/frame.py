"""Build the head frame: the clock-class rows of the nine block months and the two ranking months.

    nice -n 19 taskset -c 56-59,120-123 $PY code/frame.py

One row per departure of class `clock` (`ctx_cls` of exp-148's feature table). Inputs only, plus
the label `y` on 2025 rows. Writes `inputs/frame.parquet` and `inputs/frame.json` (write-once).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common as C

RAW_COLS = [C.ID, "PHASE_mvt", C.TIME, "TAXITIME_SEC_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt",
            "FLIGHT_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt"]


def raw_departures() -> pd.DataFrame:
    parts = []
    for p in sorted(C.RAW.glob("training_2025-*.parquet")):
        d = pd.read_parquet(p, columns=RAW_COLS)
        parts.append(d[d.PHASE_mvt == "DEP"])
    rk = pd.read_parquet(C.RAW / "ranking.parquet", columns=RAW_COLS)
    rk = rk[rk.PHASE_mvt == "DEP"].copy()
    rk["TAXITIME_SEC_mvt"] = np.nan                       # blank in the file; never a label here
    raw = pd.concat(parts + [rk], ignore_index=True)
    if raw[C.ID].duplicated().any():
        raise ValueError("raw departures: repeated ids")
    return raw.drop(columns="PHASE_mvt")


def main() -> None:
    out, rec_path = C.FRAME, C.FRAME.with_suffix(".json")
    C.refuse_existing(out, rec_path)
    t0 = time.time()
    feat = pd.read_parquet(C.FEATURES)
    feat = feat[feat.ctx_cls == "clock"]
    keep = [C.ID, "ctx_airport", "ctx_year", "ctx_cls"] + \
        [c for c in feat.columns if c.startswith(("obs_", "shp_", "v23_"))]
    feat = feat[keep]
    prox = pd.read_parquet(C.PROXY, columns=[C.ID, "month", "clock_current250_row", "clock_implied_row"])
    f = feat.merge(prox, on=C.ID, how="left", validate="one_to_one")
    if f.month.isna().any():
        raise ValueError("a clock row has no proxy row")
    raw = raw_departures()
    f = f.merge(raw, on=C.ID, how="left", validate="one_to_one")
    if f[C.TIME].isna().any():
        raise ValueError("a clock row has no raw departure")
    t = pd.to_datetime(f[C.TIME], utc=True)
    f["mvt_s"] = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    f["utc_day"] = t.dt.strftime("%Y-%m-%d")
    f = f.rename(columns={"ctx_airport": "airport", "TAXITIME_SEC_mvt": "y"})
    if not (f.airport.astype(str) == f.ADEP_mvt.astype(str)).all():
        raise ValueError("airport of the feature table and ADEP_mvt differ")
    if f.AIRCRAFT_TYPE_mvt.isna().any():
        raise ValueError("a clock row has no aircraft type")
    is26 = f.month.isin(C.MONTHS_2026)
    if f.loc[~is26, "y"].isna().any() or f.loc[is26, "y"].notna().any():
        raise ValueError("labels: a 2025 clock row has none, or a 2026 row has one")
    if f.obs_obs_taxi.isna().any():
        raise ValueError("a clock row has no obs_obs_taxi")

    op = pd.read_parquet(C.OPDI)
    lead = [c for c in op.columns if c.endswith(("_lead_s", "_dist_m"))]
    op.loc[op.duplicate_opdi_identity.astype(bool), lead] = np.nan
    op = op[[C.ID] + lead].rename(columns={c: f"opdi_{c}" for c in lead})
    f = f.merge(op, on=C.ID, how="left", validate="one_to_one")
    for c in C.OPDI_RAW:
        if c not in f:
            raise ValueError(f"OPDI column {c} is missing")

    f = f.sort_values(["month", C.ID]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, index=False)
    by = f.groupby("month")
    rec = {"what": "exp-159 head frame: clock-class rows, inputs and the 2025 label",
           "rows": int(len(f)), "columns": int(f.shape[1]),
           "rows_by_month": by.size().to_dict(),
           "coverage_by_month": {c: by[c].apply(lambda s: float(s.notna().mean())).round(4).to_dict()
                                 for c in ["clock_implied_row", "clock_current250_row",
                                           "v23_implied_offblock_to_current250_s",
                                           "opdi_exit_parking_position_lead_s", "opdi_exit_apron_lead_s",
                                           "opdi_entry_taxiway_lead_s"]},
           "inputs_sha256": {"features": C.sha(C.FEATURES), "proxy": C.sha(C.PROXY), "opdi": C.sha(C.OPDI),
                             "raw": {p.name: C.sha(p) for p in sorted(C.RAW.glob("training_2025-*.parquet"))}
                             | {"ranking.parquet": C.sha(C.RAW / "ranking.parquet")}},
           "output_sha256": C.sha(out), "seconds": round(time.time() - t0, 1),
           "code_sha256": {"frame.py": C.sha(__file__), "common.py": C.sha(C.__file__)}}
    C.write_json(rec_path, rec)
    print(pd.DataFrame(rec["coverage_by_month"]).assign(rows=pd.Series(rec["rows_by_month"])).to_string())


if __name__ == "__main__":
    main()
