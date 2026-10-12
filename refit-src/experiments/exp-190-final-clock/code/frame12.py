"""The twelve-month head frame: the nine-month frame plus the clock rows of April, May, September 2025.

    $PY code/frame12.py frame        (then: EXP159_SET=12 $PY code/track.py --months 2025-04 2025-05 2025-09
                                             --out inputs/track-r5.parquet --procs 3)
    $PY code/frame12.py track        (track12 = the nine-month track table + track-r5)

New rows: the clock class of `inputs/features-r5.parquet` (`features12.py`: valid's month builder on
exp-046 blocks of the three months), joined to the raw departures and OPDI exactly as `frame.py`
does. Two things do not exist for the new months and stay NaN: exp-137's row clocks
(`clock_implied_row`, `clock_current250_row`; the proxy kernel never ran on them) and exp-127's
`v23_*` columns (no arm reads them). The old rows are copied from `inputs/frame.parquet` unchanged.
Reads the label of 2025 rows as `y` only. Write-once.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

os.environ["EXP159_SET"] = "12"
import common as C                    # noqa: E402
import frame as FR                    # noqa: E402

OLD_FRAME = C.WORK / "inputs/frame.parquet"
OLD_TRACK = C.WORK / "inputs/track.parquet"
TRACK5 = C.WORK / "inputs/track-r5.parquet"


def build_frame() -> None:
    rec_path = C.FRAME.with_suffix(".json")
    C.refuse_existing(C.FRAME, rec_path)
    t0 = time.time()
    old = pd.read_parquet(OLD_FRAME)
    feat = pd.read_parquet(C.FEATURES5)
    feat = feat[feat.ctx_cls == "clock"]
    months = "2025-" + feat.ctx_month.astype(int).map("{:02d}".format)
    if set(months) != set(C.MONTHS_NEW):
        raise SystemExit(f"new months {sorted(set(months))}")
    keep = [C.ID, "ctx_airport", "ctx_year", "ctx_cls"] + [c for c in feat.columns if c.startswith(("obs_", "shp_"))]
    f = feat[keep].copy()
    f["month"] = months.to_numpy()
    for c in [c for c in old.columns if c.startswith("v23_")] + ["clock_current250_row", "clock_implied_row"]:
        f[c] = np.nan
    raw = FR.raw_departures()
    f = f.merge(raw, on=C.ID, how="left", validate="one_to_one")
    if f[C.TIME].isna().any():
        raise SystemExit("a new clock row has no raw departure")
    t = pd.to_datetime(f[C.TIME], utc=True)
    f["mvt_s"] = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    f["utc_day"] = t.dt.strftime("%Y-%m-%d")
    if not (f.utc_day.str[:7] == f.month).all():
        raise SystemExit("a new row's UTC day is outside its month")
    f = f.rename(columns={"ctx_airport": "airport", "TAXITIME_SEC_mvt": "y"})
    if not (f.airport.astype(str) == f.ADEP_mvt.astype(str)).all():
        raise SystemExit("airport of the feature table and ADEP_mvt differ")
    if f.AIRCRAFT_TYPE_mvt.isna().any() or f.y.isna().any() or f.obs_obs_taxi.isna().any():
        raise SystemExit("a new clock row has no aircraft type, no label or no obs_obs_taxi")
    op = pd.read_parquet(C.OPDI)
    lead = [c for c in op.columns if c.endswith(("_lead_s", "_dist_m"))]
    op.loc[op.duplicate_opdi_identity.astype(bool), lead] = np.nan
    op = op[[C.ID] + lead].rename(columns={c: f"opdi_{c}" for c in lead})
    f = f.merge(op, on=C.ID, how="left", validate="one_to_one")
    if set(f.columns) != set(old.columns):
        raise SystemExit(f"columns differ from the nine-month frame: {sorted(set(f.columns) ^ set(old.columns))}")
    f = f[old.columns]
    for c in old.columns:                                  # the nine-month frame's dtypes
        if f[c].dtype != old[c].dtype:
            f[c] = f[c].astype(old[c].dtype)
    if set(f[C.ID]) & set(old[C.ID]):
        raise SystemExit("a new id is in the nine-month frame")
    out = pd.concat([old, f], ignore_index=True).sort_values(["month", C.ID]).reset_index(drop=True)
    back = out[out.month.isin(old.month.unique())].reset_index(drop=True)
    if not back.equals(old.sort_values(["month", C.ID]).reset_index(drop=True)):
        raise SystemExit("the old rows changed")
    out.to_parquet(C.FRAME, index=False)
    by = out.groupby("month")
    cover = ["obs_obs_taxi", "shp_r_aobt", "clock_implied_row", "opdi_exit_parking_position_lead_s",
             "opdi_exit_apron_lead_s", "opdi_entry_taxiway_lead_s"]
    rec = {"what": "exp-159 twelve-month head frame", "rows": int(len(out)), "new_rows": int(len(f)),
           "columns": int(out.shape[1]), "rows_by_month": by.size().to_dict(),
           "y_mean_by_month": by["y"].mean().round(2).to_dict(),
           "coverage_by_month": {c: by[c].apply(lambda s: float(s.notna().mean())).round(4).to_dict() for c in cover},
           "inputs_sha256": {"frame": C.sha(OLD_FRAME), "features_r5": C.sha(C.FEATURES5), "opdi": C.sha(C.OPDI)},
           "output_sha256": C.sha(C.FRAME), "seconds": round(time.time() - t0, 1),
           "code_sha256": {"frame12.py": C.sha(__file__), "frame.py": C.sha(FR.__file__), "common.py": C.sha(C.__file__)}}
    C.write_json(rec_path, rec)
    print(pd.DataFrame(rec["coverage_by_month"]).assign(rows=pd.Series(rec["rows_by_month"]),
                                                        y=pd.Series(rec["y_mean_by_month"])).to_string())


def build_track() -> None:
    rec_path = C.TRACK.with_suffix(".json")
    C.refuse_existing(C.TRACK, rec_path)
    old, new = pd.read_parquet(OLD_TRACK), pd.read_parquet(TRACK5)
    if list(old.columns) != list(new.columns) or set(old[C.ID]) & set(new[C.ID]):
        raise SystemExit("track tables: columns differ or an id repeats")
    out = pd.concat([old, new.astype(old.dtypes.to_dict())], ignore_index=True)
    ids = pd.read_parquet(C.FRAME, columns=[C.ID, "month"])
    if set(out[C.ID]) != set(ids[C.ID]):
        raise SystemExit("track12 ids are not the frame12 ids")
    out.to_parquet(C.TRACK, index=False)
    j = ids.merge(out, on=C.ID)
    cover = j.groupby("month")[["trk_matched", "trk_implied_taxi", "trk_push_detected"]].agg(
        lambda s: float(s.notna().mean()) if s.name != "trk_matched" else float(s.mean())).round(4)
    C.write_json(rec_path, {"what": "exp-159 twelve-month track table", "rows": int(len(out)),
                            "coverage_by_month": cover.to_dict(),
                            "inputs_sha256": {"track": C.sha(OLD_TRACK), "track_r5": C.sha(TRACK5), "frame12": C.sha(C.FRAME)},
                            "output_sha256": C.sha(C.TRACK), "code_sha256": {"frame12.py": C.sha(__file__)}})
    print(cover.to_string())


if __name__ == "__main__":
    {"frame": build_frame, "track": build_track}[sys.argv[1]]()
