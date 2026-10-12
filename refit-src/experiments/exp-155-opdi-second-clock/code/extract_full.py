"""OPDI v002 departure-side events for the ten airports, all of 2025 and January/July 2026.

Run in $PRC_OPDI_RAW/ on the workstation (one process, one download stream):
    taskset -c 1,65 nice -n 19 python3 opdi_extract_full.py

For every 10-day event file that overlaps the target periods it keeps, for flights whose
departure airport is one of the ten, the surface events within 8 km of that airport and the
identity events (first_seen, take-off, first-xing-fl50). Flights are taken from the monthly
flight lists by id, so a flight on a window edge keeps its events from both files.
Outputs: full/flights.parquet, full/events_<start>_<end>.parquet, full/manifest.csv.
"""
import datetime as dt
import hashlib
import os
import subprocess
import shutil

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ARP = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619),
       "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4936, -3.5668),
       "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
       "LTFM": (41.2753, 28.7519)}
B = "https://www.eurocontrol.int/performance/data/download/OPDI/v002"
BASE = dt.date(2022, 1, 1)
PERIODS = [(dt.date(2024, 12, 31), dt.date(2026, 2, 2)), (dt.date(2026, 6, 30), dt.date(2026, 8, 2))]
MONTHS = ["202412"] + [f"2025{m:02d}" for m in range(1, 13)] + ["202601", "202602", "202606", "202607", "202608"]
SURFACE = ["exit-parking_position", "entry-parking_position", "exit-apron", "entry-apron",
           "entry-taxiway", "exit-taxiway", "entry-runway", "exit-runway",
           "entry-deicing_pad", "exit-deicing_pad"]
IDENT = ["first_seen", "take-off", "first-xing-fl50"]
os.makedirs("full", exist_ok=True)
manifest = []


def sha(name):
    h = hashlib.sha256()
    with open(name, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def get(url, name):
    cached = os.environ.get("PRC_OPDI_RAW", "/path/to/opdi-raw") + "/" + name
    if not os.path.exists(name) and os.path.isfile(cached):
        shutil.copyfile(cached, name)
    if not os.path.exists(name) or os.path.getsize(name) == 0:
        subprocess.run(["curl", "-sfL", "-m", "3600", "--retry", "3", "-o", name, url])
    ok = os.path.exists(name) and os.path.getsize(name) > 0
    manifest.append((name, os.path.getsize(name) if ok else -1, sha(name) if ok else "", url))
    print("FILE", *manifest[-1][:3], flush=True)
    return ok


fls = []
for ym in MONTHS:
    n = f"flight_list_{ym}.parquet"
    if get(f"{B}/flight_list/{n}", n):
        f = pd.read_parquet(n, columns=["id", "icao24", "flt_id", "dof", "adep", "ades", "registration",
                                        "typecode", "first_seen", "last_seen"])
        for col in ("dof", "first_seen", "last_seen"):
            f[col] = pd.to_datetime(f[col], utc=True, errors="coerce")
        f["id"] = f.id.astype("uint64").view("int64")
        fls.append(f[f.adep.isin(ARP)])
fl = pd.concat(fls).drop_duplicates("id")
fl.to_parquet("full/flights.parquet", index=False, compression="zstd")
print("flights", len(fl), fl.groupby("adep").size().to_dict(), flush=True)
ids = fl.id.to_numpy()
meta = fl.set_index("id")[["adep"]]

ks = sorted({k for a, b in PERIODS for k in range((a - BASE).days // 10, (b - BASE).days // 10 + 1)})
for k in ks:
    s = BASE + dt.timedelta(days=10 * k)
    e = s + dt.timedelta(days=10)
    n = f"flight_events_{s:%Y%m%d}_{e:%Y%m%d}.parquet"
    out = f"full/events_{s:%Y%m%d}_{e:%Y%m%d}.parquet"
    if os.path.exists(out):
        print("SKIP", out, flush=True)
        continue
    if not get(f"{B}/flight_events/{n}", n):
        print("MISSING", n, flush=True)
        continue
    ev = ds.dataset(n).to_table(
        columns=["id", "flight_id", "type", "event_time", "longitude", "latitude", "altitude", "info"],
        filter=ds.field("type").isin(SURFACE + IDENT)).to_pandas()
    ev["flight_id"] = ev.flight_id.astype("uint64").view("int64")
    ev = ev[ev.flight_id.isin(ids)].rename(columns={"id": "event_id"})
    ev = ev.join(meta, on="flight_id")
    la = ev.adep.map(lambda a: ARP[a][0]).to_numpy()
    lo = ev.adep.map(lambda a: ARP[a][1]).to_numpy()
    ev["dist_m"] = np.hypot((ev.longitude.to_numpy() - lo) * 111320 * np.cos(np.radians(la)),
                            (ev.latitude.to_numpy() - la) * 110540).astype("float32")
    ev = ev[ev.type.isin(IDENT) | (ev.dist_m < 8000)]
    ev.to_parquet(out + ".tmp", index=False, compression="zstd")
    os.replace(out + ".tmp", out)
    print("WINDOW", out, len(ev), ev.flight_id.nunique(), flush=True)

pd.DataFrame(manifest, columns=["file", "bytes", "sha256", "url"]).to_csv("full/manifest.csv", index=False)
print("done", flush=True)
