"""exp-185: OPDI v002 departure-side events for the February and June 2026 windows.

The processing is exp-155 `code/extract_full.py` (branch exp155-opdi-second-clock, remote copy
$PRC_OPDI_ROOT/code/extract_full.py) line for line. Changes: the windows are given on
the command line (the 10-day files that exp-155 did not hold), the output folder is an argument,
and a raw file that already exists in the exp-155 folder is linked, not copied or downloaded again.
MONTHS is unchanged, so `flights.parquet` must equal the exp-155 one byte for byte.

Run in the work folder (it holds the raw files and the links to them):
    python3 extract_final.py <out_dir> <window_start> [<window_start> ...]
e.g. python3 extract_final.py full 20260209 20260219 20260301 20260530 20260609 20260619
"""
import datetime as dt
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ARP = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619),
       "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4936, -3.5668),
       "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
       "LTFM": (41.2753, 28.7519)}
B = "https://www.eurocontrol.int/performance/data/download/OPDI/v002"
BASE = dt.date(2022, 1, 1)
MONTHS = ["202412"] + [f"2025{m:02d}" for m in range(1, 13)] + ["202601", "202602", "202606", "202607", "202608"]
SURFACE = ["exit-parking_position", "entry-parking_position", "exit-apron", "entry-apron",
           "entry-taxiway", "exit-taxiway", "entry-runway", "exit-runway",
           "entry-deicing_pad", "exit-deicing_pad"]
IDENT = ["first_seen", "take-off", "first-xing-fl50"]
EXP155 = os.environ.get("PRC_OPDI_ROOT", "/path/to/prc-exp155-opdi") + "/"
OUT = sys.argv[1]
WINDOWS = [dt.datetime.strptime(w, "%Y%m%d").date() for w in sys.argv[2:]]
assert WINDOWS and all((w - BASE).days % 10 == 0 for w in WINDOWS), "window starts must be on the 10-day grid"
os.makedirs(OUT, exist_ok=True)
manifest = []


def sha(name):
    h = hashlib.sha256()
    with open(name, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def get(url, name):
    if not os.path.exists(name) and os.path.isfile(EXP155 + name):
        os.symlink(EXP155 + name, name)
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
fl.to_parquet(f"{OUT}/flights.parquet", index=False, compression="zstd")
print("flights", len(fl), fl.groupby("adep").size().to_dict(), flush=True)
ids = fl.id.to_numpy()
meta = fl.set_index("id")[["adep"]]

for s in WINDOWS:
    e = s + dt.timedelta(days=10)
    n = f"flight_events_{s:%Y%m%d}_{e:%Y%m%d}.parquet"
    out = f"{OUT}/events_{s:%Y%m%d}_{e:%Y%m%d}.parquet"
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

pd.DataFrame(manifest, columns=["file", "bytes", "sha256", "url"]).to_csv(f"{OUT}/manifest.csv", index=False)
print("done", flush=True)
