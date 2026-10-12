"""exp-045-harvest: size the ten airport boxes from open runway geometry.

Source: OurAirports `runways.csv` and `airports.csv` (public domain, no account,
<https://ourairports.com/data/>, mirror <https://davidmegginson.github.io/ourairports-data/>).
The box of an airport is the bounding box of every open runway endpoint plus the
airport reference point, grown by MARGIN_M metres on all four sides.  The margin
carries the aprons, the terminal piers and the de-icing pads, which sit inside or
immediately beside the runway hull at all ten fields.

Writes `analysis/boxes.json` (the boxes the kernel reads) and
`analysis/runways-10.csv` (the source rows actually used, so the box is
reproducible from the repo without re-downloading 17 MB).

    python3 code/boxes.py <runways.csv> <airports.csv>
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

IDENTS = ["EHAM", "LFPG", "EGLL", "LTFM", "LEMD", "EDDF", "LEBL", "LIRF", "EDDM", "LSZH"]
MARGIN_M = 3000.0
SOURCE = {
    "provider": "OurAirports",
    "licence": "public domain (https://ourairports.com/data/)",
    "runways_url": "https://davidmegginson.github.io/ourairports-data/runways.csv",
    "airports_url": "https://davidmegginson.github.io/ourairports-data/airports.csv",
}
EXP = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(runways: str, airports: str, retrieved: str) -> None:
    rw_path, ap_path = Path(runways), Path(airports)
    rw_rows = [r for r in csv.DictReader(open(rw_path)) if r["airport_ident"] in IDENTS]
    ap_rows = {r["ident"]: r for r in csv.DictReader(open(ap_path)) if r["ident"] in IDENTS}
    assert len(ap_rows) == len(IDENTS), sorted(set(IDENTS) - set(ap_rows))

    out = {}
    for ident in IDENTS:
        pts = [(float(ap_rows[ident]["latitude_deg"]), float(ap_rows[ident]["longitude_deg"]))]
        used = 0
        for r in rw_rows:
            if r["airport_ident"] != ident or r["closed"] == "1":
                continue
            for a, o in (("le_latitude_deg", "le_longitude_deg"), ("he_latitude_deg", "he_longitude_deg")):
                if r[a] and r[o]:
                    pts.append((float(r[a]), float(r[o])))
                    used += 1
        lat = [p[0] for p in pts]
        lon = [p[1] for p in pts]
        mid = (min(lat) + max(lat)) / 2.0
        dlat = MARGIN_M / 110540.0
        dlon = MARGIN_M / (111320.0 * math.cos(math.radians(mid)))
        # round outward, so the stored box is never smaller than the computed one
        box = {
            "lat": [math.floor((min(lat) - dlat) * 1e4) / 1e4, math.ceil((max(lat) + dlat) * 1e4) / 1e4],
            "lon": [math.floor((min(lon) - dlon) * 1e4) / 1e4, math.ceil((max(lon) + dlon) * 1e4) / 1e4],
        }
        box["hull"] = {"lat": [min(lat), max(lat)], "lon": [min(lon), max(lon)]}
        box["runway_endpoints"] = used
        box["km"] = [
            round((box["lat"][1] - box["lat"][0]) * 110.54, 2),
            round((box["lon"][1] - box["lon"][0]) * 111.32 * math.cos(math.radians(mid)), 2),
        ]
        out[ident] = box

    rec = {
        "built_on": retrieved,
        "margin_m": MARGIN_M,
        "source": dict(SOURCE, runways_sha256=sha256(rw_path), airports_sha256=sha256(ap_path),
                       runways_bytes=rw_path.stat().st_size, airports_bytes=ap_path.stat().st_size,
                       retrieved=retrieved),
        "boxes": out,
    }
    (EXP / "analysis").mkdir(parents=True, exist_ok=True)
    with open(EXP / "analysis/boxes.json", "w") as fh:
        json.dump(rec, fh, indent=2)
    with open(EXP / "analysis/runways-10.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rw_rows[0].keys()))
        w.writeheader()
        w.writerows(rw_rows)
    for k, v in out.items():
        print(f"{k}  lat {v['lat']}  lon {v['lon']}  {v['km'][0]} x {v['km'][1]} km  "
              f"{v['runway_endpoints']} endpoints")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "2026-09-08")
