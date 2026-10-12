#!/usr/bin/env python3
"""Fetch X-Plane Gateway apt.dat files and build a stand-to-runway lookup."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable


AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
MAIN = Path(os.environ.get("PRC_ROOT", Path(__file__).resolve().parents[2])).resolve()
RANKING = MAIN / "data/raw/ranking.parquet"
EARTH_RADIUS_M = 6_371_008.8
PREFIX = re.compile(r"^(?:STAND|GATE|PARKING|RAMP|APRON|POSITION|POS|SPOT)\s*", re.I)
SEPARATORS = re.compile(r"[\s/_-]+")
ZEROES = re.compile(r"(?<=[A-Z])0+(?=\d)")


def fetch_json(url: str, timeout: int, attempts: int = 2) -> dict:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prc-exp127/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last = exc
            print(f"retry {attempt}/{attempts} {url}: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(3)
    assert last is not None
    raise last


def download_airport(icao: str, cache: Path) -> dict:
    dat_path = cache / f"{icao}.dat"
    source_path = cache / f"{icao}.source.json"
    if dat_path.exists() and source_path.exists():
        source = json.loads(source_path.read_text())
        raw = dat_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() == source["sha256"]:
            return source

    timeout = 1_200 if icao == "LTFM" else 300
    airport_obj = fetch_json(f"https://gateway.x-plane.com/apiv1/airport/{icao}", timeout)["airport"]
    sid = airport_obj["recommendedSceneryId"]
    scenery_obj = fetch_json(f"https://gateway.x-plane.com/apiv1/scenery/{sid}", timeout)["scenery"]
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(scenery_obj["masterZipBlob"])))
    names = archive.namelist()
    dat_names = [n for n in names if n.lower().endswith("apt.dat")]
    if not dat_names:
        dat_names = [n for n in names if n.lower().endswith(".dat")]
    if not dat_names:
        raise ValueError(f"No apt.dat file in scenery archive for {icao}")
    dat_name = dat_names[0]
    raw = archive.read(dat_name)
    digest = hashlib.sha256(raw).hexdigest()
    dat_path.write_bytes(raw)
    source = {
        "airport": icao,
        "scenery_id": sid,
        "file": dat_name,
        "bytes": len(raw),
        "sha256": digest,
        "date_accepted": scenery_obj.get("dateAccepted"),
        "author": scenery_obj.get("userName"),
        "license": scenery_obj.get("license"),
        "airport_api_url": f"https://gateway.x-plane.com/apiv1/airport/{icao}",
        "scenery_api_url": f"https://gateway.x-plane.com/apiv1/scenery/{sid}",
    }
    source_path.write_text(json.dumps(source, indent=2) + "\n")
    print(f"fetched {icao} scenery={sid} dat={dat_name} bytes={len(raw)} sha256={digest}", flush=True)
    return source


def parse_dat(icao: str, path: Path) -> tuple[list[dict], list[dict]]:
    stands: list[dict] = []
    runways: list[dict] = []
    text = path.read_text(encoding="latin-1")
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "1300" and len(parts) >= 7:
            stands.append(
                {
                    "airport": icao,
                    "xplane_stand_name": " ".join(parts[6:]),
                    "latitude": float(parts[1]),
                    "longitude": float(parts[2]),
                    "heading": float(parts[3]) % 360.0,
                    "type": parts[4],
                    "aircraft_types": parts[5],
                }
            )
        elif parts[0] == "100" and len(parts) >= 20:
            # A row 100 has 9 common fields, then 9 fields for each runway end.
            for name_i, lat_i, lon_i in ((8, 9, 10), (17, 18, 19)):
                runways.append(
                    {
                        "airport": icao,
                        "runway_end": parts[name_i],
                        "latitude": float(parts[lat_i]),
                        "longitude": float(parts[lon_i]),
                    }
                )
    if not stands:
        raise ValueError(f"No row 1300 stands parsed for {icao} from {path}")
    if not runways:
        raise ValueError(f"No row 100 runway ends parsed for {icao} from {path}")
    return stands, runways


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value.upper().strip())


def no_zeroes(value: str) -> str:
    return ZEROES.sub("", value.upper().strip())


def strip_prefix(value: str) -> str:
    return PREFIX.sub("", value.upper().strip()).strip()


def last_token(value: str) -> str:
    pieces = SEPARATORS.split(strip_prefix(value))
    return pieces[-1] if pieces else ""


RULES: list[tuple[str, Callable[[str], str]]] = [
    ("exact_casefold", lambda s: s.upper().strip()),
    ("strip_facility_prefix", strip_prefix),
    ("compact_spaces", compact),
    ("strip_prefix_compact_spaces", lambda s: compact(strip_prefix(s))),
    ("last_token", last_token),
    ("last_token_zero_normalized", lambda s: no_zeroes(last_token(s))),
    ("strip_prefix_zero_normalized", lambda s: no_zeroes(strip_prefix(s))),
    ("strip_prefix_compact_zero_normalized", lambda s: no_zeroes(compact(strip_prefix(s)))),
]


def choose_match(stand_name: str, xplane_stands: list[dict]) -> tuple[dict | None, str]:
    for rule_name, normalize in RULES:
        key = normalize(stand_name)
        if not key:
            continue
        hits = [x for x in xplane_stands if normalize(x["xplane_stand_name"]) == key]
        if len(hits) == 1:
            return hits[0], rule_name
    return None, "unmatched_or_ambiguous"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def ranking_stands() -> dict[str, list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required from the project venv") from exc
    frame = pq.read_table(RANKING, columns=["ADEP_mvt", "PHASE_mvt", "STAND_mvt"]).to_pandas()
    frame = frame.loc[(frame["PHASE_mvt"] == "DEP") & frame["STAND_mvt"].notna()]
    result: dict[str, list[str]] = {}
    for airport, group in frame.groupby("ADEP_mvt"):
        result[str(airport)] = sorted({str(x).strip() for x in group["STAND_mvt"] if str(x).strip()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory for downloaded source .dat files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    ranking_names = ranking_stands()
    source_rows = []
    geometry_rows = []
    match_summary = []
    for icao in AIRPORTS:
        source = download_airport(icao, args.cache_dir)
        source_rows.append(source)
        xstands, runways = parse_dat(icao, args.cache_dir / f"{icao}.dat")
        names = ranking_names.get(icao, [])
        matched = 0
        for ranking_name in names:
            xstand, match_rule = choose_match(ranking_name, xstands)
            if xstand is None:
                continue
            matched += 1
            for runway in runways:
                geometry_rows.append(
                    {
                        "airport": icao,
                        "stand_name_mvt": ranking_name,
                        "xplane_stand_name": xstand["xplane_stand_name"],
                        "stand_latitude": f"{xstand['latitude']:.8f}",
                        "stand_longitude": f"{xstand['longitude']:.8f}",
                        "heading_true_deg": f"{xstand['heading']:.3f}",
                        "match_rule": match_rule,
                        "runway_end": runway["runway_end"],
                        "runway_end_latitude": f"{runway['latitude']:.8f}",
                        "runway_end_longitude": f"{runway['longitude']:.8f}",
                        "distance_to_runway_end_m": f"{haversine_m(xstand['latitude'], xstand['longitude'], runway['latitude'], runway['longitude']):.3f}",
                        "stand_source": "xplane_gateway",
                        "stand_source_id": source["scenery_id"],
                        "stand_source_url": source["scenery_api_url"],
                        "stand_source_license": "GPL-2.0-or-later",
                        "stand_source_ref_tag": "",
                        "stand_source_sha256": source["sha256"],
                        "scenery_id": source["scenery_id"],
                        "source_sha256": source["sha256"],
                    }
                )
        match_summary.append(
            {
                "airport": icao,
                "ranking_stand_names": len(names),
                "uniquely_matched_names": matched,
                "unique_match_share": round(matched / len(names), 6) if names else None,
                "xplane_stand_rows": len(xstands),
                "runway_ends": len(runways),
            }
        )
        print(f"parsed {icao}: {matched}/{len(names)} unique ranking stand names, {len(xstands)} X-Plane stands, {len(runways)} runway ends", flush=True)

    columns = [
        "airport", "stand_name_mvt", "xplane_stand_name", "stand_latitude", "stand_longitude",
        "heading_true_deg", "match_rule", "runway_end", "runway_end_latitude", "runway_end_longitude",
        "distance_to_runway_end_m", "stand_source", "stand_source_id", "stand_source_url",
        "stand_source_license", "stand_source_ref_tag", "stand_source_sha256", "scenery_id", "source_sha256",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(geometry_rows)
    manifest = {
        "experiment": "exp-127-clock-precision",
        "input": "PRC ranking.parquet departures and their STAND_mvt names",
        "source": "X-Plane Scenery Gateway recommended airport scenery; row 1300 stands and row 100 runway ends",
        "source_license_note": "Gateway upload terms specify GNU GPL version 2; the scenery 105504 pack README also offers version 2 or any later version. Retain GPLv2 terms and check each source pack; see docs/GEOMETRY-DATA.md. The scenery API metadata has no licence field.",
        "format_reference": "https://developer.x-plane.com/article/airport-data-apt-dat-12-00-file-format-specification/",
        "earth_radius_m": EARTH_RADIUS_M,
        "distance_method": "haversine great-circle distance from row 1300 stand coordinates to each row 100 runway-end coordinate",
        "output_format": "long: one record per uniquely matched PRC stand name and runway end",
        "source_files": source_rows,
        "match_summary": match_summary,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(geometry_rows)} stand/runway rows to {args.output}; sha256={manifest['output_sha256']}", flush=True)


if __name__ == "__main__":
    main()
