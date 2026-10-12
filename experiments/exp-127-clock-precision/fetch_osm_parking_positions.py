#!/usr/bin/env python3
"""Fetch OSM aeroway=parking_position nodes with reference tags at LTFM and LSZH."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


ENDPOINT = "https://overpass-api.de/api/interpreter"
AREAS = {
    "LTFM": (41.20, 28.60, 41.40, 28.92),
    "LSZH": (47.38, 8.45, 47.55, 8.65),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    boxes = ";\n  ".join(
        f'node["aeroway"="parking_position"]({south},{west},{north},{east})'
        for south, west, north, east in AREAS.values()
    )
    query = f"[out:json][timeout:180];\n(\n  {boxes};\n);\nout body;"
    request = Request(
        ENDPOINT,
        data=("data=" + quote_plus(query)).encode(),
        headers={
            "User-Agent": "prc-exp127/1.0 (X-Plane stand-position research)",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urlopen(request, timeout=240) as response:
        raw = response.read()
    payload = json.loads(raw)
    rows = []
    for node in payload.get("elements", []):
        if node.get("type") != "node":
            continue
        lat, lon = float(node["lat"]), float(node["lon"])
        airport = next(
            (
                code
                for code, (south, west, north, east) in AREAS.items()
                if south <= lat <= north and west <= lon <= east
            ),
            None,
        )
        if not airport:
            continue
        tags = node.get("tags", {})
        refs = {k: v for k, v in tags.items() if k == "ref" or k.startswith("ref:")}
        if not refs:
            continue
        rows.append(
            {
                "airport": airport,
                "osm_node_id": node["id"],
                "latitude": f"{lat:.8f}",
                "longitude": f"{lon:.8f}",
                "ref_tags_json": json.dumps(refs, sort_keys=True, ensure_ascii=False),
                "name": tags.get("name", ""),
                "source_url": f"https://www.openstreetmap.org/node/{node['id']}",
                "source_license": "ODbL-1.0",
            }
        )
    rows.sort(key=lambda r: (r["airport"], int(r["osm_node_id"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["airport", "osm_node_id", "latitude", "longitude", "ref_tags_json", "name", "source_url", "source_license"]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "endpoint": ENDPOINT,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "element_count": len(rows),
        "counts_by_airport": {a: sum(r["airport"] == a for r in rows) for a in AREAS},
        "all_node_count_before_ref_filter": sum(x.get("type") == "node" for x in payload.get("elements", [])),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "license": "OpenStreetMap data is available under ODbL 1.0; attribute OpenStreetMap contributors.",
        "copyright_url": "https://www.openstreetmap.org/copyright",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
