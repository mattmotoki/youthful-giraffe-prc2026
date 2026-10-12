#!/usr/bin/env python3
"""Add unambiguous OSM ref-tag parking positions missing from the Gateway map."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from build_stand_geometry import RULES, haversine_m, ranking_stands


EXPERIMENT = Path(__file__).resolve().parent


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--osm-nodes", type=Path, required=True)
    parser.add_argument("--osm-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    current = read_rows(args.geometry)
    fields = list(current[0])
    gateway_rows = [r for r in current if r.get("stand_source", "xplane_gateway") == "xplane_gateway"]
    gateway_names = defaultdict(set)
    runway_ends: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in gateway_rows:
        gateway_names[row["airport"]].add(row["stand_name_mvt"])
        runway_ends[row["airport"]][row["runway_end"]] = {
            "runway_end_latitude": row["runway_end_latitude"],
            "runway_end_longitude": row["runway_end_longitude"],
            "scenery_id": row["scenery_id"],
            "source_sha256": row["source_sha256"],
        }

    osm_nodes = read_rows(args.osm_nodes)
    nodes_by_airport: dict[str, list[dict[str, str]]] = defaultdict(list)
    for node in osm_nodes:
        node["refs"] = json.loads(node.pop("ref_tags_json"))
        nodes_by_airport[node["airport"]].append(node)

    ranking_names = ranking_stands()
    added_rows: list[dict[str, str]] = []
    summary = {}
    for airport in ("LTFM", "LSZH"):
        added_names: set[str] = set()
        ambiguous_names: set[str] = set()
        for stand_name in ranking_names.get(airport, []):
            if stand_name in gateway_names[airport]:
                continue
            chosen = None
            chosen_rule = ""
            for rule, normalize in RULES:
                hits: dict[str, tuple[dict[str, str], str, str]] = {}
                for node in nodes_by_airport.get(airport, []):
                    for tag, value in node["refs"].items():
                        if normalize(value) == normalize(stand_name):
                            hits[node["osm_node_id"]] = (node, tag, value)
                if len(hits) == 1:
                    chosen = next(iter(hits.values()))
                    chosen_rule = rule
                    break
                if len(hits) > 1:
                    ambiguous_names.add(stand_name)
            if not chosen:
                continue
            node, ref_tag, ref_value = chosen
            for runway_name, runway in sorted(runway_ends[airport].items()):
                row = {name: "" for name in fields}
                row.update(
                    {
                        "airport": airport,
                        "stand_name_mvt": stand_name,
                        "xplane_stand_name": "",
                        "stand_latitude": node["latitude"],
                        "stand_longitude": node["longitude"],
                        "heading_true_deg": "",
                        "match_rule": f"osm_ref_{chosen_rule}",
                        "runway_end": runway_name,
                        "runway_end_latitude": runway["runway_end_latitude"],
                        "runway_end_longitude": runway["runway_end_longitude"],
                        "distance_to_runway_end_m": f"{haversine_m(float(node['latitude']), float(node['longitude']), float(runway['runway_end_latitude']), float(runway['runway_end_longitude'])):.3f}",
                        "stand_source": "openstreetmap",
                        "stand_source_id": node["osm_node_id"],
                        "stand_source_url": node["source_url"],
                        "stand_source_license": "ODbL-1.0",
                        "stand_source_ref_tag": f"{ref_tag}={ref_value}",
                        "stand_source_sha256": json.loads(args.osm_manifest.read_text())["response_sha256"],
                        "scenery_id": runway["scenery_id"],
                        "source_sha256": runway["source_sha256"],
                    }
                )
                added_rows.append(row)
            added_names.add(stand_name)
        summary[airport] = {
            "gateway_uniquely_matched_names": len(gateway_names[airport]),
            "osm_added_names": len(added_names),
            "osm_unique_total_names": len(gateway_names[airport]) + len(added_names),
            "ranking_stand_names": len(ranking_names.get(airport, [])),
            "ambiguous_names_seen": len(ambiguous_names),
            "added_stand_names": sorted(added_names),
        }

    merged = gateway_rows + added_rows
    merged.sort(key=lambda r: (r["airport"], r["stand_name_mvt"], r["runway_end"], r["stand_source"]))
    write_rows(args.geometry, merged, fields)

    manifest = json.loads(args.manifest.read_text())
    osm_manifest = json.loads(args.osm_manifest.read_text())
    manifest["osm_fallback"] = {
        "endpoint": osm_manifest["endpoint"],
        "retrieved_utc": osm_manifest["retrieved_utc"],
        "query": osm_manifest["query"],
        "response_sha256": osm_manifest["response_sha256"],
        "output_sha256": osm_manifest["output_sha256"],
        "license": osm_manifest["license"],
        "match_summary": summary,
        "method": "unique normalized match against OSM ref/ref:* tags; add only ranking stand names not already matched by Gateway",
    }
    manifest["combined_unique_matches"] = {
        airport: summary[airport]["osm_unique_total_names"] for airport in ("LTFM", "LSZH")
    }
    import hashlib

    manifest["output_sha256"] = hashlib.sha256(args.geometry.read_bytes()).hexdigest()
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"added {len(added_rows)} stand/runway rows; total geometry rows={len(merged)}")


if __name__ == "__main__":
    main()
