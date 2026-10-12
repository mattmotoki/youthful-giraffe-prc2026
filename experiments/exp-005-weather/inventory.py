#!/usr/bin/env python3
"""Inventory the CDS ERA5 snapshot against its manifest."""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "external" / "weather" / "copernicus_era5"
RANKING_AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
PRODUCTS = ["core", "extra_2024", "extra_2025", "extra_2026_01_07"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"ERA5 snapshot manifest absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    recs = {Path(r["file"]).name: r for r in manifest["files"]}
    raw = data_dir / "raw"
    rows = []
    for ap in RANKING_AIRPORTS + ["LTAI"]:
        for prod in PRODUCTS:
            name = f"{ap}_core_2024-01-01_2026-07-31.nc" if prod == "core" else f"{ap}_{prod}.nc"
            path = raw / name
            rec = recs.get(name)
            if not path.exists():
                rows.append((ap, prod, "MISSING", "-", "-", "-"))
                continue
            digest = sha256(path)
            if rec is None:
                rows.append((ap, prod, "present", "no-manifest-row", str(path.stat().st_size), f"{digest[:12]}"))
                continue
            ok = "hash-ok" if rec["sha256"] == digest else "HASH-MISMATCH"
            rows.append((ap, prod, "present", ok, f"{rec['hours']}h {rec['first_hour_utc'][:10]}..{rec['last_hour_utc'][:10]}",
                         f"cell=({rec['latitude']},{rec['longitude']})"))
    w = [max(len(str(r[i])) for r in rows) for i in range(6)]
    print(f"{'apt':<{w[0]}}  {'product':<{w[1]}}  {'state':<{w[2]}}  {'check':<{w[3]}}  {'coverage':<{w[4]}}  cell")
    for r in rows:
        print(f"{r[0]:<{w[0]}}  {r[1]:<{w[1]}}  {r[2]:<{w[2]}}  {r[3]:<{w[3]}}  {r[4]:<{w[4]}}  {r[5]}")
    missing = [(a, p) for a, p, s, *_ in rows if s == "MISSING" and a in RANKING_AIRPORTS]
    print(f"\nmanifest rows: {len(manifest['files'])}  raw .nc files: {len(list(raw.glob('*.nc')))}")
    print(f"missing for ranking airports: {missing}")


if __name__ == "__main__":
    main()
