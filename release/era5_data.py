#!/usr/bin/env python3
"""Pack or verify the separate CC BY ERA5 table required by v32."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "release" / "ERA5-DATA-MANIFEST.json"
NOTICE = """Copernicus ERA5 data bundle for PRC Data Challenge 2026 v32

Data licence: Creative Commons Attribution 4.0 (CC BY 4.0).
https://creativecommons.org/licenses/by/4.0/

Contains modified Copernicus Climate Change Service information 2026.
Neither the European Commission nor ECMWF is responsible for any use of this information.

The table is derived from the two products and exact frozen retrieval described in
docs/weather-provenance.md. It remains under its upstream data licence. The repository's
GNU GPLv3 licence applies to source code, not to this data file.
"""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, spec: dict) -> dict:
    if not path.is_file():
        raise ValueError(f"exact ERA5 table absent: {path}")
    size = path.stat().st_size
    got = digest(path)
    if size != spec["bytes"] or got != spec["sha256"]:
        raise ValueError(f"ERA5 custody mismatch: bytes={size}, sha256={got}")

    import pandas as pd
    table = pd.read_parquet(path)
    columns = list(table.columns)
    forbidden_rx = re.compile(
        r"(^|_)(mvt_?id|movement_?id|taxitime|taxi_?time|block_?time|target|label|y_?true)($|_)",
        re.IGNORECASE,
    )
    forbidden = sorted(c for c in columns if forbidden_rx.search(c))
    if forbidden:
        raise ValueError(f"competition-row or label columns present: {forbidden}")
    if len(table) != spec["rows"]:
        raise ValueError(f"wrong row count: {len(table)}")
    got_schema = [{"name": c, "dtype": str(table[c].dtype)} for c in columns]
    if got_schema != spec["schema"]:
        raise ValueError(f"wrong ordered schema: {got_schema}")
    version = table.attrs.get("weather_features_version")
    if version != spec["weather_features_version"]:
        raise ValueError(
            "weather_features_version metadata mismatch: "
            f"expected={spec['weather_features_version']!r}, actual={version!r}"
        )
    airports = sorted(table["airport"].dropna().unique().tolist())
    if airports != spec["airports"]:
        raise ValueError(f"wrong airports: {airports}")
    if table.duplicated(spec["key"]).any():
        raise ValueError("duplicate airport-hour keys")
    return {"bytes": size, "sha256": got, "rows": len(table), "columns": columns,
            "weather_features_version": version}


def verify_bundle(bundle: Path, spec: dict) -> dict:
    if not bundle.is_dir():
        raise ValueError(f"ERA5 bundle absent: {bundle}")
    expected = {spec["file"], MANIFEST_PATH.name, "NOTICE.txt"}
    actual = {p.name for p in bundle.iterdir()}
    if actual != expected:
        raise ValueError(f"unexpected bundle contents: expected={sorted(expected)}, actual={sorted(actual)}")
    result = verify_file(bundle / spec["file"], spec)
    if json.loads((bundle / MANIFEST_PATH.name).read_text()) != spec:
        raise ValueError("bundle manifest differs from the source manifest")
    if (bundle / "NOTICE.txt").read_text() != NOTICE:
        raise ValueError("bundle notice differs from the required CC BY notice")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("source", type=Path)
    pack.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    args = parser.parse_args()
    spec = json.loads(MANIFEST_PATH.read_text())

    if args.command == "pack":
        result = verify_file(args.source, spec)
        out = args.out.resolve()
        if out.exists():
            raise SystemExit(f"refusing to overwrite existing bundle: {out}")
        out.mkdir(parents=True)
        shutil.copyfile(args.source, out / spec["file"])
        shutil.copyfile(MANIFEST_PATH, out / MANIFEST_PATH.name)
        (out / "NOTICE.txt").write_text(NOTICE)
        verify_bundle(out, spec)
        print(json.dumps({"status": "PASS", "bundle": str(out), **result}, sort_keys=True))
        return 0

    result = verify_bundle(args.bundle, spec)
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
