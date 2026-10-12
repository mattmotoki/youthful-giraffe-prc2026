"""exp-185: fill a Gateway cache folder for exp-127 build_stand_geometry.py with the PINNED sceneries.

    .venv/bin/python prefetch_gateway.py <exp-127 manifest.json> <cache_dir>

build_stand_geometry.py downloads each airport's CURRENT recommended scenery unless its cache folder holds
`<icao>.dat` and `<icao>.source.json` whose SHA-256 agree. This script downloads the scenery id that the
saved manifest names (`source_files`), checks the apt.dat SHA-256 against the manifest, and writes the two
files in the builder's own form, so the builder reads the same sources as the saved stand_geometry.csv.
"""
import base64
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

man, cache = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
cache.mkdir(parents=True, exist_ok=True)
for src in man["source_files"]:
    icao, sid = src["airport"], src["scenery_id"]
    dat, meta = cache / f"{icao}.dat", cache / f"{icao}.source.json"
    if dat.exists() and hashlib.sha256(dat.read_bytes()).hexdigest() == src["sha256"]:
        print(f"{icao} cached", flush=True)
        continue
    for attempt in range(3):
        try:
            req = urllib.request.Request(src["scenery_api_url"], headers={"User-Agent": "prc-exp127/1.0"})
            with urllib.request.urlopen(req, timeout=1200) as r:
                obj = json.load(r)["scenery"]
            break
        except Exception as exc:  # network retry, then fail
            print(f"retry {icao} {exc}", flush=True)
            time.sleep(5)
    else:
        raise SystemExit(f"{icao}: scenery {sid} not fetched")
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(obj["masterZipBlob"])))
    names = [n for n in z.namelist() if n.lower().endswith("apt.dat")] or [n for n in z.namelist() if n.lower().endswith(".dat")]
    raw = z.read(names[0])
    got = hashlib.sha256(raw).hexdigest()
    if got != src["sha256"] or len(raw) != src["bytes"]:
        raise SystemExit(f"{icao}: scenery {sid} apt.dat sha {got} bytes {len(raw)} != manifest {src['sha256']} {src['bytes']}")
    dat.write_bytes(raw)
    meta.write_text(json.dumps(src, indent=2) + "\n")
    print(f"{icao} scenery {sid} sha {got[:12]} ok", flush=True)
