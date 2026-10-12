"""exp-043-pilot: stream one adsb.lol daily globe_history release and keep the EHAM ground slice.

Runs as a Kaggle CPU kernel with internet on.  The laptop's data volume has 4.5 GB
free, so the archive is NEVER written to disk: the two split release assets are read
over HTTP as one continuous byte stream, fed straight into `tarfile` in stream mode
(`r|`), and each member is decompressed, screened and dropped inside the loop.  Peak
disk is the output slice (a few MB) and peak memory is one trace file.

The SHA-256 of each asset is computed on the same bytes that are extracted, and
checked against the digest GitHub publishes in the release metadata.  That is a
stronger provenance statement than hashing a separate on-disk copy, and it is the
only one available without the disk to hold one.

Outputs, in /kaggle/working:
  points_<day>.parquet   kept trace points (on-ground in the box, plus the first
                         airborne point after each ground segment)
  census_<day>.parquet   one row per aircraft-day seen inside the box, ground or not
  run_<day>.json         provenance, member census, timings, screen counts
"""
import gzip
import hashlib
import json
import os
import re
import sys
import tarfile
import time
from collections import Counter

import pandas as pd
import requests

CFG = {
    "days": ["2025-07-07"],
    "repo": "adsblol/globe_history_2025",
    "instance": "planes-readsb-prod-0",
    "lat": [52.25, 52.40],          # EHAM box, assignments/pilot.md step 1
    "lon": [4.65, 4.85],
    "probe_bytes": 200000000,               # >0: read only this prefix of the stream and
                                    # report the archive's shape, then stop
    "out": "/kaggle/working",
}

TIMEOUT = 60
RETRIES = 8
CHUNK = 8 << 20


# ---------------------------------------------------------------- byte stream
def _prefixes(lo: float, hi: float) -> list[bytes]:
    """One-decimal text prefixes covering [lo, hi].

    Every decimal rendering of a number in the range starts with one of these, so
    screening a trace file for them has no false negatives.  False positives (a
    longitude prefix matching inside some other number) only cost a JSON parse.
    """
    a, b = int(lo * 10), int(hi * 10)
    return [f"{n / 10:g}".encode() for n in range(a, b + 1)]


class AssetStream:
    """The release's split assets read as one stream, resumable with Range."""

    def __init__(self, assets, log):
        self.assets, self.log = assets, log
        self.i = 0
        self.resp = None
        self.pos = 0                      # bytes delivered from the current asset
        self.hash = None
        self.done = []                    # {name, bytes, sha256}

    def _open(self, start: int):
        a = self.assets[self.i]
        headers = {"Range": f"bytes={start}-"} if start else {}
        r = requests.get(a["url"], headers=headers, stream=True, timeout=TIMEOUT)
        r.raise_for_status()
        if start and r.status_code != 206:
            raise RuntimeError(f"{a['name']}: server ignored Range (status {r.status_code})")
        self.resp = r

    def read(self, n: int) -> bytes:
        while self.i < len(self.assets):
            a = self.assets[self.i]
            if self.resp is None:
                self.pos, self.hash = 0, hashlib.sha256()
                for k in range(RETRIES):
                    try:
                        self._open(0)
                        break
                    except Exception as e:                      # noqa: BLE001
                        self.log(f"  open {a['name']} failed ({e}); retry {k + 1}")
                        time.sleep(2 * (k + 1))
                else:
                    raise RuntimeError(f"cannot open {a['name']}")
            try:
                b = self.resp.raw.read(n)
            except Exception as e:                              # noqa: BLE001
                self.log(f"  stream broke on {a['name']} at {self.pos}: {e}; resuming")
                b = b""
                for k in range(RETRIES):
                    try:
                        self._open(self.pos)
                        b = self.resp.raw.read(n)
                        break
                    except Exception as e2:                     # noqa: BLE001
                        self.log(f"  resume failed ({e2}); retry {k + 1}")
                        time.sleep(2 * (k + 1))
                else:
                    raise RuntimeError(f"cannot resume {a['name']} at {self.pos}")
            if b:
                self.pos += len(b)
                self.hash.update(b)
                return b
            # asset exhausted, or the connection closed early
            if self.pos < a["size"]:
                self.log(f"  short read on {a['name']}: {self.pos} of {a['size']}; resuming")
                self._open(self.pos)
                continue
            self.done.append({"name": a["name"], "bytes": self.pos,
                              "sha256": self.hash.hexdigest()})
            self.resp.close()
            self.resp = None
            self.i += 1
        return b""


# ------------------------------------------------------------------- decoding
def trace_rows(obj, day_lo, lat_lo, lat_hi, lon_lo, lon_hi):
    """`(kept points, census row)` for one aircraft's trace, or `(None, None)`."""
    t0 = obj.get("timestamp")
    tr = obj.get("trace")
    if t0 is None or not tr:
        return None, None
    icao = obj.get("icao")
    reg, typ = obj.get("r"), obj.get("t")

    n = len(tr)
    lat = [None] * n
    lon = [None] * n
    box = [False] * n
    grd = [False] * n
    call = [None] * n
    for k, p in enumerate(tr):
        try:
            la, lo = p[1], p[2]
        except (IndexError, TypeError):
            continue
        if la is None or lo is None:
            continue
        lat[k], lon[k] = la, lo
        box[k] = lat_lo <= la <= lat_hi and lon_lo <= lo <= lon_hi
        grd[k] = p[3] == "ground"
        ac = p[8] if len(p) > 8 else None
        if isinstance(ac, dict):
            f = ac.get("flight")
            if f:
                call[k] = f.strip() or None
    if not any(box):
        return None, None

    # the aircraft object is written only when it changes, so carry the callsign
    # forward and then backward over the whole trace before anything is dropped
    last = None
    for k in range(n):
        if call[k] is None:
            call[k] = last
        else:
            last = call[k]
    last = None
    for k in range(n - 1, -1, -1):
        if call[k] is None:
            call[k] = last
        else:
            last = call[k]

    keep = []
    for k in range(n):
        if lat[k] is None:
            continue
        first_air = grd[k - 1] and not grd[k] if k else False
        if not ((box[k] and grd[k]) or first_air):
            continue
        p = tr[k]
        keep.append({
            "icao": icao, "reg": reg, "type": typ,
            "t": t0 + p[0], "lat": lat[k], "lon": lon[k],
            "on_ground": grd[k], "in_box": box[k],
            "first_airborne": bool(first_air),
            "alt_ft": None if grd[k] or not isinstance(p[3], (int, float)) else float(p[3]),
            "gs_kt": float(p[4]) if len(p) > 4 and isinstance(p[4], (int, float)) else None,
            "track": float(p[5]) if len(p) > 5 and isinstance(p[5], (int, float)) else None,
            "flags": int(p[6]) if len(p) > 6 and isinstance(p[6], int) else None,
            "callsign": call[k],
            "src": p[9] if len(p) > 9 else None,
            "idx": k,
        })

    ib = [k for k in range(n) if box[k]]
    ibg = [k for k in ib if grd[k]]
    cen = {
        "icao": icao, "reg": reg, "type": typ,
        "n_points": n,
        "n_in_box": len(ib),
        "n_in_box_ground": len(ibg),
        "n_in_box_air": len(ib) - len(ibg),
        "t_first_in_box": t0 + tr[ib[0]][0],
        "t_last_in_box": t0 + tr[ib[-1]][0],
        "t_first_ground": t0 + tr[ibg[0]][0] if ibg else None,
        "t_last_ground": t0 + tr[ibg[-1]][0] if ibg else None,
        "callsigns_in_box": "|".join(sorted({call[k] for k in ib if call[k]})),
        "n_kept": len(keep),
    }
    return keep, cen


# ------------------------------------------------------------------- one day
def assets_for(day: str, log):
    tag = f"v{day.replace('-', '.')}-{CFG['instance']}"
    url = f"https://api.github.com/repos/{CFG['repo']}/releases/tags/{tag}"
    r = requests.get(url, timeout=TIMEOUT, headers={"Accept": "application/vnd.github+json"})
    r.raise_for_status()
    rel = r.json()
    out = [{"name": a["name"], "size": a["size"], "url": a["browser_download_url"],
            "digest": (a.get("digest") or "").split(":")[-1] or None}
           for a in sorted(rel["assets"], key=lambda a: a["name"])]
    log(f"{day}: tag {tag}, {len(out)} assets, "
        f"{sum(a['size'] for a in out) / 1e9:.3f} GB, published {rel['published_at']}")
    for a in out:
        log(f"  {a['name']}  {a['size']}  sha256(published)={a['digest']}")
    return tag, out


def run_day(day: str, log) -> dict:
    t_start = time.time()
    tag, assets = assets_for(day, log)
    lat_lo, lat_hi = CFG["lat"]
    lon_lo, lon_hi = CFG["lon"]
    lat_pre, lon_pre = _prefixes(lat_lo, lat_hi), _prefixes(lon_lo, lon_hi)
    log(f"  screen prefixes lat={lat_pre} lon={lon_pre}")

    st = AssetStream(assets, log)
    probe = int(CFG["probe_bytes"])
    shapes, samples = Counter(), []
    n_mem = n_file = n_gz = n_json = n_hit = 0
    raw_bytes = plain_bytes = 0
    points, census = [], []
    err = Counter()

    tf = tarfile.open(fileobj=st, mode="r|")
    try:
        for m in tf:
            n_mem += 1
            if probe and st.pos > probe:
                log(f"  probe limit {probe} bytes reached after {n_mem} members")
                break
            base = os.path.basename(m.name)
            shapes[re.sub(r"[0-9a-f]{6,}", "<hex>", m.name)] += 1
            if not m.isfile():
                continue
            n_file += 1
            blob = tf.extractfile(m)
            if blob is None:
                continue
            blob = blob.read()
            raw_bytes += len(blob)
            if blob[:2] == b"\x1f\x8b":
                n_gz += 1
                try:
                    blob = gzip.decompress(blob)
                except Exception:                               # noqa: BLE001
                    err["gunzip"] += 1
                    continue
            plain_bytes += len(blob)
            if probe and len(samples) < 6 and base.startswith("trace"):
                samples.append({"member": m.name, "size": m.size,
                                "head": blob[:900].decode("utf-8", "replace")})
            if not (any(p in blob for p in lat_pre) and any(p in blob for p in lon_pre)):
                continue
            n_json += 1
            try:
                obj = json.loads(blob)
            except Exception:                                   # noqa: BLE001
                err["json"] += 1
                continue
            if not isinstance(obj, dict):
                continue
            k, c = trace_rows(obj, day, lat_lo, lat_hi, lon_lo, lon_hi)
            if c is None:
                continue
            n_hit += 1
            points.extend(k)
            census.append(c)
            if n_hit % 200 == 0:
                log(f"  {n_hit} aircraft in box, {n_mem} members, "
                    f"{st.pos / 1e9:.2f} GB of asset {st.i}, {time.time() - t_start:.0f} s")
    finally:
        tf.close()

    out = CFG["out"]
    pdf = pd.DataFrame(points)
    cdf = pd.DataFrame(census)
    if not probe:
        pdf.to_parquet(f"{out}/points_{day}.parquet", index=False)
        cdf.to_parquet(f"{out}/census_{day}.parquet", index=False)

    rec = {
        "day": day, "tag": tag, "box": {"lat": CFG["lat"], "lon": CFG["lon"]},
        "probe_bytes": probe,
        "assets_published": assets,
        "assets_read": st.done,
        "digest_match": [
            {"name": d["name"], "ok": d["sha256"] == a["digest"], "published": a["digest"],
             "computed": d["sha256"], "bytes": d["bytes"], "size_published": a["size"]}
            for d, a in zip(st.done, assets)],
        "members": n_mem, "files": n_file, "gzipped": n_gz,
        "raw_member_bytes": raw_bytes, "decompressed_bytes": plain_bytes,
        "screened_in": n_json, "aircraft_in_box": n_hit,
        "points_kept": int(len(pdf)), "errors": dict(err),
        "member_shapes": dict(shapes.most_common(20)),
        "samples": samples,
        "seconds": round(time.time() - t_start, 1),
    }
    if len(cdf):
        rec["aircraft_with_ground_point"] = int((cdf["n_in_box_ground"] > 0).sum())
        rec["ground_points_in_box"] = int(cdf["n_in_box_ground"].sum())
    with open(f"{out}/run_{day}.json", "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    log(json.dumps({k: v for k, v in rec.items()
                    if k not in ("samples", "assets_published")}, indent=2, default=str))
    return rec


def main() -> None:
    def log(s):
        print(s, flush=True)

    log(f"CFG {json.dumps(CFG)}")
    for day in CFG["days"]:
        run_day(day, log)
    log("done")


if __name__ == "__main__":
    main()
