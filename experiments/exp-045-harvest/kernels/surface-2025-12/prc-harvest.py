"""exp-045-harvest: stream one adsb.lol daily globe_history release and keep the
ground slice of TEN airport boxes in a single pass.

This is exp-043-pilot's kernel, extended from one box to ten and from a census to
a product.  What is inherited unchanged: the archive is NEVER written to disk (the
split release assets are read over HTTP as one continuous byte stream, fed straight
into `tarfile` in stream mode, each member decompressed, used and dropped inside the
loop), and the SHA-256 of every asset is computed on the same bytes the filter
consumed and checked against the digest GitHub publishes.

What changed, and why:

* **No text screen.**  The pilot screened each decompressed trace for one-decimal
  text prefixes of its single box before parsing it, and that rejected 21 % of the
  day.  Across ten boxes spanning 40.4 N to 52.4 N and 3.6 W to 28.8 E the union of
  those prefixes contains "49", "50" and "2", which occur in almost every trace, so
  the screen has no discriminating power left; a screen tight enough to discriminate
  (the comma-terminated latitude/longitude pair of each box) has to scan the same
  17 GB with an alternation and costs more than the parse it saves.  Every trace is
  therefore parsed, which also makes the census of aircraft inside a box complete by
  construction rather than by an argument about prefixes.
* **The trace is filtered with numpy, not with a Python loop over every point.**
  Only latitude and longitude are extracted for all 65,000 traces of the day; the
  trace's own bounding box is compared with the ten boxes, and the ~99 % of traces
  that come nowhere near an airport are dropped before any per-point work.  The
  remaining few thousand get the full treatment.
* **The first airborne point after a ground run is kept only when the ground point
  before it is inside a box**, and it is tagged with that box's airport.  The pilot
  kept it unconditionally, which is safe with one box and wrong with ten.
* **The kernel also writes the ground-segment summaries**, using exp-043's frozen
  rule inlined verbatim below, so the points parquet never has to leave Kaggle.

Outputs, in /kaggle/working:
  points_<day>.parquet     kept trace points, tagged with the airport box
  segments_<day>.parquet   one row per aircraft ground segment (the summaries)
  callsigns_<day>.parquet  distinct callsigns seen on the ground, by airport
  census_<day>.parquet     one row per aircraft-day per box, ground or not
  run_<day>.json           provenance, digest check, member census, stage timings
  session.json             which days ran, which failed, and why
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter

import numpy as np
import pandas as pd
import requests

CFG = {
    "days": ["2025-12-01", "2025-12-02", "2025-12-03", "2025-12-04", "2025-12-05", "2025-12-06", "2025-12-07", "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12", "2025-12-13", "2025-12-14", "2025-12-15", "2025-12-16", "2025-12-17", "2025-12-18", "2025-12-19", "2025-12-20", "2025-12-21", "2025-12-22", "2025-12-23", "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2025-12-29", "2025-12-30", "2025-12-31"],
    "boxes": {"EHAM": {"lat": [52.2607, 52.3899], "lon": [4.6647, 4.8472]}, "LFPG": {"lat": [48.9657, 49.0539], "lon": [2.4719, 2.6513]}, "EGLL": {"lat": [51.4376, 51.5049], "lon": [-0.5328, -0.3899]}, "LTFM": {"lat": [41.2265, 41.326], "lon": [28.6708, 28.8359]}, "LEMD": {"lat": [40.4285, 40.5598], "lon": [-3.6115, -3.4971]}, "EDDF": {"lat": [49.9713, 50.073], "lon": [8.4551, 8.629]}, "LEBL": {"lat": [41.2551, 41.3365], "lon": [2.0313, 2.1397]}, "LIRF": {"lat": [41.7548, 41.8732], "lon": [12.1902, 12.3117]}, "EDDM": {"lat": [48.3135, 48.3941], "lon": [11.7104, 11.8618]}, "LSZH": {"lat": [47.4182, 47.5103], "lon": [8.4948, 8.6104]}},
    "budget_s": 39600,          # stop starting days after this much wall clock
    "limit_members": 0,         # >0: smoke test, stop after this many tar members
    "out": "/kaggle/working",
}

TIMEOUT = 60
RETRIES = 8
CHUNK = 8 << 20
DAY_RETRIES = 2                 # a failed day is re-queued once inside the session

try:
    import orjson

    def loads(b):
        return orjson.loads(b)
    JSON_LIB = "orjson"
except ImportError:                                             # pragma: no cover
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "orjson"],
                       check=True, timeout=300)
        import orjson

        def loads(b):
            return orjson.loads(b)
        JSON_LIB = "orjson-installed"
    except Exception:                                           # noqa: BLE001
        def loads(b):
            return json.loads(b)
        JSON_LIB = "json"


# ---------------------------------------------------------------- byte stream
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

    def drain(self):
        """Finish every asset, so each one's SHA-256 covers all of its bytes.

        The tar's end-of-archive marker stops the reader inside the last split
        part; without this the final asset is never hashed at all (exp-043's
        `surface-gate` run verified one of its two assets for that reason)."""
        while self.read(CHUNK):
            pass


# --------------------------------------------------------------- the ten boxes
class Boxes:
    def __init__(self, boxes: dict):
        self.name = list(boxes)
        self.lat_lo = np.array([boxes[k]["lat"][0] for k in self.name])
        self.lat_hi = np.array([boxes[k]["lat"][1] for k in self.name])
        self.lon_lo = np.array([boxes[k]["lon"][0] for k in self.name])
        self.lon_hi = np.array([boxes[k]["lon"][1] for k in self.name])

    def candidates(self, la_lo, la_hi, lo_lo, lo_hi) -> np.ndarray:
        """Boxes whose rectangle overlaps a trace's own bounding box."""
        return np.flatnonzero((self.lat_lo <= la_hi) & (self.lat_hi >= la_lo)
                              & (self.lon_lo <= lo_hi) & (self.lon_hi >= lo_lo))


def _floats(seq) -> np.ndarray:
    """A float array from a column of the trace, with nulls as NaN."""
    try:
        return np.asarray(seq, dtype=np.float64)
    except (TypeError, ValueError):
        return np.array([x if isinstance(x, (int, float)) else np.nan for x in seq],
                        dtype=np.float64)


def trace_rows(obj, boxes: Boxes):
    """`(points, census)` for one aircraft's trace: lists of per-box records."""
    t0 = obj.get("timestamp")
    tr = obj.get("trace")
    if t0 is None or not tr:
        return None, None
    lat = _floats([p[1] for p in tr])
    lon = _floats([p[2] for p in tr])
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        return None, None
    cand = boxes.candidates(lat[ok].min(), lat[ok].max(), lon[ok].min(), lon[ok].max())
    if not len(cand):
        return None, None

    n = len(tr)
    icao = obj.get("icao")
    reg, typ = obj.get("r"), obj.get("t")
    grd = np.fromiter((p[3] == "ground" for p in tr), bool, n)
    gs = _floats([p[4] if len(p) > 4 else None for p in tr])
    trk = _floats([p[5] if len(p) > 5 else None for p in tr])
    alt = _floats([p[3] if isinstance(p[3], (int, float)) else None for p in tr])
    t = t0 + _floats([p[0] for p in tr])

    # readsb writes the aircraft object only when it changes, so carry the callsign
    # forward and then backward over the whole trace before anything is dropped
    call = [None] * n
    for k, p in enumerate(tr):
        ac = p[8] if len(p) > 8 else None
        if isinstance(ac, dict):
            f = ac.get("flight")
            if f:
                call[k] = f.strip() or None
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
    call = np.array(call, dtype=object)

    points, census = [], []
    for c in cand:
        box = (ok & (lat >= boxes.lat_lo[c]) & (lat <= boxes.lat_hi[c])
               & (lon >= boxes.lon_lo[c]) & (lon <= boxes.lon_hi[c]))
        if not box.any():
            continue
        name = boxes.name[c]
        # the first airborne point after a ground run that was inside THIS box
        air = np.zeros(n, bool)
        air[1:] = ~grd[1:] & grd[:-1] & box[:-1]
        keep = (box & grd) | air
        ib = np.flatnonzero(box)
        ibg = np.flatnonzero(box & grd)
        census.append({
            "airport": name, "icao": icao, "reg": reg, "type": typ,
            "n_points": n, "n_in_box": int(len(ib)), "n_in_box_ground": int(len(ibg)),
            "n_in_box_air": int(len(ib) - len(ibg)),
            "t_first_in_box": float(t[ib[0]]), "t_last_in_box": float(t[ib[-1]]),
            "t_first_ground": float(t[ibg[0]]) if len(ibg) else np.nan,
            "t_last_ground": float(t[ibg[-1]]) if len(ibg) else np.nan,
            "callsigns_in_box": "|".join(sorted({x for x in call[ib] if x})),
            "n_kept": int(keep.sum()),
        })
        if not keep.any():
            continue
        k = np.flatnonzero(keep)
        points.append({
            "airport": name, "icao": icao,
            "t": t[k], "lat": lat[k], "lon": lon[k],
            "gs_kt": gs[k], "track": trk[k], "alt_ft": np.where(grd[k], np.nan, alt[k]),
            "on_ground": grd[k], "first_airborne": air[k], "in_box": box[k],
            "callsign": call[k],
        })
    return points, census


def frame(chunks: list[dict]) -> pd.DataFrame:
    """Column-wise concatenation of the per-trace slices (never a row of dicts)."""
    if not chunks:
        return pd.DataFrame(columns=["airport", "icao", "t", "lat", "lon", "gs_kt", "track",
                                     "alt_ft", "on_ground", "first_airborne", "in_box",
                                     "callsign"])
    n = [len(c["t"]) for c in chunks]
    out = {
        "airport": np.repeat([c["airport"] for c in chunks], n),
        "icao": np.repeat([c["icao"] for c in chunks], n),
    }
    for col in ("t", "lat", "lon", "gs_kt", "track", "alt_ft",
                "on_ground", "first_airborne", "in_box", "callsign"):
        out[col] = np.concatenate([c[col] for c in chunks])
    df = pd.DataFrame(out)
    for col in ("gs_kt", "track", "alt_ft"):
        df[col] = df[col].astype("float32")
    df["airport"] = df["airport"].astype("category")
    return df


# ------------------------------------------------------------------- one day
def release_for(day: str) -> tuple[str, str]:
    """`(repo, tag)`.  2025-12-31 is published in the 2026 repository, not the 2025 one."""
    repo = "adsblol/globe_history_2026" if day >= "2025-12-31" else f"adsblol/globe_history_{day[:4]}"
    return repo, f"v{day.replace('-', '.')}-planes-readsb-prod-0"


def assets_for(day: str, log):
    repo, tag = release_for(day)
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    for k in range(RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={"Accept": "application/vnd.github+json"})
            r.raise_for_status()
            break
        except Exception as e:                                  # noqa: BLE001
            log(f"  release metadata failed ({e}); retry {k + 1}")
            time.sleep(3 * (k + 1))
    else:
        raise RuntimeError(f"cannot read release {repo} {tag}")
    rel = r.json()
    out = [{"name": a["name"], "size": a["size"], "url": a["browser_download_url"],
            "digest": (a.get("digest") or "").split(":")[-1] or None}
           for a in sorted(rel["assets"], key=lambda a: a["name"])]
    log(f"{day}: {repo} {tag}, {len(out)} assets, "
        f"{sum(a['size'] for a in out) / 1e9:.3f} GB, published {rel['published_at']}")
    return repo, tag, out


def run_day(day: str, boxes: Boxes, log) -> dict:
    t_start = time.time()
    repo, tag, assets = assets_for(day, log)
    st = AssetStream(assets, log)
    limit = int(CFG["limit_members"])
    shapes = Counter()
    err = Counter()
    n_mem = n_file = n_gz = n_json = n_hit = 0
    raw_bytes = plain_bytes = skipped_bytes = 0
    t_gunzip = t_parse = t_filter = 0.0
    pts, cen = [], []

    tf = tarfile.open(fileobj=st, mode="r|")
    try:
        for m in tf:
            n_mem += 1
            if limit and n_mem > limit:
                log(f"  member limit {limit} reached")
                break
            shapes[re.sub(r"[0-9a-f]{6,}", "<hex>", m.name)] += 1
            if not m.isfile():
                # a trace-named member that is a link or a directory would be data the
                # filter never sees, so count it rather than dropping it silently
                if os.path.basename(m.name).startswith("trace_"):
                    err["trace_not_a_file"] += 1
                continue
            n_file += 1
            # the day also carries ./acas and ./heatmap members, and the heatmaps are
            # 8-17 MB each; name them before reading, so half a GB a day never becomes
            # a Python object
            if not os.path.basename(m.name).startswith("trace"):
                skipped_bytes += m.size
                continue
            blob = tf.extractfile(m)
            if blob is None:
                err["trace_unreadable"] += 1
                continue
            blob = blob.read()
            raw_bytes += len(blob)
            if blob[:2] == b"\x1f\x8b":
                n_gz += 1
                t0 = time.time()
                try:
                    blob = gzip.decompress(blob)
                except Exception:                               # noqa: BLE001
                    err["gunzip"] += 1
                    continue
                t_gunzip += time.time() - t0
            plain_bytes += len(blob)
            n_json += 1
            t0 = time.time()
            try:
                obj = loads(blob)
            except Exception:                                   # noqa: BLE001
                err["json"] += 1
                continue
            t_parse += time.time() - t0
            if not isinstance(obj, dict):
                continue
            t0 = time.time()
            try:
                k, c = trace_rows(obj, boxes)
            except Exception:                                   # noqa: BLE001
                err["trace"] += 1
                continue
            finally:
                t_filter += time.time() - t0
            if not c:
                continue
            n_hit += 1
            pts.extend(k)
            cen.extend(c)
            if n_hit % 2000 == 0:
                log(f"  {n_hit} aircraft in a box, {n_mem} members, "
                    f"{st.pos / 1e9:.2f} GB of asset {st.i}, {time.time() - t_start:.0f} s")
    finally:
        tf.close()
    if not limit:
        st.drain()

    out = CFG["out"]
    pdf = frame(pts)
    cdf = pd.DataFrame(cen)
    t0 = time.time()
    sdf = all_segments(pdf)
    t_seg = time.time() - t0

    g = pdf[pdf.on_ground & pdf.in_box]
    cs = (g.dropna(subset=["callsign"])
          .groupby(["airport", "callsign"], observed=True)
          .agg(n_points=("t", "size"), n_icao=("icao", "nunique"),
               t_first=("t", "min"), t_last=("t", "max"))
          .reset_index()) if len(g) else pd.DataFrame(
        columns=["airport", "callsign", "n_points", "n_icao", "t_first", "t_last"])

    files = {}
    for name, df in (("points", pdf), ("segments", sdf), ("callsigns", cs), ("census", cdf)):
        path = f"{out}/{name}_{day}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        files[name] = {"rows": int(len(df)), "bytes": os.path.getsize(path),
                       "sha256": _sha256(path)}

    by_airport = {}
    for a in boxes.name:
        ga = g[g.airport == a]
        ca = cdf[cdf.airport == a] if len(cdf) else cdf
        sa = sdf[sdf.airport == a] if len(sdf) else sdf
        dt = ga.sort_values("t").groupby("icao", observed=True)["t"].diff().dropna() if len(ga) else pd.Series(dtype=float)
        dt = dt[(dt > 0) & (dt <= 900)]
        by_airport[a] = {
            "aircraft_in_box_any": int(len(ca)),
            "aircraft_in_box_with_ground": int((ca.n_in_box_ground > 0).sum()) if len(ca) else 0,
            "ground_points": int(len(ga)),
            "distinct_callsigns_on_ground": int(ga.callsign.nunique()) if len(ga) else 0,
            "segments": int(len(sa)),
            "departure_segments": int((sa.kind == "departure").sum()) if len(sa) else 0,
            "median_interval_s": float(dt.median()) if len(dt) else None,
        }

    rec = {
        "day": day, "repo": repo, "tag": tag, "json_lib": JSON_LIB,
        "boxes": {k: {"lat": [float(boxes.lat_lo[i]), float(boxes.lat_hi[i])],
                      "lon": [float(boxes.lon_lo[i]), float(boxes.lon_hi[i])]}
                  for i, k in enumerate(boxes.name)},
        "assets_published": assets,
        "assets_read": st.done,
        "digest_match": [
            {"name": d["name"], "ok": d["sha256"] == a["digest"], "published": a["digest"],
             "computed": d["sha256"], "bytes": d["bytes"], "size_published": a["size"]}
            for d, a in zip(st.done, assets)],
        "digests_verified": sum(1 for d, a in zip(st.done, assets) if d["sha256"] == a["digest"]),
        "digests_expected": len(assets),
        "members": n_mem, "files": n_file, "gzipped": n_gz, "traces": n_json,
        "raw_member_bytes": raw_bytes, "decompressed_bytes": plain_bytes,
        "skipped_member_bytes": skipped_bytes,
        "aircraft_in_box": n_hit,
        "outputs": files,
        "by_airport": by_airport,
        "errors": dict(err),
        "member_shapes": dict(shapes.most_common(8)),
        "stage_seconds": {"gunzip": round(t_gunzip, 1), "parse": round(t_parse, 1),
                          "filter": round(t_filter, 1), "segments": round(t_seg, 1)},
        "seconds": round(time.time() - t_start, 1),
    }
    with open(f"{out}/run_{day}.json", "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    log(json.dumps({k: v for k, v in rec.items()
                    if k not in ("assets_published", "member_shapes")}, indent=2, default=str))
    return rec


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    def log(s):
        print(s, flush=True)

    t_session = time.time()
    log(f"CFG days={len(CFG['days'])} boxes={list(CFG['boxes'])} "
        f"budget={CFG['budget_s']} json={JSON_LIB} rule_sha256={RULE_SHA256}")
    boxes = Boxes(CFG["boxes"])
    ses = {"days": CFG["days"], "done": [], "failed": [], "not_started": [],
           "json_lib": JSON_LIB, "rule_sha256": RULE_SHA256}
    last = 0.0
    for i, day in enumerate(CFG["days"]):
        spent = time.time() - t_session
        if spent + max(last, 600.0) * 1.3 > CFG["budget_s"]:
            log(f"budget: {spent:.0f} s spent, stopping before {day}")
            ses["not_started"] = CFG["days"][i:]
            break
        for attempt in range(DAY_RETRIES):
            try:
                rec = run_day(day, boxes, log)
                last = rec["seconds"]
                ses["done"].append({"day": day, "seconds": rec["seconds"],
                                    "digests": [rec["digests_verified"], rec["digests_expected"]],
                                    "points": rec["outputs"]["points"]["rows"],
                                    "segments": rec["outputs"]["segments"]["rows"]})
                break
            except Exception as e:                              # noqa: BLE001
                log(f"{day}: attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                if attempt == DAY_RETRIES - 1:
                    ses["failed"].append({"day": day, "reason": f"{type(e).__name__}: {e}"})
                else:
                    time.sleep(30)
        with open(f"{CFG['out']}/session.json", "w") as fh:
            json.dump(dict(ses, seconds=round(time.time() - t_session, 1)), fh, indent=2)
    ses["seconds"] = round(time.time() - t_session, 1)
    with open(f"{CFG['out']}/session.json", "w") as fh:
        json.dump(ses, fh, indent=2)
    log(f"session: {len(ses['done'])} days done, {len(ses['failed'])} failed, "
        f"{len(ses['not_started'])} not started, {ses['seconds']:.0f} s")
    log("done")



# --- inlined verbatim from experiments/exp-043-pilot/code/segments.py (sha256 f54bad079021f877)
"""exp-043-pilot: turn kept trace points into departure ground segments, and match them.

One "departure event" is a ground -> airborne transition inside an aircraft-day
(the `first_airborne` point the kernel keeps).  Its ground segment is the run of
on-ground points before it with no time gap larger than `GAP`.  From that segment:

  obs_wheels_off  the last on-ground point before the transition
  obs_offblock    the first SUSTAINED motion (ground speed above 3 kt held for at
                  least 30 s) after the last stationary period at the stand.

**Which stationary period is "the stand" is the one real decision here, and the
data settled it.**  Measured on days 1-2, the network sees a departing aircraft for
a median of 150 s before it starts moving and not at all before that: transponders
come on near pushback.  Requiring the assignment's long stand stop (600 s) would
have discarded 95 % of segments for a margin the network does not provide.  But a
segment cannot simply be anchored at its first motion either, because an aircraft
received continuously through its turnaround has the PREVIOUS arrival's taxi-in
inside the same segment.  So:

    anchor = the end of the last stationary run of at least TURNAROUND seconds
             (a turnaround at the stand), or the start of the segment if there is
             none (the ordinary case: the trace begins at the stand)
    obs_offblock = the start of the first sustained motion run at or after anchor
    stand_stop_s = the stationary time actually observed immediately before it

`stand_stop_s` is the confidence margin, not a filter: `strict` keeps only the
segments where it reaches `STAND_STOP`, `loose` keeps every motion onset including
the traces that begin already moving.  The two are reported side by side and never
mixed, and `RESULTS.md` reports the accuracy as a function of the margin, because
the margin is the only thing that separates a seen pushback from a missed one.
"""

import numpy as np
import pandas as pd

GAP = 1800.0        # s: a larger hole in reception splits the ground segment
MOVE_KT = 3.0       # ground speed above this is motion (assignments/pilot.md)
HOLD_S = 30.0       # motion must be held this long to count as sustained
STAND_STOP = 60.0   # observed stationary time that makes a motion onset trustworthy
TURNAROUND = 600.0  # a stationary run at least this long is a stand turnaround
DISP_M = 250.0      # metres from the stand that count as "the aircraft has left it"


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """`[(start, stop_exclusive, value)]` over a boolean array."""
    if not len(mask):
        return []
    cut = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    edges = [0, *cut.tolist(), len(mask)]
    return [(a, b, bool(mask[a])) for a, b in zip(edges[:-1], edges[1:])]


def _moving(gs: np.ndarray, lat: np.ndarray, lon: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Motion per point.  Ground speed where it was received, displacement where
    it was not: a null `gs` is unknown, and calling it stationary would invent a
    stand stop that the network never saw."""
    mv = gs > MOVE_KT
    miss = np.isnan(gs)
    if miss.any() and len(t) > 1:
        # metres per second from the neighbouring positions, 1 kt = 0.5144 m/s
        dx = np.gradient(lon) * 111320.0 * np.cos(np.radians(np.nanmean(lat)))
        dy = np.gradient(lat) * 110540.0
        dt = np.gradient(t)
        with np.errstate(invalid="ignore", divide="ignore"):
            spd = np.hypot(dx, dy) / np.where(dt == 0, np.nan, dt) / 0.5144
        mv[miss] = spd[miss] > MOVE_KT
    return np.nan_to_num(mv, nan=0.0).astype(bool)


def _span(t: np.ndarray, a: int, b: int) -> float:
    """Seconds a run covers, counting the step out of it: a single point that is
    followed 60 s later by another moving point has held motion for 60 s."""
    if b - a >= 2:
        return float(t[b - 1] - t[a])
    return float(t[b] - t[a]) if b < len(t) else 0.0


def segments(points: pd.DataFrame) -> pd.DataFrame:
    """One row per departure event seen in `points` (the kernel's slice)."""
    out = []
    for (icao,), g in points.sort_values("t").groupby(["icao"], sort=False):
        g = g.reset_index(drop=True)
        t = g["t"].to_numpy("float64")
        grd = g["on_ground"].to_numpy(bool)
        air = g["first_airborne"].to_numpy(bool)
        gs = g["gs_kt"].to_numpy("float64")
        lat, lon = g["lat"].to_numpy("float64"), g["lon"].to_numpy("float64")
        for k in np.flatnonzero(air):
            j = k - 1
            if j < 0 or not grd[j]:
                continue
            i = j
            while i > 0 and grd[i - 1] and t[i] - t[i - 1] <= GAP:
                i -= 1
            sl = slice(i, j + 1)
            ts, gss = t[sl], gs[sl]
            mv = _moving(gss, lat[sl], lon[sl], ts)
            runs = _runs(mv)
            sustained = [(a, b) for a, b, v in runs if v and _span(ts, a, b) >= HOLD_S]
            turns = [b for a, b, v in runs if not v and _span(ts, a, b) >= TURNAROUND]
            anchor = turns[-1] - 1 if turns else 0
            after = [a for a, _ in sustained if a >= anchor]
            strict = loose = np.nan
            margin = np.nan
            if after:
                k0 = after[0]
                loose = ts[k0]
                prev = [(a, b) for a, b, v in runs if not v and b <= k0]
                margin = _span(ts, *prev[-1]) if prev else 0.0
                if margin >= STAND_STOP:
                    strict = loose
            # a second, independent onset that never reads the broadcast ground
            # speed: the first time the aircraft is DISP_M metres from where it
            # was standing and never comes back inside that radius.  Ground speed
            # is noisy at a stand (jitter fires a low gate at a random time) while
            # a 50 m displacement is far outside any GPS error.
            disp = np.nan
            if after:
                k0 = after[0]
                st_pts = [k for k in range(anchor, k0) if not mv[k]]
                ref = st_pts if st_pts else [anchor]
                rla = float(np.median(lat[sl][ref]))
                rlo = float(np.median(lon[sl][ref]))
                dx = (lon[sl] - rlo) * 111320.0 * np.cos(np.radians(rla))
                dy = (lat[sl] - rla) * 110540.0
                far = np.hypot(dx, dy) > DISP_M
                far[:anchor] = False
                out_idx = np.flatnonzero(far)
                if len(out_idx):
                    # the LAST re-entry inside the radius is the true departure
                    back = np.flatnonzero(~far[out_idx[0]:]) 
                    k1 = out_idx[0]
                    if len(back):
                        later = np.flatnonzero(far[out_idx[0] + back[-1]:])
                        if len(later):
                            k1 = out_idx[0] + back[-1] + later[0]
                    disp = ts[k1]

            cs = g["callsign"].iloc[sl].dropna()
            out.append({
                "icao": icao,
                "callsign_modal": cs.mode().iat[0] if len(cs) else None,
                "callsign_n": int(cs.nunique()),
                "reg": g["reg"].iloc[j], "type": g["type"].iloc[j],
                "obs_wheels_off": t[j],
                "t_airborne": t[k],
                "seg_start": ts[0], "seg_points": int(j - i + 1),
                "seg_span_s": float(ts[-1] - ts[0]),
                "seg_gap_max_s": float(np.diff(ts).max()) if len(ts) > 1 else 0.0,
                "seg_dt_median_s": float(np.median(np.diff(ts))) if len(ts) > 1 else np.nan,
                "stand_stop_s": margin,
                "turnaround_anchored": bool(turns),
                "obs_offblock_strict": strict,
                "obs_offblock_loose": loose,
                "obs_offblock_disp": disp,
                "n_ground_points": int(grd[sl].sum()),
            })
    return pd.DataFrame(out)

# --- inlined verbatim from experiments/exp-045-harvest/code/allsegments.py (sha256 3959e146b756c4b8)
"""exp-045-harvest: every ground segment, not only the departure ones.

`exp-043-pilot/code/segments.py` is the frozen rule and this file does not touch
it: `all_segments()` calls the pilot's `_runs`, `_moving`, `_span` and its five
constants, and reproduces the pilot's per-segment computation line for line, so a
segment whose kind is `departure` carries exactly the numbers `segments()` gives.
`code/check_rule.py` asserts that equality on the pilot's own stored day before
any kernel runs.

What is added:

* **Arrival and other segments.**  The pilot enumerates ground runs backwards from
  a ground -> airborne transition, so a landing that never departs again inside the
  day is invisible to it.  Here every run of on-ground points (split by a hole
  larger than `GAP`, or by an airborne point) becomes a row, and `kind` says which
  end of the flight it is:

      departure  the run is followed by the kept first-airborne point
      arrival    it is not, it begins at runway speed, and it ends stationary
      other      everything else: a tow, a ground vehicle, a through-taxi, or a
                 run whose two ends were both lost to reception

  `starts_fast` and `ends_stationary` are kept as columns, so a reader can redraw
  the boundary without re-deriving anything.
* **The stand position.**  The median latitude and longitude of the stationary
  points the rule anchored on, which is the position the aircraft pushed back from.
* **The airport.**  Points carry the box they fell in, and segments are cut inside
  `(airport, icao)`, so an aircraft that flies EHAM -> EDDF on the same day gives
  one departure row and one arrival row.
"""

import numpy as np
import pandas as pd


LAND_KT = 30.0      # a run that begins above this speed began as a landing rollout
STOP_KT = 3.0       # a run that ends below this speed ended at a stand


def _segment_row(ts, gss, lat, lon, mv, runs) -> dict:
    """The pilot's rule on one ground run: anchor, onset, margin, stand position."""
    sustained = [(a, b) for a, b, v in runs if v and _span(ts, a, b) >= HOLD_S]
    turns = [b for a, b, v in runs if not v and _span(ts, a, b) >= TURNAROUND]
    anchor = turns[-1] - 1 if turns else 0
    after = [a for a, _ in sustained if a >= anchor]
    strict = loose = disp = margin = np.nan
    stand_lat = stand_lon = np.nan
    if after:
        k0 = after[0]
        loose = ts[k0]
        prev = [(a, b) for a, b, v in runs if not v and b <= k0]
        margin = _span(ts, *prev[-1]) if prev else 0.0
        if margin >= STAND_STOP:
            strict = loose
        st_pts = [k for k in range(anchor, k0) if not mv[k]]
        ref = st_pts if st_pts else [anchor]
        stand_lat = float(np.median(lat[ref]))
        stand_lon = float(np.median(lon[ref]))
        dx = (lon - stand_lon) * 111320.0 * np.cos(np.radians(stand_lat))
        dy = (lat - stand_lat) * 110540.0
        far = np.hypot(dx, dy) > 250.0                      # segments.DISP_M
        far[:anchor] = False
        out_idx = np.flatnonzero(far)
        if len(out_idx):
            back = np.flatnonzero(~far[out_idx[0]:])
            k1 = out_idx[0]
            if len(back):
                later = np.flatnonzero(far[out_idx[0] + back[-1]:])
                if len(later):
                    k1 = out_idx[0] + back[-1] + later[0]
            disp = ts[k1]
    return {
        "obs_offblock_strict": strict,
        "obs_offblock_loose": loose,
        "obs_offblock_disp": disp,
        "stand_stop_s": margin,
        "turnaround_anchored": bool(turns),
        "stand_lat": stand_lat,
        "stand_lon": stand_lon,
    }


def all_segments(points: pd.DataFrame) -> pd.DataFrame:
    """One row per aircraft ground segment in the kernel's slice."""
    out = []
    cols = ["t", "on_ground", "first_airborne", "gs_kt", "lat", "lon", "callsign"]
    for (airport, icao), g in points.sort_values("t").groupby(["airport", "icao"], sort=False, observed=True):
        g = g.reset_index(drop=True)
        t = g["t"].to_numpy("float64")
        grd = g["on_ground"].to_numpy(bool)
        air = g["first_airborne"].to_numpy(bool)
        gs = g["gs_kt"].to_numpy("float64")
        lat, lon = g["lat"].to_numpy("float64"), g["lon"].to_numpy("float64")
        gidx = np.flatnonzero(grd)
        if not len(gidx):
            continue
        # a ground run breaks at an airborne point (index jump) or at a hole > GAP
        brk = np.flatnonzero((np.diff(gidx) > 1) | (np.diff(t[gidx]) > GAP)) + 1
        for a, b in zip([0, *brk.tolist()], [*brk.tolist(), len(gidx)]):
            i, j = gidx[a], gidx[b - 1]
            sl = slice(i, j + 1)
            ts, gss = t[sl], gs[sl]
            la, lo = lat[sl], lon[sl]
            mv = _moving(gss, la, lo, ts)
            runs = _runs(mv)
            row = _segment_row(ts, gss, la, lo, mv, runs)
            ends_air = bool(j + 1 < len(t) and air[j + 1])
            starts_fast = bool(mv[0] and (gss[0] > LAND_KT if np.isfinite(gss[0]) else False))
            ends_stationary = bool(not mv[-1])
            kind = ("departure" if ends_air
                    else "arrival" if (starts_fast and ends_stationary)
                    else "other")
            cs = g["callsign"].iloc[sl].dropna()
            row.update({
                "airport": airport,
                "icao": icao,
                "kind": kind,
                # the two names the assignment asks for, bound to the variants the pilot
                # froze: `obs_offblock` is the 250 m displacement onset (143 s RMSE at
                # EHAM against the block time, versus 184 s for the speed onset,
                # exp-043-pilot/analysis/match.json), and `first_motion` is that speed
                # onset.  Both variants stay in the row, so a later stage can re-choose
                # without re-deriving anything.
                "obs_offblock": row["obs_offblock_disp"],
                "first_motion": row["obs_offblock_loose"],
                "callsign_modal": cs.mode().iat[0] if len(cs) else None,
                "callsign_n": int(cs.nunique()),
                "callsigns_all": "|".join(sorted(cs.unique())) if len(cs) else None,
                "seg_start": float(ts[0]),
                "seg_end": float(ts[-1]),
                "obs_wheels_off": float(ts[-1]) if ends_air else np.nan,
                "t_airborne": float(t[j + 1]) if ends_air else np.nan,
                "seg_points": int(j - i + 1),
                "n_ground_points": int(grd[sl].sum()),
                "seg_span_s": float(ts[-1] - ts[0]),
                "seg_gap_max_s": float(np.diff(ts).max()) if len(ts) > 1 else 0.0,
                "seg_dt_median_s": float(np.median(np.diff(ts))) if len(ts) > 1 else np.nan,
                "starts_fast": starts_fast,
                "ends_stationary": ends_stationary,
                "ends_airborne": ends_air,
            })
            out.append(row)
    order = ["airport", "icao", "callsign_modal", "callsign_n", "callsigns_all", "kind",
             "seg_start", "seg_end", "obs_offblock", "first_motion",
             "obs_offblock_disp", "obs_offblock_strict",
             "obs_offblock_loose", "stand_stop_s", "turnaround_anchored",
             "obs_wheels_off", "t_airborne", "stand_lat", "stand_lon",
             "seg_points", "n_ground_points", "seg_span_s", "seg_gap_max_s",
             "seg_dt_median_s", "starts_fast", "ends_stationary", "ends_airborne"]
    df = pd.DataFrame(out)
    return df[order] if len(df) else pd.DataFrame(columns=order)

RULE_SHA256 = "a8cf1b7441fc51d4a7d118eafc8d42e7770542b5ac30b8f77286f1381337ba09"


if __name__ == "__main__":
    main()
