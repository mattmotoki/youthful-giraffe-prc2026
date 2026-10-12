"""exp-045-harvest: re-derive one month's ground-segment summaries from its stored points.

The assignment provides for exactly this: "if the pilot changes the rule, re-derive the
summaries from the stored points rather than re-fetching".  The pilot's reviewer repaired
the rule on 2026-09-08 (a first-airborne point with readsb's null altitude is not evidence
of a take-off), after four months had already been harvested under the shipped rule.  The
points are untouched by the change — the repair reads `alt_ft`, which every points file
carries — so this kernel mounts a fetch kernel's output, applies the current rule to it,
and writes the summaries again.  Nothing is re-fetched and no archive is read.

The mounted month is whatever `kernel_sources` names in the metadata; the code takes every
`points_*.parquet` under /kaggle/input.

Outputs, in /kaggle/working:
  segments_<day>.parquet   the summaries under the current rule
  callsigns_<day>.parquet  the ground callsign table, unchanged in definition
  rederive_<day>.json      row counts, the rule hash, and what the rule changed
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

CFG = {"out": "/kaggle/working", "input": "/kaggle/input"}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = sorted(glob.glob(f"{CFG['input']}/*/points_*.parquet"))
    print(f"rule {RULE_VERSION} {RULE_SHA256}  REQUIRE_ALT={REQUIRE_ALT}  "
          f"{len(files)} points files", flush=True)
    out = CFG["out"]
    total = {"days": 0, "segments": 0, "departures": 0, "dropped_by_require_alt": 0}
    for path in files:
        t0 = time.time()
        day = os.path.basename(path)[7:17]
        pdf = pd.read_parquet(path)
        sdf = all_segments(pdf)
        # what the repair actually changed on this day, priced rather than argued
        raw_dep = int((sdf.ends_airborne | (sdf.airborne_alt_ok != sdf.ends_airborne)).sum())
        dropped = int(((~sdf.ends_airborne) & sdf.kind.ne("departure")
                       & sdf.airborne_alt_ok.eq(False)).sum()) if REQUIRE_ALT else 0
        g = pdf[pdf.on_ground & pdf.in_box]
        cs = (g.dropna(subset=["callsign"])
              .groupby(["airport", "callsign"], observed=True)
              .agg(n_points=("t", "size"), n_icao=("icao", "nunique"),
                   t_first=("t", "min"), t_last=("t", "max")).reset_index())
        files_out = {}
        for name, df in (("segments", sdf), ("callsigns", cs)):
            p = f"{out}/{name}_{day}.parquet"
            df.to_parquet(p, index=False, compression="zstd")
            files_out[name] = {"rows": int(len(df)), "bytes": os.path.getsize(p),
                               "sha256": _sha256(p)}
        rec = {"day": day, "source": path, "rule_sha256": RULE_SHA256,
               "rule_version": RULE_VERSION,
               "require_alt": bool(REQUIRE_ALT),
               "points_rows": int(len(pdf)), "outputs": files_out,
               "departures": int((sdf.kind == "departure").sum()),
               "segments_without_a_real_altitude": int((~sdf.airborne_alt_ok
                                                        & sdf.ends_stationary.notna()).sum()),
               "seconds": round(time.time() - t0, 1)}
        with open(f"{out}/rederive_{day}.json", "w") as fh:
            json.dump(rec, fh, indent=2)
        total["days"] += 1
        total["segments"] += len(sdf)
        total["departures"] += rec["departures"]
        print(f"{day}: {len(sdf)} segments, {rec['departures']} departures, "
              f"{rec['seconds']} s", flush=True)
    with open(f"{out}/rederive_session.json", "w") as fh:
        json.dump(total, fh, indent=2)
    print(json.dumps(total), flush=True)
    print("done", flush=True)


# <<< SEGMENT RULE >>>

if __name__ == "__main__":
    main()
