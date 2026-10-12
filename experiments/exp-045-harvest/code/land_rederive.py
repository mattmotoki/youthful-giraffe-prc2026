"""exp-045-harvest: land one re-derive's outputs on the summaries tree.

The same landing for both re-derives: the one that runs in a Kaggle session
(`pull_rederive.sh`) and the one that runs on this laptop (`rederive_local.py`).
It was a here-document inside the puller until the volume filled on 2026-09-08
and bash could not spool one.

  python land_rederive.py <dir-of-outputs> <artifacts/exp-045> <exp-dir> <label>
                          [<kaggle-log.json> <plain-log>]

`label` goes into analysis/rederived/<day>.json as the thing that wrote the day,
so manifest.json can say which rule wrote each day's summaries and where it ran.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    tmp, art, exp, label = (Path(sys.argv[1]), Path(sys.argv[2]),
                            Path(sys.argv[3]), sys.argv[4])
    if len(sys.argv) > 6:                      # the session log, JSON records to text
        src, dst = Path(sys.argv[5]), Path(sys.argv[6])
        if src.exists():
            raw = json.loads(src.read_text())
            dst.write_text("".join(r.get("data", "") for r in raw))

    n = days = 0
    for p in sorted(tmp.glob("segments_*.parquet")):
        day = p.name[9:19]
        df = pd.read_parquet(p)
        for airport, g in df.groupby("airport", observed=True):
            d = art / "summaries" / str(airport)
            d.mkdir(parents=True, exist_ok=True)
            g.drop(columns=["airport"]).to_parquet(d / f"{day}.parquet", index=False,
                                                   compression="zstd")
            n += 1
        days += 1
    for p in sorted(tmp.glob("callsigns_*.parquet")):
        (art / "callsigns").mkdir(parents=True, exist_ok=True)
        pd.read_parquet(p).to_parquet(art / "callsigns" / p.name, index=False,
                                      compression="zstd")
    for p in sorted(tmp.glob("rederive_2*.json")):
        rec = json.loads(p.read_text())
        rec["kernel"] = label
        out = exp / "analysis/rederived" / f"{rec['day']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2))
    print(f"re-derived {days} days into {n} airport-day summaries under the current rule")


if __name__ == "__main__":
    main()
