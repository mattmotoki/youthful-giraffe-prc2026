"""The one thing the ranking-side rebuild needs from the raw data, cached.

`prc.blend.build_candidate` reads the ranking file for exactly two per-row
quantities and nothing else:

  * the **regime** (`prc.blend.regimes_from_raw`), from `AIRCRAFT_TYPE_mvt`,
    `FLIGHT_ID_mvt`, `AOBT_3_flt` and `LOBT_flt`;
  * **`d_takeoff_aobt`** = `MVT_TIME_UTC_mvt - AOBT_3_flt`, the residual
    guard's argument.

Neither is in any committed file, and neither is derivable from the published
members: `N1-nmtail-v4-ranking.parquet` carries `has_nm` and `exp-013`'s
`nonm-v1-ranking.parquet` carries a `seg` flag, but the `nm_late` split
(16,090 rows by `exp-014`'s regime census) and `d_takeoff_aobt` are absent.
That is why the ranking-side vectors wait for `data/raw/ranking.parquet`; the
submission template is a second, separate dependency, needed only by
`prc.data.write_submission` in `finalise.py`.

This script turns the ranking file into a 344,841-row side table in seconds, so
that every staged candidate afterwards is a pure function of committed bytes
plus this one cache.  Run it once when the data lands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

OUT = F.EXP / "candidates" / "ranking-gates.parquet"


def build(out: Path = OUT) -> pd.DataFrame:
    r = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt",
                                "AOBT_3_flt", "LOBT_flt", "MVT_TIME_UTC_mvt", "ADEP_mvt"])
    r = r[r["PHASE_mvt"] == "DEP"].copy()
    g = pd.DataFrame({
        "MVT_ID_mvt": r["MVT_ID_mvt"].astype("float64").to_numpy(),
        "regime": B.regimes_from_raw(r, B.LATE_CUT).astype(str),
        "d_takeoff_aobt": (r["MVT_TIME_UTC_mvt"] - r["AOBT_3_flt"]).dt.total_seconds()
                          .to_numpy("float64"),
        "month": r["month"].to_numpy(),
        # `ADEP_mvt` is the departure airport; `exp-007`'s day-state table calls
        # the same column `apt` and `prc.phi_features` reads it the same way.
        # Review round 1 finding: an earlier version probed for a column named
        # `airport`, which the ranking file does not have, so the field was
        # always blank and every per-airport split downstream was empty.
        "airport": r["ADEP_mvt"].astype(str).to_numpy(),
        "day": r["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").to_numpy(),
    })
    assert (g["airport"].str.len() == 4).all(), "ADEP_mvt is not a 4-letter ICAO code"
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(out, index=False)
    print(f"{out}: {len(g)} rows, sha256 {D.sha256(out)}")
    print(g["regime"].value_counts().to_string())
    return g


def load(path: Path = OUT) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(F.ROOT)} is absent.  Build it with\n"
            f"    .venv/bin/python experiments/exp-026-rebuild/code/regime_table.py\n"
            f"which needs data/raw/ranking.parquet (the live 43,564,605-byte file).\n"
            f"Nothing else in the ranking-side rebuild touches the raw data.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    try:
        build()
    except FileNotFoundError as e:
        raise SystemExit(
            f"{e}\n\n"
            "This script needs data/raw/ranking.parquet (the live 43,564,605-byte\n"
            "file, sha256 9db75e3a…) plus the January 2025 training file, which is\n"
            "what prc.data.data_dir() probes for. The submission template\n"
            "data/raw/submitting.parquet is a SEPARATE dependency and is needed\n"
            "only later, by finalise.py.") from None
