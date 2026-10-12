"""exp-185: the exp-045 receipt (census.py) for months of the final file.

    .venv/bin/python experiments/exp-185-final-data/code/census_final.py 2026-02 2026-06

Run-only wrapper: exp-045 census.py reads `data/raw/ranking.parquet` for every 2026 month (`ranking()`), which
holds no February or June row. The wrapper gives `ranking()` the departures of the final file (prc.data final
mode, by SHA-256) and calls census.main for the months given; it adds them to exp-045 analysis/receipt.json and
leaves the other months as they are.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ["PRC_RANKING_MODE"] = "final"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp-045-harvest/code"))
import census as C  # noqa: E402
from prc import data as D  # noqa: E402

months = sys.argv[1:]
assert months and all(m in D.rank_months() for m in months), months


def final_ranking():
    r = D.load_ranking(columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt",
                                "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
    r = r[r.PHASE_mvt == "DEP"].drop(columns=[c for c in ("PHASE_mvt", "month") if c in r.columns])
    r["month"] = r.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    return r


C.ranking = final_ranking
C.main(months)
