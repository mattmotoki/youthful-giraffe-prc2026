"""Q16 identity and bar measurements. Reads only data/raw. One thread, no external fetch.

Run: .venv/bin/python experiments/exp-042-surface/code/identity_check.py
Writes: experiments/exp-042-surface/analysis/identity.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prc import data as prc_data

ROOT = Path(__file__).resolve().parents[1]
RAW = prc_data.data_dir()
OUT = ROOT / "analysis" / "identity.json"


def main() -> None:
    rank = pd.read_parquet(RAW / "ranking.parquet")
    dep = rank[rank.PHASE_mvt == "DEP"].copy()
    dep["date"] = dep.MVT_TIME_UTC_mvt.dt.date

    known = dep.CALLSIGN_flt.notna()
    grp = dep[known].groupby(["date", "ADEP_mvt", "CALLSIGN_flt"]).size()

    res = {
        "ranking_departures": int(len(dep)),
        "callsign_present": float(known.mean()),
        "no_nm_rows": int((~known).sum()),
        "no_nm_frac": float((~known).mean()),
        "no_nm_by_airport": dep.loc[~known, "ADEP_mvt"].value_counts().to_dict(),
        "callsign_icao_shaped": float(dep.CALLSIGN_flt.dropna().str.match(r"^[A-Z]{3}").mean()),
        "ambiguous_frac_date_adep_callsign": float(grp[grp > 1].sum() / grp.sum()),
        "months": dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").value_counts().to_dict(),
    }

    tr = pd.read_parquet(RAW / "training_2025-07-01_2025-08-01.parquet")
    d = tr[tr.PHASE_mvt == "DEP"].copy()
    t = d[d.CALLSIGN_flt.notna()].copy()

    eq = t.FLIGHT_mvt == t.CALLSIGN_flt
    t["fpre"] = t.FLIGHT_mvt.str.extract(r"^([A-Z]+)")[0]
    t["fsuf"] = t.FLIGHT_mvt.str.replace(r"^[A-Z]+", "", regex=True)
    t["cpre"] = t.CALLSIGN_flt.str.extract(r"^([A-Z]+)")[0]
    sub = t.dropna(subset=["fpre", "cpre"])
    prefix_map = sub.groupby("fpre")["cpre"].apply(lambda s: s.mode().iat[0])
    recon = t.fpre.map(prefix_map).fillna("") + t.fsuf.fillna("")

    res["callsign_reconstruction_july2025"] = {
        "rows": int(len(t)),
        "flight_equals_callsign": float(eq.mean()),
        "prefix_map_correct": float((t.fpre.map(prefix_map) == t.cpre).mean()),
        "reconstructed_exact": float((recon == t.CALLSIGN_flt).mean()),
        "reconstructed_exact_on_unequal_subset": float(
            (recon[~eq] == t.CALLSIGN_flt[~eq]).mean()
        ),
    }

    d["off"] = (d.BLOCK_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
    bar = {}
    for ap in ["ALL", "EHAM"]:
        s = d if ap == "ALL" else d[d.ADEP_mvt == ap]
        o = s["off"].dropna()
        bar[ap] = {
            "rows": int(len(s)),
            "with_nm": int(len(o)),
            "median_s": float(o.median()),
            "rmse_about_airport_median_s": float(np.sqrt(((o - o.median()) ** 2).mean())),
            "p01_s": float(o.quantile(0.01)),
            "p99_s": float(o.quantile(0.99)),
            "within_60s": float((o.abs() <= 60).mean()),
        }
    res["block_minus_aobt_july2025"] = bar

    wk = d[
        (d.ADEP_mvt == "EHAM")
        & (d.MVT_TIME_UTC_mvt >= "2025-07-07")
        & (d.MVT_TIME_UTC_mvt < "2025-07-14")
    ]
    res["pilot_week_eham_2025_07_07_13"] = {
        "departures": int(len(wk)),
        "no_nm": int(wk.CALLSIGN_flt.isna().sum()),
        "target_median_s": float(wk.TAXITIME_SEC_mvt.median()),
        "target_std_s": float(wk.TAXITIME_SEC_mvt.std()),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, sort_keys=True, default=str))
    print(json.dumps(res, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
