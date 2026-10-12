"""§3b, second reading: are the arrivals that crowd a placeholder row's stand
OTHER aircraft, or the departing aircraft's own inbound?

`mechanism.py` counted arrivals and could not tell one from the other.  This
splits them by the three-letter prefix of `FLIGHT_mvt` -- an airline proxy --
against the base rate of that prefix at that stand, which is the denominator the
first reading was missing.  Two windows, both anchored on the departure's own
block time, so this measurement is a DIAGNOSIS on 2025 rows and not a feature:
the ranking file blanks the block time.

Writes `analysis/rotation_reading.csv` (band x class) and
`analysis/rotation_legal.csv` (the schedule-anchored form, which IS legal on the
388, for both years).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
from prc.tail_features import tail_signals  # noqa: E402

W = 1800.0


def _one(masked: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    sig = tail_signals(masked)
    keep = sig.index[sig["noac"].to_numpy() == 1]
    dep = raw.loc[keep]
    arr = masked[masked["PHASE_mvt"].to_numpy() == "ARR"]
    arr = arr.dropna(subset=["ADES_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt"])
    arr = arr.assign(pref=arr["FLIGHT_mvt"].astype(str).str[:3])
    grp = {k: g for k, g in arr.groupby(["ADES_mvt", "STAND_mvt"], sort=False)}
    rows = []
    for mid, apt, st, blk, T, fl in zip(dep.MVT_ID_mvt, dep.ADEP_mvt, dep.STAND_mvt,
                                        dep.BLOCK_TIME_UTC_mvt, dep.MVT_TIME_UTC_mvt,
                                        dep.FLIGHT_mvt):
        g = grp.get((apt, st))
        r = {"MVT_ID_mvt": float(mid)}
        if g is None or pd.isna(blk):
            rows.append(r)
            continue
        p = str(fl)[:3]
        t = g.BLOCK_TIME_UTC_mvt
        win = g[(t > blk - pd.Timedelta(seconds=W)) & (t <= blk + pd.Timedelta(seconds=W))]
        taxi = g[(t > blk) & (t <= T)]
        r.update(base_same=float((g.pref == p).mean()), n_win=len(win),
                 win_same=int((win.pref == p).sum()), n_taxi=len(taxi),
                 taxi_same=int((taxi.pref == p).sum()),
                 taxi_other=int((taxi.pref != p).sum()))
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(f: pd.DataFrame) -> pd.Series:
    """Observed shares are EVENT weighted, so the base rate must be too.

    Round 2 of the review found the first version comparing an event-weighted
    observed share (arrivals with the prefix / arrivals) against a row-weighted
    mean of the per-stand base rates.  The comparable denominator weights each
    row's stand base rate by the number of arrivals that row contributes.
    """
    w = f.n_win.to_numpy(float)
    wt = f.n_taxi.to_numpy(float)
    return pd.Series({
        "n": len(f),
        "n_win_arrivals": float(f.n_win.sum()),
        "n_taxi_arrivals": float(f.n_taxi.sum()),
        "same_pref_share_win": f.win_same.sum() / max(f.n_win.sum(), 1),
        "stand_base_same_rate": float((f.base_same.to_numpy(float) * w).sum() / max(w.sum(), 1)),
        "stand_base_same_rate_taxi": float((f.base_same.to_numpy(float) * wt).sum() / max(wt.sum(), 1)),
        "stand_base_same_rate_rowmean": f.base_same.mean(),
        "P_any_other_in_taxi": (f.taxi_other > 0).mean(),
        "P_2plus_in_taxi": (f.n_taxi >= 2).mean(),
        "same_pref_share_taxi": f.taxi_same.sum() / max(f.n_taxi.sum(), 1),
    })


def main() -> None:
    parts = []
    for m in D.MONTHS:
        raw = D.load_month(m)
        masked, _ = D.mask_like_ranking(raw)
        parts.append(_one(masked, raw))
        print(m, flush=True)
    d = pd.concat(parts, ignore_index=True)
    a = pd.read_parquet(EXP / "work" / "noac_2025.parquet",
                        columns=["MVT_ID_mvt", "ph", "band", "rot_same_near"])
    d = d.merge(a, on="MVT_ID_mvt", validate="one_to_one").dropna(subset=["n_win"])
    d["cls"] = np.where(d.ph == 1, "placeholder", "rest")
    out = []
    for band in ["ALL"] + sorted(d.band.unique()):
        f0 = d if band == "ALL" else d[d.band == band]
        for cls in ("placeholder", "rest"):
            f = f0[f0.cls == cls]
            if len(f) < 5:
                continue
            out.append(dict(band=band, cls=cls, **summarise(f).to_dict()))
    t = pd.DataFrame(out)
    t.to_csv(EXP / "analysis" / "rotation_reading.csv", index=False)
    with pd.option_context("display.width", 220):
        print(t.round(3).to_string(index=False))

    # the legal, schedule-anchored form, on both years
    b = pd.read_parquet(EXP / "work" / "noac_2026.parquet", columns=["rot_same_near"])
    n = a.rot_same_near.fillna(0.0).to_numpy()
    leg = [{"year": "2025", "scope": "ALL", "n": len(a),
            "share_with_same_airline_near_sched": float(n.mean()),
            "P_ph_near": float(a.ph[n == 1].mean()), "P_ph_not": float(a.ph[n == 0].mean())}]
    for band, g in a.groupby("band", sort=False):
        k = g.rot_same_near.fillna(0.0).to_numpy()
        if min((k == 1).sum(), (k == 0).sum()) < 5:
            continue
        leg.append({"year": "2025", "scope": band, "n": len(g),
                    "share_with_same_airline_near_sched": float(k.mean()),
                    "P_ph_near": float(g.ph[k == 1].mean()),
                    "P_ph_not": float(g.ph[k == 0].mean())})
    leg.append({"year": "2026", "scope": "the 388", "n": len(b),
                "share_with_same_airline_near_sched": float(b.rot_same_near.fillna(0.0).mean()),
                "P_ph_near": np.nan, "P_ph_not": np.nan})
    lt = pd.DataFrame(leg)
    lt.to_csv(EXP / "analysis" / "rotation_legal.csv", index=False)
    print()
    print(lt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
