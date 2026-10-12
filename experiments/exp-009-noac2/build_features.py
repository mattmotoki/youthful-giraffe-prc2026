"""Candidate observables for p(placeholder), all label-free at prediction time.

The assignment's list, minus the arrival-side evidence (refuted, finding 1):

- per-airport-day noac departure count and share (exp-002 finding 2 showed
  P(placeholder) runs 0.69 -> 0.24 as the day's noac count rises)
- the stand's next arrival *relative to SCHED*: if the stand was re-occupied
  shortly after the scheduled off-block then the aircraft did leave near SCHED,
  so `MVT − SCHED` is the right answer whether or not the record is a
  placeholder -- this is the one observable that can be right without resolving
  the ambiguity
- `FLIGHT_mvt` shape (length, digit/letter pattern, repeated-prefix form)
- hour, day of week, runway, stand group
- operator prefix (the first three characters of `FLIGHT_mvt`)
"""
from __future__ import annotations
import numpy as np, pandas as pd
import prc.data as D
from prc.tail_features import tail_signals, RANKING_AIRPORTS

S = lambda a, b: (a - b).dt.total_seconds()


def noac_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-departure features for the noac rows of one month frame."""
    sig = tail_signals(frame)
    dep = frame.loc[sig.index]
    out = sig.copy()
    out["MVT_ID_mvt"] = dep["MVT_ID_mvt"]
    day = dep["MVT_TIME_UTC_mvt"].dt.floor("D")
    apt = dep["ADEP_mvt"].astype(str)
    g = pd.DataFrame({"apt": apt.to_numpy(), "day": day.to_numpy(),
                      "noac": out["noac"].to_numpy()}, index=sig.index)
    grp = g.groupby(["apt", "day"], observed=True)["noac"]
    out["noac_day_n"] = grp.transform("sum").astype(float)
    out["noac_day_share"] = grp.transform("mean").astype(float)
    out["dep_day_n"] = grp.transform("size").astype(float)

    # the stand's next arrival relative to SCHED: was the stand re-occupied soon
    # after the scheduled off-block?
    out["arr_after_sched_gap"] = out["gap_arr_after_sched"]
    out["arr_within_1h_of_sched"] = (out["gap_arr_after_sched"] < 3600).astype(float)
    out["arr_within_3h_of_sched"] = (out["gap_arr_after_sched"] < 10800).astype(float)

    f = dep["FLIGHT_mvt"].astype(str)
    out["flight_len"] = f.str.len().astype(float)
    out["flight_n_digits"] = f.str.count(r"\d").astype(float)
    out["flight_n_alpha"] = f.str.count(r"[A-Za-z]").astype(float)
    # LIRF writes some records as ICAO+callsign concatenated (ITYTY680, RYRR90BN)
    out["flight_repeated_prefix"] = f.str.match(r"^([A-Z]{3})\1?").fillna(False).astype(float)
    out["flight_prefix"] = f.str[:3]
    out["op_prefix"] = f.str[:3]
    out["mvt_sched_log"] = np.sign(out["mvt_sched"]) * np.log1p(out["mvt_sched"].abs())
    return out[out["noac"] == 1].copy()


def build(months=None) -> pd.DataFrame:
    months = months or list(D.MONTHS)
    parts = []
    for m in months:
        full = D.load_month(m)
        masked, labels = D.mask_like_ranking(full)
        x = noac_features(masked)
        # leakage guard: the same features must come out of the unmasked frame
        y = noac_features(full)
        num = [c for c in x.columns if x[c].dtype.kind in "fiu"]
        assert np.allclose(x[num].to_numpy(float), y[num].to_numpy(float), equal_nan=True), m
        x["y"] = labels.reindex(x.index).astype(float)
        x["month"] = m
        parts.append(x.reset_index(drop=True))
        print(f"{m}: {len(x)} noac rows", flush=True)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    out = D.ROOT / "artifacts" / "exp-009-noac2"
    out.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_parquet(out / "noac_train.parquet")
    print(f"\n{len(df)} training noac rows -> {out/'noac_train.parquet'}")
    rk = D.load_ranking()
    parts = []
    for mo, g in rk.groupby("month"):
        x = noac_features(g); x["month"] = mo; parts.append(x.reset_index(drop=True))
    r = pd.concat(parts, ignore_index=True)
    r.to_parquet(out / "noac_ranking.parquet")
    print(f"{len(r)} ranking noac rows -> {out/'noac_ranking.parquet'}")
