"""Q3: is the candidate file itself sane?

Checks the bytes that would be uploaded: the floor, the ceiling, the two small
regimes that carry most of the squared error, the per-month level against the
2025 months of the same season, and every row where the three members disagree
by more than 1,800 s. Members are compared AS BLENDED -- floored and residual-
guarded by `prc.blend.build_candidate` -- with the raw disagreement reported
beside it, because the preparation is part of the shipped estimator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames14 as F  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

OUT = F.EXP / "analysis"
CFG = F.ROOT / "release" / "config" / "exp-012-blend2" / "blend-config-1.json"
BIG = 20000.0
DISAGREE = 1800.0
QS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)


def qrow(label: str, v: np.ndarray, extra=None) -> dict:
    v = np.asarray(v, dtype="float64")
    r = dict(slice=label, n=len(v), mean=float(v.mean()), std=float(v.std()),
             min=float(v.min()), max=float(v.max()))
    for q in QS:
        r[f"p{q*100:g}"] = float(np.quantile(v, q))
    if extra:
        r.update(extra)
    return r


def main() -> None:
    cfg = json.loads(CFG.read_text())
    cand = F.candidate()
    print(f"candidate sha256 {D.sha256(F.ROOT / cfg['candidate']['path'])}", flush=True)

    rk = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                 "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt",
                                 "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt",
                                 "STAND_mvt", "RUNWAY_mvt"])
    dep = rk[rk["PHASE_mvt"] == "DEP"].copy()
    dep["pred"] = cand.reindex(dep["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
    dep["regime"] = B.regimes_from_raw(dep, float(cfg["late_cut"]))
    dep["obs"] = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    dep["aobt_lobt"] = (dep["AOBT_3_flt"] - dep["LOBT_flt"]).dt.total_seconds()
    dep["mvt_sched"] = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    dep["airport"] = dep["ADEP_mvt"].astype(str)
    dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")

    # ------------------------------------------------- floor and ceiling ----
    checks = dict(
        n=int(len(dep)), n_nan=int((~np.isfinite(dep["pred"])).sum()),
        n_negative=int((dep["pred"] < 0).sum()),
        n_below_floor=int((dep["pred"] < float(cfg["floor"]) - 1e-9).sum()),
        n_at_floor=int(np.isclose(dep["pred"], float(cfg["floor"])).sum()),
        n_above_20000=int((dep["pred"] > BIG).sum()),
        min=float(dep["pred"].min()), max=float(dep["pred"].max()))
    print("\n=== file checks ===\n" + json.dumps(checks, indent=1), flush=True)

    big = dep[dep["pred"] > BIG].sort_values("pred", ascending=False)
    big[["MVT_ID_mvt", "airport", "month", "regime", "pred", "obs", "aobt_lobt", "mvt_sched",
         "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"]] \
        .to_csv(OUT / "q3_predictions_above_20000.csv", index=False)
    if len(big):
        print(f"\n=== {len(big)} predictions above {BIG:.0f} s ===")
        print(big[["airport", "month", "regime", "pred", "obs", "aobt_lobt", "mvt_sched",
                   "FLIGHT_mvt", "STAND_mvt"]].head(30).round(0).to_string(index=False), flush=True)

    # -------------------------------- regimes against their 2025 selves ----
    rows = []
    for m in ("2026-01", "2026-07"):
        s = dep[dep["month"] == m]
        rows.append(qrow(f"{m} all", s["pred"].to_numpy()))
        for rg in B.REGIMES:
            g = s[s["regime"] == rg]
            if len(g):
                rows.append(qrow(f"{m} {rg}", g["pred"].to_numpy()))
    # the 2025 label distributions for the same regimes, per season
    SEASON = {"winter": ("2025-01", "2025-02", "2025-11", "2025-12"),
              "summer": ("2025-06", "2025-07", "2025-08", "2025-09")}
    for season, months in SEASON.items():
        t = D.load_months(list(months),
                          columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "AIRCRAFT_TYPE_mvt",
                                   "FLIGHT_ID_mvt", "AOBT_3_flt", "LOBT_flt"])
        t = t[(t["PHASE_mvt"] == "DEP") & t["TAXITIME_SEC_mvt"].notna()].copy()
        t["regime"] = B.regimes_from_raw(t, float(cfg["late_cut"]))
        rows.append(qrow(f"2025 {season} TRUTH all", t["TAXITIME_SEC_mvt"].to_numpy("float64")))
        for rg in B.REGIMES:
            g = t[t["regime"] == rg]
            if len(g):
                rows.append(qrow(f"2025 {season} TRUTH {rg}", g["TAXITIME_SEC_mvt"].to_numpy("float64")))
    Q = pd.DataFrame(rows)
    Q.to_csv(OUT / "q3_level_vs_2025.csv", index=False)
    print("\n=== level: candidate against the 2025 truth of the same season ===")
    print(Q.round(0).to_string(index=False), flush=True)

    # ------------------------------------------- per airport-month level ----
    am = dep.groupby(["month", "airport"], observed=True)["pred"].agg(
        n="size", mean="mean", p50="median", p90=lambda s: s.quantile(0.9)).reset_index()
    tr = D.load_months(list(D.MONTHS), columns=["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                               "TAXITIME_SEC_mvt"])
    tr = tr[(tr["PHASE_mvt"] == "DEP") & tr["TAXITIME_SEC_mvt"].notna()].copy()
    tr["airport"] = tr["ADEP_mvt"].astype(str)
    tr["season"] = np.where(tr["MVT_TIME_UTC_mvt"].dt.month.isin([1, 2, 11, 12]), "winter",
                            np.where(tr["MVT_TIME_UTC_mvt"].dt.month.isin([6, 7, 8, 9]), "summer", "mid"))
    ts = tr.groupby(["season", "airport"], observed=True)["TAXITIME_SEC_mvt"].agg(
        n25="size", mean25="mean", p50_25="median", p90_25=lambda s: s.quantile(0.9)).reset_index()
    am["season"] = np.where(am["month"] == "2026-01", "winter", "summer")
    am = am.merge(ts, on=["season", "airport"], how="left")
    am["d_mean"] = am["mean"] - am["mean25"]
    am.to_csv(OUT / "q3_airport_month_level.csv", index=False)
    print("\n=== per airport-month mean against the 2025 same-season truth ===")
    print(am.round(0).to_string(index=False), flush=True)

    # ------------------------------------------------ member disagreement ----
    files = cfg["ranking"]
    frame, names = B.stack_members(files, F.ROOT)
    m = frame.merge(dep[["MVT_ID_mvt", "airport", "month", "regime", "pred", "obs",
                         "aobt_lobt", "mvt_sched", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt",
                         "AIRCRAFT_TYPE_mvt"]], on="MVT_ID_mvt", how="inner")
    raw = m[names].to_numpy("float64")
    P = raw.copy()
    a = m["obs"].to_numpy("float64")
    lo, hi = cfg["guard_bounds"]
    for j in range(P.shape[1]):
        P[:, j] = np.maximum(P[:, j], float(cfg["floor"]))
        P[:, j] = B.apply_residual_guard(P[:, j], a, float(lo), float(hi), float(cfg["floor"]))
    m["spread_raw"] = raw.max(axis=1) - raw.min(axis=1)
    m["spread"] = P.max(axis=1) - P.min(axis=1)
    for j, n in enumerate(names):
        m[f"prep_{n}"] = P[:, j]
    dis = m[m["spread"] > DISAGREE].sort_values("spread", ascending=False)
    cols = ["MVT_ID_mvt", "airport", "month", "regime", "pred", "spread", "spread_raw"] + \
           [f"prep_{n}" for n in names] + ["obs", "aobt_lobt", "mvt_sched", "FLIGHT_mvt",
                                           "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"]
    dis[cols].to_csv(OUT / "q3_member_disagreement.csv", index=False)
    print(f"\n=== {len(dis)} rows where the prepared members disagree by more than "
          f"{DISAGREE:.0f} s ({len(m[m.spread_raw > DISAGREE])} before preparation) ===")
    print(dis[["airport", "month", "regime", "pred", "spread"] +
              [f"prep_{n}" for n in names] +
              ["obs", "aobt_lobt", "mvt_sched", "STAND_mvt"]].head(40).round(0).to_string(index=False),
          flush=True)
    summ = dis.groupby(["month", "regime"], observed=True).agg(
        n=("spread", "size"), spread_mean=("spread", "mean"), pred_mean=("pred", "mean")).reset_index()
    print("\n" + summ.round(0).to_string(index=False), flush=True)
    json.dump(checks, open(OUT / "q3_file_checks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
