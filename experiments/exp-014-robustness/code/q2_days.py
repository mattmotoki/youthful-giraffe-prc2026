"""Q2: airport-day exposure in January 2026, and the 2025 analogue that prices it.

Three things per exposed airport-day: what the candidate PREDICTS, what the
observable take-off minus AOBT actually IS on the same rows, and what the blend
DID on the most similar 2025 days -- the january fold's worst days, where the
label exists and the blend's own out-of-fold prediction can be scored.

The judgement column distinguishes three behaviours:
  tracking      predicted taxi moves with the day's observable one-for-one
                (the members fit a RESIDUAL to take-off minus AOBT, so this is
                the structural default)
  saturating    the implied residual r_hat = obs - pred grows with severity:
                the model is not pricing the extra, it is passing it through
  extrapolating a day-level INPUT is past its training range, so the tree is
                answering from its top bin
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames14 as F  # noqa: E402
sys.path.insert(0, str(F.ROOT))
sys.path.insert(0, str(F.ROOT / "experiments" / "exp-012-blend2"))
import prc.data as D  # noqa: E402
import prc.blend as B  # noqa: E402
import blend as BL  # noqa: E402

OUT = F.EXP / "analysis"
MEMBERS = F.ROOT / "experiments" / "exp-012-blend2" / "members-cand1.json"
CLEAR = 30.0   # seconds of daily mean above the airport's 2025 maximum


# ------------------------------------------------------------ day means ----
def day_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (airport, day): n, mean/median/p90 of take-off minus AOBT, NM-join rate."""
    d = df[df["PHASE_mvt"] == "DEP"].copy()
    d["obs"] = (d["MVT_TIME_UTC_mvt"] - d["AOBT_3_flt"]).dt.total_seconds()
    d["day"] = d["MVT_TIME_UTC_mvt"].dt.floor("D")
    g = d.groupby([d["ADEP_mvt"].astype(str), "day"], observed=True)
    t = g.agg(n=("obs", "size"), n_obs=("obs", "count"), obs_mean=("obs", "mean"),
              obs_med=("obs", "median"), obs_p90=("obs", lambda s: s.quantile(0.9)))
    t.index.names = ["airport", "day"]
    return t.reset_index()


def main() -> None:
    import json

    # ---- 2025 daily envelope, per airport --------------------------------
    tr25 = []
    for m in D.MONTHS:
        tr25.append(day_table(D.load_month(m, columns=["PHASE_mvt", "ADEP_mvt",
                                                       "MVT_TIME_UTC_mvt", "AOBT_3_flt"])))
    T25 = pd.concat(tr25, ignore_index=True)
    T25["month"] = T25["day"].dt.strftime("%Y-%m")
    env = T25.groupby("airport")["obs_mean"].agg(max25="max", p95=lambda s: s.quantile(0.95),
                                                 mean25="mean").reset_index()
    T25.to_csv(OUT / "q2_days_2025.csv", index=False)

    # ---- January 2026 days ------------------------------------------------
    rk = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                 "AOBT_3_flt", "LOBT_flt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt"])
    T26 = day_table(rk)
    T26["month"] = T26["day"].dt.strftime("%Y-%m")
    T26 = T26.merge(env, on="airport", how="left")
    T26["over_max"] = T26["obs_mean"] - T26["max25"]
    jan = T26[T26["month"] == "2026-01"].copy()
    exposed = jan[jan["over_max"] > CLEAR].sort_values("over_max", ascending=False)
    T26.to_csv(OUT / "q2_days_2026.csv", index=False)
    print(f"January 2026 airport-days clearing the 2025 maximum by >{CLEAR:.0f} s: "
          f"{len(exposed)}", flush=True)
    print(exposed[["airport", "day", "n", "obs_mean", "max25", "over_max"]]
          .to_string(index=False), flush=True)

    # ---- candidate predictions on those rows ------------------------------
    dep = rk[rk["PHASE_mvt"] == "DEP"].copy()
    dep["obs"] = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    dep["day"] = dep["MVT_TIME_UTC_mvt"].dt.floor("D")
    dep["airport"] = dep["ADEP_mvt"].astype(str)
    dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    cand = F.candidate()
    dep["pred"] = cand.reindex(dep["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
    dep["r_hat"] = dep["obs"] - dep["pred"]
    key = set(zip(exposed["airport"], exposed["day"]))
    dep["exposed"] = [(a, d) in key for a, d in zip(dep["airport"], dep["day"])]
    dep["eham_jan"] = (dep["airport"] == "EHAM") & (dep["month"] == "2026-01")

    def dist(g: pd.DataFrame, label: str) -> dict:
        q = lambda s, p: float(np.nanquantile(s, p))
        return dict(slice=label, n=len(g),
                    pred_mean=float(g["pred"].mean()), pred_p50=q(g["pred"], .5),
                    pred_p90=q(g["pred"], .9), pred_p99=q(g["pred"], .99),
                    pred_max=float(g["pred"].max()),
                    obs_mean=float(g["obs"].mean()), obs_p50=q(g["obs"].dropna(), .5),
                    obs_p90=q(g["obs"].dropna(), .9), obs_p99=q(g["obs"].dropna(), .99),
                    r_hat_mean=float(g["r_hat"].mean()), r_hat_p50=q(g["r_hat"].dropna(), .5))

    rows = [dist(dep[dep["month"] == "2026-01"], "january 2026, all"),
            dist(dep[dep["exposed"]], f"january 2026, {len(exposed)} exposed airport-days"),
            dist(dep[dep["eham_jan"]], "EHAM january 2026, whole month"),
            dist(dep[(dep["month"] == "2026-01") & ~dep["exposed"]], "january 2026, other days"),
            dist(dep[dep["month"] == "2026-07"], "july 2026, all")]
    Dst = pd.DataFrame(rows)
    Dst.to_csv(OUT / "q2_distributions_2026.csv", index=False)
    print("\n" + Dst.round(1).to_string(index=False), flush=True)

    # ---- the 2025 analogue: the january fold's worst days ------------------
    files = json.loads(MEMBERS.read_text())["members"]
    data, names = BL.load(files)
    Tr, W = BL.rotate(data, names, "nnls")
    print("\nrotated blend (candidate 1 members):", flush=True)
    print(Tr[["fold", "fit_on", "rmse"]].round(2).to_string(index=False), flush=True)

    ana = []
    for fold in ("january", "july_loo", "december"):
        d = data[fold].copy()
        w = np.array([W[fold][n] for n in names], dtype="float64")
        d["pred"] = np.maximum(d[names].to_numpy("float64") @ w, B.FLOOR)
        month = D.FOLDS[fold].valid_month
        raw = D.load_month(month, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
                                           "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
        raw = raw[raw["PHASE_mvt"] == "DEP"]
        j = pd.DataFrame({"MVT_ID_mvt": raw["MVT_ID_mvt"].to_numpy("float64"),
                          "apt": raw["ADEP_mvt"].astype(str).to_numpy(),
                          "day": raw["MVT_TIME_UTC_mvt"].dt.floor("D").to_numpy(),
                          "obs": (raw["MVT_TIME_UTC_mvt"] - raw["AOBT_3_flt"]).dt.total_seconds().to_numpy()})
        d = d.merge(j, on="MVT_ID_mvt", how="left")
        dm = d.groupby(["apt", "day"], observed=True)["obs"].mean().rename("day_obs_mean")
        d = d.merge(dm, on=["apt", "day"], how="left")
        # this month's own days, ranked inside the airport by severity
        d["day_rank"] = d.groupby("apt", observed=True)["day_obs_mean"].rank(pct=True)
        # the training envelope for that fold, from its TRAINING months only
        tr_env = (T25[T25["month"].isin(D.FOLDS[fold].train_months)]
                  .groupby("airport")["obs_mean"].agg(["max", lambda s: s.quantile(0.95)]))
        tr_env.columns = ["tr_max", "tr_p95"]
        d = d.merge(tr_env, left_on="apt", right_index=True, how="left")
        d["over_tr_max"] = d["day_obs_mean"] - d["tr_max"]
        for label, m in (("whole fold", np.ones(len(d), bool)),
                         ("worst 5 % of days (by that airport's own daily mean)", d["day_rank"] > 0.95),
                         ("days above the fold's TRAINING maximum", d["over_tr_max"] > 0),
                         (f"days above the training maximum by >{CLEAR:.0f} s", d["over_tr_max"] > CLEAR)):
            m = np.asarray(m)
            if m.sum() < 20:
                continue
            g = d[m]
            e = g["pred"] - g["y_true"]
            ana.append(dict(fold=fold, slice=label, n=int(m.sum()),
                            n_days=int(g.groupby(["apt", "day"], observed=True).ngroups),
                            rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
                            y_mean=float(g["y_true"].mean()), pred_mean=float(g["pred"].mean()),
                            obs_mean=float(g["obs"].mean()),
                            r_true=float((g["obs"] - g["y_true"]).mean()),
                            r_hat=float((g["obs"] - g["pred"]).mean())))
    An = pd.DataFrame(ana)
    An.to_csv(OUT / "q2_analogue_2025.csv", index=False)
    print("\n=== the 2025 analogue: the blend's own out-of-fold behaviour ===")
    print(An.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
