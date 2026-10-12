"""Q2b: per-day judgement, and the board-error direction read off the 2025 analogue.

Day level, because the exposure is a day-level object. For each 2025 fold the
script measures the blend's own bias per (airport, day) against how far that day
sits past its airport's TRAINING maximum daily mean of take-off minus AOBT --
the same quantity that flags the 18 January 2026 days. The relation between the
two is what lets a 2026 day with no label be priced.

Judgement per exposed day:
  tracking      |r_hat - r_hat(that airport's ordinary days)| <= 60 s
  saturating    r_hat grows with severity (the model passes the extra through
                instead of pricing it)
  extrapolating a day-level model INPUT is past its training range
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
sys.path.insert(0, str(F.ROOT / "experiments" / "exp-012-blend2"))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402
import blend as BL  # noqa: E402

OUT = F.EXP / "analysis"
MEMBERS = F.ROOT / "experiments" / "exp-012-blend2" / "members-cand1.json"
NR = 344841


def fold_days() -> pd.DataFrame:
    """Per (airport, day) blend bias and RMSE on each 2025 fold, with the day's
    distance past its airport's training maximum."""
    T25 = pd.read_csv(OUT / "q2_days_2025.csv", parse_dates=["day"])
    files = json.loads(MEMBERS.read_text())["members"]
    data, names = BL.load(files)
    _, W = BL.rotate(data, names, "nnls")
    rows = []
    for fold in ("january", "december", "july_loo"):
        d = data[fold].copy()
        w = np.array([W[fold][n] for n in names], dtype="float64")
        d["pred"] = np.maximum(d[names].to_numpy("float64") @ w, B.FLOOR)
        month = D.FOLDS[fold].valid_month
        raw = D.load_month(month, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
                                           "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
        raw = raw[raw["PHASE_mvt"] == "DEP"]
        d = d.merge(pd.DataFrame({
            "MVT_ID_mvt": raw["MVT_ID_mvt"].to_numpy("float64"),
            "apt": raw["ADEP_mvt"].astype(str).to_numpy(),
            "day": raw["MVT_TIME_UTC_mvt"].dt.floor("D").to_numpy(),
            "obs": (raw["MVT_TIME_UTC_mvt"] - raw["AOBT_3_flt"]).dt.total_seconds().to_numpy()}),
            on="MVT_ID_mvt", how="left")
        d["e"] = d["pred"] - d["y_true"]
        g = d.groupby(["apt", "day"], observed=True).agg(
            n=("e", "size"), bias=("e", "mean"), rmse=("e", lambda s: float(np.sqrt((s ** 2).mean()))),
            y_mean=("y_true", "mean"), pred_mean=("pred", "mean"), obs_mean=("obs", "mean"),
        ).reset_index()
        g["r_true"] = g["obs_mean"] - g["y_mean"]
        g["r_hat"] = g["obs_mean"] - g["pred_mean"]
        env = (T25[T25["day"].dt.strftime("%Y-%m").isin(D.FOLDS[fold].train_months)]
               .groupby("airport")["obs_mean"].agg(tr_max="max", tr_p95=lambda s: s.quantile(0.95),
                                                   tr_med="median"))
        g = g.merge(env, left_on="apt", right_index=True, how="left")
        g["over_tr_max"] = g["obs_mean"] - g["tr_max"]
        g["fold"] = fold
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    FD = fold_days()
    FD.to_csv(OUT / "q2b_fold_days.csv", index=False)

    # ---- how the blend behaves as a day leaves the training envelope ------
    bins = [(-1e9, -300), (-300, -100), (-100, 0), (0, 30), (30, 200), (200, 1e9)]
    rows = []
    for lo, hi in bins:
        m = (FD["over_tr_max"] > lo) & (FD["over_tr_max"] <= hi)
        if m.sum() == 0:
            continue
        g = FD[m]
        wn = g["n"].to_numpy("float64")
        rows.append(dict(band=f"({lo:.0f}, {hi:.0f}]", n_days=int(m.sum()), n_rows=int(g["n"].sum()),
                         bias=float(np.average(g["bias"], weights=wn)),
                         rmse=float(np.sqrt(np.average(g["rmse"] ** 2, weights=wn))),
                         r_true=float(np.average(g["r_true"], weights=wn)),
                         r_hat=float(np.average(g["r_hat"], weights=wn)),
                         obs_mean=float(np.average(g["obs_mean"], weights=wn))))
    Bnd = pd.DataFrame(rows)
    Bnd.to_csv(OUT / "q2b_bias_vs_exceedance.csv", index=False)
    print("=== the blend's bias as a 2025 day leaves its airport's training envelope ===")
    print("(all three folds pooled; `over_tr_max` is the day's mean take-off minus AOBT "
          "less that airport's training maximum daily mean)")
    print(Bnd.round(1).to_string(index=False), flush=True)

    ex = FD[FD["over_tr_max"] > 0].sort_values("over_tr_max", ascending=False)
    print(f"\n=== every 2025 day past its airport's training maximum ({len(ex)} days) ===")
    print(ex[["fold", "apt", "day", "n", "over_tr_max", "obs_mean", "y_mean", "pred_mean",
              "bias", "rmse", "r_true", "r_hat"]].round(1).to_string(index=False), flush=True)

    # ---- the 2026 exposed days, priced from the analogue -------------------
    T26 = pd.read_csv(OUT / "q2_days_2026.csv", parse_dates=["day"])
    jan = T26[T26["month"] == "2026-01"].copy()
    exposed = jan[jan["over_max"] > 30].copy()
    # predictions per 2026 airport-day
    rk = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                 "AOBT_3_flt"])
    dep = rk[rk["PHASE_mvt"] == "DEP"].copy()
    cand = F.candidate()
    dep["pred"] = cand.reindex(dep["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
    dep["obs"] = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    dep["apt"] = dep["ADEP_mvt"].astype(str)
    dep["day"] = dep["MVT_TIME_UTC_mvt"].dt.floor("D")
    dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    g26 = dep.groupby(["apt", "day"], observed=True).agg(
        n=("pred", "size"), pred_mean=("pred", "mean"), obs_mean=("obs", "mean"),
        pred_p90=("pred", lambda s: s.quantile(0.9)),
        obs_p90=("obs", lambda s: s.quantile(0.9))).reset_index()
    g26["r_hat"] = g26["obs_mean"] - g26["pred_mean"]
    E = exposed.merge(g26, left_on=["airport", "day"], right_on=["apt", "day"],
                      how="left", suffixes=("_all", ""))
    # the airport's own ordinary-day r_hat in the same ranking month, as the reference
    ord_ref = (g26.merge(jan[["airport", "day", "over_max"]], left_on=["apt", "day"],
                         right_on=["airport", "day"], how="inner"))
    ord_ref = ord_ref[ord_ref["over_max"] <= 0]
    ref = ord_ref.groupby("apt").apply(
        lambda s: float(np.average(s["r_hat"], weights=s["n"])), include_groups=False).rename("r_hat_ord")
    E = E.merge(ref, left_on="airport", right_index=True, how="left")
    E["d_r_hat"] = E["r_hat"] - E["r_hat_ord"]

    # the day-level input exposure (the tail member's day columns)
    tl = F.tail_ranking()
    tr_tail = F.tail_training()
    caps = {c: float(np.nanmax(tr_tail[c].to_numpy("float64")))
            for c in ("apt_day_p90_mvt_aobt", "apt_day_med_mvt_aobt", "nonm_day_count",
                      "apt_hour_med_mvt_aobt")}
    tl2 = tl.merge(dep[["MVT_ID_mvt", "day"]], on="MVT_ID_mvt", how="left")
    inp = tl2.groupby(["apt", "day"], observed=True).agg(
        p90_in=("apt_day_p90_mvt_aobt", "max"), nonm_in=("nonm_day_count", "max")).reset_index()
    E = E.merge(inp, left_on=["airport", "day"], right_on=["apt", "day"], how="left",
                suffixes=("", "_i"))
    E["p90_over_train"] = E["p90_in"] - caps["apt_day_p90_mvt_aobt"]

    def judge(r):
        tags = []
        if r["p90_over_train"] > 0:
            tags.append("extrapolating")
        if abs(r["d_r_hat"]) <= 60:
            tags.append("tracking")
        elif r["d_r_hat"] > 60:
            tags.append("saturating")
        else:
            tags.append("over-shooting")
        return "+".join(tags)

    E["judgement"] = E.apply(judge, axis=1)
    cols = ["airport", "day", "n", "obs_mean", "max25", "over_max", "pred_mean", "r_hat",
            "r_hat_ord", "d_r_hat", "p90_in", "p90_over_train", "judgement"]
    E[cols].to_csv(OUT / "q2b_exposed_days_2026.csv", index=False)
    print(f"\n=== the {len(E)} exposed January 2026 airport-days ===")
    print(E[cols].round(1).to_string(index=False), flush=True)

    # EHAM's whole month, day by day
    eh = g26[(g26["apt"] == "EHAM") & (g26["day"].dt.strftime("%Y-%m") == "2026-01")].copy()
    eh = eh.merge(jan[["airport", "day", "over_max", "max25"]], left_on=["apt", "day"],
                  right_on=["airport", "day"], how="left")
    eh.to_csv(OUT / "q2b_eham_january.csv", index=False)
    print("\n=== EHAM, January 2026, day by day ===")
    print(eh[["day", "n", "obs_mean", "max25", "over_max", "pred_mean", "obs_p90",
              "pred_p90", "r_hat"]].round(1).to_string(index=False), flush=True)

    # ---- board error direction, from the analogue --------------------------
    print("\n=== board error direction on the exposed days, priced from the analogue ===")
    for lo, hi, label in ((0, 1e9, "any day past the training maximum"),
                          (30, 1e9, "past it by more than 30 s"),
                          (200, 1e9, "past it by more than 200 s")):
        m = (FD["over_tr_max"] > lo) & (FD["over_tr_max"] <= hi)
        if m.sum() == 0:
            continue
        g = FD[m]
        wn = g["n"].to_numpy("float64")
        b = float(np.average(g["bias"], weights=wn))
        n26 = int(E[E["over_max"] > lo]["n"].sum())
        add = np.sqrt(318.81 ** 2 + n26 / NR * b ** 2) - 318.81
        print(f"  2025 days {label}: {int(m.sum())} days, {int(g['n'].sum())} rows, "
              f"bias {b:+.0f} s -> applied to {n26} January 2026 rows, "
              f"board estimate moves {add:+.2f} s if the analogue holds", flush=True)


if __name__ == "__main__":
    main()
