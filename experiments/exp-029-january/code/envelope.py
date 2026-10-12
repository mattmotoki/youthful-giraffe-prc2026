"""The label-free envelope flag, rebuilt from committed files alone.

The quantity is `span = MVT_TIME_UTC_mvt - AOBT_3_flt` in seconds, on DEP rows
that carry an NM join (`AOBT_3_flt` non-null).  Both columns survive into the
ranking file, so every number below is computable without a label.

    day_span_mean[apt, day]  = mean of span over that airport-day
    envelope[apt]            = max over the TRAINING months of day_span_mean
    exceedance[apt, day]     = day_span_mean - envelope[apt]
    flagged                  = exceedance > 0

That is exp-014 `code/q2_days.py`'s `obs_mean` / `max25` / `over_max`, renamed.
The fold side uses each fold's own training months for the envelope; the ranking
side uses all twelve 2025 months, because the submission fit sees all twelve.

Addresses (all in git):
  experiments/exp-014-robustness/analysis/q2_days_2025.csv   3,652 airport-days
  experiments/exp-014-robustness/analysis/q2_days_2026.csv     620 airport-days
  experiments/exp-007-winter/work/rows_{january,july,december}.parquet
  experiments/exp-007-winter/work/ranking_daystate.parquet
  experiments/exp-026-rebuild/oof/stack5-{fold}.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments" / "exp-026-rebuild" / "code"))
import frames as F26  # noqa: E402

DAYS25 = ROOT / "experiments/exp-014-robustness/analysis/q2_days_2025.csv"
DAYS26 = ROOT / "experiments/exp-014-robustness/analysis/q2_days_2026.csv"
ROWS = "experiments/exp-007-winter/work/rows_{m}.parquet"
STACK5 = "experiments/exp-026-rebuild/oof/stack5-{fold}.parquet"
#: the validation month of each fold lives in a file named after the PRIMARY
#: fold, not after `july_loo`; july and july_loo validate the same month.
ROWS_FILE = {"january": "january", "july_loo": "july", "december": "december"}
FOLDS = ("july_loo", "december", "january")
CLEAR = 30.0


def days_2025() -> pd.DataFrame:
    t = pd.read_csv(DAYS25, parse_dates=["day"])
    return t.rename(columns={"obs_mean": "day_span_mean", "airport": "apt"})


def envelope(train_months) -> pd.Series:
    """max daily span mean per airport over `train_months`."""
    t = days_2025()
    t = t[t["month"].isin(list(train_months))]
    return t.groupby("apt")["day_span_mean"].max().rename("envelope")


def fold_days(fold: str) -> pd.DataFrame:
    """Per (apt, day) of the fold's validation month: span mean, envelope, exceedance.

    The day means are recomputed from the row file rather than read from the
    2025 table, so the two independent producers have to agree; `main` checks
    that they do.
    """
    r = pd.read_parquet(ROOT / ROWS.format(m=ROWS_FILE[fold]),
                        columns=["MVT_ID_mvt", "apt", "day", "span"])
    d = (r.groupby(["apt", "day"], observed=True)["span"]
         .agg(n="size", n_span="count", day_span_mean="mean").reset_index())
    env = envelope(D.FOLDS[fold].train_months)
    d = d.merge(env, left_on="apt", right_index=True, how="left")
    d["exceedance"] = d["day_span_mean"] - d["envelope"]
    d["fold"] = fold
    return d


def fold_rows(fold: str) -> pd.DataFrame:
    """stack-5's out-of-fold prediction with the flag and exceedance attached."""
    p = pd.read_parquet(ROOT / STACK5.format(fold=fold))
    r = pd.read_parquet(ROOT / ROWS.format(m=ROWS_FILE[fold]),
                        columns=["MVT_ID_mvt", "apt", "day", "span", "regime", "y"])
    m = p.merge(r, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if m["apt"].isna().any():
        raise ValueError(f"{fold}: {int(m['apt'].isna().sum())} rows without an airport-day")
    if not np.allclose(m["y_true"].to_numpy("float64"), m["y"].to_numpy("float64")):
        raise ValueError(f"{fold}: the row file's label disagrees with the OOF label")
    d = fold_days(fold)[["apt", "day", "day_span_mean", "envelope", "exceedance"]]
    m = m.merge(d, on=["apt", "day"], how="left", validate="many_to_one")
    # `prc.blend`'s four-regime label, the taxonomy the served stack switches on
    g = F26.gates(fold, check=False)[["MVT_ID_mvt", "regime"]].rename(
        columns={"regime": "breg"})
    m = m.merge(g, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if m["breg"].isna().any():
        raise ValueError(f"{fold}: {int(m['breg'].isna().sum())} rows without a blend regime")
    m["flagged"] = m["exceedance"] > 0
    m["fold"] = fold
    return m


def ranking_days() -> pd.DataFrame:
    """January and July 2026 airport-days against the twelve-month envelope."""
    t = pd.read_csv(DAYS26, parse_dates=["day"])
    t = t.rename(columns={"obs_mean": "day_span_mean", "airport": "apt",
                          "max25": "envelope", "over_max": "exceedance"})
    return t[["apt", "day", "month", "n", "n_obs", "day_span_mean", "envelope", "exceedance"]]


def verify_against_raw() -> None:
    """Recompute the twelve-month day table from `data/raw` and compare.

    Every fold-side number here rests on exp-014's committed `q2_days_2025.csv`,
    which was produced on the workstation from data this laptop did not have
    until 2026-09-07 16:0x.  Now that it does, the table is checked against the
    monthly files themselves rather than trusted.
    """
    try:
        D.data_dir()
    except Exception as exc:                                     # noqa: BLE001
        print(f"[verify] skipped, no raw data ({type(exc).__name__})", flush=True)
        return
    parts = []
    for m in D.MONTHS:
        f = D.load_month(m, columns=["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                     "AOBT_3_flt"])
        f = f[f["PHASE_mvt"] == "DEP"]
        parts.append(pd.DataFrame({
            "apt": f["ADEP_mvt"].astype(str).to_numpy(),
            "day": f["MVT_TIME_UTC_mvt"].dt.floor("D").to_numpy(),
            "span": (f["MVT_TIME_UTC_mvt"] - f["AOBT_3_flt"]).dt.total_seconds().to_numpy()}))
    t = pd.concat(parts, ignore_index=True)
    got = (t.groupby(["apt", "day"], observed=True)["span"]
           .agg(n="size", day_span_mean="mean").reset_index())
    # exp-014 built its table one month at a time, so two airport-days that
    # straddle a monthly file boundary appear twice there (LEMD 2025-08-31 and
    # LTFM 2025-02-28, each with a one-row all-null partner).  Aggregate the
    # reference by key before comparing, or the duplicate reads as a 647-row
    # discrepancy that is not one.
    ref = (days_2025().groupby(["apt", "day"], observed=True)
           .agg(n=("n", "sum"),
                day_span_mean=("day_span_mean", lambda s: s.dropna().iloc[0]
                               if s.notna().any() else np.nan)).reset_index())
    j = got.merge(ref[["apt", "day", "n", "day_span_mean"]], on=["apt", "day"],
                  how="outer", suffixes=("", "_ref"), indicator=True,
                  validate="one_to_one")
    bad = int((j["_merge"] != "both").sum())
    gap = float(np.abs(j["day_span_mean"] - j["day_span_mean_ref"]).max())
    nn = int((j["n"] - j["n_ref"]).abs().max())
    print(f"[verify] twelve-month day table against data/raw: {len(j)} airport-days, "
          f"{bad} unmatched, max |delta day mean| {gap:.3e} s, max |delta rows| {nn}",
          flush=True)
    if bad or nn or gap > 1e-6:
        raise SystemExit("exp-014's committed day table disagrees with the raw data")


def main() -> None:
    out = EXP / "analysis"
    verify_against_raw()
    # --- gate 1: the two producers of the day means agree -------------------
    t25 = days_2025()
    for fold in FOLDS:
        d = fold_days(fold)
        ref = t25[t25["month"] == D.FOLDS[fold].valid_month]
        j = d.merge(ref[["apt", "day", "day_span_mean", "n", "n_obs"]], on=["apt", "day"],
                    suffixes=("", "_ref"), validate="one_to_one")
        if len(j) != len(d):
            raise ValueError(f"{fold}: {len(d)} days against {len(j)} matched")
        gap = np.abs(j["day_span_mean"] - j["day_span_mean_ref"]).max()
        print(f"[gate] {fold:9s} {len(d):3d} airport-days, day-mean max |delta| "
              f"vs exp-014's table {gap:.2e} s, rows {int(j['n'].sum())} vs "
              f"{int(j['n_ref'].sum())}", flush=True)
        if gap > 1e-6:
            raise SystemExit(f"{fold}: day means disagree by {gap} s")

    # --- gate 2: the five analogue days -------------------------------------
    A = pd.concat([fold_days(f) for f in FOLDS], ignore_index=True)
    A.to_csv(out / "fold_days.csv", index=False)
    ex = A[A["exceedance"] > 0].sort_values(["fold", "exceedance"], ascending=[True, False])
    print("\n=== fold airport-days past their airport's training envelope ===")
    print(ex[["fold", "apt", "day", "n", "day_span_mean", "envelope", "exceedance"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"), flush=True)

    # --- the rows, with stack-5 on them --------------------------------------
    R = pd.concat([fold_rows(f) for f in FOLDS], ignore_index=True)
    R.to_parquet(EXP / "analysis" / "fold_rows.parquet", index=False)
    print("\n=== stack-5 on the flagged rows ===")
    rec = []
    for f in FOLDS:
        g = R[R["fold"] == f]
        for lab, m in (("whole fold", np.ones(len(g), bool)),
                       ("flagged", g["flagged"].to_numpy()),
                       (f"flagged, >{CLEAR:.0f} s", (g["exceedance"] > CLEAR).to_numpy())):
            s = g[m]
            if not len(s):
                continue
            e = s["y_pred"].to_numpy("float64") - s["y_true"].to_numpy("float64")
            rec.append(dict(fold=f, slice=lab, days=s.groupby(["apt", "day"]).ngroups,
                            n=len(s), rmse=float(np.sqrt((e ** 2).mean())),
                            bias=float(e.mean())))
    T = pd.DataFrame(rec)
    T.to_csv(out / "stack5_on_flagged.csv", index=False)
    print(T.round(1).to_string(index=False), flush=True)

    # --- the ranking side ----------------------------------------------------
    K = ranking_days()
    K["flagged"] = K["exceedance"] > 0
    K.to_csv(out / "ranking_days.csv", index=False)
    jan = K[K["month"] == "2026-01"]
    print("\n=== January 2026 ===")
    print(f"  airport-days {len(jan)}, flagged {int(jan['flagged'].sum())}, "
          f">{CLEAR:.0f} s {int((jan['exceedance'] > CLEAR).sum())}", flush=True)
    print(f"  rows {int(jan['n'].sum())}, flagged {int(jan.loc[jan['flagged'], 'n'].sum())} "
          f"({jan.loc[jan['flagged'], 'n'].sum() / jan['n'].sum():.4%}), "
          f">{CLEAR:.0f} s {int(jan.loc[jan['exceedance'] > CLEAR, 'n'].sum())}", flush=True)
    print(jan[jan["flagged"]].sort_values("exceedance", ascending=False)
          [["apt", "day", "n", "day_span_mean", "envelope", "exceedance"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"), flush=True)


if __name__ == "__main__":
    main()
