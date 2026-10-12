"""The measurements Sol's round-1 review asks for.  No new arm; four re-reads.

The review (`review/round1-sol.md`, verdict NEEDS REVISION with FAIL kept)
raised four findings that need a number rather than a wording change:

  B1  the +0.049 s re-mix is partly IN SAMPLE -- the exposed-slice behaviour is
      pooled over all five analogue days while the january-rotation constants
      are fitted on the July and December days.  The out-of-sample pairing is
      january-only behaviour with July+December constants.
  B3  the assignment asks for CALENDAR-DAY clusters; `correct.py` uses
      `airport|calendar day`.  Recompute on the literal unit.
  B5  COMMON.md wants overall / <=3600 / >3600 / no-NM per fold for every
      reported arm, not for three of them.
  M9  the implied-`r` row weighting counts rows whose `r_hat` is missing.

Every number here is a re-reading of vectors already measured.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "exp-026-rebuild" / "code"))
import correct as C  # noqa: E402
import envelope as E  # noqa: E402
import remix as RX  # noqa: E402
import prc.data as D  # noqa: E402
import frames as F26  # noqa: E402

#: the arms the RESULTS tables report, in table order.
REPORTED = ("step|lofo|winsor", "step|lofo|raw", "step-half|lofo|winsor",
            "linear|lofo|winsor", "linear-capped|lofo|winsor",
            "stress|lofo|winsor", "stress-all|lofo|winsor",
            "step-winter|lofo|winsor", "step-bulk|lofo|winsor",
            "step-regime|lofo|winsor")


# --------------------------------------------------------------------- B1 ---
def b1_out_of_sample(R: pd.DataFrame) -> pd.DataFrame:
    """The re-mix under both sources of exposed-slice behaviour.

    The served constants for the january rotation are fitted on the July and
    December days.  Pairing them with behaviour POOLED over all five days scores
    them partly on their own fitting rows; pairing them with the january fold's
    own two days does not.  The second is the honest one and the review is right
    that the first must not be called "out of sample".
    """
    K = pd.read_csv(EXP / "analysis" / "ranking_days.csv", parse_dates=["day"])
    j = K[K["month"] == "2026-01"]
    x = j["exceedance"].to_numpy("float64")
    mix = {"in envelope": int(j.loc[x <= 0, "n"].sum()),
           "0-30 s over": int(j.loc[(x > 0) & (x <= E.CLEAR), "n"].sum()),
           ">30 s over": int(j.loc[x > E.CLEAR, "n"].sum())}
    tot = sum(mix.values())
    par = json.loads((EXP / "analysis" / "arm_params.json").read_text())
    served = par["step|lofo|winsor"]["('january', None, None)"]
    sj = RX.slices(R[R["fold"] == "july_loo"], "y_pred")
    july = float(np.sqrt((sj["mse"] * sj["n"]).sum() / sj["n"].sum()))

    out = []
    for pooled in (True, False):
        h = RX.behaviour(R, "y_pred", pooled, mix, tot)
        c = {"in envelope": 0.0, "0-30 s over": served["lo"], ">30 s over": served["hi"]}
        for mixname, col in (("fold", "share_fold"), ("January 2026", "share_2026")):
            mse0 = float((h["mse"] * h[col]).sum())
            mse1 = float(sum(h.loc[l, col] * (h.loc[l, "mse"] + 2 * c[l] * h.loc[l, "bias"]
                                              + c[l] ** 2) for l in h.index))
            j0, j1 = np.sqrt(mse0), np.sqrt(mse1)
            out.append(dict(
                exposed_from="5 days (partly in sample)" if pooled else "january only (out of sample)",
                mix=mixname, jan_gain=j0 - j1,
                board_gain=(F26.board_estimate({"july_loo": july, "january": j0})
                            - F26.board_estimate({"july_loo": july, "january": j1}))))
    return pd.DataFrame(out)


# --------------------------------------------------------------------- B3 ---
def b3_date_clusters(R: pd.DataFrame) -> pd.DataFrame:
    """The same interval on the assignment's literal unit: the calendar day.

    `correct.py` clusters on `airport|calendar day`, which is exp-026
    `stack5.boot_delta`'s unit and the finer of the two.  The assignment says
    "calendar-day-clustered".  Both are reported from here.
    """
    K = pd.read_parquet(EXP / "analysis" / "corrections.parquet")
    R = R.merge(K[["MVT_ID_mvt", "fold"] + [f"c::{a}" for a in REPORTED]],
                on=["MVT_ID_mvt", "fold"], how="left", validate="one_to_one")
    rec = []
    for f in E.FOLDS:
        g = R[R["fold"] == f]
        y, ids = g["y_true"].to_numpy("float64"), g["MVT_ID_mvt"].to_numpy("float64")
        ref = g["y_pred"].to_numpy("float64")
        units = {"airport|day": C.clusters(g),
                 "day": pd.factorize(g["day"].dt.strftime("%Y-%m-%d"))[0]}
        for a in REPORTED:
            p = np.maximum(ref + g[f"c::{a}"].to_numpy("float64"), 60.0)
            for uname, cl in units.items():
                d, lo, hi = C.boot_gain(y, p, ref, cl, ids, f)
                rec.append(dict(fold=f, arm=a, unit=uname, k=int(cl.max() + 1),
                                gain=d, lo=lo, hi=hi))
    return pd.DataFrame(rec)


# --------------------------------------------------------------------- B5 ---
def b5_decomposition(R: pd.DataFrame) -> pd.DataFrame:
    """COMMON.md's four slices for every reported arm, not three of them."""
    K = pd.read_parquet(EXP / "analysis" / "corrections.parquet")
    R = R.merge(K[["MVT_ID_mvt", "fold"] + [f"c::{a}" for a in REPORTED]],
                on=["MVT_ID_mvt", "fold"], how="left", validate="one_to_one")
    B = pd.concat([F26.gates(f, check=False)[["MVT_ID_mvt", "nm_missing"]]
                   for f in E.FOLDS], ignore_index=True)
    R = R.merge(B, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if R["nm_missing"].isna().any():
        raise SystemExit("some rows have no nm_missing flag")
    rec = []
    for f in E.FOLDS:
        g = R[R["fold"] == f]
        y, ids = g["y_true"].to_numpy("float64"), g["MVT_ID_mvt"].to_numpy("float64")
        ref = g["y_pred"].to_numpy("float64")
        cols = {"stack-5 keep-noac (base)": ref}
        for a in REPORTED:
            cols[a] = np.maximum(ref + g[f"c::{a}"].to_numpy("float64"), 60.0)
        sl = {"rmse": np.ones(len(g), bool), "le3600": y <= 3600, "gt3600": y > 3600,
              "no_nm": g["nm_missing"].to_numpy() == 1}
        for lab, p in cols.items():
            row = dict(fold=f, variant=lab, rmse_clean=D.rmse_clean(y, p, ids, f))
            for k, m in sl.items():
                row[k] = float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
            rec.append(row)
    return pd.DataFrame(rec)


# --------------------------------------------------------------------- M9 ---
def m9_weighting() -> dict:
    """Re-weight the 2026 implied `r` on the rows that actually have one."""
    T = pd.read_csv(EXP / "analysis" / "implied_r_2026.csv")
    col = "r_hat_n" if "r_hat_n" in T.columns else None
    fl = T[T["flagged"]] if "flagged" in T.columns else T
    out = {"columns": list(T.columns), "flagged_days": int(len(fl))}
    w_all = fl["n"].to_numpy("float64")
    r = fl["r_hat_mean"].to_numpy("float64")
    out["weighted_all_rows"] = float(np.nansum(w_all * r) / np.nansum(w_all[~np.isnan(r)]))
    if col:
        w_obs = fl[col].to_numpy("float64")
        out["weighted_observed_rows"] = float(np.nansum(w_obs * r) / np.nansum(w_obs[~np.isnan(r)]))
    return out


def main() -> None:
    R = pd.read_parquet(EXP / "analysis" / "fold_rows.parquet")

    print("=== B1: the re-mix under both sources of slice behaviour ===")
    b1 = b1_out_of_sample(R)
    print(b1.round(4).to_string(index=False))
    b1.to_csv(EXP / "analysis" / "review_b1_remix.csv", index=False)

    print("\n=== B3: airport|day against the assignment's calendar day ===")
    b3 = b3_date_clusters(R)
    b3.to_csv(EXP / "analysis" / "review_b3_clusters.csv", index=False)
    piv = b3[b3.fold == "january"].pivot(index="arm", columns="unit",
                                         values=["gain", "lo", "hi"])
    print(piv.round(4).to_string())
    print("\n  clusters per fold:")
    print(b3.groupby(["fold", "unit"])["k"].first().to_string())
    pos = b3[(b3.fold == "january") & (b3.gain > 0)]
    print(f"\n  january arms with a positive gain: {len(pos)//2}; "
          f"intervals clear of zero: "
          f"{int((pos['lo'] > 0).sum())}")
    neg = b3[(b3.fold == "january") & (b3.hi < 0)]
    if len(neg):
        print("  january arms whose interval excludes zero NEGATIVELY:")
        print(neg[["arm", "unit", "gain", "lo", "hi"]].round(4).to_string(index=False))

    print("\n=== B5: the four required slices, every reported arm ===")
    b5 = b5_decomposition(R)
    b5.to_csv(EXP / "analysis" / "review_b5_decomp.csv", index=False)
    print(b5.round(3).to_string(index=False))

    print("\n=== M9: the implied-r weighting ===")
    print(json.dumps(m9_weighting(), indent=2))


if __name__ == "__main__":
    main()
