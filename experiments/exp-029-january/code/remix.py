"""What the correction is worth on the board, and what the fold gate can see.

Two separate questions, and the fold gate answers only the second.

1. **Power.** The january fold carries 1,129 flagged rows of 153,706 (0.73 %).
   `ceilings.csv` says the best additive per-day constant on them is worth
   0.029 s of whole-fold clean RMSE. The queue's gate is 0.53 s. So no additive
   correction of any shape can pass that gate; the instrument is 18x too coarse
   for the object it is asked to measure. This script states the arithmetic.

2. **Value.** January 2026 carries 11,968 flagged rows of 152,719 (7.84 %), a
   10.7x mass ratio, so the same per-row effect is worth ~10x more there. The
   transplant is exp-014's re-mix: hold each slice's measured behaviour fixed
   and change only the mix. Its own weakness is stated with it -- the 2025
   analogue tops out at 73 s of exceedance and January 2026 reaches 1,350 s.
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
import envelope as E  # noqa: E402
import base_moved as BM  # noqa: E402
import prc.data as D  # noqa: E402
import frames as F26  # noqa: E402

ARM = "step|lofo|winsor"
#: The three bases on one instrument.  exp-014 priced this lever on candidate 1.
BASES = ("candidate 1", "blend-5", "stack-5")


def slices(g: pd.DataFrame, base: str = "y_pred") -> pd.DataFrame:
    """(n, RMSE, bias) on the three exposure slices, clean rows only."""
    ex = D.excluded_ids(str(g["fold"].iloc[0]))
    g = g[~g["MVT_ID_mvt"].isin(ex)]
    e = g[base].to_numpy("float64") - g["y_true"].to_numpy("float64")
    x = g["exceedance"].to_numpy("float64")
    out = []
    for lab, m in (("in envelope", x <= 0), ("0-30 s over", (x > 0) & (x <= E.CLEAR)),
                   (">30 s over", x > E.CLEAR)):
        if not m.any():
            out.append(dict(slice=lab, n=0, mse=np.nan, bias=np.nan))
            continue
        out.append(dict(slice=lab, n=int(m.sum()), mse=float((e[m] ** 2).mean()),
                        bias=float(e[m].mean())))
    return pd.DataFrame(out)


def behaviour(R: pd.DataFrame, base: str, pooled: bool, mix: dict, tot26: int
              ) -> pd.DataFrame:
    """Slice (n, MSE, bias) for one base under one choice of exposed-slice source.

    `pooled=True` takes the two exposed slices from all three folds together --
    2,416 rows, five airport-days.  `pooled=False` takes them from the january
    fold alone -- 1,129 rows, two days -- which is exp-014's own re-mix.  The
    two bracket the number, and the width of that bracket is the finding about
    the re-mix, not a detail of it.
    """
    jf = slices(R[R["fold"] == "january"], base).set_index("slice")
    beh = jf.copy()
    if pooled:
        for lab in ("0-30 s over", ">30 s over"):
            n = mse = bias = 0.0
            for f in E.FOLDS:
                s = slices(R[R["fold"] == f], base).set_index("slice").loc[lab]
                if s["n"]:
                    n += s["n"]; mse += s["n"] * s["mse"]; bias += s["n"] * s["bias"]
            beh.loc[lab, ["mse", "bias"]] = [mse / n, bias / n]
    beh["rmse"] = np.sqrt(beh["mse"])
    beh["share_2026"] = [mix[i] / tot26 for i in beh.index]
    beh["share_fold"] = beh["n"] / beh["n"].sum()
    return beh


def main() -> None:
    R = pd.read_parquet(EXP / "analysis" / "fold_rows.parquet")
    c1 = BM.candidate1()
    R = R.merge(pd.concat([c1[f] for f in E.FOLDS], ignore_index=True),
                on="MVT_ID_mvt", how="left", validate="one_to_one")
    R = R.rename(columns={"y_pred": "stack-5", "y_pred_anchor": "blend-5",
                          "cand1": "candidate 1"})
    K = pd.read_csv(EXP / "analysis" / "ranking_days.csv", parse_dates=["day"])
    jan26 = K[K["month"] == "2026-01"].copy()
    x = jan26["exceedance"].to_numpy("float64")
    mix = {"in envelope": int(jan26.loc[x <= 0, "n"].sum()),
           "0-30 s over": int(jan26.loc[(x > 0) & (x <= E.CLEAR), "n"].sum()),
           ">30 s over": int(jan26.loc[x > E.CLEAR, "n"].sum())}
    tot26 = sum(mix.values())

    # --- slice behaviour, three bases x two sources for the exposed slices ----
    tab = []
    for b in BASES:
        for pooled in (True, False):
            h = behaviour(R, b, pooled, mix, tot26)
            for lab in h.index:
                tab.append(dict(base=b, exposed_from="5 days" if pooled else "january only",
                                slice=lab, n=int(h.loc[lab, "n"]),
                                rmse=float(h.loc[lab, "rmse"]),
                                bias=float(h.loc[lab, "bias"])))
    S = pd.DataFrame(tab)
    S.to_csv(EXP / "analysis" / "remix_bases.csv", index=False)
    print("=== slice behaviour (clean), three bases ===")
    print(S.pivot_table(index=["base", "exposed_from"], columns="slice",
                        values="bias").round(1).to_string())
    print("\n  (bias = pred - truth; the in-envelope column is january's own on every row)")

    beh = behaviour(R, "stack-5", True, mix, tot26)
    print("\n=== stack-5, five-day exposed slices, and the two mixes ===")
    print(beh[["n", "rmse", "bias", "share_fold", "share_2026"]].round(4).to_string())

    jan_fold = float(np.sqrt((beh["mse"] * beh["share_fold"]).sum()))
    jan_remix = float(np.sqrt((beh["mse"] * beh["share_2026"]).sum()))
    sj = slices(R[R["fold"] == "july_loo"], "stack-5")
    july = float(np.sqrt((sj["mse"] * sj["n"]).sum() / sj["n"].sum()))
    b_fold = F26.board_estimate({"july_loo": july, "january": jan_fold})
    b_remix = F26.board_estimate({"july_loo": july, "january": jan_remix})
    print(f"\njanuary half: fold mix {jan_fold:.3f} s, January-2026 mix {jan_remix:.3f} s "
          f"({jan_remix - jan_fold:+.3f})")
    print(f"board estimate: fold mix {b_fold:.3f} s, re-mixed {b_remix:.3f} s "
          f"({b_remix - b_fold:+.3f})")

    # --- what a correction is worth under each mix ---------------------------
    par = json.loads((EXP / "analysis" / "arm_params.json").read_text())[ARM]
    served = par["('january', None, None)"]
    rows = []

    def value(c_lo: float, c_hi: float, label: str, h: pd.DataFrame | None = None,
              july_of_base: float | None = None) -> dict:
        """Apply a correction to a base's OWN slice behaviour, under both mixes."""
        h = beh if h is None else h
        jl = july if july_of_base is None else july_of_base
        c = {"in envelope": 0.0, "0-30 s over": c_lo, ">30 s over": c_hi}
        d = {}
        for mixname, col in (("fold", "share_fold"), ("January 2026", "share_2026")):
            mse0 = float((h["mse"] * h[col]).sum())
            mse1 = 0.0
            for lab in h.index:
                b = h.loc[lab, "bias"]
                mse1 += h.loc[lab, col] * (h.loc[lab, "mse"] + 2 * c[lab] * b + c[lab] ** 2)
            j0, j1 = float(np.sqrt(mse0)), float(np.sqrt(mse1))
            board0 = F26.board_estimate({"july_loo": jl, "january": j0})
            board1 = F26.board_estimate({"july_loo": jl, "january": j1})
            d[mixname] = (j0 - j1, board0 - board1)
        return dict(correction=label, c_lo=c_lo, c_hi=c_hi,
                    jan_gain_fold=d["fold"][0], board_gain_fold=d["fold"][1],
                    jan_gain_2026=d["January 2026"][0],
                    board_gain_2026=d["January 2026"][1])

    rows.append(value(served["lo"], served["hi"], f"served ({ARM})"))
    rows.append(value(-beh.loc["0-30 s over", "bias"], -beh.loc[">30 s over", "bias"],
                      "oracle band constants, stack-5 (in-sample)"))
    for b in ("candidate 1", "blend-5"):
        h = behaviour(R, b, True, mix, tot26)
        sjb = slices(R[R["fold"] == "july_loo"], b)
        jl = float(np.sqrt((sjb["mse"] * sjb["n"]).sum() / sjb["n"].sum()))
        rows.append(value(-h.loc["0-30 s over", "bias"], -h.loc[">30 s over", "bias"],
                          f"oracle band constants on {b}, its own base "
                          f"-- what exp-014 priced", h, jl))
    V = pd.DataFrame(rows)
    print("\n=== value of the correction, by mix (positive = RMSE down) ===")
    print(V.round(4).to_string(index=False))

    # --- how big would the January 2026 bias have to be? ---------------------
    share = beh.loc[">30 s over", "share_2026"]
    need = []
    for target in (0.53, 1.0, 2.0, 4.276):
        # solve for |b| such that an oracle constant on the >30 s slice alone
        # moves the board by `target`
        lo, hi = 0.0, 4000.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            mse = 0.0
            for lab in beh.index:
                m = beh.loc[lab, "mse"]
                if lab == ">30 s over":
                    m = m + mid ** 2          # a larger bias raises the slice MSE too
                mse += beh.loc[lab, "share_2026"] * m
            base_j = float(np.sqrt(mse))
            new_j = float(np.sqrt(mse - beh.loc[">30 s over", "share_2026"] * mid ** 2))
            g = (F26.board_estimate({"july_loo": july, "january": base_j})
                 - F26.board_estimate({"july_loo": july, "january": new_j}))
            lo, hi = (mid, hi) if g < target else (lo, mid)
        need.append(dict(board_target_s=target, required_abs_bias_s=round(0.5 * (lo + hi), 1)))
    N = pd.DataFrame(need)
    print("\n=== the bias stack-5 would need to carry on January 2026's >30 s rows "
          f"({mix['>30 s over']:,} rows, {share:.2%}) for the correction to be worth ===")
    print(N.to_string(index=False))
    print(f"\nmeasured on the 2025 analogue: |bias| = "
          f"{abs(beh.loc['>30 s over', 'bias']):.1f} s")

    (EXP / "analysis" / "remix.json").write_text(json.dumps(
        {"mix_2026": mix, "rows_2026": tot26,
         "slice": beh.reset_index().to_dict("records"),
         "january_fold_mix": jan_fold, "january_2026_mix": jan_remix,
         "july_loo": july, "board_fold_mix": b_fold, "board_remixed": b_remix,
         "value": V.to_dict("records"), "required_bias": N.to_dict("records")},
        indent=2))
    V.to_csv(EXP / "analysis" / "remix_value.csv", index=False)
    beh.reset_index().to_csv(EXP / "analysis" / "remix_slices.csv", index=False)


if __name__ == "__main__":
    main()
