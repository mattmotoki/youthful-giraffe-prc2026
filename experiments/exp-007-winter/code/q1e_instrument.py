"""What the two unreachable rows do to the january fold as an INSTRUMENT.

If 55 % of a fold's squared error sits on two rows no feature can reach, then a
january comparison between two models is mostly a comparison of how each one
happened to predict those two rows. This measures that directly: every
three-fold retro-v1 out-of-fold line, january RMSE as reported and with the two
rows removed, and the resulting change of ranking.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import prc.data as D  # noqa: E402
import common as C  # noqa: E402

OOF = HERE.parent.parent / "exp-001-retro-v1" / "oof"
ROLL_IDS = None


def main():
    global ROLL_IDS
    base = pd.read_parquet(HERE.parent / "work" / "rows_january.parquet")
    import q1c_trimmed as q1c
    ROLL_IDS = set(base.loc[q1c.rollover_mask(base), D.ID])
    print("unreachable january ids:", ROLL_IDS)

    srcs = [("exp-001", OOF), ("exp-006", HERE.parent.parent / "exp-006-lirf" / "oof")]
    rows = []
    for tag, dr in srcs:
        for path in sorted(dr.glob("*-january.parquet")):
            ln = f"{tag}/{path.name[:-len('-january.parquet')]}"
            d = pd.read_parquet(path)
            if "y_true" not in d.columns:
                continue
            if (d["airport"] != "LIRF").sum() == 0:
                continue          # LIRF-only files carry none of the two rows
            d = d[d["y_true"].notna()]
            e = d["y_true"] - d["y_pred"]
            keep = ~d[D.ID].isin(ROLL_IDS)
            nl = d["airport"] != "LIRF"
            rows.append({
                "line": ln, "n": len(d),
                "jan_all": C.rmse(e), "jan_all_clean": C.rmse(e[keep]),
                "jan_nonlirf": C.rmse(e[nl]), "jan_nonlirf_clean": C.rmse(e[nl & keep]),
                "sse_share_2rows": float((e[~keep] ** 2).sum() / (e ** 2).sum()),
            })
    R = pd.DataFrame(rows).sort_values("jan_all")
    R["rank_as_reported"] = R["jan_all"].rank().astype(int)
    R["rank_clean"] = R["jan_all_clean"].rank().astype(int)
    R = R.sort_values("jan_all_clean")
    out = ["# exp-007 question 1e — the january fold as an instrument\n",
           "Every retro-v1 out-of-fold line that ran the `january` fold, ranked by january RMSE",
           "as reported and with the two unreachable date-rollover rows removed.\n",
           "| line | january RMSE as reported | rank | january RMSE, 2 rows removed | rank | non-LIRF as reported | non-LIRF clean | SSE share of the 2 rows |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        out.append(f"| `{r['line']}` | {r['jan_all']:.2f} | {r['rank_as_reported']} | **{r['jan_all_clean']:.2f}** | "
                   f"{r['rank_clean']} | {r['jan_nonlirf']:.2f} | **{r['jan_nonlirf_clean']:.2f}** | "
                   f"{100*r['sse_share_2rows']:.1f} % |")
    sp_rep = R["jan_all"].max() - R["jan_all"].min()
    sp_cln = R["jan_all_clean"].max() - R["jan_all_clean"].min()
    out.append(f"\nSpread across the {len(R)} lines: **{sp_rep:.1f} s** as reported, "
               f"**{sp_cln:.1f} s** with the two rows removed.\n")
    tau = R[["rank_as_reported", "rank_clean"]].corr(method="kendall").iloc[0, 1]
    out.append(f"Kendall rank correlation between the two orderings: **{tau:.2f}**.\n")
    # the two rows add a nearly constant SSE, because every line predicts them near 1,000 s
    core = R[R["jan_all_clean"] < 250]
    sse_add = (core["n"] * (core["jan_all"] ** 2 - core["jan_all_clean"] ** 2))
    out.append(f"The squared error the two rows add is nearly the same for every line: "
               f"{sse_add.min()/1e9:.3f}e9 to {sse_add.max()/1e9:.3f}e9 s² across the "
               f"{len(core)} lines below 250 s clean, a spread of "
               f"{100*(sse_add.max()/sse_add.min()-1):.1f} %. Every model predicts them "
               f"between 900 and 1,300 s, so they behave as an additive constant, not as a "
               f"tie-breaker. That is why the ranking survives.\n")
    slope = np.polyfit(core["jan_all_clean"], core["jan_all"], 1)[0]
    # analytic check: reported = sqrt(clean^2 + C/n) with C the two rows' constant
    # squared error, so d(reported)/d(clean) = clean/reported at the operating point.
    analytic = float((core["jan_all_clean"] / core["jan_all"]).mean())
    out.append(f"(The empirical slope is a check on the arithmetic, not a fit: with "
               f"`reported = sqrt(clean² + C/n)` and C constant, "
               f"`d(reported)/d(clean) = clean/reported`, which averages "
               f"**{analytic:.3f}** over these lines against the fitted {slope:.3f}.)\n")
    out.append(f"What they do instead is (i) raise the LEVEL by "
               f"{(core['jan_all'] - core['jan_all_clean']).mean():.0f} s, which is what breaks any "
               f"arithmetic against the board, and (ii) COMPRESS differences by a factor "
               f"**{slope:.2f}**: a 10 s improvement in the january error a model can actually "
               f"attack shows up as {10*slope:.1f} s in the reported january RMSE.\n")
    p = HERE.parent / "analysis" / "q1_instrument.md"
    p.write_text("\n".join(out))
    print(p.read_text())


if __name__ == "__main__":
    main()
