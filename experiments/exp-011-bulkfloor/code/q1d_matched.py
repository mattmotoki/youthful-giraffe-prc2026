"""The like-for-like comparison the first pass missed: the model's RMSE on
exactly the rows the near-duplicate bound is computed on."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
sys.path.insert(0, "experiments/exp-011-bulkfloor/code")
import prc.data as d
import common as C
from q1a_pairs import pooled_sigma, COLS
from q1b_residuals import load_best, month_frame

MID = ["stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "h2"]
TIGHT = MID + ["gap_bin", "q_bin"]


def main():
    out = []; W = out.append
    W("# Q1(d) — the model scored on the rows the bound is computed on\n")
    W("The near-duplicate bound only exists on rows that have a near-duplicate: "
      "9 % of the bulk at the tight match, 80 % at the loose one, and those rows "
      "are a biased sample (busy stands, repeated identical operations). Comparing "
      "a bound computed on 9 % of rows with a model RMSE computed on 100 % is not a "
      "comparison. Here both are on the same rows.\n")
    rows = []
    for fold in ("july_loo", "december"):
        m = d.FOLDS[fold].valid_month
        f = month_frame(m)
        o, name = load_best(fold)
        j = f.merge(o, on="MVT_ID_mvt")
        b = j[j.has_nm & j.noac.ne(1) & j.y_true.le(3600)].copy()
        for label, keys in (("stand x runway x operator x 2h", MID),
                            ("... + AOBT-LOBT and queue bins", TIGHT)):
            n = b.groupby(keys, observed=True)["y_true"].transform("size")
            sel = (n >= 2).to_numpy()
            s = pooled_sigma(b, keys)
            rows.append({"fold": fold, "match": label, "rows with a duplicate": int(sel.sum()),
                         "share of bulk %": 100 * sel.mean(),
                         "bound (pooled within-group sd)": s["sigma"],
                         "model RMSE on those rows": d.rmse(b.y_true[sel], b.y_pred[sel]),
                         "model RMSE on all bulk rows": d.rmse(b.y_true, b.y_pred)})
    T = pd.DataFrame(rows).set_index("fold")
    W(C.md_table(T, floatfmt="{:.1f}"))
    W("")
    C.write("q1d_matched.md", "\n".join(out))
    print(T.to_string())


if __name__ == "__main__":
    main()
