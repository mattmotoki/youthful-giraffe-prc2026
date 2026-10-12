"""Review round 1, B5 and B6: the per-airport floor the assignment asked for,
with the bound and the model on the same rows, and `key_raw` as the group key
(stand names repeat across airports)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
sys.path.insert(0, "experiments/exp-011-bulkfloor/code")
import prc.data as d
import common as C
from q1a_pairs import pooled_sigma
from q1b_residuals import load_best, month_frame

TIGHT = ["key_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "h2", "gap_bin", "q_bin"]
MID = ["key_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "h2"]


def main():
    out = []; W = out.append
    W("# Q1(e) — the floor per airport, both quantities on the same rows\n")
    W("The bound and the model RMSE are both computed on the rows that have a "
      "near-duplicate, which is the only comparison that means anything "
      "(review round 1, B5/B6). Group key is `key_raw` = airport|stand, because "
      "stand names repeat across airports.\n")
    rows = []
    for fold in ("july_loo", "december"):
        m = d.FOLDS[fold].valid_month
        f = month_frame(m)
        f["key_raw"] = f.apt.astype(str) + "|" + f.stand_raw.astype(str)
        o, name = load_best(fold)
        j = f.merge(o, on="MVT_ID_mvt")
        b = j[j.has_nm & j.noac.ne(1) & j.y_true.le(3600)].copy()
        for label, keys in (("loose", MID), ("tight", TIGHT)):
            for a, ga in list(b.groupby("apt", observed=True)) + [("ALL", b)]:
                n = ga.groupby(keys, observed=True)["y_true"].transform("size")
                sel = (n >= 2).to_numpy()
                if sel.sum() < 50:
                    continue
                s = pooled_sigma(ga, keys)
                rows.append({"fold": fold, "match": label, "airport": a,
                             "rows with a duplicate": int(sel.sum()),
                             "share of the airport's bulk %": 100 * sel.mean(),
                             "bound": s["sigma"],
                             "model on those rows": d.rmse(ga.y_true[sel], ga.y_pred[sel])})
                rows[-1]["model minus bound"] = rows[-1]["model on those rows"] - rows[-1]["bound"]
    T = pd.DataFrame(rows)
    for fold in ("july_loo", "december"):
        for label in ("tight", "loose"):
            s = T[(T.fold == fold) & (T["match"] == label)].set_index("airport")
            W(f"## {fold}, {label} match "
              f"({'stand x runway x operator x 2h' if label=='loose' else '... + AOBT-LOBT and queue bins'})\n")
            W(C.md_table(s.drop(columns=["fold", "match"]).sort_values("model minus bound"),
                         floatfmt="{:.1f}"))
            W("")
    T.to_csv(C.ANA / "airport_floor.csv", index=False)
    C.write("q1e_airport_floor.md", "\n".join(out))


if __name__ == "__main__":
    main()
