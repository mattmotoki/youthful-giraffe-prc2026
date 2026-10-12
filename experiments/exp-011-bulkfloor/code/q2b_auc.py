"""How much of the train-versus-validation AUC is genuine covariate shift and how
much is a month index hiding in the feature set."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
sys.path.insert(0, "experiments/exp-011-bulkfloor/code")
import prc.data as d
import common as C
from q2_shift import frame, fit_ratio, FEATURES

MONTHY = ["doy", "n_dep_stand_month", "n_arr_stand_month", "cs_arr_count"]


def main():
    out = []; W = out.append
    W("# Q2(b) — how much of the separability is real shift\n")
    W("A classifier that separates the validation month from the training months "
      "at AUC 0.98 is only measuring covariate shift if the features it uses are "
      "features, not clocks. The model's set contains a day-of-year and three "
      "month-level counts; dropping them one group at a time:\n")
    rows = []
    for fold in ("july_loo", "december"):
        F = d.FOLDS[fold]
        tr = pd.concat([frame(m) for m in F.train_months], ignore_index=True)
        va = frame(F.valid_month)
        TR = tr[(tr.has_nm & tr.y.le(3600) & tr.off.notna() & tr.noac.ne(1)).to_numpy()]
        VA = va[(va.has_nm & va.y.le(3600) & va.noac.ne(1)).to_numpy()]
        for label, feats in [
            ("all model features", FEATURES),
            ("minus `doy`", [c for c in FEATURES if c != "doy"]),
            ("minus `doy` and the month-level counts", [c for c in FEATURES if c not in MONTHY]),
            ("minus those and the neighbour block",
             [c for c in FEATURES if c not in MONTHY and not c.startswith("nbr_")]),
        ]:
            w, auc, _, _ = fit_ratio(TR, VA, feats, 6)
            rows.append({"fold": fold, "classifier features": label, "n": len(feats), "AUC": auc,
                         "w p01": float(np.percentile(w, 1)), "w p50": float(np.percentile(w, 50)),
                         "w p99": float(np.percentile(w, 99)),
                         "effective sample size %": 100 * float(
                             w.sum() ** 2 / (w ** 2).sum() / len(w))})
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in rows[-1].items()}), flush=True)
    W(C.md_table(pd.DataFrame(rows).set_index("fold"), floatfmt="{:.3f}"))
    W("")
    C.write("q2b_auc.md", "\n".join(out))


if __name__ == "__main__":
    main()
