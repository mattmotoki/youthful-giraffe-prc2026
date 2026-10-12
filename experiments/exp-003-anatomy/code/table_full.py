"""Render the COMMON.md four-slice x three-fold table for every ablation row."""
import json
import pandas as pd
import prc.data as d

rows = []
for line in open(d.ROOT / "experiments/exp-003-anatomy/climb.log"):
    r = json.loads(line)
    for fold, s in r["res"].items():
        rows.append({"model": r["name"].replace("_3fold", ""), "fold": fold, "n_feat": s["n_feat"],
                     "bulk": s["rmse_bulk"], "all": s["rmse_all"],
                     "le3600": s["rmse_le3600"], "gt3600": s["rmse_gt3600"],
                     "noNM": s["rmse_noNM"]})
T = pd.DataFrame(rows).drop_duplicates(subset=["model", "fold"], keep="last")
order = ["july", "december", "january"]
T["fold"] = pd.Categorical(T["fold"], order, ordered=True)
T = T.sort_values(["model", "fold"])
lines = ["| model | features | fold | RMSE all | RMSE y<=3600 | RMSE y>3600 | RMSE no-NM | RMSE bulk |",
         "|---|---|---|---|---|---|---|---|"]
for m, g in T.groupby("model", sort=True):
    for i, (_, r) in enumerate(g.iterrows()):
        lines.append(f"| {m if i == 0 else ''} | {int(r.n_feat) if i == 0 else ''} | {r.fold} | "
                     f"{r['all']:.1f} | {r.le3600:.1f} | {r.gt3600:.0f} | {r.noNM:.0f} | **{r.bulk:.2f}** |")
p = d.ROOT / "experiments/exp-003-anatomy/analysis/ablation_full.md"
p.write_text("# Ablation: the four COMMON.md slices on all three folds\n\n"
             "bulk = NM join and taxi <= 3600 s, the metric the assignment scores.\n\n"
             + "\n".join(lines) + "\n")
print(p, len(T), "model-fold rows")
missing = [(m, f) for m in T.model.unique() for f in order
           if not ((T.model == m) & (T.fold == f)).any()]
print("missing model-fold cells:", missing)
