"""Render climb.log into the RESULTS ablation table."""
import json, sys
import pandas as pd
import prc.data as d

rows = []
for line in open(d.ROOT / "experiments/exp-003-anatomy/climb.log"):
    r = json.loads(line)
    for fold, s in r["res"].items():
        rows.append({"model": r["name"], "blocks": "+".join(r["blocks"]),
                     "target": r.get("target", "offset"), "fold": fold,
                     "n_feat": s["n_feat"], **{k: v for k, v in s.items() if k.startswith(("rmse", "n_bulk"))}})
T = pd.DataFrame(rows)
T = T.drop_duplicates(subset=["model", "fold"], keep="last")
piv = T.pivot(index=["model", "n_feat"], columns="fold", values="rmse_bulk")
order = [c for c in ["july", "december", "january"] if c in piv.columns]
piv = piv[order].reset_index()
print(piv.to_string(index=False))
print()
print(T[T.fold == "july"][["model", "rmse_all", "rmse_le3600", "rmse_gt3600", "rmse_noNM",
                           "rmse_bulk", "rmse_bulk_unseen_stand"]].to_string(index=False))
