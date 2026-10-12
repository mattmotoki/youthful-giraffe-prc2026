"""The four COMMON.md slices plus the NM-tail slice, for every arm and fold."""
import json
import pandas as pd
import prc.data as d

LOG = d.ROOT / "experiments/exp-010-nmtail/climb.log"
ORDER = ["july", "july_loo", "december", "january"]
LABEL = {"C-phi": "head, `mode=\"phi\"` (**recommended**)",
         "C-p": "head, `mode=\"p\"` (two-point)",
         "C-phi_screened": "head + screened second stage"}


def main():
    rows = []
    for line in open(LOG):
        r = json.loads(line)
        if r["name"] not in LABEL:
            continue
        for arm, key in [("head", "head"), ("R0", "base_R0"), ("tail_model", "base_tail_model")]:
            s = r.get(key)
            if not s:
                continue
            rows.append({"model": LABEL[r["name"]] if arm == "head" else f"base: {arm}",
                         "fold": r["fold"], "all": s["rmse_all"], "le3600": s["rmse_le3600"],
                         "gt3600": s["rmse_gt3600"], "noNM": s["rmse_noNM"],
                         "bulk": s["rmse_bulk"], "nm_tail": s["rmse_nm_tail"],
                         "n_nm_tail": s["n_nm_tail"]})
    T = pd.DataFrame(rows).drop_duplicates(subset=["model", "fold"], keep="last")
    T["fold"] = pd.Categorical(T["fold"], ORDER, ordered=True)
    T = T.sort_values(["model", "fold"])
    out = ["| model | fold | RMSE all | RMSE <=3600 | RMSE >3600 | RMSE no-NM | RMSE bulk | **RMSE NM tail** | n |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m, g in T.groupby("model", sort=True, observed=True):
        for i, (_, r) in enumerate(g.iterrows()):
            out.append(f"| {m if i == 0 else ''} | {r.fold} | {r['all']:.1f} | {r.le3600:.1f} | "
                       f"{r.gt3600:.0f} | {r.noNM:.0f} | {r.bulk:.1f} | **{r.nm_tail:.0f}** | "
                       f"{int(r.n_nm_tail)} |")
    p = d.ROOT / "experiments/exp-010-nmtail/analysis/slices_full.md"
    p.write_text("# The four COMMON.md slices, plus the NM tail, for every arm and fold\n\n"
                 "All arms at one configuration (`prc.nmtail` VERSION 2 defaults: 900 rounds, "
                 "lr 0.04, 63 leaves, seed 7). `bulk` = NM join and taxi <= 3600 s.\n\n"
                 + "\n".join(out) + "\n")
    print(p, len(T), "model-fold rows")


if __name__ == "__main__":
    main()
