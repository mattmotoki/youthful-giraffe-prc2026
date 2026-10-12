"""Fill the shift table in RESULTS.md from climb.log."""
import json, sys
import pandas as pd
import prc.data as d

R = d.ROOT / "experiments/exp-011-bulkfloor/RESULTS.md"
LOG = d.ROOT / "experiments/exp-011-bulkfloor/climb.log"
ORDER = ["july", "july_loo", "december", "january"]

rows, rank = {}, {}
for line in open(LOG):
    r = json.loads(line)
    if r.get("name") == "shift":
        rows[r["fold"]] = r
    if r.get("name") == "shift_ranking":
        rank[r["month"]] = r

t = ["| fold | classifier AUC (with `doy`) | effective sample size | weight p1 / p50 / p99 | unweighted bulk | weighted bulk | paired difference, 95 % day-clustered CI |",
     "|---|---|---|---|---|---|---|"]
for f in ORDER:
    r = rows.get(f)
    if not r:
        continue
    t.append(f"| {f} | {r['auc']:.4f} ({r['auc_with_doy']:.4f}) | {100*r['ess']:.1f} % | "
             f"{r['w_p01']:.3f} / {r['w_p50']:.3f} / {r['w_p99']:.2f} | "
             f"**{r['bulk_unweighted']:.2f}** | {r['bulk_weighted']:.2f} | "
             f"**+{r['bulk_weighted']-r['bulk_unweighted']:.2f}** [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}] |")
text = "\n".join(t)
if rank:
    t2 = ["", "### The same classifier against the two ranking months", "",
          "Trained on all twelve 2025 months against each 2026 month's bulk rows:", "",
          "| month | rows | classifier AUC | effective sample size | weight p1 / p50 / p99 / max |",
          "|---|---|---|---|---|"]
    for m in sorted(rank):
        r = rank[m]
        t2.append(f"| {m} | {r['n_valid']:,} | {r['auc']:.4f} | {100*r['ess']:.1f} % | "
                  f"{r['w_p01']:.3f} / {r['w_p50']:.3f} / {r['w_p99']:.2f} / {r['w_max']:.1f} |")
    text += "\n" + "\n".join(t2)
s = R.read_text()
if "SHIFT_TABLE_PLACEHOLDER" in s:
    R.write_text(s.replace("SHIFT_TABLE_PLACEHOLDER", text))
else:
    print("placeholder gone; table follows\n")
print(text)
