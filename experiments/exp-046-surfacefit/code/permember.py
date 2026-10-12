"""Per-member alignment <r, d_i> for all ten members, and the identity that voids the check."""
import sys, json
from pathlib import Path
import numpy as np
EXP = Path("/path/to/competitions/prc-data-challenge-2026/experiments/exp-046-surfacefit")
sys.path.insert(0, str(EXP / "code")); sys.path.insert(0, str(EXP.parents[1]))
import arms as A, chainfold as CF, rung511 as R, bagarm as BAG

rule = A.serving_rule()[0]
blk = {}
for f in A.FOLDS:
    m = A.FOLD_MONTH[f]
    b = CF.build_month(m, A.prefix_months_for(m), rule); R.assert_mode3(b, m, "shipped"); blk[m] = b
pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})

def load(size, s):
    return {f: v.astype("float64") for f, v in
            np.load(EXP / f"work/d-bundle-placebo{s}-{size}-free.npz").items()}
m255 = {s: load(255, s) for s in range(1, 11)}
m511 = {s: load(511, s) for s in range(1, 11)}
b255_k5 = {f: v.astype("float64") for f, v in np.load(EXP / "work/d-bag5-255-free.npz").items()}

out = {}
for f in A.FOLDS:
    y = pan[f].y_true.to_numpy("float64"); pb = pan[f].pbase.to_numpy("float64")
    r = y - pb - b255_k5[f]
    al = []
    for s in range(1, 11):
        d = BAG.build_bag(m511, pan, [s])[0][f] - BAG.build_bag(m255, pan, [s])[0][f]
        al.append(float(np.mean(r * d)))
    first, second = al[:5], al[5:]
    out[f] = {"per_member": al, "mean_1_5": float(np.mean(first)),
              "mean_6_10": float(np.mean(second)), "mean_all": float(np.mean(al)),
              "sd_across_members": float(np.std(al, ddof=1)),
              "sem_of_a_five_member_mean": float(np.std(al, ddof=1) / np.sqrt(5)),
              "gap_1_5_minus_6_10": float(np.mean(first) - np.mean(second))}
    o = out[f]
    print(f"{f:10s} " + " ".join(f"{x:6.1f}" for x in al))
    print(f"           1-5 mean {o['mean_1_5']:7.2f}   6-10 mean {o['mean_6_10']:7.2f}   "
          f"all {o['mean_all']:7.2f}   sd {o['sd_across_members']:6.2f}   "
          f"sem(5) {o['sem_of_a_five_member_mean']:6.2f}   "
          f"gap/sem {o['gap_1_5_minus_6_10']/ (o['sem_of_a_five_member_mean']*np.sqrt(2)):5.2f}")
Path(EXP / "analysis/a-per-member.json").write_text(json.dumps(out, indent=2, sort_keys=True))
print("\nwritten to analysis/a-per-member.json")
