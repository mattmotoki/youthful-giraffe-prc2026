"""The second repair: a blend instead of a correction.

§6's diagnosis is that the CORRECTION form injects my base head's error into the
shipped vector, because `y_chain - y_base` carries both heads' noise and my base
head is 610 s worse than the shipped one.  A blend does not subtract anything:

    arm = (1 - w) * shipped + w * y_head

with `w` fitted OUT OF FOLD on the other two folds and clipped to [0, 1].  Run
for every head, and the difference from the base head's blend is what that arm
adds to the shipped vector when the carrier cannot inject the base head's own
weakness.

This family is reported separately from `repair.py`'s carriers; it was written
after that family was measured and its p-values are not pooled with them.  Holm
runs over the heads it contains.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prc.tail_model import BAND_MIN  # noqa: E402
from evaluate import day_boot_gain, rmse, FOLDS  # noqa: E402

HEADS = ("y_base", "y_chain", "y_rot")


def main() -> None:
    m = pd.read_parquet(EXP / "work" / "repair_eval.parquet")
    heads = [h for h in HEADS if h in m.columns]
    live = (m.mvt_sched.to_numpy(float) < BAND_MIN)
    m["live"] = live
    w = {}
    for h in heads:
        for f in FOLDS:
            o = ((m.fold != f) & m.live).to_numpy()
            d = (m[h].to_numpy(float) - m.shipped.to_numpy(float))[o]
            e = (m.y_true.to_numpy(float) - m.shipped.to_numpy(float))[o]
            w[(h, f)] = float(np.clip(np.dot(d, e) / max(np.dot(d, d), 1e-9), 0.0, 1.0))
    wt = pd.DataFrame([{"head": h, "fold": f, "w": v} for (h, f), v in w.items()])

    for h in heads:
        ww = m.fold.map(lambda z, h_=h: w[(h_, z)]).to_numpy()
        m[f"blend_{h}"] = np.where(live, (1 - ww) * m.shipped + ww * m[h], m.shipped)

    rows = []
    for h in heads:
        col = f"blend_{h}"
        for fold in list(FOLDS) + ["POOLED"]:
            d = m if fold == "POOLED" else m[m.fold == fold]
            r = {"head": h, "fold": fold, "n": len(d),
                 "n_moved": int((d[col] != d.shipped).sum()),
                 "shift_rms_fold": float(np.sqrt(np.mean((d[col] - d.shipped) ** 2))),
                 "rmse_shipped": rmse(d.y_true, d.shipped), "rmse_arm": rmse(d.y_true, d[col])}
            r.update(day_boot_gain(d, "shipped", col))
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "blend_arms.csv", index=False)
    wt.to_csv(EXP / "analysis" / "blend_w.csv", index=False)

    # each arm's own contribution: the paired difference against the base blend
    contrib = []
    for h in [x for x in heads if x != "y_base"]:
        d = day_boot_gain(m.assign(a=m["blend_y_base"], b=m[f"blend_{h}"]), "a", "b")
        contrib.append(dict(d, what=f"blend({h[2:]}) minus blend(base)"))
    pd.DataFrame(contrib).to_csv(EXP / "analysis" / "blend_contribution.csv", index=False)

    with pd.option_context("display.width", 200):
        print("out-of-fold blend weights")
        print(wt.round(3).to_string(index=False))
        print()
        print(t.round(2).to_string(index=False))
        print()
        for d in contrib:
            print(f"{d['what']}: {d['gain']:+.2f} s "
                  f"[{d['lo']:+.2f}, {d['hi']:+.2f}], P(worse) {d['p_worse']:.3f}")
        p = t[t.fold == "POOLED"].sort_values("p_worse")
        k = len(p)
        p = p.assign(p_holm=np.maximum.accumulate(
            np.minimum(1.0, p.p_worse.to_numpy() * (k - np.arange(k)))))
        print(f"\nHolm over the {k} blend arms:")
        print(p[["head", "gain", "lo", "hi", "p_worse", "p_holm"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
