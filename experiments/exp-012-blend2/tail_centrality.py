"""Does a transformation that worsens a member's tail also make it more CENTRAL?

The question separates two things the rotation cannot. A rotation says what a
change cost the blend; this says why. A member can be worse in the tail and keep
its weight if it is *differently* wrong, and it loses twice if its tail moves
toward the consensus.

The quantity is the change in a member's mean residual correlation with the OTHER
members, computed on TAIL ROWS ONLY, between two arms that differ in one thing.

The scale comes from a pair this lane already holds. The Huber arm at 2,100 and
8,725 rounds differs only in budget and moves the tail by 26 s RMSE:

    HUB2100   tail rmse 2588.1   mean tail correlation 0.777
    HUB8725   tail rmse 2561.5   mean tail correlation 0.773

So **0.004 of correlation per 26 s of tail RMSE** under a round-budget change.
exp-018 pointed out that a clip compresses the learned response far more directly
than a budget does, so the same measurement under a clip is the informative one:
a delta at or below 0.004 says the coupling is absent under that intervention
too, and a much larger one says a clip really does move a member toward the
middle.

  python tail_centrality.py <arm_a.parquet> <arm_b.parquet> [label_a] [label_b] \\
      [--fold F] [--exclude MEMBER]

Each arm is a member OOF for one fold with MVT_ID_mvt, y_true, y_pred.

`--exclude` drops a member from the comparison set, and is REQUIRED when the arm
under test is itself one of them. Without it that member contributes a
self-correlation of 1.0 to arm A, and its merely-high correlation with arm A to
arm B, which inflates the level and puts a spurious difference into the delta —
the whole quantity being measured. Caught on the Huber self-test, where the
contamination moved the level from 0.777 to 0.844.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import prc.blend as B  # noqa: E402

OTHERS = {
    "T-tailv2":  "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v2.parquet",
    "C2-cat":    "experiments/exp-012-blend2/pinned/C2-cat--C2-d8p-{fold}.parquet",
    "L2-hub":    "experiments/exp-012-blend2/pinned/L2-hub--L2-hub-{fold}.parquet",
    "D1-winter": "experiments/exp-012-blend2/published/D1-winter-deice-v1-{fold}.parquet",
}
TAIL_CUT = 3600.0
# Below this, a rate is a division of ~0 by ~0. The winter feature-clip pair moves
# the tail by 0.1 s and produced a spurious "0.0050 per 26 s" from a centrality
# change of 0.0000; the ratio is only a rate when the denominator is real.
MIN_DT = 5.0
REFERENCE = "0.004 of correlation per 26 s of tail RMSE (Huber budget pair)"


def load(path: str, fold: str, name: str) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / str(path).format(fold=fold) if not Path(path).is_absolute()
                        else path)
    if "fold" in d.columns:
        d = d[d["fold"] == fold]
    return d[["MVT_ID_mvt", "y_pred"]].rename(columns={"y_pred": name})


def main(argv: list[str]) -> int:
    import provenance
    fold, exclude, pos = "july_loo", [], []
    it = iter(argv)
    for tok in it:
        if tok == "--fold":
            fold = next(it)
        elif tok == "--exclude":
            exclude.append(next(it))
        else:
            pos.append(tok)
    if len(pos) < 2:
        print(__doc__); return 2
    arm_a, arm_b = pos[0], pos[1]
    label_a = pos[2] if len(pos) > 2 else "arm A"
    label_b = pos[3] if len(pos) > 3 else "arm B"
    bad = [e for e in exclude if e not in OTHERS]
    if bad:
        print(f"  --exclude names no member: {bad}; known {list(OTHERS)}"); return 2
    others_used = {k: v for k, v in OTHERS.items() if k not in exclude}
    names = {Path(v.format(fold=fold)).name for v in others_used.values()}
    for arm in (arm_a, arm_b):
        if Path(arm).name in names:
            print(f"  {Path(arm).name} is itself in the comparison set; "
                  f"pass --exclude for it or the delta is contaminated"); return 2
    print(provenance.stamp())
    print(f"  comparing against {list(others_used)}"
          + (f"; excluded {exclude}" if exclude else "") + "\n")
    lo, hi = json.loads((ROOT / "experiments/exp-008-diversity/guard_bounds.json").read_text())[fold]
    base = pd.read_parquet(ROOT / OTHERS["L2-hub"].format(fold=fold))[
        ["MVT_ID_mvt", "y_true", "d_takeoff_aobt"]]
    for n, p in others_used.items():
        base = base.merge(load(p, fold, n), on="MVT_ID_mvt", how="left")
    for p, n in ((arm_a, "_A"), (arm_b, "_B")):
        d = pd.read_parquet(p)
        if "fold" in d.columns:
            d = d[d["fold"] == fold]
        base = base.merge(d[["MVT_ID_mvt", "y_pred"]].rename(columns={"y_pred": n}),
                          on="MVT_ID_mvt", how="left")
    miss = [c for c in list(others_used) + ["_A", "_B"] if base[c].isna().any()]
    if miss:
        print(f"  members not covering every row: {miss}"); return 1
    a = base["d_takeoff_aobt"].to_numpy("float64"); y = base["y_true"].to_numpy("float64")
    prep = lambda v: B.apply_residual_guard(
        np.maximum(np.asarray(v, "float64"), B.FLOOR), a, float(lo), float(hi), B.FLOOR)
    others = [y - prep(base[n].to_numpy("float64")) for n in others_used]
    tail = y > TAIL_CUT
    print(f"  fold {fold}, tail = {int(tail.sum()):,} rows of {len(y):,} "
          f"({100*tail.mean():.2f} %)\n")
    print(f"  {'arm':>26} {'tail rmse':>10} {'mean tail corr':>15}")
    out = {}
    for col, lab in (("_A", label_a), ("_B", label_b)):
        r = y - prep(base[col].to_numpy("float64"))
        tr = float(np.sqrt((r[tail] ** 2).mean()))
        c = float(np.mean([np.corrcoef(r[tail], o[tail])[0, 1] for o in others]))
        out[lab] = (tr, c)
        print(f"  {lab:>26} {tr:10.1f} {c:15.3f}")
    (ta, ca), (tb, cb) = out[label_a], out[label_b]
    dt, dc = tb - ta, cb - ca
    print(f"\n  tail RMSE change {dt:+.1f} s;  centrality change {dc:+.4f}")
    if abs(dt) < MIN_DT:
        print(f"\n  NO RATE — tail RMSE moved only {abs(dt):.1f} s (< {MIN_DT:.0f} s).")
        print("  This pair does not calibrate a coupling rate and does not test one:")
        print("  the ratio would be noise over noise. Reported as a failed calibration.")
        return 0
    if abs(dt) > 1e-9:
        print(f"  coupling: {abs(dc)/abs(dt)*26:+.4f} of correlation per 26 s of tail RMSE")
        print(f"  reference: {REFERENCE}")
        print("\n  " + ("COUPLED — the worse tail is materially more central, so a clip costs the "
                        "blend twice" if abs(dc) / abs(dt) * 26 > 0.008 else
                        "NOT COUPLED at this scale — worse but differently wrong, so exp-018's "
                        "objection holds under a clip as well as under a budget"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
