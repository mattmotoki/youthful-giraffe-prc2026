"""Re-screen the four saturation candidates by the BLEND criterion (desk, 2026-09-06).

Every screen in this experiment ranked candidates by their standalone fold RMSE
and looked at the blend gain second. The Huber budget case showed that ordering
can invert the answer: a member 3.57 s worse on its own score made the flat blend
0.18 s better, because its errors moved into the tail and its correlation with
the other members fell. So the criterion is the ROTATED BLEND RMSE with the
candidate in, against the same rotation without it — honest weights, fitted only
on the folds a candidate is not being scored on.

This is OOF-only and needs no fits.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import blend as BL  # noqa: E402
import prc.blend as B  # noqa: E402

CORE = json.loads((HERE / (os.environ.get("SCREEN_CORE") or "members-3b.json")).read_text())["members"]
#: Three of the four were never on all three folds -- which is exactly WHY they
#: were screened in sample originally, and it is a limit on this re-screen rather
#: than something the re-screen can fix. Each is scored on the folds it has.
CANDIDATES = {
    "A0-plain":  ("experiments/exp-001-retro-v1/oof/A0-plain-jloo-{fold}.parquet", ["july_loo"]),
    "X1-deice":  ("experiments/exp-001-retro-v1/oof/X1-deice-{fold}.parquet", ["july_loo", "january"]),
    "E12-mlp":   ("experiments/exp-012-blend2/oof/E12-mlp-{fold}.parquet", ["july_loo", "december"]),
    "F1-fresh":  ("experiments/exp-012-blend2/published/F1-fresh-s1-{fold}.parquet",
                  ["july_loo", "december", "january"]),
    #: retro-v1's v3-frame arms. Two folds only (no december), so these are
    #: two-fold rotations: fit on one month, score the other.
    "V3-allin":    ("experiments/exp-001-retro-v1/oof/V3-allin-{fold}.parquet", ["july_loo", "january"]),
    "V3-full":     ("experiments/exp-001-retro-v1/oof/V3-full-{fold}.parquet", ["july_loo", "january"]),
    "V3-fresh":    ("experiments/exp-001-retro-v1/oof/V3-fresh-{fold}.parquet", ["july_loo", "january"]),
    "V3-rollover": ("experiments/exp-001-retro-v1/oof/V3-rollover-{fold}.parquet", ["july_loo", "january"]),
    "V3-wins3600": ("experiments/exp-001-retro-v1/oof/V3-wins3600-{fold}.parquet", ["july_loo", "january"]),
}


def rotated_on(files: dict, mode: str, folds: list) -> float:
    """Mean rotated RMSE over the given folds, each scored on weights fitted on
    the others. With one fold there is nothing to rotate on, so the weights are
    fitted in sample and the number is an upper bound, marked * in the output."""
    data, names = BL.load(files, folds=tuple(folds))
    tot = []
    for f in folds:
        src = [g for g in folds if g != f] or [f]
        fit = pd.concat([data[g] for g in src], ignore_index=True)
        w = B.fit_weights(fit[names].to_numpy("float64"),
                          fit["y_true"].to_numpy("float64"), mode)
        d = data[f]
        p = np.maximum(d[names].to_numpy("float64") @ w, B.FLOOR)
        tot.append(float(np.sqrt(((d["y_true"].to_numpy("float64") - p) ** 2).mean())))
    return float(np.mean(tot))


def standalone(path_tmpl: str, folds) -> dict:
    out = {}
    for f in folds:
        p = ROOT / path_tmpl.format(fold=f)
        if not p.exists():
            return {}
        d = pd.read_parquet(p)
        if "fold" in d.columns:
            d = d[d["fold"] == f]
        g = d.dropna(subset=["y_true", "y_pred"])
        out[f] = float(np.sqrt(((g.y_true - g.y_pred) ** 2).mean()))
    return out


def main(mode="nnls"):
    import provenance
    print(provenance.stamp() + "\n")
    print(f"{'candidate':>10} {'folds':>22} {'standalone':>11} {'core':>9} "
          f"{'+cand':>9} {'gain':>8}")
    rows = []
    for n, (tmpl, folds) in CANDIDATES.items():
        sa = standalone(tmpl, folds)
        if len(sa) != len(folds):
            print(f"{n:>10}   -- missing a declared fold"); continue
        core_b = rotated_on(CORE, mode, folds)
        files = dict(CORE); files[n] = tmpl
        try:
            cand_b = rotated_on(files, mode, folds)
        except Exception as e:
            print(f"{n:>10}   -- {type(e).__name__}: {str(e)[:44]}"); continue
        star = "*" if len(folds) == 1 else " "
        print(f"{n:>10} {'+'.join(folds):>22} {np.mean(list(sa.values())):11.2f} "
              f"{core_b:9.3f} {cand_b:9.3f} {core_b-cand_b:+7.3f}{star}")
        rows.append({"name": n, "standalone": float(np.mean(list(sa.values()))),
                     "blend": cand_b, "gain": core_b - cand_b})
    print("  * one fold only: weights fitted in sample, so an upper bound\n")
    # NO cross-candidate ordering is printed, and that is deliberate. The
    # candidates sit on DIFFERENT fold sets, and the folds have very different
    # levels (december ~210, july_loo ~290, january ~343), so a mean standalone
    # RMSE over july_loo+december is not comparable with one over all three. An
    # earlier version of this script ranked them anyway and reported that the two
    # orderings "disagree" -- which measured the fold sets, not the members.
    # What IS comparable is each candidate's own gain against its own core.
    print("\n  Each row compares a candidate against the SAME core on the SAME folds.")
    print("  Rows are not comparable with each other: different fold sets, different levels.")
    pos = [r["name"] for r in rows if r["gain"] > 0.01]
    print(f"  Candidates that add anything under the blend criterion: "
          f"{', '.join(pos) if pos else 'none'}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
