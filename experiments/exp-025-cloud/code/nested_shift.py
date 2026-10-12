"""exp-025-cloud: what one nested refit costs a member, fold by fold.

For the clean instrument (the nested validation rule) each scored fold's rotation partners
are refit with the scored month also dropped. This prices the difference between
that member and the published one, per member and per fold:

    shift = RMSE(nested, ten training months) - RMSE(published, eleven)

**The shift mixes two things and cannot separate them.** Dropping a month closes
the label channel AND removes a twelfth of the training data. Both push the same
way, so a shift is an upper bound on the channel, never a measurement of it.
Say so wherever the number is quoted.

    .venv/bin/python experiments/exp-025-cloud/code/nested_shift.py \
        --nested experiments/exp-025-cloud/oof-nested --out .../nested-shift.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

#: Where each member's PUBLISHED (unnested, eleven-month) OOF lives. `{fold}` is
#: the base fold. A single-file member gives (path, fold-column) instead.
PUBLISHED = {
    "F1-fresh": ("experiments/exp-012-blend2/published/F1-fresh-s1-{fold}.parquet", None),
    "L2-hub8725": ("experiments/exp-012-blend2/published/L2-hub8725-{fold}.parquet", None),
    "V3-allin": ("experiments/exp-012-blend2/published/V3-allin-v1-{fold}.parquet", None),
    "T-tailv4": ("experiments/exp-009-noac2/oof/exp-009-noac2-oof-v3.parquet", "fold"),
}
#: nested fold -> (base fold it validates, the scored fold whose rotation it serves)
NESTED = {
    "july_loo_nx_dec": ("july_loo", "december"),
    "july_loo_nx_jan": ("july_loo", "january"),
    "december_nx_jul": ("december", "july_loo"),
    "december_nx_jan": ("december", "january"),
    "january_nx_jul": ("january", "july_loo"),
    "january_nx_dec": ("january", "december"),
}


def rmse(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.sqrt(((y - p) ** 2).mean()))


#: Same-environment (Kaggle, unnested) baselines that did not come from this
#: directory. F1-fresh's `july_loo` is the acceptance test of RESULTS 9.3.
EXTRA_KAGGLE: dict[tuple[str, str], str] = {}   # all baselines now live in oof-nested/


def kaggle_unnested(member: str, fold: str, nd: Path) -> pd.DataFrame | None:
    """The member's UNNESTED fold, run on Kaggle.

    This is the only correct baseline for a nested shift. Measuring against the
    published workstation OOF instead mixes in the environment, which RESULTS
    10.6 shows is 98 % of the difference for the one fold where both exist.
    """
    cand = nd / f"{member}--{fold}.parquet"
    if not cand.exists():
        rel = EXTRA_KAGGLE.get((member, fold))
        cand = ROOT / rel if rel else None
    if cand is None or not cand.exists():
        return None
    return pd.read_parquet(cand)[["MVT_ID_mvt", "y_true", "y_pred"]].copy()


def published(member: str, fold: str) -> pd.DataFrame | None:
    tmpl, foldcol = PUBLISHED[member]
    path = ROOT / tmpl.format(fold=fold)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if foldcol:
        df = df[df[foldcol] == fold]
    return df[["MVT_ID_mvt", "y_true", "y_pred"]].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nested", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nd = Path(a.nested)

    rows = []
    for f in sorted(nd.glob("*--*.parquet")):
        member, nested_fold = f.stem.split("--", 1)
        if nested_fold not in NESTED or member not in PUBLISHED:
            print(f"skip {f.name}: unknown member or fold")
            continue
        base, scored = NESTED[nested_fold]
        nst = pd.read_parquet(f)[["MVT_ID_mvt", "y_true", "y_pred"]]
        rec = {"member": member, "nested_fold": nested_fold, "base_fold": base,
               "serves_scored_fold": scored, "n_nested": len(nst),
               "rmse_nested": rmse(nst.y_true, nst.y_pred)}

        for label, base_df in (("kaggle", kaggle_unnested(member, base, nd)),
                               ("published", published(member, base))):
            if base_df is None:
                continue
            m = nst.merge(base_df, on="MVT_ID_mvt", how="inner", suffixes=("_nx", "_b"))
            # y_true must agree: both files score the same month, so a
            # disagreement means the frames are not the same rows.
            assert not (m.y_true_nx != m.y_true_b).any(), f"{member} {nested_fold} y_true differs"
            d = (m.y_pred_nx - m.y_pred_b).to_numpy(float)
            rec.update({
                f"rmse_base_{label}": rmse(m.y_true_b, m.y_pred_b),
                f"rmse_nested_on_{label}_rows": rmse(m.y_true_nx, m.y_pred_nx),
                f"shift_vs_{label}": rmse(m.y_true_nx, m.y_pred_nx) - rmse(m.y_true_b, m.y_pred_b),
                f"n_matched_{label}": len(m),
                f"pred_mean_abs_{label}": float(np.abs(d).mean()),
                f"pred_corr_{label}": float(np.corrcoef(m.y_pred_nx, m.y_pred_b)[0, 1]),
            })
        if "shift_vs_kaggle" not in rec:
            rec["note"] = ("no same-environment baseline yet; shift_vs_published mixes "
                           "the machine in and RESULTS 10.6 measures that at +0.275 s")
        rows.append(rec)

    if not rows:
        raise SystemExit(f"no nested OOF under {nd}")
    tab = pd.DataFrame(rows).sort_values(["member", "nested_fold"])
    cols = [c for c in ("member", "nested_fold", "base_fold", "serves_scored_fold",
                        "rmse_base_kaggle", "rmse_nested", "shift_vs_kaggle",
                        "rmse_base_published", "shift_vs_published",
                        "pred_mean_abs_kaggle") if c in tab.columns]
    print(tab[cols].to_string(index=False))
    if "shift_vs_kaggle" in tab:
        print("\nper member, the SAME-ENVIRONMENT shift (channel + one month of data):")
        g = tab.dropna(subset=["shift_vs_kaggle"]).groupby("member")["shift_vs_kaggle"]
        print(pd.DataFrame({"n": g.size(), "mean": g.mean(),
                            "min": g.min(), "max": g.max()}).to_string())
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
