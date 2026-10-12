"""The repair: four narrower ways to put the chain on the shipped vector.

§4's first arm added the whole difference of two self-fitted heads to the
shipped out-of-fold vector and lost.  The diagnosis is in the same table: the
self-fitted base head is worse than the shipped vector on those rows and far
from it, so a correction defined relative to it is mostly noise on rows the
shipped model already fits (exp-033 §8's negative, seen again).  Inside the
self-fitted family the arm wins on the pooled folds, so the information is real
and the CARRIER is wrong.  The current numbers live in RESULTS.md §5 and §6 and
are deliberately not repeated here: three rounds of review moved them.

Four narrower carriers, all measured, none dropped:

    full     shipped + (y_chain - y_base)                     -- §4's arm
    clf      shipped + (p_chain - p_base) * off_np_base       -- the classifier
             change alone, moved along the mixture line the shipped head sits on
    reg      shipped + [(1-p_base)*off_base - (1-p_base)*off_chain]  -- the
             regressor change alone
    clip     min(shipped, g_prev2 - q)                        -- the assignment's
             other named form, the physical clip

and each with a shrinkage `lam` fitted OUT OF FOLD (on the other two folds) so
the size of the move is not chosen on the fold it is scored on.

Holm over the family of four carriers is applied to the pooled gains, because
four arms were tried on one set of 491 rows.
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
from prc.tail_model import BAND_MIN, MIN_TAXI  # noqa: E402
from build import LAB  # noqa: E402
from evaluate import day_boot_gain, rmse, load_shipped, FOLDS  # noqa: E402
from step2 import margins, V9  # noqa: E402

CLIP_LEVEL = 0.001
CLIP_VAR = "g_prev2"


def corrections(m: pd.DataFrame, arm: str = "chain") -> pd.DataFrame:
    """The four raw corrections, before shrinkage, zero above BAND_MIN."""
    live = (m.mvt_sched.to_numpy(float) < BAND_MIN).astype(float)
    off_base = (m.mvt_sched - m.bulk_base).to_numpy(float)
    off_arm = (m.mvt_sched - m[f"bulk_{arm}"]).to_numpy(float)
    out = pd.DataFrame(index=m.index)
    out["full"] = live * (m[f"y_{arm}"] - m.y_base).to_numpy(float)
    out["clf"] = live * (m[f"p_{arm}"] - m.p_base).to_numpy(float) * off_base
    out["reg"] = live * (1.0 - m.p_base.to_numpy(float)) * (off_base - off_arm)
    out["clip"] = live * np.minimum(m.bound.to_numpy(float) - m.shipped.to_numpy(float), 0.0)
    return out


def shift_388(q: pd.DataFrame, arms=("chain",)) -> pd.DataFrame:
    """How far each carrier would move the 388 ranking noac rows off v9.

    The gate's second half (`docs/QUEUE-2026-09-07.md` §4) is 500 s RMS on these
    rows.  This block was missing from the code until the round-1 review found
    `analysis/shift_388.csv` had no producer; the numbers it now writes
    reproduce that file.
    """
    r = pd.read_parquet(EXP / "work" / "ranking_head.parquet")
    n26 = pd.read_parquet(EXP / "work" / "noac_2026.parquet")
    r = r.merge(n26[["MVT_ID_mvt", "apt", CLIP_VAR]], on="MVT_ID_mvt", validate="one_to_one")
    v9 = pd.read_parquet(V9).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    r["shipped"] = v9.reindex(r.MVT_ID_mvt.to_numpy()).to_numpy()
    assert r.shipped.notna().all() and len(r) == 388
    qv = np.array([q.loc[(str(ap), CLIP_VAR), f"q{CLIP_LEVEL:g}"] for ap in r.apt.astype(str)])
    r["bound"] = r[CLIP_VAR].to_numpy(float) - qv
    r.loc[~np.isfinite(r.bound), "bound"] = np.inf
    rows = []
    for arm in arms:
        C = corrections(r, arm)
        for name in C.columns:
            c = C[name].to_numpy(float)
            served = np.clip(r.shipped.to_numpy(float) + c, MIN_TAXI, None)
            d = served - r.shipped.to_numpy(float)
            rows.append({"head": arm, "arm": name, "n_moved": int((c != 0).sum()),
                         "shift_rms_388": float(np.sqrt(np.mean(c ** 2))),
                         "shift_rms_388_served": float(np.sqrt(np.mean(d ** 2))),
                         "max_abs": float(np.abs(c).max()), "mean": float(c.mean()),
                         "under_budget": bool(np.sqrt(np.mean(c ** 2)) < 500.0)})
    return pd.DataFrame(rows)


HEADS = ("chain", "rot")


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "dep_2025.parquet")
    q = margins(a).set_index(["apt", "variant"])
    f = pd.read_parquet(EXP / "work" / "fold_arms.parquet")
    n25 = pd.read_parquet(EXP / "work" / "noac_2025.parquet")
    f = f.merge(n25[["MVT_ID_mvt", "apt", CLIP_VAR]], on="MVT_ID_mvt", validate="one_to_one")
    ship = load_shipped().rename(columns={"y_pred": "shipped", "y_true": "y_ship"})
    m = f.merge(ship, on=["MVT_ID_mvt", "fold"], how="left", validate="one_to_one")
    assert m.shipped.notna().all() and np.allclose(m.y_true, m.y_ship)
    qv = np.array([q.loc[(str(ap), CLIP_VAR), f"q{CLIP_LEVEL:g}"] for ap in m.apt.astype(str)])
    m["bound"] = m[CLIP_VAR].to_numpy(float) - qv
    m.loc[~np.isfinite(m.bound), "bound"] = np.inf

    heads = [h for h in HEADS if f"y_{h}" in m.columns]
    rows, lam_rows = [], []
    for head in heads:
        C = corrections(m, head)
        # out-of-fold shrinkage: lam for fold f is the RMSE-optimal scale on the
        # OTHER two folds, clipped to [0, 1]
        lam = {}
        for arm in C.columns:
            for fold in FOLDS:
                o = (m.fold != fold).to_numpy()
                e = (m.y_true - m.shipped).to_numpy(float)[o]
                c = C[arm].to_numpy(float)[o]
                lam[(arm, fold)] = float(np.clip(np.dot(c, e) / max(np.dot(c, c), 1e-9), 0.0, 1.0))
                lam_rows.append({"head": head, "arm": arm, "fold": fold, "lam": lam[(arm, fold)]})
        for arm in C.columns:
            m[f"{head}_raw_{arm}"] = m.shipped + C[arm]
            m[f"{head}_shr_{arm}"] = m.shipped + C[arm] * m.fold.map(
                lambda z, a_=arm: lam[(a_, z)]).to_numpy()
        for arm in C.columns:
            for kind in ("raw", "shr"):
                col = f"{head}_{kind}_{arm}"
                for fold in list(FOLDS) + ["POOLED"]:
                    d = m if fold == "POOLED" else m[m.fold == fold]
                    r = {"head": head, "arm": arm, "kind": kind, "fold": fold, "n": len(d),
                         "n_moved": int((d[col] != d.shipped).sum()),
                         "shift_rms_fold": float(np.sqrt(np.mean((d[col] - d.shipped) ** 2))),
                         "rmse_shipped": rmse(d.y_true, d.shipped),
                         "rmse_arm": rmse(d.y_true, d[col])}
                    r.update(day_boot_gain(d, "shipped", col))
                    rows.append(r)
    t = pd.DataFrame(rows)
    lam_tab = pd.DataFrame(lam_rows)
    t.to_csv(EXP / "analysis" / "repair_arms.csv", index=False)
    lam_tab.to_csv(EXP / "analysis" / "repair_lambda.csv", index=False)
    m.to_parquet(EXP / "work" / "repair_eval.parquet")

    sh = shift_388(q, arms=tuple(heads))
    sh.to_csv(EXP / "analysis" / "shift_388.csv", index=False)

    with pd.option_context("display.width", 240):
        print("out-of-fold shrinkage")
        print(lam_tab.round(3).to_string(index=False))
        print()
        print("the carriers on the shipped vector, 2025 noac fold rows")
        print(t.round(2).to_string(index=False))
        print()
        print("shift on the 388 ranking rows against v9 (budget 500 s RMS)")
        print(sh.round(2).to_string(index=False))

    # Holm over every DISTINCT carrier tried on these 491 rows.  `clip` does not
    # depend on the head, so it enters the family once, and a carrier whose
    # out-of-fold shrinkage came out at zero moves no row: its gain is exactly
    # 0 and its `p_worse` is 0 by construction, so it is not a test and does not
    # enter the family either.
    p = t[t.fold == "POOLED"].copy()
    # `p.head` is the DataFrame METHOD; the column has to be indexed.  Round 3
    # of the review found this dropping every clip row from the family.
    p = p[~((p["head"] != heads[0]) & (p.arm == "clip"))]
    p = p[p.n_moved > 0]
    for kind in ("raw", "shr"):
        s = p[p.kind == kind].sort_values("p_worse")
        k = len(s)
        # Holm: step down the ascending p-values, multiply by the remaining count,
        # then take a running MAXIMUM so the adjusted sequence is monotone.
        s["p_holm"] = np.maximum.accumulate(
            np.minimum(1.0, s.p_worse.to_numpy() * (k - np.arange(k))))
        print(f"\nHolm over the {k} distinct carriers ({kind}):")
        print(s[["head", "arm", "gain", "lo", "hi", "p_worse", "p_holm"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
