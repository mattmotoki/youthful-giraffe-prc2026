"""K3-tail: `T-tailv4` refit with K2's window block in its own feature frame.

One arm and one control, on the three rotation folds:

    PRC_THREADS=6 python code/member.py --arm base   --folds july_loo
    PRC_THREADS=6 python code/member.py --arm window --folds july_loo,december,january

`--arm base` is the environment control. `T-tailv4` is CPU-only and
bit-identical across machines (`exp-025-cloud` section 10.7), so the committed
OOF is the like-for-like baseline and this arm must land on it exactly: the
check is the vector, not the RMSE. `--arm window` is the treatment, and the
ONLY difference between the two runs is `window=` on `fit` and `predict`.

The member recipe is `exp-009-noac2/deliver.py`'s, unchanged: `TailModel` at
`noac_head="v4"`, `n_jobs=6`, phi on, no keep mask, no weights, no placeholder
regime. Nothing about the noac head or the band mixture moves -- the block
reaches the `nm` and `nonm` heads alone, which is exp-027's measured
`offload_noac` rule met by this member's own architecture.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]

THREADS = int(os.environ.get("PRC_THREADS", "6"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import prc.data as D  # noqa: E402
from prc.tail_model import TailModel  # noqa: E402
import wblock  # noqa: E402

FOLDS = ("july_loo", "december", "january")
NOAC_PIN = "v4"
#: `T-tailv4`'s committed out-of-fold file: the baseline this arm is measured
#: against, and the file `frames.OOF["T-tailv4"]` addresses.
PUBLISHED = ROOT / "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v3.parquet"


def metrics(y, p, fold, arm):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e2 = (y - p) ** 2
    tail = y > 3600
    r = lambda m: float(np.sqrt(e2[m].mean())) if m.sum() else float("nan")
    return dict(fold=fold, arm=arm, n=int(len(y)), rmse=r(np.ones(len(y), bool)),
                rmse_le3600=r(~tail), rmse_gt3600=r(tail),
                n_gt3600=int(tail.sum()))


def dump_importance(model, fold: str, arm: str) -> None:
    """Split-gain importance of the two bulk heads, so the block's share is a
    number rather than a claim.

    Read from the fitted booster AFTER the fit; it cannot move a prediction.
    (`july_loo`'s window arm was run before this call existed, which is why its
    file is absent; its out-of-fold vector is the same one, and the repeat run
    in section 3 reproduces it bit for bit.)
    """
    rows = []
    for head, mdl in (("nm", model.nm_head), ("nonm", model.nonm_head)):
        if mdl is None:
            continue
        b = mdl.booster_
        for name, gain, split in zip(b.feature_name(),
                                     b.feature_importance("gain"),
                                     b.feature_importance("split")):
            rows.append({"fold": fold, "arm": arm, "head": head, "feature": name,
                         "gain": float(gain), "split": int(split),
                         "window": name in set(model.window_cols)})
    d = pd.DataFrame(rows)
    d["gain_share"] = d["gain"] / d.groupby("head")["gain"].transform("sum")
    d.to_csv(EXP / "analysis" / f"importance-{arm}-{fold}.csv", index=False)
    sh = d[d["window"]].groupby("head")["gain_share"].sum().to_dict()
    print(f"[{fold}/{arm}] window share of split gain: "
          + "  ".join(f"{k} {v:.3f}" for k, v in sh.items()), flush=True)


def run(fold: str, arm: str) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    train, valid, y_valid = D.fold_frames(fold)
    wtr = wvl = None
    if arm == "window":
        wtr = wblock.build(train)
        wvl = wblock.build(valid)
        print(f"[{fold}] block {wtr.shape} train / {wvl.shape} valid "
              f"at {time.time()-t0:.0f}s", flush=True)
    m = TailModel().fit(train, n_jobs=THREADS, noac_head=NOAC_PIN, window=wtr)
    print(f"[{fold}] fitted at {time.time()-t0:.0f}s "
          f"({len(m.window_cols)} window columns)", flush=True)
    dump_importance(m, fold, arm)
    pred = m.predict(valid, window=wvl)
    y = y_valid.reindex(pred.index).to_numpy(float)
    oof = pd.DataFrame({"MVT_ID_mvt": pred["MVT_ID_mvt"].to_numpy(),
                        "fold": fold, "y_true": y,
                        "y_pred": pred["y_pred"].to_numpy()})
    row = metrics(y, oof["y_pred"], fold, arm)
    row["wall_s"] = round(time.time() - t0, 1)
    print(f"[{fold}/{arm}] rmse {row['rmse']:.4f}  le3600 {row['rmse_le3600']:.4f}  "
          f"gt3600 {row['rmse_gt3600']:.2f} on {row['n_gt3600']} rows  "
          f"{row['wall_s']:.0f}s", flush=True)
    return oof, row


def check_published(oof: pd.DataFrame, fold: str) -> dict:
    """The control's gate: the vector, not the RMSE."""
    pub = pd.read_parquet(PUBLISHED)
    pub = pub[pub["fold"] == fold]
    m = oof.merge(pub[["MVT_ID_mvt", "y_pred", "y_true"]], on="MVT_ID_mvt",
                  suffixes=("", "_pub"), how="inner")
    if len(m) != len(oof) or len(m) != len(pub):
        raise SystemExit(f"{fold}: coverage {len(m)} against {len(oof)} / {len(pub)}")
    d = float(np.abs(m["y_pred"].to_numpy() - m["y_pred_pub"].to_numpy()).max())
    dy = float(np.abs(m["y_true"].to_numpy() - m["y_true_pub"].to_numpy()).max())
    return {"fold": fold, "rows": int(len(m)), "max_abs_pred_diff": d,
            "max_abs_label_diff": dy,
            "published_rmse": round(float(np.sqrt(np.mean(
                (m["y_true_pub"] - m["y_pred_pub"]) ** 2))), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("base", "window"), required=True)
    ap.add_argument("--folds", default=",".join(FOLDS))
    a = ap.parse_args()
    folds = tuple(f.strip() for f in a.folds.split(",") if f.strip())
    rows, checks = [], []
    for fold in folds:
        oof, row = run(fold, a.arm)
        out = EXP / "oof" / f"k3-tail-{a.arm}-{fold}.parquet"
        oof.to_parquet(out, index=False)
        row["path"] = str(out.relative_to(ROOT))
        row["sha256"] = D.sha256(out)
        rows.append(row)
        if a.arm == "base":
            c = check_published(oof, fold)
            checks.append(c)
            print(f"[gate] {fold}: {c['rows']} rows, max abs difference against the "
                  f"committed T-tailv4 {c['max_abs_pred_diff']:.6g} "
                  f"(published {c['published_rmse']})", flush=True)
            # An environment fault, not a result: `T-tailv4` is deterministic and
            # bit-identical across machines, so anything but 0.0 here says this
            # box does not reproduce the member and no treatment number from it
            # would be like-for-like.
            if c["max_abs_pred_diff"] != 0.0 or c["max_abs_label_diff"] != 0.0:
                raise SystemExit(f"{fold}: the flag-off path does not reproduce the "
                                 f"committed T-tailv4 ({c['max_abs_pred_diff']:.6g})")
    rec = {"arm": a.arm, "folds": list(folds), "threads": THREADS,
           "lightgbm": __import__("lightgbm").__version__,
           "tail_model_version": __import__("prc.tail_model", fromlist=["VERSION"]).VERSION,
           "rows": rows, "identity_checks": checks}
    p = EXP / "analysis" / f"member-{a.arm}-{'_'.join(folds)}.json"
    p.write_text(json.dumps(rec, indent=2))
    print(f"\n-> {p.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
