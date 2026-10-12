"""What a member change is worth at the head, without paying for a rotation.

A rotation costs three fold sessions plus a ranking session before it says
anything. This screen costs seconds and says most of it: project the member's
per-row change onto the head's residual at the best coefficient, PER REGIME,
because the harness's anchor is a per-regime hybrid NNLS (`exp-026 rule.py`) and
a change concentrated in one regime is not diluted across the fold.

**It ranks; it does not predict.** Six calibration points -- two changes whose
head gain is already known, on all three folds -- put the realised gain at
between 0.58x and 2.98x the bound:

    change                            july_loo    december     january
    F1-fresh's member -> K2's         1.23x        0.67x        0.58x
    arm 1, K2 -> three seeds          1.28x        2.98x        1.70x

The two july_loo points agreeing at 1.2-1.3x was a coincidence of that fold, and
an earlier version of this file asserted 1.25x as a constant on the strength of
it (RESULTS section 14b records the correction). Two effects pull against each
other: the bound is an upper bound for a FIXED head, while the rotation refits
the stacker, the anchor and the block and can beat it. What survives is the
ranking -- the bound separates the arms by an order of magnitude -- so this
decides whether the next three sessions are worth buying, and nothing more.

    python screen.py --arms seeds phi --fold july_loo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
for p in ("experiments/exp-030-window2/code", "experiments/exp-028-window/code",
          "experiments/exp-026-rebuild/code", ""):
    sys.path.insert(0, str(ROOT / p))
import frames as F  # noqa: E402
import stack5 as S  # noqa: E402
import prc.data as D  # noqa: E402

#: The head the screen prices against: K2's own arm, the file the board holds.
HEAD = ROOT / "experiments/exp-027-step/oof/exp-027-k2-k2-hold-{fold}.parquet"
K2 = ROOT / "experiments/exp-027-step/oof/k2-window-{fold}.parquet"
#: arm -> the member file whose change is screened.  `<f1>` is the calibration.
MEMBER = {"seeds": EXP / "oof" / "seeds-{fold}.parquet",
          "phi": EXP / "oof" / "k2p-{fold}.parquet",
          "mod": EXP / "oof" / "k2m-{fold}.parquet",
          "wide": EXP / "oof" / "k2w-{fold}.parquet"}
#: (member change, head column, that head's clean RMSE, the realised head gain).
CALIB = {"july_loo": ("y_pred_v6", 284.7219, 0.3281),
         "december": ("y_pred_v6", 205.6534, 0.3544),
         "january": ("y_pred_v6", 218.8166, 0.3934)}
#: arm 1's own realised head gain (`analysis/price-vs-k2.json`, the `k2-hold`
#: arm before the mask), the second calibration point on every fold.
REALISED = {"seeds": {"july_loo": 0.0121, "december": 0.0421, "january": 0.0269}}
#: kept at 1.0: the six calibration points above refuse a constant.
RATIO = 1.0


def bound(delta, resid, rmse, reg, ok, per_regime=True) -> float:
    """The MSE the best coefficient could take out of the head, in RMSE seconds."""
    tot = 0.0
    groups = np.unique(reg[ok]) if per_regime else np.array(["ALL"])
    for g in groups:
        m = (ok & (reg == g)) if per_regime else ok
        x, r = delta[m], resid[m]
        k = np.isfinite(x) & np.isfinite(r)
        x, r = x[k], r[k]
        if x.size == 0 or np.dot(x, x) == 0:
            continue
        tot += float(np.dot(x, r) ** 2 / np.dot(x, x))
    return tot / ok.sum() / (2 * rmse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--fold", default="july_loo")
    a = ap.parse_args()
    f = a.fold

    fr = F.fold_frame(f, S.B5).set_index("MVT_ID_mvt")
    head = pd.read_parquet(str(HEAD).format(fold=f)).set_index("MVT_ID_mvt")
    km = pd.read_parquet(str(K2).format(fold=f)).set_index("MVT_ID_mvt")["y_pred"]
    y = head["y_true"]
    ok = ~np.asarray(pd.Index(head.index).isin(D.excluded_ids(f)))
    reg = fr["regime"].astype(str).reindex(head.index).to_numpy()
    rmse = D.rmse_clean(y.to_numpy(), head["y_pred_nomask"].to_numpy(),
                        head.index.to_numpy(), f)
    resid = (head["y_pred_nomask"] - y).to_numpy()

    out = {"fold": f, "head_rmse_clean": round(rmse, 4), "ratio": RATIO, "arms": {}}
    print(f"{'case':14s} {'global':>9s} {'per-regime':>11s} {'bound':>9s} {'realised':>9s}")
    if f in CALIB:
        col, r6, real = CALIB[f]
        fm = pd.read_parquet(str(F.OOF["F1-fresh"]).format(fold=f)).set_index("MVT_ID_mvt")["y_pred"]
        d = (km - fm.reindex(km.index)).reindex(head.index).to_numpy()
        r = (head[col] - y).to_numpy()
        g, pr = bound(d, r, r6, reg, ok, False), bound(d, r, r6, reg, ok)
        out["calibration"] = {"global": round(g, 4), "per_regime": round(pr, 4),
                              "realised": real, "ratio": round(real / pr, 3)}
        print(f"{'CALIB F1->K2':14s} {g:+9.4f} {pr:+11.4f} {pr * RATIO:+9.4f} {real:+9.4f}"
              f"   {real / pr:.2f}x")
    for arm in a.arms:
        p = Path(str(MEMBER[arm]).format(fold=f))
        if not p.exists():
            print(f"{arm:14s} pending ({p.relative_to(ROOT)})")
            continue
        m = pd.read_parquet(p).set_index("MVT_ID_mvt")["y_pred"]
        d = (m - km.reindex(m.index)).reindex(head.index).to_numpy()
        g, pr = bound(d, resid, rmse, reg, ok, False), bound(d, resid, rmse, reg, ok)
        mg = D.rmse_clean(y.reindex(m.index).to_numpy(), m.to_numpy(),
                          m.index.to_numpy(), f)
        real = REALISED.get(arm, {}).get(f)
        out["arms"][arm] = {"member_rmse": round(mg, 4),
                            "realised": real,
                            "ratio": round(real / pr, 3) if real and pr else None,
                            "member_gain": round(D.rmse_clean(
                                y.reindex(km.index).to_numpy(), km.to_numpy(),
                                km.index.to_numpy(), f) - mg, 4),
                            "global": round(g, 4), "per_regime": round(pr, 4),
                            "implied_head_gain": round(pr * RATIO, 4)}
        print(f"{arm:14s} {g:+9.4f} {pr:+11.4f} {pr * RATIO:+9.4f} "
              f"{('%+.4f' % real) if real else '        -':>9s}"
              f"{('   %.2fx' % (real / pr)) if real and pr else ''}")
    q = EXP / "analysis" / f"screen-{f}.json"
    q.write_text(json.dumps(out, indent=1))
    print("wrote", q.relative_to(ROOT))


if __name__ == "__main__":
    main()
