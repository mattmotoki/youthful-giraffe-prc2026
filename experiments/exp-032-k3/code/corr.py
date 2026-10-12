"""Why the member gain does not reach the board: measure the correlation.

`experiments/exp-012-blend2/run_hub8725_folds.sh` states the rule this
experiment ran into, in the desk's own words at the time:

    Measured by MEMBER RMSE the 8,725-round arm is 3.57 s worse than the
    shipped 2,100. Measured by what it adds to the BLEND [...] it is 0.18 s
    BETTER on july_loo, because its errors move into the tail where the blend
    is weak and its correlation with the other members falls 0.9309 -> 0.9132.

L2-hub8725 was therefore made WORSE as a member on purpose, to buy
decorrelation. K3 is the mirror image: the window block makes `T-tailv4` much
better on its own folds. This script asks what it costs in correlation, which
is the currency the blend actually pays in.

Reported per fold, on the non-`noac` rows the stacker fits (the `noac` rows are
held at the anchor in every arm, so they cannot move an arm apart):

  rho_K2     correlation of the member's residual with `K2-window`'s residual
  rho_rest   mean correlation with the residuals of the other six members
  rmse       the member's own clean fold RMSE, for the contrast

-> analysis/member-corr.csv, and a table on stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
sys.path.insert(0, str(ROOT))
import frames as F  # noqa: E402
import stack5 as S  # noqa: E402
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
K2, K3, K3G = "K2-window", "K3-tail", "K3-tail-gated"
F.OOF[K2] = "experiments/exp-027-step/oof/k2-window-{fold}.parquet"
F.OOF[K3] = "experiments/exp-032-k3/oof/k3-tail-window-{fold}.parquet"
F.OOF[K3G] = "experiments/exp-032-k3/oof/k3-tail-gated-{fold}.parquet"

#: The three readings of `T-tailv4`'s slot that the rotation scored.
VARIANTS = ("T-tailv4", K3, K3G)
OTHERS = ("C2-cat", "L2-hub8725", "D1-winter", "V3-allin", "N1-nmtail")


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation on the rows where both residuals are finite."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    rows = []
    for fold in FOLDS:
        m = F.fold_frame(fold, S.B5 + (K2, K3, K3G))
        nn = (m["regime"].to_numpy() != "noac")
        y = m["y_true"].to_numpy("float64")
        res = {n: (y - m[n].to_numpy("float64"))[nn] for n in S.B5 + (K2, K3, K3G)}
        ids = m["MVT_ID_mvt"]
        for name in VARIANTS:
            # the other six members of the arm this variant sits in: K2 holds
            # F1-fresh's slot in every scored arm, so F1-fresh is not among them
            rest = OTHERS + (K2,)
            # `N1-nmtail` is defined only inside `nm_late` and is NaN elsewhere,
            # so every pair is taken on the rows where BOTH members are finite.
            cs = [_rho(res[name], res[o]) for o in rest]
            rmse = D.rmse_clean(m["y_true"], m[name], ids, fold)
            rows.append({"fold": fold, "member": name, "n": int(nn.sum()),
                         "rmse_clean": round(float(rmse), 4),
                         "rho_K2": round(_rho(res[name], res[K2]), 4),
                         "rho_rest": round(float(np.nanmean(cs)), 4)})
            print(f"{fold:9s} {name:14s} rmse {rmse:9.4f}   "
                  f"rho_K2 {rows[-1]['rho_K2']:.4f}   rho_rest {rows[-1]['rho_rest']:.4f}",
                  flush=True)
    d = pd.DataFrame(rows)
    out = ROOT / "experiments/exp-032-k3/analysis/member-corr.csv"
    d.to_csv(out, index=False)
    print("\n--- change against the plain T-tailv4 slot, by fold ---")
    for fold in FOLDS:
        b = d[(d["fold"] == fold) & (d["member"] == "T-tailv4")].iloc[0]
        for name in (K3, K3G):
            r = d[(d["fold"] == fold) & (d["member"] == name)].iloc[0]
            print(f"{fold:9s} {name:14s} rmse {r.rmse_clean - b.rmse_clean:+8.4f}   "
                  f"rho_K2 {r.rho_K2 - b.rho_K2:+.4f}   rho_rest {r.rho_rest - b.rho_rest:+.4f}")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
