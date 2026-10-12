"""Why the FAIL: is it the basis, the transfer, or the sample size?

Three questions a negative has to answer before it is written down.

1. **Can the basis express anything on these rows at all?**  Arm `O` fits the
   pooled objective ON the nine reading months at the same registered capacity
   and reads it there in sample.  It is an oracle and can never be a candidate;
   it is the ceiling of the representation on this population.
2. **Does the sample resolve what we are looking for?**  The half-width of the
   nine-month interval against the effect the board would need.
3. **Is the frozen base itself already carrying the schedule/stand signal?**
   Regress the residual on the same basis with no penalty and read the R^2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "code"))
from prc.tail_model import MIN_TAXI  # noqa: E402
from arms import (FIT_MONTHS, READ_MONTHS, CAPACITY, basis, solve, bisect,
                  move_388, COLS)  # noqa: E402
from read import endpoint, boot  # noqa: E402
from anatomy import SIG_NOAC, N388, board, GAP  # noqa: E402

SRC = ROOT / "experiments" / "exp-033-noac" / "work"


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "base_lomo.parquet")
    b = pd.read_parquet(SRC / "noac_2026_d.parquet")
    cfg = json.loads((EXP / "analysis" / "arms.json").read_text())
    mu, sd = cfg["mu"], cfg["sd"]
    X388, v7 = basis(b, mu, sd), b.v7.to_numpy(float)

    rd = a[a.month.isin(READ_MONTHS)].reset_index(drop=True)
    X9 = basis(rd, mu, sd)
    r9 = (rd.y_true - rd.base).to_numpy(float)
    ft = a[a.month.isin(FIT_MONTHS)].reset_index(drop=True)
    Xf, rf = basis(ft, mu, sd), (ft.y_true - ft.base).to_numpy(float)

    # 1. the oracle: fitted on the nine months, read on the nine months
    G9 = X9.T @ X9 / len(r9)
    c9 = X9.T @ r9 / len(r9)
    lam, w_o, st = bisect(lambda l: solve(G9, c9, l), X388, v7)
    d_o = np.clip(rd.base.to_numpy(float) + X9 @ w_o, MIN_TAXI, None) - rd.base.to_numpy(float)
    e_o = endpoint(r9, d_o)

    # the same oracle at its own unrestricted amplitude (no capacity at all)
    w_u = np.linalg.lstsq(X9, r9, rcond=None)[0]
    d_u = X9 @ w_u
    e_u = endpoint(r9, d_u)

    # 2. power
    days = rd.day.to_numpy().astype("datetime64[D]").astype("int64")
    ds = {"O": d_o}
    bs = boot(r9, ds, days)
    half = (bs["O"]["hi"] - bs["O"]["lo"]) / 2.0

    # what the board's gap needs, expressed as a nine-month RMSE gain at the
    # same capacity: rho_c = 0.309 (500 s budget) or 0.259 (unbounded amplitude)
    def gain_rmse_at(rho, S, sigma):
        g = 2.0 * rho * sigma * S - S ** 2
        return float(sigma - np.sqrt(max(sigma ** 2 - g, 0.0)))
    sigma9 = float(np.sqrt(np.mean(r9 ** 2)))
    need_250 = gain_rmse_at(0.2592, 250.0, sigma9)
    need_500 = gain_rmse_at(0.3085, 500.0, sigma9)

    # 3. how much of the residual the basis explains at all
    r2_9 = 1.0 - float(np.sum((r9 - X9 @ w_u) ** 2) / np.sum((r9 - r9.mean()) ** 2))
    w_uf = np.linalg.lstsq(Xf, rf, rcond=None)[0]
    r2_f = 1.0 - float(np.sum((rf - Xf @ w_uf) ** 2) / np.sum((rf - rf.mean()) ** 2))
    # and the transfer: the fit-month unrestricted fit applied to the nine
    e_t = endpoint(r9, X9 @ w_uf)

    rec = {
        "oracle_capacity_250": {**e_o, "lo": bs["O"]["lo"], "hi": bs["O"]["hi"],
                                "lambda": lam, "bisect": st,
                                "move_388": move_388(w_o, X388, v7),
                                "w": w_o.tolist()},
        "oracle_unrestricted": {**e_u, "w": w_u.tolist()},
        "half_width_nine_month_s": half,
        "sigma_nine": sigma9,
        "needed_gain_rmse_nine_at_250s": need_250,
        "needed_gain_rmse_nine_at_500s": need_500,
        "r2_basis_on_nine": r2_9, "r2_basis_on_fit": r2_f,
        "fit_month_unrestricted_on_nine": e_t,
        "cols": list(COLS),
    }
    (EXP / "analysis" / "power.json").write_text(json.dumps(rec, indent=2, default=float))

    print("=== 1. the ceiling of the representation on this population ===")
    print(f"oracle (pooled objective fitted ON the nine months, capacity 250 s on the 388):")
    print(f"  gain {e_o['gain_rmse']:+.2f} s [{bs['O']['lo']:+.2f}, {bs['O']['hi']:+.2f}], "
          f"rho {e_o['rho']:+.4f}, scale {e_o['scale']:.3f}")
    print(f"oracle at its own unrestricted amplitude (no capacity):")
    print(f"  gain {e_u['gain_rmse']:+.2f} s, rms_d {e_u['rms_d']:.1f} s, rho {e_u['rho']:+.4f}")
    print(f"  -> at rho {e_u['rho']:.4f} the whole 388 at its optimal amplitude is worth "
          f"{board(-N388 * e_u['rho'] ** 2 * SIG_NOAC ** 2):+.3f} s of board "
          f"({abs(board(-N388 * e_u['rho'] ** 2 * SIG_NOAC ** 2)) / GAP * 100:.1f} % of the gap)")
    print("\n=== 2. does the sample resolve what the board needs? ===")
    print(f"nine-month interval half-width      {half:8.2f} s of noac RMSE")
    print(f"gain the board's gap needs at 250 s {need_250:8.2f} s")
    print(f"gain the board's gap needs at 500 s {need_500:8.2f} s")
    print("\n=== 3. what the basis explains of the residual ===")
    print(f"R^2 of the 8-column basis, unpenalised: nine months {r2_9:.5f}, "
          f"fit months {r2_f:.5f}")
    print(f"fit-month unrestricted fit read on the nine months: "
          f"gain {e_t['gain_rmse']:+.2f} s, rms_d {e_t['rms_d']:.1f} s, rho {e_t['rho']:+.4f}")


if __name__ == "__main__":
    main()
