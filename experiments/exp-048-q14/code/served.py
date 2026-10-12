"""Two things round 1 of the FALLBACK review made the experiment measure.

**Blocking 1.** How far is this experiment's base from the vector actually served
on the 388?  The reviewer's direct test: fit the same noac head on all twelve
2025 months and compare its prediction on the 388 with `v7`, the served bytes.
The answer is the transfer caveat this reading carries, and it belongs in the
record with a number rather than an adjective.

**Non-blocking 3.** How much of each arm's update is a constant level and how
much is shape?  With `d = m + s` (`m` the mean, `s` zero-mean), the contract's
gain splits exactly:

    2A - B = [2 m mean(r) - m^2] + [2 mean(r s) - mean(s^2)]
             ------ level ------   ------- shape --------

so the level and shape contributions can be read off without refitting anything.
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
from prc.tail_model import NOAC_CATS_V2, MIN_TAXI  # noqa: E402
from base import fit_path, apply_path, encode, SRC  # noqa: E402
from arms import FIT_MONTHS, READ_MONTHS  # noqa: E402


def split_gain(r: np.ndarray, d: np.ndarray) -> dict:
    m = float(np.mean(d))
    s = d - m
    lev = 2.0 * m * float(np.mean(r)) - m ** 2
    sha = 2.0 * float(np.mean(r * s)) - float(np.mean(s ** 2))
    B = float(np.mean(d * d))
    return {"mean_d": m, "rms_d": float(np.sqrt(B)), "shape_rms": float(np.sqrt(np.mean(s ** 2))),
            "energy_share_in_level": (m ** 2 / B) if B > 0 else 0.0,
            "gain_mse_level": lev, "gain_mse_shape": sha, "gain_mse_total": lev + sha,
            "level_share_of_gain": (lev / (lev + sha)) if (lev + sha) != 0 else float("nan"),
            "mean_r": float(np.mean(r))}


def main() -> None:
    a = pd.read_parquet(SRC / "noac_2025_lab.parquet")
    b = pd.read_parquet(SRC / "noac_2026_d.parquet")
    cats = {c: pd.Index(sorted(set(a[c].astype(str)) | set(b[c].astype(str))))
            for c in NOAC_CATS_V2}

    # --- blocking 1: this head on the 388, against the served bytes ---------
    head = fit_path(a, cats)
    p388 = apply_path(head, b, cats)
    v7 = b.v7.to_numpy(float)
    dif = p388 - v7
    served = {"n": 388, "rms_diff": float(np.sqrt(np.mean(dif ** 2))),
              "median_abs_diff": float(np.median(np.abs(dif))),
              "rows_within_1s": int((np.abs(dif) <= 1.0).sum()),
              "rows_within_100s": int((np.abs(dif) <= 100.0).sum()),
              "corr": float(np.corrcoef(p388, v7)[0, 1]),
              "by_band": {}}
    for bn, g in b.assign(dif=dif).groupby("band", observed=True):
        served["by_band"][str(bn)] = {"n": len(g),
                                      "rms_diff": float(np.sqrt(np.mean(g.dif ** 2))),
                                      "median_v7": float(g.v7.median())}

    # the band-free variant's LOMO RMSE over all 1,507 rows, for the §2 comparison
    bc = json.loads((EXP / "analysis" / "basecheck.json").read_text())
    served["base_rmse_all_1507"] = {k: v["rmse_base_all"] for k, v in bc.items()}

    # --- non-blocking 3: level against shape --------------------------------
    arms = pd.read_parquet(EXP / "work" / "arms_2025.parquet")
    s388 = pd.read_parquet(EXP / "work" / "arms_388.parquet")
    ft = arms[arms.month.isin(FIT_MONTHS)]
    rd = arms[arms.month.isin(READ_MONTHS)]
    rows = []
    for k in ("P", "W"):
        for tag, g in (("fit months", ft), ("nine reading months", rd)):
            r = (g.y_true - g.base).to_numpy(float)
            rows.append({"arm": k, "scope": tag, **split_gain(r, g[f"d_{k}"].to_numpy(float))})
        d = s388[f"d_{k}"].to_numpy(float)
        m = float(np.mean(d))
        rows.append({"arm": k, "scope": "the 388", "mean_d": m,
                     "rms_d": float(np.sqrt(np.mean(d ** 2))),
                     "shape_rms": float(np.sqrt(np.mean((d - m) ** 2))),
                     "energy_share_in_level": m ** 2 / float(np.mean(d ** 2))})
    lev = pd.DataFrame(rows)
    lev.to_csv(EXP / "analysis" / "level_shape.csv", index=False)

    # round 2 required this: the out-of-sample level and shape terms with their
    # own day-clustered intervals, held to the same standard as the gate itself
    r9 = (rd.y_true - rd.base).to_numpy(float)
    d9 = {k: rd[f"d_{k}"].to_numpy(float) for k in ("P", "W")}
    days = rd.day.to_numpy().astype("datetime64[D]").astype("int64")
    uq, inv = np.unique(days, return_inverse=True)
    posl = [np.flatnonzero(inv == i) for i in range(len(uq))]
    rng = np.random.default_rng(23)
    draws = [np.concatenate([posl[j] for j in rng.integers(0, len(uq), len(uq))])
             for _ in range(2000)]
    bl = {k: np.empty(2000) for k in ("level_P", "level_W", "shape_P", "shape_W", "mean_r")}
    for t, ii in enumerate(draws):
        rr = r9[ii]
        bl["mean_r"][t] = rr.mean()
        for k in ("P", "W"):
            g = split_gain(rr, d9[k][ii])
            bl[f"level_{k}"][t] = g["gain_mse_level"]
            bl[f"shape_{k}"][t] = g["gain_mse_shape"]
    def q(v, tag):
        return {"term": tag, "point": float(np.mean(v)),
                "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
    boots = [q(bl["mean_r"], "mean residual, nine months (s)"),
             q(bl["level_P"], "level term P (s^2)"), q(bl["level_W"], "level term W (s^2)"),
             q(bl["shape_P"], "shape term P (s^2)"), q(bl["shape_W"], "shape term W (s^2)")]
    dlp = bl["level_P"] - bl["level_W"]
    dsp = bl["shape_P"] - bl["shape_W"]
    boots.append({**q(dlp, "level P - level W (s^2)"), "p_neg": float((dlp < 0).mean())})
    boots.append({**q(dsp, "shape P - shape W (s^2)"), "p_pos": float((dsp > 0).mean())})
    bt = pd.DataFrame(boots)
    bt.to_csv(EXP / "analysis" / "level_shape_boot.csv", index=False)

    per = arms.assign(r=arms.y_true - arms.base).groupby("month").agg(
        n=("r", "size"), mean_r=("r", "mean"), rmse=("r", lambda x: float(np.sqrt((x ** 2).mean()))))
    per["set"] = np.where(per.index.isin(FIT_MONTHS), "fit", "read")
    per.to_csv(EXP / "analysis" / "residual_level_per_month.csv")

    out = {"served_vs_head": served, "level_shape": lev.to_dict("records"),
           "residual_mean_fit": float((arms[arms.month.isin(FIT_MONTHS)].y_true
                                       - arms[arms.month.isin(FIT_MONTHS)].base).mean()),
           "residual_mean_read": float((arms[arms.month.isin(READ_MONTHS)].y_true
                                        - arms[arms.month.isin(READ_MONTHS)].base).mean()),
           "per_month": per.reset_index().to_dict("records"),
           "level_shape_boot": bt.to_dict("records")}
    (EXP / "analysis" / "served.json").write_text(json.dumps(out, indent=2, default=float))

    pd.set_option("display.width", 220)
    print("=== blocking 1: this experiment's head against the SERVED bytes on the 388 ===")
    print(f"RMS difference {served['rms_diff']:,.0f} s | median |diff| "
          f"{served['median_abs_diff']:,.0f} s | rows within 1 s "
          f"{served['rows_within_1s']} of 388 | within 100 s {served['rows_within_100s']} "
          f"| corr {served['corr']:.4f}")
    for bn, v in sorted(served["by_band"].items(), key=lambda x: -x[1]["n"]):
        print(f"  {bn:7s} n {v['n']:3d}  RMS diff {v['rms_diff']:9,.0f} s  "
              f"median v7 {v['median_v7']:9,.0f} s")
    print("base LOMO RMSE over all 1,507 rows:",
          {k: round(v, 1) for k, v in served["base_rmse_all_1507"].items()})
    print("\n=== non-blocking 3: level against shape ===")
    print(lev.round(3).to_string(index=False))
    print("\n=== the level and shape terms, day-clustered (round 2) ===")
    print(bt.round(3).to_string(index=False))
    print("\n=== the residual's level, per month ===")
    print(per.round(1).to_string())


if __name__ == "__main__":
    main()
