"""The nine-month reading: the gate exp-033 §9.3 asks for.

Everything scored here is on the nine 2025 months that have never been a
validation month in this record and took no part in choosing anything in
SPEC.md.  The endpoint is the contract's (`experiments/exp-038-packet/code/
align.py`): `A = mean(r d)`, `B = mean(d d)`, full-amplitude MSE gain `2A - B`,
quoted in RMSE seconds.  Intervals are a 2,000-draw block bootstrap over
calendar days, the same draws shared across arms so `W - P` is paired.
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
from arms import FIT_MONTHS, READ_MONTHS  # noqa: E402

DRAWS = 2000
SEED = 7


def endpoint(r: np.ndarray, d: np.ndarray) -> dict:
    A, B = float(np.mean(r * d)), float(np.mean(d * d))
    mse0 = float(np.mean(r * r))
    g = 2.0 * A - B
    return {"n": int(len(r)), "A": A, "B": B, "gain_mse": g,
            "rmse_base": float(np.sqrt(mse0)),
            "rmse_arm": float(np.sqrt(max(mse0 - g, 0.0))),
            "gain_rmse": float(np.sqrt(mse0) - np.sqrt(max(mse0 - g, 0.0))),
            "scale": (A / B) if B > 0 else None,
            "gain_mse_fitted": (A * A / B) if B > 0 else 0.0,
            "rms_d": float(np.sqrt(B)),
            "rho": float(np.corrcoef(r, d)[0, 1]) if B > 0 else None}


def _draw_index(days: np.ndarray, rng) -> list:
    """Bootstrap resample of calendar days; returns row-index arrays."""
    uq, inv = np.unique(days, return_inverse=True)
    pos = [np.flatnonzero(inv == i) for i in range(len(uq))]
    k = len(uq)
    return [np.concatenate([pos[j] for j in rng.integers(0, k, k)])
            for _ in range(DRAWS)]


def boot(r: np.ndarray, ds: dict, days: np.ndarray, seed: int = SEED) -> dict:
    """Day-clustered intervals on each arm's RMSE gain and on every pairwise gap."""
    rng = np.random.default_rng(seed)
    idx = _draw_index(days, rng)
    keys = list(ds)
    g = {k: np.empty(DRAWS) for k in keys}
    for t, ii in enumerate(idx):
        rr = r[ii]
        mse0 = float(np.mean(rr * rr))
        b0 = np.sqrt(mse0)
        for k in keys:
            dd = ds[k][ii]
            gm = 2.0 * float(np.mean(rr * dd)) - float(np.mean(dd * dd))
            g[k][t] = b0 - np.sqrt(max(mse0 - gm, 0.0))
    out = {k: {"lo": float(np.percentile(g[k], 2.5)),
               "hi": float(np.percentile(g[k], 97.5)),
               "p_pos": float((g[k] > 0).mean())} for k in keys}
    for i, k in enumerate(keys):
        for k2 in keys[i + 1:]:
            dif = g[k] - g[k2]
            out[f"{k}-{k2}"] = {"mean": float(dif.mean()),
                                "lo": float(np.percentile(dif, 2.5)),
                                "hi": float(np.percentile(dif, 97.5)),
                                "p_pos": float((dif > 0).mean())}
    return out


def table(frame: pd.DataFrame, arms) -> pd.DataFrame:
    r = (frame.y_true - frame.base).to_numpy(float)
    rows = []
    for k in arms:
        e = endpoint(r, frame[f"d_{k}"].to_numpy(float))
        rows.append({"arm": k, **e})
    return pd.DataFrame(rows)


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "arms_2025.parquet")
    arms = [c[2:] for c in a.columns if c.startswith("d_")]
    rd = a[a.month.isin(READ_MONTHS)].reset_index(drop=True)
    ft = a[a.month.isin(FIT_MONTHS)].reset_index(drop=True)
    r = (rd.y_true - rd.base).to_numpy(float)
    ds = {k: rd[f"d_{k}"].to_numpy(float) for k in arms}
    days = rd.day.to_numpy().astype("datetime64[D]").astype("int64")

    res = {"arms": arms, "read_months": list(READ_MONTHS), "draws": DRAWS}

    # --- the gate: pooled over the nine reading months ----------------------
    pooled = table(rd, arms)
    bs = boot(r, ds, days)
    pooled["lo"] = [bs[k]["lo"] for k in pooled.arm]
    pooled["hi"] = [bs[k]["hi"] for k in pooled.arm]
    pooled["p_pos"] = [bs[k]["p_pos"] for k in pooled.arm]
    res["pooled_nine"] = pooled.to_dict("records")
    res["pooled_nine_pairs"] = {k: v for k, v in bs.items() if "-" in k}
    pooled.to_csv(EXP / "analysis" / "pooled_nine.csv", index=False)

    # --- the three fit months, for contrast (in-sample, not a gate) ---------
    res["fit_months_in_sample"] = table(ft, arms).to_dict("records")

    # --- every reading month separately -------------------------------------
    per = []
    for m in READ_MONTHS:
        g = rd[rd.month == m]
        rr = (g.y_true - g.base).to_numpy(float)
        dm = {k: g[f"d_{k}"].to_numpy(float) for k in arms}
        dy = g.day.to_numpy().astype("datetime64[D]").astype("int64")
        b = boot(rr, dm, dy, seed=SEED + 1)
        for k in arms:
            e = endpoint(rr, dm[k])
            per.append({"month": m, "arm": k, **e, **{x: b[k][x] for x in ("lo", "hi", "p_pos")}})
    per = pd.DataFrame(per)
    per.to_csv(EXP / "analysis" / "per_month_nine.csv", index=False)
    res["per_month"] = per.to_dict("records")

    # --- the month challenges: leave one reading month out ------------------
    ch = []
    for m in READ_MONTHS:
        g = rd[rd.month != m]
        rr = (g.y_true - g.base).to_numpy(float)
        for k in arms:
            e = endpoint(rr, g[f"d_{k}"].to_numpy(float))
            ch.append({"left_out": m, "arm": k, "n": e["n"],
                       "gain_rmse": e["gain_rmse"], "gain_mse": e["gain_mse"]})
    ch = pd.DataFrame(ch)
    ch.to_csv(EXP / "analysis" / "month_challenge.csv", index=False)
    res["month_challenge"] = ch.to_dict("records")

    # --- raw-tail accounting: by mvt_sched band and by label band -----------
    acc = []
    for k in arms:
        for name, mask in (("y<=3600", rd.y_true <= 3600.0), ("y>3600", rd.y_true > 3600.0)):
            g = rd[mask.to_numpy()]
            e = endpoint((g.y_true - g.base).to_numpy(float), g[f"d_{k}"].to_numpy(float))
            acc.append({"arm": k, "split": "label", "cell": name, **e})
        for bnd, g in rd.groupby("band", observed=True):
            e = endpoint((g.y_true - g.base).to_numpy(float), g[f"d_{k}"].to_numpy(float))
            acc.append({"arm": k, "split": "mvt_sched band", "cell": bnd, **e})
    acc = pd.DataFrame(acc)
    acc.to_csv(EXP / "analysis" / "tail_accounting.csv", index=False)
    res["tail_accounting"] = acc.to_dict("records")

    (EXP / "analysis" / "reading.json").write_text(json.dumps(res, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("=== the gate: pooled over the nine reading months (1,016 rows) ===")
    print(pooled[["arm", "n", "rmse_base", "rmse_arm", "gain_rmse", "lo", "hi",
                  "p_pos", "rms_d", "rho", "scale", "gain_mse"]].round(4).to_string(index=False))
    print("\npaired gaps (same draws):")
    for k, v in res["pooled_nine_pairs"].items():
        print(f"  {k}: {v['mean']:+.2f} s [{v['lo']:+.2f}, {v['hi']:+.2f}] p(>0)={v['p_pos']:.3f}")
    print("\n=== the three fit months, in sample (not a gate) ===")
    print(pd.DataFrame(res["fit_months_in_sample"])[
        ["arm", "n", "rmse_base", "rmse_arm", "gain_rmse", "rms_d", "rho"]].round(3).to_string(index=False))
    print("\n=== per reading month ===")
    print(per.pivot(index="month", columns="arm", values="gain_rmse").round(2).to_string())
    print("\n=== month challenge (one reading month left out) ===")
    print(ch.pivot(index="left_out", columns="arm", values="gain_rmse").round(2).to_string())
    print("\n=== raw-tail accounting ===")
    print(acc[["arm", "split", "cell", "n", "rmse_base", "gain_rmse", "rms_d"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
