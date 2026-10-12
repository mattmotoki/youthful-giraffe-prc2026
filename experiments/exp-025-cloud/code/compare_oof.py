"""exp-025-cloud: compare a Kaggle fold OOF against the committed workstation one.

  .venv/bin/python experiments/exp-025-cloud/code/compare_oof.py \
      --cloud <kaggle oof>.parquet \
      --local experiments/exp-018-fresh/oof/july_loo.parquet --fold july_loo

A bitwise match is not expected and is not the test: LightGBM does not
guarantee identical histograms across thread counts (Kaggle 4 vCPU against the
workstation's 6 threads) and CatBoost GPU is not bitwise reproducible across
GPU models (P100 against TITAN RTX). The test is the fold RMSE against the
committed one, read against the repo's single-seed measurability threshold of
about 3 s, plus the per-row spread and the segment table COMMON.md requires.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from prc import data  # noqa: E402


def segments(df: pd.DataFrame, col: str) -> dict:
    y = df["y_true"].to_numpy(dtype="float64")
    p = df[col].to_numpy(dtype="float64")
    lo = y <= 3600
    out = {"rmse": float(np.sqrt(np.mean((y - p) ** 2))),
           "rmse_le3600": float(np.sqrt(np.mean((y[lo] - p[lo]) ** 2))),
           "rmse_gt3600": float(np.sqrt(np.mean((y[~lo] - p[~lo]) ** 2))),
           "n": int(len(y)), "n_gt3600": int((~lo).sum())}
    if "no_nm" in df.columns:
        m = df["no_nm"].to_numpy(dtype=bool)
        out["rmse_no_nm"] = (float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) if m.any()
                             else float("nan"))
        out["n_no_nm"] = int(m.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--raw", default=None,
                    help="raw month file, to add the no-NM segment COMMON.md asks for")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = pd.read_parquet(a.cloud)[["MVT_ID_mvt", "y_true", "y_pred"]]
    l = pd.read_parquet(a.local)[["MVT_ID_mvt", "y_true", "y_pred"]]
    m = c.merge(l, on="MVT_ID_mvt", suffixes=("_cloud", "_local"))
    rep = {"fold": a.fold, "n_cloud": len(c), "n_local": len(l), "n_matched": len(m),
           "cloud_file": a.cloud, "local_file": a.local,
           "cloud_sha256": data.sha256(a.cloud), "local_sha256": data.sha256(a.local)}
    if len(m) == 0:
        raise SystemExit("no MVT_ID_mvt in common: the two files describe different rows")
    if not np.allclose(m["y_true_cloud"], m["y_true_local"], equal_nan=True):
        d = int((m["y_true_cloud"] != m["y_true_local"]).sum())
        rep["LABELS_DIFFER_rows"] = d
        print(f"*** labels differ on {d} rows: these are not the same data ***")

    if a.raw:
        raw = pd.read_parquet(a.raw, columns=["MVT_ID_mvt", "AOBT_3_flt"])
        raw["no_nm"] = raw["AOBT_3_flt"].isna()
        m = m.merge(raw[["MVT_ID_mvt", "no_nm"]], on="MVT_ID_mvt", how="left")

    for tag, col in (("cloud", "y_pred_cloud"), ("local", "y_pred_local")):
        rep[tag] = segments(m.rename(columns={"y_true_cloud": "y_true"}), col)
    d = m["y_pred_cloud"].to_numpy() - m["y_pred_local"].to_numpy()
    rep["delta"] = {
        "rmse_cloud_minus_local": rep["cloud"]["rmse"] - rep["local"]["rmse"],
        "identical": bool(np.array_equal(m["y_pred_cloud"], m["y_pred_local"])),
        "max_abs": float(np.abs(d).max()), "mean_abs": float(np.abs(d).mean()),
        "median_abs": float(np.median(np.abs(d))),
        "p99_abs": float(np.percentile(np.abs(d), 99)),
        "rms": float(np.sqrt(np.mean(d ** 2))),
        "corr": float(np.corrcoef(m["y_pred_cloud"], m["y_pred_local"])[0, 1]),
        "rows_over_1s": int((np.abs(d) > 1).sum()),
        "rows_over_60s": int((np.abs(d) > 60).sum()),
    }
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
