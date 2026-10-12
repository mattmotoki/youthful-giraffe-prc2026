"""Create the two fixed time-field input views; preserve the original."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_time_day_052 import PIN, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--check", type=Path, required=True)
    ap.add_argument("--out-z", type=Path, required=True)
    ap.add_argument("--out-t", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    a = ap.parse_args()
    check = json.loads(a.check.read_text())
    assert check["status"] == "RETAINED_FALSE_DAY" and check["manifest_sha256"] == PIN
    assert check["eligible"] == 1000 and check["old_expression_within_12"] == 1000
    assert sha(a.inputs / "MANIFEST.json") == PIN
    parent = json.loads((a.inputs / "MANIFEST.json").read_text())
    for name, digest in parent["files"].items():
        assert sha(a.inputs / name) == digest, name
    assert not any(p.exists() for p in [a.out_z, a.out_t, a.receipt])
    meta = pd.read_parquet(a.inputs / "rows.parquet")
    old = np.load(a.inputs / "numeric.npy", mmap_mode="r")
    assert old.shape == (2429888, 118) and old.dtype == np.float32
    assert parent["numeric"][57] == "seconds_in_day"
    times = meta.MVT_TIME_UTC_mvt
    assert str(times.dtype) == "datetime64[us, UTC]"
    train = (meta.year == 2025).to_numpy() & ~meta.month.isin([1, 7]).to_numpy()
    months = sorted(meta.loc[train, "month"].unique().tolist())
    assert months == [2, 3, 4, 5, 6, 8, 9, 10, 11, 12] and train.sum() == 1740628
    raw = (times.dt.hour * 3600 + times.dt.minute * 60 + times.dt.second +
           times.dt.microsecond / 1e6).to_numpy(float)
    val = np.sign(raw) * np.log1p(np.abs(raw))
    center, scale = float(np.nanmedian(val[train])), float(max(np.nanstd(val[train]), .1))
    assert np.isfinite(center) and np.isfinite(scale)
    corrected = np.nan_to_num((val - center) / scale, nan=0.).clip(-12, 12).astype("float32")
    finite = np.isfinite(val).astype("float32")
    same = [c for c in range(118) if c not in [57, 116]]
    receipt = {"source_row": 902, "parent_manifest_sha256": PIN, "check_sha256": sha(a.check),
               "code_sha256": sha(__file__), "changed_columns": [57, 116],
               "train_months": months, "train_rows": int(train.sum()),
               "train_ids_sha256": hashlib.sha256(meta.loc[train, "MVT_ID_mvt"].to_numpy("<i8").tobytes()).hexdigest(),
               "T_center": center, "T_scale": scale, "views": {}}
    for form, out in [("Z", a.out_z), ("T", a.out_t)]:
        out.mkdir(parents=True, exist_ok=False)
        numeric = np.lib.format.open_memmap(out / "numeric.npy", mode="w+", dtype="float32", shape=old.shape)
        for lo in range(0, len(old), 65536):
            hi = min(lo + 65536, len(old))
            numeric[lo:hi] = old[lo:hi]
            numeric[lo:hi, 57] = 0. if form == "Z" else corrected[lo:hi]
            numeric[lo:hi, 116] = 0. if form == "Z" else finite[lo:hi]
            assert np.array_equal(numeric[lo:hi, same], old[lo:hi, same])
            assert np.isfinite(numeric[lo:hi]).all()
        numeric.flush()
        del numeric
        for name in ["rows.parquet", "category.npy", "anchor.npy", "labels.npy"]:
            (out / name).symlink_to((a.inputs / name).resolve())
        rec = json.loads(json.dumps(parent))
        if form == "T":
            rec["center"][57], rec["scale"][57] = center, scale
        rec.update({"parent_manifest_sha256": PIN, "source_code_sha256": sha(__file__),
                    "array_rule": "only57/116: zero both" if form == "Z" else
                                  "only57/116: true UTC day seconds, ten-training-month scaler",
                    "time_day_form": form, "scaler_train_months": months,
                    "changed_numeric_columns": [57, 116], "unchanged_columns_exact": True,
                    "check_sha256": sha(a.check),
                    "files": {name: sha(out / name) for name in parent["files"]}})
        for name in ["rows.parquet", "category.npy", "anchor.npy", "labels.npy"]:
            assert rec["files"][name] == parent["files"][name], name
        (out / "MANIFEST.json").write_text(json.dumps(rec, indent=1) + "\n")
        receipt["views"][form] = {"path": str(out.resolve()), "manifest_sha256": sha(out / "MANIFEST.json"),
                                  "numeric_sha256": rec["files"]["numeric.npy"],
                                  "other_columns_exact": True, "other_arrays_exact": True}
        print("ready", form, receipt["views"][form], flush=True)
    assert sha(a.inputs / "numeric.npy") == parent["files"]["numeric.npy"]
    receipt["original_unchanged"] = True
    a.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
