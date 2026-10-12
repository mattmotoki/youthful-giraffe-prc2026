"""Read-only replay of saved column57; no label or checkpoint reads."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PIN = "33a8cc487a044d01fc40d01f856dc688ee5b86721ac0fc021a7b8dd607000352"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ranges(x):
    return dict(zip(["min", "p01", "p25", "median", "p75", "p99", "max"],
                    np.quantile(x, [0, .01, .25, .5, .75, .99, 1]).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    assert sha(a.inputs / "MANIFEST.json") == PIN
    rec = json.loads((a.inputs / "MANIFEST.json").read_text())
    files = {k: sha(a.inputs / k) for k in ["numeric.npy", "rows.parquet"]}
    assert all(files[k] == rec["files"][k] for k in files)
    assert len(rec["numeric"]) == 59 and rec["numeric"][57] == "seconds_in_day"
    m = pd.read_parquet(a.inputs / "rows.parquet",
                        columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "year", "month", "ADEP_mvt"])
    z = np.load(a.inputs / "numeric.npy", mmap_mode="r")
    assert z.shape == (len(m), 118) and z.dtype == np.float32
    assert m.MVT_ID_mvt.dtype == np.int64 and m.MVT_ID_mvt.is_unique
    ts = m.MVT_TIME_UTC_mvt
    assert str(ts.dtype) == "datetime64[us, UTC]", str(ts.dtype)
    ix = np.linspace(0, len(m) - 1, 1000, dtype=np.int64)
    keep = (z[ix, 116] == 1) & (np.abs(z[ix, 57]) < 12) & ts.iloc[ix].notna().to_numpy()
    use = ix[keep]
    assert len(use) >= 900
    v = z[use, 57].astype(float) * rec["scale"][57] + rec["center"][57]
    raw = np.sign(v) * np.expm1(np.abs(v))
    sample = ts.iloc[use]
    old = (sample.astype("int64").to_numpy() / 1e9) % 86400
    true = (sample.dt.hour * 3600 + sample.dt.minute * 60 + sample.dt.second +
            sample.dt.microsecond / 1e6).to_numpy()
    olderr, trueerr = np.abs(raw - old), np.abs(raw - true)
    oldmatch, truematch = bool((olderr <= 12).all()), bool((trueerr <= 12).all())
    status = "RETAINED_FALSE_DAY" if oldmatch and not truematch else "ALREADY_TRUE_DAY" if truematch else "SOURCE_CONFLICT"
    train = (m.year == 2025).to_numpy() & ~m.month.isin([1, 7]).to_numpy()
    domains = {"2025_all": (m.year == 2025).to_numpy(), "train_ten_months": train,
               "2026_January": ((m.year == 2026) & (m.month == 1)).to_numpy(),
               "2026_July": ((m.year == 2026) & (m.month == 7)).to_numpy()}
    result = {"status": status, "source_row": 902, "manifest_sha256": PIN,
              "source_sha256": sha(__file__), "files": files, "timestamp_dtype": str(ts.dtype),
              "shape": list(z.shape), "column": 57, "flag": 116,
              "center": rec["center"][57], "scale": rec["scale"][57],
              "index_rule": "np.linspace(0,n-1,1000,dtype=int64)",
              "indices_sha256": hashlib.sha256(ix.astype("<i8").tobytes()).hexdigest(),
              "eligible_indices_sha256": hashlib.sha256(use.astype("<i8").tobytes()).hexdigest(),
              "sample_ids_sha256": hashlib.sha256(m.MVT_ID_mvt.iloc[use].to_numpy(dtype="<i8").tobytes()).hexdigest(),
              "sample_count": 1000, "eligible": len(use),
              "old_expression_max_abs_error": float(olderr.max()),
              "old_expression_within_12": int((olderr <= 12).sum()),
              "true_day_max_abs_error": float(trueerr.max()),
              "true_day_within_12": int((trueerr <= 12).sum()),
              "sample_examples": [{"index": int(use[j]), "id": int(m.MVT_ID_mvt.iloc[use[j]]),
                                   "timestamp": str(sample.iloc[j]), "reversed": float(raw[j]),
                                   "old_expression": float(old[j]), "true_day": float(true[j])}
                                  for j in [0, len(use)//2, len(use)-1]],
              "ranges": {name: {"rows": int(mask.sum()), "z57": ranges(z[mask, 57]),
                                 "flag_present": int((z[mask, 116] == 1).sum())}
                         for name, mask in domains.items()},
              "labels_read": False, "models_read": False, "inputs_changed": False}
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["status", "eligible", "old_expression_max_abs_error",
                                            "true_day_within_12", "ranges"]}), flush=True)


if __name__ == "__main__":
    main()
