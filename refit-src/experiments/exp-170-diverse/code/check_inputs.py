"""Independent checks of real input domains, windows and frozen held rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from fit import windows
    meta = pd.read_parquet(a.inputs / "rows.parquet")
    rec = json.loads((a.inputs / "MANIFEST.json").read_text())
    num = np.load(a.inputs / "numeric.npy", mmap_mode="r")
    cat = np.load(a.inputs / "category.npy", mmap_mode="r")
    y = np.load(a.inputs / "labels.npy", mmap_mode="r")
    train = (meta.year.to_numpy() == 2025) & ~meta.month.isin([1, 7]).to_numpy()
    held = (meta.year.to_numpy() == 2025) & meta.month.isin([1, 7]).to_numpy()
    frozen = pd.read_parquet(a.frozen)
    ids = frozen.loc[frozen.month.isin([1, 7]), "MVT_ID_mvt"].to_numpy("int64")
    checks = {}
    def check(name, ok):
        checks[name] = bool(ok)
        if not ok:
            raise AssertionError(name)
    check("held IDs match all frozen jjrefit rows", set(ids) == set(meta.loc[held, "MVT_ID_mvt"]))
    check("all held and rank IDs unique", not meta.MVT_ID_mvt.duplicated().any())
    check("all model numbers finite", np.isfinite(num).all())
    check("all train labels finite", np.isfinite(y[train]).all())
    check("ranking labels absent", np.isnan(y[meta.year.to_numpy() == 2026]).all())
    check("ten train months", set(meta.loc[train, "month"]) == {2, 3, 4, 5, 6, 8, 9, 10, 11, 12})
    check("all twelve months loaded", set(meta.loc[meta.year == 2025, "month"]) == set(range(1, 13)))
    check("no label or BLOCK input", not any("BLOCK" in c or "TAXITIME" in c for c in rec["numeric"] + rec["category"]))
    check("no shared arrival-label shape input", not any(c.startswith("shp_") for c in rec["numeric"]))
    check("numeric feature width", num.shape[1] == 2 * len(rec["numeric"]))
    for i, c in enumerate(rec["category"]):
        check("category bounds " + c, cat[:, i].min() >= 0 and cat[:, i].max() <= len(rec["vocab"][c]))
    windows_checked = 0
    for domain, mask in [("train", train), ("held", held)]:
        for halo in [0, 16, 48]:
            seen = np.zeros(len(meta), dtype="uint8")
            for lo, hi, c, end in windows(meta, mask, halo):
                check_name = domain + " windows stay within day and domain"
                if not (meta.group.iloc[lo] == meta.group.iloc[hi - 1] and mask[lo:hi].all()
                        and lo <= c < end <= hi and meta.MVT_TIME_UTC_mvt.iloc[lo] <= meta.MVT_TIME_UTC_mvt.iloc[hi - 1]):
                    raise AssertionError(check_name)
                seen[c:end] += 1
                windows_checked += 1
            check(domain + " exact output coverage halo " + str(halo),
                  np.array_equal(seen, mask.astype("uint8")))
    result = {"checks": checks, "passed": len(checks), "windows_checked": windows_checked,
              "train_rows": int(train.sum()), "held_rows": int(held.sum()),
              "held_ids_sha256": hashlib.sha256(np.sort(ids).tobytes()).hexdigest()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
