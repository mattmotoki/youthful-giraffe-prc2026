"""Training-only whole-airport-day clock blackouts on a reviewed fixed-time input."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_time_day_052 import sha
from native_weekpair import training_rate_month

COVERAGE = "5512d3202b61e2c9b1f03b0f35df6e18728facccb90399ef74e22072d0c76f9c"
VIEWS = {"Z": "d052ad0239f6ebca4c05320eac7ba0766dbbfcd158afc06a318fa880ce92d3b1",
         "T": "96a7f802e4945d01cbd93397eed4864767891731a53f85ebd7fd0c6962278b49"}


def chosen_days(airport, month, rate):
    ts = pd.Timestamp(f"2025-{month:02d}-01")
    dates = [(ts + pd.Timedelta(days=i)).date().isoformat() for i in range(ts.days_in_month)]
    def key(date):
        return int.from_bytes(hashlib.sha256(f"prc-exp170-darkday-603|{airport}|{date}".encode()).digest()[:8], "big")
    count = int(np.floor(rate * len(dates) + .5))
    return sorted(sorted(dates, key=key)[:count])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--coverage", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annual", action="store_true")
    a = ap.parse_args()
    selection = json.loads(a.selection.read_text())
    assert selection["status"] == "PASS" and selection["chosen_form"] in VIEWS
    form = selection["chosen_form"]
    assert sha(a.inputs / "MANIFEST.json") == VIEWS[form]
    assert selection["chosen_input_manifest_sha256"] == VIEWS[form]
    assert sha(a.coverage) == COVERAGE and not a.out.exists()
    rec = json.loads((a.inputs / "MANIFEST.json").read_text())
    assert rec["time_day_form"] == form and rec["unchanged_columns_exact"]
    for name, digest in rec["files"].items():
        assert sha(a.inputs / name) == digest, name
    meta = pd.read_parquet(a.inputs / "rows.parquet")
    assert meta.MVT_ID_mvt.is_unique and meta.MVT_ID_mvt.dtype == np.int64
    old = np.load(a.inputs / "numeric.npy", mmap_mode="r")
    assert old.shape == (2429888, 118) and old.dtype == np.float32 and len(rec["numeric"]) == 59
    names = [n for n in rec["numeric"] if n.startswith(("obs_", "trk_", "clock_", "opdi_")) or n == "nm_minus_obs"]
    assert len(names) == 38
    values = [rec["numeric"].index(n) for n in names]
    cols = values + [c + 59 for c in values]
    assert len(set(cols)) == 76 and not set(cols) & {57, 116}
    train = (meta.year == 2025).to_numpy()
    if not a.annual:
        train &= ~meta.month.isin([1, 7]).to_numpy()
    assert int(train.sum()) == (2085047 if a.annual else 1740628)
    coverage = pd.read_csv(a.coverage)
    assert not coverage.duplicated(["airport", "month"]).any()
    rate = coverage.set_index(["airport", "month"]).rate
    selected, detail = set(), []
    airports = sorted(meta.ADEP_mvt.astype(str).unique())
    months = list(range(1, 13)) if a.annual else [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    for airport in airports:
        for month in months:
            mapped = training_rate_month(month)
            r = float(rate.loc[(airport, mapped)])
            assert 0 <= r <= 1
            days = chosen_days(airport, month, r)
            selected.update((airport, day) for day in days)
            detail.append({"airport": airport, "month": month, "coverage_month": mapped,
                           "rate": r, "calendar_days": pd.Timestamp(f"2025-{month:02d}-01").days_in_month,
                           "selected_dates": days})
    pairs = list(zip(meta.ADEP_mvt.astype(str), meta.date.astype(str)))
    mask = train & np.fromiter((p in selected for p in pairs), dtype=bool, count=len(meta))
    # The same mask applies to every runway/row in each training airport-date.
    check = pd.DataFrame({"airport": meta.ADEP_mvt, "date": meta.date, "train": train, "mask": mask})
    assert (check.loc[train].groupby(["airport", "date"])["mask"].nunique() == 1).all()
    assert not mask[~train].any() and not mask[meta.year.to_numpy() == 2026].any()
    same = [c for c in range(118) if c not in cols]
    a.out.mkdir(parents=True, exist_ok=False)
    numeric = np.lib.format.open_memmap(a.out / "numeric.npy", mode="w+", dtype="float32", shape=old.shape)
    for lo in range(0, len(old), 65536):
        hi = min(lo + 65536, len(old))
        block = numeric[lo:hi]
        block[:] = old[lo:hi]
        hit = np.flatnonzero(mask[lo:hi])
        block[np.ix_(hit, cols)] = 0.
        assert np.array_equal(block[:, same], old[lo:hi, same])
        assert np.array_equal(block[~mask[lo:hi]], old[lo:hi][~mask[lo:hi]])
        assert (block[np.ix_(hit, cols)] == 0).all() and np.isfinite(block).all()
    numeric.flush()
    del numeric
    for name in ["rows.parquet", "category.npy", "anchor.npy", "labels.npy"]:
        (a.out / name).symlink_to((a.inputs / name).resolve())
    files = {name: sha(a.out / name) for name in rec["files"]}
    for name in ["rows.parquet", "category.npy", "anchor.npy", "labels.npy"]:
        assert files[name] == rec["files"][name]
    for record in detail:
        domain = train & (meta.ADEP_mvt.to_numpy() == record["airport"]) & (meta.month.to_numpy() == record["month"])
        record["train_rows"] = int(domain.sum())
        record["masked_rows"] = int((domain & mask).sum())
        record["train_dates"] = int(meta.loc[domain, "date"].nunique())
        record["masked_train_dates"] = int(meta.loc[domain & mask, "date"].nunique())
        record["masked_train_date_share"] = (record["masked_train_dates"] / record["train_dates"]
                                              if record["train_dates"] else 0.)
        record["masked_train_row_share"] = (record["masked_rows"] / record["train_rows"]
                                             if record["train_rows"] else 0.)
        flags = [col + 59 for col in values]
        before = (old[np.ix_(np.flatnonzero(domain), flags)] == 1).sum(axis=0)
        after = (old[np.ix_(np.flatnonzero(domain & ~mask), flags)] == 1).sum(axis=0)
        record["clock_flag_present_before"] = dict(zip(names, before.astype(int).tolist()))
        record["clock_flag_present_after"] = dict(zip(names, after.astype(int).tolist()))
    result = {**rec, "parent_manifest_sha256": VIEWS[form], "source_code_sha256": sha(__file__),
              "selection_sha256": sha(a.selection), "coverage_sha256": COVERAGE,
              "array_rule": "only38 clock values/38 flags zero on selected full training airport-days; fixed time exact",
              "darkday_seed": 603, "darkday_hash_salt": "prc-exp170-darkday-603",
              "darkday_training_domain": "2025_all" if a.annual else "2025_ten_months",
              "darkday_masked_names": names, "darkday_masked_indices": cols,
              "changed_numeric_columns": cols,
              "masked_rows": int(mask.sum()), "masked_airport_dates": int(check.loc[mask].drop_duplicates(["airport", "date"]).shape[0]),
              "day_details": detail, "all_other_columns_exact": True,
              "nontraining_rows_exact": True, "fixed57_116_exact": True, "files": files}
    (a.out / "MANIFEST.json").write_text(json.dumps(result, indent=1) + "\n")
    assert sha(a.inputs / "numeric.npy") == rec["files"]["numeric.npy"]
    print(json.dumps({"manifest_sha256": sha(a.out / "MANIFEST.json"), "masked_rows": int(mask.sum()),
                      "airport_dates": result["masked_airport_dates"], "fixed_form": form,
                      "ranking2026_exact": True, "held2025_exact": not a.annual}), flush=True)


if __name__ == "__main__":
    main()
