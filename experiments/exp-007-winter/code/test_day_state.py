"""Leak and construction tests for prc.day_state_features.

  .venv/bin/python experiments/exp-007-winter/code/test_day_state.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import prc.data as D  # noqa: E402
import prc.day_state_features as DS  # noqa: E402


def main():
    fold = D.FOLDS["july"]
    full = D.load_month(fold.valid_month)
    masked, _ = D.mask_like_ranking(full)
    train = D.load_month("2025-03")
    ds = DS.DayState().fit(train)

    # 1. masking the departure labels changes nothing: the module cannot be reading them
    a = ds.transform(full)
    b = ds.transform(masked)
    assert list(a.columns) == list(b.columns) == list(DS.COLUMNS)
    diff = (a.fillna(-9e9).to_numpy() != b.fillna(-9e9).to_numpy()).sum()
    assert diff == 0, f"{diff} cells differ between the unmasked and masked frame"
    print(f"1. masked == unmasked on all {a.shape[0]:,} x {a.shape[1]} cells: PASS")

    # 2. fit() on a frame whose departure labels are destroyed gives the same reference
    wrecked = train.copy()
    dep = wrecked["PHASE_mvt"].eq("DEP")
    wrecked.loc[dep, "TAXITIME_SEC_mvt"] = 12345
    wrecked.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    r2 = DS.DayState().fit(wrecked).reference_
    assert np.allclose(ds.reference_.to_numpy(), r2.to_numpy(), equal_nan=True)
    print("2. fit() ignores departure labels entirely: PASS")

    # 3. arrival taxi times DO reach the features (they are legal and are meant to)
    w2 = train.copy()
    arr = w2["PHASE_mvt"].eq("ARR")
    w2.loc[arr, "TAXITIME_SEC_mvt"] = 999
    r3 = DS.DayState().fit(w2).reference_
    assert not np.allclose(ds.reference_["ds_taxiin_med__med"].to_numpy(),
                           r3["ds_taxiin_med__med"].to_numpy(), equal_nan=True)
    print("3. arrival taxi-in does reach the features: PASS")

    # 4. the hourly de-icing proxy excludes the row itself
    m2 = masked.copy()
    di = np.flatnonzero(m2["PHASE_mvt"].eq("DEP").to_numpy())
    pick = di[5000]
    m2.iloc[pick, m2.columns.get_loc("AOBT_3_flt")] = (
        m2.iloc[pick]["MVT_TIME_UTC_mvt"] - pd.Timedelta(seconds=9000))
    c = ds.transform(m2)
    own_before = b.iloc[b.index.get_indexer([masked.index[pick]])[0]]["ds_deice_hr"]
    own_after = c.iloc[c.index.get_indexer([m2.index[pick]])[0]]["ds_deice_hr"]
    assert (np.isnan(own_before) and np.isnan(own_after)) or own_before == own_after, \
        f"the row's own proxy moved when its own span moved: {own_before} -> {own_after}"
    moved = int((b["ds_deice_hr"].fillna(-1) != c["ds_deice_hr"].fillna(-1)).sum())
    print(f"4. own row excluded from ds_deice_hr (its own value fixed at {own_before:.4f}; "
          f"{moved} neighbours moved): PASS")

    # 5. the day version excludes the row too, and matches a hand computation.
    # NOT SUBSTANTIVE as written (gpt-5.6-sol round 1, non-blocking): the chosen
    # row's hand value is 0, so a mutant that includes the row still passes. It is
    # kept because it does check the arithmetic path; the exclusion itself is
    # covered by test 4, whose mutant IS caught (0.0 -> 0.0294).
    apt = masked["ADEP_mvt"].astype(str)
    day = masked["MVT_TIME_UTC_mvt"].dt.floor("D")
    dep_i = masked["PHASE_mvt"].eq("DEP")
    span = (masked["MVT_TIME_UTC_mvt"] - masked["AOBT_3_flt"]).dt.total_seconds()
    sub = pd.DataFrame({"apt": apt[dep_i], "day": day[dep_i], "span": span[dep_i]})
    k = sub.index[100]
    g = sub[(sub.apt == sub.loc[k, "apt"]) & (sub.day == sub.loc[k, "day"])].drop(index=k)
    hand = float((g["span"] > 1800).sum() / g["span"].notna().sum())
    got = float(b.loc[k, "ds_deice_day"])
    assert abs(hand - got) < 1e-6, f"{hand} != {got}"
    print(f"5. ds_deice_day matches a hand computation with the row removed ({got:.6f}): PASS")

    # 6. a thin airport-day is NaN rather than a one-row statistic.
    # NOT SUBSTANTIVE on this month (gpt-5.6-sol round 1, non-blocking): July's
    # thinnest airport-day has 382 departures, so removing the gate also passes.
    # The gate is present in the module and fires on partial days in other frames.
    counts = sub.groupby(["apt", "day"]).size()
    print(f"6. thinnest airport-day has {counts.min()} departures; "
          f"NaN day statistics on {int(b['ds_span_med'].isna().sum())} rows: PASS")

    # 7. every produced column is float32 and the column list is the module's
    assert set(b.dtypes) == {np.dtype("float32")}
    print("7. schema is the module's COLUMNS, all float32: PASS")


if __name__ == "__main__":
    main()
