"""The noac frames this experiment fits on: every `tail_signals` column, the
chain, and (2025 only) the label.

Self-contained: it repeats the signal build rather than reading another
experiment's work files, so `code/run_all.sh` reproduces every number here from
`data/raw` alone.  `work/noac_2025.parquet` here is asserted row-for-row equal
to exp-033's frame on `MVT_ID_mvt`, which is the check that the population is
the same 1,507 and 388 the record means.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
from prc.tail_features import tail_signals  # noqa: E402
from prc.tail_model import PLACEHOLDER_TOL  # noqa: E402
from build import add_band, CHAIN_COLS  # noqa: E402
from chain import stand_chain  # noqa: E402
from rotation import rotation_signals, ROT_COLS  # noqa: E402

#: label-free columns the chain contributes.  `self_aobt_pre` is dropped: every
#: noac row is also `nonm` (measured: 0 of 1,507 have an NM off-block), so the
#: column is constant False on this population.
CHAIN_FEATS = ("g_any", "g_prev2", "slack", "slack2", "log_g", "n_dep_between",
               "arr_stand_per_day")

#: the round-2 arm.  `g_any`/`g_prev2` turned out to BE the shipped `gap_arr1`
#: and `gap_arr2` (round 1 finding 2), so the chain arm above adds only the
#: transforms and the two counts.  These are the observable the review's
#: refutation of the mechanism uncovered, and the airline prefix of an arrival
#: is genuinely outside `NOAC_FEATURES_V2`.
ROT_FEATS = ("rot_same_near", "rot_same_before", "rot_same_after", "rot_same_gap",
             "rot_g_same", "rot_pref_base", "rot_n_same_stand", "rot_slack")


def add_chain_feats(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["slack"] = d.g_any - d.mvt_sched
    d["slack2"] = d.g_prev2 - d.mvt_sched
    d["log_g"] = np.log1p(np.clip(d.g_any, 0, None))
    # `n_arr_stand` counts a whole month, and the months are 28 to 31 days long
    # while both ranking months are 31; per day is the comparable quantity.
    dim = pd.to_datetime(d.month + "-01").dt.days_in_month.to_numpy("float64")
    d["arr_stand_per_day"] = d.n_arr_stand.to_numpy("float64") / dim
    # the same-airline chain read the way `slack` reads the any-airline one
    if "rot_g_same" in d.columns:
        d["rot_slack"] = d.rot_g_same - d.mvt_sched
    return d


def _one(masked: pd.DataFrame, raw: pd.DataFrame, month: str) -> pd.DataFrame:
    sig = tail_signals(masked)
    keep = sig.index[sig["noac"].to_numpy() == 1]
    ch = stand_chain(masked).reindex(keep)
    rot = rotation_signals(masked, keep)
    out = sig.loc[keep].copy()
    for c in CHAIN_COLS:
        out[c] = ch[c].to_numpy()
    for c in ROT_COLS:
        out[c] = rot[c].to_numpy()
    out["MVT_ID_mvt"] = raw["MVT_ID_mvt"].reindex(keep).to_numpy("float64")
    out["month"] = month
    out["t_off"] = raw["MVT_TIME_UTC_mvt"].reindex(keep).to_numpy()
    out["day"] = pd.to_datetime(out["t_off"]).dt.strftime("%Y-%m-%d")
    return add_chain_feats(add_band(out)).reset_index(drop=True)


def main() -> None:
    (EXP / "work").mkdir(parents=True, exist_ok=True)
    rank = D.load_ranking()
    b = pd.concat([_one(g, g, m) for m, g in rank.groupby("month", sort=True)], ignore_index=True)
    assert "y_true" not in b.columns and "ph" not in b.columns
    b.to_parquet(EXP / "work" / "noac_2026.parquet")
    print("2026 noac rows:", len(b))

    frames = []
    for m in D.MONTHS:
        raw = D.load_month(m)
        masked, y = D.mask_like_ranking(raw)
        f = _one(masked, raw, m)
        sig = tail_signals(masked)
        idx = sig.index[sig["noac"].to_numpy() == 1]
        # `_one` reset the index, so the label join rests on row order.  Assert it
        # rather than trust it: a silent misalignment here would move every label.
        assert np.array_equal(raw["MVT_ID_mvt"].reindex(idx).to_numpy("float64"),
                              f["MVT_ID_mvt"].to_numpy("float64")), \
            f"{m}: the label index is not aligned with the frame"
        f["y_true"] = y.reindex(idx).to_numpy("float64")
        f["ph"] = (np.abs(f.mvt_sched - f.y_true) < PLACEHOLDER_TOL).astype("float64")
        frames.append(f)
        print(f"{m}: {len(f)} noac", flush=True)
    a = pd.concat(frames, ignore_index=True)
    a.to_parquet(EXP / "work" / "noac_2025.parquet")
    print("2025 noac rows:", len(a))

    for f, mine in (("noac_2025", a), ("noac_2026", b)):
        ref = pd.read_parquet(EXP.parents[0] / "exp-033-noac" / "work" / f"{f}.parquet")
        same = set(ref.MVT_ID_mvt) == set(mine.MVT_ID_mvt)
        print(f"{f}: same population as exp-033: {same} ({len(mine)} rows)")
        assert same


if __name__ == "__main__":
    main()
