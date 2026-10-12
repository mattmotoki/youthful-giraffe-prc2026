"""The non-duplicated part of prc.stand_features (exp-003's recommended bulk set).

exp-003 recommends 18 `occupancy_features` columns and 6
`airport_neighbour_features` columns. Measured against what this experiment
already builds — its own v3 arrival-side columns plus `prc.lirf_features` — **15
of the 24 are near-duplicates**, and 8 of those correlate at |r| = 1.000 under a
different name:

    dep_next_mvt_gap   = lf_stand_next_dep_gap        taxi_ub_pm = stand_prev_arr_block_gap
    dep_prev_mvt_gap   = stand_prev_dep_gap           taxi_lb_nm = lf_stand_next_block_gap
    dep_next_aobt_gap  = lf_stand_next_dep_gap        off_ub_nm  = lf_stand_next_block_gap
    dep_prev_aobt_gap  = stand_prev_dep_gap           off_lb_pm  = stand_prev_arr_block_gap

with six more between 0.99 and 1.00 (`arr_td_gap_pa`, `arr_turnaround_pa`,
`off_lb_pa`, `taxi_ub_pa`, `off_ub_na`, `taxi_lb_na`) and `arr_taxi_pa` at 0.900.
Adding those would put four modules' versions of the same stand-occupancy
measurement into one frame, which costs fitting budget and buys nothing.

What is genuinely new is the **airport-neighbour family**: the neighbours' own
`take-off − AOBT`, a live readout of surface conditions from other flights'
actuals rather than a count of them. Those six correlate 0.14 to 0.24 with the
target and at most 0.50 with anything here. `same_type_pa` and `same_type_na`
are new too (0.20 and 0.10 against everything) though weak on their own, and
`arr_taxi_na` is half-new at 0.498.
"""

from __future__ import annotations

import pandas as pd

import prc.stand_features as SF

#: kept: everything whose maximum correlation against an existing column is < 0.9
KEEP = (
    "nbr_apt_30m", "nbr_apt_prev10_nmtaxi", "nbr_apt_next10_nmtaxi",
    "nbr_apt_prev10_mvtsched", "nbr_rwy_prev10_nmtaxi", "nbr_stand_prev5_nmtaxi",
    "arr_taxi_na", "same_type_pa", "same_type_na",
)
#: dropped as duplicates, with the column each one reproduces
DUPLICATES = {
    "dep_next_mvt_gap": "lf_stand_next_dep_gap", "dep_prev_mvt_gap": "stand_prev_dep_gap",
    "dep_next_aobt_gap": "lf_stand_next_dep_gap", "dep_prev_aobt_gap": "stand_prev_dep_gap",
    "taxi_ub_pm": "stand_prev_arr_block_gap", "off_lb_pm": "stand_prev_arr_block_gap",
    "taxi_lb_nm": "lf_stand_next_block_gap", "off_ub_nm": "lf_stand_next_block_gap",
    "arr_td_gap_pa": "stand_prev_arr_land_gap", "arr_turnaround_pa": "stand_prev_arr_land_gap",
    "off_lb_pa": "stand_prev_arr_land_gap", "taxi_ub_pa": "stand_prev_arr_land_gap",
    "off_ub_na": "lf_stand_next_block_gap", "taxi_lb_na": "lf_stand_next_block_gap",
    "arr_taxi_pa": "stand_prev_arr_taxiin",
}


def _one(raw: pd.DataFrame) -> pd.DataFrame:
    df = SF.add_stand_keys(raw)
    out = pd.concat([SF.occupancy_features(df, key="key_raw"),
                     SF.airport_neighbour_features(df)], axis=1)
    dep = raw[raw["PHASE_mvt"].eq("DEP")]
    out = out.reindex(dep.index)[list(KEEP)].copy()
    out.insert(0, "MVT_ID_mvt", dep["MVT_ID_mvt"].to_numpy())
    return out.reset_index(drop=True)


def _attach(X: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    j = out.set_index("MVT_ID_mvt").reindex(X["MVT_ID_mvt"].to_numpy())
    j.index = X.index
    return pd.concat([X, j.add_prefix("st_")], axis=1)


def fit_transform(train_months, Xtr: pd.DataFrame, raw_by_month: dict,
                  others: list[tuple[pd.DataFrame, pd.DataFrame]]):
    """No fit: both functions are per-month and label-free, so nothing is learned."""
    tr = pd.concat([_one(raw_by_month[m]) for m in train_months], ignore_index=True)
    return _attach(Xtr, tr), [_attach(X, _one(raw)) for X, raw in others]
