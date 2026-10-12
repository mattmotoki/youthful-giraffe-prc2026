"""Are the fitted features and the served features the same function? — no data needed.

`serve.py` cannot run until `data/raw/ranking.parquet` exists, so until then its
feature path is unexecuted code that a submission would depend on.  This test
executes it against a real month.

**The construction.** A fold's canonical panel carries `t`, `d_takeoff_aobt`,
`d_aobt_lobt`, `noac`, `nm_missing` and `airport`.  Those are exactly invertible
into the raw movement columns that produced them:

    MVT_TIME_UTC_mvt = t
    AOBT_3_flt       = t - d_takeoff_aobt
    LOBT_flt         = AOBT_3_flt - d_aobt_lobt
    AIRCRAFT_TYPE_mvt = null iff noac      FLIGHT_ID_mvt = null iff nm_missing
    ADEP_mvt          = airport            PHASE_mvt     = "DEP"

so the raw frame below has a real month's distribution, its real airports and its
real 2.06 % of rows with no NM join — it is not synthetic data, it is the same
data written in the raw file's columns.  Pushing it back through
`features.panel_from_movements` and `features.window_features` must reproduce the
fitted design matrix exactly.

Also checked here, because they are the ways this pair breaks in practice:

1. the regime `panel_from_movements` derives equals the one the members froze;
2. `apt` survives an airport being absent from the frame (the fit/serve break
   that a per-frame `pd.Categorical` would cause);
3. arrival rows in the movement frame change nothing — the window population is
   departures on both sides;
4. the `shuffle` control leaves every window column's multiset unchanged.

    .venv/bin/python experiments/exp-028-window/tests/shape_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import features as Fe  # noqa: E402
import panel as P  # noqa: E402

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def raw_from_panel(p: pd.DataFrame, add_arrivals: int = 0) -> pd.DataFrame:
    t = pd.to_datetime(p["t"].to_numpy("float64"), unit="s", utc=True)
    aobt = t - pd.to_timedelta(p["d_takeoff_aobt"].to_numpy("float64"), unit="s")
    lobt = aobt - pd.to_timedelta(p["d_aobt_lobt"].to_numpy("float64"), unit="s")
    noac = p["noac"].to_numpy().astype(bool)
    nonm = p["nm_missing"].to_numpy().astype(bool)
    raw = pd.DataFrame({
        "MVT_ID_mvt": p["MVT_ID_mvt"].to_numpy("float64"),
        "PHASE_mvt": "DEP",
        "AIRCRAFT_TYPE_mvt": np.where(noac, None, "A320"),
        "FLIGHT_ID_mvt": np.where(nonm, np.nan, 1.0),
        "MVT_TIME_UTC_mvt": t,
        "AOBT_3_flt": aobt,
        "LOBT_flt": lobt,
        "ADEP_mvt": p["airport"].astype(str).to_numpy(),
    })
    if add_arrivals:
        rng = np.random.default_rng(3)
        take = rng.choice(len(raw), size=add_arrivals, replace=False)
        arr = raw.iloc[take].copy()
        arr["PHASE_mvt"] = "ARR"
        arr["MVT_ID_mvt"] = arr["MVT_ID_mvt"] + 1e9   # ids outside the departure space
        raw = pd.concat([raw, arr], ignore_index=True)
    return raw


def check(fold: str) -> None:
    p = P.load_fold(fold)
    X_fit = Fe.window_features(p)

    raw = raw_from_panel(p, add_arrivals=20_000)
    q = Fe.panel_from_movements(raw, base=pd.Series(p["base"].to_numpy("float64"),
                                                    index=p["MVT_ID_mvt"].to_numpy("float64")))
    X_srv = Fe.window_features(q)

    for c in ("MVT_ID_mvt", "t", "d_takeoff_aobt", "d_aobt_lobt", "base"):
        a, b = p[c].to_numpy("float64"), q[c].to_numpy("float64")
        bad = int((~(np.isclose(a, b, rtol=0, atol=1e-6) | (np.isnan(a) & np.isnan(b)))).sum())
        assert bad == 0, f"{fold}: panel column {c} differs on {bad} rows"
    for c in ("airport", "regime"):
        bad = int((p[c].astype(str).to_numpy() != q[c].astype(str).to_numpy()).sum())
        assert bad == 0, f"{fold}: panel column {c} differs on {bad} rows"
    assert list(X_fit.columns) == list(X_srv.columns), f"{fold}: column order differs"
    a, b = X_fit.to_numpy("float64"), X_srv.to_numpy("float64")
    bad = ~(np.isclose(a, b, rtol=0, atol=1e-6) | (np.isnan(a) & np.isnan(b)))
    assert not bad.any(), (f"{fold}: design matrices differ on {int(bad.sum())} cells, "
                           f"worst column {X_fit.columns[bad.any(0).argmax()]}")

    # 2. an absent airport must not renumber the others
    keep = p["airport"].to_numpy() != "LEBL"
    sub = Fe.window_features(p[keep].reset_index(drop=True))
    ref = X_fit.loc[keep, "apt"].to_numpy()
    assert np.array_equal(sub["apt"].to_numpy(), ref), f"{fold}: apt renumbers when LEBL is absent"

    # 4. the shuffle control preserves every window column's multiset
    sh = Fe.shuffle_within_day(X_fit, p, seed=5)
    for c in Fe.window_cols(X_fit):
        u = np.sort(X_fit[c].to_numpy()); v = np.sort(sh[c].to_numpy())
        assert np.array_equal(u, v) or np.allclose(u, v, equal_nan=True), \
            f"{fold}: shuffle changed the multiset of {c}"
    moved = float((X_fit["w_ta_mean"].to_numpy() != sh["w_ta_mean"].to_numpy()).mean())
    assert moved > 0.5, f"{fold}: shuffle moved only {moved:.1%} of rows"

    print(f"  {fold:9s} panel {p.shape} design {X_fit.shape}  "
          f"fit==serve on all {a.size:,} cells  shuffle moved {moved:.1%}")


def main() -> None:
    print("[shape] fitted features against served features, on real months")
    for f in P.FOLDS:
        check(f)
    print("[shape] all checks passed")


if __name__ == "__main__":
    main()
