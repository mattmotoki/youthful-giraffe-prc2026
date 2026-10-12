"""The exposure test: arrivals at the row's own stand in a window, against
`length x lambda` with lambda the stand's own arrival rate that month.

BOTH readings this module produced are WITHDRAWN (RESULTS.md 3b).  The first
used the full taxi length on each side of the block time, whose before-window
reaches back past the aircraft's own arrival and counts it.  The second, the
length-matched form below, removes that confounder but still cannot tell the
departing aircraft's own inbound from another aircraft, so "busy on both
sides" is what a rotation looks like when the recorded off-block predates the
arrival.  `rotation_check.py` is the test that decides, by splitting the
arrivals on the arriving flight's airline prefix.  This module is kept
because its numbers are quoted in the record as the withdrawn readings.
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
from build import LAB  # noqa: E402
from step3 import day_boot  # noqa: E402

KEY = ["apt", "stand"]


def counts(raw: pd.DataFrame) -> pd.DataFrame:
    """Arrivals at the stand inside `(BLOCK, T]`, and the stand's arrival rate."""
    masked, y = D.mask_like_ranking(raw)
    sig = tail_signals(masked)
    keep = sig.index[sig["noac"].to_numpy() == 1]
    arr = raw[raw.PHASE_mvt == "ARR"]
    a = pd.DataFrame({
        "apt": arr["ADES_mvt"].to_numpy(),
        "stand": arr["STAND_mvt"].to_numpy(),
        "t": arr["BLOCK_TIME_UTC_mvt"].to_numpy("datetime64[ns]"),
    }).dropna().sort_values("t", kind="mergesort")
    a["c"] = a.groupby(KEY, sort=False).cumcount().to_numpy() + 1.0

    d = pd.DataFrame({
        "apt": raw["ADEP_mvt"].reindex(keep).to_numpy(),
        "stand": raw["STAND_mvt"].reindex(keep).to_numpy(),
        "t_off": raw["MVT_TIME_UTC_mvt"].reindex(keep).to_numpy("datetime64[ns]"),
        "block": raw["BLOCK_TIME_UTC_mvt"].reindex(keep).to_numpy("datetime64[ns]"),
        "MVT_ID_mvt": raw["MVT_ID_mvt"].reindex(keep).to_numpy("float64"),
    })
    d["_row"] = np.arange(len(d), dtype="int64")

    def rank(col):
        left = d.loc[d[col].notna() & d.apt.notna() & d.stand.notna(),
                     KEY + [col, "_row"]].sort_values(col, kind="mergesort")
        j = pd.merge_asof(left, a, left_on=col, right_on="t", by=KEY,
                          direction="backward", allow_exact_matches=True)
        out = np.zeros(len(d))
        out[j["_row"].to_numpy("int64")] = np.nan_to_num(j["c"].to_numpy("float64"), nan=0.0)
        return out

    y_all = y.reindex(keep).to_numpy("float64")
    d["n_arr_in_taxi"] = rank("t_off") - rank("block")
    d["pre_full"] = d["block"] - pd.to_timedelta(np.clip(y_all, 0, None), unit="s")
    d["n_arr_placebo_full"] = rank("block") - rank("pre_full")
    # FIXED-LENGTH windows, the only length-matched comparison: the full-length
    # placebo reaches back past the aircraft's OWN arrival whenever the taxi is
    # longer than the turnaround, and then counts it.  `after_w` is the stand in
    # the w seconds AFTER the block time (free if the block time is real),
    # `before_w` the w seconds BEFORE it (occupied under every reading).
    for w in (900, 1800):
        d[f"post_{w}"] = np.minimum(d["block"] + pd.Timedelta(seconds=w), d["t_off"])
        d[f"pre_{w}"] = d["block"] - pd.Timedelta(seconds=w)
        d[f"after_{w}"] = rank(f"post_{w}") - rank("block")
        d[f"before_{w}"] = rank("block") - rank(f"pre_{w}")
        d[f"exp_after_{w}"] = (d[f"post_{w}"] - d["block"]).dt.total_seconds().clip(lower=0)
        d[f"exp_before_{w}"] = float(w)
    span = (raw.MVT_TIME_UTC_mvt.max() - raw.MVT_TIME_UTC_mvt.min()).total_seconds()
    n = a.groupby(KEY, sort=False).size().rename("n")
    d["lam"] = pd.to_numeric(pd.Series(d.set_index(KEY).index.map(n)),
                             errors="coerce").fillna(0.0).to_numpy() / span
    d["y_true"] = y.reindex(keep).to_numpy("float64")
    cols = ["MVT_ID_mvt", "n_arr_in_taxi", "n_arr_placebo_full", "lam", "y_true"]
    for w in (900, 1800):
        cols += [f"after_{w}", f"before_{w}", f"exp_after_{w}", f"exp_before_{w}"]
    return d[cols]


def main() -> None:
    parts = []
    for m in D.MONTHS:
        parts.append(counts(D.load_month(m)))
        print(f"  {m}", flush=True)
    c = pd.concat(parts, ignore_index=True)
    n = pd.read_parquet(EXP / "work" / "noac_2025.parquet")
    d = n.merge(c.drop(columns=["y_true"]), on="MVT_ID_mvt", validate="one_to_one")
    d["expected"] = d.y_true.clip(lower=0) * d.lam
    d = d[np.isfinite(d.expected) & (d.expected > 0)]

    def ratio(f):
        return float(f.n_arr_in_taxi.sum() / f.expected.sum())

    def ratio_placebo(f):
        return float(f.n_arr_placebo_full.sum() / f.expected.sum())

    rows = []
    for lab in ["ALL"] + LAB:
        for cls, s in (("placeholder", 1.0), ("rest", 0.0)):
            f = d if lab == "ALL" else d[d.band == lab]
            f = f[f.ph == s]
            if len(f) < 10:
                continue
            pt, lo, hi = day_boot(f, ratio, nboot=2000)
            pp, plo, phi = day_boot(f, ratio_placebo, nboot=2000)
            rows.append({"band": lab, "class": cls, "n": len(f),
                         "y_med": float(f.y_true.median()),
                         "obs": float(f.n_arr_in_taxi.sum()),
                         "exp": float(f.expected.sum()),
                         "obs_over_exp": pt, "lo": lo, "hi": hi,
                         "P_any_arrival": float((f.n_arr_in_taxi > 0).mean()),
                         "placebo_obs": float(f.n_arr_placebo_full.sum()),
                         "placebo_over_exp": pp, "pl_lo": plo, "pl_hi": phi,
                         "P_any_placebo": float((f.n_arr_placebo_full > 0).mean())})
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "mechanism.csv", index=False)

    # the length-matched test
    fx = []
    for w in (900, 1800):
        for lab in ["ALL"] + LAB:
            for cls, sv in (("placeholder", 1.0), ("rest", 0.0)):
                f = d if lab == "ALL" else d[d.band == lab]
                f = f[f.ph == sv]
                if len(f) < 10:
                    continue
                ea = (f[f"exp_after_{w}"] * f.lam).sum()
                eb = (f[f"exp_before_{w}"] * f.lam).sum()
                def ra(z, w=w):
                    return float(z[f"after_{w}"].sum() / max((z[f"exp_after_{w}"] * z.lam).sum(), 1e-12))
                def rb(z, w=w):
                    return float(z[f"before_{w}"].sum() / max((z[f"exp_before_{w}"] * z.lam).sum(), 1e-12))
                pa, alo, ahi = day_boot(f, ra, nboot=2000)
                pb, blo, bhi = day_boot(f, rb, nboot=2000)
                fx.append({"w": w, "band": lab, "class": cls, "n": len(f),
                           "after_obs": float(f[f"after_{w}"].sum()), "after_exp": float(ea),
                           "after_over_exp": pa, "a_lo": alo, "a_hi": ahi,
                           "before_obs": float(f[f"before_{w}"].sum()), "before_exp": float(eb),
                           "before_over_exp": pb, "b_lo": blo, "b_hi": bhi})
    fxt = pd.DataFrame(fx)
    fxt.to_csv(EXP / "analysis" / "mechanism_matched.csv", index=False)
    with pd.option_context("display.width", 200):
        print("\narrivals at the row's own stand inside (BLOCK, take-off], "
              "against the stand's own arrival rate times the taxi time")
        print(t[["band", "class", "n", "y_med", "obs", "exp", "obs_over_exp", "lo", "hi",
                 "P_any_arrival", "placebo_over_exp", "pl_lo", "pl_hi",
                 "P_any_placebo"]].round(3).to_string(index=False))
        print("\nlength-matched: the stand in the w seconds AFTER the block time "
              "against the w seconds BEFORE it")
        print(fxt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
