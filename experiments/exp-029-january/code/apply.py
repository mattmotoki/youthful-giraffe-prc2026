"""Serve the envelope correction on top of any ranking-side vector.

    corrected = max(pred + c(exceedance), FLOOR)          c(x) = 0 for x <= 0

The correction is a function of a SERVED VECTOR and a label-free per-row
exceedance, so it composes: it takes stack-5 today and the exp-028 window arm
later without change.

**It is measured and it is not adopted.**  `RESULTS.md` §3 reports the whole
lever at +0.05 to +0.07 s of board estimate on stack-5, against the queue's
1-2 s, because the under-prediction it repairs has fallen from -143.5 s on
exp-014's candidate 1 to -41.0 s on stack-5.  This file exists so the number
can be re-read the hour a base with a larger envelope-day bias appears, and so
the flag itself -- which is the durable part -- has a serve-time address.

`noac` rows are held at the base.  The desk's BOARD READING 3 (2026-09-07, on
v3 = stack-5 at 264.3861) quotes every candidate as clean-with-noac-at-the-
anchor + 4.28, because bulk fold gains transfer to the board and `noac` fold
gains do not, so a correction credited on a `noac` row is credit the board will
not pay.  On this ranking file the guard moves nothing -- **zero of the 388
`noac` rows is flagged**, every one sits on an in-envelope airport-day
(`RESULTS.md` §7a) -- and it is here so the object stays correct under the
currency if a later month does put one on a flagged day.  `--credit-noac`
restores the old behaviour.

Usage
-----
    python apply.py --pred <parquet with MVT_ID_mvt,y_pred> --out <parquet>
                    [--arm step|step-half|linear-capped] [--source auto|raw|committed]
                    [--credit-noac]

`--source raw` derives the flag from `data/raw/ranking.parquet` and is the
authority; `--source committed` uses `oof/ranking_flag.parquet`, which this
machine can reach without the data; `auto` (the default) uses raw when the data
directory resolves and committed otherwise, and when BOTH are available it
cross-checks them row by row and aborts on any disagreement.

Why `linear` is not offered: fitted on the five 2025 analogue days its slope is
NEGATIVE on the january rotation (-0.489 s per second of exceedance), so at
January 2026's largest exceedance of 1,350 s it serves -603 s of correction on
a day where the 2025 evidence says the base under-predicts. `linear-capped`
clips to the fitting days' own optimal constants and is the only linear form
that may be served at all. `RESULTS.md` §2 carries the table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import correct as C  # noqa: E402
import envelope as E  # noqa: E402
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

FLAG = EXP / "oof" / "ranking_flag.parquet"


# ------------------------------------------------------------------ the flag -
def flag_from_raw() -> pd.DataFrame:
    """`MVT_ID_mvt`, airport-day, and exceedance, derived from the ranking file.

    The envelope is the maximum daily mean of `MVT_TIME_UTC_mvt - AOBT_3_flt`
    over the TWELVE 2025 months, per airport, because the served fit sees all
    twelve.  When the 2025 months are on disk it is recomputed from them; when
    they are not it is read from exp-014's committed twelve-month day table,
    which is the same quantity from the same rows.
    """
    rk = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
                                 "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
    d = rk[rk["PHASE_mvt"] == "DEP"].copy()
    d["apt"] = d["ADEP_mvt"].astype(str)
    d["day"] = d["MVT_TIME_UTC_mvt"].dt.floor("D")
    d["span"] = (d["MVT_TIME_UTC_mvt"] - d["AOBT_3_flt"]).dt.total_seconds()
    day = d.groupby(["apt", "day"], observed=True)["span"].mean().rename("day_span_mean")
    try:
        env = _envelope_from_months()
        src = "the twelve 2025 monthly files"
    except Exception as exc:                                    # noqa: BLE001
        env = E.days_2025().groupby("apt")["day_span_mean"].max().rename("envelope")
        src = f"exp-014's committed twelve-month day table ({type(exc).__name__})"
    print(f"[flag] envelope from {src}", flush=True)
    out = d[["MVT_ID_mvt", "apt", "day"]].merge(day, on=["apt", "day"], how="left")
    out = out.merge(env, left_on="apt", right_index=True, how="left")
    out["exceedance"] = out["day_span_mean"] - out["envelope"]
    out["flagged"] = out["exceedance"] > 0
    out["month"] = out["day"].dt.strftime("%Y-%m")
    return out


def _envelope_from_months() -> pd.Series:
    rows = []
    for m in D.MONTHS:
        f = D.load_month(m, columns=["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                     "AOBT_3_flt"])
        f = f[f["PHASE_mvt"] == "DEP"]
        rows.append(pd.DataFrame({
            "apt": f["ADEP_mvt"].astype(str).to_numpy(),
            "day": f["MVT_TIME_UTC_mvt"].dt.floor("D").to_numpy(),
            "span": (f["MVT_TIME_UTC_mvt"] - f["AOBT_3_flt"]).dt.total_seconds().to_numpy()}))
    t = pd.concat(rows, ignore_index=True)
    day = t.groupby(["apt", "day"], observed=True)["span"].mean()
    return day.groupby("apt").max().rename("envelope")


def flag_from_committed() -> pd.DataFrame:
    if not FLAG.exists():
        raise SystemExit(f"{FLAG} is missing; run code/ranking_flag.py first")
    return pd.read_parquet(FLAG)


def get_flag(source: str) -> pd.DataFrame:
    have_raw = True
    try:
        D.data_dir()
    except Exception:                                           # noqa: BLE001
        have_raw = False
    if source == "raw" or (source == "auto" and have_raw):
        raw = flag_from_raw()
        if FLAG.exists():
            com = flag_from_committed()
            j = raw.merge(com[["MVT_ID_mvt", "exceedance"]], on="MVT_ID_mvt",
                          how="outer", suffixes=("", "_c"), indicator=True)
            bad = int((j["_merge"] != "both").sum())
            gap = float(np.nanmax(np.abs(j["exceedance"] - j["exceedance_c"])))
            print(f"[flag] cross-check against the committed flag: {bad} unmatched "
                  f"rows, max |delta exceedance| {gap:.3e} s", flush=True)
            if bad or gap > 1e-6:
                raise SystemExit("the raw and committed flags disagree")
        return raw
    if source == "raw":
        raise SystemExit("--source raw asked for, but the data directory does not resolve")
    return flag_from_committed()


def noac_guard(fl: pd.DataFrame, c: np.ndarray) -> tuple:
    """Zero the correction on rows with no `AIRCRAFT_TYPE_mvt`.

    BOARD READING 3: the board does not pay for movement on the `noac` rows, so
    a served correction must not claim any.  Needs the raw ranking file for the
    aircraft type; without it the guard cannot run and the caller is told, since
    silently serving an uncorrected-for-noac vector would be the very thing the
    reading forbids.
    """
    try:
        rk = D.load_ranking()[["MVT_ID_mvt", "AIRCRAFT_TYPE_mvt"]]
    except Exception as e:                                      # noqa: BLE001
        raise SystemExit(f"the noac guard needs data/raw/ranking.parquet ({e}); "
                         "pass --credit-noac to serve without it, knowingly")
    rk = rk.assign(MVT_ID_mvt=rk["MVT_ID_mvt"].astype("float64"))
    m = fl[["MVT_ID_mvt"]].merge(rk, on="MVT_ID_mvt", how="left", validate="one_to_one")
    noac = m["AIRCRAFT_TYPE_mvt"].isna().to_numpy()
    hit = int((noac & (c != 0.0)).sum())
    return np.where(noac, 0.0, c), hit


# ---------------------------------------------------------------- the fit ----
def fit(arm: str, winsor: bool = True) -> dict:
    """Fit the correction on ALL five 2025 analogue days -- the serve-time fit.

    There is no held-out fold at serve time, so the served parameters use every
    analogue day.  `analysis/arms.csv` reports the same shapes fitted with a
    fold held out, and those are the numbers the gate is read on.
    """
    src = EXP / "analysis" / "fold_rows.parquet"
    if not src.exists():
        raise SystemExit(f"{src.relative_to(ROOT)} is git-ignored (18 MB); run "
                         f"code/envelope.py first, it rebuilds it in about 30 s")
    R = pd.read_parquet(src)
    t = C.day_targets(R[R["flagged"].to_numpy()], winsor)
    if arm == "step":
        return {"arm": arm, "kind": "step", **C.fit_step(t)}
    if arm == "step-half":
        return {"arm": arm, "kind": "step", **C.fit_step(t, shrink=0.5)}
    if arm == "linear-capped":
        return {"arm": arm, "kind": "linear", **C.fit_linear(t, cap=True)}
    raise SystemExit(f"unknown arm {arm!r}")


def correction(p: dict, x: np.ndarray) -> np.ndarray:
    return C.apply_step(p, x) if p["kind"] == "step" else C.apply_linear(p, x)


# ---------------------------------------------------------------------- run --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, help="parquet with MVT_ID_mvt and y_pred")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--arm", default="step-half",
                    choices=["step", "step-half", "linear-capped"])
    ap.add_argument("--source", default="auto", choices=["auto", "raw", "committed"])
    ap.add_argument("--raw-residual", action="store_true",
                    help="fit on unwinsorised residuals")
    ap.add_argument("--credit-noac", action="store_true",
                    help="correct the noac rows too (pre-BOARD-READING-3 behaviour)")
    a = ap.parse_args()

    p = fit(a.arm, winsor=not a.raw_residual)
    print(f"[fit] {json.dumps(p)}", flush=True)
    fl = get_flag(a.source)
    x = fl["exceedance"].to_numpy("float64")
    c = correction(p, x)
    if not a.credit_noac:
        c, n0 = noac_guard(fl, c)
        print(f"[noac] {n0:,} flagged noac rows held at the base "
              f"(BOARD READING 3); --credit-noac to correct them", flush=True)
    fl = fl.assign(correction=c)

    t = (fl[fl["correction"] != 0].groupby(["month", "apt", "day"], observed=True)
         .agg(rows=("MVT_ID_mvt", "size"), exceedance=("exceedance", "first"),
              c=("correction", "first")).reset_index())
    t["total_s"] = t["rows"] * t["c"]
    print(f"\n[touch] {int(t['rows'].sum()):,} of {len(fl):,} ranking rows "
          f"({t['rows'].sum()/len(fl):.2%}) across {len(t)} airport-days; "
          f"total {t['total_s'].sum():,.0f} s")
    print(t.groupby("month")[["rows", "total_s"]].sum().to_string())

    if a.pred is None:
        print("\n[dry run] no --pred given; the flag and the fit are printed above")
        return
    pred = pd.read_parquet(a.pred)
    m = pred.merge(fl[["MVT_ID_mvt", "exceedance", "correction"]], on="MVT_ID_mvt",
                   how="left", validate="one_to_one")
    if m["correction"].isna().any():
        raise SystemExit(f"{int(m['correction'].isna().sum())} predictions have no flag")
    out = pred.copy()
    out["y_pred"] = np.maximum(pred["y_pred"].to_numpy("float64")
                               + m["correction"].to_numpy("float64"), B.FLOOR)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    moved = int((out["y_pred"].to_numpy() != pred["y_pred"].to_numpy()).sum())
    print(f"\n[write] {a.out}  {len(out):,} rows, {moved:,} changed, "
          f"sha256 {D.sha256(a.out)}")


if __name__ == "__main__":
    main()
