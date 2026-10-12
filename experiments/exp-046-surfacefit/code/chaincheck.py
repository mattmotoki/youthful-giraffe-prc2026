"""The chain rows' SECOND independent check (desk, 2026-09-09, with the research addendum).

`analysis/chain-spec.md` shipped 3,179 turnaround placements whose key ends in a +-60 s
take-off agreement.  **8,478 pairs satisfy every other clause of the key and fail that
one**; section 21.4 recorded them as a MAYBE whose revival is a second independent check.
This is that check.

Everything here is INDEPENDENT of the key by construction.  The key reads callsigns, stands,
operator, type and two clocks (the arrival's in-block and the take-off).  It never reads a
POSITION and never reads the off-block clock of the departure, so all three checks below are
new evidence rather than the key restated:

  (0) **Why the take-off check failed.**  A pair whose wheels-off is poorly observed fails
      the clause for an instrument reason; a pair whose wheels-off is WELL observed and
      still disagrees is evidence AGAINST the placement and leaves the pool.  This runs
      first, because a check applied to the second class would be measuring the wrong thing.
  (i) **The off-block clock.**  The segment's own off-block P against the placed
      departure's `AOBT_3_flt` A.  Label-free -- `mask_like_ranking` blanks
      `BLOCK_TIME_UTC_mvt` and the target on departure rows and leaves `AOBT_3_flt`, which
      is present on 98.5 % of ranking departures and is a NOISY off-block (RMS 466 s
      against the label on 2025-07, corr 0.51), already used in this lane as a label-free
      clock (`code/janwarn.py`).
  (ii) **Runway agreement.**  The wheels-off POSITION implies a runway; compare with
      `RUNWAY_mvt` of the placed departure.  The runway centroids are learned from the
      pass-1/2 matched pairs, so no runway database enters and nothing is fitted to the
      chain rows.
  (iii) **Stand-position agreement.**  Learn `STAND_mvt -> segment start position` from the
      pass-1/2 matched pairs, then test the unclaimed segment's own start position against
      the placed departure's stand.

Every threshold is set on the SHIPPED 3,179 and every rate is reported against the two
placebos of `code/chainpass.py` -- the day placebo (departures moved +24 h, segments left
where they are) and the stand placebo (`STAND_mvt` permuted inside each airport).

The pool is reconstructed with a PROBE COPY of the shipped matcher
(`analysis/chain-check/block_probe.py`), which differs from `code/block.py` by fourteen
added lines and nothing removed, so the pass logic is identical by construction rather than
by agreement.  The first thing this file does is reconcile its gated count against the
shipped reference and stop if it disagrees.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/chaincheck.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arms as A                                                      # noqa: E402
import prc.data as D                                                  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "block_probe", EXP / "analysis/chain-check/block_probe.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["block_probe"] = B
_spec.loader.exec_module(B)

RANK = ["2026-01", "2026-07"]
#: the shipped counts the probe must reproduce before anything else runs
REFERENCE = {"2026-01": 1382, "2026-07": 1797}
UNGATED_REFERENCE = 8478
PLACEBO_SHIFT_H = 24
SEED = 46
OUT = EXP / "analysis/chain-check.json"

SEG_COLS = ["seg_key", "icao", "obs_offblock", "wheels_lat", "wheels_lon",
            "stand_lat", "stand_lon", "seg_points", "n_ground_points",
            "seg_gap_max_s", "seg_dt_median_s", "seg_span_s", "t_airborne",
            "ends_airborne", "airborne_alt_ok", "wheels_gs_kt"]
COH_COLS = ["MVT_ID_mvt", "aobt_s", "RUNWAY_mvt"]


def prefix_train() -> pd.DataFrame:
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in D.MONTHS], ignore_index=True)
    return tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")


def km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km, vectorised."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def run(dep: pd.DataFrame, tr: pd.DataFrame, arr: pd.DataFrame, rule: str) -> dict:
    """One build, and everything the probe stashed, enriched and split."""
    B.PROBE.clear()
    blk = B.build(dep, tr, require_rule=rule, arrivals=arr)
    pr = B.PROBE
    seg, coh = pr.get("seg"), pr.get("coh")
    pairs = pr.get("pairs")
    j12 = pr.get("j12")
    if pairs is None or not len(pairs):
        return {"blk": blk, "pairs": pd.DataFrame(), "j12": j12, "seg": seg, "coh": coh,
                "report": blk.attrs.get("turnaround", {})}
    p = pairs.merge(seg[SEG_COLS], on="seg_key", how="left", validate="one_to_one")
    p = p.merge(coh[COH_COLS], on="MVT_ID_mvt", how="left", validate="one_to_one")
    p["dt_takeoff"] = p.obs_wheels_off - p.mvt_s
    p["gated"] = p.dt_takeoff.abs() <= B.CHECK_S
    return {"blk": blk, "pairs": p, "j12": j12, "seg": seg, "coh": coh,
            "report": blk.attrs.get("turnaround", {})}


def learn_maps(j12: pd.DataFrame, seg: pd.DataFrame, coh: pd.DataFrame) -> dict:
    """Runway centroids and stand centroids, from the PASS-1/2 matched pairs only.

    Nothing about the chain rows enters, and no external database: the position of a
    runway is where matched departures were seen to leave the ground, and the position of
    a stand is where matched departures were seen to start moving.
    """
    if j12 is None or not len(j12):
        return {"runway": {}, "stand": {}}
    m = j12[["seg_key", "MVT_ID_mvt"]].merge(
        seg[["seg_key", "airport", "wheels_lat", "wheels_lon", "stand_lat", "stand_lon"]],
        on="seg_key", how="left")
    m = m.merge(coh[["MVT_ID_mvt", "RUNWAY_mvt", "STAND_mvt"]], on="MVT_ID_mvt", how="left")
    rw, st = {}, {}
    g = m.dropna(subset=["airport", "RUNWAY_mvt", "wheels_lat", "wheels_lon"])
    for (a, r), q in g.groupby(["airport", "RUNWAY_mvt"], observed=True):
        if len(q) >= 20:                       # a runway nobody used is not a runway
            rw.setdefault(str(a), {})[str(r)] = (float(q.wheels_lat.median()),
                                                 float(q.wheels_lon.median()), int(len(q)))
    g = m.dropna(subset=["airport", "STAND_mvt", "stand_lat", "stand_lon"])
    for (a, sd), q in g.groupby(["airport", "STAND_mvt"], observed=True):
        if len(q) >= 3:
            st.setdefault(str(a), {})[str(sd)] = (float(q.stand_lat.median()),
                                                  float(q.stand_lon.median()), int(len(q)))
    return {"runway": rw, "stand": st}


def apply_checks(p: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """The three checks, as columns.  No threshold is applied here."""
    p = p.copy()
    if not len(p):
        return p
    # (i) the off-block clock: the segment's own P against the departure's AOBT
    p["dt_offblock"] = p.obs_offblock - p.aobt_s
    p["abs_dt_offblock"] = p.dt_offblock.abs()
    # (ii) the runway the wheels-off position implies, and whether it is the row's runway
    imp, dist = [], []
    for a, la, lo in zip(p.airport.astype(str), p.wheels_lat, p.wheels_lon):
        cand = maps["runway"].get(a)
        if not cand or not np.isfinite(la) or not np.isfinite(lo):
            imp.append(None); dist.append(np.nan); continue
        names = list(cand)
        d = km(la, lo, np.array([cand[n][0] for n in names]),
               np.array([cand[n][1] for n in names]))
        k = int(np.argmin(d))
        imp.append(names[k]); dist.append(float(d[k]))
    p["runway_implied"] = imp
    p["runway_implied_km"] = dist
    p["runway_agrees"] = (p.runway_implied.notna() & p.RUNWAY_mvt.notna()
                          & (p.runway_implied.astype(str) == p.RUNWAY_mvt.astype(str)))
    # (iii) the stand the departure claims, against where the segment actually started
    sd = []
    for a, s_, la, lo in zip(p.airport.astype(str), p.STAND_mvt.astype(str),
                             p.stand_lat, p.stand_lon):
        c = maps["stand"].get(a, {}).get(s_)
        sd.append(float(km(la, lo, c[0], c[1])) if c and np.isfinite(la) and np.isfinite(lo)
                  else np.nan)
    p["stand_km"] = sd
    # (0) how well the wheels-off itself is observed -- the split the addendum asks for
    p["wheels_gap_s"] = p.t_airborne - p.obs_wheels_off
    return p


def rates(p: pd.DataFrame, col: str, thr: float, how: str) -> dict:
    if not len(p):
        return {"rows": 0, "with_the_field": 0, "pass_rate": None}
    v = p[col].to_numpy("float64") if how != "eq" else None
    if how == "le":
        ok, have = (v <= thr), np.isfinite(v)
    elif how == "ge":
        ok, have = (v >= thr), np.isfinite(v)
    else:
        ok = p[col].to_numpy(bool); have = np.ones(len(p), bool)
    return {"rows": int(len(p)), "with_the_field": int(have.sum()),
            "passes": int((ok & have).sum()),
            "pass_rate": (float((ok & have).sum() / have.sum()) if have.sum() else None)}


def main() -> None:
    rk = D.load_ranking()
    rk["_m"] = rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    tr = prefix_train()
    rule, _ = A.serving_rule()
    rng = np.random.default_rng(SEED)
    out = {"what": __doc__.strip().splitlines()[0],
           "probe": "analysis/chain-check/block_probe.py",
           "matcher_sha256": A.MATCHER_SHA,
           "reference": REFERENCE, "months": {}}

    real, plac = {}, {"day": {}, "stand": {}}
    for m in RANK:
        dep = rk[rk.PHASE_mvt.eq("DEP") & rk._m.eq(m)].drop(columns="_m").copy()
        arr = rk[rk.PHASE_mvt.eq("ARR") & rk._m.eq(m)]
        r = run(dep, tr, arr, rule)
        got = int(r["report"].get("placed", -1))
        if got != REFERENCE[m]:
            raise SystemExit(f"{m}: the probe placed {got}, the shipped block placed "
                             f"{REFERENCE[m]} -- the probe is not the shipped pass")
        gated = int(r["pairs"].gated.sum())
        if gated != REFERENCE[m]:
            raise SystemExit(f"{m}: the probe's gated pairs {gated} != {REFERENCE[m]}")
        real[m] = r
        print(f"{m}: reconciled -- placed {got}, pairs {len(r['pairs'])}, "
              f"gated {gated}, ungated {len(r['pairs']) - gated}", flush=True)

        sh = dep.copy()
        sh["MVT_TIME_UTC_mvt"] = sh.MVT_TIME_UTC_mvt + pd.Timedelta(hours=PLACEBO_SHIFT_H)
        plac["day"][m] = run(sh, tr, arr, rule)
        st = dep.copy()
        st["STAND_mvt"] = (st.groupby("ADEP_mvt", group_keys=False).STAND_mvt
                             .apply(lambda s: pd.Series(rng.permutation(s.to_numpy()),
                                                        index=s.index)))
        plac["stand"][m] = run(st, tr, arr, rule)
        print(f"   placebos: day placed {plac['day'][m]['report'].get('placed')}, "
              f"stand placed {plac['stand'][m]['report'].get('placed')}", flush=True)

    tot_g = sum(int(real[m]["pairs"].gated.sum()) for m in RANK)
    tot_u = sum(int((~real[m]["pairs"].gated).sum()) for m in RANK)
    out["pool"] = {"gated_shipped": tot_g, "ungated": tot_u,
                   "ungated_reference": UNGATED_REFERENCE}
    print(f"\nPOOL: gated {tot_g} (reference 3,179), ungated {tot_u} "
          f"(reference {UNGATED_REFERENCE})", flush=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # maps and checks, per month, on the real and both placebos
    for m in RANK:
        mp = learn_maps(real[m]["j12"], real[m]["seg"], real[m]["coh"])
        real[m]["maps"] = mp
        real[m]["p"] = apply_checks(real[m]["pairs"], mp)
        for lab in ("day", "stand"):
            # the MAP is the real one: a placebo corrupts the placement, not the geography
            plac[lab][m]["p"] = apply_checks(plac[lab][m]["pairs"], mp)
        out["months"].setdefault(m, {})["maps"] = {
            "runways_learned": {a: len(v) for a, v in mp["runway"].items()},
            "stands_learned": {a: len(v) for a, v in mp["stand"].items()}}
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    P = pd.concat([real[m]["p"].assign(month=m) for m in RANK], ignore_index=True)
    PL = {lab: pd.concat([plac[lab][m]["p"].assign(month=m) for m in RANK],
                         ignore_index=True) for lab in ("day", "stand")}
    P.to_parquet(EXP / "work/chain-pairs.parquet", index=False)
    for lab, q in PL.items():
        q.to_parquet(EXP / f"work/chain-pairs-placebo-{lab}.parquet", index=False)
    out["frames"] = {"real": "work/chain-pairs.parquet",
                     "placebo": "work/chain-pairs-placebo-{day,stand}.parquet"}
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("frames written; shipped rows", int(P.gated.sum()),
          "ungated", int((~P.gated).sum()),
          "| day placebo pairs", len(PL["day"]), "stand placebo pairs", len(PL["stand"]))


if __name__ == "__main__":
    main()
