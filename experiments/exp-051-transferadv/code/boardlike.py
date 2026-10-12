"""exp-051 iteration 1, item 2: the board-like half-day fits (SPEC.md section I1.2).

On fold f, the served head holds the served season's own 2025 month and no fold member does.
The board-like fit adds HALF of fold f's own days to member 1's fitting set and is scored on
the OTHER half.  Stacked in uploaded order, four levels per half:

    L0  member 1 + the half-days
    L1  + 2025-11                       (v20)
    L2  + 2025-10 + 2025-03             (v21)
    L3  + 2025-08 on july, 2025-02 on december and january   (v22, recorded rule's fold rule)

3 folds x 2 halves x 4 levels = 24 fits, one at a time, each write-once.  Everything else is
exp-046's member-1 recipe unchanged: `chainfold.fit_arm`, 255 leaves, HOLD off, the guard cut
from the other two folds, LightGBM seed 46, placebo seed 1.

    .venv/bin/python experiments/exp-051-transferadv/code/boardlike.py --plan
    .venv/bin/python experiments/exp-051-transferadv/code/boardlike.py --reproduce
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-051-transferadv/code/boardlike.py --run

--plan builds every fitting set and prints support and hashes; it fits nothing and prices
nothing.  --reproduce refits the plain L1 member (2025-11 over member 1) on january through
this file's path and compares it to exp-046's saved vector.  --run takes the 24 fits in a
fixed order, skipping any fit whose artifact exists, and holds while the 1-minute load is
above 10.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import common as C
from common import A, CF, NO

LOAD_MAX = 10.0
ORDER = [(f, h, lv) for f in ("january", "december", "july_loo") for h in (0, 1)
         for lv in (0, 1, 2, 3)]


def load1() -> float:
    return os.getloadavg()[0]


def hold_for_load(tag: str) -> float:
    while True:
        la = load1()
        if la <= LOAD_MAX:
            return la
        print(f"  {tag}: 1-minute load {la:.2f} > {LOAD_MAX}, holding 60 s", flush=True)
        time.sleep(60)


def fit_one(ctx: dict, f: str, h: int | None, level: int, overwrite: bool = False) -> dict:
    """One fit.  h=None is the plain path (no half-days), used only by --reproduce."""
    npz, js = C.fit_path(f, h, level)
    NO.refuse_existing([npz, js], overwrite=overwrite)
    pan, ship, scols, hv = ctx["pan"], ctx["ship"], ctx["scols"], ctx["hv"][f]
    parts, prov_rows = [], {}
    if h is not None:
        hr = C.half_fit_rows(pan, ship, f, h, hv)
        parts.append(hr)
        prov_rows["half_rows"] = int(len(hr))
        prov_rows["half_days"] = sorted(pd.to_datetime(hr.day).dt.strftime("%Y-%m-%d").unique())
        prov_rows["half_rows_by_airport"] = hr.airport.astype(str).value_counts().sort_index().to_dict()
        prov_rows["half_ids_sha16"] = C.ids_sha16(hr.MVT_ID_mvt)
        scored_days = set(pd.to_datetime(pan[f].day[hv != h]).dt.strftime("%Y-%m-%d"))
        if scored_days & set(prov_rows["half_days"]):
            raise SystemExit(f"{f}: a fitting half-day is also a scored day")
    ext = C.ext_frame(scols, C.LEVELS[level][f], ctx["cache"])
    if ext is not None:
        prov_rows["ext_months"] = C.LEVELS[level][f]
        prov_rows["ext_rows"] = int(len(ext))
        if parts:
            ext = ext.reindex(columns=parts[0].columns)
        parts.append(ext)
    if not parts:
        raise SystemExit("a fit with neither half-days nor extension is member 1 itself")
    extra = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    setting_was, hold_was = CF.SETTING, CF.HOLD
    CF.SETTING, CF.HOLD = {"num_leaves": C.LEAVES}, ()
    prov: dict = {}
    t1 = time.time()
    try:
        d = CF.fit_arm(pan, prov=prov, extra=extra, folds=(f,))
    finally:
        CF.SETTING, CF.HOLD = setting_was, hold_was
    secs = time.time() - t1
    want = prov_rows.get("half_rows", 0) + prov_rows.get("ext_rows", 0)
    if prov[f]["extra_rows"] != want:
        raise SystemExit(f"{f}: fit_arm carried {prov[f]['extra_rows']} extra rows, not {want}")
    vec = d[f].astype("float64")
    C.WORK.mkdir(parents=True, exist_ok=True)
    NO.refuse_existing([npz], overwrite=overwrite)
    np.savez_compressed(npz, **{f: vec})
    rec = {"fold": f, "half_added": None if h is None else C.HALF_NAME[h],
           "scores_half": None if h is None else C.HALF_NAME[1 - h], "level": level,
           "step": C.STEP.get(level), "rows": prov_rows, "fit_arm_provenance": prov[f],
           "leaves": C.LEAVES, "hold": [], "lgb_seed": A.SEED, "placebo_seed": C.SEED_MEMBER,
           "seconds_fit": round(secs, 1), "load_1min_at_launch": ctx.get("load_at_launch"),
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "vector": str(npz.relative_to(C.ROOT)), "vector_sha256": C.sha256(npz),
           "rows_moved": int((vec != 0).sum())}
    C.write_json(js, rec, overwrite=overwrite)
    print(f"  fitted {npz.name} in {secs:.0f}s: fit rows {prov[f]['total_fit_rows']:,} "
          f"(extra {prov[f]['extra_rows']:,}), moved {rec['rows_moved']:,}", flush=True)
    return rec


def plan(ctx: dict) -> dict:
    pan_ship, pan, ship, scols = ctx["pan_ship"], ctx["pan"], ctx["ship"], ctx["scols"]
    out = {"what": "exp-051 I1.2 plan: support and hashes, no fit, no gain", "folds": {}}
    for f in A.FOLDS:
        hv = ctx["hv"][f]
        days = pd.to_datetime(pan_ship[f].day).dt.strftime("%Y-%m-%d")
        rec = {"rows": int(len(pan_ship[f])), "days": int(days.nunique()),
               "usable_ship": int(pan_ship[f].usable.sum()),
               "usable_member1_panel": int(pan[f].usable.sum()),
               "member1_fit_rows_other_folds": int(sum(pan[g].usable.sum()
                                                       for g in A.FOLDS if g != f))}
        for h in (0, 1):
            hr = C.half_fit_rows(pan, ship, f, h, hv)
            sel = hv == h
            rec[f"half_{C.HALF_NAME[h]}"] = {
                "days": sorted(days[sel].unique()), "n_days": int(days[sel].nunique()),
                "fold_rows": int(sel.sum()),
                "usable_rows_ship": int((sel & pan_ship[f].usable.to_numpy(bool)).sum()),
                "fit_rows_member1_panel": int(len(hr)),
                "fit_rows_by_airport": hr.airport.astype(str).value_counts().sort_index().to_dict(),
                "fit_ids_sha16": C.ids_sha16(hr.MVT_ID_mvt)}
        rec["ext_rows_by_level"] = {}
        for lv in (1, 2, 3):
            e = C.ext_frame(scols, C.LEVELS[lv][f], ctx["cache"])
            rec["ext_rows_by_level"][lv] = int(len(e))
        rec["total_fit_rows_by_half_level"] = {
            f"{C.HALF_NAME[h]}-L{lv}": rec["member1_fit_rows_other_folds"]
            + rec[f"half_{C.HALF_NAME[h]}"]["fit_rows_member1_panel"]
            + (rec["ext_rows_by_level"][lv] if lv else 0)
            for h in (0, 1) for lv in (0, 1, 2, 3)}
        out["folds"][f] = rec
    # #4's halves (withinsplit) were built on the six-airport class rows; say whether they
    # are the same day split as the one used here, for december and january
    for f in ("december", "january"):
        ws = json.loads((C.E46 / f"analysis/withinsplit-{f}.json").read_text())["plan"]
        E = pan_ship[f][pan_ship[f].usable.to_numpy(bool)
                        & pan_ship[f].airport.astype(str).isin(ws["airports"]).to_numpy()]
        hE = WS_halves_map(E.day)
        hF = WS_halves_map(pan_ship[f].day)
        out["folds"][f]["same_day_split_as_withinsplit"] = bool(hE == {k: hF[k] for k in hE})
    out["extension_month_rows"] = {mo: int(len(v)) for mo, v in ctx["cache"].items()}
    # the saved plain vectors: sha256 and the identity checks that say which stack each is
    pl = {lv: C.load_vec(p) for lv, p in C.PLAIN.items()}
    out["plain"] = {lv: {"path": str(p.relative_to(C.ROOT)), "sha256": C.sha256(p)}
                    for lv, p in C.PLAIN.items()}
    p0308 = C.load_vec(C.E46 / "work/d-onemonth-twomonth-plus0308-255-free.npz")
    p0302 = C.load_vec(C.E46 / "work/d-onemonth-twomonth-plus0302-255-free.npz")
    out["plain_identity"] = {
        "seasonal_m1_july_eq_plus0308_july": bool(np.array_equal(pl[3]["july_loo"], p0308["july_loo"])),
        "seasonal_m1_dec_eq_plus0302_dec": bool(np.array_equal(pl[3]["december"], p0302["december"])),
        "seasonal_m1_jan_eq_plus0302_jan": bool(np.array_equal(pl[3]["january"], p0302["january"])),
        "max_abs_seasonal_m1_minus_plus0308_july": float(np.abs(pl[3]["july_loo"] - p0308["july_loo"]).max()),
        "max_abs_seasonal_m1_minus_plus0302_jan": float(np.abs(pl[3]["january"] - p0302["january"]).max())}
    return out


def WS_halves_map(days: pd.Series) -> dict:
    d = pd.to_datetime(days).dt.strftime("%Y-%m-%d")
    h = C.WS.halves(days)
    return dict(zip(d.to_numpy(), h))


def context() -> dict:
    pan_ship, pan, ship, scols = C.build_panels()
    return {"pan_ship": pan_ship, "pan": pan, "ship": ship, "scols": scols,
            "hv": {f: C.half_of_rows(pan_ship, f) for f in A.FOLDS}, "cache": {}}


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--plan", action="store_true")
    ps.add_argument("--reproduce", action="store_true")
    ps.add_argument("--run", action="store_true")
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    if a.plan:
        out_p = C.ANALYSIS / "plan.json"
        NO.refuse_existing([out_p], overwrite=a.overwrite)
        ctx = context()
        out = plan(ctx)
        C.write_json(out_p, out, overwrite=a.overwrite)
        print(json.dumps({f: {k: v for k, v in r.items() if not k.startswith("half_")}
                          for f, r in out["folds"].items()}, indent=1, default=str))
        for f, r in out["folds"].items():
            for h in ("A", "B"):
                x = r[f"half_{h}"]
                print(f"{f} half {h}: {x['n_days']} days, fold rows {x['fold_rows']:,}, usable "
                      f"{x['usable_rows_ship']:,}, fit rows {x['fit_rows_member1_panel']:,} "
                      f"ids {x['fit_ids_sha16']}")
        print(json.dumps(out["plain_identity"], indent=1))
        print(f"PLAN ONLY -- written {out_p.relative_to(C.ROOT)} ({time.time()-t0:.0f}s)")
        return
    if a.reproduce:
        npz, js = C.fit_path("january", None, 1)
        NO.refuse_existing([npz, js], overwrite=a.overwrite)
        ctx = context()
        ctx["load_at_launch"] = hold_for_load("reproduce")
        rec = fit_one(ctx, "january", None, 1, overwrite=a.overwrite)
        got = np.load(npz)["january"].astype("float64")
        want = C.load_vec(C.PLAIN[1])["january"]
        diff = np.abs(got - want)
        rep = {"what": "I1.2 integrity: plain L1 (2025-11 over member 1) refitted on january "
                       "through boardlike.fit_one, against exp-046's saved vector",
               "saved": str(C.PLAIN[1].relative_to(C.ROOT)), "saved_sha256": C.sha256(C.PLAIN[1]),
               "refit": rec["vector"], "refit_sha256": rec["vector_sha256"],
               "bit_identical": bool(np.array_equal(got, want)),
               "max_abs_diff_s": float(diff.max()), "rows_differing": int((diff > 0).sum()),
               "rows": int(len(got)), "passes_1e-9": bool(diff.max() <= 1e-9)}
        C.write_json(C.ANALYSIS / "reproduce-plain-L1-january.json", rep, overwrite=a.overwrite)
        print(json.dumps(rep, indent=1))
        return
    if a.run:
        ctx = context()
        for f, h, lv in ORDER:
            npz, js = C.fit_path(f, h, lv)
            if npz.exists() and js.exists():
                print(f"skip {npz.name}: exists", flush=True)
                continue
            ctx["load_at_launch"] = hold_for_load(npz.name)
            print(f"launch {npz.name} at load {ctx['load_at_launch']:.2f} "
                  f"({time.strftime('%H:%M:%S', time.gmtime())} UTC)", flush=True)
            fit_one(ctx, f, h, lv, overwrite=a.overwrite)
        print(f"all {len(ORDER)} fits present ({time.time()-t0:.0f}s)")
        return
    raise SystemExit("pass --plan, --reproduce or --run")


if __name__ == "__main__":
    main()
