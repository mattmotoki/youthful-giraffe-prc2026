"""exp-051 iteration 2: the June carrier on july_loo (SPEC.md section I2).

Iteration 1's carrier was half of the scored month's own days.  It crushed v20 on every fold,
and the zero-fit distance probe showed its strength is month-level, not day adjacency.  The
served head's same-season month is a WHOLE month of the same season that is NOT the scored
rows' own month (2025-07 for 2026-07).  The fold-side stand-in used here is 2025-06: a whole
same-season month (c1 from July), built, in no uploaded step, sharing no day with July.

Only july_loo gets a carrier.  january's member already holds a c1 month (2025-12), and its
other c1 month (2025-02) is v22's own step month; december likewise holds 2025-01.  Those
folds take the plain member of the same seed.

Arms, all fitted on july_loo only (`folds=("july_loo",)`):

    C  carrier, members s = 1, 2, 3:  member s + 2025-06, then the steps in uploaded order
         C0 = +06   C1 = +06 +11   C2 = +06 +11 +10 +03   C3 = +06 +11 +10 +03 +08
       C0 at s = 1 is exp-046's d-onemonth-only202506 (reproduced first, see --integrity)
    S  sham, member 1: as C, with 2025-06's rbase permuted within airport (seed 5103)
    June, record only, member 1:  P4 = P3 + 06 (06 last)   and   BL4 = BL3 + 06 on both halves

    .venv/bin/python experiments/exp-051-transferadv/code/carrier.py --plan
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-051-transferadv/code/carrier.py --run

--run takes the integrity fits first and stops if either misses 1e-9 s; then the rest in a
fixed order, skipping any fit whose artifact exists, holding while the 1-minute load is > 10.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

import common as C
from common import A, BA, CF, CP, NO, R
import memberspread as MS

JUNE = "2025-06"
JUNE_ROWS = 98_396                    # exp-046 section 47.71, the row count at the fit
SHAM_SEED = 5103
LOAD_MAX = 10.0
F = "july_loo"
STEPS_JULY = {0: [], 1: ["2025-11"], 2: ["2025-11", "2025-10", "2025-03"],
              3: ["2025-11", "2025-10", "2025-03", "2025-08"]}
ONLY0306 = C.E46 / "work/d-onemonth-only202506-255-free.npz"


def panels(seeds: list) -> dict:
    """The shipped panel and member s's placebo-s fitting panel for each seed, exactly as
    `onemonthbag.main` builds a member: rng seed s over the fold months in A.FOLDS order."""
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        mth = A.FOLD_MONTH[f]
        b = CF.build_month(mth, A.prefix_months_for(mth), rule)
        R.assert_mode3(b, mth, "shipped")
        blk[mth] = b
    pan_ship = CF.panel_for(blk, R.THR, {mth: CF.GAP_CUT_S for mth in blk})
    ship, scols = A.shipped_cols(pan_ship)
    pans = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        blocks = {}
        for mth, b in blk.items():
            pb, _ = CP.placebo_block(b, R.THR, CF.GAP_CUT_S, CP.MOVED[mth], rng)
            blocks[mth] = pb
        real = A.month_block
        A.month_block = (lambda bs: (lambda month, prefix_months, rule=None: bs[month]))(blocks)
        try:
            pans[s] = A.panel_with_block(BA.load())
        finally:
            A.month_block = real
        for f in A.FOLDS:
            if not np.array_equal(pans[s][f].MVT_ID_mvt.to_numpy("float64"),
                                  pan_ship[f].MVT_ID_mvt.to_numpy("float64")):
                raise SystemExit(f"seed {s} {f}: the member panel is not in the shipped order")
    return {"pan_ship": pan_ship, "pans": pans, "ship": ship, "scols": scols, "cache": {},
            "hv": C.half_of_rows(pan_ship, F)}


def june_frame(ctx: dict, sham: bool) -> tuple[pd.DataFrame, dict]:
    j = C.ext_frame(ctx["scols"], [JUNE], ctx["cache"])
    if len(j) != JUNE_ROWS:
        raise SystemExit(f"{JUNE} gives {len(j)} usable rows, not section 47.71's {JUNE_ROWS}")
    info = {"june_rows": int(len(j)), "june_ids_sha16": C.ids_sha16(j.MVT_ID_mvt)}
    if not sham:
        return j, info
    j = j.copy()
    rb = j.rbase.to_numpy("float64").copy()
    rng = np.random.default_rng(SHAM_SEED)
    air = j.airport.astype(str).to_numpy()
    new = rb.copy()
    for a in np.unique(air):
        idx = np.flatnonzero(air == a)
        new[idx] = rb[idx][rng.permutation(len(idx))]
    for a in np.unique(air):
        idx = air == a
        if not np.array_equal(np.sort(rb[idx]), np.sort(new[idx])):
            raise SystemExit(f"sham: airport {a}'s rbase values changed, not just their order")
    j["rbase"] = new
    j["y_true"] = j.pbase.to_numpy("float64") + new
    info.update({"sham_seed": SHAM_SEED, "sham_permuted_within": "airport",
                 "sham_corr_with_true_rbase": float(np.corrcoef(rb, new)[0, 1]),
                 "sham_rbase_sha16": hashlib.sha256(new.tobytes()).hexdigest()[:16]})
    return j, info


def paths(tag: str):
    return C.WORK / f"i2-{tag}.npz", C.ANALYSIS / "fits2" / f"i2-{tag}.json"


def plan_fits() -> list:
    """(tag, seed, kind, level, half) in the order they are taken."""
    out = [("integrity-C0-s1", 1, "C", 0, None), ("integrity-P1-s2", 2, "P", 1, None)]
    out += [(f"C-s1-L{L}", 1, "C", L, None) for L in (1, 2, 3)]
    for s in (2, 3):
        out += [(f"C-s{s}-L{L}", s, "C", L, None) for L in (0, 1, 2, 3)]
    out += [(f"S-s1-L{L}", 1, "S", L, None) for L in (0, 1, 2, 3)]
    out += [("June-P4-s1", 1, "P4", 4, None), ("June-BL4-A", 1, "BL4", 4, 0),
            ("June-BL4-B", 1, "BL4", 4, 1)]
    return out


def tag_of(kind: str, s: int, level: int) -> str:
    return f"{kind}-s{s}-L{level}"


def fitting_extra(ctx: dict, s: int, kind: str, level: int, half) -> tuple[pd.DataFrame, dict]:
    rows = {}
    parts = []
    if kind in ("C", "S"):
        j, info = june_frame(ctx, sham=(kind == "S"))
        rows.update(info)
        parts.append(j)
        months = STEPS_JULY[level]
    elif kind == "P":                                        # plain member s, the integrity fit
        months = STEPS_JULY[level]
    elif kind == "P4":                                       # June appended last, plain
        months = STEPS_JULY[3] + [JUNE]
    elif kind == "BL4":                                      # iteration 1's half-days, + June last
        hr = C.half_fit_rows(ctx["pans"][s], ctx["ship"], F, half, ctx["hv"])
        rows["half_rows"] = int(len(hr))
        rows["half_ids_sha16"] = C.ids_sha16(hr.MVT_ID_mvt)
        parts.append(hr)
        months = STEPS_JULY[3] + [JUNE]
    else:
        raise SystemExit(kind)
    ext = C.ext_frame(ctx["scols"], months, ctx["cache"]) if months else None
    if ext is not None:
        rows["ext_months"] = months
        rows["ext_rows"] = int(len(ext))
        if kind in ("C", "S"):
            # June leads the frame, so every later month must carry June's columns, the
            # check `common.ext_frame` makes between months (a missing one would be NaN-filled)
            miss = [c for c in parts[0].columns if c not in ext.columns]
            if miss:
                raise SystemExit(f"{tag_of(kind, s, level)}: the step months lack {miss[:8]}")
        if parts:
            ext = ext.reindex(columns=parts[0].columns)
        parts.append(ext)
    extra = parts[0] if len(parts) == 1 else pd.concat(
        [parts[0]] + [p.reindex(columns=parts[0].columns) for p in parts[1:]], ignore_index=True)
    return extra, rows


def fit_one(ctx: dict, tag: str, s: int, kind: str, level: int, half, overwrite=False) -> dict:
    npz, js = paths(tag)
    NO.refuse_existing([npz, js], overwrite=overwrite)
    extra, rows = fitting_extra(ctx, s, kind, level, half)
    setting_was, hold_was = CF.SETTING, CF.HOLD
    CF.SETTING, CF.HOLD = {"num_leaves": C.LEAVES}, ()
    prov: dict = {}
    t1 = time.time()
    try:
        d = CF.fit_arm(ctx["pans"][s], prov=prov, extra=extra, folds=(F,))
    finally:
        CF.SETTING, CF.HOLD = setting_was, hold_was
    secs = time.time() - t1
    if prov[F]["extra_rows"] != len(extra):
        raise SystemExit(f"{tag}: fit_arm carried {prov[F]['extra_rows']} extra rows, not {len(extra)}")
    vec = d[F].astype("float64")
    C.WORK.mkdir(parents=True, exist_ok=True)
    NO.refuse_existing([npz], overwrite=overwrite)
    np.savez_compressed(npz, **{F: vec})
    rec = {"tag": tag, "fold": F, "seed": s, "kind": kind, "level": level,
           "half_added": None if half is None else C.HALF_NAME[half], "rows": rows,
           "fit_arm_provenance": prov[F], "leaves": C.LEAVES, "hold": [], "lgb_seed": A.SEED,
           "seconds_fit": round(secs, 1), "load_1min_at_launch": ctx.get("load_at_launch"),
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "vector": str(npz.relative_to(C.ROOT)), "vector_sha256": C.sha256(npz),
           "rows_moved": int((vec != 0).sum())}
    C.write_json(js, rec, overwrite=overwrite)
    print(f"  fitted {npz.name} in {secs:.0f}s: fit rows {prov[F]['total_fit_rows']:,} "
          f"(extra {prov[F]['extra_rows']:,}), moved {rec['rows_moved']:,}", flush=True)
    return rec


def integrity(tag: str) -> dict:
    npz, _ = paths(tag)
    got = np.load(npz)[F].astype("float64")
    ref = ONLY0306 if tag == "integrity-C0-s1" else MS.member_path(1, 2)
    want = C.load_vec(ref)[F]
    diff = np.abs(got - want)
    rep = {"tag": tag, "reference": str(ref.relative_to(C.ROOT)), "reference_sha256": C.sha256(ref),
           "refit_sha256": C.sha256(npz), "bit_identical": bool(np.array_equal(got, want)),
           "max_abs_diff_s": float(diff.max()), "rows_differing": int((diff > 0).sum()),
           "rows": int(len(got)), "passes_1e-9": bool(diff.max() <= 1e-9)}
    C.write_json(C.ANALYSIS / f"i2-{tag}.json", rep)
    print(json.dumps(rep, indent=1), flush=True)
    return rep


def hold_for_load(tag: str) -> float:
    while True:
        la = os.getloadavg()[0]
        if la <= LOAD_MAX:
            return la
        print(f"  {tag}: 1-minute load {la:.2f} > {LOAD_MAX}, holding 60 s", flush=True)
        time.sleep(60)


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--plan", action="store_true")
    ps.add_argument("--run", action="store_true")
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    order = plan_fits()
    if a.plan:
        ctx = panels([1, 2, 3])
        out = {"what": "exp-051 I2 plan: fitting-set sizes and hashes, no fit, no gain",
               "fits": []}
        for tag, s, kind, level, half in order:
            extra, rows = fitting_extra(ctx, s, kind, level, half)
            other = int(sum(ctx["pans"][s][g].usable.sum() for g in A.FOLDS if g != F))
            out["fits"].append({"tag": tag, "seed": s, "kind": kind, "level": level,
                                "half": half, "extra_rows": int(len(extra)),
                                "member_other_fold_rows": other,
                                "total_fit_rows": other + int(len(extra)), "rows": rows})
            print(f"{tag:18} seed {s} {kind:4} L{level}  extra {len(extra):>8,}  total "
                  f"{other + len(extra):>8,}", flush=True)
        C.write_json(C.ANALYSIS / "i2-plan.json", out, overwrite=a.overwrite)
        print(f"PLAN ONLY ({time.time()-t0:.0f}s)")
        return
    if a.run:
        ctx = panels([1, 2, 3])
        for tag, s, kind, level, half in order:
            npz, js = paths(tag)
            if npz.exists() and js.exists():
                print(f"skip {npz.name}: exists", flush=True)
            else:
                ctx["load_at_launch"] = hold_for_load(npz.name)
                print(f"launch {npz.name} at load {ctx['load_at_launch']:.2f} "
                      f"({time.strftime('%H:%M:%S', time.gmtime())} UTC)", flush=True)
                fit_one(ctx, tag, s, kind, level, half, overwrite=a.overwrite)
            if tag.startswith("integrity"):
                rp = C.ANALYSIS / f"i2-{tag}.json"
                rep = json.loads(rp.read_text()) if rp.exists() else integrity(tag)
                if not rep["passes_1e-9"]:
                    raise SystemExit(f"{tag}: the refit misses 1e-9 s; the arm does not launch")
        print(f"all {len(order)} fits present ({time.time()-t0:.0f}s)")
        return
    raise SystemExit("pass --plan or --run")


if __name__ == "__main__":
    main()
