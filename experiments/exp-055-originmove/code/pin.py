"""exp-055 iteration 1, step 1: pins, meaning checks and reproductions.  ZERO FIT, LABEL-FREE.

SPEC sections 2-3 (registration commit 2509842).  Reads IDs, dates, airports, input-only
regimes and prediction columns only: no `y_true`, no `clean`, and none of exp-051's loss
records.  Every check is a hard stop.

1. `H` on each fold and on each ranking month reproduces Q15's ordered id and regime hashes;
   the K = 10 heads reproduce Q15's ordered vector hashes and are 0 outside `H`; every uploaded
   step changes only ranking `H` rows.
2. Days: the panel `day` against the UTC date of `MVT_TIME_UTC_mvt` in the raw month (count of
   rows that differ); the ranking day inside its class-vector month; the day sets.
3. Q15's A2 movement ratio `q` is reproduced to 1e-12 (month ratio and the airport-mix ratio).
4. exp-051's meanings: every K = 1 vector rebuilt from its npz (BL stitched by the OOF `half`
   column, itself checked against `withinsplit.halves`) equals OOF - pbase to 1e-9 s; both K = 3
   bags rebuilt by the bag rule from the pinned members equal the OOF bag columns to 1e-9 s (the
   carrier also with `only202506` in seed 1's place, reported); C3 = P3 off July; every bag is 0
   outside `H`; each step in d-space equals the OOF column difference to 1e-9 s.
5. Ordered SHA-256 of (ids, vector) for every vector and of (ids, column) for every OOF column.

    .venv/bin/python experiments/exp-055-originmove/code/pin.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import common as C
import withinsplit as WS                                              # noqa: E402  (exp-046)
from prc import data as D                                             # noqa: E402

OUT = C.ANALYSIS / "pin.json"
TOL = 1e-9


def main() -> None:
    C.NO.refuse_existing([OUT], overwrite=False)
    t0 = time.time()
    C.check(C.E53 / "analysis/pin.json")
    C.check(C.E53 / "analysis/routec.json")
    qpin = json.loads((C.E53 / "analysis/pin.json").read_text())       # hash keys only
    a2rec = json.loads((C.E53 / "analysis/routec.json").read_text())["A2_movement_ratio"]
    rec = {"what": __doc__.strip().splitlines()[0], "registration": "SPEC.md, commit 2509842",
           "tolerance_s": TOL, "folds": {}, "ranking": {}, "q_reproduction": {}, "e51": {},
           "ordered_sha256": {}}

    # ---- 1-2. fold panels, H, days, K = 10 heads ------------------------------------------
    heads = {}
    for lv, p in C.HEADS.items():
        C.check(p)
        z = np.load(p)
        heads[lv] = {f: z[f].astype("float64") for f in C.FOLDS}
    pan = {}
    for f in C.FOLDS:
        p = C.load_panel(f)
        if "y_true" in p.columns or "clean" in p.columns:
            raise SystemExit("a label column was read")
        ids = p.MVT_ID_mvt.to_numpy("float64")
        reg = C.fold_regime(p)
        q = qpin["folds"][f]
        if C.sha_arr(ids) != q["ids_sha256"]:
            raise SystemExit(f"{f}: ordered ids differ from Q15's pin")
        if C.sha_arr(C.regime_index(reg)) != q["regime_sha256"]:
            raise SystemExit(f"{f}: the regime mask differs from Q15's pin")
        H = reg == "H"
        if int(H.sum()) != C.H_ROWS[f]:
            raise SystemExit(f"{f}: {int(H.sum())} H rows, not {C.H_ROWS[f]}")
        for lv in C.LEVELS10:
            if C.sha_arr(heads[lv][f]) != q["head_vector_sha256"][lv]:
                raise SystemExit(f"{f} {lv}: head vector differs from Q15's pin")
            if np.any(heads[lv][f][~H] != 0):
                raise SystemExit(f"{f} {lv}: the K = 10 head moves a row outside H")
        pb = p.pbase.to_numpy("float64")
        if not np.array_equal(pb + heads["v22"][f], p.p_v22.to_numpy("float64")):
            raise SystemExit(f"{f}: p_v22 != pbase + d_v22")
        # the panel day against the raw take-off date (input column)
        mth = C.FOLD_MONTH[f]
        raw = pd.read_parquet(D.month_file(mth), columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
        raw = raw[raw.PHASE_mvt == "DEP"]
        tk = pd.Series(raw.MVT_TIME_UTC_mvt.to_numpy(), index=raw.MVT_ID_mvt.to_numpy("float64"))
        if not tk.index.is_unique or not np.isin(ids, tk.index.to_numpy()).all():
            raise SystemExit(f"{f}: panel ids do not map onto {mth}'s departures")
        tday = pd.to_datetime(tk.reindex(ids).to_numpy(), utc=True).strftime("%Y-%m-%d")
        day = p.day.astype(str).to_numpy()
        days = np.sort(np.unique(day))
        hv = WS.halves(p.day)
        nmreg = p.regime.astype(str).to_numpy()
        pan[f] = {"ids": ids, "H": H, "reg": reg, "day": day, "pbase": pb, "hv": hv,
                  "ap": p.airport.astype(str).to_numpy(), "nm": nmreg,
                  "usable": p.usable.to_numpy(bool), "guard": p.guard_held.to_numpy(bool)}
        rec["folds"][f] = {
            "month": mth, "rows": int(len(p)), "H_rows": int(H.sum()),
            "regime_rows": {r: int((reg == r).sum()) for r in C.REGIMES},
            "ids_sha256": C.sha_arr(ids), "regime_sha256": C.sha_arr(C.regime_index(reg)),
            "H_mask_sha256": C.sha_arr(H), "matches_q15_pin": True,
            "days": int(len(days)), "first_day": str(days[0]), "last_day": str(days[-1]),
            "days_with_H_rows": int(len(np.unique(day[H]))),
            "rows_panel_day_ne_takeoff_date": int((tday != day).sum()),
            "H_rows_panel_day_ne_takeoff_date": int((tday[H] != day[H]).sum()),
            "nm_regime_rows_in_H": {r: int(((nmreg == r) & H).sum()) for r in np.unique(nmreg[H])},
            "head_vector_sha256": {lv: C.sha_arr(heads[lv][f]) for lv in C.LEVELS10}}
        print(f"{f}: {len(p)} rows, H {int(H.sum())}, {len(days)} days "
              f"({days[0]}..{days[-1]}), panel day != take-off date on "
              f"{rec['folds'][f]['rows_panel_day_ne_takeoff_date']} rows; pins match Q15", flush=True)

    # ---- ranking side ------------------------------------------------------------------------
    rk = C.load_ranking()
    rid = rk.MVT_ID_mvt.to_numpy("float64")
    rreg = rk.regime.to_numpy()
    if C.sha_arr(rid) != qpin["ranking"]["ids_sha256"]:
        raise SystemExit("ranking ids differ from Q15's pin")
    if C.sha_arr(C.regime_index(rreg)) != qpin["ranking"]["regime_sha256"]:
        raise SystemExit("ranking regime differs from Q15's pin")
    up = C.load_uploads(rid)
    rmo = rk.month.to_numpy()
    rday = rk.day.to_numpy()
    for m in C.RMONTHS:
        sel = rmo == m
        Hm = sel & (rreg == "H")
        if int(Hm.sum()) != C.H_ROWS_RANK[m]:
            raise SystemExit(f"{m}: {int(Hm.sum())} ranking H rows, not {C.H_ROWS_RANK[m]}")
        d_ = np.sort(np.unique(rday[sel]))
        rec["ranking"][m] = {"rows": int(sel.sum()), "H_rows": int(Hm.sum()), "days": int(len(d_)),
                             "first_day": str(d_[0]), "last_day": str(d_[-1]),
                             "rows_outside_month": int((~pd.Series(rday[sel]).str.startswith(m)).sum()),
                             "days_with_H_rows": int(len(np.unique(rday[Hm])))}
        print(f"ranking {m}: {int(sel.sum())} rows, H {int(Hm.sum())}, {len(d_)} days "
              f"({d_[0]}..{d_[-1]}), rows outside the month "
              f"{rec['ranking'][m]['rows_outside_month']}", flush=True)
    rec["ranking"]["ids_sha256"] = C.sha_arr(rid)
    rec["ranking"]["regime_sha256"] = C.sha_arr(C.regime_index(rreg))
    rec["ranking"]["matches_q15_pin"] = True
    for k in (1, 2, 3):
        pri, nxt = C.LEVELS10[k - 1], C.LEVELS10[k]
        ch = up[nxt] != up[pri]
        out = {r: int((ch & (rreg == r)).sum()) for r in C.REGIMES}
        if out["G"] or out["A"] or out["C"]:
            raise SystemExit(f"{C.STEP[k]}: the uploaded step changes ranking rows outside H: {out}")
        rec["ranking"][f"upload_rows_changed_{C.STEP[k]}"] = out
    for lv in C.LEVELS10:
        rec["ordered_sha256"][f"upload_{lv}"] = C.sha_arr(rid, up[lv])

    # ---- 3. Q15's q, reproduced ------------------------------------------------------------
    rap = rk.airport.to_numpy()
    rH = rreg == "H"
    qrep = {}
    for f, m in C.RANK_MONTH.items():
        H = pan[f]["H"]
        fap = pan[f]["ap"]
        AP = sorted(set(fap) | set(rap))
        piR_H = {a: ((rmo == m) & (rap == a) & rH).sum() / (rmo == m).sum() for a in AP}
        for k in (1, 2, 3):
            pri, nxt = C.LEVELS10[k - 1], C.LEVELS10[k]
            dr = (up[nxt] - up[pri])[rH & (rmo == m)]
            df = (heads[nxt][f] - heads[pri][f])[H]
            ratio = float(np.sqrt(np.mean(dr ** 2)) / np.sqrt(np.mean(df ** 2)))
            ms_r = {a: float(np.mean((up[nxt] - up[pri])[rH & (rmo == m) & (rap == a)] ** 2))
                    for a in AP if (rH & (rmo == m) & (rap == a)).any()}
            ms_f = {a: float(np.mean(((heads[nxt][f] - heads[pri][f])[H & (fap == a)]) ** 2))
                    for a in AP if (H & (fap == a)).any()}
            both = [a for a in AP if a in ms_r and a in ms_f]
            w = np.array([piR_H[a] for a in both])
            mix = (float(np.sqrt(np.sum(w * np.array([ms_r[a] for a in both])) / w.sum()))
                   / float(np.sqrt(np.sum(w * np.array([ms_f[a] for a in both])) / w.sum())))
            key = f"{C.STEP[k]}:{f}->{m}"
            want = a2rec[key]
            d1, d2 = abs(ratio - want["ratio"]), abs(mix - want["ratio_fold_at_ranking_airport_mix"])
            if d1 > 1e-12 or d2 > 1e-12:
                raise SystemExit(f"{key}: q {ratio!r} / {mix!r} does not reproduce Q15's "
                                 f"{want['ratio']!r} / {want['ratio_fold_at_ranking_airport_mix']!r}")
            qrep[key] = {"q": ratio, "q_recorded": want["ratio"], "abs_diff": d1,
                         "q_airport_mix": mix, "q_airport_mix_recorded": want["ratio_fold_at_ranking_airport_mix"],
                         "rms_ranking_s": float(np.sqrt(np.mean(dr ** 2))),
                         "rms_fold_s": float(np.sqrt(np.mean(df ** 2)))}
            print(f"q {key}: {ratio:.12f} (Q15 {want['ratio']:.12f}), airport-mix {mix:.6f}", flush=True)
    rec["q_reproduction"] = qrep

    # ---- 4. exp-051 meanings -----------------------------------------------------------------
    worst = {}

    def note(key: str, a: np.ndarray, b: np.ndarray) -> float:
        dm = float(np.max(np.abs(a - b))) if len(a) else 0.0
        worst[key] = {"max_abs_diff_s": dm, "rows_differing": int((a != b).sum()),
                      "bit_identical": bool(np.array_equal(a, b))}
        if dm > TOL:
            raise SystemExit(f"{key}: max |diff| {dm!r} > {TOL}")
        return dm

    vec = {}                                 # vec[f][name] = d-space vector (float64)
    oofcols = {}
    for (it, f), path in C.OOF51.items():
        C.check(path)
        names = pq.read_schema(path).names
        cols = [c for c in names if c not in ("y_true",)]
        o = pd.read_parquet(path, columns=cols)
        if "y_true" in o.columns:
            raise SystemExit("y_true was read")
        if not np.array_equal(o.MVT_ID_mvt.to_numpy("float64"), pan[f]["ids"]):
            raise SystemExit(f"{it}-{f}: OOF row order is not the panel's")
        if set(o.fold.unique()) != {f}:
            raise SystemExit(f"{it}-{f}: fold column is not {f}")
        oofcols[(it, f)] = o
        for c in cols:
            if c.startswith("y_pred"):
                rec["ordered_sha256"][f"oof_{it}_{f}_{c}"] = C.sha_arr(pan[f]["ids"], o[c].to_numpy("float64"))
    # the column mapping of exp-051 populations.py
    want_i1 = ["y_pred", "half"] + [f"y_pred_P{L}" for L in range(4)] + [f"y_pred_BL{L}" for L in range(3)]
    want_i2j = (["y_pred"] + [f"y_pred_C3bag_L{L}" for L in range(3)] + [f"y_pred_P3bag_L{L}" for L in range(4)]
                + [f"y_pred_S1_L{L}" for L in range(4)] + ["y_pred_JuneP4", "y_pred_JuneBL4"])
    for f in C.FOLDS:
        if [c for c in want_i1 if c not in oofcols[("i1", f)].columns]:
            raise SystemExit(f"i1-{f}: columns missing")
    if [c for c in want_i2j if c not in oofcols[("i2", "july_loo")].columns]:
        raise SystemExit("i2-july_loo: columns missing")
    for f in ("december", "january"):
        o = oofcols[("i2", f)]
        if not np.array_equal(o.y_pred.to_numpy("float64"), o.y_pred_P3bag_L3.to_numpy("float64")):
            raise SystemExit(f"i2-{f}: y_pred is not P3bag_L3 (C3 = P3 off July)")

    outside = {}
    for f in C.FOLDS:
        pb, H = pan[f]["pbase"], pan[f]["H"]
        o1 = oofcols[("i1", f)]
        half = o1.half.to_numpy()
        if not np.array_equal(half.astype(int), pan[f]["hv"]):
            raise SystemExit(f"{f}: the OOF half column is not withinsplit.halves(day)")
        v = {}
        for L in range(4):
            v[f"P1_L{L}"] = C.npz_fold(C.member_path(L, 1), f)
            A_ = C.npz_fold(C.bl_path(f, "A", L), f)
            B_ = C.npz_fold(C.bl_path(f, "B", L), f)
            v[f"BL_L{L}"] = np.where(half == 1, A_, B_)
            note(f"{f}:P1_L{L} vs OOF i1 y_pred_P{L}", pb + v[f"P1_L{L}"], o1[f"y_pred_P{L}"].to_numpy("float64"))
            col = "y_pred" if L == 3 else f"y_pred_BL{L}"
            note(f"{f}:BL_L{L} vs OOF i1 {col}", pb + v[f"BL_L{L}"], o1[col].to_numpy("float64"))
        # plain members s = 1..3 on every fold; the plain bag
        for s in C.SEEDS:
            for L in range(4):
                v[f"P{s}_L{L}"] = C.npz_fold(C.member_path(L, s), f)
        o2 = oofcols[("i2", f)]
        for L in range(4):
            v[f"P3bag_L{L}"] = C.bag({s: v[f"P{s}_L{L}"] for s in C.SEEDS}, C.SEEDS, pb,
                                     pan[f]["usable"], pan[f]["guard"])
            note(f"{f}:P3bag_L{L} rebuilt vs OOF i2 y_pred_P3bag_L{L}", pb + v[f"P3bag_L{L}"],
                 o2[f"y_pred_P3bag_L{L}"].to_numpy("float64"))
            if np.any(v[f"P3bag_L{L}"][~H] != 0):
                raise SystemExit(f"{f} P3bag_L{L}: the bag moves a row outside H")
        if f == "july_loo":
            for s in C.SEEDS:
                for L in range(4):
                    v[f"C{s}_L{L}"] = C.npz_fold(C.carrier_path(L, s), f)
            for L in range(4):
                v[f"S1_L{L}"] = C.npz_fold(C.sham_path(L), f)
                note(f"{f}:S1_L{L} vs OOF i2 y_pred_S1_L{L}", pb + v[f"S1_L{L}"],
                     o2[f"y_pred_S1_L{L}"].to_numpy("float64"))
                v[f"C3bag_L{L}"] = C.bag({s: v[f"C{s}_L{L}"] for s in C.SEEDS}, C.SEEDS, pb,
                                         pan[f]["usable"], pan[f]["guard"])
                col = "y_pred" if L == 3 else f"y_pred_C3bag_L{L}"
                note(f"{f}:C3bag_L{L} rebuilt vs OOF i2 {col}", pb + v[f"C3bag_L{L}"], o2[col].to_numpy("float64"))
                if np.any(v[f"C3bag_L{L}"][~H] != 0):
                    raise SystemExit(f"{f} C3bag_L{L}: the bag moves a row outside H")
            # record: only202506 in seed 1's level-0 place rebuilds exp-051's OOF bag
            alt = C.bag({1: C.npz_fold(C.ONLY0306, f), 2: v["C2_L0"], 3: v["C3_L0"]}, C.SEEDS, pb,
                        pan[f]["usable"], pan[f]["guard"])
            note(f"{f}:C3bag_L0 with only202506 vs OOF i2 y_pred_C3bag_L0", pb + alt,
                 o2["y_pred_C3bag_L0"].to_numpy("float64"))
            note(f"{f}:integrity-C0-s1 vs only202506 (record)", v["C1_L0"], C.npz_fold(C.ONLY0306, f))
        # steps in d-space against the OOF column differences
        cmap = {"P1": ("i1", lambda L: f"y_pred_P{L}"),
                "BL": ("i1", lambda L: "y_pred" if L == 3 else f"y_pred_BL{L}"),
                "P3bag": ("i2", lambda L: f"y_pred_P3bag_L{L}")}
        if f == "july_loo":
            cmap["C3bag"] = ("i2", lambda L: "y_pred" if L == 3 else f"y_pred_C3bag_L{L}")
            cmap["S1"] = ("i2", lambda L: f"y_pred_S1_L{L}")
        for nm, (it, cf) in cmap.items():
            o = oofcols[(it, f)]
            for L in (1, 2, 3):
                dv = v[f"{nm}_L{L}"] - v[f"{nm}_L{L-1}"]
                dc = o[cf(L)].to_numpy("float64") - o[cf(L - 1)].to_numpy("float64")
                note(f"{f}:step {C.STEP[L]} {nm} d-space vs OOF difference", dv, dc)
        # changed rows outside H for every K = 1 vector (they are not in m)
        for name, x in v.items():
            if "bag" in name:
                continue
            outside[f"{f}:{name}"] = {r: int(((x != 0) & (pan[f]["reg"] == r)).sum()) for r in C.REGIMES}
        for name, x in v.items():
            rec["ordered_sha256"][f"vector_{f}_{name}"] = C.sha_arr(pan[f]["ids"], x)
        vec[f] = v
        print(f"{f}: every exp-051 vector rebuilt within {TOL} s of its OOF column", flush=True)
    rec["e51"]["checks"] = worst
    rec["e51"]["rows_moved_by_regime_K1_vectors"] = outside
    rec["e51"]["max_abs_diff_over_all_checks_s"] = max(w_["max_abs_diff_s"] for w_ in worst.values())
    # one-seed bag against the raw member on H (SPEC rejected reading 6), July
    f = "july_loo"
    H = pan[f]["H"]
    oneseed = {}
    for arm in ("P", "C"):
        for s in C.SEEDS:
            for L in range(4):
                x = vec[f][f"{arm}{s}_L{L}"]
                b1 = C.bag({s: x}, (s,), pan[f]["pbase"], pan[f]["usable"], pan[f]["guard"])
                oneseed[f"{arm}{s}_L{L}"] = {"max_abs_diff_on_H_s": float(np.max(np.abs(b1[H] - x[H]))),
                                             "rows_differing_on_H": int((b1[H] != x[H]).sum())}
    rec["e51"]["one_seed_bag_vs_raw_member_on_H"] = oneseed
    rec["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, rec)
    print(f"max |diff| over all exp-051 checks {rec['e51']['max_abs_diff_over_all_checks_s']:.3e} s; "
          f"written {C.rel(OUT)} ({rec['seconds']} s)")


if __name__ == "__main__":
    main()
