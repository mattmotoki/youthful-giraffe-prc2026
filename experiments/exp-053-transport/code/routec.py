"""exp-053 iteration 2, route C: zero-fit transport test of the composition instrument between labelled months.

SPEC §10 (registration 2, commit 01e4703), after iteration 1 (commit 4926895).  For each step and
each ordered pair of distinct folds (source s, target t), the target's measured step gain F_t
is predicted at four coarsening levels from the source: (m, a, r) the source's supported head-cell
gains at the target's head-cell masses; (m, a) the source's airport gains at the target's airport
masses; (m, r) the source's gain per head row times the target's head share; (m) the source's
all-row gain.  Primary: the 18 single-step ordered pairs; secondary: leave-one-month-out with the
other two folds pooled.  Then the ladder on the ranking target (read-only board comparison last),
and the descriptive addenda A1 (order intervals) and A2 (month-level movement ratio).

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-053-transport/code/routec.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import common as C

OUT = C.ANALYSIS / "routec.json"
PIN = C.ANALYSIS / "pin.json"
INSTR = C.ANALYSIS / "instrument.json"
BOOT = 2000
SEED0 = 5301
LS = (1, 2, 3)
ALL_STEPS = C.STEPS + (C.CUMULATIVE,)
NAMES = [s[0] for s in C.STEPS]
PAIRS = [(s, t) for s in C.FOLDS for t in C.FOLDS if s != t]
LEVELS = ("S", "P", "R", "F")          # (m, a, r), (m, a), (m, r), (m)
RMONTHS = ("2026-01", "2026-07")


def pct(a) -> list:
    a = np.asarray(a, dtype="float64")
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def main() -> None:
    C.NO.refuse_existing([OUT], overwrite=False)
    t0 = time.time()
    pin = json.loads(PIN.read_text())
    ins = json.loads(INSTR.read_text())
    heads = C.load_heads()

    # ---- panels, pinned exactly as in iteration 1 ------------------------------------------
    pan = {}
    for f in C.FOLDS:
        p = C.load_panel(f)
        ids = p.MVT_ID_mvt.to_numpy("float64")
        reg = C.fold_regime(p).astype(str)
        if C.sha_arr(ids) != pin["folds"][f]["ids_sha256"]:
            raise SystemExit(f"{f}: id order differs from the pin")
        if C.sha_arr(np.array([C.REGIMES.index(r) for r in reg], "int8")) != pin["folds"][f]["regime_sha256"]:
            raise SystemExit(f"{f}: regime mask differs from the pin")
        y = p.y_true.to_numpy("float64")
        pb = p.pbase.to_numpy("float64")
        sq = {lv: (y - pb - heads[lv][f]) ** 2 for lv in C.LEVELS}
        pan[f] = {"ap": p.airport.astype(str).to_numpy(), "reg": reg,
                  "day": p.day.astype(str).reset_index(drop=True),
                  "v": {name: sq[pri] - sq[nxt] for name, pri, nxt in ALL_STEPS},
                  "d": {lv: heads[lv][f] for lv in C.LEVELS}}
        for name, _, _ in ALL_STEPS:
            if abs(pan[f]["v"][name].sum() - pin["steps"][name]["by_fold"][f]["paired_sse"]) > 1e-6 * abs(
                    pin["steps"][name]["by_fold"][f]["paired_sse"]):
                raise SystemExit(f"{f} {name}: paired SSE differs from the pin")
    AP = sorted(set().union(*[set(pan[f]["ap"]) for f in C.FOLDS]))
    ai = {a: i for i, a in enumerate(AP)}

    # ---- iteration 1's draws, rebuilt (SPEC §4.7, §10.2) ------------------------------------
    nb = {f: {L: (pd.Period(C.FOLD_MONTH[f]).days_in_month - 1) // L + 1 for L in LS} for f in C.FOLDS}
    picks = {}
    for L in LS:
        rng = np.random.default_rng(SEED0 + L)
        for f in C.FOLDS:
            k = nb[f][L]
            pk = rng.integers(0, k, size=(BOOT, k))
            picks[(f, L)] = np.stack([np.bincount(r, minlength=k) for r in pk]).astype("float64")

    # ---- per fold: block sums; point support ------------------------------------------------
    blk, sup_H, sup_A, days_H, days_A = {}, {}, {}, {}, {}
    for f in C.FOLDS:
        q = pan[f]
        a_idx = np.array([ai[a] for a in q["ap"]])
        H = q["reg"] == "H"
        days_H[f] = np.array([q["day"][H & (a_idx == i)].nunique() for i in range(len(AP))])
        days_A[f] = np.array([q["day"][a_idx == i].nunique() for i in range(len(AP))])
        nH = np.bincount(a_idx[H], minlength=len(AP))
        nA = np.bincount(a_idx, minlength=len(AP))
        sup_H[f] = (nH >= C.N_MIN) & (days_H[f] >= C.D_MIN)
        sup_A[f] = (nA >= C.N_MIN) & (days_A[f] >= C.D_MIN)
        want = {a: v_["supported"] for a, v_ in ins["folds"][f]["H_cells"].items()}
        for a, s_ in want.items():
            if bool(sup_H[f][ai[a]]) != s_:
                raise SystemExit(f"{f} {a}: head support differs from iteration 1")
        for L in LS:
            b = C.day_block(q["day"], L)
            k = nb[f][L]
            NH = np.zeros((k, len(AP)))
            np.add.at(NH, (b[H], a_idx[H]), 1.0)
            NA = np.zeros((k, len(AP)))
            np.add.at(NA, (b, a_idx), 1.0)
            NT = np.bincount(b, minlength=k).astype("float64")
            SH, SA, ST = {}, {}, {}
            for name, _, _ in ALL_STEPS:
                SH[name] = np.zeros((k, len(AP)))
                np.add.at(SH[name], (b[H], a_idx[H]), q["v"][name][H])
                SA[name] = np.zeros((k, len(AP)))
                np.add.at(SA[name], (b, a_idx), q["v"][name])
                ST[name] = np.bincount(b, weights=q["v"][name], minlength=k)
            blk[(f, L)] = {"NH": NH, "NA": NA, "NT": NT, "SH": SH, "SA": SA, "ST": ST}

    def stats(f: str, L: int, point: bool) -> dict:
        B_ = blk[(f, L)]
        W = np.ones((1, nb[f][L])) if point else picks[(f, L)]
        return {"cH": W @ B_["NH"], "cA": W @ B_["NA"], "tot": W @ B_["NT"],
                "sH": {n_: W @ B_["SH"][n_] for n_ in B_["SH"]},
                "sA": {n_: W @ B_["SA"][n_] for n_ in B_["SA"]},
                "sT": {n_: W @ B_["ST"][n_] for n_ in B_["ST"]}}

    def pool(a: dict, b: dict) -> dict:
        return {"cH": a["cH"] + b["cH"], "cA": a["cA"] + b["cA"], "tot": a["tot"] + b["tot"],
                "sH": {n_: a["sH"][n_] + b["sH"][n_] for n_ in a["sH"]},
                "sA": {n_: a["sA"][n_] + b["sA"][n_] for n_ in a["sA"]},
                "sT": {n_: a["sT"][n_] + b["sT"][n_] for n_ in a["sT"]}}

    def predict(src: dict, tgt: dict, supH: np.ndarray, supA: np.ndarray, name: str) -> dict:
        with np.errstate(invalid="ignore", divide="ignore"):
            gH = src["sH"][name] / src["cH"]
            gA = src["sA"][name] / src["cA"]
            piH_t = tgt["cH"] / tgt["tot"][:, None]
            piA_t = tgt["cA"] / tgt["tot"][:, None]
            Fs = src["sT"][name] / src["tot"]
            Ft = tgt["sT"][name] / tgt["tot"]
            hs = src["cH"].sum(1) / src["tot"]
            ht = tgt["cH"].sum(1) / tgt["tot"]
        emp = (src["cH"][:, supH] == 0).any(1) | (src["cA"][:, supA] == 0).any(1)
        S = np.where(emp, np.nan, (piH_t[:, supH] * gH[:, supH]).sum(1))
        P = np.where(emp, np.nan, (piA_t[:, supA] * gA[:, supA]).sum(1))
        return {"S": S, "P": P, "R": ht * Fs / hs, "F": Fs, "target": Ft, "Fs": Fs,
                "U": (piH_t[:, ~supH]).sum(1), "empty": emp}

    rec = {"what": __doc__.strip().splitlines()[0],
           "registration": "SPEC §10, commit 01e4703",
           "inputs": {"pin": {"path": str(PIN.relative_to(C.ROOT)), "sha256": C.sha_file(PIN)},
                      "instrument": {"path": str(INSTR.relative_to(C.ROOT)), "sha256": C.sha_file(INSTR)}},
           "airports": AP}

    # ---- reproduce iteration 1 (board S, v20, 3-day) ----------------------------------------
    rk = C.load_ranking()
    rsub = {m: rk[rk.month == m] for m in RMONTHS}
    piR_H = {m: np.array([((rsub[m].airport == a) & (rsub[m].regime == "H")).sum() / len(rsub[m]) for a in AP])
             for m in RMONTHS}
    piR_A = {m: np.array([(rsub[m].airport == a).sum() / len(rsub[m]) for a in AP]) for m in RMONTHS}
    bmap = {"july_loo": "2026-07", "january": "2026-01"}
    sh = C.SHARES

    def rank_levels(f: str, L: int, point: bool, name: str) -> dict:
        st = stats(f, L, point)
        m = bmap[f]
        with np.errstate(invalid="ignore", divide="ignore"):
            gH = st["sH"][name] / st["cH"]
            gA = st["sA"][name] / st["cA"]
            F = st["sT"][name] / st["tot"]
            hf = st["cH"].sum(1) / st["tot"]
        emp = (st["cH"][:, sup_H[f]] == 0).any(1)
        S = np.where(emp, np.nan, np.nansum(gH[:, sup_H[f]] * piR_H[m][sup_H[f]], axis=1))
        P = np.nansum(gA[:, sup_A[f]] * piR_A[m][sup_A[f]], axis=1)
        return {"S": S, "P": P, "R": piR_H[m].sum() * F / hf, "F": F}

    def board(name: str, L: int, point: bool) -> dict:
        kap = pin["steps"][name]["kappa_s_per_s2"]
        j, n_ = rank_levels("july_loo", L, point, name), rank_levels("january", L, point, name)
        return {x: kap * (sh["july_loo"] * j[x] + sh["january"] * n_[x]) for x in LEVELS}

    rep = pct(board("v20", 3, False)["S"])
    want = ins["board"]["v20"]["intervals"]["L3"]["S_s"]
    diff = max(abs(rep[0] - want[0]), abs(rep[1] - want[1]))
    if diff > 1e-9:
        raise SystemExit(f"iteration 1's draws do not reproduce: {rep} vs {want}")
    pt = board("v20", 3, True)
    if abs(pt["S"][0] - ins["board"]["v20"]["S_s"]) > 1e-12 or abs(pt["P"][0] - ins["board"]["v20"]["P_s"]) > 1e-12:
        raise SystemExit("iteration 1's point S or P does not reproduce")
    rec["reproduce_iteration1"] = {"board_v20_S_3day": rep, "recorded": want, "max_abs_diff_s": diff}
    print(f"iteration 1 reproduced: v20 board S 3-day {rep} (diff {diff:.1e})", flush=True)

    # ---- primary: 18 ordered single-step pairs --------------------------------------------
    def run_set(pairs_src: list, label: str) -> dict:
        """pairs_src: list of (key, source-builder, target fold, supH, supA)."""
        out = {"predictions": {}, "intervals": {}}
        per_draw = {L: {x: [] for x in LEVELS} for L in LS}
        abs_pt = {x: [] for x in LEVELS}
        for name in NAMES + [C.CUMULATIVE[0]]:
            for key, build, t, supH, supA in pairs_src:
                pp = predict(build(True, 3), stats(t, 3, True), supH, supA, name)
                row = {"target_F": float(pp["target"][0]), "source_F": float(pp["Fs"][0]),
                       "unsupported_target_H_mass": float(pp["U"][0])}
                for x in LEVELS:
                    e = float(pp[x][0] - pp["target"][0])
                    row[x] = {"pred": float(pp[x][0]), "err": e, "rel_abs_err": abs(e) / abs(pp["target"][0])}
                ds = predict(build(False, 3), stats(t, 3, False), supH, supA, name)
                row["ci3_absdiff_S_minus_F"] = pct(np.abs(ds["S"] - ds["target"]) - np.abs(ds["F"] - ds["target"]))
                row["ci3_err"] = {x: pct(ds[x] - ds["target"]) for x in LEVELS}
                row["empty_draws_3day"] = int(ds["empty"].sum())
                out["predictions"][f"{name}:{key}"] = row
                if name in NAMES:
                    for x in LEVELS:
                        abs_pt[x].append(abs(row[x]["err"]))
                    for L in LS:
                        d_ = predict(build(False, L), stats(t, L, False), supH, supA, name)
                        for x in LEVELS:
                            per_draw[L][x].append(np.abs(d_[x] - d_["target"]))
        n = len(abs_pt["F"])
        mae = {x: float(np.mean(abs_pt[x])) for x in LEVELS}
        sp = [(k_, r_) for k_, r_ in out["predictions"].items() if k_.split(":")[0] in NAMES]
        out["n"] = n
        out["MAE"] = mae
        out["MAE_ratio_to_F"] = {x: mae[x] / mae["F"] for x in LEVELS}
        out["mean_rel_abs_err"] = {x: float(np.mean([r_[x]["rel_abs_err"] for _, r_ in sp])) for x in LEVELS}
        out["wins"] = {f"{x}>{y}": int(sum(abs(r_[x]["err"]) < abs(r_[y]["err"]) for _, r_ in sp))
                       for x, y in (("S", "F"), ("P", "F"), ("R", "F"), ("S", "R"), ("S", "P"))}
        out["direction_toward_target"] = {
            x: int(sum(np.sign(r_[x]["pred"] - r_["source_F"]) == np.sign(r_["target_F"] - r_["source_F"])
                       for _, r_ in sp)) for x in ("S", "P", "R")}
        for L in LS:
            M = {x: np.mean(np.stack(per_draw[L][x]), axis=0) for x in LEVELS}
            out["intervals"][f"L{L}"] = {"S-F": pct(M["S"] - M["F"]), "R-F": pct(M["R"] - M["F"]),
                                         "P-F": pct(M["P"] - M["F"]), "S-R": pct(M["S"] - M["R"]),
                                         "dropped_draws": int(np.isnan(M["S"]).sum())}
        i3 = out["intervals"]["L3"]
        # SPEC §10.3 registers CT1 and CT2 on the 18 ordered pairs only; the LOMO set is descriptive
        out["CT1"] = (bool(mae["S"] < mae["F"] and i3["S-F"][1] is not None and i3["S-F"][1] < 0
                           and out["wins"]["S>F"] >= 12) if n == 18 else None)
        out["CT2"] = (bool(mae["S"] < mae["R"] and i3["S-R"][1] is not None and i3["S-R"][1] < 0)
                      if n == 18 else None)
        print(f"{label}: n {n}, MAE S {mae['S']:.1f} P {mae['P']:.1f} R {mae['R']:.1f} F {mae['F']:.1f}; "
              f"wins {out['wins']}; 3-day S-F {i3['S-F']} S-R {i3['S-R']}; CT1 {out['CT1']} CT2 {out['CT2']}",
              flush=True)
        return out

    pair_src = [(f"{s}->{t}", (lambda pt_, L, s=s: stats(s, L, pt_)), t, sup_H[s], sup_A[s]) for s, t in PAIRS]
    rec["primary_pairs"] = run_set(pair_src, "pairs")

    # ---- secondary: leave-one-month-out, two sources pooled ---------------------------------
    lomo_src = []
    for t in C.FOLDS:
        s1, s2 = [f for f in C.FOLDS if f != t]
        nH = blk[(s1, 3)]["NH"].sum(0) + blk[(s2, 3)]["NH"].sum(0)
        nA = blk[(s1, 3)]["NA"].sum(0) + blk[(s2, 3)]["NA"].sum(0)
        supH = (nH >= C.N_MIN) & (days_H[s1] + days_H[s2] >= C.D_MIN)
        supA = (nA >= C.N_MIN) & (days_A[s1] + days_A[s2] >= C.D_MIN)
        lomo_src.append((f"{s1}+{s2}->{t}", (lambda pt_, L, s1=s1, s2=s2: pool(stats(s1, L, pt_), stats(s2, L, pt_))),
                         t, supH, supA))
    rec["secondary_lomo"] = run_set(lomo_src, "lomo")

    # ---- the ladder on the ranking target (SPEC §10.4) --------------------------------------
    lad = {}
    for name in NAMES + [C.CUMULATIVE[0]]:
        pt_ = board(name, 3, True)
        row = {x: float(pt_[x][0]) for x in LEVELS}
        for L in LS:
            dr = board(name, L, False)
            row[f"L{L}"] = {x: pct(dr[x]) for x in LEVELS}
            row[f"L{L}"].update({f"{x}-F": pct(dr[x] - dr["F"]) for x in ("S", "P", "R")})
        row["ratio_to_F"] = {x: row[x] / row["F"] for x in LEVELS}
        lad[name] = row
    rec["ladder_ranking"] = lad

    # ---- A1: are the step orders resolved (iteration 1's draws, board seconds) ---------------
    a1 = {}
    for L in LS:
        bd = {n_: board(n_, L, False) for n_ in NAMES}
        a1[f"L{L}"] = {"S_v20-S_v21": pct(bd["v20"]["S"] - bd["v21"]["S"]),
                       "S_v21-S_v22": pct(bd["v21"]["S"] - bd["v22"]["S"]),
                       "F_v21-F_v20": pct(bd["v21"]["F"] - bd["v20"]["F"]),
                       "F_v20-F_v22": pct(bd["v20"]["F"] - bd["v22"]["F"])}
    pts = {n_: board(n_, 3, True) for n_ in NAMES}
    a1["point"] = {"S_v20-S_v21": float(pts["v20"]["S"][0] - pts["v21"]["S"][0]),
                   "S_v21-S_v22": float(pts["v21"]["S"][0] - pts["v22"]["S"][0]),
                   "F_v21-F_v20": float(pts["v21"]["F"][0] - pts["v20"]["F"][0]),
                   "F_v20-F_v22": float(pts["v20"]["F"][0] - pts["v22"]["F"][0])}
    rec["A1_order_intervals"] = a1

    # ---- A2: month-level label-free movement ratio -------------------------------------------
    rid = rk.MVT_ID_mvt.to_numpy("float64")
    up = {}
    for lv in C.LEVELS:
        C.check(C.UPLOAD[lv], C.UPLOAD_SHA[lv])
        u = pd.read_parquet(C.UPLOAD[lv])
        u["MVT_ID_mvt"] = u.MVT_ID_mvt.astype("float64")
        up[lv] = u.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(rid).to_numpy("float64")
    rH, rap, rmo = rk.regime.to_numpy() == "H", rk.airport.to_numpy(), rk.month.to_numpy()
    a2 = {}
    for f, m in bmap.items():
        q = pan[f]
        H = q["reg"] == "H"
        for name, pri, nxt in ALL_STEPS:
            dr_ = (up[nxt] - up[pri])[rH & (rmo == m)]
            df_ = (q["d"][nxt] - q["d"][pri])[H]
            ms_r = {a: float(np.mean((up[nxt] - up[pri])[rH & (rmo == m) & (rap == a)] ** 2))
                    for a in AP if (rH & (rmo == m) & (rap == a)).any()}
            ms_f = {a: float(np.mean(((q["d"][nxt] - q["d"][pri])[H & (q["ap"] == a)]) ** 2))
                    for a in AP if (H & (q["ap"] == a)).any()}
            both = [a for a in AP if a in ms_r and a in ms_f]
            w = np.array([piR_H[m][ai[a]] for a in both])
            rw_r = float(np.sqrt(np.sum(w * np.array([ms_r[a] for a in both])) / w.sum()))
            rw_f = float(np.sqrt(np.sum(w * np.array([ms_f[a] for a in both])) / w.sum()))
            a2[f"{name}:{f}->{m}"] = {
                "rms_ranking": float(np.sqrt(np.mean(dr_ ** 2))), "rms_fold": float(np.sqrt(np.mean(df_ ** 2))),
                "ratio": float(np.sqrt(np.mean(dr_ ** 2)) / np.sqrt(np.mean(df_ ** 2))),
                "ratio_fold_at_ranking_airport_mix": rw_r / rw_f,
                "ranking_H_mass_without_fold_rows": float(piR_H[m][[ai[a] for a in ms_r if a not in ms_f]].sum())}
    rec["A2_movement_ratio"] = a2

    # ---- read only (score-feedback rule), last: the ladder against the board ---------------------------------
    B = np.array([C.BOARD_GAIN[k] for k in NAMES])
    ro = {"board_gain_s": dict(zip(NAMES, B.tolist()))}
    Fv = np.array([lad[k]["F"] for k in NAMES])
    for x in LEVELS:
        X = np.array([lad[k][x] for k in NAMES])
        ro[x] = {"values_s": dict(zip(NAMES, X.tolist())), "E_abs_s": float(np.abs(X - B).sum()),
                 "sign_agrees": bool(np.all(np.sign(X) == np.sign(B))),
                 "order_agrees": [NAMES[i] for i in np.argsort(-X)] == [NAMES[i] for i in np.argsort(-B)],
                 "gap_v21_v22_accounted_s": float(Fv[1] + Fv[2] - X[1] - X[2])}
    ro["note"] = "read only; nothing is fitted, selected, scaled or changed from these (D11)"
    rec["ladder_read_only"] = ro
    rec["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, rec)
    print(json.dumps({k: rec[k] for k in ("A1_order_intervals",)}, indent=1))
    print(json.dumps(ro, indent=1))
    print(f"written {OUT.relative_to(C.ROOT)} ({rec['seconds']} s)")


if __name__ == "__main__":
    main()
