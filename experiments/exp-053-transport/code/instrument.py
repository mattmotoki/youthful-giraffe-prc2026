"""exp-053 iteration 1, step 2: the vector-only transport-sensitivity instrument.  ZERO FIT.

SPEC §3-§8, frozen at commit d1e1706; pins and census at 6d61776 (`analysis/pin.json`).
For each shipped extension step (v20|v19, v21|v20, v22|v21, and the cumulative v22|v19):
stratum paired losses on (month, airport, ownership regime); the unstandardized month gain F,
the ranking-standardized gain S on supported head cells, the airport-only control P, the
unsupported ranking head mass U, the exact decomposition of S - F, 1-/2-/3-day block
bootstrap intervals, Kish day counts; board weighting and seconds at the face's own kappa;
December as a zero-weight control; the predictor tables of SPEC §6; the label-free
ranking-movement check; the OOF receipts; and the iteration-2 routing rule of SPEC §8.
Board gains enter only the last, read-only comparison block (score-feedback rule), after every fold quantity
and after the route is chosen.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-053-transport/code/instrument.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import common as C

OUT = C.ANALYSIS / "instrument.json"
CELLS = C.ANALYSIS / "instrument-cells.csv"
OOF = {f: C.EXP / f"oof/i1-{f}.parquet" for f in C.FOLDS}
PIN = C.ANALYSIS / "pin.json"
BOOT = 2000
SEED0 = 5301                         # SPEC §4.7: one generator per L, seed 5301 + L
LS = (1, 2, 3)
ALL_STEPS = C.STEPS + (C.CUMULATIVE,)
NAMES = [s[0] for s in C.STEPS]
TARGETS = {"july_loo": ("2026-07",), "december": ("2026-01", "2026-07"), "january": ("2026-01",)}
NM_REG = ("nm_bulk", "nm_late", "nonm", "noac")
RMONTHS = ("2026-01", "2026-07")
#: SPEC §8 route B (ii): built extension months, and the pairs already measured on a fold by
#: exp-046 #4 (§47.74) and recorded rule (§47.76)
BUILT = ("2025-02", "2025-03", "2025-06", "2025-08", "2025-10", "2025-11")
MEASURED = {("december", frozenset({"2025-10", "2025-02"})),
            ("january", frozenset({"2025-03", "2025-11"}))}


def pct(a) -> list:
    a = np.asarray(a, dtype="float64")
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def width(ci) -> float | None:
    return None if ci[0] is None else ci[1] - ci[0]


def midx(m: str) -> int:
    y, mo = m.split("-")
    return int(y) * 12 + int(mo) - 1


def main() -> None:
    C.NO.refuse_existing([OUT, CELLS] + list(OOF.values()), overwrite=False)
    t0 = time.time()
    pin = json.loads(PIN.read_text())
    heads = C.load_heads()
    rk = C.load_ranking()
    if C.sha_arr(rk.MVT_ID_mvt.to_numpy("float64")) != pin["ranking"]["ids_sha256"]:
        raise SystemExit("ranking id order differs from the pin")
    if C.sha_arr(np.array([C.REGIMES.index(r) for r in rk.regime], "int8")) != pin["ranking"]["regime_sha256"]:
        raise SystemExit("ranking regime differs from the pin")
    rsub = {m: rk[rk.month == m] for m in RMONTHS}
    rcnt = {m: rsub[m].groupby(["airport", "regime"]).size() for m in RMONTHS}
    rcnt_a = {m: rsub[m].groupby("airport").size() for m in RMONTHS}
    pcen = pd.DataFrame(pin["census"]["fold_cells"])
    rcen = pd.DataFrame(pin["census"]["ranking_cells"])
    scu = pin["support_conditions"]["SC_U_airports"]
    scj = pin["support_conditions"]["SC_J_airports"]
    conds = {f"SC-U({k})": {**{f: scu[f"{k}@{C.FOLD_MONTH[f]}"] for f in C.FOLDS},
                            **{m: scu[f"{k}@{m}"] for m in RMONTHS}} for k in NAMES}
    conds["SC-J"] = {**{f: scj[C.FOLD_MONTH[f]] for f in C.FOLDS}, **{m: scj[m] for m in RMONTHS}}

    # SPEC §4.7: block draws, one generator per L, months drawn in FOLDS order (independent)
    nb = {f: {L: (pd.Period(C.FOLD_MONTH[f]).days_in_month - 1) // L + 1 for L in LS} for f in C.FOLDS}
    picks = {}
    for L in LS:
        rng = np.random.default_rng(SEED0 + L)
        for f in C.FOLDS:
            k = nb[f][L]
            pk = rng.integers(0, k, size=(BOOT, k))
            picks[(f, L)] = np.stack([np.bincount(r, minlength=k) for r in pk]).astype("float64")

    rec = {"what": __doc__.strip().splitlines()[0],
           "pin": {"path": str(PIN.relative_to(C.ROOT)), "sha256": C.sha_file(PIN)},
           "bootstrap": {"draws": BOOT, "seeds": {f"L{L}": SEED0 + L for L in LS},
                         "blocks": {f: nb[f] for f in C.FOLDS}},
           "folds": {}, "months": {}, "board": {}, "comparison_faces": {}, "routing": {},
           "predictor_tables": {}, "movement": {}, "comparison_read_only": {}}
    cell_rows = []
    oofs = {}
    draws = {}
    routeA = {(c, n_): {} for c in conds for n_, _, _ in ALL_STEPS}

    for f in C.FOLDS:
        p = C.load_panel(f)
        ids = p.MVT_ID_mvt.to_numpy("float64")
        if C.sha_arr(ids) != pin["folds"][f]["ids_sha256"]:
            raise SystemExit(f"{f}: id order differs from the pin")
        reg = C.fold_regime(p).astype(str)
        if C.sha_arr(np.array([C.REGIMES.index(r) for r in reg], "int8")) != pin["folds"][f]["regime_sha256"]:
            raise SystemExit(f"{f}: regime mask differs from the pin")
        nonm = C.no_nm_mask(ids, f)
        if C.sha_arr(nonm) != pin["folds"][f]["no_nm_sha256"]:
            raise SystemExit(f"{f}: no-NM mask differs from the pin")
        y = p.y_true.to_numpy("float64")
        pb = p.pbase.to_numpy("float64")
        ap = p.airport.astype(str).to_numpy()
        day = p.day.astype(str).reset_index(drop=True)
        dayv = day.to_numpy()
        n = len(p)
        sq = {lv: (y - pb - heads[lv][f]) ** 2 for lv in C.LEVELS}
        v = {name: sq[pri] - sq[nxt] for name, pri, nxt in ALL_STEPS}
        # --- SPEC §4.6: all-row reconciliation against the pin and the record
        for name, _, _ in ALL_STEPS:
            pf = pin["steps"][name]["by_fold"][f]
            if abs(v[name].sum() - pf["paired_sse"]) > 1e-6 * max(1.0, abs(pf["paired_sse"])):
                raise SystemExit(f"{f} {name}: paired SSE {v[name].sum()!r} != pinned {pf['paired_sse']!r}")
            if "recorded_mse_gain_s2" in pf and abs(v[name].mean() - pf["recorded_mse_gain_s2"]) > 1e-9:
                raise SystemExit(f"{f} {name}: F {v[name].mean()!r} != recorded {pf['recorded_mse_gain_s2']!r}")
        for name, _, _ in ALL_STEPS:
            if np.any(v[name][reg != "H"] != 0):
                raise SystemExit(f"{f} {name}: a non-H row has a nonzero paired loss")

        # --- strata (SPEC §3) and support, checked against the pin's census
        cells = sorted(set(zip(ap, reg)))
        code = {c: i for i, c in enumerate(cells)}
        sc = np.array([code[c] for c in zip(ap, reg)])
        ns = np.bincount(sc, minlength=len(cells)).astype("float64")
        aps = sorted(set(ap))
        acode = {a: i for i, a in enumerate(aps)}
        sa = np.array([acode[a] for a in ap])
        na = np.bincount(sa, minlength=len(aps)).astype("float64")
        D_s = {c: int(day[sc == code[c]].nunique()) for c in cells}
        kish_s = {c: C.kish(day[sc == code[c]].value_counts().to_numpy()) for c in cells}
        D_a = {a: int(day[sa == acode[a]].nunique()) for a in aps}
        sup = {c: bool(c[1] == "H" and ns[code[c]] >= C.N_MIN and D_s[c] >= C.D_MIN) for c in cells}
        sup_a = {a: bool(na[acode[a]] >= C.N_MIN and D_a[a] >= C.D_MIN) for a in aps}
        pc = pcen[pcen.fold == f].set_index(["airport", "regime"])
        for c in cells:
            if int(pc.loc[c, "fold_rows"]) != int(ns[code[c]]) or bool(pc.loc[c, "supported"]) != sup[c]:
                raise SystemExit(f"{f} {c}: stratum count or support differs from the pin's census")
        g = {name: np.bincount(sc, weights=v[name], minlength=len(cells)) / ns for name, _, _ in ALL_STEPS}
        ga = {name: np.bincount(sa, weights=v[name], minlength=len(aps)) / na for name, _, _ in ALL_STEPS}
        sse_s = {lv: np.bincount(sc, weights=sq[lv], minlength=len(cells)) for lv in C.LEVELS}
        pif = ns / n
        pif_a = na / n
        Hcells = [c for c in cells if c[1] == "H"]
        supc = [code[c] for c in Hcells if sup[c]]
        pir = {m: {c: float(k) / len(rsub[m]) for c, k in rcnt[m].items()} for m in TARGETS[f]}
        pir_a = {m: {a: float(k) / len(rsub[m]) for a, k in rcnt_a[m].items()} for m in TARGETS[f]}

        # --- block sums for the bootstrap
        bsum = {}
        for L in LS:
            b = C.day_block(day, L)
            k = nb[f][L]
            if b.min() < 0 or b.max() >= k:
                raise SystemExit(f"{f}: a {L}-day block id is outside 0..{k - 1}")
            Nbs = np.zeros((k, len(cells)))
            np.add.at(Nbs, (b, sc), 1.0)
            Nba = np.zeros((k, len(aps)))
            np.add.at(Nba, (b, sa), 1.0)
            Sbs, Sba = {}, {}
            for name, _, _ in ALL_STEPS:
                Sbs[name] = np.zeros((k, len(cells)))
                np.add.at(Sbs[name], (b, sc), v[name])
                Sba[name] = np.zeros((k, len(aps)))
                np.add.at(Sba[name], (b, sa), v[name])
            W = picks[(f, L)]
            cnt_s, cnt_a = W @ Nbs, W @ Nba
            with np.errstate(invalid="ignore", divide="ignore"):
                gs_b = {name: (W @ Sbs[name]) / cnt_s for name, _, _ in ALL_STEPS}
                ga_b = {name: (W @ Sba[name]) / cnt_a for name, _, _ in ALL_STEPS}
            tot = W @ Nbs.sum(1)
            bsum[L] = {"b": b, "cnt_s": cnt_s, "cnt_a": cnt_a, "gs": gs_b, "ga": ga_b,
                       "F": {name: (W @ Sbs[name].sum(1)) / tot for name, _, _ in ALL_STEPS}}

        # --- month estimates (SPEC §4.2, §4.4, §4.5)
        for m in TARGETS[f]:
            key = f"{f}->{m}"
            rank_H = {c: pir[m].get(c, 0.0) for c in set(Hcells) | {c for c in pir[m] if c[1] == "H"}}
            U = sum(w for c, w in rank_H.items() if not sup.get(c, False))
            hr_tot = sum(rank_H.values())
            hf_tot = float(sum(pif[code[c]] for c in Hcells))
            wrow = np.zeros(n)
            for c in Hcells:
                if sup[c]:
                    wrow[sc == code[c]] = rank_H.get(c, 0.0) / ns[code[c]]
            nH = int((reg == "H").sum())
            wun = np.where(reg == "H", 1.0 / nH, 0.0)
            kd_s = C.kish(pd.Series(wrow).groupby(dayv).sum().to_numpy())
            kd_u = C.kish(pd.Series(wun).groupby(dayv).sum().to_numpy())
            wa = np.array([pir_a[m].get(a, 0.0) for a in aps])
            supa = np.array([sup_a[a] for a in aps])
            wr = np.array([rank_H.get(cells[i], 0.0) for i in supc])
            mrec = {"kish_days_standardized": kd_s, "kish_days_unstandardized": kd_u,
                    "kish_days_loss": 1 - kd_s / kd_u,
                    "rank_H_mass": hr_tot, "fold_H_mass": hf_tot, "unsupported_rank_H_mass": U,
                    "unsupported_share_of_rank_H": U / hr_tot,
                    "supported_target_mass": 1 - U, "unsupported_target_mass": U,
                    "unsupported_cells": sorted(c[0] for c, w in rank_H.items()
                                                if not sup.get(c, False) and w > 0),
                    "airport_only_unsupported_mass": float(sum(w for a, w in pir_a[m].items()
                                                               if not sup_a.get(a, False))),
                    "steps": {}}
            for name, _, _ in ALL_STEPS:
                F = float(v[name].mean())
                S = float(sum(rank_H.get(c, 0.0) * g[name][code[c]] for c in Hcells if sup[c]))
                P = float((wa[supa] * ga[name][supa]).sum())
                mix = cov = drop = 0.0
                for c in Hcells:
                    a, gc = c[0], g[name][code[c]]
                    if sup[c]:
                        pr_a, pf_a = pir_a[m].get(a, 0.0), pif_a[acode[a]]
                        hr = rank_H.get(c, 0.0) / pr_a if pr_a > 0 else 0.0
                        hf = pif[code[c]] / pf_a
                        mix += gc * (pr_a - pf_a) * (hr + hf) / 2
                        cov += gc * (hr - hf) * (pr_a + pf_a) / 2
                    else:
                        drop -= pif[code[c]] * gc
                if abs((mix + cov + drop) - (S - F)) > 1e-8 * max(1.0, abs(F)):
                    raise SystemExit(f"{key} {name}: the decomposition does not close")
                gr = S / (hr_tot - U)
                gf = F / hf_tot
                st = {"F_s2": F, "S_s2": S, "P_s2": P, "S_minus_F_s2": S - F, "S_over_F": S / F,
                      "decomposition": {"airport_mix": mix, "head_coverage": cov,
                                        "fold_cells_without_support": drop,
                                        "missing_support_mass_U": U},
                      "multiplicative": {"head_mass_ratio": hr_tot / hf_tot,
                                         "supported_fraction": (hr_tot - U) / hr_tot,
                                         "per_row_ratio": gr / gf,
                                         "gain_per_H_row_rank_supported": gr,
                                         "gain_per_H_row_fold": gf},
                      "intervals": {}}
                qd = {}
                for L in LS:
                    bs = bsum[L]
                    emp_s = (bs["cnt_s"][:, supc] == 0).any(1)
                    emp_a = (bs["cnt_a"][:, supa] == 0).any(1)
                    Fb = bs["F"][name]
                    Sb = np.where(emp_s, np.nan, np.nansum(bs["gs"][name][:, supc] * wr, axis=1))
                    Pb = np.where(emp_a, np.nan, np.nansum(bs["ga"][name][:, supa] * wa[supa], axis=1))
                    qd[L] = {"F": Fb, "S": Sb, "P": Pb}
                    st["intervals"][f"L{L}"] = {
                        "F_s2": pct(Fb), "S_s2": pct(Sb), "P_s2": pct(Pb),
                        "S_minus_F_s2": pct(Sb - Fb), "P_minus_F_s2": pct(Pb - Fb),
                        "width_ratio_S_to_F": (width(pct(Sb)) / width(pct(Fb))) if width(pct(Fb)) else None,
                        "empty_draws_S": int(emp_s.sum()), "empty_draws_P": int(emp_a.sum()),
                        "unreliable": bool(emp_s.sum() > 0.01 * BOOT or emp_a.sum() > 0.01 * BOOT)}
                mrec["steps"][name] = st
                draws[(key, name)] = qd
            rec["months"][key] = mrec

        # --- route A evidence (SPEC §8), 3-day draws of this fold
        bs3 = bsum[3]
        W3 = picks[(f, 3)]
        for cname, per in conds.items():
            msk = (reg == "H") & np.isin(ap, per[f])
            nP, DP = int(msk.sum()), int(day[msk].nunique())
            for name, _, _ in ALL_STEPS:
                if nP == 0:
                    routeA[(cname, name)][f] = {"airports": per[f], "rows": 0, "days": 0,
                                                "prior_advantage_clear": False}
                    continue
                b = bs3["b"][msk]
                k = nb[f][3]
                sb = np.bincount(b, weights=v[name][msk], minlength=k)
                cb = np.bincount(b, minlength=k).astype("float64")
                with np.errstate(invalid="ignore", divide="ignore"):
                    ci = pct((W3 @ sb) / (W3 @ cb))
                ok = bool(nP >= C.N_MIN and DP >= C.D_MIN and ci[1] is not None and ci[1] < 0)
                routeA[(cname, name)][f] = {"airports": per[f], "rows": nP, "days": DP,
                                            "mean_s2": float(v[name][msk].mean()), "ci3_s2": ci,
                                            "prior_advantage_clear": ok}

        # --- per-cell table (SPEC §4.1)
        for c in cells:
            i = code[c]
            row = {"fold": f, "month": C.FOLD_MONTH[f], "airport": c[0], "regime": c[1],
                   "fold_rows": int(ns[i]), "fold_days": D_s[c], "fold_kish_days": kish_s[c],
                   "fold_mass": float(pif[i]), "supported": sup[c], "structural_zero": c[1] != "H"}
            for m in TARGETS[f]:
                row[f"rank_rows_{m}"] = int(rcnt[m].get(c, 0))
                row[f"rank_mass_{m}"] = pir[m].get(c, 0.0)
                rr = rcen[(rcen.rank_month == m) & (rcen.airport == c[0]) & (rcen.regime == c[1])]
                row[f"rank_mode4_rows_{m}"] = int(rr.rank_mode4_rows.sum())
            for L in LS:
                row[f"ci{L}_empty_draws"] = int((bsum[L]["cnt_s"][:, i] == 0).sum())
            for name, pri, nxt in ALL_STEPS:
                row[f"g_s2_{name}"] = float(g[name][i])
                row[f"rmse_gain_s_{name}"] = float(np.sqrt(sse_s[pri][i] / ns[i]) - np.sqrt(sse_s[nxt][i] / ns[i]))
                for L in LS:
                    ci = pct(bsum[L]["gs"][name][:, i]) if c[1] == "H" else [0.0, 0.0]
                    row[f"ci{L}_lo_{name}"], row[f"ci{L}_hi_{name}"] = ci
            cell_rows.append(row)
        for m in TARGETS[f]:
            for c, k in rcnt[m].items():
                if c not in code:
                    rr = rcen[(rcen.rank_month == m) & (rcen.airport == c[0]) & (rcen.regime == c[1])]
                    cell_rows.append({"fold": f, "month": C.FOLD_MONTH[f], "airport": c[0],
                                      "regime": c[1], "fold_rows": 0, "fold_days": 0,
                                      "fold_kish_days": 0.0, "fold_mass": 0.0, "supported": False,
                                      "structural_zero": c[1] != "H", f"rank_rows_{m}": int(k),
                                      f"rank_mass_{m}": pir[m][c],
                                      f"rank_mode4_rows_{m}": int(rr.rank_mode4_rows.sum())})
        for name, _, _ in ALL_STEPS:
            if abs(float((g[name] * ns).sum()) - v[name].sum()) > 1e-6 * max(1.0, abs(v[name].sum())):
                raise SystemExit(f"{f} {name}: the cell table does not sum to the paired SSE")

        # --- predictor tables (SPEC §6)
        pops = {"all": np.ones(n, bool), "le3600": y <= C.CUT, "gt3600": y > C.CUT, "no_nm": nonm}
        for r in C.REGIMES:
            pops[f"regime_{r}"] = reg == r
        nmr = p.regime.astype(str).to_numpy()
        for r in NM_REG:
            pops[f"nm_{r}"] = nmr == r
        rec["predictor_tables"][f] = {
            pop: {"rows": int(msk.sum()),
                  "rmse_s": {lv: float(np.sqrt(sq[lv][msk].mean())) if msk.any() else None for lv in C.LEVELS},
                  "step_s2_per_fold_row": {nm_: float(v[nm_][msk].sum() / n) for nm_, _, _ in ALL_STEPS},
                  "step_s2_per_pop_row": {nm_: float(v[nm_][msk].mean()) if msk.any() else None
                                          for nm_, _, _ in ALL_STEPS}}
            for pop, msk in pops.items()}

        # --- label-free movement, fold side (SPEC §5.4)
        H = reg == "H"
        rec["movement"][f"fold:{f}"] = {
            name: {a: float(np.sqrt(np.mean((heads[nxt][f] - heads[pri][f])[H & (ap == a)] ** 2)))
                   for a in aps if (H & (ap == a)).any()} for name, pri, nxt in ALL_STEPS}

        # --- OOF receipt (SPEC §6), written after every check has passed
        oofs[f] = pd.DataFrame({"MVT_ID_mvt": ids, "fold": f, "y_true": y, "y_pred": pb + heads["v22"][f],
                                "y_pred_v19": pb + heads["v19"][f], "y_pred_v20": pb + heads["v20"][f],
                                "y_pred_v21": pb + heads["v21"][f], "airport": ap, "regime": reg})
        rec["folds"][f] = {"month": C.FOLD_MONTH[f], "rows": n, "sse_reconciled": True,
                           "H_cells": {a: {"rows": int(ns[code[(a, r)]]), "days": D_s[(a, r)],
                                           "supported": sup[(a, r)]} for (a, r) in Hcells},
                           "airports_supported": sup_a}
        print(f"{f}: {len(cells)} cells, SSE reconciled, bootstrap done ({time.time() - t0:.0f} s)", flush=True)

    # ---- ranking-side movement (uploaded bytes, label-free) --------------------------------
    rid = rk.MVT_ID_mvt.to_numpy("float64")
    up = {}
    for lv in C.LEVELS:
        C.check(C.UPLOAD[lv], C.UPLOAD_SHA[lv])
        u = pd.read_parquet(C.UPLOAD[lv])
        u["MVT_ID_mvt"] = u.MVT_ID_mvt.astype("float64")
        up[lv] = u.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(rid).to_numpy("float64")
    rH, rap, rmo = rk.regime.to_numpy() == "H", rk.airport.to_numpy(), rk.month.to_numpy()
    for m in RMONTHS:
        rec["movement"][f"ranking:{m}"] = {
            name: {a: float(np.sqrt(np.mean((up[nxt] - up[pri])[rH & (rmo == m) & (rap == a)] ** 2)))
                   for a in sorted(set(rap)) if (rH & (rmo == m) & (rap == a)).any()}
            for name, pri, nxt in ALL_STEPS}

    # ---- board combination and seconds (SPEC §4.3) -------------------------------------------
    sh = C.SHARES
    for name, _, _ in ALL_STEPS:
        kap, face = pin["steps"][name]["kappa_s_per_s2"], pin["steps"][name]["face_s"]
        mj = rec["months"]["july_loo->2026-07"]["steps"][name]
        ma = rec["months"]["january->2026-01"]["steps"][name]
        X = {x: sh["july_loo"] * mj[f"{x}_s2"] + sh["january"] * ma[f"{x}_s2"] for x in "FSP"}
        if abs(kap * X["F"] - face) > 1e-9:
            raise SystemExit(f"{name}: kappa * F_board {kap * X['F']!r} != face {face!r}")
        br = {"kappa_s_per_s2": kap, "face_s": face, **{f"{x}_s": kap * X[x] for x in "FSP"},
              **{f"{x}_s2": X[x] for x in "FSP"}, "S_over_F": X["S"] / X["F"], "P_over_F": X["P"] / X["F"],
              "S_minus_F_s": kap * (X["S"] - X["F"]), "intervals": {}}
        for L in LS:
            qj, qa = draws[("july_loo->2026-07", name)][L], draws[("january->2026-01", name)][L]
            b_ = {x: kap * (sh["july_loo"] * qj[x] + sh["january"] * qa[x]) for x in "FSP"}
            with np.errstate(invalid="ignore", divide="ignore"):
                br["intervals"][f"L{L}"] = {
                    "F_s": pct(b_["F"]), "S_s": pct(b_["S"]), "P_s": pct(b_["P"]),
                    "S_minus_F_s": pct(b_["S"] - b_["F"]), "P_minus_F_s": pct(b_["P"] - b_["F"]),
                    "S_over_F": pct(b_["S"] / b_["F"]), "P_over_F": pct(b_["P"] / b_["F"]),
                    "width_ratio_S_to_F": width(pct(b_["S"])) / width(pct(b_["F"])),
                    "empty_draws_S": int(np.isnan(b_["S"]).sum()),
                    "empty_draws_P": int(np.isnan(b_["P"]).sum()),
                    "unreliable": bool(np.isnan(b_["S"]).sum() > 0.01 * BOOT)}
        rec["board"][name] = br
        print(f"{name}: face {face:.5f} s | S {br['S_s']:+.5f} s | P {br['P_s']:+.5f} s | S/F "
              f"{br['S_over_F']:.3f} | 3-day S {br['intervals']['L3']['S_s']}", flush=True)

    # ---- comparison 1 (SPEC §4.8.1): S against the faces, no board number -----------------------
    Fv = np.array([rec["board"][k]["F_s"] for k in NAMES])
    Sv = np.array([rec["board"][k]["S_s"] for k in NAMES])
    Pv = np.array([rec["board"][k]["P_s"] for k in NAMES])
    order = lambda x: [NAMES[i] for i in np.argsort(-x)]
    ratio = Sv / Fv
    hi, lo = int(np.argmax(ratio)), int(np.argmin(ratio))
    ci_hi = rec["board"][NAMES[hi]]["intervals"]["L3"]["S_over_F"]
    ci_lo = rec["board"][NAMES[lo]]["intervals"]["L3"]["S_over_F"]
    rec["comparison_faces"] = {
        "F_s": dict(zip(NAMES, Fv.tolist())), "S_s": dict(zip(NAMES, Sv.tolist())),
        "P_s": dict(zip(NAMES, Pv.tolist())),
        "sign_kept_S": bool(np.all(np.sign(Sv) == np.sign(Fv))),
        "sign_kept_P": bool(np.all(np.sign(Pv) == np.sign(Fv))),
        "order_F": order(Fv), "order_S": order(Sv), "order_P": order(Pv),
        "order_kept_S": order(Sv) == order(Fv), "order_kept_P": order(Pv) == order(Fv),
        "S_over_F": dict(zip(NAMES, ratio.tolist())),
        "S_over_F_spread": float(ratio.max() / ratio.min()),
        "falsifier_spread_ge_2_and_disjoint_3day": bool(ratio.max() / ratio.min() >= 2
                                                        and ci_lo[1] < ci_hi[0]),
        "S_minus_F_s_ci3": {k: rec["board"][k]["intervals"]["L3"]["S_minus_F_s"] for k in NAMES}}

    # ---- routing (SPEC §8), fold evidence only -----------------------------------------------
    bw_mass = {}
    for cname, per in conds.items():
        mass = {m: float(rsub[m][(rsub[m].regime == "H") & rsub[m].airport.isin(per[m])].shape[0]
                         / len(rsub[m])) for m in RMONTHS}
        bw_mass[cname] = sh["july_loo"] * mass["2026-07"] + sh["january"] * mass["2026-01"]
    ra = []
    for (cname, name), per in routeA.items():
        clear = [f for f in C.FOLDS if per[f]["prior_advantage_clear"]]
        ra.append({"condition": cname, "contrast": name, "folds_clear": clear,
                   "fires": len(clear) >= 2, "board_weighted_rank_mass_of_P": bw_mass[cname],
                   "by_fold": per})
    fired = sorted((r for r in ra if r["fires"]), key=lambda r: -r["board_weighted_rank_mass_of_P"])
    rel = {k: abs(rec["board"][k]["S_s2"] - rec["board"][k]["F_s2"]) / abs(rec["board"][k]["F_s2"]) for k in NAMES}
    bw = pin["census"]["board_weighted"]
    small = bool(max(rel.values()) <= 0.10 and bw["unsupported_share_of_rank_H"] <= 0.10)
    fold_m = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
    cyc = lambda d: min(d % 12, (-d) % 12)
    pairs = []
    for f, fm in fold_m.items():
        for i, o in enumerate(BUILT):
            for r_ in BUILT[i + 1:]:
                do, dr = midx(o) - midx(fm), midx(r_) - midx(fm)
                if cyc(do) != cyc(dr) or abs(abs(do) - abs(dr)) < 6:
                    continue
                pairs.append({"fold": f, "pair": [o, r_], "cyclic_distance": cyc(do),
                              "time_distance": [abs(do), abs(dr)],
                              "already_measured": (f, frozenset({o, r_})) in MEASURED})
    ident = [q for q in pairs if not q["already_measured"]]
    route = "A" if fired else ("B" if (small and ident) else "C")
    rec["routing"] = {"route_A_tests": ra, "route_A_fired": [f"{r['condition']} x {r['contrast']}" for r in fired],
                      "route_A_choice": (f"{fired[0]['condition']} x {fired[0]['contrast']}" if fired else None),
                      "route_B": {"relative_change_S_vs_F": rel,
                                  "unsupported_share_of_rank_H": bw["unsupported_share_of_rank_H"],
                                  "i_composition_small": small, "ii_pairs_time_gap_ge_6": pairs,
                                  "ii_identifiable_unmeasured": ident},
                      "route": route}
    print(f"routing: route {route}; A fired {rec['routing']['route_A_fired']}; B (i) {small}, "
          f"(ii) {len(ident)} unmeasured pairs", flush=True)

    # ---- comparison 2 (SPEC §4.8.2): board gains, read only (score-feedback rule), last ------------------------
    B = np.array([C.BOARD_GAIN[k] for k in NAMES])
    rec["comparison_read_only"] = {
        "board_gain_s": dict(zip(NAMES, B.tolist())), "board_order": order(B),
        **{x: {"values_s": dict(zip(NAMES, a.tolist())), "sign_agrees": bool(np.all(np.sign(a) == np.sign(B))),
               "order_agrees": order(a) == order(B), "E_abs_s": float(np.abs(a - B).sum())}
           for x, a in (("F", Fv), ("S", Sv), ("P", Pv))},
        "gap_v21_v22_s": {"faces": float(Fv[1] + Fv[2]), "board": float(B[1] + B[2]),
                          "gap": float(Fv[1] + Fv[2] - B[1] - B[2]),
                          "accounted_by_S": float(Fv[1] + Fv[2] - Sv[1] - Sv[2]),
                          "accounted_by_P": float(Fv[1] + Fv[2] - Pv[1] - Pv[2])},
        "note": "read only; nothing is fitted, selected, scaled or changed from these (D11)"}
    for f in C.FOLDS:
        OOF[f].parent.mkdir(parents=True, exist_ok=True)
        C.NO.refuse_existing([OOF[f]], overwrite=False)
        oofs[f].to_parquet(OOF[f], index=False)
        rec["folds"][f]["oof"] = {"path": str(OOF[f].relative_to(C.ROOT)), "sha256": C.sha_file(OOF[f])}
    pd.DataFrame(cell_rows).to_csv(CELLS, index=False)
    rec["cells_csv"] = {"path": str(CELLS.relative_to(C.ROOT)), "sha256": C.sha_file(CELLS)}
    rec["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, rec)
    print(json.dumps(rec["comparison_faces"], indent=1))
    print(json.dumps(rec["comparison_read_only"], indent=1))
    print(f"written {OUT.relative_to(C.ROOT)} ({rec['seconds']} s)")


if __name__ == "__main__":
    main()
