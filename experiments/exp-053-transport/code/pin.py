"""exp-053 iteration 1, step 1: pins, all-row face reconciliation, and the input-only support
census.  ZERO FIT.  No stratum loss is computed here (SPEC, registration order).

1. Every source of SPEC §2 is sha-checked; the ordered ids, the ownership masks, the no-NM mask
   and every head vector are hashed into the record.
2. Structural zeros (SPEC §3): each fold head vector is 0 outside H; the ownership columns of
   class vectors v22..v25 are equal row for row; each uploaded step changes only ranking H rows.
3. Face reconciliation (SPEC §2): each step's all-row face from the vectors equals the recorded
   full-precision face to 1e-9 s, and each fold's all-row mean paired loss equals the recorded
   `mse_gain_s2`.  The per-fold paired SSE is recorded for the stratum tables to sum back to.
4. Census (counts only): per (month, airport, regime) fold rows, distinct days, Kish days,
   ranking rows and masses, the H support flag, mode-4 ranking rows; per airport-month clock
   coverage; the SC-U and SC-J memberships of SPEC §8.

    .venv/bin/python experiments/exp-053-transport/code/pin.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import common as C

OUT = C.ANALYSIS / "pin.json"
EXT = C.ANALYSIS / "extcount.json"
#: SPEC §8 SC-U(k): the months each step adds, per head (fold f / ranking month)
ADDED = {"v20": {f: ["2025-11"] for f in C.FOLDS},
         "v21": {f: ["2025-10", "2025-03"] for f in C.FOLDS},
         "v22": {"july_loo": ["2025-08"], "december": ["2025-02"], "january": ["2025-02"]}}
ADDED_RANK = {"v20": {"2026-07": ["2025-11"], "2026-01": ["2025-11"]},
              "v21": {"2026-07": ["2025-10", "2025-03"], "2026-01": ["2025-10", "2025-03"]},
              "v22": {"2026-07": ["2025-08"], "2026-01": ["2025-02"]}}


def main() -> None:
    C.NO.refuse_existing([OUT], overwrite=False)
    t0 = time.time()
    rec = {"what": __doc__.strip().splitlines()[0], "sources": {}, "folds": {}, "steps": {},
           "ranking": {}, "census": {}, "support_conditions": {}}

    # ---- 1. sources --------------------------------------------------------------------
    heads = C.load_heads()
    for lv, (p, want) in C.HEADS.items():
        rec["sources"][f"head_{lv}"] = {"path": str(p.relative_to(C.ROOT)), "sha256": want}
    pan, reg, nonm = {}, {}, {}
    for f in C.FOLDS:
        p = C.load_panel(f)
        pan[f] = p
        ids = p.MVT_ID_mvt.to_numpy("float64")
        if not p.MVT_ID_mvt.is_unique:
            raise SystemExit(f"{f}: panel ids are not unique")
        for lv in C.LEVELS:
            if len(heads[lv][f]) != len(p):
                raise SystemExit(f"{f}: head {lv} has {len(heads[lv][f])} rows, panel {len(p)}")
        pb = p.pbase.to_numpy("float64")
        if not np.array_equal(p.d_v22.to_numpy("float64"), heads["v22"][f]):
            raise SystemExit(f"{f}: the panel's d_v22 is not the pinned v22 head")
        if not np.array_equal(pb + heads["v22"][f], p.p_v22.to_numpy("float64")):
            raise SystemExit(f"{f}: p_v22 != pbase + d_v22")
        reg[f] = C.fold_regime(p)
        nonm[f] = C.no_nm_mask(ids, f)
        H = reg[f] == "H"
        moved_out = {lv: int((heads[lv][f][~H] != 0).sum()) for lv in C.LEVELS}
        if any(moved_out.values()):
            raise SystemExit(f"{f}: a head vector moves rows outside H: {moved_out}")
        rec["sources"][f"panel_{f}"] = {"path": str(C.PANEL[f].relative_to(C.ROOT)),
                                        "sha256": C.PANEL_SHA[f]}
        rec["folds"][f] = {
            "month": C.FOLD_MONTH[f], "rows": int(len(p)),
            "ids_sha256": C.sha_arr(ids), "pbase_sha256": C.sha_arr(pb),
            "regime_sha256": C.sha_arr(np.array([C.REGIMES.index(r) for r in reg[f]], "int8")),
            "regime_rows": {r: int((reg[f] == r).sum()) for r in C.REGIMES},
            "no_nm_rows": int(nonm[f].sum()), "no_nm_sha256": C.sha_arr(nonm[f]),
            "days": int(p.day.nunique()),
            "head_rows_moved": {lv: int((heads[lv][f] != 0).sum()) for lv in C.LEVELS},
            "head_rows_moved_outside_H": moved_out,
            "head_vector_sha256": {lv: C.sha_arr(heads[lv][f]) for lv in C.LEVELS}}
        print(f"{f}: {len(p)} rows, regimes {rec['folds'][f]['regime_rows']}, no-NM "
              f"{int(nonm[f].sum())}; every head is 0 outside H", flush=True)

    # ---- 3. face reconciliation (all-row only) -----------------------------------------
    mse = {lv: {} for lv in C.LEVELS}
    for f in C.FOLDS:
        y = pan[f].y_true.to_numpy("float64")
        pb = pan[f].pbase.to_numpy("float64")
        for lv in C.LEVELS:
            mse[lv][f] = float(np.mean((y - pb - heads[lv][f]) ** 2))
    board = {lv: float(np.sqrt(sum(C.SHARES[f] * mse[lv][f] for f in C.BOARD_FOLDS)))
             for lv in C.LEVELS}
    for name, pri, nxt in C.STEPS + (C.CUMULATIVE,):
        face = board[pri] - board[nxt]
        st = {"prior": pri, "next": nxt, "face_s": face, "board_rmse_prior": board[pri],
              "board_rmse_next": board[nxt], "by_fold": {}}
        for f in C.FOLDS:
            y = pan[f].y_true.to_numpy("float64")
            pb = pan[f].pbase.to_numpy("float64")
            v = (y - pb - heads[pri][f]) ** 2 - (y - pb - heads[nxt][f]) ** 2
            st["by_fold"][f] = {"paired_sse": float(v.sum()), "mse_gain_s2": float(v.mean()),
                                "rows_moved_by_step": int((heads[nxt][f] != heads[pri][f]).sum())}
        g_board = sum(C.SHARES[f] * st["by_fold"][f]["mse_gain_s2"] for f in C.BOARD_FOLDS)
        st["board_s2_gain"] = g_board
        st["kappa_s_per_s2"] = face / g_board
        if name in C.FACE_RECORD:
            path, key = C.FACE_RECORD[name]
            pr = json.loads(path.read_text())["prices"][key]
            want = pr["board_all_row"]["gain_s"]
            st["recorded_face_s"] = want
            st["recorded_path"] = f"{path.relative_to(C.ROOT)}:prices.{key}"
            st["face_diff_s"] = face - want
            if abs(face - want) > 1e-9:
                raise SystemExit(f"{name}: face {face!r} != recorded {want!r}")
            for f in C.FOLDS:
                w = pr["by_fold"][f]["mse_gain_s2"]
                d = st["by_fold"][f]["mse_gain_s2"] - w
                st["by_fold"][f]["recorded_mse_gain_s2"] = w
                st["by_fold"][f]["diff"] = d
                if abs(d) > 1e-7:
                    raise SystemExit(f"{name} {f}: mse gain differs from the record by {d!r}")
        rec["steps"][name] = st
        print(f"{name}: face {face:.12f} s" + (f" = recorded {st['recorded_face_s']!r} "
              f"(diff {st['face_diff_s']:.2e})" if "recorded_face_s" in st else " (cumulative)"),
              flush=True)

    # ---- 2. ranking side: class-vector stability and uploaded step rows -----------------
    rk = C.load_ranking()
    rid = rk.MVT_ID_mvt.to_numpy("float64")
    rec["sources"]["ranking"] = {"path": str(C.RANKING.relative_to(C.ROOT)), "sha256": C.RANKING_SHA}
    stab = {}
    for v in ("v22", "v23", "v24"):
        eq = {}
        for m in ("2026-01", "2026-07"):
            C.check(C.CLASS[v][m], C.CLASS_SHA[v][m])
            c = pd.read_parquet(C.CLASS[v][m])
            c["MVT_ID_mvt"] = c.MVT_ID_mvt.astype("float64")
            c = c.set_index("MVT_ID_mvt")
            mine = rk[rk.month == m].set_index("MVT_ID_mvt")
            if not c.index.sort_values().equals(mine.index.sort_values()):
                raise SystemExit(f"class {v} {m}: ids differ from v25")
            c = c.reindex(mine.index)
            for col in C.OWNERSHIP_COLS:
                same = bool(np.array_equal(c[col].to_numpy(bool), mine[col].to_numpy(bool)))
                eq[f"{m}:{col}"] = same
                if not same and col in ("occupied_by_exp_046", "guard_held_by_exp_046", "noac"):
                    raise SystemExit(f"class {v} {m}: {col} differs from v25")
        stab[v] = eq
    rec["ranking"]["class_vectors_equal_to_v25"] = stab
    for v in C.CLASS_SHA:
        for m in C.CLASS_SHA[v]:
            rec["sources"][f"class_{v}_{m}"] = {"path": str(C.CLASS[v][m].relative_to(C.ROOT)),
                                                "sha256": C.CLASS_SHA[v][m]}
    up = {}
    for lv in C.LEVELS:
        C.check(C.UPLOAD[lv], C.UPLOAD_SHA[lv])
        u = pd.read_parquet(C.UPLOAD[lv])
        u["MVT_ID_mvt"] = u.MVT_ID_mvt.astype("float64")
        s = u.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(rid)
        if s.isna().any():
            raise SystemExit(f"upload {lv} lacks {int(s.isna().sum())} ranking departures")
        up[lv] = s.to_numpy("float64")
        rec["sources"][f"upload_{lv}"] = {"path": str(C.UPLOAD[lv].relative_to(C.ROOT)),
                                          "sha256": C.UPLOAD_SHA[lv]}
    rreg = rk.regime.to_numpy()
    rec["ranking"]["rows"] = int(len(rk))
    rec["ranking"]["regime_rows"] = {m: {r: int(((rk.month == m).to_numpy() & (rreg == r)).sum())
                                         for r in C.REGIMES} for m in ("2026-01", "2026-07")}
    rec["ranking"]["ids_sha256"] = C.sha_arr(rid)
    rec["ranking"]["regime_sha256"] = C.sha_arr(np.array([C.REGIMES.index(r) for r in rreg], "int8"))
    for name, pri, nxt in C.STEPS + (C.CUMULATIVE,):
        ch = up[nxt] != up[pri]
        by = {r: int((ch & (rreg == r)).sum()) for r in C.REGIMES}
        rec["ranking"][f"upload_rows_changed_{name}"] = by
        if by["G"] or by["A"] or by["C"]:
            raise SystemExit(f"{name}: the uploaded step changes ranking rows outside H: {by}")
        print(f"{name}: upload changes {by['H']} ranking rows, all in H", flush=True)

    # ---- 4. census (counts only) -------------------------------------------------------
    rows = []
    for f in C.FOLDS:
        p = pan[f]
        a = p.airport.astype(str).to_numpy()
        day = p.day.astype(str).to_numpy()
        for (ap, r), idx in pd.Series(np.arange(len(p))).groupby([a, reg[f]]).groups.items():
            ii = np.asarray(idx)
            dc = pd.Series(day[ii]).value_counts().to_numpy()
            rows.append({"fold": f, "month": C.FOLD_MONTH[f], "airport": ap, "regime": r,
                         "fold_rows": int(len(ii)), "fold_days": int(len(dc)),
                         "fold_kish_days": C.kish(dc)})
    cen = pd.DataFrame(rows)
    for f in C.FOLDS:
        n = rec["folds"][f]["rows"]
        cen.loc[cen.fold == f, "fold_mass"] = cen.loc[cen.fold == f, "fold_rows"] / n
    rrows = []
    for m in ("2026-01", "2026-07"):
        sub = rk[rk.month == m]
        g = sub.groupby(["airport", "regime"])
        cnt = g.size()
        m4 = g.chain_mode_4.sum()
        for (ap, r), nrow in cnt.items():
            rrows.append({"rank_month": m, "airport": ap, "regime": r, "rank_rows": int(nrow),
                          "rank_mass": nrow / len(sub), "rank_mode4_rows": int(m4[(ap, r)])})
    rcen = pd.DataFrame(rrows)
    cen["supported"] = ((cen.regime == "H") & (cen.fold_rows >= C.N_MIN)
                        & (cen.fold_days >= C.D_MIN))
    rec["census"]["fold_cells"] = cen.to_dict(orient="records")
    rec["census"]["ranking_cells"] = rcen.to_dict(orient="records")

    # support summary per target month: fold f's H cells against ranking month's H cells
    summ = {}
    for f, targets in {"july_loo": ["2026-07"], "january": ["2026-01"],
                       "december": ["2026-01", "2026-07"]}.items():
        for m in targets:
            fh = cen[(cen.fold == f) & (cen.regime == "H")].set_index("airport")
            rh = rcen[(rcen.rank_month == m) & (rcen.regime == "H")].set_index("airport")
            aps = sorted(set(fh.index) | set(rh.index))
            sup = {ap: bool(ap in fh.index and fh.loc[ap, "supported"]) for ap in aps}
            rmass = {ap: float(rh.loc[ap, "rank_mass"]) if ap in rh.index else 0.0 for ap in aps}
            U = sum(rmass[ap] for ap in aps if not sup[ap])
            Hr = sum(rmass.values())
            summ[f"{f}->{m}"] = {
                "rank_H_mass": Hr, "unsupported_H_mass": U,
                "unsupported_share_of_rank_H": U / Hr if Hr else None,
                "fold_H_mass": float(fh.fold_mass.sum()),
                "unsupported_airports": [ap for ap in aps if not sup[ap] and rmass[ap] > 0],
                "fold_H_cells_failing_support": [ap for ap in fh.index if not fh.loc[ap, "supported"]],
                "rank_mode4_H_rows": int(rh.rank_mode4_rows.sum())}
            print(f"{f}->{m}: ranking H mass {Hr:.4f}, fold H mass {summ[f'{f}->{m}']['fold_H_mass']:.4f}, "
                  f"unsupported {U:.4f} ({summ[f'{f}->{m}']['unsupported_airports']})", flush=True)
    bw_H = sum(C.SHARES[f] * summ[f"{f}->{C.RANK_MONTH[f]}"]["rank_H_mass"] for f in C.BOARD_FOLDS)
    bw_U = sum(C.SHARES[f] * summ[f"{f}->{C.RANK_MONTH[f]}"]["unsupported_H_mass"] for f in C.BOARD_FOLDS)
    rec["census"]["support_summary"] = summ
    rec["census"]["board_weighted"] = {"rank_H_mass": bw_H, "unsupported_H_mass": bw_U,
                                       "unsupported_share_of_rank_H": bw_U / bw_H,
                                       "structural_zero_mass": 1 - bw_H,
                                       "supported_target_mass": 1 - bw_U}
    print(f"board-weighted: ranking H {bw_H:.4f}, unsupported {bw_U:.4f} "
          f"({bw_U / bw_H:.3%} of ranking H)", flush=True)

    # ---- clock coverage and SPEC §8 support conditions ----------------------------------
    cov = {}
    for f in C.FOLDS:
        p = pan[f]
        g = pd.DataFrame({"a": p.airport.astype(str), "c": p.obs_has_clock.astype(bool)}).groupby("a").c
        cov[C.FOLD_MONTH[f]] = {ap: {"rows": int(n), "with_clock": int(k)}
                                for ap, n, k in zip(g.size().index, g.size(), g.sum())}
    for m in ("2026-01", "2026-07"):
        sub = rk[rk.month == m]
        g = sub.groupby("airport").obs_has_clock
        cov[m] = {ap: {"rows": int(n), "with_clock": int(k)}
                  for ap, n, k in zip(g.size().index, g.size(), g.sum())}
    rec["support_conditions"]["clock_coverage"] = cov
    fit_months = {"2025-07": ["2025-12", "2025-01"], "2025-12": ["2025-07", "2025-01"],
                  "2025-01": ["2025-07", "2025-12"],
                  "2026-07": ["2025-07", "2025-12", "2025-01"],
                  "2026-01": ["2025-07", "2025-12", "2025-01"]}
    scj = {}
    for m, fm in fit_months.items():
        jump = []
        for ap, c in cov[m].items():
            num = sum(cov[x].get(ap, {"with_clock": 0})["with_clock"] for x in fm)
            den = sum(cov[x].get(ap, {"rows": 0})["rows"] for x in fm)
            pooled = num / den if den else 0.0
            here = c["with_clock"] / c["rows"] if c["rows"] else 0.0
            if here > 0 and here > 2 * pooled:
                jump.append(ap)
        scj[m] = sorted(jump)
    rec["support_conditions"]["SC_J_airports"] = scj
    ext = json.loads(EXT.read_text())
    rec["sources"]["extcount"] = {"path": str(EXT.relative_to(C.ROOT)), "sha256": C.sha_file(EXT)}
    all_aps = sorted(set().union(*[set(cov[m]) for m in cov]))
    scu = {}
    for k, per in ADDED.items():
        for f, months in per.items():
            n = {ap: sum(ext["months"][mo]["by_airport"].get(ap, 0) for mo in months) for ap in all_aps}
            scu[f"{k}@{C.FOLD_MONTH[f]}"] = sorted(ap for ap in all_aps if n[ap] < C.N_MIN)
    for k, per in ADDED_RANK.items():
        for m, months in per.items():
            n = {ap: sum(ext["months"][mo]["by_airport"].get(ap, 0) for mo in months) for ap in all_aps}
            scu[f"{k}@{m}"] = sorted(ap for ap in all_aps if n[ap] < C.N_MIN)
    rec["support_conditions"]["SC_U_airports"] = scu
    print(f"SC-J airports: {scj}", flush=True)
    print(f"SC-U airports: {scu}", flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, rec)
    print(f"written {OUT.relative_to(C.ROOT)} ({rec['seconds']} s)")


if __name__ == "__main__":
    main()
