"""exp-056-q12advocate: price an iteration on V (SPEC 3.7-3.9, 4). Applies the registered gate.

Runs only after all three rotations' predictions are frozen: it checks each fit record's freeze
time against the V label read. Writes per-row losses of every arm and design (artifacts/, sha in
the record), per-date sums, the paired contrasts with moving-block intervals over consecutive
UTC dates (1, 2, 3 days; fixed draws), the board recombination at the ranking shares, the gate
verdict, the five diagnostics, the COMMON.md tables and the OOF files of base, zero and true.
`--iteration 2` prices route A's fits (`it2-fit-<r>.json`) with the same rule and adds the
contrast of iteration 2's `true` over iteration 1's.

    .venv/bin/python experiments/exp-056-q12advocate/code/price.py
    .venv/bin/python experiments/exp-056-q12advocate/code/price.py --iteration 2
    .venv/bin/python experiments/exp-056-q12advocate/code/price.py --dry --scenario null
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402
import fit as F                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

GATE_CONTROLS = ("zero", "shuffle", "gmean", "donor", "sham")
MAIN_ARMS = ("zero", "true", "gmean", "donor", "shuffle")
SHAMS = tuple(f"sham{s:02d}" for s in E.SHAM_SEEDS)
#: dry outputs carry the pricing revision (p1: no secondary read; p2: sparse = n_A < 50, too few rows)
PRICE_REV = 3


def arm_loss(y, p, pop, h):
    return (y - E.final_pred(p, h, pop)) ** 2


def fold_stats(la, lb, di, k, draws):
    """Gain of B over A on one fold: point, per-date sums, block intervals of the mean."""
    v = la - lb
    num = np.bincount(di, weights=v, minlength=k)
    cnt = np.bincount(di, minlength=k).astype("float64")
    out = {"mse_gain_s2_per_row": float(v.mean()),
           "rmse_gain_s": float(np.sqrt(la.mean()) - np.sqrt(lb.mean())),
           "date_sums_s2": [float(x) for x in num]}
    for L, idx in draws.items():
        out[f"ci_block_{L}d"] = E.ci(num[idx].sum(1) / cnt[idx].sum(1))
    return out


def board_stats(la: dict, lb: dict, di: dict, k: dict, draws: dict):
    ma = sum(E.BOARD[r] * la[r].mean() for r in E.BOARD)
    mb = sum(E.BOARD[r] * lb[r].mean() for r in E.BOARD)
    out = {"board_gain_s": float(np.sqrt(ma) - np.sqrt(mb))}
    for L in E.BLOCK_LENGTHS:
        m0 = np.zeros(E.BOOT)
        m1 = np.zeros(E.BOOT)
        for r, s in E.BOARD.items():
            e0 = np.bincount(di[r], weights=la[r], minlength=k[r])
            e1 = np.bincount(di[r], weights=lb[r], minlength=k[r])
            c = np.bincount(di[r], minlength=k[r]).astype("float64")
            idx = draws[r][L]
            m0 += s * e0[idx].sum(1) / c[idx].sum(1)
            m1 += s * e1[idx].sum(1) / c[idx].sum(1)
        out[f"ci_block_{L}d"] = E.ci(np.sqrt(m0) - np.sqrt(m1))
    return out


def rmse_table(y, pred, nonm, ids, fold):
    e2 = (y - pred) ** 2
    t = {"rmse": float(np.sqrt(e2.mean())),
         "rmse_le3600": float(np.sqrt(e2[y <= 3600].mean())),
         "rmse_gt3600": float(np.sqrt(e2[y > 3600].mean())) if (y > 3600).any() else None,
         "rmse_nonm": float(np.sqrt(e2[nonm].mean())) if nonm.any() else None,
         "rows": int(len(y)), "rows_gt3600": int((y > 3600).sum()), "rows_nonm": int(nonm.sum())}
    t["rmse_clean"] = float(E.D.rmse_clean(y, pred, ids, fold))
    return t


def gate(res: dict) -> dict:
    """SPEC 3.8, applied mechanically: true beats each control on both board folds (fold point >
    0), board point > 0 with the 1-day board interval above zero, and no 2- or 3-day interval
    (board or board fold) entirely below zero."""
    out = {}
    for c in GATE_CONTROLS:
        cc = res["contrasts"][f"true_vs_{c}"]["main"]
        f_pos = {r: cc["folds"][r]["mse_gain_s2_per_row"] > 0 for r in E.BOARD}
        b = cc["board"]
        rev = any(b[f"ci_block_{L}d"][1] < 0 for L in (2, 3)) or any(
            cc["folds"][r][f"ci_block_{L}d"][1] < 0 for r in E.BOARD for L in (2, 3))
        ok = all(f_pos.values()) and b["board_gain_s"] > 0 and b["ci_block_1d"][0] > 0 and not rev
        out[c] = {"fold_points_positive": f_pos, "board_gain_s": b["board_gain_s"],
                  "board_ci_1d": b["ci_block_1d"], "resolved_reversal_2d_3d": rev, "pass": ok}
    out["advance"] = all(out[c]["pass"] for c in GATE_CONTROLS)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--scenario", choices=F.SCENARIOS, default=None)
    ap.add_argument("--iteration", type=int, choices=(1, 2), default=1,
                    help="2 = route A, the support-aware prior (SPEC 3.9 and 4)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.dry != (a.scenario is not None):
        raise SystemExit("--dry needs --scenario, and --scenario needs --dry")
    n_it = a.iteration
    it = "it2-" if n_it == 2 else ""
    tag = f"dry{E.DRY_VERSION}-{a.scenario}-" if a.dry else ""
    otag = f"{tag}p{PRICE_REV}-" if a.dry else ""
    out_rec = E.ANALYSIS / f"{otag}iter{n_it}.json"
    rows_p = (E.WORK / "dry" if a.dry else E.ART) / f"{otag}iter{n_it}-rows.parquet"
    draws_p = (E.WORK / "dry" if a.dry else E.ART) / f"{otag}iter{n_it}-draws.npz"
    oof_dir = E.WORK / "dry" if a.dry else E.EXP / "oof"
    oof_paths = [oof_dir / f"{otag}{it}{arm}-iso_{r}_v.parquet" for arm in ("base", "zero", "true")
                 for r in E.ORDER]
    NO.refuse_existing([out_rec, rows_p, draws_p] + oof_paths, a.overwrite)
    t0 = time.time()
    launch = E.hold_load("price")

    inputs, fits, z, h, fits1, h1 = {}, {}, {}, {}, {}, {}
    for r in E.ORDER:
        ip = E.ANALYSIS / (f"dry{E.DRY_VERSION}-inputs-{r}.json" if a.dry else f"inputs-{r}.json")
        fp = E.ANALYSIS / f"{tag}{it}fit-{r}.json"
        inputs[r], fits[r] = E.load_json(ip), E.load_json(fp)
        checks = [(inputs[r], "npz"), (fits[r], "pred_npz")]
        if n_it == 2:          # iteration 1's corrections, for the iteration-2-over-1 contrast
            fits1[r] = E.load_json(E.ANALYSIS / f"{tag}fit-{r}.json")
            checks.append((fits1[r], "pred_npz"))
        for rec, key in checks:
            if E.sha_file(E.ROOT / rec[key]) != rec[f"{key}_sha256"]:
                raise SystemExit(f"{rec[key]} is not the frozen file")
        z[r] = np.load(E.ROOT / inputs[r]["npz"])
        h[r] = np.load(E.ROOT / fits[r]["pred_npz"])
        if n_it == 2:
            h1[r] = np.load(E.ROOT / fits1[r]["pred_npz"])
    last_freeze = max(fits[r]["freeze_unix"] for r in E.ORDER)

    # ---- the only label reads of this job: V, after every rotation is frozen --------------
    y, p, pop, di, k, nonm, ids = {}, {}, {}, {}, {}, {}, {}
    for r in E.ORDER:
        V = E.ROT[r][0]
        ids[r] = z[r]["ids_V"]
        if a.dry:
            y[r] = F.synthetic_labels(z[r], a.scenario, "V", 301 + E.ORDER.index(r))
        else:
            y[r] = E.dep_labels(V, ids[r], "scoring month V, after the freeze")
        if not np.all(np.isfinite(y[r])):
            raise SystemExit(f"{V}: a departure without a label")
        p[r], pop[r], nonm[r] = z[r]["p_V"], z[r]["pop_V"], z[r]["nonm_V"]
        di[r] = z[r]["day_V"]
        k[r] = int(di[r].max()) + 1
    for rd in E.LABEL_READS:
        if rd["unix"] <= last_freeze:
            raise SystemExit("a V label was read before every rotation's predictions were frozen")

    draws = {r: {L: E.block_draws(k[r], L, E.draw_seed(r, L)) for L in E.BLOCK_LENGTHS}
             for r in E.ORDER}

    # ---- per-row losses of every arm and design ---------------------------------------------
    loss = {r: {"base": (y[r] - p[r]) ** 2} for r in E.ORDER}
    for r in E.ORDER:
        for arm in list(MAIN_ARMS) + list(SHAMS):
            for dsg in F.DESIGNS:
                loss[r][f"{arm}|{dsg}"] = arm_loss(y[r], p[r], pop[r], h[r][f"h_{arm}_{dsg}"])
        for dsg in F.DESIGNS:
            loss[r][f"sham|{dsg}"] = np.mean([loss[r][f"{s}|{dsg}"] for s in SHAMS], axis=0)
        if n_it == 2:
            loss[r]["it1true|main"] = arm_loss(y[r], p[r], pop[r], h1[r]["h_true_main"])

    def contrast(A_: str, B_: str) -> dict:
        folds = {r: fold_stats(loss[r][A_], loss[r][B_], di[r], k[r], draws[r]) for r in E.ORDER}
        board = board_stats({r: loss[r][A_] for r in E.BOARD}, {r: loss[r][B_] for r in E.BOARD},
                            {r: di[r] for r in E.BOARD}, {r: k[r] for r in E.BOARD},
                            {r: draws[r] for r in E.BOARD})
        return {"A": A_, "B": B_, "folds": folds, "board": board}

    res = {"what": __doc__.strip().splitlines()[0], "dry": a.dry, "scenario": a.scenario,
           "launch": launch, "code_sha256": E.code_shas(), "boot": E.BOOT,
           "draw_seeds": {r: {L: E.draw_seed(r, L) for L in E.BLOCK_LENGTHS} for r in E.ORDER},
           "board_shares": E.BOARD, "label_reads": E.LABEL_READS,
           "inputs": {r: {"record_npz_sha256": inputs[r]["npz_sha256"],
                          "fit_pred_npz_sha256": fits[r]["pred_npz_sha256"],
                          "fit_freeze_unix": fits[r]["freeze_unix"]} for r in E.ORDER},
           "picks": {r: {arm: {"lam": fits[r]["arms"][arm]["lam"], "beta": fits[r]["arms"][arm]["beta"]}
                         | ({"c": fits[r]["arms"][arm].get("c")} if n_it == 2 else {})
                         for arm in fits[r]["arms"]} for r in E.ORDER},
           "contrasts": {}}
    if n_it == 2:
        res["iteration"] = 2
        res["contrasts"]["true_it2_vs_true_it1"] = {"main": contrast("it1true|main", "true|main")}
    # registered contrasts (SPEC 3.7)
    for c in GATE_CONTROLS:
        res["contrasts"][f"true_vs_{c}"] = {dsg: contrast(f"{c}|{dsg}", f"true|{dsg}")
                                            for dsg in F.DESIGNS}
    for arm in list(MAIN_ARMS) + ["sham"]:
        res["contrasts"][f"{arm}_vs_base"] = {"main": contrast("base", f"{arm}|main")}
    for arm in ("zero", "true"):
        res["contrasts"][f"{arm}_expo_vs_expo_ref"] = contrast(f"{arm}|expo_ref", f"{arm}|expo")
        res["contrasts"][f"{arm}_originS_vs_originI"] = contrast(f"{arm}|originI", f"{arm}|originS")
    res["sham_seeds"] = {s: contrast(f"{s}|main", "true|main")["board"] for s in SHAMS}
    res["gate"] = gate(res)

    # ---- secondary read, not the gate (SPEC 3.9): exp-054 7.6's sparse-stand route ---------
    # rows of P whose stand key has fewer than 500 departures in the base months A
    sparse = {}
    for r in E.ORDER:
        cs = z[r]["code_V_stand"]
        nA = np.where(cs >= 0, z[r]["nA_stand"][np.maximum(cs, 0)], -1)
        sparse[r] = pop[r] & (nA >= 0) & (nA < E.N_A_BINS[1])

    def contrast_on(mask: dict, A_: str, B_: str) -> dict:
        folds = {r: fold_stats(loss[r][A_][mask[r]], loss[r][B_][mask[r]], di[r][mask[r]], k[r],
                               draws[r]) for r in E.ORDER}
        board = board_stats({r: loss[r][A_][mask[r]] for r in E.BOARD},
                            {r: loss[r][B_][mask[r]] for r in E.BOARD},
                            {r: di[r][mask[r]] for r in E.BOARD}, {r: k[r] for r in E.BOARD},
                            {r: draws[r] for r in E.BOARD})
        for f_ in folds.values():
            f_.pop("date_sums_s2")
        return {"A": A_, "B": B_, "folds": folds, "board": board}

    sec = {"stratum": "P rows with a stand key of n_A(stand) < 500", "not_the_gate": True,
           "rows": {r: int(sparse[r].sum()) for r in E.ORDER}}
    for c in GATE_CONTROLS:
        sec[f"true_vs_{c}"] = contrast_on(sparse, f"{c}|main", "true|main")
    sv = sec["true_vs_shuffle"]["folds"]
    sec["exp054_route_true_beats_shuffle_1d_above_zero_both_board_folds"] = bool(
        all(sv[r]["ci_block_1d"][0] > 0 for r in E.BOARD))
    res["secondary_sparse_stand"] = sec

    # ---- diagnostics: residual alignment and support (SPEC 3.9) ----------------------------
    diag = {}
    for r in E.ORDER:
        codes = {f: z[r][f"code_V_{f}"] for f in E.FAMS}
        qf = {}
        for f in E.PRIOR_FAMS:
            c = codes[f]
            qq = np.zeros(len(c))
            qq[c >= 0] = z[r][f"prior_true_{f}"][c[c >= 0]]
            qf[f] = qq
        q = qf["stand"] + qf["rwy"] + qf["op"]
        pz = E.final_pred(p[r], h[r]["h_zero_main"], pop[r])
        m = pop[r]
        e = y[r] - pz

        def align(mask):
            qq = q[mask]
            den = float(np.sum(qq * qq))
            return {"rows": int(mask.sum()),
                    "alignment": float(np.sum(e[mask] * qq) / den) if den > 0 else None,
                    "corr": float(np.corrcoef(e[mask], qq)[0, 1]) if mask.sum() > 2 and qq.std() > 0 else None}
        cs = codes["stand"]
        nA = np.where(cs >= 0, z[r]["nA_stand"][np.maximum(cs, 0)], -1)
        sS = z[r]["code_S_stand"]
        cntS = np.bincount(sS[(sS >= 0) & z[r]["pop_S"]], minlength=len(z[r]["nA_stand"]))
        nS = np.where(cs >= 0, cntS[np.maximum(cs, 0)], -1)
        strata_A = {"stand_missing": nA < 0, "nA_lt50": (nA >= 0) & (nA < E.N_A_BINS[0]),
                    "nA_50_499": (nA >= E.N_A_BINS[0]) & (nA < E.N_A_BINS[1]),
                    "nA_ge500": nA >= E.N_A_BINS[1]}
        strata_S = {"stand_missing": nS < 0, "nS_0": nS == 0,
                    "nS_1_9": (nS >= E.N_S_BINS[0]) & (nS < E.N_S_BINS[1]),
                    "nS_10_49": (nS >= E.N_S_BINS[1]) & (nS < E.N_S_BINS[2]),
                    "nS_ge50": nS >= E.N_S_BINS[2]}
        v_tz = loss[r]["zero|main"] - loss[r]["true|main"]
        d = {"alignment_all": align(m),
             "alignment_by_family": {f: (lambda qq: {"alignment": float(np.sum(e[m] * qq[m]) / np.sum(qq[m] ** 2))
                                                     if np.sum(qq[m] ** 2) > 0 else None})(qf[f])
                                     for f in E.PRIOR_FAMS},
             "by_nA_stand": {}, "by_nS_stand": {}}
        for name, st in strata_A.items():
            mm = m & st
            d["by_nA_stand"][name] = {**align(mm), "true_vs_zero_sse_gain_Ms2": float(v_tz[mm].sum() / 1e6)}
        for name, st in strata_S.items():
            mm = m & st
            d["by_nS_stand"][name] = {**align(mm), "true_vs_zero_sse_gain_Ms2": float(v_tz[mm].sum() / 1e6)}
        diag[r] = d
    res["diagnostics"] = diag
    res["inner_alignment_on_I"] = {r: fits[r]["inner_alignment_true_unit_prior_on_I"] for r in E.ORDER}

    # ---- COMMON.md tables -------------------------------------------------------------------
    tables = {}
    for r in E.ORDER:
        fold = f"iso_{r}_v"
        tables[r] = {"base": rmse_table(y[r], p[r], nonm[r], ids[r], fold)}
        for arm in MAIN_ARMS:
            tables[r][arm] = rmse_table(y[r], E.final_pred(p[r], h[r][f"h_{arm}_main"], pop[r]),
                                        nonm[r], ids[r], fold)
    res["tables"] = tables

    # ---- write ------------------------------------------------------------------------------
    NO.refuse_existing([out_rec, rows_p, draws_p] + oof_paths, a.overwrite)
    rows_p.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for r in E.ORDER:
        df = pd.DataFrame({"rotation": r, "MVT_ID_mvt": ids[r], "day_index": di[r],
                           "pop": pop[r]})
        for key, v in loss[r].items():
            df[f"loss:{key}"] = v
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_parquet(rows_p, index=False)
    np.savez(draws_p, **{f"{r}_{L}": draws[r][L] for r in E.ORDER for L in E.BLOCK_LENGTHS})
    oof_dir.mkdir(parents=True, exist_ok=True)
    for r in E.ORDER:
        for arm in ("base", "zero", "true"):
            pred = p[r] if arm == "base" else E.final_pred(p[r], h[r][f"h_{arm}_main"], pop[r])
            pd.DataFrame({"MVT_ID_mvt": ids[r], "fold": f"iso_{r}_v", "y_true": y[r],
                          "y_pred": pred}).to_parquet(oof_dir / f"{otag}{it}{arm}-iso_{r}_v.parquet",
                                                      index=False)
    res["artifacts"] = {"rows": {"path": str(rows_p.relative_to(E.ROOT)), "sha256": E.sha_file(rows_p)},
                        "draws": {"path": str(draws_p.relative_to(E.ROOT)), "sha256": E.sha_file(draws_p)},
                        "oof": {str(q.relative_to(E.ROOT)): E.sha_file(q) for q in oof_paths}}
    res["seconds"] = round(time.time() - t0, 1)
    out_rec.write_text(json.dumps(res, indent=1, sort_keys=True))
    g = res["gate"]
    print(json.dumps({c: g[c]["pass"] for c in GATE_CONTROLS} | {"advance": g["advance"]}), flush=True)
    for c in GATE_CONTROLS:
        print(f"  true vs {c:8s} board {g[c]['board_gain_s']:+.4f} s  1d {g[c]['board_ci_1d']}", flush=True)
    print(f"  sparse-stand secondary (rows {sec['rows']}): exp-054 route "
          f"{sec['exp054_route_true_beats_shuffle_1d_above_zero_both_board_folds']}", flush=True)
    print(f"written {out_rec.relative_to(E.ROOT)} ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
