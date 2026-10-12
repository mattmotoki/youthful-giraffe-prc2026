"""exp-051 iteration 1, item 1: recorded rule's four saved increments rescored against y - base6 - d_m1.

ZERO FIT.  exp-046 section 47.76.3 names three differences between #4 (F2 >= N2) and recorded rule
(F2 < N2): member 1's months in the fit, the target base, and the marginal contrast.  This
changes ONE half of the second: the SCORING base.  The fitted vectors are untouched; only the
residual they are scored against changes, from r = y - pbase - d_m1 (recorded rule) to
r' = y - base6 - d_m1, with base6 from exp-047's out-of-fold `base6--<fold>.parquet`.  It does
not isolate member-1 redundancy, and nothing here claims it does.

Order, as SPEC.md I1.1 registers it:
  1. rebuild the four matched draws through withinsplit's own source_rows -> draw path and
     check their id hashes against the artifacts and section 47.76.1;
  2. check the four vector files against their recorded sha16;
  3. reproduce redundfit's R_dec, R_jan, pooled R and interval EXACTLY (same seed 4775, same
     draw order), and each G against onemonth's printed mse_gain_s2;
  4. only then rescore, on the SAME day draws, and report R', pooled R', and the paired
     change R' - R.

    .venv/bin/python experiments/exp-051-transferadv/code/rescore.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common as C
from common import A, CF, NO, WS, OM, R

import redundfit as RF                                               # noqa: E402

OUT = C.ANALYSIS / "rescore-base6.json"
OOF = C.ROOT / "experiments/exp-047-queue/oof"
#: section 47.76.1's table, the recorded identities
VEC_SHA16 = {("december", "N2"): "6ee63dabeaa3b3e5", ("december", "F2"): "6fc01e74d49801bd",
             ("january", "N2"): "25b303ccd9c1d7ff", ("january", "F2"): "e868bb8aa5a1e694"}
IDS_SHA16 = {("december", "N2"): "15b6690e5a16582a", ("december", "F2"): "ba70b8fa1bd96ba2",
             ("january", "N2"): "d58a308f1c605528", ("january", "F2"): "6ff267e64e50d5ef"}
RECORDED = {"december": 0.7007, "january": 0.6252, "pooled": 0.6619,
            "pooled_ci95": [0.5250, 0.8311]}


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    NO.refuse_existing([OUT], overwrite=a.overwrite)

    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        mth = A.FOLD_MONTH[f]
        b = CF.build_month(mth, A.prefix_months_for(mth), rule)
        R.assert_mode3(b, mth, "shipped")
        blk[mth] = b
    pan = CF.panel_for(blk, R.THR, {mth: CF.GAP_CUT_S for mth in blk})
    ship, scols = A.shipped_cols(pan)
    d_m1 = C.load_vec(OM.CACHED_M1)

    # 1. the matched draws, rebuilt exactly as withinsplit fitted them
    hashes = {}
    for fold in RF.CODE:
        ws = json.loads((C.E46 / f"analysis/withinsplit-{fold}.json").read_text())["plan"]
        budget = {k: int(v) for k, v in ws["budget_per_airport"].items()}
        for arm in ("N2", "F2"):
            src = WS.source_rows(pan, ship, scols, WS.DESIGN[fold][arm][0])
            i = list(WS.DESIGN[fold]).index(arm)
            dr = WS.draw(src, budget, WS.SEED + 10 + i)
            got = C.ids_sha16(dr.MVT_ID_mvt)
            art = json.loads((C.E46 / f"analysis/onemonth-{RF.tag(fold, arm)}.json")
                             .read_text())["matched_draw"]["mvt_id_sha16"]
            hashes[f"{fold}/{arm}"] = {"rebuilt": got, "artifact": art,
                                       "section_47_76_1": IDS_SHA16[(fold, arm)],
                                       "match": got == art == IDS_SHA16[(fold, arm)],
                                       "rows": int(len(dr))}
            if not hashes[f"{fold}/{arm}"]["match"]:
                raise SystemExit(f"{fold}/{arm}: matched-draw ids do not reproduce: "
                                 f"{hashes[f'{fold}/{arm}']}")
    print("matched draws reproduce:", {k: v["rebuilt"] for k, v in hashes.items()}, flush=True)

    # 2. the vectors
    vec_sha = {}
    for fold in RF.CODE:
        for arm in ("N2", "F2"):
            p = C.E46 / f"work/d-onemonth-{RF.tag(fold, arm)}-255-free.npz"
            s = C.sha256(p)[:16]
            vec_sha[f"{fold}/{arm}"] = {"sha16": s, "recorded": VEC_SHA16[(fold, arm)],
                                        "match": s == VEC_SHA16[(fold, arm)]}
            if s != VEC_SHA16[(fold, arm)]:
                raise SystemExit(f"{p.name}: sha16 {s} is not the recorded {VEC_SHA16[(fold, arm)]}")

    # 3 + 4. redundfit's draws, in its order, then the same draws on the rescored residual
    rng = np.random.default_rng(RF.SEED)
    res, logs, logs2 = {}, {}, {}
    for fold in RF.CODE:
        p = pan[fold]
        y = p.y_true.to_numpy("float64")
        r = p.rbase.to_numpy("float64") - d_m1[fold]
        b6 = pd.read_parquet(OOF / f"base6--{fold}.parquet").set_index("MVT_ID_mvt")
        idx = p.MVT_ID_mvt.to_numpy("float64")
        p6 = b6.y_pred.reindex(idx).to_numpy("float64")
        y6 = b6.y_true.reindex(idx).to_numpy("float64")
        if not np.isfinite(p6).all():
            raise SystemExit(f"{fold}: {int((~np.isfinite(p6)).sum())} rows lack base6")
        if int((np.abs(y6 - y) > 1e-6).sum()):
            raise SystemExit(f"{fold}: base6's y_true differs from the panel's")
        r6 = y - p6 - d_m1[fold]
        codes, uniq = pd.factorize(p.day.to_numpy())
        k = len(uniq)
        draws = rng.integers(0, k, size=(RF.N_BOOT, k))
        G, G6, Gb, G6b, ent = {}, {}, {}, {}, {}
        for arm in ("N2", "F2"):
            t = RF.tag(fold, arm)
            d = np.load(C.E46 / f"work/d-onemonth-{t}-255-free.npz")
            inc = d[fold].astype("float64") - d_m1[fold]
            v = 2 * r * inc - inc * inc
            v6 = 2 * r6 * inc - inc * inc
            G[arm], G6[arm] = float(v.mean()), float(v6.mean())
            want = json.loads((C.E46 / f"analysis/onemonth-{t}.json").read_text())[
                "prices"]["i_over_cached_member_1"]["by_fold"][fold]["mse_gain_s2"]
            if abs(G[arm] - want) > 1e-9 * max(1.0, abs(want)):
                raise SystemExit(f"{t}: G {G[arm]!r} is not onemonth's {want!r}")
            c = np.bincount(codes, minlength=k).astype(float)
            Gb[arm] = np.bincount(codes, weights=v, minlength=k)[draws].sum(1) / c[draws].sum(1)
            G6b[arm] = np.bincount(codes, weights=v6, minlength=k)[draws].sum(1) / c[draws].sum(1)
            moved = inc != 0
            ent[arm] = {"G_pbase_s2": G[arm], "G_base6_s2": G6[arm],
                        "G_base6_ci95": [float(np.percentile(G6b[arm], 2.5)),
                                         float(np.percentile(G6b[arm], 97.5))],
                        "change_s2": G6[arm] - G[arm],
                        "rows_moved": int(moved.sum()),
                        "corr_inc_with_pbase_minus_base6_on_moved": float(np.corrcoef(
                            inc[moved], (p.pbase.to_numpy("float64") - p6)[moved])[0, 1])}
        rb, r6b = Gb["F2"] / Gb["N2"], G6b["F2"] / G6b["N2"]
        ok, ok6 = rb > 0, r6b > 0
        logs[fold] = np.where(ok, np.log(np.where(ok, rb, 1.0)), np.nan)
        logs2[fold] = np.where(ok6, np.log(np.where(ok6, r6b, 1.0)), np.nan)
        pb = pan[fold].pbase.to_numpy("float64")
        res[fold] = {
            "arms": ent,
            "R_pbase_point": G["F2"] / G["N2"],
            "R_pbase_ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
            "R_base6_point": G6["F2"] / G6["N2"],
            "R_base6_ci95": [float(np.percentile(r6b, 2.5)), float(np.percentile(r6b, 97.5))],
            "R_base6_minus_R_pbase_point": G6["F2"] / G6["N2"] - G["F2"] / G["N2"],
            "R_base6_minus_R_pbase_ci95": [float(np.percentile(r6b - rb, 2.5)),
                                           float(np.percentile(r6b - rb, 97.5))],
            "increments_clear_zero_base6": all(ent[x]["G_base6_ci95"][0] > 0 for x in ent),
            "rms_pbase_minus_base6_s_all_rows": float(np.sqrt(np.mean((pb - p6) ** 2))),
            "rows": int(len(p)), "days": int(k)}
        print(f"{fold:9} R(pbase) {res[fold]['R_pbase_point']:.4f}  R(base6) "
              f"{res[fold]['R_base6_point']:.4f} {res[fold]['R_base6_ci95']}", flush=True)

    def pooled(lg):
        both = lg["december"] + lg["january"]
        okp = np.isfinite(both)
        return np.exp(0.5 * both[okp]), int(okp.sum())

    pd_, n_ = pooled(logs)
    pd6, n6 = pooled(logs2)
    both_ok = np.isfinite(logs["december"] + logs["january"]) & \
        np.isfinite(logs2["december"] + logs2["january"])
    diff = (np.exp(0.5 * (logs2["december"] + logs2["january"]))
            - np.exp(0.5 * (logs["december"] + logs["january"])))[both_ok]
    Rp = float(np.sqrt(res["december"]["R_pbase_point"] * res["january"]["R_pbase_point"]))
    Rp6 = float(np.sqrt(res["december"]["R_base6_point"] * res["january"]["R_base6_point"]))
    ci = [float(np.percentile(pd_, 2.5)), float(np.percentile(pd_, 97.5))]
    ci6 = [float(np.percentile(pd6, 2.5)), float(np.percentile(pd6, 97.5))]
    # against the full-precision artifact, not the printed digits
    rf = json.loads((C.E46 / "analysis/redundfit.json").read_text())
    repro = {"R_dec": res["december"]["R_pbase_point"], "R_jan": res["january"]["R_pbase_point"],
             "pooled": Rp, "pooled_ci95": ci,
             "R_dec_ci95": res["december"]["R_pbase_ci95"],
             "R_jan_ci95": res["january"]["R_pbase_ci95"]}
    want = {"R_dec": rf["by_fold"]["december"]["R_point"], "R_jan": rf["by_fold"]["january"]["R_point"],
            "pooled": rf["R_pooled"], "pooled_ci95": rf["R_pooled_ci95"],
            "R_dec_ci95": rf["by_fold"]["december"]["R_ci95"],
            "R_jan_ci95": rf["by_fold"]["january"]["R_ci95"]}
    reproduced = all(np.allclose(np.atleast_1d(repro[k]), np.atleast_1d(want[k]),
                                 rtol=0, atol=1e-12) for k in want)
    if not reproduced or abs(Rp - RECORDED["pooled"]) > 5e-5:
        raise SystemExit(f"redundfit does not reproduce: {repro} against {want}")
    kill = [f for f in RF.CODE if not res[f]["increments_clear_zero_base6"]]
    split = (res["december"]["R_base6_point"] < RF.CUT) != (res["january"]["R_base6_point"] < RF.CUT)
    band = ("MOVES TO STANDALONE" if ci6[0] >= RF.CUT else
            "UNMOVED" if ci6[1] < RF.CUT else "STRADDLES")
    verdict = ("NOT POOLED (kill line: " + ", ".join(kill) + ")" if kill else
               "SPLIT" if split else band)
    out = {"section": "exp-051 SPEC I1.1", "seed": RF.SEED, "n_boot": RF.N_BOOT, "cut_C": RF.CUT,
           "statistic": "R_f = G_f(F2|m1)/G_f(N2|m1); G = mean(2 r inc - inc^2) over the whole fold",
           "scoring_residual": {"D28": "r = y - pbase - d_m1", "rescored": "r' = y - base6 - d_m1"},
           "matched_draw_hashes": hashes, "vector_sha16": vec_sha,
           "reproduced_redundfit": repro, "reproduced": reproduced,
           "by_fold": res,
           "R_pbase_pooled": Rp, "R_pbase_pooled_ci95": ci, "pooled_draws_usable": n_,
           "R_base6_pooled": Rp6, "R_base6_pooled_ci95": ci6, "pooled_draws_usable_base6": n6,
           "R_base6_minus_R_pbase_pooled_point": Rp6 - Rp,
           "R_base6_minus_R_pbase_pooled_ci95": [float(np.percentile(diff, 2.5)),
                                                 float(np.percentile(diff, 97.5))],
           "verdict": verdict,
           "comparator_no_member1_4774": RF.NO_M1_4774,
           "seconds": round(time.time() - t0, 1)}
    C.write_json(OUT, out, overwrite=a.overwrite)
    print(f"pooled R(pbase) {Rp:.4f} {ci}   pooled R(base6) {Rp6:.4f} {ci6}   "
          f"change {Rp6 - Rp:+.4f} {out['R_base6_minus_R_pbase_pooled_ci95']}   verdict {verdict}")
    print(f"written to {OUT.relative_to(C.ROOT)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
