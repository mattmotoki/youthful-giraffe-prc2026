"""Sol round 1 repairs that need new measurement (blocking 1, 5 and 6).

Nothing here overwrites the round-1 record: `analysis/learner.json` and
`analysis/controls.json` stay as measured, and every arm below is an addition.

  b1  the exact diagonal Hessian.  The loss's Hessian is the within-cell
      centring projector `P`, not the identity; LightGBM takes only a diagonal,
      and the exact diagonal is `1 - 1/n_c`.  Every tree arm is refitted with it.
  b5  the shipped block's 13 `c_*` columns are codes from one persistent,
      append-only vocabulary shared across cached months, so a cross-month arm
      that uses them is not certified fold-isolated.  Every shipped arm is
      refitted on the 76 columns that remain.
  b6  two controls that test the parts of the `xmonth` path the round-1 controls
      did not:
        `pathcheck`  the EXACT xmonth code path -- frozen vocabulary, two parts
                     vstacked, cell codes offset per part -- with the two parts
                     drawn from the SCORED month's own fitting day blocks.  If
                     the path itself cost the gain, this collapses too.
        `xmonth_one` the same path fitted on ONE other month, which separates
                     "two months pooled" from "a month is crossed".

    OMP_NUM_THREADS=3 python repair.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zs                                   # noqa: E402
import learner as L                         # noqa: E402

EXP = zs.EXP
FOLDS = zs.FOLDS
TREE_ARMS = ["lgb_small", "lgb_wide", "lgb_shipped", "lgb_shipped_wide"]


def matrix(month: str, levels, shipped: bool, ids, nocv: bool):
    d = zs.dep(month)
    X, cols = zs.design(d, levels)
    if shipped:
        W = zs.shipped(month, drop_cache_vocab=nocv).reindex(
            d["MVT_ID_mvt"].astype("float64").to_numpy())
        X = np.hstack([X, W.to_numpy("float32")])
        cols = cols + list(W.columns)
    return X[L.order_index(d, ids)], cols


def fit(kind: str, Xfit, r_fit, code_fit, Xscore, hess: str) -> np.ndarray:
    fc, ufc = pd.factorize(code_fit)
    nfc = len(ufc)
    rc = zs.project(np.asarray(r_fit, "float64"), fc, nfc)
    if kind == "ridge":
        return zs.ridge_h(np.asarray(Xfit, "float64"), rc, fc, nfc,
                          np.asarray(Xscore, "float64"))
    return zs.lgb_h(Xfit, rc, fc, nfc, Xscore, kind, hess_mode=hess)


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(zs.SEED)
    pan = zs.V.load()
    S = {f: L.fold_state(f, pan[f]) for f in FOLDS}
    amps = json.load(open(EXP / "analysis" / "learner.json"))["amplitudes"]
    H: dict[str, dict[str, np.ndarray]] = {f: {} for f in FOLDS}
    fit_months: dict[str, dict[str, str]] = {}

    for f in FOLDS:
        Sf = S[f]
        ids, ei, blk = Sf["ids"], np.flatnonzero(Sf["elig"]), Sf["blk"]
        others = [g for g in FOLDS if g != f]
        frozen = zs.levels_of([zs.dep(S[g]["month"]) for g in others])
        local = zs.levels_of([zs.dep(Sf["month"])])

        # ---- b1 and b5: every tree arm again, exact-diagonal Hessian and,
        # ---- for the shipped arms, without the shared cache vocabulary.
        for shipped in (False, True):
            arms = [a for a in TREE_ARMS if (a in L.SHIPPED) == shipped]
            nocv = shipped
            Xlo, _ = matrix(Sf["month"], local, shipped, ids, nocv)
            Xfr, colf = matrix(Sf["month"], frozen, shipped, ids, nocv)
            parts, rs, cs = [], [], []
            for g in others:
                G = S[g]
                Xg, cg = matrix(G["month"], frozen, shipped, G["ids"], nocv)
                if cg != colf:
                    raise SystemExit("column mismatch across months")
                e = np.flatnonzero(G["elig"])
                parts.append(Xg[e]); rs.append(G["r"][e])
                cs.append(G["code"][e].astype("int64") + 1000000 * FOLDS.index(g))
                del Xg
            Xtr = np.vstack(parts); del parts
            rtr = np.concatenate(rs); ctr = np.concatenate(cs)
            for arm in arms:
                base = L.BASE[arm]
                tag = "hess_diag" if not shipped else "hess_diag_nocachevocab"
                h = np.zeros(len(ids))
                for b in range(zs.BLOCKS):
                    fi, sc = ei[blk[ei] != b], ei[blk[ei] == b]
                    h[sc] = fit(base, Xlo[fi], Sf["r"][fi], Sf["code"][fi], Xlo[sc], "diag")
                H[f][f"{arm}|dayblock|{tag}"] = h
                H[f][f"{arm}|xmonth|{tag}"] = fit(base, Xtr, rtr, ctr, Xfr, "diag")
                print(f"    [{f}] {arm:18s} {tag:22s} {time.time() - t0:6.1f}s", flush=True)
            del Xtr, Xlo, Xfr

        # ---- b6: the path check and the single-month fit, ridge only.
        Xfr, _ = matrix(Sf["month"], frozen, False, ids, False)
        h = np.zeros(len(ids))
        for b in range(zs.BLOCKS):
            fi, sc = ei[blk[ei] != b], ei[blk[ei] == b]
            # two parts of the SAME month, vstacked, codes offset per part --
            # the xmonth machinery, exercised without crossing a month.
            half = blk[fi] % 2
            Xp = np.vstack([Xfr[fi[half == 0]], Xfr[fi[half == 1]]])
            rp = np.concatenate([Sf["r"][fi[half == 0]], Sf["r"][fi[half == 1]]])
            cp = np.concatenate([Sf["code"][fi[half == 0]].astype("int64"),
                                 Sf["code"][fi[half == 1]].astype("int64") + 1000000])
            h[sc] = fit("ridge", Xp, rp, cp, Xfr[sc], "unit")
        H[f]["ridge|pathcheck|-"] = h
        for j, g in enumerate(others, start=1):
            G = S[g]
            lv1 = zs.levels_of([zs.dep(G["month"])])
            Xg, c1 = matrix(G["month"], lv1, False, G["ids"], False)
            Xf1, c2 = matrix(Sf["month"], lv1, False, ids, False)
            e = np.flatnonzero(G["elig"])
            H[f][f"ridge|xmonth_one_{j}|-"] = fit(
                "ridge", Xg[e], G["r"][e], G["code"][e].astype("int64"), Xf1, "unit")
            fit_months.setdefault(f"xmonth_one_{j}", {})[f] = G["month"]
            del Xg, Xf1
        print(f"    [{f}] ridge controls {time.time() - t0:6.1f}s", flush=True)
        del Xfr

    keys = sorted({k for f in FOLDS for k in H[f]})

    # The projected update per key and fold, once.
    D0: dict[str, dict[str, np.ndarray]] = {}
    for key in keys:
        D0[key] = {}
        for f in FOLDS:
            m0 = S[f]["m0"]
            cm, ucm = pd.factorize(S[f]["code"][m0])
            v = np.zeros(len(m0))
            v[m0] = zs.project(H[f][key][m0], cm, len(ucm))
            D0[key][f] = v

    # Sol round 2, blocking 1: the first version of this script scored every
    # repaired arm at the amplitude fitted on the OLD, vocabulary-bearing run,
    # so the no-vocabulary and exact-Hessian arms were fixed-amplitude
    # perturbations rather than refits of the declared estimator.  Each repaired
    # arm now gets its OWN cross-month amplitude, fitted on the other two folds'
    # day-block updates of THAT arm, exactly as code/learner.py does.
    own_amp: dict[str, dict[str, float]] = {}
    for key in keys:
        arm, cf, tag = key.split("|")
        src = f"{arm}|dayblock|{tag}"
        if src not in D0:                       # the ridge controls have no
            src = None                          # day-block sibling of their own
        own_amp[key] = {}
        for f in FOLDS:
            num = den = 0.0
            for g in FOLDS:
                if g == f:
                    continue
                m = S[g]["m0"]
                u = D0[src][g] if src else None
                if u is None:
                    continue
                num += float((S[g]["r"][m] * u[m]).sum())
                den += float((u[m] ** 2).sum())
            own_amp[key][f] = (num / den if den > 0 else 0.0) if src else amps[arm][f]

    recs, boards = {}, {}
    for key in keys:
        arm = key.split("|")[0]
        recs[key] = {}
        for f in FOLDS:
            d0 = D0[key][f]
            a = own_amp[key][f]
            recs[key][f] = L.score(S[f], a * d0, H[f][key], a, rng, True)
        ex = lambda k: {f: recs[key][f]["rmse_exact"][k] for f in FOLDS}
        av, an = zs.V.board(ex("all_v10")), zs.V.board(ex("all_new"))
        cv, cn = zs.V.board(ex("clean_v10")), zs.V.board(ex("clean_new"))
        boards[key] = {"gain_all_row": round(av - an, 4), "gain_clean": round(cv - cn, 4)}
        print(f"  {key:44s} all-row {av - an:+.4f}  clean {cv - cn:+.4f}  folds "
              + " ".join(f"{recs[key][f]['gain_all']:+.4f}" for f in FOLDS)
              + "  2A-B " + " ".join(
                  f"{recs[key][f]['mse_gain_s2']:+.0f}[{recs[key][f]['ci'][0]:+.0f},"
                  f"{recs[key][f]['ci'][1]:+.0f}]" for f in FOLDS), flush=True)

    out = {"step": "Sol round 1 repairs (blocking 1, 5, 6); "
                   "Sol round 2 blocking 1, the amplitude repair",
           "amplitude_note": "each repaired arm carries its OWN cross-month amplitude, "
                             "fitted on the other two folds' day-block updates of that arm. "
                             "The ridge path controls have no day-block sibling of their own "
                             "and keep the reference ridge amplitude, which is stated where "
                             "they are read.",
           "amplitudes": own_amp, "folds": recs,
           "board": boards, "single_month_fit_source": fit_months,
           "wall_s": round(time.time() - t0, 1)}
    (EXP / "analysis" / "repair.json").write_text(json.dumps(out, indent=2))
    # Sol round 2, blocking 4: COMMON's slice decomposition is owed for every arm,
    # so every arm's held-out h is kept for code/records.py.
    np.savez_compressed(EXP / "analysis" / "h_repair.npz",
                        **{f"{f}|{k}": H[f][k].astype("float32")
                           for f in FOLDS for k in H[f]})
    print(f"  written {EXP / 'analysis' / 'repair.json'} ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
