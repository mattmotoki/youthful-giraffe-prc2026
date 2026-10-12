"""exp-056-q12advocate: an upper bound on V over the per-airport refinement (post-hoc).

Review round 1, finding 13. Every key family is joined with its airport (e56.dep_frame), so a
row's correction uses only its own airport's columns: X'X is block diagonal by airport and the
exact sparse solve separates by airport. The fitted correction is linear in the prior, h =
data(lam) + sum_f beta_f * slope_f(lam, form), slope_f from the prior restricted to family f
(the same exact solve as fit.py). So a beta per airport, or per airport and family, is exact on
the same solves. For each arm with a prior, rotation and airport, this job scores on V itself,
with the served clip max(60, p + h):
  A  one beta per airport; lam and form common to all airports;
  B  one beta per airport and family (stand, rwy, op); lam and form common;
  C  B with lam and form chosen per airport as well;
each on the registered betas (7 per family) and on a wide grid (A: -1 to 3 by 0.05; B, C: 13
values from -1 to 3 per family), and keeps the best point. References: zero's registered pick,
zero's lam-oracle (one lam) for A and B, and zero's per-airport lam-oracle for C. Check: A with
one beta for all airports is gridoracle.py's family, so its best points must equal gridoracle's.
In-sample fits gain on noise too; the ten permuted shams calibrate that. Not a test; nothing is
chosen from it.

    .venv/bin/python experiments/exp-056-q12advocate/code/refineoracle.py jul     # reads S and V
    .venv/bin/python experiments/exp-056-q12advocate/code/refineoracle.py jul --dry planted
    .venv/bin/python experiments/exp-056-q12advocate/code/refineoracle.py --combine [--dry planted]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402
import fit as F                                                       # noqa: E402
import gridoracle as G                                                # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

REG1 = np.array(E.BETAS)
WIDE1 = G.BGRID
WIDE3 = np.array((-1.0, -0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0))
LEVELS = ("A_registered", "A_wide", "B_registered", "B_wide", "C_registered", "C_wide")


def grid3(b: np.ndarray) -> np.ndarray:
    return np.array(np.meshgrid(b, b, b, indexing="ij")).reshape(3, -1).T


GRIDS = {"A_registered": np.repeat(REG1[:, None], 3, 1), "A_wide": np.repeat(WIDE1[:, None], 3, 1),
         "B_registered": grid3(REG1), "B_wide": grid3(WIDE3)}


def out_path(r: str | None, dry: str | None) -> Path:
    tag = f"dry-{dry}-" if dry else ""
    return E.ANALYSIS / (f"{tag}refineoracle-{r}.json" if r else f"{tag}refineoracle.json")


def airport_sse(y, u, S, g, ng, B) -> np.ndarray:
    """Exact SSE of max(60, u + S b) against y, per airport group g (0..ng-1) and grid row b of
    B: the quadratic form on all rows, then an exact recomputation on the rows where the clip
    can bind anywhere in the grid's box."""
    e = y - u
    ee = np.bincount(g, weights=e * e, minlength=ng)
    Se = np.stack([np.bincount(g, weights=S[:, i] * e, minlength=ng) for i in range(3)], 1)
    SS = np.stack([np.stack([np.bincount(g, weights=S[:, i] * S[:, j], minlength=ng)
                             for j in range(3)], 1) for i in range(3)], 1)
    out = ee[:, None] - 2 * Se @ B.T + np.einsum("kf,afh,kh->ak", B, SS, B)
    lo = u + np.minimum(S * B.min(0), S * B.max(0)).sum(1)
    risk = np.flatnonzero(lo < E.MIN_PRED)
    for i0 in range(0, len(risk), 4000):
        k = risk[i0:i0 + 4000]
        lin = u[k, None] + S[k] @ B.T
        d = (y[k, None] - np.maximum(E.MIN_PRED, lin)) ** 2 - (y[k, None] - lin) ** 2
        np.add.at(out, g[k], d)
    return out


def one(r: str, dry: str | None) -> None:
    out = out_path(r, dry)
    NO.refuse_existing([out], False)
    t0 = time.time()
    launch = E.hold_load(f"refineoracle {r}")
    inp = E.load_json(E.ANALYSIS / f"inputs-{r}.json")
    fits = {1: E.load_json(E.ANALYSIS / f"fit-{r}.json")}
    grid_rec = E.load_json(G.out_path(r, dry))
    if E.sha_file(E.ROOT / inp["npz"]) != inp["npz_sha256"]:
        raise SystemExit("the inputs are not the frozen file")
    z = np.load(E.ROOT / inp["npz"])
    V, S_, _, _ = E.ROT[r]
    nlev = {f: int(inp["vocab_sizes"][f]) for f in E.FAMS}
    if dry:
        yS = F.synthetic_labels(z, dry, "S", 101 + E.ORDER.index(r))
        yV = F.synthetic_labels(z, dry, "V", 301 + E.ORDER.index(r))
    else:
        yS = E.dep_labels(S_, z["ids_S"], "post-hoc per-airport refinement bound: the S fit")
        yV = E.dep_labels(V, z["ids_V"], "post-hoc per-airport refinement bound: scoring on V itself")
    codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("S", "V")}
    rows = z["pop_S"] & np.isfinite(yS) & (yS > 0)
    X = E.design(codes["S"], nlev, rows)
    res = E.demean_by_airport(np.clip(yS[rows] - z["p_S"][rows], -E.CLIP, E.CLIP), codes["S"]["apt"][rows])
    XV = E.design(codes["V"], nlev)
    pV, P = z["p_V"], z["pop_V"]
    n_all = len(yV)
    sse_out = float(np.sum((yV[~P] - pV[~P]) ** 2))
    apt_raw = codes["V"]["apt"][P]
    groups, g = np.unique(apt_raw, return_inverse=True)
    ng = len(groups)
    yP = yV[P]
    lu = {lam: splu((X.T @ X + lam * sp.identity(X.shape[1], format="csr")).tocsc())
          for lam in E.LAMBDAS if np.isfinite(lam)}
    h0 = {lam: (XV @ lu[lam].solve(np.asarray(X.T @ res, dtype="float64")))[P] if np.isfinite(lam)
          else np.zeros(int(P.sum())) for lam in E.LAMBDAS}
    zsse = {lam: np.bincount(g, weights=(yP - np.maximum(E.MIN_PRED, pV[P] + h0[lam])) ** 2, minlength=ng)
            for lam in E.LAMBDAS}
    lz = fits[1]["arms"]["zero"]["lam"]
    lz = float("inf") if lz == "inf" else float(lz)
    zero = {"pick": {"lam": str(lz), "sse": sse_out + float(zsse[lz].sum())}}
    lo = min(E.LAMBDAS, key=lambda lam: zsse[lam].sum())
    zero["lam_oracle"] = {"lam": str(lo), "sse": sse_out + float(zsse[lo].sum())}
    per = np.array([zsse[lam] for lam in E.LAMBDAS])
    zero["airport_lam_oracle"] = {"lam": {str(groups[a]): str(E.LAMBDAS[i]) for a, i in enumerate(per.argmin(0))},
                                  "sse": sse_out + float(per.min(0).sum())}
    off = np.cumsum([0] + [nlev[f] for f in E.PRIOR_FAMS])
    nA = {f: z[f"nA_{f}"] for f in E.PRIOR_FAMS}
    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r, "V": V, "dry": dry, "launch": launch,
           "code_sha256": E.code_shas(), "label_reads": E.LABEL_READS, "rows_all": n_all, "rows_P": int(P.sum()),
           "airports": [int(x) for x in groups], "grids": {k: [float(x) for x in v] for k, v in
                                                          (("registered", REG1), ("A_wide", WIDE1), ("B_wide", WIDE3))},
           "zero": zero, "arms": {}, "check_vs_gridoracle": {}}
    risk_max = 0
    for arm in G.ARMS:
        pv0 = F.prior_of(z, arm, nlev)
        cells = {}                                   # (c, lam) -> level -> (per-airport min sse, argmin index)
        for c in (None,) + tuple(E.C_GRID):
            pv = pv0 if c is None else pv0 * E.support_weights(nA, c)
            for lam in E.LAMBDAS:
                Sm = np.zeros((int(P.sum()), 3))
                for i, f in enumerate(E.PRIOR_FAMS):
                    pf = np.zeros_like(pv)
                    pf[off[i]:off[i + 1]] = pv[off[i]:off[i + 1]]
                    Sm[:, i] = (XV @ (lu[lam].solve(lam * pf) if np.isfinite(lam) else pf))[P]
                u = pV[P] + h0[lam]
                cell = {}
                for lev, B in GRIDS.items():
                    s = airport_sse(yP, u, Sm, g, ng, B)
                    cell[lev] = (s.min(1), s.argmin(1), s.sum(0).min(), int(s.sum(0).argmin()))
                    risk_max = max(risk_max, int((u + np.minimum(Sm * B.min(0), Sm * B.max(0)).sum(1) < E.MIN_PRED).sum()))
                cells[(c, lam)] = cell
        a_rec = {}
        for lev in GRIDS:
            # per-airport beta, common (c, lam)
            key = min(cells, key=lambda k: cells[k][lev][0].sum())
            m, idx = cells[key][lev][0], cells[key][lev][1]
            a_rec[lev] = {"sse": sse_out + float(m.sum()), "c": key[0], "lam": str(key[1]),
                          "betas": {str(groups[a]): [float(x) for x in GRIDS[lev][idx[a]]] for a in range(ng)}}
            # one beta for all airports (the grid-oracle family), common (c, lam)
            k1 = min(cells, key=lambda k: cells[k][lev][2])
            a_rec[lev + "_one_beta"] = {"sse": sse_out + float(cells[k1][lev][2]), "c": k1[0], "lam": str(k1[1]),
                                        "beta": [float(x) for x in GRIDS[lev][cells[k1][lev][3]]]}
        for lev in ("B_registered", "B_wide"):
            best = np.array([[cells[k][lev][0][a] for k in cells] for a in range(ng)])
            ks = list(cells)
            a_rec["C" + lev[1:]] = {"sse": sse_out + float(best.min(1).sum()),
                                    "points": {str(groups[a]): {"c": ks[j][0], "lam": str(ks[j][1]),
                                                                "betas": [float(x) for x in GRIDS[lev][cells[ks[j]][lev][1][a]]]}
                                               for a, j in enumerate(best.argmin(1))}}
        rec["arms"][arm] = a_rec
        gr = grid_rec["arms"][arm]
        rec["check_vs_gridoracle"][arm] = {
            "registered": a_rec["A_registered_one_beta"]["sse"] - gr["best_registered_beta"]["sse"],
            "wide": a_rec["A_wide_one_beta"]["sse"] - gr["best_free"]["sse"]}
        zp, zl, za = zero["pick"]["sse"], zero["lam_oracle"]["sse"], zero["airport_lam_oracle"]["sse"]
        print(f"  {r} {arm:8s} over zero lam-oracle, s2/row: A reg {(zl - a_rec['A_registered']['sse']) / n_all:+7.2f}"
              f" wide {(zl - a_rec['A_wide']['sse']) / n_all:+7.2f}; B reg {(zl - a_rec['B_registered']['sse']) / n_all:+7.2f}"
              f" wide {(zl - a_rec['B_wide']['sse']) / n_all:+7.2f}; C (over zero airport lam-oracle) reg "
              f"{(za - a_rec['C_registered']['sse']) / n_all:+7.2f} wide {(za - a_rec['C_wide']['sse']) / n_all:+7.2f}; "
              f"check {max(abs(v) for v in rec['check_vs_gridoracle'][arm].values()):.2e}", flush=True)
    worst = max(abs(v) for x in rec["check_vs_gridoracle"].values() for v in x.values())
    rec["check_max_abs_sse_diff"] = worst
    rec["risk_rows_max"] = risk_max
    rec["seconds"] = round(time.time() - t0, 1)
    if worst > 1e-6 * zero["pick"]["sse"]:
        raise SystemExit(f"one-beta level does not reproduce the grid oracle: {worst}")
    NO.refuse_existing([out], False)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"written {out.name} ({rec['seconds']:.0f}s; clip-risk rows at most {risk_max}; check {worst:.2e})")


def combine(dry: str | None) -> None:
    out = out_path(None, dry)
    NO.refuse_existing([out], False)
    d = {r: E.load_json(out_path(r, dry)) for r in E.ORDER}

    def board(a_sse: dict, b_sse: dict) -> float:
        m = lambda x, r: x[r] / d[r]["rows_all"]                                  # noqa: E731
        return float(np.sqrt(sum(s * m(a_sse, r) for r, s in E.BOARD.items()))
                     - np.sqrt(sum(s * m(b_sse, r) for r, s in E.BOARD.items())))
    z = {k: {r: d[r]["zero"][k]["sse"] for r in E.ORDER} for k in ("pick", "lam_oracle", "airport_lam_oracle")}
    res = {"what": "per-airport refinement bound, board recombination (post-hoc; V in-sample; not a test)",
           "dry": dry, "zero": {"lam_oracle_over_pick_board_s": board(z["pick"], z["lam_oracle"]),
                                "airport_lam_oracle_over_pick_board_s": board(z["pick"], z["airport_lam_oracle"])},
           "levels": {}}
    sh = [f"sham{s:02d}" for s in E.SHAM_SEEDS]
    for lev in LEVELS:
        ref = z["airport_lam_oracle"] if lev.startswith("C") else z["lam_oracle"]
        gain = {arm: board(ref, {r: d[r]["arms"][arm][lev]["sse"] for r in E.ORDER}) for arm in G.ARMS}
        folds = {arm: {r: (ref[r] - d[r]["arms"][arm][lev]["sse"]) / d[r]["rows_all"] for r in E.ORDER}
                 for arm in G.ARMS}
        v = np.array([gain[s] for s in sh])
        res["levels"][lev] = {"reference": "zero airport lam-oracle" if lev.startswith("C") else "zero lam-oracle",
                              "board_gain_s": gain, "fold_gain_s2_per_row": folds,
                              "true_rank_of_11": int(1 + (v > gain["true"]).sum()),
                              "sham_mean_s": float(v.mean()), "sham_max_s": float(v.max()),
                              "true_minus_sham_mean_s": float(gain["true"] - v.mean())}
    res["check_max_abs_sse_diff"] = max(d[r]["check_max_abs_sse_diff"] for r in E.ORDER)
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"zero lam-oracle over zero's pick {res['zero']['lam_oracle_over_pick_board_s']:+.4f} s; per-airport lam-oracle "
          f"{res['zero']['airport_lam_oracle_over_pick_board_s']:+.4f} s")
    for lev, x in res["levels"].items():
        g = x["board_gain_s"]
        print(f"{lev:13s} over {x['reference']}: true {g['true']:+.4f} s, rank {x['true_rank_of_11']} of 11, sham mean "
              f"{x['sham_mean_s']:+.4f}, max {x['sham_max_s']:+.4f}; gmean {g['gmean']:+.4f}, donor {g['donor']:+.4f}, "
              f"shuffle {g['shuffle']:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rotation", nargs="?", choices=E.ORDER)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--dry", choices=F.SCENARIOS, default=None, help="synthetic labels; no label read")
    a = ap.parse_args()
    if a.combine == (a.rotation is not None):
        raise SystemExit("give one rotation, or --combine")
    combine(a.dry) if a.combine else one(a.rotation, a.dry)
