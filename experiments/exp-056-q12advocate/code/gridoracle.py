"""exp-056-q12advocate: an upper bound on V over the registered correction family (post-hoc).

Run after both iterations were priced (review round 1, finding 11). For one rotation and every arm
with a prior (true, the ten permuted shams, gmean, donor, shuffle), fit.py's main-design
correction fitted on S is, at each (lam, prior form), exactly `data(lam) + beta * slope(lam,
form)` (the same exact sparse solve). The forms are the iteration-1 prior and the iteration-2
prior weighted by c / (c + n_A) for every c in C_GRID. This job scores every lam in LAMBDAS, every
form, and beta on a grid from -1 to 3 in steps of 0.05 (it contains the registered betas) on V
itself, with the served clip max(60, p + h), and keeps the best point. No selection rule on the
inner months can pick a better point of this family, so the best point's gain over `zero`'s
registered pick bounds what any choice inside the family (beta extended) could have measured on
V. Also kept: the bound over the registered betas only (0 to 1), and `zero`'s lam-oracle (beta
0, best lam on V), which separates the prior from the lam choice. As a check, the registered
picks of `true` must reproduce the priced fold points. Not a test; nothing is chosen from it.

    .venv/bin/python experiments/exp-056-q12advocate/code/gridoracle.py jul     # reads S and V
    .venv/bin/python experiments/exp-056-q12advocate/code/gridoracle.py jul --dry planted
    .venv/bin/python experiments/exp-056-q12advocate/code/gridoracle.py --combine [--dry planted]
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

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

ARMS = ("true",) + tuple(f"sham{s:02d}" for s in E.SHAM_SEEDS) + ("gmean", "donor", "shuffle")
BGRID = np.round(np.arange(-1.0, 3.0 + 1e-9, 0.05), 2)
REG_B = np.isin(BGRID, np.round(np.array(E.BETAS), 2))


def out_path(r: str | None, dry: str | None) -> Path:
    tag = f"dry-{dry}-" if dry else ""
    return E.ANALYSIS / (f"{tag}gridoracle-{r}.json" if r else f"{tag}gridoracle.json")


def one(r: str, dry: str | None) -> None:
    out = out_path(r, dry)
    NO.refuse_existing([out], False)
    t0 = time.time()
    launch = E.hold_load(f"gridoracle {r}")
    inp = E.load_json(E.ANALYSIS / f"inputs-{r}.json")
    fits = {1: E.load_json(E.ANALYSIS / f"fit-{r}.json"), 2: E.load_json(E.ANALYSIS / f"it2-fit-{r}.json")}
    if E.sha_file(E.ROOT / inp["npz"]) != inp["npz_sha256"]:
        raise SystemExit("the inputs are not the frozen file")
    for it in ("iter1.json", "iter2.json"):
        if not (E.ANALYSIS / it).exists():
            raise SystemExit(f"{it} is missing: this bound runs only after both iterations")
    z = np.load(E.ROOT / inp["npz"])
    V, S, _, _ = E.ROT[r]
    nlev = {f: int(inp["vocab_sizes"][f]) for f in E.FAMS}
    if dry:
        yS = F.synthetic_labels(z, dry, "S", 101 + E.ORDER.index(r))
        yV = F.synthetic_labels(z, dry, "V", 301 + E.ORDER.index(r))
    else:
        yS = E.dep_labels(S, z["ids_S"], "post-hoc grid-oracle bound: the S fit")
        yV = E.dep_labels(V, z["ids_V"], "post-hoc grid-oracle bound: scoring on V itself")
    codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("S", "V")}
    rows = z["pop_S"] & np.isfinite(yS) & (yS > 0)
    X = E.design(codes["S"], nlev, rows)
    res = E.demean_by_airport(np.clip(yS[rows] - z["p_S"][rows], -E.CLIP, E.CLIP), codes["S"]["apt"][rows])
    XV = E.design(codes["V"], nlev)
    pV, popV = z["p_V"], z["pop_V"]
    P = popV
    sse_out = float(np.sum((yV[~P] - pV[~P]) ** 2))
    n_all = len(yV)
    lu = {lam: splu((X.T @ X + lam * sp.identity(X.shape[1], format="csr")).tocsc())
          for lam in E.LAMBDAS if np.isfinite(lam)}
    h0 = {lam: (XV @ lu[lam].solve(np.asarray(X.T @ res, dtype="float64"))) if np.isfinite(lam)
          else np.zeros(n_all) for lam in E.LAMBDAS}

    def sse_curve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """SSE over all V rows for every beta in BGRID; only P rows move."""
        pr = np.maximum(E.MIN_PRED, (pV[P] + a[P])[:, None] + b[P][:, None] * BGRID[None, :])
        return sse_out + ((yV[P][:, None] - pr) ** 2).sum(0)

    zero_curve = {lam: float(sse_curve(h0[lam], np.zeros(n_all))[0]) for lam in E.LAMBDAS}
    lz = fits[1]["arms"]["zero"]["lam"]
    lz = float("inf") if lz == "inf" else float(lz)
    sse_zero_pick = zero_curve[lz]
    lam_or = min(zero_curve, key=zero_curve.get)
    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r, "V": V, "dry": dry, "launch": launch,
           "code_sha256": E.code_shas(), "label_reads": E.LABEL_READS, "rows_all": n_all,
           "beta_grid": [float(b) for b in BGRID], "lambdas": [str(x) for x in E.LAMBDAS],
           "c_grid": list(E.C_GRID), "zero_pick": {"lam": str(lz), "sse": sse_zero_pick},
           "zero_lam_oracle": {"lam": str(lam_or), "sse": zero_curve[lam_or]},
           "zero_sse_by_lam": {str(k): v for k, v in zero_curve.items()}, "arms": {}}
    nA = {f: z[f"nA_{f}"] for f in E.PRIOR_FAMS}
    for arm in ARMS:
        pv0 = F.prior_of(z, arm, nlev)
        best = {"free": None, "registered": None}
        picks = {}
        for c in (None,) + tuple(E.C_GRID):
            pv = pv0 if c is None else pv0 * E.support_weights(nA, c)
            for lam in E.LAMBDAS:
                slope = XV @ (lu[lam].solve(lam * pv) if np.isfinite(lam) else pv)
                curve = sse_curve(h0[lam], slope)
                for kind, mask in (("free", np.ones(len(BGRID), bool)), ("registered", REG_B)):
                    i = int(np.flatnonzero(mask)[np.argmin(curve[mask])])
                    if best[kind] is None or curve[i] < best[kind]["sse"]:
                        best[kind] = {"lam": str(lam), "c": c, "beta": float(BGRID[i]), "sse": float(curve[i])}
                # the registered picks of this arm, to check against the priced fold points
                for it in (1, 2):
                    a = fits[it]["arms"][arm]
                    al = float("inf") if a["lam"] == "inf" else float(a["lam"])
                    if al == lam and a.get("c") == c and (it == 2 or c is None):
                        j = int(np.flatnonzero(np.isclose(BGRID, a["beta"]))[0])
                        picks[it] = float(curve[j])
        rec["arms"][arm] = {"best_free": best["free"], "best_registered_beta": best["registered"],
                            "sse_at_registered_pick": {str(k): v for k, v in picks.items()}}
        print(f"  {r} {arm:8s} free {(sse_zero_pick - best['free']['sse']) / n_all:+8.2f} "
              f"(lam {best['free']['lam']}, c {best['free']['c']}, beta {best['free']['beta']:+.2f})  "
              f"registered-beta {(sse_zero_pick - best['registered']['sse']) / n_all:+8.2f} s2/row "
              f"over zero's pick", flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([out], False)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"written {out.name} ({rec['seconds']:.0f}s)")


def combine(dry: str | None) -> None:
    out = out_path(None, dry)
    NO.refuse_existing([out], False)
    d = {r: E.load_json(out_path(r, dry)) for r in E.ORDER}
    mse = lambda r, sse: sse / d[r]["rows_all"]                                  # noqa: E731

    def board(a_sse: dict, b_sse: dict) -> float:
        return float(np.sqrt(sum(s * mse(r, a_sse[r]) for r, s in E.BOARD.items()))
                     - np.sqrt(sum(s * mse(r, b_sse[r]) for r, s in E.BOARD.items())))
    zp = {r: d[r]["zero_pick"]["sse"] for r in E.ORDER}
    zo = {r: d[r]["zero_lam_oracle"]["sse"] for r in E.ORDER}
    res = {"what": "grid-oracle bound, board recombination (post-hoc; V in-sample; not a test)",
           "dry": dry, "zero_lam_oracle_over_zero_pick": {
               "board_s": board(zp, zo), "fold_s2_per_row": {r: mse(r, zp[r] - zo[r]) for r in E.ORDER}},
           "arms": {}}
    for arm in ARMS:
        x = {}
        for kind in ("best_free", "best_registered_beta"):
            b = {r: d[r]["arms"][arm][kind]["sse"] for r in E.ORDER}
            x[kind] = {"over_zero_pick_board_s": board(zp, b), "over_zero_lam_oracle_board_s": board(zo, b),
                       "over_zero_pick_fold_s2_per_row": {r: mse(r, zp[r] - b[r]) for r in E.ORDER},
                       "points": {r: {k: d[r]["arms"][arm][kind][k] for k in ("lam", "c", "beta")} for r in E.ORDER}}
        x["registered_picks_over_zero_pick_fold_s2_per_row"] = {
            it: {r: mse(r, zp[r] - d[r]["arms"][arm]["sse_at_registered_pick"][it]) for r in E.ORDER}
            for it in ("1", "2") if all(it in d[r]["arms"][arm]["sse_at_registered_pick"] for r in E.ORDER)}
        res["arms"][arm] = x
    sh = [f"sham{s:02d}" for s in E.SHAM_SEEDS]
    for kind in ("best_free", "best_registered_beta"):
        t = res["arms"]["true"][kind]["over_zero_lam_oracle_board_s"]
        v = [res["arms"][s][kind]["over_zero_lam_oracle_board_s"] for s in sh]
        res[f"true_rank_of_11_{kind}"] = int(1 + sum(x > t for x in v))
        res[f"sham_max_{kind}_over_zero_lam_oracle_board_s"] = float(max(v))
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"zero lam-oracle over zero's pick: board {res['zero_lam_oracle_over_zero_pick']['board_s']:+.4f} s")
    for arm, x in res["arms"].items():
        print(f"{arm:8s} free: over zero pick {x['best_free']['over_zero_pick_board_s']:+.4f} s, over zero "
              f"lam-oracle {x['best_free']['over_zero_lam_oracle_board_s']:+.4f}; registered beta: "
              f"{x['best_registered_beta']['over_zero_pick_board_s']:+.4f} / "
              f"{x['best_registered_beta']['over_zero_lam_oracle_board_s']:+.4f}")
    print("true rank of 11 (free, registered):", res["true_rank_of_11_best_free"],
          res["true_rank_of_11_best_registered_beta"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rotation", nargs="?", choices=E.ORDER)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--dry", choices=F.SCENARIOS, default=None, help="synthetic labels; no label read")
    a = ap.parse_args()
    if a.combine == (a.rotation is not None):
        raise SystemExit("give one rotation, or --combine")
    combine(a.dry) if a.combine else one(a.rotation, a.dry)
