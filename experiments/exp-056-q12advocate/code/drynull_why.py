"""exp-056-q12advocate: why the null dry run's free-beta bound ranks `true` first (label-free).

Review round 2, finding 5. RESULTS 5.1 said that the null dry run's free-beta value (+0.2642 s, `true`
first of 11) came from the synthetic floor max(1, .). Sol's rerun without the floor, the label > 0 row
filter and the served clip max(60, .) still gave `true` +0.15854 s, first, against a sham maximum of
+0.05287 s, so the floor is not the cause. This job tests one other cause, stated before it runs: under
the null the only structure in V's synthetic residual is its airport-month offset (normal, sd 30 s, per
airport). The family's correction on V is data(lam) + beta * slope(lam, form); fit.py demeans the S
residual by airport, but neither term is demeaned on V's rows (at lam = inf the slope is the prior
itself), so beta moves each airport's mean on V and a free beta can fit part of V's offset.
Prediction: with V's offset set to zero, or with data and slope demeaned by airport over V's population
rows, `true`'s free-beta advantage over the ten permuted shams goes away.

Variants, all on the draws of `gridoracle.py --dry null` (seed 101 + i for S, 301 + i for V, i the
rotation's index), the same family (every lam, every prior form, beta -1 to 3 by 0.05) and the same
board recombination over `zero`'s lam-oracle:
  registered      as gridoracle's dry null (floor, row filter, served clip); must reproduce its SSEs
  plain           no floor, no row filter, no served clip (Sol's rerun)
  plain_nooffset  plain, with V's airport offset set to zero (S keeps its offset; fit.py demeans it)
  plain_vdemean   plain, with data(lam) and slope demeaned by airport over V's population rows
Also, per prior, the share of its sum of squares on V's population rows (lam = inf, no support
weight) that lies in its airport means. No label is read.

    .venv/bin/python experiments/exp-056-q12advocate/code/drynull_why.py
"""
from __future__ import annotations

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

OUT = E.ANALYSIS / "drynull-why.json"
VARIANTS = ("registered", "plain", "plain_nooffset", "plain_vdemean")
SHAMS = tuple(f"sham{s:02d}" for s in E.SHAM_SEEDS)


def null_parts(z, k: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The airport offset and the t3 noise of fit.synthetic_labels(z, 'null', k, seed), same draws."""
    rng = np.random.default_rng(seed)
    apt = z[f"code_{k}_apt"]
    off = rng.normal(0, 30, int(apt.max()) + 1)[np.maximum(apt, 0)]
    eps = 250.0 * rng.standard_t(3, len(apt))
    if not np.array_equal(np.maximum(1.0, z[f"p_{k}"] + off + eps), F.synthetic_labels(z, "null", k, seed)):
        raise SystemExit(f"the null draws of {k} are not fit.synthetic_labels'")
    return off, eps


def airport_share(v: np.ndarray, apt: np.ndarray) -> float:
    a = np.maximum(apt, 0)
    n = np.bincount(a).astype("float64")
    m = np.bincount(a, weights=v) / np.maximum(n, 1.0)
    return float((n * m ** 2).sum() / max(float((v ** 2).sum()), 1e-300))


def one(r: str) -> dict:
    t0 = time.time()
    i = E.ORDER.index(r)
    inp = E.load_json(E.ANALYSIS / f"inputs-{r}.json")
    if E.sha_file(E.ROOT / inp["npz"]) != inp["npz_sha256"]:
        raise SystemExit("the inputs are not the frozen file")
    z = np.load(E.ROOT / inp["npz"])
    nlev = {f: int(inp["vocab_sizes"][f]) for f in E.FAMS}
    offS, epsS = null_parts(z, "S", 101 + i)
    offV, epsV = null_parts(z, "V", 301 + i)
    pS, pV, P = z["p_S"], z["p_V"], z["pop_V"]
    aptS, aptV = z["code_S_apt"], z["code_V_apt"]
    yS = {"floor": np.maximum(1.0, pS + offS + epsS), "plain": pS + offS + epsS}
    yV = {"registered": np.maximum(1.0, pV + offV + epsV), "plain": pV + offV + epsV,
          "plain_nooffset": pV + epsV, "plain_vdemean": pV + offV + epsV}
    rows = z["pop_S"] & np.isfinite(yS["floor"]) & (yS["floor"] > 0)
    if not np.array_equal(rows, z["pop_S"] & np.isfinite(yS["plain"])):
        raise SystemExit("the row filter removes rows under the floor")
    X = E.design({f: z[f"code_S_{f}"] for f in E.FAMS}, nlev, rows)
    XV = E.design({f: z[f"code_V_{f}"] for f in E.FAMS}, nlev)
    lu = {lam: splu((X.T @ X + lam * sp.identity(X.shape[1], format="csr")).tocsc())
          for lam in E.LAMBDAS if np.isfinite(lam)}
    n_all = len(pV)

    def h_data(y: np.ndarray) -> dict:
        res = E.demean_by_airport(np.clip(y[rows] - pS[rows], -E.CLIP, E.CLIP), aptS[rows])
        rhs = np.asarray(X.T @ res, dtype="float64")
        return {lam: (XV @ lu[lam].solve(rhs)) if np.isfinite(lam) else np.zeros(n_all) for lam in E.LAMBDAS}
    h0 = {"registered": h_data(yS["floor"])}
    h0["plain"] = h0["plain_nooffset"] = h0["plain_vdemean"] = h_data(yS["plain"])
    sse_out = {v: float(np.sum((yV[v][~P] - pV[~P]) ** 2)) for v in VARIANTS}

    def vdm(x: np.ndarray) -> np.ndarray:
        return E.demean_by_airport(x[P], aptV[P])

    def curve(v: str, a: np.ndarray, s: np.ndarray) -> np.ndarray:
        """SSE over all V rows for every beta in gridoracle's BGRID; only P rows move."""
        if v == "registered":
            pr = np.maximum(E.MIN_PRED, (pV[P] + a[P])[:, None] + s[P][:, None] * G.BGRID[None, :])
            return sse_out[v] + ((yV[v][P][:, None] - pr) ** 2).sum(0)
        aa, ss = (vdm(a), vdm(s)) if v == "plain_vdemean" else (a[P], s[P])
        e = yV[v][P] - pV[P] - aa
        return sse_out[v] + float(e @ e) - 2.0 * G.BGRID * float(e @ ss) + G.BGRID ** 2 * float(ss @ ss)

    zero_oracle = {v: min(float(curve(v, h0[v][lam], np.zeros(n_all))[0]) for lam in E.LAMBDAS) for v in VARIANTS}
    out = {"rows_all": n_all, "zero_lam_oracle_sse": zero_oracle, "arms": {}, "airport_share_V": {}}
    nA = {f: z[f"nA_{f}"] for f in E.PRIOR_FAMS}
    for arm in G.ARMS:
        pv0 = F.prior_of(z, arm, nlev)
        out["airport_share_V"][arm] = airport_share((XV @ pv0)[P], aptV[P])
        best = {v: {"free": None, "registered": None} for v in VARIANTS}
        for c in (None,) + tuple(E.C_GRID):
            pv = pv0 if c is None else pv0 * E.support_weights(nA, c)
            for lam in E.LAMBDAS:
                slope = XV @ (lu[lam].solve(lam * pv) if np.isfinite(lam) else pv)
                for v in VARIANTS:
                    cu = curve(v, h0[v][lam], slope)
                    for kind, mask in (("free", np.ones(len(G.BGRID), bool)), ("registered", G.REG_B)):
                        j = int(np.flatnonzero(mask)[np.argmin(cu[mask])])
                        if best[v][kind] is None or cu[j] < best[v][kind]["sse"]:
                            best[v][kind] = {"lam": str(lam), "c": c, "beta": float(G.BGRID[j]), "sse": float(cu[j])}
        out["arms"][arm] = best
    g = E.load_json(G.out_path(r, "null"))
    diffs = [abs(out["arms"][a]["registered"][k]["sse"] - g["arms"][a][f"best_{'free' if k == 'free' else 'registered_beta'}"]["sse"])
             / g["arms"][a][f"best_{'free' if k == 'free' else 'registered_beta'}"]["sse"]
             for a in G.ARMS for k in ("free", "registered")]
    out["check_registered_vs_gridoracle_max_rel_diff"] = float(max(diffs))
    out["seconds"] = round(time.time() - t0, 1)
    print(f"{r}: registered variant against dry-null-gridoracle-{r}.json, largest relative SSE difference "
          f"{max(diffs):.2e} ({out['seconds']:.0f}s)", flush=True)
    return out


def main() -> None:
    NO.refuse_existing([OUT], False)
    launch = E.hold_load("drynull_why")
    d = {r: one(r) for r in E.ORDER}
    if max(d[r]["check_registered_vs_gridoracle_max_rel_diff"] for r in E.ORDER) > 1e-9:
        raise SystemExit("the registered variant does not reproduce gridoracle's dry null")

    def board(v: str, sse: dict) -> float:
        mse = lambda r, x: x / d[r]["rows_all"]                                   # noqa: E731
        return float(np.sqrt(sum(s * mse(r, d[r]["zero_lam_oracle_sse"][v]) for r, s in E.BOARD.items()))
                     - np.sqrt(sum(s * mse(r, sse[r]) for r, s in E.BOARD.items())))
    res = {"what": __doc__.strip().splitlines()[0], "launch": launch, "code_sha256": E.code_shas(),
           "label_reads": E.LABEL_READS, "rotations": d, "board_over_zero_lam_oracle_s": {}}
    for v in VARIANTS:
        x = {}
        for kind in ("free", "registered"):
            b = {a: board(v, {r: d[r]["arms"][a][v][kind]["sse"] for r in E.ORDER}) for a in G.ARMS}
            sh = np.array([b[s] for s in SHAMS])
            x[kind] = {"by_arm": b, "true": b["true"], "true_rank_of_11": int(1 + (sh > b["true"]).sum()),
                       "sham_mean": float(sh.mean()), "sham_max": float(sh.max()),
                       "true_points": {r: {k: d[r]["arms"]["true"][v][kind][k] for k in ("lam", "c", "beta")}
                                       for r in E.ORDER}}
            print(f"{v:15s} {kind:10s} true {b['true']:+.5f} s, rank {x[kind]['true_rank_of_11']} of 11; "
                  f"shams mean {sh.mean():+.5f}, max {sh.max():+.5f}")
        res["board_over_zero_lam_oracle_s"][v] = x
    for r in E.ORDER:
        s = d[r]["airport_share_V"]
        sh = np.array([s[k] for k in SHAMS])
        print(f"{r}: share of the prior's sum of squares on V's P rows in airport means: true {s['true']:.3f}; "
              f"shams mean {sh.mean():.3f}, max {sh.max():.3f}")
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"written {OUT.name}")


if __name__ == "__main__":
    main()
