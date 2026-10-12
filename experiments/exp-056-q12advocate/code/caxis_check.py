"""exp-056-q12advocate: the c axis is a no-op for the arms without a prior (review round 1, finding 6).

Iteration 2 gives each arm with a prior the grid c x lam x beta and each arm without one (zero,
const, literal_sham) the grid lam x beta. This check shows that the c axis cannot change those
arms, so the one-c grid is the five-c grid with four exact copies removed. No label is read.

1. On the frozen inputs of every rotation, each such arm's prior vector has no nonzero entry, and
   neither has prior x c / (c + n_A) for any c in C_GRID (the weights are finite and positive).
   So fit.py's slope part is the zero vector at every (c, lam), and parts[c] is the same for
   every c.
2. With parts identical across c, `select_c` returns the same (lam, beta) and SSE as `select`
   (it visits beta up, lam down, c down and keeps the first strict minimum). Checked on the
   real selection functions with random parts, labels and ties, 200 cases.
3. The records agree: in every rotation, iteration 2's zero, const and literal_sham picks and
   prediction hashes equal iteration 1's.

    .venv/bin/python experiments/exp-056-q12advocate/code/caxis_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402
import fit as F                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

OUT = E.ANALYSIS / "caxis-check.json"
NOPRIOR = ("zero", "const", "literal_sham")


def main() -> None:
    NO.refuse_existing([OUT], False)
    res = {"what": __doc__.strip().splitlines()[0], "code_sha256": E.code_shas(), "priors": {}, "records": {}}
    for r in E.ORDER:
        inp = E.load_json(E.ANALYSIS / f"inputs-{r}.json")
        if E.sha_file(E.ROOT / inp["npz"]) != inp["npz_sha256"]:
            raise SystemExit("the inputs are not the frozen file")
        z = np.load(E.ROOT / inp["npz"])
        nlev = {f: int(inp["vocab_sizes"][f]) for f in E.FAMS}
        nA = {f: z[f"nA_{f}"] for f in E.PRIOR_FAMS}
        for arm in NOPRIOR:
            pv0 = F.prior_of(z, arm, nlev)
            w = {c: E.support_weights(nA, c) for c in E.C_GRID}
            res["priors"][f"{r}|{arm}"] = {
                "prior_nonzero": int(np.count_nonzero(pv0)),
                "weights_finite_positive": bool(all(np.all(np.isfinite(v) & (v > 0)) for v in w.values())),
                "weighted_nonzero": {str(c): int(np.count_nonzero(pv0 * w[c])) for c in E.C_GRID}}
        f1 = E.load_json(E.ANALYSIS / f"fit-{r}.json")["arms"]
        f2 = E.load_json(E.ANALYSIS / f"it2-fit-{r}.json")["arms"]
        for arm in NOPRIOR:
            res["records"][f"{r}|{arm}"] = {
                "same_pick": (f1[arm]["lam"], f1[arm]["beta"]) == (f2[arm]["lam"], f2[arm]["beta"]),
                "it2_c": f2[arm].get("c"), "same_h_sha256": f1[arm]["h_sha256"] == f2[arm]["h_sha256"]}
    # the selection functions: parts identical across c -> the same (lam, beta) and SSE
    rng = np.random.default_rng(56006)
    cases, agree = 200, 0
    for i in range(cases):
        n = int(rng.integers(20, 200))
        y = {k: np.round(rng.normal(600, 300, n)) for k in ("S", "I")}
        p = {k: np.round(rng.normal(600, 300, n)) for k in ("S", "I")}
        pop = {k: rng.random(n) < 0.9 for k in ("S", "I")}
        # coarse values make exact SSE ties common, so the tie order is exercised
        parts = {d: {lam: (np.round(rng.normal(0, 50, n), -1) * (i % 3 != 0), np.zeros(n))
                     for lam in E.LAMBDAS} for d in ("S>I", "I>S")}
        b1, _ = E.select(parts, y, p, pop)
        b2, _ = E.select_c({c: parts for c in E.C_GRID}, y, p, pop)
        agree += int((b1["lam"], b1["beta"], b1["sse"]) == (b2["lam"], b2["beta"], b2["sse"]))
    res["selection_unit_check"] = {"cases": cases, "agree": agree}
    ok = (all(v["prior_nonzero"] == 0 and v["weights_finite_positive"] and not any(v["weighted_nonzero"].values())
              for v in res["priors"].values())
          and all(v["same_pick"] and v["same_h_sha256"] for v in res["records"].values()) and agree == cases)
    res["pass"] = bool(ok)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"priors zero under every c: {all(not any(v['weighted_nonzero'].values()) for v in res['priors'].values())}; "
          f"records agree: {all(v['same_pick'] and v['same_h_sha256'] for v in res['records'].values())}; "
          f"selection {agree}/{cases}; pass {ok}")
    if not ok:
        raise SystemExit("the c-axis check fails")


if __name__ == "__main__":
    main()
