"""exp-056-q12advocate: post-hoc oracle bound on the taxi-in prior's reach (after both iterations).

Not a registered test and not a correction: V is in-sample here, and nothing is chosen from it.
For one rotation, take the iteration-1 `zero` correction's residual on V's P rows, e = y − max(60,
p + h_zero). Fit, by least squares on V itself, airport intercepts (the oracle month offset) and
then airport intercepts plus linear terms of the prior. The SSE the prior terms remove beyond the
intercepts is the most any additive linear use of that prior could gain on V. Regressor sets:
`q` (the summed unit prior, 1 term), `family` (q_stand, q_rwy, q_op), `support` (each family ×
its own key's departure-count stratum in A: < 50, 50–499, ≥ 500; 9 terms), `routeA` (each family
unweighted and × c/(c + n_A) for c in C_GRID; 18 terms), `airport` (support × airport; up to 90
terms). In-sample fits gain on pure noise too, so each set is also fitted with the ten permuted
priors (`sham01`–`sham10`, the same airport and family value distribution without key identity);
the taxi-in-specific bound is true's gain minus the sham mean (and max).

    .venv/bin/python experiments/exp-056-q12advocate/code/q12oracle.py jul     # reads V labels
    .venv/bin/python experiments/exp-056-q12advocate/code/q12oracle.py --combine
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

PRIORS = ("true",) + tuple(f"sham{s:02d}" for s in E.SHAM_SEEDS)
SETS = ("q", "family", "support", "routeA", "airport")


def unit(z, prior: str, f: str) -> np.ndarray:
    c = z[f"code_V_{f}"]
    q = np.zeros(len(c))
    q[c >= 0] = z[f"prior_{prior}_{f}"][c[c >= 0]]
    return q


def n_of(z, f: str) -> np.ndarray:
    c = z[f"code_V_{f}"]
    return np.where(c >= 0, z[f"nA_{f}"][np.maximum(c, 0)], -1)


def design(z, prior: str, which: str, apt: np.ndarray) -> np.ndarray:
    qf = {f: unit(z, prior, f) for f in E.PRIOR_FAMS}
    if which == "q":
        return (qf["stand"] + qf["rwy"] + qf["op"])[:, None]
    if which == "family":
        return np.column_stack([qf[f] for f in E.PRIOR_FAMS])
    cols = []
    for f in E.PRIOR_FAMS:
        n = n_of(z, f)
        if which == "routeA":
            cols.append(qf[f])
            cols += [qf[f] * np.where(n >= 0, c / (c + np.maximum(n, 0)), 0.0) for c in E.C_GRID]
            continue
        for lo, hi in ((0, E.N_A_BINS[0]), (E.N_A_BINS[0], E.N_A_BINS[1]), (E.N_A_BINS[1], np.inf)):
            s = (n >= lo) & (n < hi)
            if which == "support":
                cols.append(qf[f] * s)
            else:
                cols += [qf[f] * s * (apt == a) for a in np.unique(apt[apt >= 0])]
    X = np.column_stack(cols)
    return X[:, np.abs(X).sum(0) > 0]


def sse_after(e: np.ndarray, D: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(D, e, rcond=None)
    r = e - D @ beta
    return float(r @ r)


def one(r: str) -> None:
    out = E.ANALYSIS / f"oracle-{r}.json"
    NO.refuse_existing([out], False)
    t0 = time.time()
    launch = E.hold_load(f"oracle {r}")
    inp = E.load_json(E.ANALYSIS / f"inputs-{r}.json")
    fit = E.load_json(E.ANALYSIS / f"fit-{r}.json")
    for rec, key in ((inp, "npz"), (fit, "pred_npz")):
        if E.sha_file(E.ROOT / rec[key]) != rec[f"{key}_sha256"]:
            raise SystemExit(f"{rec[key]} is not the frozen file")
    for it in ("iter1.json", "iter2.json"):
        if not (E.ANALYSIS / it).exists():
            raise SystemExit(f"{it} is missing: the oracle runs only after both iterations")
    z = np.load(E.ROOT / inp["npz"])
    h = np.load(E.ROOT / fit["pred_npz"])
    V = E.ROT[r][0]
    y = E.dep_labels(V, z["ids_V"], "post-hoc oracle bound, after both iterations were priced")
    p, pop = z["p_V"], z["pop_V"]
    m = pop & np.isfinite(y)
    e = (y - E.final_pred(p, h["h_zero_main"], pop))[m]
    apt_all = z["code_V_apt"]
    apt = apt_all[m]
    A = (apt[:, None] == np.unique(apt)[None, :]).astype("float64")
    n_all = int(len(y))
    sse0 = float(e @ e)
    sse_apt = sse_after(e, A)
    res = {"what": __doc__.strip().splitlines()[0], "rotation": r, "V": V, "launch": launch,
           "code_sha256": E.code_shas(), "label_reads": E.LABEL_READS, "rows_all": n_all,
           "rows_P": int(m.sum()), "mse_zero_all_rows": float(np.mean((y - E.final_pred(p, h["h_zero_main"], pop)) ** 2)),
           "sse_zero_P": sse0, "sse_airport_oracle_P": sse_apt, "sets": {}}
    for which in SETS:
        g = {}
        for prior in PRIORS:
            X = design(z, prior, which, apt_all)[m]
            g[prior] = {"terms": int(X.shape[1]),
                        "gain_s2_per_row_all": (sse_apt - sse_after(e, np.hstack([A, X]))) / n_all}
        sh = np.array([g[s]["gain_s2_per_row_all"] for s in PRIORS[1:]])
        res["sets"][which] = {"by_prior": g, "true_gain": g["true"]["gain_s2_per_row_all"],
                              "sham_mean": float(sh.mean()), "sham_max": float(sh.max()),
                              "true_rank_of_11": int(1 + (sh > g["true"]["gain_s2_per_row_all"]).sum())}
        print(f"  {r} {which:8s} terms {g['true']['terms']:3d}  true {g['true']['gain_s2_per_row_all']:8.3f}"
              f"  sham mean {sh.mean():8.3f} max {sh.max():8.3f} s2/row", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([out], False)
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"written {out.name}")


def combine() -> None:
    out = E.ANALYSIS / "oracle.json"
    NO.refuse_existing([out], False)
    d = {r: E.load_json(E.ANALYSIS / f"oracle-{r}.json") for r in E.ORDER}
    res = {"what": "post-hoc oracle bound, board recombination (V in-sample; not a test)", "sets": {}}
    for which in SETS:
        row = {}
        for key in ("true_gain", "sham_mean", "sham_max"):
            mse_apt = {r: d[r]["mse_zero_all_rows"] - (d[r]["sse_zero_P"] - d[r]["sse_airport_oracle_P"]) / d[r]["rows_all"]
                       for r in E.ORDER}
            fold = {r: float(np.sqrt(mse_apt[r]) - np.sqrt(mse_apt[r] - d[r]["sets"][which][key])) for r in E.ORDER}
            board = float(np.sqrt(sum(s * mse_apt[r] for r, s in E.BOARD.items()))
                          - np.sqrt(sum(s * (mse_apt[r] - d[r]["sets"][which][key]) for r, s in E.BOARD.items())))
            row[key] = {"fold_rmse_gain_s": fold, "board_rmse_gain_s": board}
        row["terms"] = {r: d[r]["sets"][which]["by_prior"]["true"]["terms"] for r in E.ORDER}
        row["true_rank_of_11"] = {r: d[r]["sets"][which]["true_rank_of_11"] for r in E.ORDER}
        res["sets"][which] = row
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    for which, x in res["sets"].items():
        print(f"{which:8s} board: true {x['true_gain']['board_rmse_gain_s']:+.4f} s, sham mean "
              f"{x['sham_mean']['board_rmse_gain_s']:+.4f}, sham max {x['sham_max']['board_rmse_gain_s']:+.4f}; "
              f"rank {x['true_rank_of_11']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rotation", nargs="?", choices=E.ORDER)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    if a.combine == (a.rotation is not None):
        raise SystemExit("give one rotation, or --combine")
    combine() if a.combine else one(a.rotation)
