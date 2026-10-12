"""exp-056-q12advocate: the time-origin control with exactly equal fit rows (post-hoc; review round 1, finding 5).

SPEC 3.5 thins S and I to the same population count before any label read; fit.py then keeps the
rows with a finite, positive label, so the two fit sets end 5 to 6 rows apart. Here the larger
kept set is thinned again by the same registered key (the smallest sha256("exp056-origin|<id>"))
to the smaller set's count, so both fits use exactly the same number of rows. Arms (zero, true),
picks (lam, beta, c of each iteration), residual, solve and V scoring are fit.py's and price.py's;
only the row sets change. Checks: the registered row sets reproduce fit.py's recorded originS and
originI prediction hashes byte for byte, and the combine reproduces iter1.json's and iter2.json's
origin contrasts from them. Post-hoc; nothing is chosen from it.

    .venv/bin/python experiments/exp-056-q12advocate/code/origin_r1.py jul    # reads S, I and V of jul
    .venv/bin/python experiments/exp-056-q12advocate/code/origin_r1.py --combine    # no label read
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
import price as PR                                                    # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

ARMS = ("zero", "true")
ROWSETS = ("registered", "exact")
OUT = E.ANALYSIS / "origin-r1.json"


def paths(r: str) -> tuple[Path, Path]:
    return E.ANALYSIS / f"origin-r1-{r}.json", E.ART / f"origin-r1-{r}-loss.npz"


def one(r: str) -> None:
    out_rec, out_npz = paths(r)
    NO.refuse_existing([out_rec, out_npz], False)
    t0 = time.time()
    launch = E.hold_load(f"origin_r1 {r}")
    inp = E.load_json(E.ANALYSIS / f"inputs-{r}.json")
    fits = {1: E.load_json(E.ANALYSIS / f"fit-{r}.json"), 2: E.load_json(E.ANALYSIS / f"it2-fit-{r}.json")}
    if E.sha_file(E.ROOT / inp["npz"]) != inp["npz_sha256"]:
        raise SystemExit("the inputs are not the frozen file")
    z = np.load(E.ROOT / inp["npz"])
    V, S, I, _ = E.ROT[r]
    nlev = {f: int(inp["vocab_sizes"][f]) for f in E.FAMS}
    y = {"S": E.dep_labels(S, z["ids_S"], "post-hoc exact time-origin control: the S fit"),
         "I": E.dep_labels(I, z["ids_I"], "post-hoc exact time-origin control: the I fit"),
         "V": E.dep_labels(V, z["ids_V"], "post-hoc exact time-origin control: scoring on V")}
    codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("V", "S", "I")}
    XV = E.design(codes["V"], nlev)
    reg = {k: z[f"pop_{k}"] & np.isfinite(y[k]) & (y[k] > 0) & z[f"thin_{k}"] for k in ("S", "I")}
    m = min(int(reg["S"].sum()), int(reg["I"].sum()))
    exact = {k: E.thin_mask(z[f"ids_{k}"], reg[k], m) for k in ("S", "I")}
    for k in ("S", "I"):
        assert int(exact[k].sum()) == m and not np.any(exact[k] & ~reg[k])
    rows = {"registered": reg, "exact": exact}
    nA = {f: z[f"nA_{f}"] for f in E.PRIOR_FAMS}
    pV, popV = z["p_V"], z["pop_V"]
    loss, rec_h, check = {}, {}, {}
    for rs in ROWSETS:
        for k in ("S", "I"):
            rr = rows[rs][k]
            X = E.design(codes[k], nlev, rr)
            res = E.demean_by_airport(np.clip(y[k][rr] - z[f"p_{k}"][rr], -E.CLIP, E.CLIP), codes[k]["apt"][rr])
            lu_c = {}
            for it in (1, 2):
                for arm in ARMS:
                    a = fits[it]["arms"][arm]
                    lam = float("inf") if a["lam"] == "inf" else float(a["lam"])
                    beta, c = a["beta"], a.get("c")
                    pv0 = F.prior_of(z, arm, nlev)
                    pv = pv0 if c is None else pv0 * E.support_weights(nA, c)
                    # fit.py's data_part + beta * slope_part, operation for operation
                    if np.isinf(lam):
                        t0_, t1_ = np.zeros_like(pv), pv.copy()
                    else:
                        if lam not in lu_c:
                            lu_c[lam] = splu((X.T @ X + lam * sp.identity(X.shape[1], format="csr")).tocsc())
                        t0_ = lu_c[lam].solve(np.asarray(X.T @ res, dtype="float64"))
                        t1_ = lu_c[lam].solve(lam * pv) if np.any(pv) else np.zeros_like(pv)
                    h = XV @ (t0_ + beta * t1_)
                    key = f"it{it}|{arm}|origin{k}|{rs}"
                    loss[key] = PR.arm_loss(y["V"], pV, popV, h)
                    rec_h[key] = E.sha_arr(h)
                    if rs == "registered":
                        check[key] = rec_h[key] == a["h_sha256"][f"origin{k}"]
    if not all(check.values()):
        raise SystemExit(f"the registered rows do not reproduce fit.py's origin predictions: {check}")
    NO.refuse_existing([out_rec, out_npz], False)
    np.savez(out_npz, **{k.replace("|", "__"): v for k, v in loss.items()})
    rec = {"what": __doc__.strip().splitlines()[0], "rotation": r, "months": {"V": V, "S": S, "I": I},
           "launch": launch, "code_sha256": E.code_shas(), "label_reads": E.LABEL_READS,
           "fit_rows": {rs: {k: int(rows[rs][k].sum()) for k in ("S", "I")} for rs in ROWSETS},
           "rows_dropped_to_equalize": {k: int(reg[k].sum()) - m for k in ("S", "I")},
           "h_sha256": rec_h, "registered_rows_reproduce_fit_py": check,
           "loss_npz": str(out_npz.relative_to(E.ROOT)), "loss_npz_sha256": E.sha_file(out_npz),
           "seconds": round(time.time() - t0, 1)}
    out_rec.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"{r}: fit rows {rec['fit_rows']}; registered rows reproduce fit.py: {all(check.values())}; "
          f"written {out_rec.name} ({rec['seconds']:.0f}s)")


def combine() -> None:
    NO.refuse_existing([OUT], False)
    loss, di, k, draws = {}, {}, {}, {}
    for r in E.ORDER:
        rec_p, npz_p = paths(r)
        rec = E.load_json(rec_p)
        if E.sha_file(npz_p) != rec["loss_npz_sha256"]:
            raise SystemExit(f"{npz_p.name} is not the recorded file")
        zz = np.load(npz_p)
        loss[r] = {key.replace("__", "|"): zz[key] for key in zz.files}
        inp = E.load_json(E.ANALYSIS / f"inputs-{r}.json")
        di[r] = np.load(E.ROOT / inp["npz"])["day_V"]
        k[r] = int(di[r].max()) + 1
        draws[r] = {L: E.block_draws(k[r], L, E.draw_seed(r, L)) for L in E.BLOCK_LENGTHS}

    def contrast(A_: str, B_: str) -> dict:
        folds = {r: PR.fold_stats(loss[r][A_], loss[r][B_], di[r], k[r], draws[r]) for r in E.ORDER}
        board = PR.board_stats({r: loss[r][A_] for r in E.BOARD}, {r: loss[r][B_] for r in E.BOARD},
                               {r: di[r] for r in E.BOARD}, {r: k[r] for r in E.BOARD},
                               {r: draws[r] for r in E.BOARD})
        return {"A": A_, "B": B_, "folds": folds, "board": board}
    res = {"what": "exact time-origin control, contrasts (post-hoc; not a test)", "contrasts": {}, "reproduce": {}}
    worst = 0.0
    for it in (1, 2):
        reg_rec = E.load_json(E.ANALYSIS / f"iter{it}.json")["contrasts"]
        for rs in ROWSETS:
            t = f"it{it}|{{}}|origin{{}}|{rs}"
            cs = {f"zero_originS_vs_originI": contrast(t.format("zero", "I"), t.format("zero", "S")),
                  f"true_originS_vs_originI": contrast(t.format("true", "I"), t.format("true", "S")),
                  f"true_vs_zero_originS": contrast(t.format("zero", "S"), t.format("true", "S")),
                  f"true_vs_zero_originI": contrast(t.format("zero", "I"), t.format("true", "I"))}
            res["contrasts"][f"it{it}|{rs}"] = cs
            if rs == "registered":
                pairs = {"zero_originS_vs_originI": reg_rec["zero_originS_vs_originI"],
                         "true_originS_vs_originI": reg_rec["true_originS_vs_originI"],
                         "true_vs_zero_originS": reg_rec["true_vs_zero"]["originS"],
                         "true_vs_zero_originI": reg_rec["true_vs_zero"]["originI"]}
                for name, x in pairs.items():
                    d = abs(cs[name]["board"]["board_gain_s"] - x["board"]["board_gain_s"])
                    d = max([d] + [abs(a - b) for L in E.BLOCK_LENGTHS for a, b in
                                   zip(cs[name]["board"][f"ci_block_{L}d"], x["board"][f"ci_block_{L}d"])])
                    res["reproduce"][f"it{it}|{name}"] = d
                    worst = max(worst, d)
    res["reproduce_max_abs_diff_s"] = worst
    if worst > 1e-9:
        raise SystemExit(f"the registered rows do not reproduce the priced contrasts: {worst}")
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    for key, cs in res["contrasts"].items():
        for name, x in cs.items():
            b = x["board"]
            print(f"{key:15s} {name:24s} board {b['board_gain_s']:+.4f} s, 1-day [{b['ci_block_1d'][0]:+.4f}, "
                  f"{b['ci_block_1d'][1]:+.4f}], 3-day [{b['ci_block_3d'][0]:+.4f}, {b['ci_block_3d'][1]:+.4f}]; "
                  f"folds s2/row " + " ".join(f"{r} {x['folds'][r]['mse_gain_s2_per_row']:+.2f}" for r in E.ORDER))
    print(f"registered rows reproduce the priced contrasts: max |diff| {worst:.1e} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rotation", nargs="?", choices=E.ORDER)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    if a.combine == (a.rotation is not None):
        raise SystemExit("give one rotation, or --combine")
    combine() if a.combine else one(a.rotation)
