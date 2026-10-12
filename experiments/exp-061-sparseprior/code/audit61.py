"""exp-061-sparseprior: THE SEPARATE AUDIT PROCESS (assignment, "Registration and frozen sources").

It runs only after every exp-061 setting, loss, verdict and hash is frozen, and it is the ONLY job
allowed to open exp-060's results.  No producing job of exp-061 imports it, and nothing it writes
is read by any producing job: `analysis/audit61.json` is a terminal artifact.

It reproduces, from exp-060's own frozen inputs and receipts:

  1. exp-060's `zero` correction --- refitted here from the exp-060 input archive and the S labels
     at exp-060's recorded zero lambda, and compared to the array hash exp-060 recorded;
  2. exp-060's `zero` V per-row loss --- recomputed from that correction and compared, row by row,
     with the column of exp-060's frozen `oos-rows.parquet`;
  3. exp-060's sixteen-condition gate --- recomputed from those frozen row losses and draws and
     compared with every number in exp-060's `analysis/oos.json`;
  4. exp-060's primary bound --- the board gains, the rank, the winner-adjusted point and the
     three-day band recombined from exp-060's per-rotation bound records and per-draw artifacts
     and compared with `analysis/bound-r2.json` (SPEC reading (g));
  5. exp-060's final 35-input hash sweep --- every artifact its audit jobs read, against the hash
     its producing record carries.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-061-sparseprior/code/audit61.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e61 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

E60A = E.E60EXP / "analysis"
TOL = 1e-12


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = E.out_path(E.ANALYSIS / "audit61.json")
    NO.refuse_existing([out], a.overwrite)
    t0 = time.time()
    launch = E.hold_load("audit")

    # exp-061 must be finished before the audit starts
    need = ([E.ANALYSIS / "oos61.json", E.ANALYSIS / "bound61.json"]
            + [E.ANALYSIS / f"{n}61-{r}.json" for n in ("inputs", "fit", "bound") for r in E.ORDER]
            + [E.ANALYSIS / f"mask61-{r}.json" for r in E.ORDER])
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"the audit runs after exp-061 is frozen; missing {missing}")

    rec = {"what": __doc__.strip().splitlines()[0], "launch": launch, "code_sha256": E.code_shas(),
           "exp061_records_present": [str(p.relative_to(E.ROOT)) for p in need],
           "zero_correction": {}, "zero_v_row_loss": {}, "gate": {}, "bound": {}, "hash_sweep": {}}
    bad = []

    oos60 = E.load_json(E60A / "oos.json")
    rows60 = pd.read_parquet(E.ROOT / oos60["artifacts"]["rows"]["path"])
    draws60 = np.load(E.ROOT / oos60["artifacts"]["draws"]["path"])

    # ---- 1 and 2: the zero correction and its V row loss ------------------------------------
    for r in E.ORDER:
        in60 = E.load_json(E60A / f"inputs60-{r}.json")
        fit60 = E.load_json(E60A / f"fit60-{r}.json")
        E.pin_months(E.load_json(E.ANALYSIS / f"inputs61-{r}.json"))
        z = np.load(E.ROOT / in60["npz"])
        nlev = {f: int(in60["vocab_sizes"][f]) for f in E.FAMS}
        codes = {k: {f: z[f"code_{k}_{f}"] for f in E.FAMS} for k in ("S", "V")}
        V, S = in60["months"]["V"], in60["months"]["S"]
        yS = E.dep_labels(S, z["ids_S"], f"audit: exp-060's zero correction, {r}")
        fit_rows = z["pop_S"] & np.isfinite(yS) & (yS > 0)
        X = E.design(codes["S"], nlev, fit_rows)
        res_S = E.demean_by_airport(np.clip(yS[fit_rows] - z["p_S"][fit_rows], -E.CLIP, E.CLIP),
                                    codes["S"]["apt"][fit_rows])
        XV = E.design(codes["V"], nlev)
        ent = {}
        for lev in ("B", "A"):
            lam_s = fit60["arms"]["zero"]["levels"][lev]["lam"]
            lam = float("inf") if lam_s == "inf" else float(lam_s)
            th = (np.zeros(X.shape[1]) if np.isinf(lam)
                  else splu((X.T @ X + lam * sp.identity(X.shape[1], format="csr")).tocsc())
                  .solve(np.asarray(X.T @ res_S, dtype="float64")))
            h = XV @ th
            got, want = E.sha_arr(h), fit60["array_sha256"][f"h_zero_{lev}"]
            ent[lev] = {"lambda": lam_s, "recomputed_sha256": got, "recorded_sha256": want,
                        "equal": got == want}
            if got != want:
                bad.append(f"exp-060 zero correction {r} level {lev}")
            if lev == "B":
                yV = E.dep_labels(V, z["ids_V"], f"audit: exp-060's zero V row loss, {r}")
                loss = (yV - E.final_pred(z["p_V"], h, z["pop_V"])) ** 2
                sub = rows60[rows60["rotation"] == r]
                if not np.array_equal(sub["MVT_ID_mvt"].to_numpy("int64"),
                                      z["ids_V"].astype("int64")):
                    bad.append(f"exp-060 oos-rows id order {r}")
                d = float(np.max(np.abs(loss - sub["loss:zero|B"].to_numpy("float64"))))
                rec["zero_v_row_loss"][r] = {
                    "rows": int(len(loss)), "max_abs_difference": d, "equal": d == 0.0,
                    "recomputed_sha256": E.sha_arr(loss),
                    "recorded_sha256": oos60["row_loss_sha256"][r]["zero|B"]}
                if d != 0.0:
                    bad.append(f"exp-060 zero V row loss {r}")
        rec["zero_correction"][r] = ent

    # ---- 3: exp-060's sixteen-condition gate, recomputed from the frozen row losses ----------
    di = {r: rows60[rows60["rotation"] == r]["day_index"].to_numpy("int64") for r in E.ORDER}
    k = {r: int(di[r].max()) + 1 for r in E.ORDER}
    loss = {r: {c: rows60[rows60["rotation"] == r][f"loss:{c}"].to_numpy("float64")
                for c in [f"{aa}|B" for aa in E.E60.ARMS]} for r in E.ORDER}
    dr = {r: {L: draws60[f"{r}_{L}"] for L in E.BLOCK_LENGTHS} for r in E.ORDER}
    for r in E.ORDER:
        for L in E.BLOCK_LENGTHS:
            want = E.block_draws(k[r], L, E.draw_seed(r, L))
            if not np.array_equal(want, dr[r][L]):
                bad.append(f"exp-060 draws {r} block {L}")

    def board_gain(A_: str, B_: str) -> dict:
        ma = sum(E.BOARD[r] * loss[r][A_].mean() for r in E.BOARD)
        mb = sum(E.BOARD[r] * loss[r][B_].mean() for r in E.BOARD)
        o = {"board_gain_s": float(np.sqrt(ma) - np.sqrt(mb))}
        for L in E.BLOCK_LENGTHS:
            m0 = np.zeros(E.BOOT)
            m1 = np.zeros(E.BOOT)
            for r, s in E.BOARD.items():
                e0 = np.bincount(di[r], weights=loss[r][A_], minlength=k[r])
                e1 = np.bincount(di[r], weights=loss[r][B_], minlength=k[r])
                cnt = np.bincount(di[r], minlength=k[r]).astype("float64")
                idx = dr[r][L]
                m0 += s * e0[idx].sum(1) / cnt[idx].sum(1)
                m1 += s * e1[idx].sum(1) / cnt[idx].sum(1)
            o[f"ci_block_{L}d"] = E.ci(np.sqrt(m0) - np.sqrt(m1))
        return o

    gate = {}
    for c in E.E60.GATE_CONTROLS:
        got = board_gain(f"{c}|B", "true|B")
        want = oos60["gate"][c]
        ok = (rel(got["board_gain_s"], want["board_gain_s"]) < TOL
              and max(rel(got["ci_block_1d"][i], want["board_ci_1d"][i]) for i in (0, 1)) < TOL)
        gate[c] = {"recomputed_board_gain_s": got["board_gain_s"],
                   "recorded_board_gain_s": want["board_gain_s"],
                   "recomputed_ci_1d": got["ci_block_1d"], "recorded_ci_1d": want["board_ci_1d"],
                   "recorded_pass": want["pass"], "equal": bool(ok)}
        if not ok:
            bad.append(f"exp-060 gate {c}")

    def board_rmse_draws(arm: str, L: int) -> np.ndarray:
        m = np.zeros(E.BOOT)
        for r, s in E.BOARD.items():
            e = np.bincount(di[r], weights=loss[r][f"{arm}|B"], minlength=k[r])
            cnt = np.bincount(di[r], minlength=k[r]).astype("float64")
            idx = dr[r][L]
            m += s * e[idx].sum(1) / cnt[idx].sum(1)
        return np.sqrt(m)

    pt = {arm: float(np.sqrt(sum(E.BOARD[r] * loss[r][f"{arm}|B"].mean() for r in E.BOARD)))
          for arm in E.E60.ARMS}
    best = min(E.E60.SHAM_ARMS, key=lambda s: pt[s])
    w60 = oos60["winner_adjusted"]["B"]
    mat = np.stack([board_rmse_draws(s, 1) for s in E.E60.SHAM_ARMS])
    g1 = mat.min(0) - board_rmse_draws("true", 1)
    wpt = pt[best] - pt["true"]
    ok = (rel(wpt, w60["point_s"]) < TOL and best == w60["winning_sham_on_the_original_rows"]
          and max(rel(E.ci(g1)[i], w60["ci_block_1d"][i]) for i in (0, 1)) < TOL)
    gate["winner_adjusted"] = {"recomputed_point_s": wpt, "recorded_point_s": w60["point_s"],
                               "recomputed_ci_1d": E.ci(g1), "recorded_ci_1d": w60["ci_block_1d"],
                               "recomputed_winner": best,
                               "recorded_winner": w60["winning_sham_on_the_original_rows"],
                               "equal": bool(ok)}
    if not ok:
        bad.append("exp-060 winner-adjusted gate")
    gate["recorded_advance"] = oos60["gate"]["advance"]
    rec["gate"] = gate

    # ---- 4: exp-060's primary bound, recombined from its own per-rotation records -------------
    b60 = E.load_json(E60A / "bound-r2.json")
    per = {r: E.load_json(E60A / f"bound-r2-{r}.json") for r in E.ORDER}
    bands = {}
    for r in E.ORDER:
        aa = per[r]["band_artifact"]
        got = E.sha_file(E.ROOT / aa["path"])
        if got != aa["sha256"]:
            bad.append(f"exp-060 band artifact {r}")
        bands[r] = np.load(E.ROOT / aa["path"])
    key = b60["winner_adjusted_three_day_band"]["level"]

    def bboard(x: dict, y_: dict) -> float:
        ma = sum(s * x[r] / per[r]["rows_all"] for r, s in E.BOARD.items())
        mb = sum(s * y_[r] / per[r]["rows_all"] for r, s in E.BOARD.items())
        return float(np.sqrt(ma) - np.sqrt(mb))

    zero = {r: per[r]["arms"]["zero"][key]["sse"] for r in E.ORDER}
    gain = {arm: bboard(zero, {r: per[r]["arms"][arm][key]["sse"] for r in E.ORDER})
            for arm in E.E60.ARMS}
    v = np.array([gain[s] for s in E.E60.SHAM_ARMS])
    rmse = {arm: np.sqrt(sum(s * bands[r][f"{arm}_mse"] for r, s in E.BOARD.items()))
            for arm in E.E60.ARMS}
    gb = np.stack([rmse[s] for s in E.E60.SHAM_ARMS]).min(0) - rmse["true"]
    ptb = {arm: float(np.sqrt(sum(s * per[r]["arms"][arm][key]["sse"] / per[r]["rows_all"]
                                  for r, s in E.BOARD.items()))) for arm in E.E60.ARMS}
    bs = min(E.E60.SHAM_ARMS, key=lambda s: ptb[s])
    lb60, wb60 = b60["levels"][key], b60["winner_adjusted_three_day_band"]
    checks = {
        "true_board_gain_s": (gain["true"], lb60["board_gain_s"]["true"]),
        "sham_max_s": (float(v.max()), lb60["sham_max_s"]),
        "true_rank_of_11": (int(1 + (v > gain["true"]).sum()), lb60["true_rank_of_11"]),
        "winner_adjusted_point_s": (ptb[bs] - ptb["true"], wb60["point_s"]),
        "winner_adjusted_3d_low": (E.ci(gb)[0], wb60["ci_block_3d"][0]),
        "winner_adjusted_3d_high": (E.ci(gb)[1], wb60["ci_block_3d"][1]),
        "gain_positive_share": (float((gb > 0).mean()), wb60["gain_positive_share"])}
    rec["bound"] = {"level": key,
                    "items": {kk: {"recomputed": float(x), "recorded": float(y_),
                                   "equal": bool(rel(float(x), float(y_)) < 1e-9)}
                              for kk, (x, y_) in checks.items()},
                    "recorded_bound_condition_met": b60["reading_inputs"]["bound_condition_met"]}
    for kk, it in rec["bound"]["items"].items():
        if not it["equal"]:
            bad.append(f"exp-060 bound {kk}")

    # ---- 5: exp-060's final 35-input hash sweep ------------------------------------------------
    sweep, sbad = {}, []

    def check(path, want, why):
        got = E.sha_file(E.ROOT / path)
        sweep[str(path)] = {"recorded": want, "measured": got, "equal": got == want,
                            "read_by": why}
        if got != want:
            sbad.append(str(path))

    for r in E.ORDER:
        inp = E.load_json(E60A / f"inputs60-{r}.json")
        check(inp["npz"], inp["npz_sha256"], ["identity60", "ceiling60_r1", "align60", "grid60"])
        cold = E.load_json(E60A / f"cold-{r}.json")
        check(cold["table"]["path"], cold["table"]["sha256"], ["freeze60", "answers60"])
        fit = E.load_json(E60A / f"fit60-{r}.json")
        check(fit["pred_npz"], fit["pred_npz_sha256"], ["price60", "bound60"])
        for arm in ("base", "zero", "true"):
            relp = f"experiments/exp-060-coldprior/oof/{arm}-iso_{r}_v.parquet"
            check(relp, oos60["artifacts"]["oof"][relp], ["align60", "ceiling60_r1", "check60"])
        rr = E.load_json(E60A / f"bound-r2-{r}.json")
        for kk in ("band_artifact", "grid_artifact"):
            check(rr[kk]["path"], rr[kk]["sha256"], ["banddraws60_r1"])
        rb = E.load_json(E60A / f"bound-{r}.json")
        check(rb["band_artifact"]["path"], rb["band_artifact"]["sha256"], ["banddraws60"])
        r1 = E.load_json(E60A / f"bound-r1-{r}.json")
        check(r1["band_artifact"]["path"], r1["band_artifact"]["sha256"], ["banddraws60"])
        g = E.load_json(E60A / f"grid60-{r}.json")
        check(g["row_losses"]["path"], g["row_losses"]["sha256"], ["the record itself"])
    for kk in ("rows", "draws"):
        aa = oos60["artifacts"][kk]
        check(aa["path"], aa["sha256"], ["bound60", "check60"])
    v60 = E.load_json(E60A / "verifyinputs60.json")
    rec["hash_sweep"] = {"count": len(sweep), "all_equal": not sbad, "mismatches": sbad,
                         "exp060_recorded_count": v60["count"],
                         "exp060_recorded_all_equal": v60["all_equal"],
                         "same_artifact_set": sorted(sweep) == sorted(v60["items"]),
                         "items": sweep}
    if sbad:
        bad += [f"exp-060 hash sweep {x}" for x in sbad]
    if not rec["hash_sweep"]["same_artifact_set"]:
        bad.append("exp-060 hash sweep artifact set")

    rec["all_reproduced"] = not bad
    rec["mismatches"] = bad
    rec["label_reads"] = E.LABEL_READS
    rec["terminal_artifact"] = ("no exp-061 producing job reads analysis/audit61.json; it is "
                                "written after every exp-061 record exists")
    rec["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({"zero_correction": all(v2["equal"] for e in rec["zero_correction"].values()
                                             for v2 in e.values()),
                      "zero_v_row_loss": all(v2["equal"] for v2 in rec["zero_v_row_loss"].values()),
                      "gate": all(v2.get("equal", True) for v2 in rec["gate"].values()
                                  if isinstance(v2, dict)),
                      "bound": all(v2["equal"] for v2 in rec["bound"]["items"].values()),
                      "hash_sweep": rec["hash_sweep"]["all_equal"],
                      "all_reproduced": rec["all_reproduced"]}, indent=1))
    if bad:
        raise SystemExit(f"REFUSING: the audit did not reproduce {len(bad)} items: {bad}")
    print(f"written {out.name} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
