"""Q09 step 3 and step 4: each arm through v10's full serving path, then priced.

    PRC_THREADS=4 python experiments/exp-040-chain/code/q09.py [--arms none,chain,derived]

Nothing about the serving path is reimplemented.  The recipe is exp-038
`code/q02.py`'s -- exp-027's mixed anchor (blend-K off the NM tail, blend-5 on
it) + the 600 s `depx` stacker + `noac` held at blend-5's anchor + v7's EHAM mask
+ exp-036's eight-airport mask -- built from exp-034's harness
(`candidate.py`), exp-026's blend rule and frames, exp-028's stacker and
exp-030's scoring.  The one thing this file changes is the address of the
`L2-hub8725` member: `F.OOF["L2-hub8725"]` points at the arm's out-of-fold file
instead of the published one, and every stage downstream is refitted the same
way for every arm.

Two gates run before any arm number is printed:

  A  blend-5 on the seven committed members reproduces `blend-config-5.json`
  F  the v10 replay, built from the COMMITTED member frames upward, lands on
     exp-036's published triple 284.1572 / 204.9218 / 218.1506

Then each arm is priced against TWO baselines, and both are in the table:

  vs v10     the contract's endpoint.  It carries the machine term as well as
             the chain, because the arm's member was fitted on Kaggle and v10's
             on the workstation (exp-025 §10.6).
  vs none    the same member producer in the same Kaggle session with only the
             column list changed.  This is the chain, and nothing else.

Two serving variants of every arm, because v10's `noac` hold is at BLEND-5's
anchor and blend-5's anchor is a function of the members: changing
`L2-hub8725` moves the `noac` rows too.

  served     the recipe exactly as v10 states it
  d11        the same with the `noac` rows pinned to v10's own values, which is
             what a candidate would have to ship to keep score-feedback rule's bytes

Outputs `analysis/q09-arms.json`, `analysis/q09-table.csv`,
`analysis/q09-slices.csv` and `packet/q09-{fold}.parquet`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-038-packet/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-034-bulk/code"))
import align as A  # noqa: E402
import candidate as C  # noqa: E402

F, R, S, SW, Sc = C.F, C.R, C.S, C.SW, C.Sc
B, D = C.B, C.D
FOLDS = C.FOLDS
THREADS = int(os.environ.get("PRC_THREADS", "4"))
SW.THREADS = THREADS

#: The member whose slot the arm fills.  A module global rather than a constant
#: because step 6 runs this same path on `C2-cat`, the other retro-v2 member
#: (`--member C2-cat --oof .../c2-{arm}--{fold}.parquet --tag q11`): everything
#: else about the serving path -- the anchor, the stacker, the `noac` hold, the
#: two masks, the gates -- is identical, and a second copy of this file would be a
#: second thing to keep right.
MEMBER = "L2-hub8725"
NAMES5 = C.NAMES5
NAMESK2 = C.NAMESK2
SEEDS = S.SEEDS
GATE = 1e-3
PUB_V10 = {"july_loo": 284.1572, "december": 204.9218, "january": 218.1506}
MASK = json.loads((ROOT / "experiments/exp-036-mask/analysis/mask.json").read_text())
PUBLISHED_OOF = F.OOF[MEMBER]
ARM_OOF = "experiments/exp-040-chain/oof/{arm}--{fold}.parquet"
#: the tail member's fixed high-band table starts here (`prc.tail_model.BAND_MIN`)
BAND_MIN = 8 * 3600.0


def frames_for(arm: str | None) -> dict:
    """v10's member frame per fold, with `L2-hub8725` taken from `arm`.

    `arm=None` is the published member, which is the v10 replay.  The frame is
    exp-026's `fold_frame`, so the floor and the residual guard are applied to
    every member exactly as `prc.blend.build_candidate` applies them.
    """
    F.OOF[MEMBER] = (PUBLISHED_OOF if arm is None
                     else ARM_OOF.replace("{arm}", arm))
    out = {}
    for f in FOLDS:
        m = F.fold_frame(f, NAMES5 + (C.K2,))
        for n in NAMESK2:
            # `N1-nmtail` is the specialist and is null outside `nm_late`; every
            # other member of v10's seven is defined on every row, and a null
            # that appeared in one of them would silently change the blend.
            if n in F.PARTIAL:
                continue
            if not np.isfinite(m[n].to_numpy("float64")).all():
                raise SystemExit(f"{f}: {n} is not defined on every row")
        out[f] = m
    F.OOF[MEMBER] = PUBLISHED_OOF
    return out


def mixed_anchor(data: dict, a5: dict) -> dict:
    """v10's own anchor: blend-K off the NM tail, blend-5 on it."""
    out = {}
    for g in FOLDS:
        src = pd.concat([data[h] for h in FOLDS if h != g], ignore_index=True)
        W5 = R.fit_hybrid(src, NAMES5)
        WK = R.fit_hybrid(src, list(NAMESK2))
        aa5 = R.apply_hybrid(data[g], NAMES5, W5)
        aaK = R.apply_hybrid(data[g], list(NAMESK2), WK)
        _, late = R._split(data[g], list(NAMESK2))
        out[g] = np.where(late, aa5, aaK)
    return out


def rotate(data, anchor, blk, noac_ref):
    out = {}
    for f in FOLDS:
        ap = data[f]
        fit = pd.concat([data[g] for g in FOLDS if g != f], ignore_index=True)
        a_fit = np.concatenate([anchor[g] for g in FOLDS if g != f])
        xf = pd.concat([blk[g] for g in FOLDS if g != f], ignore_index=True)
        bag = np.maximum(np.mean([SW.arm(fit, ap, s, a_fit, anchor[f], NAMESK2, xf, blk[f])
                                  for s in SEEDS], axis=0), B.FLOOR)
        out[f] = np.where(ap["regime"].to_numpy() == "noac", noac_ref[f], bag)
    return out


def serve(data: dict, blk: dict, ref: dict, v7: dict, apt: dict) -> dict:
    """One arm's member frames all the way to the served fold vector."""
    a5 = C.anchors(data, NAMES5)
    ref5 = {f: np.maximum(a5[f], B.FLOOR) for f in FOLDS}
    anc = mixed_anchor(data, a5)
    p = C.align(rotate(data, anc, blk, ref5), data, ref)
    pe = C.eham_mask(p, ref)[0]
    keep = MASK["keep"]
    return {f: np.where(np.isin(apt[f], keep), pe[f], v7[f]) for f in FOLDS}


def price(y, base, arm, P, name, base_name) -> dict:
    """The contract's alignment quantities for one arm against one baseline."""
    st_all, st_cl, parts, per_fold = {}, {}, {}, {}
    for f in FOLDS:
        pr = A.per_row(y[f], base[f], arm[f])
        ones = np.ones(len(pr["d"]), bool)
        st_all[f] = A.fold_stats(pr["r"], pr["d"], ones)
        st_cl[f] = A.fold_stats(pr["r"], pr["d"], P[f]["clean"])
        e10 = pr["r"] ** 2
        ec = (pr["r"] - pr["d"]) ** 2
        parts[f] = (P[f]["blk1"], e10, ec)
        per_fold[f] = A.boot_fold(P[f]["blk1"], e10, ec)
    b = A.board_gain(st_all)
    bc = A.board_gain(st_cl)
    ci = A.boot_board(parts, 2000)
    ci3 = A.boot_board({f: (P[f]["blk3"], parts[f][1], parts[f][2]) for f in FOLDS}, 2000)
    n = {f: st_all[f]["n"] for f in FOLDS}
    loo = A.scale_loo({f: st_all[f]["A"] for f in FOLDS},
                      {f: st_all[f]["B"] for f in FOLDS}, n)
    g_loo = A.board_gain({f: {"mse_v10": st_all[f]["mse_v10"],
                              "gain_mse_full": loo[f]["gain_mse"],
                              "gain_mse_fitted": loo[f]["gain_mse"]} for f in FOLDS})
    return {"arm": name, "base": base_name,
            "gain_board_allrow": b["gain_board"], "ci_lo": ci[0], "ci_hi": ci[1],
            "ci3_lo": ci3[0], "ci3_hi": ci3[1],
            "gain_board_clean": bc["gain_board"],
            "gain_board_fitted": b["gain_board_fitted"],
            "gain_board_loo_amp": g_loo["gain_board"],
            **{f"amp_loo_{f}": loo[f]["s"] for f in FOLDS},
            **{f"gain_rmse_{f}": st_all[f]["gain_rmse"] for f in FOLDS},
            **{f"ci_{f}_lo": per_fold[f][0] for f in FOLDS},
            **{f"ci_{f}_hi": per_fold[f][1] for f in FOLDS},
            **{f"A_{f}": st_all[f]["A"] for f in FOLDS},
            **{f"B_{f}": st_all[f]["B"] for f in FOLDS},
            **{f"scale_{f}": st_all[f]["scale"] for f in FOLDS},
            **{f"n_changed_{f}": st_all[f]["n_changed"] for f in FOLDS},
            **{f"max_abs_d_{f}": st_all[f]["max_abs_d"] for f in FOLDS}}


def slices(y, base, arm, ref, band, name, base_name) -> list:
    """Where the paired SSE change lands: by regime, by airport, and inside the
    `noac` rows by whether the tail member's FIXED high-band table wrote them."""
    rows = []
    for f in FOLDS:
        pr = A.per_row(y[f], base[f], arm[f])
        dsse = pr["dsse"]
        d = pr["d"]
        reg = ref[f]["regime"].to_numpy()
        cut = np.where(reg != "noac", "ordinary",
                       np.where(band[f], "noac band-table", "noac learned"))
        for key, vals in (("regime", reg), ("airport", ref[f]["airport"].astype(str).to_numpy()),
                          ("population", cut)):
            g = pd.DataFrame({"k": vals, "dsse": dsse, "d": d})
            agg = g.groupby("k").agg(n=("d", "size"),
                                     n_changed=("d", lambda x: int((x != 0).sum())),
                                     dsse=("dsse", "sum"),
                                     max_abs_d=("d", lambda x: float(np.abs(x).max()))).reset_index()
            for _, a in agg.iterrows():
                rows.append({"arm": name, "base": base_name, "fold": f, "kind": key,
                             "key": a["k"], "n": int(a["n"]),
                             "n_changed": int(a["n_changed"]), "dsse": float(a["dsse"]),
                             "max_abs_d": float(a["max_abs_d"])})
    return rows


def main() -> None:
    global MEMBER, PUBLISHED_OOF, ARM_OOF
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="none,chain,derived")
    ap.add_argument("--member", default=MEMBER,
                    help="the stack-5 member whose slot the arm fills")
    ap.add_argument("--oof", default=ARM_OOF,
                    help="the arm's OOF path, with {arm} and {fold}")
    ap.add_argument("--tag", default="q09", help="prefix for this run's outputs")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    if a.member not in NAMESK2:
        raise SystemExit(f"{a.member} is not one of v10's members: {list(NAMESK2)}")
    MEMBER, ARM_OOF = a.member, a.oof
    PUBLISHED_OOF = F.OOF[MEMBER]
    out_tag = a.tag          # NOT `tag`: the arm loop below binds that name
    print(f"[{out_tag}] member {MEMBER}  arms {arms}  oof {ARM_OOF}", flush=True)
    t0 = time.time()

    ref = Sc.reference()
    y = {f: ref[f]["y_true"].to_numpy("float64") for f in FOLDS}
    ids = {f: ref[f]["MVT_ID_mvt"].to_numpy("float64") for f in FOLDS}
    apt = {f: ref[f]["airport"].astype(str).to_numpy() for f in FOLDS}
    v7 = C.eham_mask({f: ref[f]["y_pred_v6"].to_numpy("float64") for f in FOLDS}, ref)[0]

    # the tail member's fixed high-band rows: `noac` and mvt_sched >= 8 h.
    v2 = pd.concat([pd.read_parquet(ROOT / "artifacts/exp-008/frames" / f"v2-{m}.parquet",
                                    columns=["MVT_ID_mvt", "d_takeoff_sched"])
                    for m in D.MONTHS], ignore_index=True)
    ms = dict(zip(v2["MVT_ID_mvt"].to_numpy("float64"),
                  v2["d_takeoff_sched"].to_numpy("float64")))
    band = {f: np.array([ms.get(i, np.nan) for i in ids[f]], "float64") >= BAND_MIN
            for f in FOLDS}

    data0 = frames_for(None)
    a5 = C.anchors(data0, NAMES5)
    blk = {f: SW.block_for(f, SW.panel_of(data0[f]), {"blocks": [C.BLOCK], "win": C.WIN})
           for f in FOLDS}
    # gate A is on the frame's own rows, not on the reference panel's
    for f in FOLDS:
        got = float(np.sqrt(np.mean((data0[f]["y_true"].to_numpy("float64") - a5[f]) ** 2)))
        off = got - S.PUBLISHED[f]
        if abs(off) > GATE:
            raise SystemExit(f"gate A (blend-5) fails on {f}: {off:+.4f}")
    print(f"[{out_tag}] gate A: blend-5 reproduces on all three folds", flush=True)

    vecs, recs = {}, []
    p10 = serve(data0, blk, ref, v7, apt)
    clean10 = {f: D.rmse_clean(y[f], p10[f], ids[f], f) for f in FOLDS}
    offF = {f: round(clean10[f] - PUB_V10[f], 4) for f in FOLDS}
    if max(abs(v) for v in offF.values()) > GATE:
        raise SystemExit(f"gate F (v10 replay) fails: {offF}")
    print(f"[{out_tag}] gate F: v10 replay {offF} on the published triple", flush=True)
    vecs["v10 replay"] = p10
    recs.append({"arm": "v10 replay", "member": "published", "variant": "served",
                 "clean": {f: round(clean10[f], 4) for f in FOLDS},
                 "clean_board": round(Sc.P.board(clean10), 4),
                 "gate_offset": offF})

    noac = {f: (ref[f]["regime"].to_numpy() == "noac") for f in FOLDS}
    for arm in arms:
        t = time.time()
        d_arm = frames_for(arm)
        p = serve(d_arm, blk, ref, v7, apt)
        pd11 = {f: np.where(noac[f], p10[f], p[f]) for f in FOLDS}
        for tag, vec in ((f"{arm}", p), (f"{arm} d11-noac", pd11)):
            cl = {f: D.rmse_clean(y[f], vec[f], ids[f], f) for f in FOLDS}
            vecs[tag] = vec
            recs.append({"arm": tag, "member": arm,
                         "variant": "d11" if "d11" in tag else "served",
                         "clean": {f: round(cl[f], 4) for f in FOLDS},
                         "clean_board": round(Sc.P.board(cl), 4),
                         "noac_shift_rms": {f: float(np.sqrt(np.mean(
                             (vec[f][noac[f]] - p10[f][noac[f]]) ** 2))) for f in FOLDS},
                         "noac_n_moved": {f: int((vec[f][noac[f]] != p10[f][noac[f]]).sum())
                                          for f in FOLDS},
                         "wall_s": round(time.time() - t, 1)})
        print(f"[{arm:8s}] " + "  ".join(f"{f} {cl[f]:9.4f}" for f in FOLDS)
              + f"  board {recs[-1]['clean_board']:9.4f}  {time.time()-t:.0f}s", flush=True)

    P = {f: {"blk1": A.blocks(A.day_of(ref[f]["t"].to_numpy("float64")), 1),
             "blk3": A.blocks(A.day_of(ref[f]["t"].to_numpy("float64")), 3),
             "clean": ~ref[f]["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()}
         for f in FOLDS}

    rows, sl = [], []
    bases = {"v10": p10}
    if "none" in arms:
        bases["none"] = vecs["none"]
        bases["none d11-noac"] = vecs["none d11-noac"]
    for bname, bvec in bases.items():
        for name, vec in vecs.items():
            if name == "v10 replay" and bname == "v10":
                continue
            rows.append(price(y, bvec, vec, P, name, bname))
            sl.extend(slices(y, bvec, vec, ref, band, name, bname))

    (EXP / "analysis").mkdir(parents=True, exist_ok=True)
    T = pd.DataFrame(rows)
    T.to_csv(EXP / "analysis" / f"{out_tag}-table.csv", index=False)
    pd.DataFrame(sl).to_csv(EXP / "analysis" / f"{out_tag}-slices.csv", index=False)
    (EXP / "analysis" / f"{out_tag}-arms.json").write_text(json.dumps(
        {"arms": recs, "member": MEMBER, "names": list(NAMESK2), "mask": MASK,
         "seeds": list(SEEDS), "rounds": S.ROUNDS, "block": C.BLOCK, "win": C.WIN,
         "threads": THREADS, "published_v10": PUB_V10,
         "wall_s": round(time.time() - t0, 1)}, indent=2))
    (EXP / "packet").mkdir(exist_ok=True)
    for f in FOLDS:
        out = pd.DataFrame({"MVT_ID_mvt": ids[f], "y_true": y[f],
                            "airport": apt[f], "regime": ref[f]["regime"].to_numpy(),
                            "band": band[f],
                            "day": A.day_of(ref[f]["t"].to_numpy("float64"))})
        for n, v in vecs.items():
            out[n] = v[f]
        out.to_parquet(EXP / "packet" / f"{out_tag}-{f}.parquet", index=False, compression="zstd")

    pd.set_option("display.width", 250)
    if T.empty:
        print(f"\n[{out_tag}] gates only: no arm was asked for")
        return
    print("\n=== every arm against v10 and against the `none` arm ===")
    cols = ["arm", "base", "gain_board_allrow", "ci_lo", "ci_hi", "gain_board_clean",
            "gain_board_loo_amp", "gain_rmse_july_loo", "gain_rmse_december",
            "gain_rmse_january"]
    print(T[cols].round(4).to_string(index=False))
    print(f"\n[{out_tag}] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
