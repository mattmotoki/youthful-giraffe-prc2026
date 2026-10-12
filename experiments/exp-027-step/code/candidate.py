"""The K2 candidate: v7's recipe with the K2 member in F1-fresh's slot.

The desk (BOARD READING 7): "the candidate is stack-5's stacker refit with the
K2 member's OOF in place of F1-fresh's, with the window block and EHAM mask
served as v7 does".

**The reading taken, and the readings rejected.**  "In place of F1-fresh's"
names a member column.  That column feeds two things: the blend-5 hybrid NNLS
anchor the stacker sits on, and the stacker's own member features.  Three
readings were possible.

1. *Anchor held, member swapped in the stacker only* (`k2-hold`).  The anchor
   stays the blend-5 that reproduces `blend-config-5.json` on all three folds,
   so exp-026's baseline gate still runs and the `noac` rows keep exactly the
   bytes v6 and v7 served.  **Taken as the shipping reading.**
2. *Anchor refitted with K2 as well* (`k2-refit`).  Measured here beside 1,
   because it is the reading that uses the new member everywhere.  It is not
   taken unless it wins, because it moves the anchor and therefore would move
   the 388 `noac` ranking rows, which score-feedback rule holds.
3. *K2 added as an eighth member.*  Rejected without measurement: the desk said
   "in place of", and an eighth member is a different candidate (F1-fresh and
   K2 are the same model on the same frame with one feature block added, so
   they are near-collinear and the stacker would split weight between them).

**`noac` is held at blend-5-on-B5's anchor in every arm**, whatever anchor the
arm's stacker used.  That makes every arm's 388 ranking `noac` rows bit
identical to stack-5's, v6's and v7's -- the third kill clause in `SPEC.md`
(RMS move <= 400 s) holds by construction, at zero -- and it makes the arms
comparable on the ordinary rows, which is where the whole measurement lives.

**Nothing is reimplemented.**  The stacker is `exp-028/code/stack5w.arm`, the
window block is `stack5w.block_for`, the anchor is `exp-026/code/rule`, the
frames are `exp-026/code/frames`, and the scoring, the day-clustered intervals
and the gates are `exp-030/code/score`.  This file chooses member sets and
anchors and does the arithmetic of v7's EHAM mask; that is all.

Three gates run before any K2 number is printed, each of them a rotation this
script recomputes rather than reads:

  A  blend-5 on the seven committed members reproduces `blend-config-5.json`
     (exp-026 `stack5.PUBLISHED`) within 1e-3.
  B  the no-window stacker lands on exp-026's published stack-5 within 1e-3.
  C  the 600 s window arm lands on exp-028's published v6 within 1e-3.

    PRC_THREADS=4 python candidate.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "experiments/exp-030-window2/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import frames as F  # noqa: E402
import rule as R  # noqa: E402
import stack5 as S  # noqa: E402
import stack5w as SW  # noqa: E402
import score as Sc  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

THREADS = int(os.environ.get("PRC_THREADS", "4"))
SW.THREADS = THREADS
WIN = float(os.environ.get("PRC_WIN", "600"))
BLOCK = os.environ.get("PRC_BLOCK", "depx")
#: Names every artifact this run writes, so the identity check's vectors and
#: the measured candidate's can never overwrite one another.
TAG = os.environ.get("PRC_TAG", "k2")
FOLDS = F.FOLDS
#: The K2 member: F1-fresh's frame with `window_features.py`'s 68 columns added,
#: refitted on Kaggle on the real data (`kernels/f1-window*`, RESULTS section 15).
K2 = "K2-window"
F.OOF[K2] = os.environ.get("PRC_K2", "experiments/exp-027-step/oof/k2-window-{fold}.parquet")
#: `PRC_K2` exists for one purpose: the identity check.  Point it at F1-fresh's
#: own committed out-of-fold files and `k2-nowin` must reproduce stack-5 and
#: `k2-hold` must reproduce v6, bit for bit, because the substituted column is
#: then the column it replaces.  That is the test that the substitution
#: machinery here changes nothing but the member.
NAMES5 = S.B5
NAMESK = tuple(K2 if n == "F1-fresh" else n for n in S.B5)
if sum(a != b for a, b in zip(NAMES5, NAMESK)) != 1 or len(NAMES5) != len(NAMESK):
    raise SystemExit(f"the member sets differ in more than F1-fresh's slot: "
                     f"{NAMES5} against {NAMESK}")
PUB_STACK5 = {"july_loo": 285.2029, "december": 205.6118, "january": 219.1303}
PUB_V6 = {"july_loo": 284.7219, "december": 205.6534, "january": 218.8166}
GATE = 1e-3
#: exp-026's serving tolerance.  A rotation rebuilt in this process sums the
#: same numbers in a different order from the committed one, so a gate on the
#: vectors is written at 1e-9 s and not at 0.0.
VEC_TOL = 1e-9


def frames() -> dict:
    """One frame per fold carrying all eight member columns, on stack-5's rows."""
    out = {}
    for f in FOLDS:
        m = F.fold_frame(f, NAMES5 + (K2,))
        base = F.fold_frame(f, NAMES5)
        a = m["MVT_ID_mvt"].to_numpy("float64")
        b = base["MVT_ID_mvt"].to_numpy("float64")
        if not np.array_equal(a, b):
            raise SystemExit(f"{f}: the K2 member changes the frame: "
                             f"{len(a)} rows against {len(b)}")
        if not np.isfinite(m[K2].to_numpy("float64")).all():
            raise SystemExit(f"{f}: the K2 member is not defined on every row")
        out[f] = m
    return out


def anchors(data: dict, names) -> dict:
    """blend-5's per-regime hybrid NNLS, rotated: fold `g` fitted on the other two."""
    out = {}
    for g in FOLDS:
        src = pd.concat([data[h] for h in FOLDS if h != g], ignore_index=True)
        out[g] = R.apply_hybrid(data[g], names, R.fit_hybrid(src, names))
    return out


def rotate(data: dict, names, anchor: dict, blk: dict | None, noac_ref: dict) -> dict:
    """`stack5w`'s arm over the three folds, `noac` held at `noac_ref`."""
    out = {}
    for f in FOLDS:
        ap = data[f]
        fit = pd.concat([data[g] for g in FOLDS if g != f], ignore_index=True)
        a_fit = np.concatenate([anchor[g] for g in FOLDS if g != f])
        xf = xa = None
        if blk is not None:
            xf = pd.concat([blk[g] for g in FOLDS if g != f], ignore_index=True)
            xa = blk[f]
        bag = np.maximum(np.mean([SW.arm(fit, ap, s, a_fit, anchor[f], names, xf, xa)
                                  for s in S.SEEDS], axis=0), B.FLOOR)
        out[f] = np.where(ap["regime"].to_numpy() == "noac", noac_ref[f], bag)
    return out


def align(preds: dict, data: dict, ref: dict) -> dict:
    """Put an arm's vectors in the reference's row order, or fail loudly."""
    out = {}
    for f in FOLDS:
        want = ref[f]["MVT_ID_mvt"].to_numpy("float64")
        have = data[f]["MVT_ID_mvt"].to_numpy("float64")
        if np.array_equal(want, have):
            out[f] = np.asarray(preds[f], "float64")
            continue
        idx = pd.Index(have).get_indexer(want)
        if (idx < 0).any():
            raise SystemExit(f"{f}: {int((idx < 0).sum())} reference rows are not in the frame")
        out[f] = np.asarray(preds[f], "float64")[idx]
    return out


def eham_mask(preds: dict, ref: dict) -> tuple[dict, dict]:
    """v7's serving rule: every EHAM departure keeps stack-5's own value."""
    out, n = {}, {}
    for f in FOLDS:
        d = ref[f]
        m = d["airport"].astype(str).to_numpy() == "EHAM"
        out[f] = np.where(m, d["y_pred_stack5"].to_numpy("float64"), preds[f])
        n[f] = int(m.sum())
    return out, n


#: v7's own measured board, `docs/LEDGER.md` BOARD READING 7.  The candidate
#: here replaces v7 on the board, so v7 -- not v6 -- is the comparison the
#: prediction is made against.
V7_BOARD = 263.4623


def against_v7(rec: dict, preds: dict, ref: dict, v7: dict) -> dict:
    """The arm's gain over v7, the file the board actually holds.

    `score.py` prices every arm against v6 because exp-030's arms were cuts on
    v6.  This one replaces v7, so its prediction is v7's measured board minus
    its clean gain over v7, at face.  The intervals are the same
    calendar-day-clustered bootstrap, on the clean rows and again without noac.
    """
    per, base = {}, {}
    rec["vs_v7"], rec["ci_v7"], rec["ci_v7_no_noac"] = {}, {}, {}
    for f in FOLDS:
        d = ref[f]
        y = d["y_true"].to_numpy("float64")
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        pv = np.asarray(preds[f], "float64")
        per[f] = D.rmse_clean(y, pv, ids, f)
        base[f] = D.rmse_clean(y, v7[f], ids, f)
        rec["vs_v7"][f] = round(base[f] - per[f], 4)
        ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
        nn = ok & (d["regime"].to_numpy() != "noac")
        cl = Sc.P.clusters(d)
        rec["ci_v7"][f] = [round(x, 4) for x in
                           Sc.P.boot_delta(y[ok], pv[ok], v7[f][ok], pd.factorize(cl[ok])[0])]
        rec["ci_v7_no_noac"][f] = [round(x, 4) for x in
                                   Sc.P.boot_delta(y[nn], pv[nn], v7[f][nn],
                                                   pd.factorize(cl[nn])[0])]
    rec["clean_board_v7"] = round(Sc.P.board(base), 4)
    rec["gain_board_vs_v7"] = round(Sc.P.board(base) - Sc.P.board(per), 4)
    rec["predicted_board_vs_v7"] = round(V7_BOARD - rec["gain_board_vs_v7"], 4)
    print(f"  vs v7      " + "  ".join(f"{f} {rec['vs_v7'][f]:+9.4f}" for f in FOLDS)
          + f"   board {rec['gain_board_vs_v7']:+9.4f}")
    for f in FOLDS:
        dl, lo, hi = rec["ci_v7"][f]
        dn, lon, hin = rec["ci_v7_no_noac"][f]
        print(f"    {f:9s} vs v7 {-dl:+7.3f} [{-hi:+7.3f}, {-lo:+7.3f}]"
              f"   no-noac {-dn:+7.3f} [{-hin:+7.3f}, {-lon:+7.3f}]")
    print(f"  predicted board {rec['predicted_board_vs_v7']:.4f} "
          f"(v7 measured {V7_BOARD}, at face; the family has transferred at 1.4-1.7x)")
    return rec


def main() -> None:
    t0 = time.time()
    data = frames()
    print(f"[k2] frames in {time.time()-t0:.0f}s, {THREADS} threads, "
          f"block={BLOCK} win={WIN:.0f}s", flush=True)

    a5 = anchors(data, NAMES5)
    for f in FOLDS:
        y = data[f]["y_true"].to_numpy("float64")
        got = float(np.sqrt(np.mean((y - a5[f]) ** 2)))
        d = got - S.PUBLISHED[f]
        print(f"[gate A] blend-5 {f:9s} {got:9.4f} (published {S.PUBLISHED[f]:.4f}, "
              f"delta {d:+.4f})", flush=True)
        if abs(d) > GATE:
            raise SystemExit(f"gate A fails on {f}: {d:+.4f}")
    aK = anchors(data, NAMESK)
    for f in FOLDS:
        y = data[f]["y_true"].to_numpy("float64")
        print(f"[note ] blend-K {f:9s} "
              f"{float(np.sqrt(np.mean((y - aK[f]) ** 2))):9.4f}", flush=True)

    ref5 = {f: np.maximum(a5[f], B.FLOOR) for f in FOLDS}
    blk = {}
    for f in FOLDS:
        blk[f] = SW.block_for(f, SW.panel_of(data[f]), {"blocks": [BLOCK], "win": WIN})
    print(f"[k2] block {BLOCK} = {blk[FOLDS[0]].shape[1]} columns", flush=True)

    ref = Sc.reference()
    # v7 is v6 with EHAM served from stack-5: arithmetic on the same two gated
    # vectors, so it needs no file of its own.
    v7, _ = eham_mask({f: ref[f]["y_pred_v6"].to_numpy("float64") for f in FOLDS}, ref)
    v7c = {f: D.rmse_clean(ref[f]["y_true"], v7[f], ref[f]["MVT_ID_mvt"], f) for f in FOLDS}
    print("[k2] exp-028's committed stack-5 and v6 read back and gated; v7 rebuilt "
          + "  ".join(f"{f} {v7c[f]:.4f}" for f in FOLDS)
          + f"  board {Sc.P.board(v7c):.4f}", flush=True)

    arms = {
        "stack-5 (gate B)": (NAMES5, a5, None),
        "v6 (gate C)":      (NAMES5, a5, blk),
        "k2-nowin":         (NAMESK, a5, None),
        "k2-hold":          (NAMESK, a5, blk),
        "k2-refit":         (NAMESK, aK, blk),
    }
    raw, recs = {}, []
    for lab, (names, anc, bk) in arms.items():
        t = time.time()
        p = rotate(data, names, anc, bk, ref5)
        raw[lab] = p
        clean = {f: D.rmse_clean(data[f]["y_true"], p[f],
                                 data[f]["MVT_ID_mvt"], f) for f in FOLDS}
        print(f"[{lab:16s}] " + "  ".join(f"{f} {clean[f]:9.4f}" for f in FOLDS)
              + f"  board {F.board_estimate(clean):9.4f}  {time.time()-t:.0f}s", flush=True)
        # Sol round 3, non-blocking 2: an RMSE to 1e-3 can be equal for different
        # vectors, so each gate now also compares the vector row by row against
        # the committed one, on the reference's row order, at the 1e-9 serving
        # tolerance.  Under the identity substitution these read 0.0 for stack-5
        # and about 7e-13 for v6 -- summation order, not a different model.
        for key, pub, col, name in (("stack-5", PUB_STACK5, "y_pred_stack5", "B"),
                                    ("v6", PUB_V6, "y_pred_v6", "C")):
            if not lab.startswith(key):
                continue
            off = {f: round(clean[f] - pub[f], 4) for f in FOLDS}
            if max(abs(v) for v in off.values()) > GATE:
                raise SystemExit(f"gate {name} fails on the RMSE: {off}")
            pa = align(p, data, ref)
            md = {f: float(np.abs(pa[f] - ref[f][col].to_numpy("float64")).max())
                  for f in FOLDS}
            if max(md.values()) > VEC_TOL:
                raise SystemExit(f"gate {name} fails on the vector: {md}")
            print(f"[gate {name}] {key} reproduces: RMSE {off}, "
                  f"max abs row difference {max(md.values()):.3g}", flush=True)

    for lab in ("k2-nowin", "k2-hold", "k2-refit"):
        p = align(raw[lab], data, ref)
        rec = Sc.score(f"{lab}", p, ref,
                       note=f"{lab}: members {'/'.join(NAMESK)}; "
                            f"anchor {'blend-K' if lab == 'k2-refit' else 'blend-5'}; "
                            f"block {BLOCK if lab != 'k2-nowin' else 'none'}; "
                            f"noac held at blend-5's anchor")
        Sc.show(rec)
        recs.append(rec)
        pe, n = eham_mask(p, ref)
        rece = Sc.score(f"{lab}-eham-out", pe, ref,
                        note=f"{lab} with EHAM served from stack-5, v7's mask; "
                             + ", ".join(f"{f} {n[f]}" for f in FOLDS) + " rows taken back")
        rece["airport_excluded"] = "EHAM"
        rece["n_rows_taken_from_stack5"] = n
        Sc.show(rece)
        against_v7(rece, pe, ref, v7)
        recs.append(rece)
        for f in FOLDS:
            pd.DataFrame({"MVT_ID_mvt": ref[f]["MVT_ID_mvt"].to_numpy("float64"),
                          "fold": f, "y_true": ref[f]["y_true"].to_numpy("float64"),
                          "y_pred": pe[f], "y_pred_nomask": p[f],
                          "y_pred_stack5": ref[f]["y_pred_stack5"].to_numpy("float64"),
                          "y_pred_v6": ref[f]["y_pred_v6"].to_numpy("float64")}).to_parquet(
                EXP / "oof" / f"exp-027-{TAG}-{lab}-{f}.parquet", index=False)

    out = {"members": list(NAMESK), "block": BLOCK, "win": WIN, "threads": THREADS,
           "seeds": list(S.SEEDS), "rounds": S.ROUNDS,
           "lightgbm": __import__("lightgbm").__version__,
           "gates": {"A_blend5": "pass", "B_stack5": "pass", "C_v6": "pass"},
           "arms": recs, "wall_s": round(time.time() - t0, 1)}
    (EXP / "analysis" / f"candidate-{TAG}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[k2] {out['wall_s']:.0f}s wall -> analysis/candidate-{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
