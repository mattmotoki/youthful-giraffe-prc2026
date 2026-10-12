"""The K3 candidate: K2's candidate with the K3 member in `T-tailv4`'s slot.

The assignment: *"K2's stacker (the exp-027 harness, which reproduces stack-5,
v6, v7 and K2 bit for bit) with the K3 member in its own slot, K2's member in
F1-fresh's slot, the window block and the EHAM mask served as v7."*

**The reading taken, and the readings rejected.** "In its own slot" names the
`T-tailv4` member column, which feeds two things: blend-5's per-regime hybrid
NNLS anchor and the stacker's own member features. This file takes the
**anchor-held** reading (`k3-hold`), which is exactly the reading K2 shipped on
(`exp-027` SPEC addendum, decision 1) and which score-feedback rule requires: the anchor stays
the blend-5 that reproduces `blend-config-5.json` on all three folds, so
exp-026's baseline gate keeps running and the 388 served `noac` rows keep the
bytes v6, v7 and v9 served. The anchor-refit reading is **not** measured here:
`k2-refit` already failed a pre-registered tail clause on this base
(`exp-027` section 16d) and its repair is a standing MAYBE that a K3 arm would
confound rather than settle.

`noac` is held at blend-5-on-B5's anchor in every arm, as in `exp-027`.

**Nothing is reimplemented.** The stacker is `exp-028/code/stack5w.arm`, the
block `stack5w.block_for`, the anchor `exp-026/code/rule`, the frames
`exp-026/code/frames`, the scoring and the day-clustered intervals
`exp-030/code/score`. This file substitutes TWO member columns instead of one
and does the arithmetic of v7's EHAM mask.

Four gates run before any K3 number, each a rotation recomputed here:

  A  blend-5 on the seven committed members reproduces `blend-config-5.json`
  B  the no-window stacker lands on exp-026's published stack-5
  C  the 600 s window arm lands on exp-028's published v6
  D  with K3 identity-pointed at `T-tailv4`'s own committed OOF, the K2
     candidate `k2-hold-eham-out` reproduces exp-027's 257.1653

and with BOTH members identity-pointed (`PRC_K2` and `PRC_K3` at the committed
files) every arm collapses onto stack-5 / v6 / v7, which is the test that the
two-slot substitution machinery changes nothing but the members.

    PRC_THREADS=3 python code/candidate.py
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

THREADS = int(os.environ.get("PRC_THREADS", "3"))
SW.THREADS = THREADS
WIN = float(os.environ.get("PRC_WIN", "600"))
BLOCK = os.environ.get("PRC_BLOCK", "depx")
TAG = os.environ.get("PRC_TAG", "k3")
FOLDS = F.FOLDS

#: The two substituted members. K2 is exp-027's; K3 is this experiment's tail
#: member, `T-tailv4` refitted with the same 68-column window block in its own
#: feature frame.
K2 = "K2-window"
K3 = "K3-tail"
#: The repair of `code/gate_member.py`: the window arm where the committed
#: member's own prediction is at or below 3600 s, the committed member above it.
#: `SPEC.md`'s addendum of 22:45 UTC registers it as the shipping member before
#: any rotation number below existed, because the ungated member fails the
#: assignment's 1 s tail clause and this one passes both clauses.
K3G = "K3-tail-gated"
F.OOF[K2] = os.environ.get("PRC_K2", "experiments/exp-027-step/oof/k2-window-{fold}.parquet")
F.OOF[K3] = os.environ.get("PRC_K3", "experiments/exp-032-k3/oof/k3-tail-window-{fold}.parquet")
F.OOF[K3G] = os.environ.get("PRC_K3G", "experiments/exp-032-k3/oof/k3-tail-gated-{fold}.parquet")

NAMES5 = S.B5


def _sub(**slots) -> tuple:
    """B5 with the named members put in the named slots."""
    return tuple(slots.get(n, n) for n in NAMES5)


NAMESK2 = _sub(**{"F1-fresh": K2})
NAMESK3 = _sub(**{"T-tailv4": K3})
NAMESK3G = _sub(**{"T-tailv4": K3G})
NAMESK23 = _sub(**{"F1-fresh": K2, "T-tailv4": K3})
NAMESK23G = _sub(**{"F1-fresh": K2, "T-tailv4": K3G})
for got, want in ((NAMESK2, 1), (NAMESK3, 1), (NAMESK3G, 1),
                  (NAMESK23, 2), (NAMESK23G, 2)):
    if sum(a != b for a, b in zip(NAMES5, got)) != want or len(got) != len(NAMES5):
        raise SystemExit(f"member set {got} does not differ from {NAMES5} in exactly {want} slot(s)")


def _add(base: tuple, extra: str) -> tuple:
    """`base` with `extra` inserted as an EIGHTH member, before `N1-nmtail`.

    The desk asked for the additive repair: keep `T-tailv4` in its own slot and
    let the K3 member join as a new one, rather than replace it.

    `N1-nmtail` goes last by CONVENTION, and the convention is kept here, but
    Sol round 1 (blocking 7) is right that the reason usually given for it is
    stale: `exp-026-rebuild/code/rule.py` removes the specialist BY NAME
    (`gen = [n for n in names if n != SPECIALIST]`, `rule.py:49`), not by
    slicing `[:, :k-1]`, and `stack5.member_features` builds its columns from
    the names as well. Position therefore does not affect correctness in any
    code this arm touches, and the anchor never sees eight names at all
    (`anchors(data, NAMES5)`). The order is held anyway so that every member
    tuple in the repository reads the same way.
    """
    if base[-1] != "N1-nmtail":
        raise SystemExit(f"{base} does not end in N1-nmtail")
    got = base[:-1] + (extra, base[-1])
    if len(got) != len(base) + 1 or got[-1] != "N1-nmtail" or len(set(got)) != len(got):
        raise SystemExit(f"additive member set {got} is malformed")
    return got


#: `k3-add-hold` / `k3g-add-hold`: K2's seven PLUS the K3 member, eight in all.
#: This is not a pure superset of `k2-hold`: `stack5.member_features` also emits
#: `mem_mean`, `mem_std`, `mem_range` and `n_defined` over the member columns, so
#: an eighth column moves those four aggregates as well. The same was true when
#: K2 itself entered, so the comparison against `k2-hold` stays like-for-like.
NAMESK2A = _add(NAMESK2, K3)
NAMESK2AG = _add(NAMESK2, K3G)

PUB_STACK5 = {"july_loo": 285.2029, "december": 205.6118, "january": 219.1303}
PUB_V6 = {"july_loo": 284.7219, "december": 205.6534, "january": 218.8166}
#: exp-027 section 16d, the shipped v9: `k2-hold-eham-out`.
#: `assignments/k3.md`: "Gate: clean gain >= 0.10 s over K2 (257.1653) with no
#: fold worse than -0.10 s and day-clustered intervals stated."
GAIN_BAR = 0.10
FOLD_FLOOR = -0.10
PUB_K2 = {"july_loo": 284.3195, "december": 205.2156, "january": 218.2582}
PUB_K2_BOARD = 257.1653
#: exp-028's v7, rebuilt: what gate D's arm must read instead when both slots are
#: identity-pointed, because `k2-hold-eham-out` is then v7 and not K2.
PUB_V7 = {"july_loo": 284.6251, "december": 205.5677, "january": 218.5412}
PUB_V7_BOARD = 257.4598
#: `PRC_IDENTITY=1` says the substituted columns are the committed members, so
#: every arm here must collapse onto stack-5 / v6 / v7.  It changes what the
#: gates EXPECT and nothing else; no arm, no anchor and no block moves.
IDENTITY = os.environ.get("PRC_IDENTITY") == "1"
GATE = 1e-3
VEC_TOL = 1e-9
V7_BOARD = 263.4623


def frames() -> dict:
    """One frame per fold carrying all nine member columns, on stack-5's rows."""
    out = {}
    for f in FOLDS:
        m = F.fold_frame(f, NAMES5 + (K2, K3, K3G))
        base = F.fold_frame(f, NAMES5)
        a = m["MVT_ID_mvt"].to_numpy("float64")
        b = base["MVT_ID_mvt"].to_numpy("float64")
        if not np.array_equal(a, b):
            raise SystemExit(f"{f}: a substituted member changes the frame: "
                             f"{len(a)} rows against {len(b)}")
        for n in (K2, K3, K3G):
            if not np.isfinite(m[n].to_numpy("float64")).all():
                raise SystemExit(f"{f}: {n} is not defined on every row")
        out[f] = m
    return out


def anchors(data: dict, names) -> dict:
    out = {}
    for g in FOLDS:
        src = pd.concat([data[h] for h in FOLDS if h != g], ignore_index=True)
        out[g] = R.apply_hybrid(data[g], names, R.fit_hybrid(src, names))
    return out


def rotate(data: dict, names, anchor: dict, blk: dict | None, noac_ref: dict) -> dict:
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
    out, n = {}, {}
    for f in FOLDS:
        d = ref[f]
        m = d["airport"].astype(str).to_numpy() == "EHAM"
        out[f] = np.where(m, d["y_pred_stack5"].to_numpy("float64"), preds[f])
        n[f] = int(m.sum())
    return out, n


def against(rec: dict, preds: dict, ref: dict, base: dict, label: str,
            base_board: float | None = None) -> dict:
    """The arm's gain over a reference vector, with day-clustered intervals."""
    per, bas = {}, {}
    kg, kc, kn = f"vs_{label}", f"ci_{label}", f"ci_{label}_no_noac"
    rec[kg], rec[kc], rec[kn] = {}, {}, {}
    for f in FOLDS:
        d = ref[f]
        y = d["y_true"].to_numpy("float64")
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        pv = np.asarray(preds[f], "float64")
        per[f] = D.rmse_clean(y, pv, ids, f)
        bas[f] = D.rmse_clean(y, base[f], ids, f)
        rec[kg][f] = round(bas[f] - per[f], 4)
        ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
        nn = ok & (d["regime"].to_numpy() != "noac")
        cl = Sc.P.clusters(d)
        # Sol round 1, blocking 5: `boot_delta` returns (arm - base), where
        # POSITIVE is worse, while `rec[kg]` above stores (base - arm), where
        # positive is BETTER.  Two opposite conventions in one record is a trap
        # for later code, so the intervals are negated here into the gain
        # convention -- the one the log prints and RESULTS quotes -- and the
        # record says so in `interval_convention`.
        neg = lambda t: [round(-x, 4) for x in (t[0], t[2], t[1])]
        rec[kc][f] = neg(Sc.P.boot_delta(y[ok], pv[ok], base[f][ok],
                                         pd.factorize(cl[ok])[0]))
        rec[kn][f] = neg(Sc.P.boot_delta(y[nn], pv[nn], base[f][nn],
                                         pd.factorize(cl[nn])[0]))
    rec["interval_convention"] = ("gain: positive is BETTER than the reference, "
                                  "[point, lo, hi], day-clustered 95 %")
    rec[f"clean_board_{label}"] = round(Sc.P.board(bas), 4)
    rec[f"gain_board_vs_{label}"] = round(Sc.P.board(bas) - Sc.P.board(per), 4)
    if base_board is not None:
        rec[f"predicted_board_vs_{label}"] = round(base_board - rec[f"gain_board_vs_{label}"], 4)
    print(f"  vs {label:9s} " + "  ".join(f"{f} {rec[kg][f]:+9.4f}" for f in FOLDS)
          + f"   board {rec[f'gain_board_vs_{label}']:+9.4f}")
    for f in FOLDS:
        dl, lo, hi = rec[kc][f]
        dn, lon, hin = rec[kn][f]
        print(f"    {f:9s} vs {label} {dl:+7.3f} [{lo:+7.3f}, {hi:+7.3f}]"
              f"   no-noac {dn:+7.3f} [{lon:+7.3f}, {hin:+7.3f}]")
    if base_board is not None:
        print(f"  predicted board {rec[f'predicted_board_vs_{label}']:.4f} "
              f"({label} measured {base_board}, at face)")
    return rec


def tail_delta(preds: dict, base: dict, ref: dict) -> dict:
    """`rmse_clean` on `y > 3600` against a reference; positive is better."""
    out = {}
    for f in FOLDS:
        d = ref[f]
        y = d["y_true"].to_numpy("float64")
        ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
        m = ok & (y > 3600.0)
        r = lambda p: float(np.sqrt(np.mean((y[m] - np.asarray(p, "float64")[m]) ** 2)))
        out[f] = {"n": int(m.sum()), "base": round(r(base[f]), 4),
                  "arm": round(r(preds[f]), 4), "gain": round(r(base[f]) - r(preds[f]), 4)}
    return out


def main() -> None:
    t0 = time.time()
    data = frames()
    print(f"[k3] frames in {time.time()-t0:.0f}s, {THREADS} threads, "
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

    ref5 = {f: np.maximum(a5[f], B.FLOOR) for f in FOLDS}
    blk = {f: SW.block_for(f, SW.panel_of(data[f]), {"blocks": [BLOCK], "win": WIN})
           for f in FOLDS}
    print(f"[k3] block {BLOCK} = {blk[FOLDS[0]].shape[1]} columns", flush=True)

    ref = Sc.reference()
    v7, _ = eham_mask({f: ref[f]["y_pred_v6"].to_numpy("float64") for f in FOLDS}, ref)
    v7c = {f: D.rmse_clean(ref[f]["y_true"], v7[f], ref[f]["MVT_ID_mvt"], f) for f in FOLDS}
    print("[k3] exp-028's committed stack-5 and v6 read back; v7 rebuilt "
          + "  ".join(f"{f} {v7c[f]:.4f}" for f in FOLDS)
          + f"  board {Sc.P.board(v7c):.4f}", flush=True)

    arms = {
        "stack-5 (gate B)": (NAMES5, None),
        "v6 (gate C)":      (NAMES5, blk),
        "k2-hold":          (NAMESK2, blk),
        "k3-only-hold":     (NAMESK3, blk),
        "k3g-only-hold":    (NAMESK3G, blk),
        "k3-hold":          (NAMESK23, blk),
        "k3g-hold":         (NAMESK23G, blk),
        "k3-add-hold":      (NAMESK2A, blk),
        "k3g-add-hold":     (NAMESK2AG, blk),
    }
    #: Which member set each K3 arm carries, for the record it writes.
    MEMBERS = {"k3-only-hold": NAMESK3, "k3g-only-hold": NAMESK3G,
               "k3-hold": NAMESK23, "k3g-hold": NAMESK23G,
               "k3-add-hold": NAMESK2A, "k3g-add-hold": NAMESK2AG}
    #: Gate E asks a substituted arm to reproduce v7 when the substituted column
    #: IS the column it replaces. An ADDITIVE arm cannot: under identity it
    #: carries `T-tailv4` twice, which legitimately moves `mem_mean`, `mem_std`,
    #: `mem_range` and `n_defined`. Gate D still covers its baseline.
    NO_GATE_E = {"k3-add-hold", "k3g-add-hold"}
    raw, recs = {}, []
    for lab, (names, bk) in arms.items():
        t = time.time()
        p = rotate(data, names, a5, bk, ref5)
        raw[lab] = p
        clean = {f: D.rmse_clean(data[f]["y_true"], p[f],
                                 data[f]["MVT_ID_mvt"], f) for f in FOLDS}
        print(f"[{lab:16s}] " + "  ".join(f"{f} {clean[f]:9.4f}" for f in FOLDS)
              + f"  board {F.board_estimate(clean):9.4f}  {time.time()-t:.0f}s", flush=True)
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

    # gate D: the K2 candidate this arm has to beat is rebuilt here, not quoted.
    k2p = align(raw["k2-hold"], data, ref)
    k2e, _ = eham_mask(k2p, ref)
    k2c = {f: D.rmse_clean(ref[f]["y_true"], k2e[f], ref[f]["MVT_ID_mvt"], f) for f in FOLDS}
    want, want_b, wname = ((PUB_V7, PUB_V7_BOARD, "v7") if IDENTITY
                           else (PUB_K2, PUB_K2_BOARD, "K2 (v9)"))
    offd = {f: round(k2c[f] - want[f], 4) for f in FOLDS}
    board_k2 = Sc.P.board(k2c)
    print(f"[gate D] k2-hold-eham-out " + "  ".join(f"{f} {k2c[f]:9.4f}" for f in FOLDS)
          + f"  board {board_k2:.4f} (published {wname} {want_b}, delta "
          f"{board_k2 - want_b:+.4f}); folds {offd}", flush=True)
    if max(abs(v) for v in offd.values()) > GATE:
        raise SystemExit(f"gate D fails against {wname}: {offd}")
    if IDENTITY:
        # Sol round 1 blocking 4, answered in round 2: gate D above compares
        # ROUNDED RMSE, which a common row-level drift in the shared harness
        # could survive. Under identity `k2-hold-eham-out` must BE the committed
        # v7 vector, so it is compared to it row by row -- and `v7` is built
        # from exp-028's committed columns through `Sc.reference()`, not from
        # anything this run computed, so the reference is external.
        mdv = {f: float(np.abs(k2e[f] - v7[f]).max()) for f in FOLDS}
        print(f"[gate D+] k2-hold-eham-out against the COMMITTED v7, row by row: "
              f"max abs difference {max(mdv.values()):.3g}", flush=True)
        if max(mdv.values()) > VEC_TOL:
            raise SystemExit(f"gate D+ fails: k2-hold is not v7 row-wise: {mdv}")
        # every K3 arm must be v7 too: the substituted column IS the column it
        # replaces, so the two-slot machinery can have changed nothing.
        for lab in MEMBERS:
            if lab in NO_GATE_E:
                print(f"[gate E] {lab:14s} SKIPPED: an additive arm carries the "
                      f"identity-pointed member twice, so the member aggregates "
                      f"move and v7 is not the right target", flush=True)
                continue
            pe, _ = eham_mask(align(raw[lab], data, ref), ref)
            c = {f: D.rmse_clean(ref[f]["y_true"], pe[f], ref[f]["MVT_ID_mvt"], f)
                 for f in FOLDS}
            # Sol round 1, blocking 4: this compared each arm with `k2e`, a
            # vector the SAME run produced, so a drift shared by both could
            # pass. The reference is now the committed v7 that `Sc.reference()`
            # reads off exp-028's published columns -- external to this process.
            md = {f: float(np.abs(pe[f] - v7[f]).max()) for f in FOLDS}
            print(f"[gate E] {lab:14s} " + "  ".join(f"{f} {c[f]:9.4f}" for f in FOLDS)
                  + f"  board {Sc.P.board(c):.4f}  max abs row difference against "
                  f"the COMMITTED v7 {max(md.values()):.3g}", flush=True)
            if max(md.values()) > VEC_TOL:
                raise SystemExit(f"gate E fails on {lab}: {md}")

    for lab in MEMBERS:
        p = align(raw[lab], data, ref)
        pe, n = eham_mask(p, ref)
        rece = Sc.score(f"{lab}-eham-out", pe, ref,
                        note=f"{lab} with EHAM served from stack-5, v7's mask; "
                             f"members {'/'.join(MEMBERS[lab])}; "
                             f"anchor blend-5; block {BLOCK}; noac held at blend-5's anchor; "
                             + ", ".join(f"{f} {n[f]}" for f in FOLDS) + " EHAM rows taken back")
        rece["airport_excluded"] = "EHAM"
        rece["n_rows_taken_from_stack5"] = n
        Sc.show(rece)
        against(rece, pe, ref, v7, "v7", V7_BOARD)
        against(rece, pe, ref, k2e, "k2", None)
        rece["tail_vs_k2"] = tail_delta(pe, k2e, ref)
        rece["tail_vs_v7"] = tail_delta(pe, v7, ref)
        # Sol round 1, blocking 3: `Sc.score`'s inherited `gate` field is the
        # v6 gate this harness was born with, and it read `true` on an arm that
        # FAILS the assignment.  `assignments/k3.md` gates on K2: clean gain at
        # least +0.10 s of board over K2, and no fold worse than -0.10 s.  It is
        # computed here, named for what it is, and `serve.py` refuses without it.
        rece["gate_v6_inherited"] = rece.pop("gate", None)
        gk = rece["gain_board_vs_k2"] >= GAIN_BAR
        gf = min(rece["vs_k2"].values()) >= FOLD_FLOOR
        rece["gate_k2"] = {"bar_board": GAIN_BAR, "floor_fold": FOLD_FLOOR,
                           "gain_board": rece["gain_board_vs_k2"],
                           "worst_fold": round(min(rece["vs_k2"].values()), 4),
                           "gain_passes": bool(gk), "folds_pass": bool(gf),
                           "PASS": bool(gk and gf)}
        print(f"  ASSIGNMENT GATE vs K2: gain {rece['gain_board_vs_k2']:+.4f} "
              f"(bar {GAIN_BAR:+.2f}) {'ok' if gk else 'FAIL'}, worst fold "
              f"{min(rece['vs_k2'].values()):+.4f} (floor {FOLD_FLOOR:+.2f}) "
              f"{'ok' if gf else 'FAIL'}  ->  "
              f"{'PASS' if gk and gf else 'FAIL'}", flush=True)
        print("  tail y>3600 vs k2 " + "  ".join(
            f"{f} {rece['tail_vs_k2'][f]['gain']:+7.2f}" for f in FOLDS))
        recs.append(rece)
        # Only the two shipping-shaped arms keep their vectors: the `-only-`
        # arms are diagnostics that separate K3's contribution from K2's, and
        # the identity run's vectors are stack-5/v6/v7 again, which are already
        # committed.  Every arm's numbers are in the JSON record either way.
        if IDENTITY or lab not in ("k3-hold", "k3g-hold"):
            continue
        for f in FOLDS:
            pd.DataFrame({"MVT_ID_mvt": ref[f]["MVT_ID_mvt"].to_numpy("float64"),
                          "fold": f, "y_true": ref[f]["y_true"].to_numpy("float64"),
                          "y_pred": pe[f], "y_pred_nomask": p[f],
                          "y_pred_k2": k2e[f],
                          "y_pred_stack5": ref[f]["y_pred_stack5"].to_numpy("float64"),
                          "y_pred_v6": ref[f]["y_pred_v6"].to_numpy("float64")}).to_parquet(
                EXP / "oof" / f"exp-032-{TAG}-{lab}-{f}.parquet", index=False)

    out = {"members": list(NAMESK23G), "members_ungated": list(NAMESK23),
           "block": BLOCK, "win": WIN, "threads": THREADS,
           "seeds": list(S.SEEDS), "rounds": S.ROUNDS,
           "lightgbm": __import__("lightgbm").__version__,
           "gates": {"A_blend5": "pass", "B_stack5": "pass", "C_v6": "pass",
                     "D_k2": {f: offd[f] for f in FOLDS},
                     "k2_board_rebuilt": round(board_k2, 4)},
           "arms": recs, "wall_s": round(time.time() - t0, 1)}
    (EXP / "analysis" / f"candidate-{TAG}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[k3] {out['wall_s']:.0f}s wall -> analysis/candidate-{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
