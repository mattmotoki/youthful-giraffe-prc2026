"""The exp-034 candidate: K2's stacker with the 64-column bulk member ADDED.

`assignments/bulk.md`: "K2's stacker with the bulk member **added** (ninth slot;
do not substitute), the window block and the EHAM mask served as v7".  K2's set
is seven columns, so the added member is the EIGHTH; `SPEC.md` records the count
and takes "added, do not substitute" as the binding half.

Nothing is reimplemented.  The stacker is `exp-028/code/stack5w.arm`, the window
block is `stack5w.block_for`, the anchor is `exp-026/code/rule`, the frames are
`exp-026/code/frames`, and the scoring, the day-clustered intervals and the
slices are `exp-030/code/score`.  This file chooses member sets and does the
arithmetic of v7's EHAM mask.  It is `exp-032/code/candidate.py` with a
different added member and a partial-member coverage check in place of the
finite-everywhere one.

Five gates run before any bulk number is printed, each a rotation this script
recomputes rather than reads:

  A  blend-5 on the seven committed members reproduces `blend-config-5.json`
  B  the no-window stacker lands on exp-026's published stack-5
  C  the 600 s window arm lands on exp-028's published v6
  D  `k2-hold-eham-out` rebuilds K2 (v9) -- the row this candidate must beat
  E  under `PRC_IDENTITY=1` every arm collapses onto the committed v7, row by
     row, EXCEPT the additive arms, which carry an identity-pointed member
     twice and legitimately move the four member aggregates

Arms, all with the 600 s `depx` window block inside the stacker, the blend-5
anchor on the committed seven, and `noac` held at blend-5's anchor:

  k2-hold                  K2's seven -- gate D, the comparison point
  bulk-add-hold            K2's seven + `B1-bulk64`   <- the shipping arm
  bulk-add-bulkonly-hold   the same, the member nulled outside `nm_bulk`
                           (`SPEC.md`'s pre-registered repair)

    PRC_THREADS=6 python candidate.py
    PRC_IDENTITY=1 PRC_TAG=identity PRC_BULK=<F1-fresh's own OOF> python candidate.py
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

THREADS = int(os.environ.get("PRC_THREADS", "6"))
SW.THREADS = THREADS
WIN = float(os.environ.get("PRC_WIN", "600"))
BLOCK = os.environ.get("PRC_BLOCK", "depx")
TAG = os.environ.get("PRC_TAG", "bulk")
FOLDS = F.FOLDS

#: exp-027's K2 member, and this experiment's bulk member.  `BULK_ONLY` is the
#: same column nulled outside `nm_bulk`; it is derived here, not fitted again.
K2 = "K2-window"
BULK = "B1-bulk64"
BULK_ONLY = "B1-bulk64-nmbulk"
F.OOF[K2] = os.environ.get("PRC_K2", "experiments/exp-027-step/oof/k2-window-{fold}.parquet")
F.OOF[BULK] = os.environ.get("PRC_BULK", "experiments/exp-034-bulk/oof/bulk64-{fold}.parquet")
F.OOF[BULK_ONLY] = F.OOF[BULK]
#: Both bulk columns are PARTIAL: the member predicts `max(nm_taxi - offset, 0)`
#: and there is no `nm_taxi` without an NM join.  `frames.fold_frame` reads this
#: tuple at call time, so rebinding it here is the whole registration.
F.PARTIAL = tuple(dict.fromkeys(F.PARTIAL + (BULK, BULK_ONLY)))

NAMES5 = S.B5


def _sub(**slots) -> tuple:
    return tuple(slots.get(n, n) for n in NAMES5)


def _add(base: tuple, extra: str) -> tuple:
    """`base` with `extra` inserted before `N1-nmtail`.

    `N1-nmtail` goes last by CONVENTION.  exp-032's Sol round 1 established that
    position affects nothing -- `rule.py:49` removes the specialist BY NAME and
    `stack5.member_features` is name-driven -- and the convention is kept so
    that every member tuple in the repository reads the same way.
    """
    if base[-1] != "N1-nmtail":
        raise SystemExit(f"{base} does not end in N1-nmtail")
    got = base[:-1] + (extra, base[-1])
    if len(got) != len(base) + 1 or got[-1] != "N1-nmtail" or len(set(got)) != len(got):
        raise SystemExit(f"additive member set {got} is malformed")
    return got


NAMESK2 = _sub(**{"F1-fresh": K2})
if sum(a != b for a, b in zip(NAMES5, NAMESK2)) != 1 or len(NAMESK2) != len(NAMES5):
    raise SystemExit(f"{NAMESK2} does not differ from {NAMES5} in exactly one slot")
#: The candidate and its pre-registered repair.  Not pure supersets of
#: `k2-hold`: `stack5.member_features` also emits `mem_mean`, `mem_std`,
#: `mem_range` and `n_defined`, which an eighth column moves.  The same was true
#: when K2 entered, so the comparison against `k2-hold` stays like for like.
NAMESB = _add(NAMESK2, BULK)
NAMESBO = _add(NAMESK2, BULK_ONLY)

PUB_STACK5 = {"july_loo": 285.2029, "december": 205.6118, "january": 219.1303}
PUB_V6 = {"july_loo": 284.7219, "december": 205.6534, "january": 218.8166}
#: exp-027 section 16d, the shipped v9 (`k2-hold-eham-out`), and the bar
#: `assignments/bulk.md` sets against it.
PUB_K2 = {"july_loo": 284.3195, "december": 205.2156, "january": 218.2582}
PUB_K2_BOARD = 257.1653
GAIN_BAR = 0.10
FOLD_FLOOR = -0.10
#: exp-028's v7 rebuilt: what gate D's arm reads instead when the added member
#: is identity-pointed and `k2-hold` is therefore v7 rather than K2.
PUB_V7 = {"july_loo": 284.6251, "december": 205.5677, "january": 218.5412}
PUB_V7_BOARD = 257.4598
V7_BOARD = 263.4623
IDENTITY = os.environ.get("PRC_IDENTITY") == "1"
GATE = 1e-3
VEC_TOL = 1e-9


def frames() -> dict:
    """One frame per fold carrying all nine member columns, on stack-5's rows.

    The bulk member is partial, so the check is not `isfinite everywhere` (which
    is what a full member gets) but two statements about WHERE it is defined:

      1. it is never defined on a no-NM row (`nm_missing`), and
      2. it is defined on exactly the rows where `d_takeoff_aobt` is finite.

    (2) is the exact contract and (1) follows from it but is worth its own line.
    The member predicts `max(nm_taxi - offset, 0)` with
    `nm_taxi = MVT_TIME_UTC_mvt - AOBT_3_flt`, and `prc.blend` builds
    `d_takeoff_aobt` from the same two columns -- but the member's copy comes
    from exp-003's frame builder and the harness's from `C2-cat`'s out-of-fold
    file, so agreement across two producers is evidence and not a restatement.
    The two are NOT the same population as the NM join: `nm_missing` is
    `FLIGHT_ID_mvt` null, and a handful of rows carry a flight id with no
    `AOBT_3_flt` (4 / 4 / 4 per fold).  Those rows have an NM join and no
    prediction, and the count is printed rather than tolerated silently.
    """
    out = {}
    for f in FOLDS:
        m = F.fold_frame(f, NAMES5 + (K2, BULK))
        base = F.fold_frame(f, NAMES5)
        a = m["MVT_ID_mvt"].to_numpy("float64")
        b = base["MVT_ID_mvt"].to_numpy("float64")
        if not np.array_equal(a, b):
            raise SystemExit(f"{f}: the added member changes the frame: "
                             f"{len(a)} rows against {len(b)}")
        if not np.isfinite(m[K2].to_numpy("float64")).all():
            raise SystemExit(f"{f}: {K2} is not defined on every row")
        ok = np.isfinite(m[BULK].to_numpy("float64"))
        nm = ~m["nm_missing"].to_numpy().astype(bool)
        ta = np.isfinite(m["d_takeoff_aobt"].to_numpy("float64"))
        n_nm_no_aobt = int((nm & ~ta).sum())
        if IDENTITY:
            if not ok.all():
                raise SystemExit(f"{f}: the identity-pointed member must be full")
        else:
            if int((ok & ~nm).sum()):
                raise SystemExit(f"{f}: {BULK} is defined on {int((ok & ~nm).sum())} "
                                 f"no-NM rows; it cannot be")
            if not np.array_equal(ok, ta):
                raise SystemExit(f"{f}: {BULK} is defined on {int(ok.sum())} rows and "
                                 f"d_takeoff_aobt on {int(ta.sum())}; they must be the "
                                 f"same rows")
        # the repair column: the same member, null outside `nm_bulk`
        v = m[BULK].to_numpy("float64").copy()
        v[m["regime"].to_numpy() != "nm_bulk"] = np.nan
        m[BULK_ONLY] = v
        print(f"[frames] {f:9s} {len(m):7d} rows; {BULK} defined on {int(ok.sum()):7d} "
              f"({ok.mean():.4f}) = d_takeoff_aobt exactly, {BULK_ONLY} on "
              f"{int(np.isfinite(v).sum()):7d} ({np.isfinite(v).mean():.4f}); "
              f"{n_nm_no_aobt} rows have an NM join and no AOBT_3", flush=True)
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


def calday(d: pd.DataFrame) -> np.ndarray:
    """Literal CALENDAR-DAY clusters, ignoring the airport.

    Sol round 1, blocking 2: `assignments/bulk.md` says "day-clustered" and the
    harness's `panel.clusters` is calendar-day x AIRPORT, which is a finer unit
    (about 310 clusters a fold against 31).  Both are reported from here on: the
    harness's, because every candidate row in `docs/LEDGER.md` from stack-5
    onward is quoted on it, and this one, because it is what the assignment
    asked for literally.
    """
    return pd.factorize(np.floor(d["t"].to_numpy("float64") / 86400.0).astype("int64"))[0]


def against(rec: dict, preds: dict, ref: dict, base: dict, label: str,
            base_board: float | None = None) -> dict:
    """The arm's gain over a reference vector, with clustered intervals.

    Intervals are negated into the GAIN convention -- positive is better -- so
    that one record never carries two opposite sign conventions (exp-032 Sol
    round 1, blocking 5).  Two clustering units are computed, `panel.clusters`
    (calendar-day x airport) and `calday` (calendar day alone).
    """
    per, bas = {}, {}
    kg, kc, kn = f"vs_{label}", f"ci_{label}", f"ci_{label}_no_noac"
    kd = f"ci_{label}_calday"
    rec[kg], rec[kc], rec[kn], rec[kd] = {}, {}, {}, {}
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
        cd = calday(d)
        # Sol round 2, blocking 2: 4 decimals hid a bound of -0.000035 as
        # `-0.0`, which reads as an interval touching zero beside a count that
        # says it does not.  Persist 6 decimals AND the boolean itself, taken
        # from the unrounded bounds, so no reader has to infer it from a
        # rounded number.
        def neg(t):
            lo, hi = -t[2], -t[1]
            return [round(-t[0], 6), round(lo, 6), round(hi, 6),
                    {"contains_zero": bool(lo <= 0.0 <= hi)}]
        rec[kc][f] = neg(Sc.P.boot_delta(y[ok], pv[ok], base[f][ok],
                                         pd.factorize(cl[ok])[0]))
        rec[kn][f] = neg(Sc.P.boot_delta(y[nn], pv[nn], base[f][nn],
                                         pd.factorize(cl[nn])[0]))
        rec[kd][f] = neg(Sc.P.boot_delta(y[ok], pv[ok], base[f][ok],
                                         pd.factorize(cd[ok])[0]))
    rec["interval_convention"] = (
        "`ci_k2*` and `ci_v7*`, written here, are GAIN: positive is BETTER than "
        "the reference, [point, lo, hi, {contains_zero}], 95 %; `ci_*` clusters "
        "on the harness's calendar day x airport, `ci_*_calday` on the calendar "
        "day alone. `ci_v6` and `ci_v6_no_noac` are NOT written here: they are "
        "inherited unchanged from experiments/exp-030-window2/code/score.py in "
        "the DELTA convention, where negative is better, so they carry the "
        "opposite sign to `vs_v6` beside them (Sol round 2, blocking 5)")
    rec[f"clean_board_{label}"] = round(Sc.P.board(bas), 4)
    rec[f"gain_board_vs_{label}"] = round(Sc.P.board(bas) - Sc.P.board(per), 4)
    if base_board is not None:
        rec[f"predicted_board_vs_{label}"] = round(base_board - rec[f"gain_board_vs_{label}"], 4)
    print(f"  vs {label:9s} " + "  ".join(f"{f} {rec[kg][f]:+9.4f}" for f in FOLDS)
          + f"   board {rec[f'gain_board_vs_{label}']:+9.4f}")
    for f in FOLDS:
        dl, lo, hi, zc = rec[kc][f]
        dn, lon, hin, _ = rec[kn][f]
        _, lod, hid, zd = rec[kd][f]
        z = lambda b: " 0" if b["contains_zero"] else "  "
        print(f"    {f:9s} vs {label} {dl:+7.3f} [{lo:+8.4f}, {hi:+8.4f}]{z(zc)}"
              f"   calendar-day [{lod:+8.4f}, {hid:+8.4f}]{z(zd)}"
              f"   no-noac {dn:+7.3f} [{lon:+7.3f}, {hin:+7.3f}]"
              f"   ('0' marks an interval that contains zero)")
    if base_board is not None:
        print(f"  predicted board {rec[f'predicted_board_vs_{label}']:.4f} "
              f"({label} measured {base_board}, at face)")
    return rec


def slice_delta(preds: dict, base: dict, ref: dict, mask_of) -> dict:
    """`rmse` on a slice against a reference; positive is better."""
    out = {}
    for f in FOLDS:
        d = ref[f]
        y = d["y_true"].to_numpy("float64")
        ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
        m = ok & mask_of(d, y)
        if not m.any():
            out[f] = {"n": 0}
            continue
        r = lambda p: float(np.sqrt(np.mean((y[m] - np.asarray(p, "float64")[m]) ** 2)))
        out[f] = {"n": int(m.sum()), "base": round(r(base[f]), 4),
                  "arm": round(r(preds[f]), 4), "gain": round(r(base[f]) - r(preds[f]), 4)}
    return out


def main() -> None:
    t0 = time.time()
    data = frames()
    print(f"[bulk] frames in {time.time()-t0:.0f}s, {THREADS} threads, "
          f"block={BLOCK} win={WIN:.0f}s, identity={IDENTITY}", flush=True)

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
    print(f"[bulk] block {BLOCK} = {blk[FOLDS[0]].shape[1]} columns", flush=True)

    ref = Sc.reference()
    v7, _ = eham_mask({f: ref[f]["y_pred_v6"].to_numpy("float64") for f in FOLDS}, ref)
    v7c = {f: D.rmse_clean(ref[f]["y_true"], v7[f], ref[f]["MVT_ID_mvt"], f) for f in FOLDS}
    print("[bulk] exp-028's committed stack-5 and v6 read back; v7 rebuilt "
          + "  ".join(f"{f} {v7c[f]:.4f}" for f in FOLDS)
          + f"  board {Sc.P.board(v7c):.4f}", flush=True)

    arms = {
        "stack-5 (gate B)":       (NAMES5, None),
        "v6 (gate C)":            (NAMES5, blk),
        "k2-hold":                (NAMESK2, blk),
        "bulk-add-hold":          (NAMESB, blk),
        "bulk-add-bulkonly-hold": (NAMESBO, blk),
    }
    MEMBERS = {"bulk-add-hold": NAMESB, "bulk-add-bulkonly-hold": NAMESBO}
    raw, recs = {}, []
    for lab, (names, bk) in arms.items():
        t = time.time()
        p = rotate(data, names, a5, bk, ref5)
        raw[lab] = p
        clean = {f: D.rmse_clean(data[f]["y_true"], p[f],
                                 data[f]["MVT_ID_mvt"], f) for f in FOLDS}
        print(f"[{lab:22s}] " + "  ".join(f"{f} {clean[f]:9.4f}" for f in FOLDS)
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
    k2e, n_eham = eham_mask(k2p, ref)
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
        mdv = {f: float(np.abs(k2e[f] - v7[f]).max()) for f in FOLDS}
        print(f"[gate D+] k2-hold-eham-out against the COMMITTED v7, row by row: "
              f"max abs difference {max(mdv.values()):.3g}", flush=True)
        if max(mdv.values()) > VEC_TOL:
            raise SystemExit(f"gate D+ fails: k2-hold is not v7 row-wise: {mdv}")
        for lab in MEMBERS:
            print(f"[gate E] {lab:22s} SKIPPED: an additive arm carries the "
                  f"identity-pointed member twice, so the member aggregates move "
                  f"and v7 is not the right target; gate D covers its baseline",
                  flush=True)

    # Sol round 1, blocking 1.  `SPEC.md`'s pre-registered repair nulls the
    # member COLUMN outside `nm_bulk`; the stacker is still one global model, so
    # its OUTPUT still moves on 16,581 / 6,428 / 5,073 non-`nm_bulk` rows.  The
    # assignment's words ("the member restricted to the `nm_bulk` regime rows in
    # the stacker") also carry the literal output reading, and that one was not
    # measured.  It is arithmetic on two vectors this run already has -- the
    # additive arm inside `nm_bulk`, K2 everywhere else -- so it is added here.
    # It is EXPLORATORY: it was not pre-registered, and it is recorded as such.
    EXPLORATORY = "bulk-add-nmbulk-out"
    raw_out = {}
    gates_out = {}
    for lab in MEMBERS:
        p = align(raw[lab], data, ref)
        pe, n = eham_mask(p, ref)
        rece = Sc.score(f"{lab}-eham-out", pe, ref,
                        note=f"{lab} with EHAM served from stack-5, v7's mask; "
                             f"members {'/'.join(MEMBERS[lab])}; "
                             f"anchor blend-5 on the committed seven; block {BLOCK} "
                             f"{WIN:.0f}s; noac held at blend-5's anchor; "
                             + ", ".join(f"{f} {n[f]}" for f in FOLDS)
                             + " EHAM rows taken back")
        rece["airport_excluded"] = "EHAM"
        rece["n_rows_taken_from_stack5"] = n
        Sc.show(rece)
        against(rece, pe, ref, v7, "v7", V7_BOARD)
        against(rece, pe, ref, k2e, "k2", None)
        rece["tail_vs_k2"] = slice_delta(pe, k2e, ref, lambda d, y: y > 3600.0)
        rece["bulk_vs_k2"] = slice_delta(pe, k2e, ref, lambda d, y: y <= 3600.0)
        rece["nmbulk_vs_k2"] = slice_delta(
            pe, k2e, ref, lambda d, y: d["regime"].to_numpy() == "nm_bulk")
        rece["nonm_vs_k2"] = slice_delta(
            pe, k2e, ref, lambda d, y: d["regime"].to_numpy() == "nonm")
        # Sol round 1, blocking 1: nulling the member COLUMN outside `nm_bulk`
        # does not stop the arm's OUTPUT moving there, because the stacker is
        # one global model.  Count it rather than assume it.
        rece["rows_moved_outside_nm_bulk_vs_k2"] = {
            f: int(((ref[f]["regime"].to_numpy() != "nm_bulk")
                    & (pe[f] != k2e[f])).sum()) for f in FOLDS}
        print("  rows moved outside nm_bulk vs k2 " + "  ".join(
            f"{f} {rece['rows_moved_outside_nm_bulk_vs_k2'][f]}" for f in FOLDS), flush=True)
        # `score.py`'s inherited `gate` field is the v6 gate this harness was
        # born with; `assignments/bulk.md` gates on K2 (exp-032 Sol round 1,
        # blocking 3).  Both are recorded, each named for what it is.
        rece["gate_v6_inherited"] = rece.pop("gate", None)
        gk = rece["gain_board_vs_k2"] >= GAIN_BAR
        gf = min(rece["vs_k2"].values()) >= FOLD_FLOOR
        rece["gate_k2"] = {"bar_board": GAIN_BAR, "floor_fold": FOLD_FLOOR,
                           "gain_board": rece["gain_board_vs_k2"],
                           "worst_fold": round(min(rece["vs_k2"].values()), 4),
                           "gain_passes": bool(gk), "folds_pass": bool(gf),
                           "PASS": bool(gk and gf)}
        gates_out[lab] = rece["gate_k2"]
        print(f"  ASSIGNMENT GATE vs K2: gain {rece['gain_board_vs_k2']:+.4f} "
              f"(bar {GAIN_BAR:+.2f}) {'ok' if gk else 'FAIL'}, worst fold "
              f"{min(rece['vs_k2'].values()):+.4f} (floor {FOLD_FLOOR:+.2f}) "
              f"{'ok' if gf else 'FAIL'}  ->  {'PASS' if gk and gf else 'FAIL'}", flush=True)
        print("  y>3600 vs k2 " + "  ".join(
            f"{f} {rece['tail_vs_k2'][f]['gain']:+7.2f}" for f in FOLDS)
            + "   y<=3600 " + "  ".join(
            f"{f} {rece['bulk_vs_k2'][f]['gain']:+7.3f}" for f in FOLDS), flush=True)
        recs.append(rece)
        # Both readings of "the member restricted to the nm_bulk regime rows in
        # the stacker" get the output form: the arm whose COLUMN is full, and
        # the arm whose column is already nulled.  Neither was pre-registered.
        for src, tag in (("bulk-add-hold", EXPLORATORY),
                         ("bulk-add-bulkonly-hold", "bulkonly-nmbulk-out")):
            if lab != src:
                continue
            sel = {f: (ref[f]["regime"].to_numpy() == "nm_bulk") for f in FOLDS}
            vec = {f: np.where(sel[f], pe[f], k2e[f]) for f in FOLDS}
            raw_out[tag] = vec
            n_out = {f: int((~sel[f]).sum()) for f in FOLDS}
            reco = Sc.score(f"{tag}-eham-out", vec, ref,
                            note=f"EXPLORATORY, not pre-registered (Sol round 1, blocking 1): "
                                 f"{src} inside the nm_bulk regime, K2 on every other row; "
                                 + ", ".join(f"{f} {n_out[f]}" for f in FOLDS)
                                 + " rows taken from K2")
            reco["airport_excluded"] = "EHAM"
            reco["exploratory"] = True
            reco["source_arm"] = src
            reco["n_rows_taken_from_k2"] = n_out
            Sc.show(reco)
            against(reco, vec, ref, v7, "v7", V7_BOARD)
            against(reco, vec, ref, k2e, "k2", None)
            reco["tail_vs_k2"] = slice_delta(vec, k2e, ref, lambda d, y: y > 3600.0)
            reco["bulk_vs_k2"] = slice_delta(vec, k2e, ref, lambda d, y: y <= 3600.0)
            reco["gate_v6_inherited"] = reco.pop("gate", None)
            gko = reco["gain_board_vs_k2"] >= GAIN_BAR
            gfo = min(reco["vs_k2"].values()) >= FOLD_FLOOR
            reco["gate_k2"] = {"bar_board": GAIN_BAR, "floor_fold": FOLD_FLOOR,
                               "gain_board": reco["gain_board_vs_k2"],
                               "worst_fold": round(min(reco["vs_k2"].values()), 4),
                               "gain_passes": bool(gko), "folds_pass": bool(gfo),
                               "PASS": bool(gko and gfo)}
            gates_out[tag] = reco["gate_k2"]
            print(f"  ASSIGNMENT GATE vs K2: gain {reco['gain_board_vs_k2']:+.4f} "
                  f"(bar {GAIN_BAR:+.2f}) {'ok' if gko else 'FAIL'}, worst fold "
                  f"{min(reco['vs_k2'].values()):+.4f} (floor {FOLD_FLOOR:+.2f}) "
                  f"{'ok' if gfo else 'FAIL'}  ->  "
                  f"{'PASS' if gko and gfo else 'FAIL'}  [EXPLORATORY]", flush=True)
            recs.append(reco)
        if IDENTITY:
            continue
        for f in FOLDS:
            pd.DataFrame({"MVT_ID_mvt": ref[f]["MVT_ID_mvt"].to_numpy("float64"),
                          "fold": f, "y_true": ref[f]["y_true"].to_numpy("float64"),
                          "y_pred": pe[f], "y_pred_nomask": p[f],
                          "y_pred_k2": k2e[f],
                          "y_pred_stack5": ref[f]["y_pred_stack5"].to_numpy("float64"),
                          "y_pred_v6": ref[f]["y_pred_v6"].to_numpy("float64")}).to_parquet(
                EXP / "oof" / f"exp-034-{TAG}-{lab}-{f}.parquet", index=False)

    # Sol round 2, blocking 4: every scored arm gets its out-of-fold file,
    # exploratory ones included (COMMON.md).  One loop, both arms; the earlier
    # single-arm branch is what the duplicated scoring path cost.
    if not IDENTITY:
        for tag, vec in raw_out.items():
            for f in FOLDS:
                pd.DataFrame({"MVT_ID_mvt": ref[f]["MVT_ID_mvt"].to_numpy("float64"),
                              "fold": f, "y_true": ref[f]["y_true"].to_numpy("float64"),
                              "y_pred": vec[f], "y_pred_k2": k2e[f]}).to_parquet(
                    EXP / "oof" / f"exp-034-{TAG}-{tag}-{f}.parquet", index=False)

    out = {"members": list(NAMESB), "members_repair": list(NAMESBO),
           "members_k2": list(NAMESK2),
           "exploratory_arms": ["bulk-add-nmbulk-out-eham-out",
                                "bulkonly-nmbulk-out-eham-out"],
           "block": BLOCK, "win": WIN, "threads": THREADS,
           "seeds": list(S.SEEDS), "rounds": S.ROUNDS,
           "identity": IDENTITY,
           "lightgbm": __import__("lightgbm").__version__,
           "shipping_arm": "bulk-add-hold",
           "gates": {"A_blend5": "pass", "B_stack5": "pass", "C_v6": "pass",
                     "D_k2": offd, "k2_board_rebuilt": round(board_k2, 4),
                     "assignment_gate_k2": gates_out},
           "n_eham_rows": n_eham,
           "arms": recs, "wall_s": round(time.time() - t0, 1)}
    (EXP / "analysis" / f"candidate-{TAG}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[bulk] {out['wall_s']:.0f}s wall -> analysis/candidate-{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
