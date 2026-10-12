"""The five shipped rotations, recomputed on each column of the instrument.

Nothing here is a new model.  The stacker is `exp-028/code/stack5w.arm`, the
anchor is `exp-026/code/rule`, the mixed anchor is `exp-027/code/refit_bulk`,
the EHAM mask is `exp-027/code/candidate.eham_mask`, the seeds, the rounds and
the floor are `exp-026/code/stack5`, and the member frames are
`exp-026/code/frames` with `instrument.py`'s column swap on top.  This file
chooses a column and a scored fold and does the arithmetic of the chain.

The five candidates, in the order they were shipped:

  stack-5      members only, no block
  v6           + the 600 s `depx` window block inside the same stacker
  v7           v6 with every EHAM departure served from stack-5
  K2           v7's recipe with `K2-window` in `F1-fresh`'s slot
  refit-bulk   K2 with the anchor refitted off the NM tail only

`noac` is held at blend-5's own rotated anchor in every arm on every column, as
score-feedback rule requires, so the 388 ranking `noac` rows are common to the whole table and
cancel out of every difference in it.

**Two label channels, and `--anchor` switches the second one.**
`exp-026-rebuild/RESULTS.md` §4a names them. *Channel B* is the members': the
stacker's feature columns come from member models trained on month sets that
include the scored fold. That is what `instrument.py`'s column swap closes, for
the three members exp-025 could refit. *Channel A* is the anchor's: scoring fold
`f`, the stacker's TRAINING anchor on fold `g` is the blend fitted on the other
two folds, and "the other two" includes `f`, so `f`'s labels reach the training
residual through seven NNLS weights. `--anchor nested` closes it with
`exp-026/code/nested_anchor.py`'s construction — fold `g`'s training anchor is
fitted on the SINGLE remaining fold, which is neither `f` nor `g` — and carries
that file's caution: the nested anchor is also a WEAKER anchor, one fold of
fitting rows instead of two, so a move under it is two effects at once.

    PRC_THREADS=4 python rotate.py                    # rotated anchor, four columns
    PRC_THREADS=4 python rotate.py --anchor nested    # both channels closed
    PRC_THREADS=4 python rotate.py --columns published
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp-027-step/code"))
import instrument as I  # noqa: E402
import candidate as C  # noqa: E402   (registers K2-window's address in frames.OOF)
import refit_bulk as RB  # noqa: E402   (the rule `mixed_anchor` below reproduces)
sys.path.insert(0, str(HERE.parent / "tests"))
import direction as DIR  # noqa: E402

F, R, S, SW, Sc = C.F, C.R, C.S, C.SW, C.Sc
B, D = C.B, C.D
FOLDS = I.FOLDS
THREADS = int(os.environ.get("PRC_THREADS", "4"))
SW.THREADS = THREADS
S.LGB_BASE["num_threads"] = THREADS
#: Every member column any arm reads.  One frame carries all of them, so the
#: arms differ by the member tuple they select and by nothing else.
ALL = C.NAMES5 + (C.K2,)
ARMS = ("stack-5", "v6", "v7", "K2", "refit-bulk")
#: The published fold triples every arm must land on when the column is
#: `published`.  stack-5/v6: exp-026 and exp-028.  v7, K2, refit-bulk:
#: `exp-027-step/analysis/refit-bulk.json` and `candidate-k2.json`.
PUBLISHED = {
    "stack-5":    {"july_loo": 285.2029, "december": 205.6118, "january": 219.1303},
    "v6":         {"july_loo": 284.7219, "december": 205.6534, "january": 218.8166},
    "v7":         {"july_loo": 284.6251, "december": 205.5677, "january": 218.5412},
    "K2":         {"july_loo": 284.3195, "december": 205.2156, "january": 218.2582},
    "refit-bulk": {"july_loo": 284.1774, "december": 204.9052, "january": 218.2115},
}
GATE = 1e-3
VEC_TOL = 1e-9
BLOCK, WIN = C.BLOCK, C.WIN


def _src(g: str, scored: str | None) -> list:
    """Which folds fit the anchor applied on fold `g`.

    `scored=None` is the shipped rotation: the other two folds, so the anchor
    applied on a TRAINING fold `g` was fitted with the scored fold in it and
    carries its labels (channel A).  `scored=f` drops `f` as well, which leaves
    exactly one fold for a training anchor and the same two for the apply-time
    anchor on `f` itself (`f` was never in its own).
    """
    return [h for h in FOLDS if h != g and h != scored]


def anchors(data: dict, names, scored: str | None = None) -> dict:
    """blend-5's rotated per-regime hybrid NNLS, anchor channel open or closed."""
    return {g: R.apply_hybrid(data[g], names,
                              R.fit_hybrid(pd.concat([data[h] for h in _src(g, scored)],
                                                     ignore_index=True), names))
            for g in FOLDS}


def mixed_anchor(data: dict, scored: str | None = None) -> dict:
    """`refit_bulk.mixed_anchor` with the same anchor-channel switch.

    blend-K off the NM tail, blend-5 on it, fold by fold.  Identical to
    `exp-027/code/refit_bulk.mixed_anchor` when `scored` is None.
    """
    if scored is None:
        return RB.mixed_anchor(data)      # exp-027's own function, unchanged
    out = {}
    for g in FOLDS:
        src = pd.concat([data[h] for h in _src(g, scored)], ignore_index=True)
        a5 = R.apply_hybrid(data[g], C.NAMES5, R.fit_hybrid(src, C.NAMES5))
        aK = R.apply_hybrid(data[g], C.NAMESK, R.fit_hybrid(src, C.NAMESK))
        _, late = R._split(data[g], list(C.NAMESK))
        out[g] = np.where(late, a5, aK)
    return out


def bag(data: dict, f: str, names, anchor: dict, blk: dict | None,
        noac_ref: np.ndarray) -> np.ndarray:
    """One arm's vector for fold `f`: the five-seed stacker bag, `noac` held."""
    ap = data[f]
    fit = pd.concat([data[g] for g in FOLDS if g != f], ignore_index=True)
    a_fit = np.concatenate([anchor[g] for g in FOLDS if g != f])
    xf = pd.concat([blk[g] for g in FOLDS if g != f], ignore_index=True) if blk else None
    xa = blk[f] if blk else None
    p = np.maximum(np.mean([SW.arm(fit, ap, s, a_fit, anchor[f], names, xf, xa)
                            for s in S.SEEDS], axis=0), B.FLOOR)
    return np.where(ap["regime"].to_numpy() == "noac", noac_ref, p)


def column(col: str, base: dict, blocks: dict, log, nested_anchor: bool) -> dict:
    """Every arm's vector, fold by fold, on one column of the instrument."""
    out = {a: {} for a in ARMS}
    meta = {}
    for f in FOLDS:
        t = time.time()
        data = I.build(col, f, ALL, base)
        meta[f] = I.describe(col, f, ALL)
        sc = f if nested_anchor else None
        a5 = anchors(data, C.NAMES5, sc)
        # `noac` is held at the APPLY-time anchor on `f`, which is fitted on the
        # other two folds under either switch and never sees `f`.
        ref5 = np.maximum(a5[f], B.FLOOR)
        am = mixed_anchor(data, sc)
        s5 = bag(data, f, C.NAMES5, a5, None, ref5)
        v6 = bag(data, f, C.NAMES5, a5, blocks, ref5)
        k2 = bag(data, f, C.NAMESK, a5, blocks, ref5)
        rb = bag(data, f, C.NAMESK, am, blocks, ref5)
        eham = data[f]["airport"].astype(str).to_numpy() == "EHAM"
        out["stack-5"][f] = s5
        out["v6"][f] = v6
        out["v7"][f] = np.where(eham, s5, v6)
        out["K2"][f] = np.where(eham, s5, k2)
        out["refit-bulk"][f] = np.where(eham, s5, rb)
        y = data[f]["y_true"].to_numpy("float64")
        ids = data[f]["MVT_ID_mvt"].to_numpy("float64")
        line = (f"[{col:14s} {f:9s}] " + "  ".join(
            f"{a} {D.rmse_clean(y, out[a][f], ids, f):9.4f}" for a in ARMS)
            + f"   {int(eham.sum())} EHAM rows   {time.time()-t:.0f}s")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    return {"pred": out, "sources": meta}


def boot_board(y: dict, new: dict, ref: dict, cl: dict, draws: int = 2000,
               seed: int = 7) -> list:
    """A day-clustered interval on the BOARD difference, not on one fold.

    `stack5.boot_delta` resamples day x airport clusters inside one fold.  The
    board is the two ranking months in quadrature, and July and January are
    disjoint calendars, so each month is resampled independently with its own
    clusters and the two per-draw RMSEs are combined by `frames.board_estimate`.
    Point estimate = ref board - new board, positive meaning the new arm is
    better; the interval is the 2.5/97.5 quantiles of the same quantity.
    """
    rng = np.random.default_rng(seed)
    sh = list(F.SHARES)
    bn = np.zeros((draws, len(sh)))
    br = np.zeros((draws, len(sh)))
    for j, f in enumerate(sh):
        c = cl[f]
        e_n = (new[f] - y[f]) ** 2
        e_r = (ref[f] - y[f]) ** 2
        k = c.max() + 1
        order = np.argsort(c, kind="stable")
        st = np.searchsorted(c[order], np.arange(k))
        en = np.searchsorted(c[order], np.arange(k), side="right")
        sn = np.add.reduceat(e_n[order], st)
        sr = np.add.reduceat(e_r[order], st)
        cnt = (en - st).astype("float64")
        W = rng.multinomial(k, np.full(k, 1.0 / k), size=draws).astype("float64")
        n = W @ cnt
        bn[:, j] = np.sqrt((W @ sn) / n)
        br[:, j] = np.sqrt((W @ sr) / n)
    w = np.array([F.SHARES[f] for f in sh])
    boards_n = np.sqrt(bn ** 2 @ w)
    boards_r = np.sqrt(br ** 2 @ w)
    pt = (F.board_estimate({f: float(np.sqrt(np.mean((ref[f] - y[f]) ** 2))) for f in sh})
          - F.board_estimate({f: float(np.sqrt(np.mean((new[f] - y[f]) ** 2))) for f in sh}))
    d = boards_r - boards_n
    return [round(pt, 4), round(float(np.quantile(d, 0.025)), 4),
            round(float(np.quantile(d, 0.975)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", default=",".join(I.COLUMNS))
    ap.add_argument("--anchor", choices=("rotated", "nested"), default="rotated",
                    help="`nested` also closes the anchor label channel (exp-026 §4a A)")
    a = ap.parse_args()
    cols = a.columns.split(",")
    nested_anchor = a.anchor == "nested"
    tag = "" if not nested_anchor else "-nestedanchor"
    t0 = time.time()

    # Sol round 1, finding 4: the published gate cannot see a wrong substitution,
    # because nothing is substituted on that column.  `tests/direction.py` can,
    # and runs here, before any number.  `tests/swap_equivalence.py` is the other
    # such test; it rebuilds 36 frames and stays a separate command
    # (Sol round 2, finding 7 -- this comment used to claim both ran here).
    DIR.main()

    base = I.base_frames(ALL)
    print(f"[exp-031] frames in {time.time()-t0:.0f}s, {THREADS} threads, "
          f"block={BLOCK} win={WIN:.0f}s, anchor={a.anchor}", flush=True)
    # The block is built from the gate columns alone (`stack5w.panel_of`), which
    # no member swap touches, so one block per fold serves every column.
    blocks = {f: SW.block_for(f, SW.panel_of(base[f]), {"blocks": [BLOCK], "win": WIN})
              for f in FOLDS}
    print(f"[exp-031] block {BLOCK} = {blocks[FOLDS[0]].shape[1]} columns", flush=True)

    keys = {f: {"y": base[f]["y_true"].to_numpy("float64"),
                "ids": base[f]["MVT_ID_mvt"].to_numpy("float64"),
                "cl": S.clusters(base[f]),
                "regime": base[f]["regime"].to_numpy(),
                "ok": ~base[f]["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()}
            for f in FOLDS}

    # ---- gate A, before any arm: blend-5's own rotation ------------------
    gates = {}
    a5pub = anchors(base, C.NAMES5)
    gates["A_blend5_all_rows"] = {}
    for f in FOLDS:
        y = base[f]["y_true"].to_numpy("float64")
        got = float(np.sqrt(np.mean((y - np.maximum(a5pub[f], B.FLOOR)) ** 2)))
        off = round(got - S.PUBLISHED[f], 4)
        gates["A_blend5_all_rows"][f] = off
        if abs(off) > GATE:
            raise SystemExit(f"gate A fails on {f}: {off:+.4f}")
    print(f"[gate A] blend-5 reproduces blend-config-5.json on all three folds: "
          f"{gates['A_blend5_all_rows']}", flush=True)

    # ---- the published column, gated before anything else runs -----------
    res = {}
    order = (["published"] if "published" in cols else []) + \
            [c for c in cols if c != "published"]
    with open(EXP / "logs" / f"rotate{tag}.log", "a") as log:
        log.write(f"\n=== {pd.Timestamp.utcnow():%Y-%m-%dT%H:%M} threads={THREADS} "
                  f"anchor={a.anchor} columns={order} ===\n")
        for col in order:
            res[col] = column(col, base, blocks, log, nested_anchor)
            if col != "published":
                continue
            # Under `--anchor nested` the published column is a DIFFERENT rotation
            # from the shipped one, so the triples cannot be gated; the gate that
            # still runs there is gate A above plus this experiment's own
            # `--anchor rotated` run, which does land on them.
            if nested_anchor:
                print("[gate] the published triples are not gated under "
                      "--anchor nested: closing channel A changes the rotation "
                      "the shipped numbers were measured on", flush=True)
                continue
            ref28 = Sc.reference()
            for arm in ARMS:
                off = {f: round(D.rmse_clean(keys[f]["y"], res["published"]["pred"][arm][f],
                                             keys[f]["ids"], f) - PUBLISHED[arm][f], 4)
                       for f in FOLDS}
                gates[arm] = off
                print(f"[gate] {arm:11s} against the published triple: {off}", flush=True)
                if max(abs(v) for v in off.values()) > GATE:
                    raise SystemExit(f"gate fails for {arm}: {off}")
            md = {}
            for arm, col_ in (("stack-5", "y_pred_stack5"), ("v6", "y_pred_v6")):
                pv = C.align(res["published"]["pred"][arm], base, ref28)
                md[arm] = {f: float(np.abs(pv[f] - ref28[f][col_].to_numpy("float64")).max())
                           for f in FOLDS}
                if max(md[arm].values()) > VEC_TOL:
                    raise SystemExit(f"gate fails on {arm}'s vector: {md[arm]}")
            gates["vector_max_abs"] = md
            print(f"[gate] vectors reproduce exp-028's committed files: max abs row "
                  f"difference {max(max(v.values()) for v in md.values()):.3g}", flush=True)

    # ---- the tables ------------------------------------------------------
    # Sol round 1, finding 5: the fold RMSE is kept at full precision and the
    # board and every gain are computed from it.  Rounding happens once, on the
    # way into the JSON, so a board can no longer inherit three rounded folds.
    ex = {c: {a_: {f: D.rmse_clean(keys[f]["y"], res[c]["pred"][a_][f], keys[f]["ids"], f)
                   for f in FOLDS} for a_ in ARMS} for c in order}
    bde = {c: {a_: F.board_estimate({f: ex[c][a_][f] for f in F.SHARES}) for a_ in ARMS}
           for c in order}
    clean = {c: {a_: {f: round(ex[c][a_][f], 4) for f in FOLDS} for a_ in ARMS} for c in order}
    boards = {c: {a_: round(bde[c][a_], 4) for a_ in ARMS} for c in order}
    pairs = [("v6", "stack-5"), ("v7", "v6"), ("K2", "v7"),
             ("refit-bulk", "v7"), ("K2", "refit-bulk")]
    gains = {}
    for c in order:
        gains[c] = {}
        for new, old in pairs:
            g = {"per_fold": {}, "ci": {}, "ci_no_noac": {}}
            for f in FOLDS:
                kk, pn, pr = keys[f], res[c]["pred"][new][f], res[c]["pred"][old][f]
                g["per_fold"][f] = round(ex[c][old][f] - ex[c][new][f], 4)
                ok, cl = kk["ok"], kk["cl"]
                nn = ok & (kk["regime"] != "noac")
                dd, lo, hi = S.boot_delta(kk["y"][ok], pn[ok], pr[ok],
                                          pd.factorize(cl[ok])[0])
                g["ci"][f] = [round(-dd, 4), round(-hi, 4), round(-lo, 4)]
                dd, lo, hi = S.boot_delta(kk["y"][nn], pn[nn], pr[nn],
                                          pd.factorize(cl[nn])[0])
                g["ci_no_noac"][f] = [round(-dd, 4), round(-hi, 4), round(-lo, 4)]
            g["board"] = round(bde[c][old] - bde[c][new], 4)
            g["board_ci"] = boot_board(
                {f: keys[f]["y"][keys[f]["ok"]] for f in F.SHARES},
                {f: res[c]["pred"][new][f][keys[f]["ok"]] for f in F.SHARES},
                {f: res[c]["pred"][old][f][keys[f]["ok"]] for f in F.SHARES},
                {f: pd.factorize(keys[f]["cl"][keys[f]["ok"]])[0] for f in F.SHARES})
            gains[c][f"{new} - {old}"] = g

    out = {"columns": order, "arms": list(ARMS), "threads": THREADS,
           "anchor": a.anchor,
           "anchor_note": ("channel A closed: fold g's TRAINING anchor is fitted on the "
                           "single fold that is neither g nor the scored fold "
                           "(exp-026/code/nested_anchor.py); it is also a weaker anchor, "
                           "one fitting fold instead of two"
                           if nested_anchor else
                           "the shipped rotation: fold g's anchor is fitted on the other "
                           "two folds, so channel A is open"),
           "block": BLOCK, "win": WIN, "seeds": list(S.SEEDS), "rounds": S.ROUNDS,
           "lightgbm": __import__("lightgbm").__version__,
           "members": {"stack-5/v6/v7": list(C.NAMES5), "K2/refit-bulk": list(C.NAMESK)},
           "nested_members": list(I.NESTABLE),
           "unnested_members": sorted(m for m in set(C.NAMES5) | set(C.NAMESK)
                                      if m not in I.NESTABLE),
           "gates": gates, "clean": clean, "clean_exact": ex, "board": boards,
           "board_exact": bde, "gain": gains,
           "sources": {c: res[c]["sources"] for c in order},
           "wall_s": round(time.time() - t0, 1)}
    (EXP / "analysis" / f"rotation{tag}.json").write_text(json.dumps(out, indent=2))
    for c in order:
        if c == "published" and not nested_anchor:
            continue          # already in git, see oof/README.md
        for f in FOLDS:
            pd.DataFrame({"MVT_ID_mvt": keys[f]["ids"], "fold": f, "y_true": keys[f]["y"],
                          **{f"y_pred_{a_.replace('-', '_')}": res[c]["pred"][a_][f]
                             for a_ in ARMS}}).to_parquet(
                EXP / "oof" / f"exp-031{tag}-{c}-{f}.parquet", index=False)
    print(f"\n[exp-031] {out['wall_s']:.0f}s wall -> analysis/rotation{tag}.json", flush=True)


if __name__ == "__main__":
    main()
