"""The four fold-side vectors this experiment prices against, and their gates.

Nothing here is refitted.  Every vector is read back from a committed rotation
and checked against the number its own experiment published before any arm is
scored:

    stack-5   exp-026, through `exp-030/code/score.reference()`
    v6        exp-028's 600 s window stacker, same call
    v7        v6 with EHAM served from stack-5 -- exp-030 arm 3, rebuilt here as
              arithmetic on the two gated vectors exactly as `exp-027/code/
              candidate.eham_mask` does, so it needs no file of its own
    K2 (v9)   `exp-027-step/oof/exp-027-k2-k2-hold-{fold}.parquet`, column
              `y_pred` -- K2 in F1-fresh's slot, blend-5 anchor, 600 s block,
              v7's EHAM mask.  The file the board holds at 263.0827
    refit-bulk `exp-027-step/oof/exp-027-k2-refit-bulk-{fold}.parquet`, the
              MAYBE: the same with the anchor refitted off the NM tail

`exp-030/code/score.py` prices an arm against v6 because exp-030's arms were cuts
on v6.  This experiment's arms are cuts on **K2**, the file on the board, so
`against()` here is `exp-027/code/candidate.against_v7` with the base named
rather than fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
E27 = ROOT / "experiments/exp-027-step"
sys.path.insert(0, str(ROOT / "experiments/exp-030-window2/code"))
import score as Sc  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = Sc.FOLDS
AIRPORT_HELD = "EHAM"
GATE = 1e-3
VEC_TOL = 1e-9

#: What each base's own experiment published, and the board it measured.
#:   v7   exp-030 RESULTS section 2, board `docs/LEDGER.md` BOARD READING 7
#:   K2   exp-027 `analysis/candidate-k2.json`, arm `k2-hold-eham-out`;
#:        board BOARD READING 8 (v9, 00:01 UTC on 2026-09-08)
#:   refit-bulk  exp-027 `analysis/refit-bulk.json`; never uploaded, no board
PUBLISHED = {
    "v7": {"july_loo": 284.6251, "december": 205.5677, "january": 218.5412,
           "board": 257.4598, "measured_board": 263.4623},
    "K2": {"july_loo": 284.3195, "december": 205.2156, "january": 218.2582,
           "board": 257.1653, "measured_board": 263.0827},
    "refit-bulk": {"july_loo": 284.1774, "december": 204.9052, "january": 218.2115,
                   "board": 257.0601, "measured_board": None},
}
OOF = {"K2": E27 / "oof" / "exp-027-k2-k2-hold-{fold}.parquet",
       "refit-bulk": E27 / "oof" / "exp-027-k2-refit-bulk-{fold}.parquet"}


def _clean(vec: dict, ref: dict) -> dict:
    return {f: D.rmse_clean(ref[f]["y_true"], vec[f], ref[f]["MVT_ID_mvt"], f)
            for f in FOLDS}


def _gate(name: str, vec: dict, ref: dict) -> dict:
    got = _clean(vec, ref)
    off = {f: round(got[f] - PUBLISHED[name][f], 4) for f in FOLDS}
    if max(abs(v) for v in off.values()) > GATE:
        raise SystemExit(f"{name} does not reproduce its published rotation: {off}")
    b = Sc.P.board(got)
    if abs(b - PUBLISHED[name]["board"]) > GATE:
        raise SystemExit(f"{name} board {b:.4f} against {PUBLISHED[name]['board']:.4f}")
    return got


def eham_mask(preds: dict, ref: dict) -> dict:
    """v7's serving rule: every EHAM departure keeps stack-5's own value."""
    return {f: np.where(ref[f]["airport"].astype(str).to_numpy() == AIRPORT_HELD,
                        ref[f]["y_pred_stack5"].to_numpy("float64"), preds[f])
            for f in FOLDS}


def load() -> tuple[dict, dict, dict]:
    """`(reference panel, {name: vector per fold}, {name: clean RMSE per fold})`."""
    ref = Sc.reference()            # gates stack-5 and v6 on exp-028's numbers
    vec = {"stack-5": {f: ref[f]["y_pred_stack5"].to_numpy("float64") for f in FOLDS},
           "v6": {f: ref[f]["y_pred_v6"].to_numpy("float64") for f in FOLDS}}
    vec["v7"] = eham_mask(vec["v6"], ref)
    for name, path in OOF.items():
        out = {}
        for f in FOLDS:
            o = pd.read_parquet(str(path).format(fold=f))
            if not np.array_equal(o["MVT_ID_mvt"].to_numpy("float64"),
                                  ref[f]["MVT_ID_mvt"].to_numpy("float64")):
                raise SystemExit(f"{f}: {name} is not in the reference's row order")
            out[f] = o["y_pred"].to_numpy("float64")
        vec[name] = out
    clean = {n: _gate(n, vec[n], ref) for n in ("v7", "K2", "refit-bulk")}
    # Both served arms already hold EHAM at stack-5's own value, so the K2-minus-v7
    # change is exactly zero there; state it as a gate rather than as a sentence.
    for name in ("K2", "refit-bulk"):
        for f in FOLDS:
            m = ref[f]["airport"].astype(str).to_numpy() == AIRPORT_HELD
            d = float(np.abs(vec[name][f][m] - vec["v7"][f][m]).max())
            if d > VEC_TOL:
                raise SystemExit(f"{f}: {name} differs from v7 on EHAM by {d:g}")
    return ref, vec, clean


def against(rec: dict, preds: dict, ref: dict, base: dict, name: str,
            board: float | None) -> dict:
    """The arm's gain over one named base, with the same day-clustered intervals.

    `exp-027/code/candidate.against_v7` with the base named: the arms here are
    cuts on K2, the file the board holds, so K2 -- not v6 -- is what the gate and
    the prediction use.
    """
    per, ba = {}, {}
    k = name.replace("-", "_")
    rec[f"vs_{k}"], rec[f"ci_{k}"], rec[f"ci_{k}_no_noac"] = {}, {}, {}
    for f in FOLDS:
        d = ref[f]
        y = d["y_true"].to_numpy("float64")
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        pv = np.asarray(preds[f], "float64")
        per[f] = D.rmse_clean(y, pv, ids, f)
        ba[f] = D.rmse_clean(y, base[f], ids, f)
        rec[f"vs_{k}"][f] = round(ba[f] - per[f], 4)
        ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
        nn = ok & (d["regime"].to_numpy() != "noac")
        cl = Sc.P.clusters(d)
        rec[f"ci_{k}"][f] = [round(x, 4) for x in
                             Sc.P.boot_delta(y[ok], pv[ok], base[f][ok],
                                             pd.factorize(cl[ok])[0])]
        rec[f"ci_{k}_no_noac"][f] = [round(x, 4) for x in
                                     Sc.P.boot_delta(y[nn], pv[nn], base[f][nn],
                                                     pd.factorize(cl[nn])[0])]
    rec[f"clean_board_{k}"] = round(Sc.P.board(ba), 4)
    rec[f"gain_board_vs_{k}"] = round(Sc.P.board(ba) - Sc.P.board(per), 4)
    rec[f"worst_fold_vs_{k}"] = round(min(rec[f"vs_{k}"].values()), 4)
    if board is not None:
        rec[f"predicted_board_vs_{k}"] = round(board - rec[f"gain_board_vs_{k}"], 4)
    return rec


def show(rec: dict, name: str) -> None:
    k = name.replace("-", "_")
    print(f"  vs {name:10s} " + "  ".join(f"{f} {rec[f'vs_{k}'][f]:+9.4f}" for f in FOLDS)
          + f"   board {rec[f'gain_board_vs_{k}']:+9.4f}")
    for f in FOLDS:
        dl, lo, hi = rec[f"ci_{k}"][f]
        dn, lon, hin = rec[f"ci_{k}_no_noac"][f]
        print(f"    {f:9s} vs {name:10s} {-dl:+7.3f} [{-hi:+7.3f}, {-lo:+7.3f}]"
              f"   no-noac {-dn:+7.3f} [{-hin:+7.3f}, {-lon:+7.3f}]")
    if f"predicted_board_vs_{k}" in rec:
        print(f"  predicted board {rec[f'predicted_board_vs_{k}']:.4f} "
              f"({name} measured {PUBLISHED[name]['measured_board']}, at face)")
