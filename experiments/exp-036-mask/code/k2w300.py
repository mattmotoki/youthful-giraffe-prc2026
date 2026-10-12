"""Arm 3: K2 with a 300 s aggregate set beside the 600 s block, in one stacker.

exp-030 arm 1 measured this on v6's stacker: 600 s and 300 s side by side, 95
block columns instead of 50, worth **+0.0734 s** of clean board over v6 — real,
all three folds the right way, and under that assignment's 0.10 s gate.  K2 puts
a different member in F1-fresh's slot, and that member is itself a window model
(`kernels/f1-window*`), so the second resolution may read differently beside it:
either the K2 member has already taken what a second width was worth, or the two
resolutions are complements the stacker can now use.  `assignments/mask.md` asks
the question and this file answers it.

**Nothing is reimplemented.**  The stacker, the seeds, the rounds, the rotated
blend-5 anchor, `keep-noac` and the frames are `exp-027/code/candidate.py`'s; the
two-width block is `exp-030/code/blocks.py`'s, which is exp-028's own
`stack5w.block_for` called at each width and renamed; the scoring and the
day-clustered intervals are `exp-030/code/score.py`'s and `code/bases.py`'s.
This file chooses a block specification and does v7's EHAM arithmetic.

Four gates run and raise rather than warn, each a rotation recomputed here:

  A  blend-5 on the seven committed members reproduces `blend-config-5.json`.
  B  the no-window stacker lands on exp-026's published stack-5, to 1e-9 a row.
  C  the 600 s window arm lands on exp-028's published v6, to 1e-9 a row.
  D  **K2 at 600 s reproduces the K2 vector this experiment prices against** --
     `exp-027-step/oof/exp-027-k2-k2-hold-{fold}.parquet` -- to 1e-9 a row.  D is
     the one that binds this driver to the board's own file: without it the arm
     could be a two-width stacker measured against a K2 it does not reproduce.
  E  the arm's first 50 block columns are the 600 s block, name for name and
     value for value, so the arm strictly contains v6's block.

    PRC_THREADS=4 python k2w300.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-030-window2/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-027-step/code"))
sys.path.insert(0, str(HERE))
import blocks as B2  # noqa: E402
import candidate as C  # noqa: E402
import bases as Bs  # noqa: E402
import mask as M  # noqa: E402

Sc, SW, S, B, D, F = C.Sc, C.SW, C.S, C.B, C.D, C.F
FOLDS = C.FOLDS
SPEC600 = {"blocks": ["depx"], "win": 600.0}
SPECARM = {"blocks": ["depx", "depx_w300"], "win": 600.0}
VEC_TOL = 1e-9


def vec_gate(name: str, got: dict, want: dict) -> None:
    md = {f: float(np.abs(got[f] - want[f]).max()) for f in FOLDS}
    if max(md.values()) > VEC_TOL:
        raise SystemExit(f"gate {name} fails on the vector: {md}")
    print(f"[gate {name}] reproduces, max abs row difference "
          f"{max(md.values()):.3g}", flush=True)


def main() -> None:
    t0 = time.time()
    ref, vec, clean = Bs.load()
    print(f"[w300] bases gated: "
          + "; ".join(f"{n} board {Sc.P.board(c):.4f}" for n, c in clean.items()),
          flush=True)

    data = C.frames()
    a5 = C.anchors(data, C.NAMES5)
    for f in FOLDS:
        y = data[f]["y_true"].to_numpy("float64")
        d = float(np.sqrt(np.mean((y - a5[f]) ** 2))) - S.PUBLISHED[f]
        if abs(d) > C.GATE:
            raise SystemExit(f"gate A fails on {f}: {d:+.4f}")
    print("[gate A] blend-5 reproduces on all three folds", flush=True)
    ref5 = {f: np.maximum(a5[f], B.FLOOR) for f in FOLDS}

    pan = {f: SW.panel_of(data[f]) for f in FOLDS}
    blk600 = {f: SW.block_for(f, pan[f], SPEC600) for f in FOLDS}
    blkarm = {f: B2.block_for(f, pan[f], SPECARM) for f in FOLDS}
    for f in FOLDS:
        n = blk600[f].shape[1]
        head = blkarm[f].iloc[:, :n]
        if list(head.columns) != list(blk600[f].columns):
            raise SystemExit(f"gate E: {f} column names differ")
        if not np.array_equal(head.to_numpy("float64"), blk600[f].to_numpy("float64"),
                              equal_nan=True):
            raise SystemExit(f"gate E: {f} the arm's first {n} columns are not the block")
    print(f"[gate E] the arm's first {blk600[FOLDS[0]].shape[1]} columns are exp-028's "
          f"600 s block; the arm has {blkarm[FOLDS[0]].shape[1]}", flush=True)

    runs = {"stack-5": (C.NAMES5, None, "y_pred_stack5", "B"),
            "v6":      (C.NAMES5, blk600, "y_pred_v6", "C"),
            "k2-600":  (C.NAMESK, blk600, None, "D"),
            "k2-w300": (C.NAMESK, blkarm, None, None)}
    got = {}
    for lab, (names, bk, col, gate) in runs.items():
        t = time.time()
        p = C.align(C.rotate(data, names, a5, bk, ref5), data, ref)
        got[lab] = p
        cl = {f: D.rmse_clean(ref[f]["y_true"], p[f], ref[f]["MVT_ID_mvt"], f)
              for f in FOLDS}
        print(f"[{lab:8s}] " + "  ".join(f"{f} {cl[f]:9.4f}" for f in FOLDS)
              + f"  board {Sc.P.board(cl):9.4f}  {time.time()-t:.0f}s", flush=True)
        if col:
            vec_gate(gate, p, {f: ref[f][col].to_numpy("float64") for f in FOLDS})
    # gate D: the K2 vector this experiment prices against, rebuilt here
    vec_gate("D", Bs.eham_mask(got["k2-600"], ref), vec["K2"])

    pe = Bs.eham_mask(got["k2-w300"], ref)
    rec = M.arm("k2-w300", pe, ref, vec,
                note="K2 in F1-fresh's slot, blend-5 anchor, the 600 s block and a "
                     "300 s aggregate set beside it in one stacker, v7's EHAM mask; "
                     "noac held at blend-5's anchor")
    rec.update({"members": list(C.NAMESK), "anchor_members": list(C.NAMES5),
                "anchor": "blend-5 on the committed seven, rotated",
                "blocks": SPECARM["blocks"], "win": SPECARM["win"],
                "block_columns": int(blkarm[FOLDS[0]].shape[1]),
                "airport_excluded": "EHAM",
                "seeds": list(S.SEEDS), "rounds": S.ROUNDS, "threads": C.THREADS,
                "lightgbm": __import__("lightgbm").__version__,
                "gates": {"A_blend5": "pass", "B_stack5": "pass", "C_v6": "pass",
                          "D_k2": "pass", "E_block_prefix": "pass"},
                "wall_s": round(time.time() - t0, 1)})
    print(f"\n  written: {M.save(rec)}", flush=True)
    for f in FOLDS:
        pd.DataFrame({"MVT_ID_mvt": ref[f]["MVT_ID_mvt"].to_numpy("float64"),
                      "fold": f, "y_true": ref[f]["y_true"].to_numpy("float64"),
                      "y_pred": pe[f], "y_pred_nomask": got["k2-w300"][f]}).to_parquet(
            EXP / "oof" / f"exp-036-k2-w300-{f}.parquet", index=False)
    print(f"[w300] {rec['wall_s']:.0f}s wall", flush=True)


if __name__ == "__main__":
    main()
