"""One arm's rotation: exp-028's own stacker, with a multi-width block.

This script **does not reimplement the stacker.**  It rebinds three names in
`exp-028/code/stack5w.py` and then calls that module's own `main()`:

    stack5w.block_for  -> blocks.block_for   (a block name may carry its width)
    stack5w.WIN        -> the arm's default width
    stack5w.VARIANTS   -> the arm's one variant
    stack5w.TAG        -> the arm's name, which names every artifact
    panel.EXP          -> this experiment's directory, so the artifacts land here

Everything that decides a number — the five seeds, the 300 rounds,
`stack5.LGB_BASE`, the rotated blend-5 hybrid NNLS anchor, `keep-noac`, the
clean/no-noac day-clustered bootstrap, the two gates (blend-5 must reproduce on
all three folds, and the no-window rotation must land on exp-026's published
stack-5 within 1e-3, or the run aborts) — is exp-028's, unchanged.  So is the
naming of the artifacts, which is why they are called `exp-028-stack5w-<arm>-*`:
the script that wrote them is exp-028's.

    PRC_THREADS=4 python rotate.py --arm w600w300 --blocks depx+depx_w300 --win 600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
W28 = ROOT / "experiments/exp-028-window/code"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(W28))
import blocks as B2  # noqa: E402
import panel as P  # noqa: E402
import stack5w as SW  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="names every artifact this run writes")
    ap.add_argument("--blocks", required=True, help="e.g. depx+depx_w300")
    ap.add_argument("--win", type=float, default=600.0,
                    help="the width a block name without its own `_w<n>` suffix takes")
    a = ap.parse_args()

    P.EXP = EXP
    SW.P.EXP = EXP
    SW.block_for = B2.block_for
    SW.WIN = a.win
    SW.TAG = a.arm
    SW.VARIANTS = [a.blocks]
    print(f"[rotate] arm={a.arm} blocks={a.blocks} default_win={a.win:.0f}s "
          f"threads={SW.THREADS} -> {EXP.relative_to(ROOT)}", flush=True)
    climb = EXP / "climb.log"
    before = climb.read_text() if climb.exists() else ""
    SW.main()
    # `stack5w.main` appends its own trace line to `climb.log` in exp-028's format.
    # `climb.log` here has to be COMMON.md's format only (Sol round 1), and the
    # line score.py writes is the one that carries the fold RMSEs and the
    # gap-closed fraction, so the rotation's trace is moved to its own file.
    after = climb.read_text() if climb.exists() else ""
    if after.startswith(before) and after != before:
        with open(EXP / "logs" / "rotation-trace.log", "a") as fh:
            fh.write(after[len(before):])
        climb.write_text(before)


if __name__ == "__main__":
    main()
