"""The environment control: is the K2 arm's july_loo gain the block or Kaggle?

Six of the seven member columns in the rotation are workstation artifacts.  The
K2 column is not: it was fitted on Kaggle, and exp-025 measured **0.275 s** of
july_loo member RMSE between the two machines on the same bytes and the same
code (`experiments/exp-025-cloud/RESULTS.md`).  So `candidate.py`'s K2 arm
carries a machine term as well as the window block, and on july_loo the machine
term points the wrong way -- the committed F1-fresh reads 293.6500 and the same
model on Kaggle reads 293.9248 -- which makes the rotation's gain a **lower
bound**, not an inflated one.

This script measures that term at the head, on the one fold where both Kaggle
vectors exist.  **Only july_loo is a control**: the December and January numbers
these two runs print are not clean reads, because their stackers train on the
changed july_loo column.  Two runs of `candidate.py`, identical in every other respect:

  `f1kag`  july_loo's member column is cloud's Kaggle F1-fresh acceptance fit
           (`exp-025-cloud/oof-nested/F1-fresh--july_loo-acceptance.parquet`,
           293.9248, no window block)
  `k2kag`  july_loo's member column is K2 (293.3992, block on)

December and January keep the committed workstation F1-fresh in both runs, so
the stacker fitted for fold july_loo -- which trains on December and January
only -- is the same model in both.  The difference on july_loo is then the
apply-side member column and nothing else: **both sides Kaggle, one with the
block and one without**.

`k2kag` minus `f1kag` on july_loo is the block's head-level worth, free of the
machine.  `k2kag` minus v6 is what `candidate.py` reports, which carries it --
and it carries it as an **inflation**, not as a discount: the Kaggle F1-fresh
head reads 284.7151 against the workstation one's 284.7219, so the mixed
comparison is 0.0068 s better than the controlled one.  (The first version of
this docstring had that sign backwards; Sol round 3, non-blocking 3.)

    PRC_THREADS=3 python envctl.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
WORK = EXP / "work" / "envctl"
FOLDS = ("july_loo", "december", "january")
WS = ROOT / "experiments/exp-012-blend2/published/F1-fresh-s1-{fold}.parquet"
ARMS = {
    "f1kag": {"july_loo": ROOT / "experiments/exp-025-cloud/oof-nested"
                                 "/F1-fresh--july_loo-acceptance.parquet"},
    "k2kag": {"july_loo": EXP / "oof" / "k2-window-july_loo.parquet"},
}


def stage(arm: str, over: dict) -> str:
    d = WORK / arm
    d.mkdir(parents=True, exist_ok=True)
    for f in FOLDS:
        src = over.get(f, Path(str(WS).format(fold=f)))
        if not src.exists():
            raise SystemExit(f"{arm}/{f}: {src} is missing")
        shutil.copy(src, d / f"sub-{f}.parquet")
    return str((d / "sub-{fold}.parquet").relative_to(ROOT))


def main() -> None:
    for arm, over in ARMS.items():
        tpl = stage(arm, over)
        env = dict(os.environ, PRC_K2=tpl, PRC_TAG=arm)
        print(f"\n########## {arm}: {tpl}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / "candidate.py")], env=env, cwd=str(ROOT))
        if r.returncode != 0:
            raise SystemExit(f"{arm} failed with rc={r.returncode}")


if __name__ == "__main__":
    main()
