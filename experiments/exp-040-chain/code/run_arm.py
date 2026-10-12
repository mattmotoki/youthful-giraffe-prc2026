"""Run `exp-008-diversity/run.py` unedited, on the arm's frame.

    PRC_CHAIN_ARM=chain PRC_THREADS=4 python run_arm.py \
        --kind lgb --name L2c-chain --variant plain --huber-delta 1200 \
        --folds july_loo --rounds-m6 3225 --rounds-m11 8725 --outdir <dir>

Every argument after this file's name goes to `run.py`'s own parser, so the
member's command line is the one `docs/CLOUD-COMPUTE.md` and
`experiments/exp-025-cloud/kernels/nested/prc-nested.py` already record for
`L2-hub8725`, with nothing added.  The only change is the module named `frames`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import frames_chain  # noqa: E402

# `run.py` does `import frames`; give it this one before it can find another.
sys.modules["frames"] = frames_chain

sys.path.insert(0, str(ROOT / "experiments/exp-008-diversity"))
import run as R  # noqa: E402

if R.frames is not frames_chain:
    raise SystemExit("run.py did not take the chain frame")
print(f"[arm] PRC_CHAIN_ARM={os.environ.get('PRC_CHAIN_ARM', 'none')} "
      f"frame {frames_chain.VERSION} adds {frames_chain.NEW}", flush=True)

if __name__ == "__main__":
    R.main()
