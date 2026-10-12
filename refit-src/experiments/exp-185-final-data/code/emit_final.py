#!/usr/bin/env python3
"""exp-185: run-only wrapper of exp-047 code/emit.py for the final mode.

block.py check_mode3 asserts the turnaround count of 2026-01 and 2026-07 against the counts on
ranking.parquet (1,382 and 1,797). The final file has other arrival rows, so in the final mode this
wrapper empties that table: every month then takes the guard of the other months (a count of 0 stops
the run) and the count is recorded in interface-final.json. Nothing else changes.

  PRC_RANKING_MODE=final PRC_SRC=<snapshot> python emit_final.py <months> --sources ... --rule ...
"""
import os
import runpy
import sys
from pathlib import Path

assert os.environ.get("PRC_RANKING_MODE") == "final", "emit_final.py runs in the final mode only"
ROOT = Path(os.environ.get("PRC_SRC") or Path(__file__).resolve().parents[3]).resolve()
CODE = ROOT / "experiments/exp-047-queue/code"
sys.path.insert(0, str(CODE))
import block as BK  # noqa: E402

assert Path(BK.__file__).resolve() == (CODE / "block.py").resolve(), BK.__file__
BK.MODE3_REFERENCE = {}
sys.argv = [str(CODE / "emit.py")] + sys.argv[1:]
runpy.run_path(str(CODE / "emit.py"), run_name="__main__")
