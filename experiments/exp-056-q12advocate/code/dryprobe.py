"""exp-056-q12advocate: re-run one dry job with the current code into `work/probe` (git-ignored),
to measure its peak RSS and to check that the committed dry records reproduce byte for byte.

    .venv/bin/python experiments/exp-056-q12advocate/code/dryprobe.py freeze jul --dry
    .venv/bin/python experiments/exp-056-q12advocate/code/dryprobe.py fit jul --dry --scenario planted
    .venv/bin/python experiments/exp-056-q12advocate/code/dryprobe.py price --dry --scenario planted

Dry only: it refuses a run without --dry. It reads no label.
"""
import json
import resource
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402

if "--dry" not in sys.argv:
    raise SystemExit("dryprobe runs dry jobs only")
PROBE = E.EXP / "work/probe"
(PROBE / "analysis").mkdir(parents=True, exist_ok=True)
REAL_ANALYSIS = E.ANALYSIS
E.WORK = PROBE / "work"
E.ANALYSIS = PROBE / "analysis"
job = sys.argv[1]
if job in ("fit", "price"):
    for r in E.ORDER:
        src = REAL_ANALYSIS / f"dry{E.DRY_VERSION}-inputs-{r}.json"
        dst = E.ANALYSIS / src.name
        if not dst.exists():
            shutil.copy(src, dst)
mod = __import__(job)
sys.argv = [f"{job}.py"] + sys.argv[2:]
mod.main()
print(json.dumps({"job": job, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
