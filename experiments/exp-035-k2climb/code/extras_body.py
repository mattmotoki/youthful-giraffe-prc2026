"""exp-035-k2climb arm 3: build the module columns, on a CPU session.

Arm 3 is "exp-018b's remaining climb axis, applied to the K2 frame". Both
records the assignment cites converge on the same columns:
`experiments/exp-018-fresh/COMPARISON.md` §4 leaves `phi` under this lineage's
heads as its last untested idea ("one refit"), and `experiments/exp-018b-climb`
never ran its step-2 module screen, whose own table expects `phi` to be "the
largest of the four".

`experiments/exp-018b-climb/code/extras.py` already builds all of it -- `phi`
cross-fitted per fold, `surface` VERSION 2 minus the fifteen columns this
lineage already covers, and `deice` -- and caches it per fold. That build is
**CPU-only**, so it runs here on a CPU session and costs nothing against the
GPU budget the assignment caps. Nothing in this kernel fits a model.

Output: one parquet per fold side, plus the ranking pair, in `/kaggle/working`.
"""
import json
import os
import resource
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

FOLDS = ["july_loo", "december", "january"]
WITH_RANK = True
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")


def log(*a):
    print(*a, flush=True)


log("== inputs")
IN = Path("/kaggle/input")
for d in sorted(IN.iterdir()):
    log(f"[{d.name}]")

tars = sorted(IN.rglob("prc-repo.tar.gz"))
if REPO.exists():
    shutil.rmtree(REPO)
if tars:
    with tarfile.open(tars[0]) as tf:
        tf.extractall("/tmp")
else:
    trees = sorted(IN.rglob("prc/data.py"))
    if not trees:
        raise SystemExit("no code dataset found under /kaggle/input")
    shutil.copytree(trees[0].parents[1], REPO)
if not (REPO / "prc" / "data.py").exists():
    raise SystemExit(f"{REPO} has no prc/data.py")
COMMIT = (REPO / "GIT_COMMIT.txt").read_text().strip()
log("repo commit", COMMIT[:60])

cands = sorted({p.parent for p in IN.rglob("training_2025-01-01_2025-02-01.parquet")})
if not cands:
    raise SystemExit("no raw data dataset found under /kaggle/input")
DATA = cands[0]
if (DATA / "SYNTHETIC.json").exists():
    raise SystemExit("this kernel refuses synthetic input: its output is a feature cache")
log("data dir", DATA)

env = dict(os.environ)
env["PRC_DATA_DIR"] = str(DATA)
env["PYTHONPATH"] = str(REPO)
env["OMP_NUM_THREADS"] = "4"
env["PRC_THREADS"] = "4"

steps = []
args = FOLDS + (["rank"] if WITH_RANK else [])
for what in args:
    t0 = time.time()
    log("\n" + "=" * 70 + f"\n== extras {what}\n" + "=" * 70)
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(REPO / 'experiments/exp-018b-climb/code')!r})\n"
        "import extras as E\n"
        f"what = {what!r}\n"
        "t = time.time()\n"
        "tr, va = (E.build_rank() if what == 'rank' else E.build(what))\n"
        "print('SHAPES', what, tr.shape, va.shape, round(time.time()-t, 1))\n"
    )
    p = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO),
                       capture_output=True, text=True)
    log(p.stdout[-12000:])
    if p.stderr.strip():
        log("--- stderr ---")
        log(p.stderr[-6000:])
    dt = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    log(f"[{what}] rc={p.returncode} wall={dt/60:.1f} min maxrss={rss:.2f} GB")
    if p.returncode != 0:
        raise SystemExit(f"extras {what} failed with rc={p.returncode}")
    steps.append({"what": what, "wall_s": round(dt, 1), "maxrss_gb": round(rss, 2)})

CACHE = REPO / "artifacts" / "exp-018b" / "extras"
if not CACHE.exists():
    raise SystemExit(f"{CACHE} was not written")
copied = []
for q in sorted(CACHE.glob("*.parquet")):
    d = WORK / q.name
    shutil.copy(q, d)
    copied.append({"name": q.name, "bytes": d.stat().st_size})
    log("wrote", d.name, f"{d.stat().st_size/2**20:.2f} MB")

(WORK / "extras.json").write_text(json.dumps(
    {"git_commit": COMMIT.splitlines()[0], "folds": args, "steps": steps,
     "files": copied, "python": sys.version.split()[0],
     "cpu_count": os.cpu_count()}, indent=1))
log("\nEXTRAS OK")
