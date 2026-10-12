"""exp-025-cloud: the F1-fresh member on one Kaggle fold, on the REAL data.

This is the port's acceptance test. The kernel is the full-scale one with the
submission step switched off and the data dataset switched to the private
`{os.environ['PRC_KAGGLE_DATA_SLUG']}`; it sets no `PRC_SYNTHETIC`, because the real directory has no
`SYNTHETIC.json` and the kernel keys off that file.

Ports `experiments/exp-018-fresh/code/run.py --tag fold4-final` unchanged. The
repo code arrives as a private Kaggle dataset (`source-code`), the raw
files as a second private dataset; nothing here edits the ported code, so a
difference against the committed OOF is a difference of environment, not of
implementation.

Comparison artifact: `experiments/exp-018-fresh/oof/july_loo.parquet`, which is
byte-identical to `experiments/exp-012-blend2/published/F1-fresh-s1-july_loo.parquet`,
the blend-4 member with weight 0.62.

Outputs in /kaggle/working: the fold OOF, the run's results json, the run log,
and `port.json` with wall time, peak host RSS, peak GPU memory and versions.
"""
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

# --------------------------------------------------------------------- config
CFG = {
    "fold": "july_loo",          # ranking-analogue fold, the expensive one (11 train months)
    "rounds": 1200,
    "clip_z": 6000,
    "exact_rounds": 300,
    "ghost_mode": "mixbase",
    "main_model": "both",        # LightGBM + CatBoost-GPU 50/50, as fold4-final
    "seeds": 1,
    "threads": 4,                # Kaggle gives 4 vCPU; the workstation run used 6
    "tag": "kaggle-f1-real",
    "also_submission": False,     # after the fold, fit 12 months and write a submission
    "submission_seeds": 1,
    # The submission step exists to measure the 12-month frame peak and to
    # exercise prc.data.write_submission at the real row count, not to price the
    # fit: its fit cost is the fold's times 12/11 by construction. Set these
    # lower than the fold's to buy that answer cheaply.
    "submission_rounds": 60,         # cheap: this step buys the 12-month memory peak, not a price
    "submission_main_model": "lgb",  # ditto; the CatBoost pool at 12 months is projected
}
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")


def log(*a):
    print(*a, flush=True)


def banner(t):
    log("\n" + "=" * 70 + f"\n== {t}\n" + "=" * 70)


# ------------------------------------------------------------------- the code
banner("inputs")
IN = Path("/kaggle/input")
if not IN.exists():
    raise SystemExit("/kaggle/input does not exist: no dataset is attached")
for d in sorted(IN.iterdir()):
    log(f"[{d.name}]")
    for i, q in enumerate(sorted(d.rglob("*"))):
        if i >= 40:
            log("    ...")
            break
        log(f"    {q.relative_to(d)}" + ("/" if q.is_dir() else f"  {q.stat().st_size}"))

banner("code")
# Kaggle may serve the archive verbatim or extract it server side; take whichever
# is there, and search the whole tree rather than assuming a mount depth.
tars = sorted(IN.rglob("prc-repo.tar.gz"))
if REPO.exists():
    shutil.rmtree(REPO)
if tars:
    log("untar", tars[0])
    with tarfile.open(tars[0]) as tf:
        tf.extractall("/tmp")           # the tarball's top directory is `repo`
else:
    trees = sorted(IN.rglob("prc/data.py"))
    if not trees:
        raise SystemExit("no code dataset found under /kaggle/input "
                         f"(looked for prc-repo.tar.gz and prc/data.py); "
                         f"inputs: {[d.name for d in IN.iterdir()]}")
    src = trees[0].parents[1]           # the directory that CONTAINS prc/
    log("copytree", src)
    shutil.copytree(src, REPO)
if not (REPO / "prc" / "data.py").exists():
    raise SystemExit(f"{REPO} has no prc/data.py after unpacking")
log("repo at", REPO, "commit", (REPO / "GIT_COMMIT.txt").read_text().strip()[:80])
log(sorted(p.name for p in REPO.iterdir()))

# ------------------------------------------------------------------- the data
banner("data")
cands = sorted({p.parent for p in IN.rglob("training_2025-01-01_2025-02-01.parquet")})
if not cands:
    raise SystemExit("no raw data dataset found under /kaggle/input "
                     f"(inputs: {[d.name for d in IN.iterdir()]})")
DATA = cands[0]
SYNTH = (DATA / "SYNTHETIC.json").exists()
log("data dir", DATA, "synthetic" if SYNTH else "REAL")
for p in sorted(DATA.iterdir()):
    if p.is_file():
        log(f"  {p.name:45s} {p.stat().st_size/2**20:8.1f} MB")
if SYNTH:
    log("\n*** SYNTHETIC INPUT: every number below is meaningless. It measures the\n"
        "*** code path, the runtime and the memory, and nothing else.\n")

env = dict(os.environ)
env["PRC_DATA_DIR"] = str(DATA)
env["PYTHONPATH"] = str(REPO)
env["OMP_NUM_THREADS"] = str(CFG["threads"])
env["PRC_THREADS"] = str(CFG["threads"])
if SYNTH:
    env["PRC_SYNTHETIC"] = "1"

# ------------------------------------------------------------------ monitors
peak = {"gpu_mib": 0, "rss_gb": 0.0}
stop = threading.Event()


def watch():
    while not stop.wait(5.0):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout
            peak["gpu_mib"] = max(peak["gpu_mib"], max(int(x) for x in out.split()))
        except Exception:                                   # noqa: BLE001
            pass
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable"):
                    avail = int(line.split()[1]) / 2**20
                    peak["rss_gb"] = max(peak["rss_gb"], TOTAL_GB - avail)
        except Exception:                                   # noqa: BLE001
            pass


TOTAL_GB = int(next(l for l in open("/proc/meminfo")
                    if l.startswith("MemTotal")).split()[1]) / 2**20
th = threading.Thread(target=watch, daemon=True)
th.start()


def run(cmd, label):
    banner(label)
    log("$", " ".join(cmd))
    t0 = time.time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    p = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    dt = time.time() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    log(p.stdout[-20000:])
    if p.stderr.strip():
        log("--- stderr ---")
        log(p.stderr[-8000:])
    log(f"[{label}] rc={p.returncode} wall={dt/60:.1f} min "
        f"child_maxrss={max(after, before)/2**20:.2f} GB")
    if p.returncode != 0:
        raise SystemExit(f"{label} failed with rc={p.returncode}")
    return {"label": label, "wall_s": round(dt, 1),
            "child_maxrss_gb": round(max(after, before) / 2**20, 2),
            "stdout_tail": p.stdout[-4000:]}


PY = sys.executable
steps = []
steps.append(run([PY, "experiments/exp-018-fresh/code/run.py",
                  "--tag", CFG["tag"], "--folds", CFG["fold"],
                  "--rounds", str(CFG["rounds"]), "--clip-z", str(CFG["clip_z"]),
                  "--exact-rounds", str(CFG["exact_rounds"]),
                  "--ghost-mode", CFG["ghost_mode"], "--main-model", CFG["main_model"],
                  "--seeds", str(CFG["seeds"]),
                  "--params", json.dumps({"num_threads": CFG["threads"]}),
                  "--save-oof"], f"fold {CFG['fold']}"))

if CFG["also_submission"]:
    steps.append(run([PY, "experiments/exp-018-fresh/code/submit.py",
                      "--out", str(WORK / "submission.parquet"),
                      "--rounds", str(CFG["submission_rounds"] or CFG["rounds"]),
                      "--clip-z", str(CFG["clip_z"]),
                      "--exact-rounds", str(CFG["exact_rounds"]),
                      "--ghost-mode", CFG["ghost_mode"],
                      "--main-model", CFG["submission_main_model"] or CFG["main_model"],
                      "--seeds", str(CFG["submission_seeds"]),
                      "--params", json.dumps({"num_threads": CFG["threads"]})],
                     "submission (12 months)"))

stop.set()
th.join(timeout=10)

# ------------------------------------------------------------------- collect
banner("collect")
EXP = REPO / "experiments" / "exp-018-fresh"
for src in [EXP / "oof" / f"{CFG['fold']}.parquet",
            EXP / "results" / f"{CFG['tag']}.json",
            EXP / "results" / "submission.json",
            EXP / "climb.log"]:
    if src.exists():
        dst = WORK / f"{CFG['tag']}-{src.name}"
        shutil.copy(src, dst)
        log("wrote", dst, f"{dst.stat().st_size/2**20:.2f} MB")

vers = {}
for m in ("numpy", "pandas", "pyarrow", "lightgbm", "catboost", "sklearn", "scipy"):
    try:
        vers[m] = getattr(__import__(m), "__version__", "?")
    except Exception as e:                                  # noqa: BLE001
        vers[m] = f"MISSING ({type(e).__name__})"
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

port = {
    "cfg": CFG, "synthetic": SYNTH, "data_dir": str(DATA),
    "data_files": {p.name: p.stat().st_size for p in sorted(DATA.iterdir()) if p.is_file()},
    "git_commit": (REPO / "GIT_COMMIT.txt").read_text().splitlines()[0],
    "python": sys.version.split()[0], "versions": vers, "gpu": gpu,
    "cpu_count": os.cpu_count(), "mem_total_gb": round(TOTAL_GB, 1),
    "peak_gpu_mib": peak["gpu_mib"], "peak_host_used_gb": round(peak["rss_gb"], 2),
    "steps": steps,
}
(WORK / "port.json").write_text(json.dumps(port, indent=1))
log(json.dumps({k: v for k, v in port.items() if k != "steps"}, indent=1))
for s in steps:
    log(f"{s['label']:28s} {s['wall_s']/60:7.1f} min  maxrss {s['child_maxrss_gb']:.2f} GB")
log(f"peak GPU {peak['gpu_mib']} MiB   peak host used {peak['rss_gb']:.2f} GB")
log("\nPORT OK")
