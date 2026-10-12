"""exp-035-k2climb: one Kaggle session of the K2 climb.

The body is exp-027-step's `kernels/f1-window` kernel (itself exp-025-cloud's
`f1-real` plus `PRC_WINDOW=1`) with one change: instead of one hard-coded fold
it runs the list of steps `make_kernel.py` wrote into `STEPS`, and it copies each
step's out-of-fold file to a step-specific name before the next step overwrites
`oof/<fold>.parquet`.

Two arm variables, and nothing else, differ from that frozen recipe:

  * `seed_start` -- `run.py --seed-start k --seeds 1` fits seed 7 + 13k alone,
    so one seed of a three-seed average costs one session rather than three.
    `seed_start 0, seeds 1` is the frozen path and still passes `seed=None`.
  * `scales` -- `PRC_WINDOW_SCALES`. Empty is the shipped 68-column block at
    (600, 3600) s; "300,600,1800,3600" is arm 2. `dataset.py`'s month cache
    carries the scale set in its key, so a wide run cannot silently reload a
    shipped-block table and report the shipped block's score as the arm's.

Outputs in /kaggle/working: one parquet per step (the fold OOF, or the ranking
prediction), the run's results json, and `port.json` with wall time, peak host
RSS, peak GPU memory and versions.
"""
import json
import os
import resource
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

ARM = 'phi-dj'
STEPS = [
    {
        "kind": "fold",
        "fold": "december",
        "tag": "k2p-december",
        "seed_start": 0,
        "scales": "",
        "extras": "phi",
        "rounds": 1200,
        "clip_z": 6000,
        "exact_rounds": 300,
        "ghost_mode": "mixbase",
        "main_model": "both",
        "threads": 4
    },
    {
        "kind": "fold",
        "fold": "january",
        "tag": "k2p-january",
        "seed_start": 0,
        "scales": "",
        "extras": "phi",
        "rounds": 1200,
        "clip_z": 6000,
        "exact_rounds": 300,
        "ghost_mode": "mixbase",
        "main_model": "both",
        "threads": 4
    }
]

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
COMMIT = (REPO / "GIT_COMMIT.txt").read_text().strip()
log("repo at", REPO, "commit", COMMIT[:80])

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

# ------------------------------------------------------------------ monitors
peak = {"gpu_mib": 0, "rss_gb": 0.0}
stop = threading.Event()
TOTAL_GB = int(next(l for l in open("/proc/meminfo")
                    if l.startswith("MemTotal")).split()[1]) / 2**20


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


th = threading.Thread(target=watch, daemon=True)
th.start()


def env_for(step):
    env = dict(os.environ)
    env["PRC_DATA_DIR"] = str(DATA)
    env["PYTHONPATH"] = str(REPO)
    env["OMP_NUM_THREADS"] = str(step["threads"])
    env["PRC_THREADS"] = str(step["threads"])
    env["PRC_WINDOW"] = "1"                     # the block is the member
    if step["scales"]:
        env["PRC_WINDOW_SCALES"] = step["scales"]
    else:
        env.pop("PRC_WINDOW_SCALES", None)
    # arm 3: exp-018b's module columns. Unset is the frozen frame.
    if step.get("extras"):
        env["PRC_EXTRAS"] = step["extras"]
    else:
        env.pop("PRC_EXTRAS", None)
    if SYNTH:
        env["PRC_SYNTHETIC"] = "1"
    return env


def run(cmd, label, env):
    banner(label)
    log("$", " ".join(cmd))
    log("  PRC_WINDOW_SCALES=" + env.get("PRC_WINDOW_SCALES", "<unset: shipped 600,3600>"))
    log("  PRC_EXTRAS=" + env.get("PRC_EXTRAS", "<unset: no module columns>"))
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
EXP = REPO / "experiments" / "exp-018-fresh"
steps = []
for st in STEPS:
    env = env_for(st)
    if st["kind"] == "fold":
        cmd = [PY, "experiments/exp-018-fresh/code/run.py",
               "--tag", st["tag"], "--folds", st["fold"],
               "--rounds", str(st["rounds"]), "--clip-z", str(st["clip_z"]),
               "--exact-rounds", str(st["exact_rounds"]),
               "--ghost-mode", st["ghost_mode"], "--main-model", st["main_model"],
               "--seeds", "1", "--seed-start", str(st["seed_start"]),
               "--params", json.dumps({"num_threads": st["threads"]}),
               "--save-oof"]
        rec = run(cmd, f"{st['tag']} (fold {st['fold']}, seed_start {st['seed_start']})", env)
        # copy out BEFORE the next step overwrites oof/<fold>.parquet
        src = EXP / "oof" / f"{st['fold']}.parquet"
        if not src.exists():
            raise SystemExit(f"{st['tag']}: {src} was not written")
        dst = WORK / f"{st['tag']}.parquet"
        shutil.copy(src, dst)
        log("wrote", dst, f"{dst.stat().st_size/2**20:.2f} MB")
        src.unlink()
    else:
        out = WORK / f"{st['tag']}.parquet"
        cmd = [PY, "experiments/exp-018-fresh/code/submit.py",
               "--out", str(out), "--rounds", str(st["rounds"]),
               "--clip-z", str(st["clip_z"]), "--exact-rounds", str(st["exact_rounds"]),
               "--ghost-mode", st["ghost_mode"], "--main-model", st["main_model"],
               "--seeds", str(st["seeds"]), "--seed-start", str(st["seed_start"]),
               "--params", json.dumps({"num_threads": st["threads"]})]
        rec = run(cmd, f"{st['tag']} (ranking, seeds {st['seeds']} from {st['seed_start']})", env)
        if not out.exists():
            raise SystemExit(f"{st['tag']}: {out} was not written")
        log("wrote", out, f"{out.stat().st_size/2**20:.2f} MB")
    for j in (EXP / "results" / f"{st['tag']}.json", EXP / "results" / "submission.json"):
        if j.exists():
            shutil.copy(j, WORK / f"{st['tag']}-{j.name}")
    steps.append(rec)

stop.set()
th.join(timeout=10)

banner("collect")
vers = {}
for m in ("numpy", "pandas", "pyarrow", "lightgbm", "catboost", "sklearn", "scipy"):
    try:
        vers[m] = getattr(__import__(m), "__version__", "?")
    except Exception as e:                                  # noqa: BLE001
        vers[m] = f"MISSING ({type(e).__name__})"
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

port = {
    "arm": ARM, "plan": STEPS, "synthetic": SYNTH, "data_dir": str(DATA),
    "data_files": {p.name: p.stat().st_size for p in sorted(DATA.iterdir()) if p.is_file()},
    "git_commit": COMMIT.splitlines()[0],
    "python": sys.version.split()[0], "versions": vers, "gpu": gpu,
    "cpu_count": os.cpu_count(), "mem_total_gb": round(TOTAL_GB, 1),
    "peak_gpu_mib": peak["gpu_mib"], "peak_host_used_gb": round(peak["rss_gb"], 2),
    "steps": steps,
}
(WORK / f"port-{ARM}.json").write_text(json.dumps(port, indent=1))
log(json.dumps({k: v for k, v in port.items() if k != "steps"}, indent=1))
for s in steps:
    log(f"{s['label']:44s} {s['wall_s']/60:7.1f} min  maxrss {s['child_maxrss_gb']:.2f} GB")
log(f"peak GPU {peak['gpu_mib']} MiB   peak host used {peak['rss_gb']:.2f} GB")
log("\nARM OK")
