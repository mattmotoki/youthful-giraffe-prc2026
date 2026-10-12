"""exp-025-cloud: one stack-4 member's ranking prediction, rebuilt on Kaggle.

`prc-f1-fold` ports the largest member (F1-fresh, weight 0.62). This kernel
ports the other five, one per push, by calling each member's own producer with
the arguments the release config declares
(`experiments/exp-015-release/config/blend-3.json`,
`experiments/exp-012-blend2/blend-config-4.json`) and nothing else. Change
`CFG["member"]` and the slug in `kernel-metadata.json`; the table below is the
only place a member's command lives.

Why this has to run at all: **two of stack-4's six ranking predictions are
lost** -- `T-tailv2` (`e9d97994…`) and `L2-hub` (`29adfc27…`) lived only under
the git-ignored `artifacts/`, and the workstation is gone. The other four match
committed files under `experiments/exp-012-blend2/published/` by hash, and every
member out-of-fold file is committed, so the blend weights and the stacker
stand. Both lost members are CPU-only, so restoring the head spends no GPU
quota. The other three producers are here because a new member set or a re-fit
needs them.

Two portability edits the table carries, both in the environment and neither in
the ported code:

* `PRC_GPU=0` for `C2-cat`. `exp-008-diversity/learners.py:21` defaults it to
  `1`, which is the workstation's second card; a Kaggle kernel has only GPU 0.
* `PRC_THREADS` set explicitly everywhere, because `prc.nmtail` hardcodes
  `num_threads = 8` and this box has four vCPU.
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

CFG = {"member": "N1-nmtail", "threads": 4}

#: name -> what the release config says. `pre` runs first (a cache the member
#: reads but does not build); `out` is copied to /kaggle/working afterwards.
MEMBERS = {
    "L2-hub": {
        "weight": 0.154, "gpu": False,
        "cmd": ["experiments/exp-008-diversity/run.py", "--submit", "--kind", "lgb",
                "--name", "L2-hub", "--variant", "plain", "--huber-delta", "1200",
                "--rounds-m6", "3225", "--rounds-m11", "2100"],
        "cwd": "experiments/exp-008-diversity", "env": {},
        "out": ["artifacts/exp-008/pred/L2-hub-ranking.parquet",
                "experiments/exp-008-diversity/submit-L2-hub.json"],
    },
    "T-tailv2": {
        "weight": 0.121, "gpu": False,
        "cmd": ["experiments/exp-008-diversity/tail_ranking.py"],
        "cwd": ".", "env": {},
        "out": ["artifacts/exp-012/pred/T-tailv2-ranking.parquet"],
    },
    "D1-winter": {
        "weight": 0.068, "gpu": False,
        "cmd": ["experiments/exp-012-blend2/winter_arm.py", "--name", "D1-winter-deice-v1",
                "--submit", "--seeds", "7,107,207", "--ds-columns", "deice"],
        "cwd": "experiments/exp-012-blend2", "env": {},
        "out": ["artifacts/exp-012/pred/D1-winter-deice-v1-ranking.parquet"],
    },
    "C2-cat": {
        "weight": 0.037, "gpu": True,
        "cmd": ["experiments/exp-008-diversity/run.py", "--submit", "--kind", "cat",
                "--name", "C2-d8p", "--depth", "8", "--rounds-m6", "3625",
                "--rounds-m11", "5325"],
        "cwd": "experiments/exp-008-diversity",
        "env": {"PRC_GPU": "0"},            # the workstation's default of 1 does not exist here
        "out": ["artifacts/exp-008/pred/C2-d8p-ranking.parquet",
                "experiments/exp-008-diversity/submit-C2-d8p.json"],
    },
    "N1-nmtail": {
        "weight": 0.253, "gpu": False,      # inside the nm_late regime only; 0.000 elsewhere
        # build_cache.py writes artifacts/exp-003-anatomy/mvt_all.parquet, which
        # build_reocc.py reads; the per-month feat/{m}.parquet frames that
        # nmtail_submit.py reads come from features.py's build_all() instead.
        # Getting that wrong cost one failed push.
        "pre": [["experiments/exp-003-anatomy/code/build_cache.py"],
                ["experiments/exp-003-anatomy/code/features.py"],
                ["experiments/exp-010-nmtail/code/build_reocc.py"]],
        "cmd": ["experiments/exp-012-blend2/nmtail_submit.py"],
        "cwd": "experiments/exp-012-blend2", "env": {},
        "out": ["artifacts/exp-012/pred/N1-nmtail-ranking.parquet",
                "experiments/exp-012-blend2/submit-N1-nmtail.json"],
    },
}
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")
IN = Path("/kaggle/input")


def log(*a):
    print(*a, flush=True)


def banner(t):
    log("\n" + "=" * 70 + f"\n== {t}\n" + "=" * 70)


M = MEMBERS[CFG["member"]]

banner(f"member {CFG['member']} (blend-4 weight {M['weight']})")
tars = sorted(IN.rglob("prc-repo.tar.gz"))
if REPO.exists():
    shutil.rmtree(REPO)
if tars:
    with tarfile.open(tars[0]) as tf:
        tf.extractall("/tmp")
else:
    trees = sorted(IN.rglob("prc/data.py"))
    if not trees:
        raise SystemExit(f"no code dataset; inputs {[d.name for d in IN.iterdir()]}")
    shutil.copytree(trees[0].parents[1], REPO)
if not (REPO / "prc" / "data.py").exists():
    raise SystemExit(f"{REPO} has no prc/data.py")
log("repo", REPO, (REPO / "GIT_COMMIT.txt").read_text().splitlines()[0])

cands = sorted({p.parent for p in IN.rglob("training_2025-01-01_2025-02-01.parquet")})
if not cands:
    raise SystemExit(f"no raw data; inputs {[d.name for d in IN.iterdir()]}")
DATA = cands[0]
SYNTH = (DATA / "SYNTHETIC.json").exists()
log("data", DATA, "synthetic" if SYNTH else "REAL")
if SYNTH:
    log("\n*** SYNTHETIC INPUT: this run prices the member and proves its code path.\n"
        "*** Its predictions are not a member and must never reach a blend.\n")

env = dict(os.environ)
env["PRC_DATA_DIR"] = str(DATA)
env["PYTHONPATH"] = str(REPO)
env["OMP_NUM_THREADS"] = str(CFG["threads"])
env["PRC_THREADS"] = str(CFG["threads"])
if SYNTH:
    env["PRC_SYNTHETIC"] = "1"
env.update(M.get("env", {}))

TOTAL_GB = int(next(l for l in open("/proc/meminfo")
                    if l.startswith("MemTotal")).split()[1]) / 2**20
peak = {"gpu_mib": 0, "used_gb": 0.0}
stop = threading.Event()


def watch():
    while not stop.wait(10.0):
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30).stdout
            peak["gpu_mib"] = max(peak["gpu_mib"], max(int(x) for x in o.split()))
        except Exception:                                   # noqa: BLE001
            pass
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable"):
                    peak["used_gb"] = max(peak["used_gb"],
                                          TOTAL_GB - int(line.split()[1]) / 2**20)
        except Exception:                                   # noqa: BLE001
            pass


th = threading.Thread(target=watch, daemon=True)
th.start()
steps = []


def run(script_and_args, cwd, label):
    banner(label)
    # The script path is repo-relative and `cwd` is the producer's own working
    # directory, so the two must not be joined: resolve the script against REPO.
    cmd = [sys.executable, str(REPO / script_and_args[0])] + script_and_args[1:]
    log("$ cd", cwd, "&&", " ".join(cmd))
    t0 = time.time()
    rss0 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    p = subprocess.run(cmd, env=env, cwd=str(REPO / cwd), capture_output=True, text=True)
    dt = time.time() - t0
    # ru_maxrss over children is a high-water mark, so this is the peak of the
    # largest child so far, not of this step alone. Kilobytes on Linux.
    rss = max(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, rss0) / 2**20
    log(p.stdout[-20000:])
    if p.stderr.strip():
        log("--- stderr ---")
        log(p.stderr[-8000:])
    log(f"[{label}] rc={p.returncode} wall={dt/60:.1f} min child_maxrss={rss:.2f} GB")
    steps.append({"label": label, "cmd": " ".join(script_and_args), "cwd": cwd,
                  "rc": p.returncode, "wall_s": round(dt, 1),
                  "child_maxrss_gb": round(rss, 2),
                  "stdout_tail": p.stdout[-4000:],
                  "stderr_tail": p.stderr[-2000:] if p.returncode else ""})
    if p.returncode != 0:
        raise SystemExit(f"{label} failed rc={p.returncode}")


for pre in M.get("pre", []):
    run(pre, ".", f"pre: {pre[0]}")
run(M["cmd"], M["cwd"], f"member {CFG['member']}")

stop.set()
th.join(timeout=10)

banner("collect")
collected = {}
for rel in M["out"]:
    src = REPO / rel
    if src.exists():
        dst = WORK / f"{CFG['member']}--{src.name}"
        shutil.copy(src, dst)
        collected[rel] = {"bytes": src.stat().st_size, "as": dst.name}
        log("wrote", dst.name, f"{src.stat().st_size/2**20:.2f} MB")
    else:
        collected[rel] = "MISSING"
        log("MISSING", rel)

vers = {}
for mod in ("numpy", "pandas", "pyarrow", "lightgbm", "catboost"):
    try:
        vers[mod] = getattr(__import__(mod), "__version__", "?")
    except Exception as e:                                  # noqa: BLE001
        vers[mod] = f"MISSING ({type(e).__name__})"
try:  # a CPU-only kernel has no nvidia-smi at all, and four of the five members are CPU-only
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    gpu = "none (CPU kernel)"
out = {"member": CFG["member"], "weight": M["weight"], "synthetic": SYNTH,
       "git_commit": (REPO / "GIT_COMMIT.txt").read_text().splitlines()[0],
       "python": sys.version.split()[0], "versions": vers, "gpu": gpu,
       "threads": CFG["threads"], "env_overrides": M.get("env", {}),
       "cpu_count": os.cpu_count(), "mem_total_gb": round(TOTAL_GB, 1),
       "peak_gpu_mib": peak["gpu_mib"], "peak_host_used_gb": round(peak["used_gb"], 2),
       "steps": steps, "collected": collected}
(WORK / f"member-{CFG['member']}.json").write_text(json.dumps(out, indent=1))
log(json.dumps({k: v for k, v in out.items() if k != "steps"}, indent=1))
log(f"\ntotal {sum(s['wall_s'] for s in steps)/60:.1f} min, peak host "
    f"{peak['used_gb']:.2f} GB, peak GPU {peak['gpu_mib']} MiB")
log("\nMEMBER OK")
