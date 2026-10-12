"""exp-040-chain step 6: the `C2-cat` member refitted with and without the stand chain.

`assignments/q09.md` step 6 authorises this only when step 4 on `L2-hub8725` has
`A > 0` on all three folds.  It does: A = 65.64 / 51.78 / 71.75 on
july_loo / december / january (`analysis/q09-table.csv`, arm `chain d11-noac`
against v10).  `C2-cat` is the second and last stack-5 member built on retro
frame v2, so it is the second and last place in the shipped stack where adding
the chain is not a no-op.

**GPU budget.** The assignment caps this at 2 GPU-h and the desk adds one session
at a time, in a slot k2climb leaves free.  A paired fold costs about 0.8 GPU-h
(two arms at about 0.4 each), so this kernel runs ONE fold and its runtime is
measured rather than assumed before any further fold is asked for.

**There is no same-environment baseline for this member.** exp-025's
`oof-nested/` holds `F1-fresh`, `T-tailv4` and `L2-hub8725`, not `C2-cat`, and
exp-025 §10.7 places the whole machine term of that experiment in
CatBoost-on-GPU -- which is this member's path.  So the workstation's published
fold RMSE is printed as an ORIENTATION and is not a gate; the gate is the paired
difference between two arms fitted in this session.

Two arms, one member, the producer command `docs/CLOUD-COMPUTE.md` records for
`C2-cat` (`experiments/exp-025-cloud/kernels/member/prc-member.py`), with
`PRC_GPU=0` because a Kaggle kernel has only GPU 0.
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

CFG = {
    "folds": ["july_loo"],
    "arms": ["none", "chain"],
    "threads": 4,
    "tag": "c2",
}


def cmd_for(fold, out):
    return ["experiments/exp-040-chain/code/run_arm.py", "--kind", "cat",
            "--name", "C2-d8p", "--depth", "8", "--folds", fold,
            "--rounds-m6", "3625", "--rounds-m11", "5325",
            "--outdir", str(out)]


#: The workstation's published fold RMSE (`experiments/exp-008-diversity/
#: result-C2-d8p.json`).  Orientation only: see the module docstring.
PUB_RMSE = {"july_loo": 315.168216, "december": 222.941722, "january": 348.253718}

WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")
IN = Path("/kaggle/input")


def log(*a):
    print(*a, flush=True)


def banner(t):
    log("\n" + "=" * 70 + f"\n== {t}\n" + "=" * 70)


banner("inputs")
if not IN.exists():
    raise SystemExit("/kaggle/input does not exist: no dataset is attached")
for d in sorted(IN.iterdir()):
    log(f"[{d.name}]")

banner("code")
tars = sorted(IN.rglob("prc-repo.tar.gz"))
if REPO.exists():
    shutil.rmtree(REPO)
if tars:
    log("untar", tars[0])
    with tarfile.open(tars[0]) as tf:
        tf.extractall("/tmp")
else:
    trees = sorted(IN.rglob("prc/data.py"))
    if not trees:
        raise SystemExit(f"no code dataset; inputs {[d.name for d in IN.iterdir()]}")
    shutil.copytree(trees[0].parents[1], REPO)
for need in ("prc/data.py", "experiments/exp-040-chain/code/run_arm.py",
             "experiments/exp-037-tail/code/chain.py",
             "experiments/exp-008-diversity/run.py"):
    if not (REPO / need).exists():
        raise SystemExit(f"{REPO} has no {need}")
log("repo at", REPO, "commit", (REPO / "GIT_COMMIT.txt").read_text().splitlines()[0])

banner("data")
cands = sorted({p.parent for p in IN.rglob("training_2025-01-01_2025-02-01.parquet")})
if not cands:
    raise SystemExit(f"no raw data; inputs {[d.name for d in IN.iterdir()]}")
DATA = cands[0]
SYNTH = (DATA / "SYNTHETIC.json").exists()
log("data dir", DATA, "synthetic" if SYNTH else "REAL")

env = dict(os.environ)
env["PRC_DATA_DIR"] = str(DATA)
env["PYTHONPATH"] = str(REPO)
env["OMP_NUM_THREADS"] = str(CFG["threads"])
env["PRC_THREADS"] = str(CFG["threads"])
env["PRC_GPU"] = "0"                    # a Kaggle kernel has only GPU 0
if SYNTH:
    env["PRC_SYNTHETIC"] = "1"

TOTAL_GB = int(next(l for l in open("/proc/meminfo")
                    if l.startswith("MemTotal")).split()[1]) / 2**20
peak = {"gpu_mib": 0, "rss_gb": 0.0}
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
                    peak["rss_gb"] = max(peak["rss_gb"],
                                         TOTAL_GB - int(line.split()[1]) / 2**20)
        except Exception:                                   # noqa: BLE001
            pass


th = threading.Thread(target=watch, daemon=True)
th.start()


def run(cmd, label, arm, cwd="."):
    banner(label)
    e = dict(env, PRC_CHAIN_ARM=arm)
    log("$ PRC_CHAIN_ARM=" + arm, "PRC_GPU=0 cd", cwd, "&&", " ".join(cmd))
    t0 = time.time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    p = subprocess.run(cmd, env=e, cwd=str(REPO / cwd), capture_output=True, text=True)
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
    return {"label": label, "arm": arm, "wall_s": round(dt, 1),
            "child_maxrss_gb": round(max(after, before) / 2**20, 2)}


PY_EXE = sys.executable
OUT = REPO / "experiments" / "exp-040-chain" / "oof"
OUT.mkdir(parents=True, exist_ok=True)
steps, written = [], {}

import pandas as pd  # noqa: E402

for arm in CFG["arms"]:
    for fold in CFG["folds"]:
        args = cmd_for(fold, OUT)
        steps.append(run([PY_EXE, str(REPO / args[0])] + args[1:],
                         f"c2 {arm} {fold}", arm))
        src = OUT / f"C2-d8p-{fold}.parquet"
        if not src.exists():
            raise SystemExit(f"no OOF at {src}")
        df = pd.read_parquet(src)
        keep = [c for c in ("MVT_ID_mvt", "fold", "y_true", "y_pred") if c in df.columns]
        df = df[keep].copy()
        df["fold"] = fold
        if list(df.columns) != ["MVT_ID_mvt", "fold", "y_true", "y_pred"]:
            raise SystemExit(f"{src} is not in the COMMON.md contract: {list(df.columns)}")
        dst = WORK / f"c2-{arm}--{fold}.parquet"
        df.to_parquet(dst, index=False)
        rmse = float(((df["y_true"] - df["y_pred"]) ** 2).mean() ** 0.5)
        written[f"c2-{arm}--{fold}"] = {
            "rows": len(df), "rmse": rmse, "bytes": dst.stat().st_size,
            "workstation_published": PUB_RMSE[fold],
            "offset_vs_workstation": rmse - PUB_RMSE[fold]}
        log(f"wrote {dst.name}  n={len(df)}  rmse={rmse:.6f}  "
            f"workstation {PUB_RMSE[fold]:.6f}  offset {rmse - PUB_RMSE[fold]:+.6f} "
            f"(machine term, NOT a gate)")
        src.unlink()

stop.set()
th.join(timeout=10)

vers = {}
for m in ("numpy", "pandas", "pyarrow", "catboost", "sklearn", "scipy"):
    try:
        vers[m] = getattr(__import__(m), "__version__", "?")
    except Exception as e:                                  # noqa: BLE001
        vers[m] = f"MISSING ({type(e).__name__})"
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    gpu = "none"

gpu_h = sum(s["wall_s"] for s in steps) / 3600.0
port = {
    "cfg": CFG, "synthetic": SYNTH, "data_dir": str(DATA),
    "git_commit": (REPO / "GIT_COMMIT.txt").read_text().splitlines()[0],
    "python": sys.version.split()[0], "versions": vers, "gpu": gpu,
    "cpu_count": os.cpu_count(), "mem_total_gb": round(TOTAL_GB, 1),
    "peak_gpu_mib": peak["gpu_mib"], "peak_host_used_gb": round(peak["rss_gb"], 2),
    "gpu_hours_this_session": round(gpu_h, 3),
    "steps": steps, "written": written, "workstation_published": PUB_RMSE,
}
(WORK / "port.json").write_text(json.dumps(port, indent=1))
log(json.dumps({k: v for k, v in port.items() if k != "steps"}, indent=1))
for s in steps:
    log(f"{s['label']:28s} {s['wall_s']/60:7.1f} min")
log(f"\nGPU-hours this session: {gpu_h:.3f}")
log("C2 OK")
