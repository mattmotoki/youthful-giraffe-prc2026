"""exp-025-cloud: nested-rotation OOF for the clean instrument (the nested validation rule).

To score fold `g` the blend weights and the stacker are fitted on the OTHER
folds' member OOF, and those members were trained on month sets that CONTAIN
`g`'s month. So `g`'s labels reach the vector that scores `g`, one step removed.
This kernel closes that channel: it refits one member on `prc.data`'s nested
folds -- `<fold>_nx_<mon>` is `<fold>` with one more month dropped from its
training set -- and writes the OOF under the COMMON.md contract.

Six nested folds serve the three scored folds, but they need only THREE fits per
member: a training set is the complement of a PAIR from {Jul, Dec, Jan} and it
predicts both of the months it drops. The kernel takes the fold list, so a
caller can put a training set's two folds in one session and pay the load once.
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
    "member": "C2-cat-q",
    "folds": ["november_loo"],
    "threads": 4,
    "tag": "q-7bk-member",
}

#: One entry per member whose OOF feeds the stacker, with the producer exactly as
#: the record names it. `oof` is where the producer leaves the file, `{f}` the
#: fold name. Nothing here edits a producer; the nested fold is a `prc.data`
#: entry, so every one of them takes it as an ordinary `--folds` argument.
MEMBERS = {
    "L2-hub8725-q": {
        "gpu": False,
        # exp-008-diversity/frames.FAMILY is a ROUND-BUDGET map with no entry for
        # november_loo (§7bj).  Eleven training months is the m11 family, exactly as
        # july_loo/december/january are; exp-025-cloud already made this additive
        # edit for its six nested folds.  Injected at runtime: frames.py is shared
        # code and this lane does not edit it.  The argv is built in the lambda so
        # NESTED_OUT is resolved HERE and repr'd into the child's sys.argv.
        "cmd": lambda f, t: ["-c", (
            "import sys, runpy; "
            "sys.path.insert(0, 'experiments/exp-008-diversity'); "
            "import frames; frames.FAMILY[%r] = 'm11'; "
            "sys.argv = %r; "
            "runpy.run_path('experiments/exp-008-diversity/run.py', "
            "run_name='__main__')") % (f, ["experiments/exp-008-diversity/run.py",
                "--kind", "lgb", "--name", "L2-hub8725", "--variant", "plain",
                "--huber-delta", "1200",
                "--rounds-m6", "3225", "--rounds-m11", "8725", "--folds", f, "--outdir", str(NESTED_OUT)])],
        "cwd": ".",
        "oof": None,
        "env": {},
    },
    "C2-cat-q": {
        "gpu": True,
        # exp-008-diversity/frames.FAMILY is a ROUND-BUDGET map with no entry for
        # november_loo (§7bj).  Eleven training months is the m11 family, exactly as
        # july_loo/december/january are; exp-025-cloud already made this additive
        # edit for its six nested folds.  Injected at runtime: frames.py is shared
        # code and this lane does not edit it.  The argv is built in the lambda so
        # NESTED_OUT is resolved HERE and repr'd into the child's sys.argv.
        "cmd": lambda f, t: ["-c", (
            "import sys, runpy; "
            "sys.path.insert(0, 'experiments/exp-008-diversity'); "
            "import frames; frames.FAMILY[%r] = 'm11'; "
            "sys.argv = %r; "
            "runpy.run_path('experiments/exp-008-diversity/run.py', "
            "run_name='__main__')") % (f, ["experiments/exp-008-diversity/run.py",
                "--kind", "cat", "--name", "C2-d8p", "--depth", "8",
                "--rounds-m6", "3625", "--rounds-m11", "5325", "--folds", f, "--outdir", str(NESTED_OUT)])],
        "cwd": ".",
        "oof": None,
        "env": {'PRC_GPU': '0'},
    },
    "F1-fresh": {
        "gpu": True,
        "cmd": lambda f, t: ["experiments/exp-018-fresh/code/run.py",
                             "--tag", f"nested-{f}", "--folds", f,
                             "--rounds", "1200", "--clip-z", "6000",
                             "--exact-rounds", "300", "--ghost-mode", "mixbase",
                             "--main-model", "both", "--seeds", "1",
                             "--params", json.dumps({"num_threads": t}),
                             "--save-oof"],
        "cwd": ".",
        "oof": "experiments/exp-018-fresh/oof/{f}.parquet",
        "env": {},
    },
    "L2-hub8725": {
        "gpu": False,
        "cmd": lambda f, t: ["experiments/exp-008-diversity/run.py", "--kind", "lgb",
                             "--name", "E12-hub8725", "--variant", "plain",
                             "--huber-delta", "1200", "--folds", f,
                             "--rounds-m6", "3225", "--rounds-m11", "8725",
                             # NOT /kaggle/working: run.py does
                             # `out.relative_to(ROOT)` AFTER writing, so an outdir
                             # outside the repo writes the parquet and then throws.
                             # Its own comment says this cost exp-012 a fold.
                             "--outdir", str(NESTED_OUT)],
        "cwd": "experiments/exp-008-diversity",
        "oof": "experiments/exp-025-cloud/oof-nested/E12-hub8725-{f}.parquet",
        "env": {},
    },
    "V3-allin": {
        "gpu": False,
        "cmd": lambda f, t: ["experiments/exp-001-retro-v1/run.py",
                             "--name", "V3-allin", "--frame", "v3",
                             "--target", "residual", "--clip-floor", "60",
                             "--noac-head", "--learning-rate", "0.03",
                             "--fixed-rounds", "660", "--folds", f,
                             "--no-submission"],
        "cwd": "experiments/exp-001-retro-v1",
        "oof": "experiments/exp-001-retro-v1/oof/V3-allin-{f}.parquet",
        "env": {},
    },
    "T-tailv4": {
        "gpu": False,
        # exp-009's deliver.py holds its fold list at module level and guards
        # against overwriting a published file, so the fold is injected and the
        # output name is given the `_` prefix its guard exempts.
        "cmd": lambda f, t: ["-c",
                             "import sys, pathlib; "
                             "sys.path.insert(0, 'experiments/exp-009-noac2'); "
                             "import deliver; "
                             f"deliver.FOLDS = ('{f}',); "
                             f"deliver.main(out_name='_nested-{f}.parquet')"],
        "cwd": ".",
        "oof": "experiments/exp-009-noac2/oof/_nested-{f}.parquet",
        "env": {},
    },
}
M = MEMBERS[CFG["member"]]
#: Resolved after REPO is known; the lambdas above read it at call time.
NESTED_OUT = None

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


def run(cmd, label, cwd="."):
    banner(label)
    log("$ cd", cwd, "&&", " ".join(cmd))
    t0 = time.time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    p = subprocess.run(cmd, env=env, cwd=str(REPO / cwd), capture_output=True, text=True)
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
NESTED_OUT = REPO / "experiments" / "exp-025-cloud" / "oof-nested"
NESTED_OUT.mkdir(parents=True, exist_ok=True)
steps = []
written = {}
for fold in CFG["folds"]:
    args = M["cmd"](fold, CFG["threads"])
    # Script paths are repo-relative while cwd is the producer's own directory,
    # so they are resolved against REPO. `-c` is passed through untouched.
    cmd = [PY, "-c", args[1]] if args[0] == "-c" else [PY, str(REPO / args[0])] + args[1:]
    steps.append(run(cmd, f"{CFG['member']} {fold}", cwd=M["cwd"]))

stop.set()
th.join(timeout=10)

# ------------------------------------------------------------------- collect
banner("collect")
import pandas as pd  # noqa: E402

for fold in CFG["folds"]:
    if M["oof"]:
        cands = [REPO / M["oof"].format(f=fold)]
    else:
        cands = sorted(NESTED_OUT.rglob(f"*{fold}*.parquet"))
    src = next((c for c in cands if c.exists()), None)
    if src is None:
        log("MISSING oof for", fold, "looked at", [str(c) for c in cands])
        written[fold] = "MISSING"
        continue
    df = pd.read_parquet(src)
    # The COMMON.md contract, and the fold column must name the NESTED fold so a
    # consumer can never confuse it with the eleven-month one it looks like.
    keep = [c for c in ("MVT_ID_mvt", "fold", "y_true", "y_pred") if c in df.columns]
    df = df[keep].copy()
    df["fold"] = fold
    if list(df.columns) != ["MVT_ID_mvt", "fold", "y_true", "y_pred"]:
        raise SystemExit(f"{src} is not in the COMMON.md contract: {list(df.columns)}")
    dst = WORK / f"{CFG['member']}--{fold}.parquet"
    df.to_parquet(dst, index=False)
    rmse = float(((df["y_true"] - df["y_pred"]) ** 2).mean() ** 0.5)
    written[fold] = {"rows": len(df), "rmse": rmse, "bytes": dst.stat().st_size,
                     "from": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src)}
    log(f"wrote {dst.name}  n={len(df)}  rmse={rmse:.4f}")

vers = {}
for m in ("numpy", "pandas", "pyarrow", "lightgbm", "catboost", "sklearn", "scipy"):
    try:
        vers[m] = getattr(__import__(m), "__version__", "?")
    except Exception as e:                                  # noqa: BLE001
        vers[m] = f"MISSING ({type(e).__name__})"
try:  # a CPU-only kernel has no nvidia-smi, and three of the four members are CPU
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    gpu = "none (CPU kernel)"

port = {
    "cfg": CFG, "synthetic": SYNTH, "data_dir": str(DATA),
    "data_files": {p.name: p.stat().st_size for p in sorted(DATA.iterdir()) if p.is_file()},
    "git_commit": (REPO / "GIT_COMMIT.txt").read_text().splitlines()[0],
    "python": sys.version.split()[0], "versions": vers, "gpu": gpu,
    "cpu_count": os.cpu_count(), "mem_total_gb": round(TOTAL_GB, 1),
    "peak_gpu_mib": peak["gpu_mib"], "peak_host_used_gb": round(peak["rss_gb"], 2),
    "steps": steps, "written": written, "member": CFG["member"],
}
(WORK / "port.json").write_text(json.dumps(port, indent=1))
log(json.dumps({k: v for k, v in port.items() if k != "steps"}, indent=1))
for s in steps:
    log(f"{s['label']:28s} {s['wall_s']/60:7.1f} min  maxrss {s['child_maxrss_gb']:.2f} GB")
log(f"peak GPU {peak['gpu_mib']} MiB   peak host used {peak['rss_gb']:.2f} GB")
if any(v == "MISSING" for v in written.values()):
    raise SystemExit(f"no OOF written for {[k for k, v in written.items() if v == 'MISSING']}")
log("\nNESTED OK")
