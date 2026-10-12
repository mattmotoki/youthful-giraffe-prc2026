"""exp-056-q12advocate: label-isolated base refits, predictions only.

A rotation holds out three months, V (scored), S (correction fit) and I (inner
selection), and its base members train on the other nine (`prc.data` folds
`iso_<r>_v`, `iso_<r>_s`, `iso_<r>_i`). `iso_<r>_x` is the exposure control:
the same-season month M swapped for S, validated on S (in-sample on purpose).

This is exp-025's `prc-nested.py` with three changes, each for label isolation:

1. The producer's stdout is never shown or saved. It prints fold RMSE, which is
   a loss read on a held month. The log keeps only its length and SHA-256.
   stderr is shown only when the producer fails.
2. The output keeps `MVT_ID_mvt`, `fold` and `y_pred`. `y_true` is dropped in
   the kernel, and no loss is computed here.
3. exp-008's fold-family table is extended at run time (`m11`, the member's own
   fixed budget), so no earlier experiment's file is edited.

Nothing here edits a producer. Each producer runs exactly as the record names it.
"""
import hashlib
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

# --------------------------------------------------------------------- config
CFG = {
    "member": "F1-fresh",
    "folds": ["iso_jul_v"],           # names from prc.data (iso_<r>_<v|s|i|x>)
    "threads": 4,
}

OUT_REL = "experiments/exp-056-q12advocate/oof-iso"     # inside the repo, see below

MEMBERS = {
    "F1-fresh": {
        "gpu": True,
        "cmd": lambda f, t: ["experiments/exp-018-fresh/code/run.py",
                             "--tag", f"iso-{f}", "--folds", f,
                             "--rounds", "1200", "--clip-z", "6000",
                             "--exact-rounds", "300", "--ghost-mode", "mixbase",
                             "--main-model", "both", "--seeds", "1",
                             "--params", json.dumps({"num_threads": t}),
                             "--save-oof"],
        "cwd": ".",
        "oof": "experiments/exp-018-fresh/oof/{f}.parquet",
    },
    "L2-hub8725": {
        "gpu": False,
        # exp-008's run.py looks the fold up in frames.FAMILY for its fixed round
        # count. The iso folds are put in `m11` (8,725 rounds), as exp-008 puts
        # its ten-month nested folds: the count is part of the member's identity.
        # The outdir must be inside the repo: run.py does `out.relative_to(ROOT)`.
        "cmd": lambda f, t: ["-c",
                             "import sys, runpy, pathlib; "
                             "import frames; "
                             f"frames.FAMILY.setdefault('{f}', 'm11'); "
                             "sys.argv = ['run.py', '--kind', 'lgb', '--name', 'E12-hub8725', "
                             "'--variant', 'plain', '--huber-delta', '1200', "
                             f"'--folds', '{f}', '--rounds-m6', '3225', '--rounds-m11', '8725', "
                             f"'--outdir', '{REPO / OUT_REL}']; "
                             "runpy.run_path(str(pathlib.Path('run.py').resolve()), "
                             "run_name='__main__')"],
        "cwd": "experiments/exp-008-diversity",
        "oof": OUT_REL + "/E12-hub8725-{f}.parquet",
    },
    "T-tailv4": {
        "gpu": False,
        # exp-009's deliver.py holds its fold list at module level and guards
        # against overwriting a published file, so the fold is injected and the
        # output name takes the `_` prefix its guard exempts (as exp-025 did).
        "cmd": lambda f, t: ["-c",
                             "import sys, pathlib; "
                             "sys.path.insert(0, 'experiments/exp-009-noac2'); "
                             "import deliver; "
                             f"deliver.FOLDS = ('{f}',); "
                             f"deliver.main(out_name='_iso-{f}.parquet')"],
        "cwd": ".",
        "oof": "experiments/exp-009-noac2/oof/_iso-{f}.parquet",
    },
}

WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")
M = MEMBERS[CFG["member"]]


def log(*a):
    print(*a, flush=True)


def banner(t):
    log("\n" + "=" * 70 + f"\n== {t}\n" + "=" * 70)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------- the code
banner("inputs")
IN = Path("/kaggle/input")
if not IN.exists():
    raise SystemExit("/kaggle/input does not exist: no dataset is attached")
for d in sorted(IN.iterdir()):
    log(f"[{d.name}]")

banner("code")
tars = sorted(IN.rglob("prc-repo.tar.gz"))
if REPO.exists():
    shutil.rmtree(REPO)
if tars:
    log("untar", tars[0], sha_file(tars[0]))
    with tarfile.open(tars[0]) as tf:
        tf.extractall("/tmp")           # the tarball's top directory is `repo`
else:
    trees = sorted(IN.rglob("prc/data.py"))
    if not trees:
        raise SystemExit("no code dataset found under /kaggle/input")
    src = trees[0].parents[1]
    log("copytree", src)
    shutil.copytree(src, REPO)
if not (REPO / "prc" / "data.py").exists():
    raise SystemExit(f"{REPO} has no prc/data.py after unpacking")
GIT = (REPO / "GIT_COMMIT.txt").read_text().splitlines()
log("repo commit", GIT[0])
log("prc/data.py", sha_file(REPO / "prc" / "data.py"))
(REPO / OUT_REL).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------- the data
banner("data")
cands = sorted({p.parent for p in IN.rglob("training_2025-01-01_2025-02-01.parquet")})
if not cands:
    raise SystemExit("no raw data dataset found under /kaggle/input")
DATA = cands[0]
if (DATA / "SYNTHETIC.json").exists():
    raise SystemExit("synthetic data attached; this kernel runs on the real files only")
DATA_SHA = {p.name: sha_file(p) for p in sorted(DATA.iterdir()) if p.is_file()}
for k, v in DATA_SHA.items():
    log(f"  {k:45s} {v}")

env = dict(os.environ)
env["PRC_DATA_DIR"] = str(DATA)
env["PYTHONPATH"] = str(REPO)
env["OMP_NUM_THREADS"] = str(CFG["threads"])
env["PRC_THREADS"] = str(CFG["threads"])

# the fold table the producers will read, printed before any fit
sys.path.insert(0, str(REPO))
os.environ["PRC_DATA_DIR"] = str(DATA)
import prc.data as PD  # noqa: E402
for f in CFG["folds"]:
    fo = PD.FOLDS[f]
    log(f"  {f}: valid {fo.valid_month}, train {','.join(m[5:] for m in fo.train_months)}")

# ------------------------------------------------------------------ monitors
peak = {"gpu_mib": 0, "host_used_gb": 0.0}
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
                    peak["host_used_gb"] = max(peak["host_used_gb"], TOTAL_GB - avail)
        except Exception:                                   # noqa: BLE001
            pass


th = threading.Thread(target=watch, daemon=True)
th.start()


def run(cmd, label, cwd="."):
    banner(label)
    log("$ cd", cwd, "&&", " ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, env=env, cwd=str(REPO / cwd), capture_output=True, text=True)
    dt = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    # stdout carries fold RMSE lines: record that it exists, never what it says
    so = p.stdout.encode()
    log(f"[{label}] rc={p.returncode} wall={dt/60:.1f} min child_maxrss={rss:.2f} GB "
        f"stdout {len(so)} bytes sha256 {hashlib.sha256(so).hexdigest()}")
    if p.returncode != 0:
        log("--- stderr (failure only) ---")
        log(p.stderr[-8000:])
        raise SystemExit(f"{label} failed with rc={p.returncode}")
    return {"label": label, "rc": p.returncode, "wall_s": round(dt, 1),
            "child_maxrss_gb": round(rss, 2), "stdout_bytes": len(so),
            "stdout_sha256": hashlib.sha256(so).hexdigest()}


PY = sys.executable
steps, written = [], {}
for fold in CFG["folds"]:
    args = M["cmd"](fold, CFG["threads"])
    cmd = [PY, "-c", args[1]] if args[0] == "-c" else [PY, str(REPO / args[0])] + args[1:]
    steps.append(run(cmd, f"{CFG['member']} {fold}", cwd=M["cwd"]))

    # ---- collect this fold at once: predictions only
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    src = REPO / M["oof"].format(f=fold)
    if not src.exists():
        raise SystemExit(f"MISSING output for {fold}: {src}")
    df = pd.read_parquet(src, columns=["MVT_ID_mvt", "y_pred"])
    df.insert(1, "fold", fold)
    dst = WORK / f"{CFG['member']}--{fold}.parquet"
    df.to_parquet(dst, index=False)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(df["MVT_ID_mvt"].to_numpy("float64")).tobytes())
    h.update(np.ascontiguousarray(df["y_pred"].to_numpy("float64")).tobytes())
    written[fold] = {"rows": int(len(df)), "bytes": dst.stat().st_size,
                     "file_sha256": sha_file(dst), "vector_sha256_ids_then_pred": h.hexdigest(),
                     "from": str(src.relative_to(REPO))}
    log(f"wrote {dst.name} rows={len(df)} sha256={written[fold]['file_sha256']}")
    src.unlink()                        # the producer's copy holds y_true

stop.set()
th.join(timeout=10)

vers = {}
for m in ("numpy", "pandas", "pyarrow", "lightgbm", "catboost", "sklearn", "scipy"):
    try:
        vers[m] = getattr(__import__(m), "__version__", "?")
    except Exception as e:                                  # noqa: BLE001
        vers[m] = f"MISSING ({type(e).__name__})"
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    gpu = "none (CPU kernel)"

port = {"cfg": CFG, "member": CFG["member"], "git": GIT, "data_sha256": DATA_SHA,
        "python": sys.version.split()[0], "versions": vers, "gpu": gpu,
        "cpu_count": os.cpu_count(), "mem_total_gb": round(TOTAL_GB, 1),
        "peak_gpu_mib": peak["gpu_mib"], "peak_host_used_gb": round(peak["host_used_gb"], 2),
        "steps": steps, "written": written}
(WORK / "port.json").write_text(json.dumps(port, indent=1))
for s in steps:
    log(f"{s['label']:28s} {s['wall_s']/60:7.1f} min  maxrss {s['child_maxrss_gb']:.2f} GB")
log(f"peak GPU {peak['gpu_mib']} MiB   peak host used {peak['host_used_gb']:.2f} GB")
log("\nISO OK")
