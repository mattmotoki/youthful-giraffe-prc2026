"""exp-040-chain: the `L2-hub8725` member refitted with and without the stand chain.

Q09 puts the stand-chain inputs INSIDE a serving member.  exp-037 §7 found the
chain is already in every `prc.tail_model` head, so the two places in stack-5
where it is genuinely absent are the retro-frame-v2 members `L2-hub8725` and
`C2-cat`.  This kernel is the first of those.

Three arms, one member, the SAME producer command in every one
(`experiments/exp-008-diversity/run.py`, the argv `docs/CLOUD-COMPUTE.md`
records for `L2-hub8725`):

    none      the shipped v2 frame -- the reproduction gate against
              `experiments/exp-025-cloud/oof-nested/L2-hub8725--{fold}.parquet`,
              which is the SAME-ENVIRONMENT baseline (exp-025 §10.6: a shift
              read against the workstation's published OOF is mostly the machine)
    chain     v2 + `ts_gap_arr1..3`, the chain as `prc/tail_features.py` computes
              it and as the retro v4 frame imports it
    derived   the same three + exp-037's `slack`, `slack2`, `log_g`,
              `n_dep_between`, `arr_stand_per_day`

Same seeds, same 8,725/3,225 rounds, same data, same threads.  Everything that
changes between two arms is the column list, and `PRC_CHAIN_ARM` is the only
thing this kernel varies.

CPU only: LightGBM, no GPU hour is spent.  One fold's three arms are about 2.7 h
together, so a fold is a session and the three sessions run side by side.
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

# --------------------------------------------------------------------- config
CFG = {
    "folds": ["december"],
    "arms": ["none", "chain", "derived"],
    "threads": 4,
    "submit": False,
    "tag": "chain-december",
}

#: The producer, exactly as `exp-025-cloud/kernels/nested/prc-nested.py` records
#: it for this member, with `run_arm.py` in front of `run.py` so that the module
#: named `frames` is the arm's.  `E12-hub8725` is the name the OOF file takes.
def cmd_for(arm, fold, threads, out):
    return ["experiments/exp-040-chain/code/run_arm.py", "--kind", "lgb",
            "--name", "E12-hub8725", "--variant", "plain",
            "--huber-delta", "1200", "--folds", fold,
            "--rounds-m6", "3225", "--rounds-m11", "8725",
            "--outdir", str(out)]


def submit_cmd(arm, threads):
    return ["experiments/exp-040-chain/code/run_arm.py", "--kind", "lgb",
            "--name", f"L2-hub8725-{arm}", "--variant", "plain",
            "--huber-delta", "1200", "--rounds-m6", "3225", "--rounds-m11", "8725",
            "--submit"]


#: exp-025's same-environment Kaggle baselines for this member
#: (`experiments/exp-025-cloud/results/nested-l2-hub8725-base.json`).  The `none`
#: arm must land on these; anything else is a defect in this kernel's frame swap
#: and no `chain` number may be read until it does.
BASE_RMSE = {"july_loo": 314.5038376447149,
             "december": 227.16488810275288,
             "january": 350.8636073436026}
GATE = 0.05

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
if SYNTH:
    log("\n*** SYNTHETIC INPUT: this run measures the code path and the runtime only.\n")

env = dict(os.environ)
env["PRC_DATA_DIR"] = str(DATA)
env["PYTHONPATH"] = str(REPO)
env["OMP_NUM_THREADS"] = str(CFG["threads"])
env["PRC_THREADS"] = str(CFG["threads"])
if SYNTH:
    env["PRC_SYNTHETIC"] = "1"

TOTAL_GB = int(next(l for l in open("/proc/meminfo")
                    if l.startswith("MemTotal")).split()[1]) / 2**20
peak = {"rss_gb": 0.0}
stop = threading.Event()


def watch():
    while not stop.wait(10.0):
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
    log("$ PRC_CHAIN_ARM=" + arm, "cd", cwd, "&&", " ".join(cmd))
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
        args = cmd_for(arm, fold, CFG["threads"], OUT)
        steps.append(run([PY_EXE, str(REPO / args[0])] + args[1:],
                         f"{arm} {fold}", arm))
        src = OUT / f"E12-hub8725-{fold}.parquet"
        if not src.exists():
            raise SystemExit(f"no OOF at {src}")
        df = pd.read_parquet(src)
        keep = [c for c in ("MVT_ID_mvt", "fold", "y_true", "y_pred") if c in df.columns]
        df = df[keep].copy()
        df["fold"] = fold
        if list(df.columns) != ["MVT_ID_mvt", "fold", "y_true", "y_pred"]:
            raise SystemExit(f"{src} is not in the COMMON.md contract: {list(df.columns)}")
        dst = WORK / f"{arm}--{fold}.parquet"
        df.to_parquet(dst, index=False)
        rmse = float(((df["y_true"] - df["y_pred"]) ** 2).mean() ** 0.5)
        rec = {"rows": len(df), "rmse": rmse, "bytes": dst.stat().st_size}
        if arm == "none" and not SYNTH:
            rec["gate_baseline"] = BASE_RMSE[fold]
            rec["gate_offset"] = rmse - BASE_RMSE[fold]
            rec["gate_pass"] = bool(abs(rec["gate_offset"]) <= GATE)
            log(f"[gate] {fold} none {rmse:.6f} against exp-025 "
                f"{BASE_RMSE[fold]:.6f}  offset {rec['gate_offset']:+.6f}  "
                f"{'PASS' if rec['gate_pass'] else 'FAIL'}")
        written[f"{arm}--{fold}"] = rec
        log(f"wrote {dst.name}  n={len(df)}  rmse={rmse:.6f}")
        src.unlink()

if CFG["submit"]:
    for arm in CFG["arms"]:
        args = submit_cmd(arm, CFG["threads"])
        steps.append(run([PY_EXE, str(REPO / args[0])] + args[1:],
                         f"{arm} ranking", arm))
        src = REPO / "artifacts" / "exp-008" / "pred" / f"L2-hub8725-{arm}-ranking.parquet"
        if not src.exists():
            raise SystemExit(f"no ranking prediction at {src}")
        dst = WORK / f"{arm}--ranking.parquet"
        shutil.copy(src, dst)
        d = pd.read_parquet(dst)
        written[f"{arm}--ranking"] = {"rows": len(d), "bytes": dst.stat().st_size,
                                      "min": float(d["y_pred"].min()),
                                      "median": float(d["y_pred"].median()),
                                      "finite": bool(d["y_pred"].notna().all())}
        log(f"wrote {dst.name}  n={len(d)}")

stop.set()
th.join(timeout=10)

vers = {}
for m in ("numpy", "pandas", "pyarrow", "lightgbm", "sklearn", "scipy"):
    try:
        vers[m] = getattr(__import__(m), "__version__", "?")
    except Exception as e:                                  # noqa: BLE001
        vers[m] = f"MISSING ({type(e).__name__})"

port = {
    "cfg": CFG, "synthetic": SYNTH, "data_dir": str(DATA),
    "data_files": {p.name: p.stat().st_size for p in sorted(DATA.iterdir()) if p.is_file()},
    "git_commit": (REPO / "GIT_COMMIT.txt").read_text().splitlines()[0],
    "python": sys.version.split()[0], "versions": vers,
    "cpu_count": os.cpu_count(), "mem_total_gb": round(TOTAL_GB, 1),
    "peak_host_used_gb": round(peak["rss_gb"], 2),
    "steps": steps, "written": written, "gate_rmse": BASE_RMSE, "gate_tol": GATE,
}
(WORK / "port.json").write_text(json.dumps(port, indent=1))
log(json.dumps({k: v for k, v in port.items() if k != "steps"}, indent=1))
for s in steps:
    log(f"{s['label']:28s} {s['wall_s']/60:7.1f} min  maxrss {s['child_maxrss_gb']:.2f} GB")
bad = [k for k, v in written.items()
       if isinstance(v, dict) and v.get("gate_pass") is False]
if bad:
    raise SystemExit(f"reproduction gate failed: {bad}")
log("\nCHAIN OK")
