"""exp-025-cloud: how much does the ENVIRONMENT move the prediction?

The workstation artifact cannot be reproduced until the competition data lands,
so this kernel measures the next best thing on data both machines can hold: the
same exp-018 fold, the same code, the same synthetic input, LightGBM only (no
CatBoost, because CatBoost GPU is not bitwise reproducible across GPU models and
would mask the effect being measured), run at two thread counts.

Diffed against the laptop's run of the identical command
(`experiments/exp-025-cloud/code/compare_oof.py`), it separates two things the
port will otherwise confound:

* **package and OS**: Kaggle lightgbm 4.6.0 / numpy 2.0.2 / Python 3.12 against
  the laptop's 4.7.0 / 2.4.6 / 3.11 -- the 2-thread arms differ only in this;
* **thread count**: LightGBM does not promise identical histograms across
  `num_threads`, and the workstation used 6 where Kaggle has 4. The two arms
  here differ only in this.

No GPU. Nothing here may enter the ledger: the input is synthetic.
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

CFG = {
    "fold": "july_loo",
    "rounds": 1200,
    "clip_z": 6000,
    "exact_rounds": 300,
    "ghost_mode": "mixbase",
    "main_model": "lgb",
    "seeds": 1,
    "thread_arms": [2, 4],
}
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")
IN = Path("/kaggle/input")


def log(*a):
    print(*a, flush=True)


def banner(t):
    log("\n" + "=" * 70 + f"\n== {t}\n" + "=" * 70)


banner("code")
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
log("repo at", REPO, (REPO / "GIT_COMMIT.txt").read_text().splitlines()[0])

banner("data")
cands = sorted({p.parent for p in IN.rglob("training_2025-01-01_2025-02-01.parquet")})
if not cands:
    raise SystemExit(f"no raw data; inputs {[d.name for d in IN.iterdir()]}")
DATA = cands[0]
SYNTH = (DATA / "SYNTHETIC.json").exists()
log("data dir", DATA, "synthetic" if SYNTH else "REAL")
if SYNTH:
    log("\n*** SYNTHETIC INPUT: the RMSE below is meaningless. What is real is the\n"
        "*** DIFFERENCE between two runs of the same code on the same bytes.\n")

steps = []
for thr in CFG["thread_arms"]:
    tag = f"cpu-{thr}thr"
    banner(f"fold {CFG['fold']} at {thr} threads")
    env = dict(os.environ)
    env["PRC_DATA_DIR"] = str(DATA)
    env["PYTHONPATH"] = str(REPO)
    env["OMP_NUM_THREADS"] = str(thr)
    env["PRC_THREADS"] = str(thr)
    if SYNTH:
        env["PRC_SYNTHETIC"] = "1"
    cmd = [sys.executable, "experiments/exp-018-fresh/code/run.py",
           "--tag", tag, "--folds", CFG["fold"],
           "--rounds", str(CFG["rounds"]), "--clip-z", str(CFG["clip_z"]),
           "--exact-rounds", str(CFG["exact_rounds"]),
           "--ghost-mode", CFG["ghost_mode"], "--main-model", CFG["main_model"],
           "--seeds", str(CFG["seeds"]),
           "--params", json.dumps({"num_threads": thr}), "--save-oof"]
    log("$", " ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    dt = time.time() - t0
    log(p.stdout[-8000:])
    if p.stderr.strip():
        log("--- stderr ---")
        log(p.stderr[-4000:])
    if p.returncode != 0:
        raise SystemExit(f"{tag} failed rc={p.returncode}")
    src = REPO / "experiments" / "exp-018-fresh" / "oof" / f"{CFG['fold']}.parquet"
    dst = WORK / f"{tag}-{CFG['fold']}.parquet"
    shutil.copy(src, dst)
    log(f"[{tag}] wall={dt/60:.1f} min -> {dst.name}")
    steps.append({"tag": tag, "threads": thr, "wall_s": round(dt, 1),
                  "maxrss_gb": round(resource.getrusage(
                      resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 2),
                  "stdout_tail": p.stdout[-2000:]})

banner("the two arms against each other")
import numpy as np
import pandas as pd

frames = {s["tag"]: pd.read_parquet(WORK / f"{s['tag']}-{CFG['fold']}.parquet")
          for s in steps}
tags = list(frames)
a, b = frames[tags[0]], frames[tags[1]]
m = a.merge(b, on="MVT_ID_mvt", suffixes=("_a", "_b"))
d = m["y_pred_a"].to_numpy() - m["y_pred_b"].to_numpy()
y = m["y_true_a"].to_numpy()
cmp = {"a": tags[0], "b": tags[1], "n": int(len(m)),
       "identical": bool(np.array_equal(m["y_pred_a"], m["y_pred_b"])),
       "max_abs": float(np.abs(d).max()), "rms": float(np.sqrt(np.mean(d ** 2))),
       "rows_over_1s": int((np.abs(d) > 1).sum()),
       "rmse_a": float(np.sqrt(np.nanmean((y - m["y_pred_a"]) ** 2))),
       "rmse_b": float(np.sqrt(np.nanmean((y - m["y_pred_b"]) ** 2)))}
cmp["rmse_a_minus_b"] = cmp["rmse_a"] - cmp["rmse_b"]
log(json.dumps(cmp, indent=1))

vers = {}
for mod in ("numpy", "pandas", "pyarrow", "lightgbm"):
    vers[mod] = getattr(__import__(mod), "__version__", "?")
out = {"cfg": CFG, "synthetic": SYNTH, "python": sys.version.split()[0],
       "versions": vers, "cpu_count": os.cpu_count(),
       "git_commit": (REPO / "GIT_COMMIT.txt").read_text().splitlines()[0],
       "steps": steps, "thread_arm_comparison": cmp}
(WORK / "cpu_ab.json").write_text(json.dumps(out, indent=1))
log("\nCPU AB OK")
