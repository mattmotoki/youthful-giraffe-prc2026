"""exp-025-cloud step 1: Kaggle environment probe. No data, no internet.

Prints what settles the port: the GPU model and memory, the CPU count, RAM,
disk, the package versions the exp-018 lineage imports, whether CatBoost and
XGBoost see the GPU, and a timed LightGBM/CatBoost micro-benchmark that scales
to the real fit (1.8 M rows x ~140 features x 1200 rounds x 255 leaves).
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

OUT = {}


def sec(name):
    print(f"\n===== {name} " + "=" * (60 - len(name)), flush=True)


sec("host")
OUT["python"] = sys.version.split()[0]
OUT["platform"] = platform.platform()
OUT["cpu_count"] = os.cpu_count()
try:
    OUT["cpu_affinity"] = len(os.sched_getaffinity(0))
except AttributeError:
    OUT["cpu_affinity"] = None
try:
    mem = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        mem[k] = v.strip()
    OUT["mem_total"] = mem.get("MemTotal")
    OUT["mem_available"] = mem.get("MemAvailable")
except OSError:
    pass
try:
    OUT["cgroup_mem_max"] = open("/sys/fs/cgroup/memory.max").read().strip()
except OSError:
    try:
        OUT["cgroup_mem_max"] = open(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes").read().strip()
    except OSError:
        pass
for p in ("/kaggle/working", "/kaggle/input", "/tmp", "/"):
    if os.path.exists(p):
        t, u, f = shutil.disk_usage(p)
        OUT[f"disk_{p}"] = f"total={t/2**30:.1f}G free={f/2**30:.1f}G"
for k, v in OUT.items():
    print(f"{k:20s} {v}")

sec("nvidia-smi")
try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=120).stdout)
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=120).stdout.strip()
    print(q)
    OUT["gpus"] = [l.strip() for l in q.splitlines() if l.strip()]
except Exception as e:                                    # noqa: BLE001
    OUT["gpus"] = f"nvidia-smi failed: {e!r}"
    print(OUT["gpus"])

sec("versions")
PKGS = ["numpy", "pandas", "polars", "pyarrow", "lightgbm", "catboost", "xgboost",
        "sklearn", "scipy", "duckdb", "optuna", "torch", "psutil"]
vers = {}
for m in PKGS:
    try:
        mod = __import__(m)
        vers[m] = getattr(mod, "__version__", "?")
    except Exception as e:                                 # noqa: BLE001
        vers[m] = f"MISSING ({type(e).__name__})"
    print(f"{m:12s} {vers[m]}")
OUT["versions"] = vers

sec("accelerator visibility")
acc = {}
try:
    import torch
    acc["torch_cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        acc["torch_devices"] = [torch.cuda.get_device_name(i)
                                for i in range(torch.cuda.device_count())]
        acc["torch_mem_gb"] = [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)
                               for i in range(torch.cuda.device_count())]
except Exception as e:                                     # noqa: BLE001
    acc["torch_cuda"] = f"error {e!r}"
print(json.dumps(acc, indent=1))

rng = np.random.default_rng(0)
NB, FB = 40_000, 40
Xb = rng.normal(size=(NB, FB)).astype("float32")
yb = Xb[:, 0] * 3 + rng.normal(size=NB)

try:
    import catboost as cb
    t0 = time.time()
    m = cb.CatBoostRegressor(iterations=50, depth=6, verbose=0, task_type="GPU", devices="0")
    m.fit(Xb, yb)
    acc["catboost_gpu"] = f"OK ({time.time()-t0:.1f}s for 50 iters on {NB}x{FB})"
except Exception as e:                                     # noqa: BLE001
    acc["catboost_gpu"] = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
print("catboost_gpu:", acc["catboost_gpu"])

try:
    import xgboost as xgb
    t0 = time.time()
    m = xgb.XGBRegressor(n_estimators=50, max_depth=6, device="cuda", tree_method="hist",
                         verbosity=0)
    m.fit(Xb, yb)
    acc["xgboost_gpu"] = f"OK ({time.time()-t0:.1f}s for 50 iters on {NB}x{FB})"
except Exception as e:                                     # noqa: BLE001
    acc["xgboost_gpu"] = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
print("xgboost_gpu:", acc["xgboost_gpu"])

try:
    import lightgbm as lgb
    ds = lgb.Dataset(Xb, label=yb)
    lgb.train(dict(objective="l2", device_type="gpu", verbose=-1, num_leaves=31), ds,
              num_boost_round=20)
    acc["lightgbm_gpu"] = "OK"
except Exception as e:                                     # noqa: BLE001
    acc["lightgbm_gpu"] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
print("lightgbm_gpu:", acc["lightgbm_gpu"])
OUT["accel"] = acc

sec("micro-benchmark (scales to the real fit)")
# The exp-018 main head: 255 leaves, lr 0.05, feature_fraction 0.7, bagging 0.8,
# 13 categorical columns, ~140 features. Fit 200k x 140 for 100 rounds and
# report seconds; the real july_loo fit is 1.80 M rows x 1200 rounds.
import lightgbm as lgb
import pandas as pd

N, F, CATS = 200_000, 140, 13
rng = np.random.default_rng(7)
X = pd.DataFrame(rng.normal(size=(N, F)).astype("float32"),
                 columns=[f"f{i}" for i in range(F)])
for i in range(CATS):
    X[f"c{i}"] = rng.integers(0, [10, 60, 3000, 900, 300, 8, 6, 5, 4, 2000, 400, 200, 1500][i],
                              size=N).astype("int32")
cat_cols = [f"c{i}" for i in range(CATS)]
y = (X["f0"] * 100 + X["f1"] * 50 + rng.normal(size=N) * 30).to_numpy()

bench = {}
for threads in sorted({4, os.cpu_count() or 4}):
    p = dict(objective="l2", learning_rate=0.05, num_leaves=255, min_data_in_leaf=60,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
             max_cat_threshold=64, cat_smooth=20.0, cat_l2=10.0, num_threads=threads,
             verbose=-1, seed=7)
    ds = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=False)
    t0 = time.time()
    lgb.train(p, ds, num_boost_round=100)
    dt = time.time() - t0
    bench[f"lgb_100r_200k_{threads}thr_s"] = round(dt, 1)
    scale = (1_800_000 / N) * (1200 / 100)
    bench[f"lgb_projected_1.8M_1200r_{threads}thr_min"] = round(dt * scale / 60, 1)
    print(f"lgb {threads} threads: {dt:.1f}s -> projected {dt*scale/60:.1f} min for "
          f"1.8M rows x 1200 rounds", flush=True)

try:
    import catboost as cb
    Xc = X.copy()
    cidx = [Xc.columns.get_loc(c) for c in cat_cols]
    pool = cb.Pool(Xc, label=y, cat_features=cidx)
    t0 = time.time()
    cb.CatBoostRegressor(loss_function="RMSE", iterations=200, learning_rate=0.06, depth=8,
                         l2_leaf_reg=6.0, random_seed=7, verbose=0, task_type="GPU",
                         devices="0", max_ctr_complexity=2).fit(pool)
    dt = time.time() - t0
    bench["cat_gpu_200it_200k_s"] = round(dt, 1)
    scale = (1_800_000 / N) * (3000 / 200)
    bench["cat_gpu_projected_1.8M_3000it_min"] = round(dt * scale / 60, 1)
    print(f"catboost GPU: {dt:.1f}s -> projected {dt*scale/60:.1f} min for 1.8M x 3000 iters",
          flush=True)
except Exception as e:                                     # noqa: BLE001
    bench["cat_gpu_200it_200k_s"] = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
    print("catboost bench failed:", bench["cat_gpu_200it_200k_s"])
OUT["bench"] = bench

sec("memory headroom of a 1.8 M x 155 float64 frame")
try:
    import psutil
    proc = psutil.Process()
    before = proc.memory_info().rss / 2**30
    big = pd.DataFrame(np.zeros((1_800_000, 155), dtype="float64"))
    after = proc.memory_info().rss / 2**30
    OUT["frame_18M_155_gb"] = round(after - before, 2)
    print(f"1.8M x 155 float64 frame costs {after-before:.2f} GB "
          f"(rss {before:.2f} -> {after:.2f} GB)")
    del big
except Exception as e:                                     # noqa: BLE001
    print("memory probe failed:", repr(e))

sec("json")
print(json.dumps(OUT, indent=1, default=str))
with open("/kaggle/working/probe.json", "w") as fh:
    json.dump(OUT, fh, indent=1, default=str)
print("\nPROBE OK")
