"""exp-056-q12advocate: the matched-size exposure pair, predictions only (SPEC 3.5).

`prc-iso.py` with one addition: the training departures of a fold are thinned to a
fixed count before the fit, so the exposed base (A - M + S) and the isolated base
(A) of a rotation train on exactly the same number of departures. The larger side
of each pair is thinned; the smaller side is the unthinned fold already fitted by
`prc-iso.py`:

    alias        base fold    train months    thinned to    pairs with
    iso_jul_xm   iso_jul_x    A - M + S       |A|           iso_jul_s  (unthinned A)
    iso_jan_sm   iso_jan_s    A               |A - M + S|   iso_jan_x  (unthinned)
    iso_dec_sm   iso_dec_s    A               |A - M + S|   iso_dec_x  (unthinned)

The kept departures are the n with the smallest sha256("exp056-size|<MVT_ID_mvt>")
among all departures of the training months (label-free, fixed before any fit).
Features are still built from whole months; only the rows that reach the learner
change. Each producer takes the subset through a run-time patch in its own process,
so no producer file is edited:

- F1-fresh: exp-018 `dataset.load` of the fold's training months is filtered.
- L2-hub8725: exp-008 `frames.fold_frames` drops the training rows. Its guard range
  still comes from the whole training months (min and max of BLOCK - AOBT).
- T-tailv4: `TailModel.fit` gets `keep`, its own training-side mask. `keep` reaches
  the two bulk trees and the noac heads; by design it does not reach the high-delay
  band head, and phi is fitted on the whole months (`prc/tail_model.py`).

Stdout is sealed and `y_true` is dropped exactly as in `prc-iso.py`.
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
    "folds": ["iso_jul_xm"],          # aliases from THIN below
    "threads": 4,
}

#: alias -> base prc.data fold, its training departures (label-free count on the raw
#: files) and the count kept. Fixed before any fit (SPEC 3.5).
THIN = {
    "iso_jul_xm": {"base": "iso_jul_x", "train_departures": 1527241, "n_keep": 1519202},
    "iso_jan_sm": {"base": "iso_jan_s", "train_departures": 1623160, "n_keep": 1601214},
    "iso_dec_sm": {"base": "iso_dec_s", "train_departures": 1571364, "n_keep": 1549745},
}
SALT = "exp056-size"

OUT_REL = "experiments/exp-056-q12advocate/oof-iso"     # inside the repo, see below
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")


def prelude(f):
    """Code run first in the producer's own process: register the alias fold (same
    months as its base fold) and load the kept ids. `_rec` writes the row counts
    before and after each thinning to /tmp/thin-<alias>.json (counts only)."""
    b = THIN[f]["base"]
    return f"""
import sys, json, pathlib, runpy
import numpy as np
import pandas as pd
sys.path.insert(0, {str(REPO)!r})
import prc.data as PD
_b = PD.FOLDS[{b!r}]
PD.FOLDS[{f!r}] = PD.Fold({f!r}, _b.train_months, _b.valid_month)
KEEP = np.load('/tmp/keep-{f}.npy')
_seen = []
def _rec(n0, n1):
    _seen.append([int(n0), int(n1)])
    pathlib.Path('/tmp/thin-{f}.json').write_text(json.dumps(_seen))
"""


def f1_cmd(f, t):
    argv = ["run.py", "--tag", f"iso-{f}", "--folds", f, "--rounds", "1200",
            "--clip-z", "6000", "--exact-rounds", "300", "--ghost-mode", "mixbase",
            "--main-model", "both", "--seeds", "1",
            "--params", json.dumps({"num_threads": t}), "--save-oof"]
    return ["-c", prelude(f) + f"""
sys.path.insert(0, str(pathlib.Path('experiments/exp-018-fresh/code').resolve()))
import dataset as _D
_load = _D.load
def _thin_load(months):
    df = _load(months)
    if list(months) == list(_b.train_months):
        n0 = len(df)
        df = df[df['MVT_ID_mvt'].isin(KEEP)].reset_index(drop=True)
        _rec(n0, len(df))
    return df
_D.load = _thin_load
sys.argv = {argv!r}
runpy.run_path(str(pathlib.Path('experiments/exp-018-fresh/code/run.py').resolve()),
               run_name='__main__')
"""]


def l2_cmd(f, t):
    # exp-008's run.py looks the fold up in frames.FAMILY for its fixed round count
    # (`m11`, 8,725 rounds, as in prc-iso.py). The outdir must be inside the repo.
    argv = ["run.py", "--kind", "lgb", "--name", "E12-hub8725", "--variant", "plain",
            "--huber-delta", "1200", "--folds", f, "--rounds-m6", "3225",
            "--rounds-m11", "8725", "--outdir", str(REPO / OUT_REL)]
    return ["-c", prelude(f) + f"""
sys.path.insert(0, str(pathlib.Path('.').resolve()))
import frames
frames.FAMILY.setdefault({f!r}, 'm11')
_ff = frames.fold_frames
def _thin_ff(fold):
    tr, va = _ff(fold)
    if fold == {f!r}:
        n0 = len(tr)
        tr = tr[tr['MVT_ID_mvt'].isin(KEEP)].reset_index(drop=True)
        _rec(n0, len(tr))
    return tr, va
frames.fold_frames = _thin_ff
sys.argv = {argv!r}
runpy.run_path(str(pathlib.Path('run.py').resolve()), run_name='__main__')
"""]


def t_cmd(f, t):
    # exp-009's deliver.py: fold injected, `_` output prefix (as in prc-iso.py).
    return ["-c", prelude(f) + f"""
sys.path.insert(0, 'experiments/exp-009-noac2')
import deliver
import prc.tail_model as TM
_fit = TM.TailModel.fit
def _thin_fit(self, train_frames, *a, **k):
    if isinstance(train_frames, (list, tuple)):
        train_frames = pd.concat(train_frames, ignore_index=True)
    if k.get('keep') is not None:
        raise SystemExit('the producer already passes keep')
    dep = train_frames['PHASE_mvt'].to_numpy() == 'DEP'
    kk = train_frames['MVT_ID_mvt'].isin(KEEP).to_numpy()
    k['keep'] = pd.Series(kk, index=train_frames.index)
    _rec(int(dep.sum()), int((dep & kk).sum()))
    return _fit(self, train_frames, *a, **k)
TM.TailModel.fit = _thin_fit
deliver.FOLDS = ({f!r},)
deliver.main(out_name='_iso-{f}.parquet')
"""]


MEMBERS = {
    "F1-fresh": {"gpu": True, "cmd": f1_cmd, "cwd": ".",
                 "oof": "experiments/exp-018-fresh/oof/{f}.parquet"},
    "L2-hub8725": {"gpu": False, "cmd": l2_cmd, "cwd": "experiments/exp-008-diversity",
                   "oof": OUT_REL + "/E12-hub8725-{f}.parquet"},
    "T-tailv4": {"gpu": False, "cmd": t_cmd, "cwd": ".",
                 "oof": "experiments/exp-009-noac2/oof/_iso-{f}.parquet"},
}
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# the kept training departures of each alias, fixed before any fit (label-free:
# ids and phase only)
banner("thin")
THIN_REC = {}
for f in CFG["folds"]:
    fo = PD.FOLDS[THIN[f]["base"]]
    log(f"  {f} = {THIN[f]['base']}: valid {fo.valid_month}, "
        f"train {','.join(m[5:] for m in fo.train_months)}")
    ids = []
    for m in fo.train_months:
        t = pd.read_parquet(PD.month_file(m), columns=["MVT_ID_mvt", "PHASE_mvt"])
        ids.append(t.loc[t["PHASE_mvt"].to_numpy() == "DEP", "MVT_ID_mvt"].to_numpy("int64"))
    ids = np.concatenate(ids)
    if len(ids) != THIN[f]["train_departures"] or len(np.unique(ids)) != len(ids):
        raise SystemExit(f"{f}: {len(ids)} training departures, not {THIN[f]['train_departures']}")
    hv = np.array([int(hashlib.sha256(f"{SALT}|{int(i)}".encode()).hexdigest()[:15], 16)
                   for i in ids], dtype="int64")
    keep = np.sort(ids[np.argsort(hv, kind="stable")[:THIN[f]["n_keep"]]])
    np.save(f"/tmp/keep-{f}.npy", keep)
    THIN_REC[f] = {**THIN[f], "salt": SALT, "kept": int(len(keep)),
                   "kept_ids_sorted_sha256": hashlib.sha256(
                       np.ascontiguousarray(keep.astype("float64")).tobytes()).hexdigest()}
    log(f"  kept {len(keep)} of {len(ids)}  sha256 {THIN_REC[f]['kept_ids_sorted_sha256']}")

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
    tp = Path(f"/tmp/thin-{fold}.json")
    if not tp.exists():
        raise SystemExit(f"{fold}: the thinning patch never ran")
    THIN_REC[fold]["producer_rows_before_after"] = json.loads(tp.read_text())
    log(f"  {fold} producer training rows before/after thinning: "
        f"{THIN_REC[fold]['producer_rows_before_after']}")
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
        "steps": steps, "written": written, "thin": THIN_REC}
(WORK / "port.json").write_text(json.dumps(port, indent=1))
for s in steps:
    log(f"{s['label']:28s} {s['wall_s']/60:7.1f} min  maxrss {s['child_maxrss_gb']:.2f} GB")
log(f"peak GPU {peak['gpu_mib']} MiB   peak host used {peak['host_used_gb']:.2f} GB")
log("\nISO THIN OK")
