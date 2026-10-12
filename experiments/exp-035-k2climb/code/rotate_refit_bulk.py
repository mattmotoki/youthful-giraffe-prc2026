"""Run exp-027's `refit_bulk.py` arm on a climbed member, and file it here.

Round 3's blocking finding 1: §20's form 2 put the climbed member in the **hold**
anchor and compared it with v11, whose head is the **refit-bulk** anchor, so it
changed two things at once and its "the member buys nothing" reading was
confounded.  The like-for-like cell is the climbed member through the same
refit-bulk arm, and that arm is a laptop stacker rotation — **0 GPU-h**.

Nothing is reimplemented: `experiments/exp-027-step/code/refit_bulk.py` is the
arm, its gates are its own, and this file only points `PRC_K2` at one of
`oof/<arm>-{fold}.parquet`, tags the run so it cannot collide with exp-027's
committed vectors, moves the artifacts here and records the member files with
their SHA-256.  `refit_bulk.py` now honours `PRC_TAG` exactly as `candidate.py`
does; unset it writes exp-027's own names, so the frozen path is unchanged.

    PRC_THREADS=3 python rotate_refit_bulk.py --arm mod
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
STEP = ROOT / "experiments/exp-027-step"
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--threads", default=os.environ.get("PRC_THREADS", "3"))
    a = ap.parse_args()

    tag = f"k2c-{a.arm}"
    tpl = f"experiments/exp-035-k2climb/oof/{a.arm}-{{fold}}.parquet"
    for f in FOLDS:
        if not (ROOT / tpl.format(fold=f)).exists():
            raise SystemExit(f"{tpl.format(fold=f)} is absent; run land.py --write {a.arm}")
    members = {f: {"path": tpl.format(fold=f),
                   "sha256": D.sha256(ROOT / tpl.format(fold=f))} for f in FOLDS}

    env = dict(os.environ)
    env.update(PRC_K2=tpl, PRC_TAG=tag, PRC_THREADS=str(a.threads), PYTHONPATH=str(ROOT))
    log = EXP / "logs" / f"rotate-refit-bulk-{tag}.log"
    t0 = time.time()
    print(f"[refit-bulk] arm {a.arm}  tag {tag}  member {tpl}", flush=True)
    with open(log, "w") as fh:
        p = subprocess.run([sys.executable, str(STEP / "code" / "refit_bulk.py")],
                           env=env, cwd=str(STEP / "code"), stdout=fh,
                           stderr=subprocess.STDOUT, text=True)
    print(open(log).read()[-6000:])
    if p.returncode != 0:
        raise SystemExit(f"refit_bulk.py failed rc={p.returncode}; see {log.relative_to(ROOT)}")

    moved = []
    src = STEP / "analysis" / f"refit-bulk-{tag}.json"
    dst = EXP / "analysis" / f"refit-bulk-{tag}.json"
    shutil.move(src, dst)
    moved.append(str(dst.relative_to(ROOT)))
    for q in sorted((STEP / "oof").glob(f"exp-027-{tag}-refit-bulk-*.parquet")):
        d = EXP / "oof" / q.name.replace(f"exp-027-{tag}-", f"cand-{tag}-")
        shutil.move(q, d)
        moved.append(str(d.relative_to(ROOT)))
    rec = json.loads(dst.read_text())
    rec["exp_035"] = {"arm": a.arm, "member_files": members, "moved": moved,
                      "log": str(log.relative_to(ROOT)),
                      "wall_s": round(time.time() - t0, 1)}
    dst.write_text(json.dumps(rec, indent=2))
    print(f"[refit-bulk] {rec['exp_035']['wall_s']:.0f}s -> {dst.relative_to(ROOT)}")
    for m in moved:
        print("   ", m)


if __name__ == "__main__":
    main()
