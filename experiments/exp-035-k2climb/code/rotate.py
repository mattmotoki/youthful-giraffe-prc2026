"""Run exp-027's candidate harness on a climbed member, and file its artifacts here.

Nothing is reimplemented. `experiments/exp-027-step/code/candidate.py` is the
rotation, its gates A/B/C are its own, and this file only

  * points `PRC_K2` at one of `oof/<arm>-{fold}.parquet`,
  * gives the run a tag that cannot collide with exp-027's,
  * moves the fold vectors and the json it writes into this experiment, and
  * records which member files went in, with their SHA-256.

That last part is the reason this exists rather than a shell line: a rotation
whose inputs are not pinned is not a measurement anyone can re-read.

    python rotate.py --arm seeds
    python rotate.py --arm seeds --identity     # PRC_K2 -> F1-fresh's own files
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
IDENTITY = "experiments/exp-012-blend2/published/F1-fresh-s1-{fold}.parquet"
#: exp-027's own committed K2 member.  `--k2base` reruns the rotation on it, which
#: (a) reproduces exp-027's published arm numbers as a gate and (b) saves the row
#: vectors for the two arms exp-027 did not save, so a climbed arm can be priced
#: against K2 with the rotation's own clustered interval and not only as a scalar.
K2BASE = "experiments/exp-027-step/oof/k2-window-{fold}.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--k2base", action="store_true")
    ap.add_argument("--threads", default=os.environ.get("PRC_THREADS", "3"))
    a = ap.parse_args()

    tag = f"k2c-{a.arm}" + ("-identity" if a.identity else "")
    if a.identity:
        tpl = IDENTITY
    elif a.k2base:
        tpl = K2BASE
    else:
        tpl = f"experiments/exp-035-k2climb/oof/{a.arm}-{{fold}}.parquet"
        for f in FOLDS:
            p = ROOT / tpl.format(fold=f)
            if not p.exists():
                raise SystemExit(f"{p.relative_to(ROOT)} is absent; run land.py --write {a.arm}")
    members = {f: {"path": tpl.format(fold=f),
                   "sha256": D.sha256(ROOT / tpl.format(fold=f))} for f in FOLDS}

    env = dict(os.environ)
    env["PRC_K2"] = tpl
    env["PRC_TAG"] = tag
    env["PRC_THREADS"] = str(a.threads)
    env["PYTHONPATH"] = str(ROOT)
    log = EXP / "logs" / f"rotate-{tag}.log"
    t0 = time.time()
    print(f"[rotate] arm {a.arm}  tag {tag}  member {tpl}", flush=True)
    with open(log, "w") as fh:
        p = subprocess.run([sys.executable, str(STEP / "code" / "candidate.py")],
                           env=env, cwd=str(STEP / "code"), stdout=fh,
                           stderr=subprocess.STDOUT, text=True)
    print(open(log).read()[-8000:])
    if p.returncode != 0:
        raise SystemExit(f"candidate.py failed with rc={p.returncode}; see {log.relative_to(ROOT)}")

    moved = []
    src = STEP / "analysis" / f"candidate-{tag}.json"
    dst = EXP / "analysis" / f"candidate-{tag}.json"
    shutil.move(src, dst)
    moved.append(str(dst.relative_to(ROOT)))
    for q in sorted((STEP / "oof").glob(f"exp-027-{tag}-*.parquet")):
        d = EXP / "oof" / q.name.replace(f"exp-027-{tag}-", f"cand-{tag}-")
        shutil.move(q, d)
        moved.append(str(d.relative_to(ROOT)))
    rec = json.loads(dst.read_text())
    rec["exp_035"] = {"arm": a.arm, "identity": a.identity, "k2base": a.k2base, "member_files": members,
                      "log": str(log.relative_to(ROOT)), "moved": moved,
                      "wall_s": round(time.time() - t0, 1)}
    dst.write_text(json.dumps(rec, indent=2))
    print(f"[rotate] {rec['exp_035']['wall_s']:.0f}s -> {dst.relative_to(ROOT)}")
    for m in moved:
        print("   ", m)


if __name__ == "__main__":
    main()
