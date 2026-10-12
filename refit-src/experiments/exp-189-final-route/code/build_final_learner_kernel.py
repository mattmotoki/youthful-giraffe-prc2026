"""exp-189: the private Kaggle kernel that fits the annual learners L (exp-134) and Lp (exp-137) and exports
the proxy features, all in the final mode (`PRC_RANKING_MODE=final`: the four months of the final file).

    python3 experiments/exp-189-final-route/code/build_final_learner_kernel.py --commit <main commit> \
        --exp130 <owner/dataset> --exp134 <owner/dataset> --harvest <owner/kernel> [...] --out <kernel dir>

Copy of exp-148 `kaggle/build_learner_kernels.py` (PROVENANCE.md). Changes: one group `final` with the fits
`features` (proxy-features.parquet: 2025-01/07 and the four 2026 months) and `final` (L and Lp; train on the
twelve 2025 months, score the four 2026 months); the code is read from `git show <commit>:<path>` (not from
the working tree); `PRC_RANKING_MODE=final` in the fit environment; the dataset and kernel sources are
arguments (finaldata's final exp-130 package carries a `source/prc/data.py` with the ranking modes, and
the harvest list adds the February/June kernels). The kernel id is `<kaggle-user>/prc-exp189-learners-final`,
private (`is_private` true); after each push an anonymous request of its page must give 404.

`--mode ranking --fits final --kernel-id <kaggle-user>/prc-exp189-learners-rank` builds the check kernel of 10-08
18:45 UTC: the same code in the ranking mode (no `PRC_RANKING_MODE`) on the inputs of the saved fit
(`prc-exp148-learners-a`), so its L/Lp equal the saved vectors when the Kaggle image and the code change
nothing.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = ["experiments/exp-147-foldserve/code/learner_wp.py",
        "experiments/exp-147-foldserve/code/pilot.py",
        "prc/__init__.py", "prc/weekpair.py"]
FITS = ["features", "final"]
KERNEL_ID = os.environ.get("PRC_KAGGLE_KERNEL", "<kaggle-user>/prc-exp189-learners-final")

TEMPLATE = '''#!/usr/bin/env python3
"""exp-189 learner fits in the final mode: {fits}. Built by build_final_learner_kernel.py from main {commit};
code tgz sha256 {code_sha}."""
import base64, io, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

FITS = {fits!r}
LEARNERS = "L,Lp"
CODE_TGZ = "{blob}"
W = Path("/kaggle/working")
C = W / "code"
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_TGZ)), mode="r:gz") as t:
    t.extractall(C)
served = [p for p in Path("/kaggle/input").rglob("data.py") if p.parent.name == "prc" and p.parent.parent.name == "source"]
if len(served) != 1:
    raise RuntimeError(f"expected one attached exp-130 source/prc; found {{served}}")
for f in served[0].parent.glob("*.py"):
    if not (C / "prc" / f.name).exists():
        shutil.copy(f, C / "prc" / f.name)
env = dict(os.environ{mode_env})
# exp-189 desk row (10-08 18:50 UTC): the 2025 months with the ranking stand table (the saved fit's), the 2026
# months with the final one. The ranking table is the final table without the rows of the stand names only the
# final file holds, checked by SHA-256. The mixed build fills learner_wp's cache, which every fit then loads.
MIXED = {mixed!r}
if MIXED:
    import csv, gc, hashlib
    import pandas as pd
    os.environ["PRC_RANKING_MODE"] = "final"
    sys.path[:0] = [str(C), str(C / "experiments/exp-147-foldserve/code")]
    import learner_wp as LW
    PX = LW.setup(Path("/kaggle/input"))[0]
    final_csv = PX.exact_file("stand_geometry.csv").read_bytes()
    drop = {{tuple(k) for k in MIXED["drop_keys"]}}
    lines = final_csv.decode().splitlines(keepends=True)
    rank_csv = "".join([lines[0]] + [l for l, r in zip(lines[1:], csv.reader(lines[1:])) if (r[0], r[1]) not in drop]).encode()
    got = (hashlib.sha256(final_csv).hexdigest(), hashlib.sha256(rank_csv).hexdigest())
    if got != (MIXED["final_sha256"], MIXED["ranking_sha256"]):
        raise RuntimeError(f"stand tables {{got}} are not {{MIXED['final_sha256']}}, {{MIXED['ranking_sha256']}}")
    stands_2025 = PX._stand_map(pd.read_csv(io.BytesIO(rank_csv)))
    _bmr, _ba = PX.build_month_raw, PX.build_all
    def _bmr_mixed(month, dep, arr, point_files, stands):
        return _bmr(month, dep, arr, point_files, stands_2025 if month.startswith("2025") else stands)
    def _ba_mixed(months, rank_months):
        raw, audit = _ba(months, rank_months)
        audit["stand_tables"] = {{"2025": MIXED["ranking_sha256"], "2026": MIXED["final_sha256"],
                                  "dropped_stand_names": len(drop)}}
        return raw, audit
    PX.build_month_raw, PX.build_all = _bmr_mixed, _ba_mixed
    raw, audit = LW.raw_months_cached(PX, Path("/tmp/exp189-cache"))
    (W / "proxy-stands.json").write_text(json.dumps({{k: audit.get(k) for k in ("stand_tables", "cache_key", "cache",
                                                     "point_files_verified")}}, indent=1) + "\\n")
    del raw, audit
    gc.collect()
    print("mixed proxy cache ready", flush=True)
status = []
for fit in FITS:
    t0 = time.time()
    with open(W / f"log-{{fit}}.txt", "w") as log:
        r = subprocess.run([sys.executable, str(C / "experiments/exp-147-foldserve/code/learner_wp.py"),
                            "--fit", fit, "--input-root", "/kaggle/input", "--out", str(W / "out"),
                            "--cache", "/tmp/exp189-cache", "--learners", LEARNERS],
                           stdout=log, stderr=subprocess.STDOUT, env=env)
    status.append({{"fit": fit, "returncode": r.returncode, "seconds": round(time.time() - t0, 1)}})
    print(json.dumps(status[-1]), flush=True)
    (W / "status.json").write_text(json.dumps(status, indent=1) + "\\n")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--exp130", required=True, help="finaldata's final exp-130 package (Kaggle dataset)")
    ap.add_argument("--exp134", required=True, help="the exp-134 input package of the final mode (Kaggle dataset)")
    ap.add_argument("--harvest", nargs="+", required=True, help="every prc-harvest kernel the proxies read")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("final", "ranking"), default="final")
    ap.add_argument("--fits", nargs="+", default=FITS)
    ap.add_argument("--kernel-id", default=KERNEL_ID)
    ap.add_argument("--title", default="PRC exp189 learners final")
    ap.add_argument("--docker-image", default=None, help="pin the Kaggle image (kernel metadata docker_image)")
    ap.add_argument("--mixed-stands", action="store_true",
                    help="2025 months with the ranking stand table, 2026 months with the final one")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; write-once")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", a.commit], check=True,
                            capture_output=True, text=True).stdout.strip()
    files = {rel: subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{rel}"], check=True,
                                 capture_output=True).stdout for rel in CODE}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for rel, data in files.items():
            info = tarfile.TarInfo(rel)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            t.addfile(info, io.BytesIO(data))
    blob = buf.getvalue()
    a.out.mkdir(parents=True)
    mode_env = ', PRC_RANKING_MODE="final"' if a.mode == "final" else ""
    mixed = None
    if a.mixed_stands:
        if a.mode != "final":
            raise SystemExit("--mixed-stands is for the final mode")
        g = ROOT / "experiments/exp-127-clock-precision"
        old, fin = (g / "stand_geometry.csv").read_bytes(), (g / "stand_geometry-final.csv").read_bytes()
        keys = lambda b: {(r["airport"], r["stand_name_mvt"]) for r in csv.DictReader(io.StringIO(b.decode()))}
        mixed = {"final_sha256": hashlib.sha256(fin).hexdigest(), "ranking_sha256": hashlib.sha256(old).hexdigest(),
                 "drop_keys": sorted(keys(fin) - keys(old))}
    (a.out / "kernel.py").write_text(TEMPLATE.format(fits=a.fits, commit=commit, blob=base64.b64encode(blob).decode(),
                                                     code_sha=hashlib.sha256(blob).hexdigest()[:16],
                                                     mode_env=mode_env, mixed=mixed))
    meta = {"id": a.kernel_id, "title": a.title, "code_file": "kernel.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": [a.exp130, a.exp134],
            "kernel_sources": list(a.harvest), "competition_sources": [], "model_sources": []}
    if a.docker_image:
        meta["docker_image"] = a.docker_image
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    rec = {"commit": commit, "code_sha256": {rel: hashlib.sha256(d).hexdigest() for rel, d in files.items()},
           "tgz_sha256": hashlib.sha256(blob).hexdigest(), "fits": a.fits, "mode": a.mode, "metadata": meta,
           "mixed_stands": None if mixed is None else {k: v for k, v in mixed.items() if k != "drop_keys"} |
           {"dropped_stand_names": len(mixed["drop_keys"])}}
    (a.out / "build.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
