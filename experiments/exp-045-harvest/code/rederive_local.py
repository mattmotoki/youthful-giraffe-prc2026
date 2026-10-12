"""exp-045-harvest: run the re-derive on this laptop instead of in a Kaggle session.

The re-derive needs no internet and no archive: it reads a month's stored points and
writes the summaries again under the current rule.  A day of points is 6 MB, so a
whole month is under 200 MB in transit and nothing is kept afterwards -- while a
Kaggle session is the scarce thing, five in the account and shared with the other
workers.  So the backfill does not queue behind the fetch: it runs here.

The kernel file is executed exactly as it stands, with `CFG` pointed at local
directories, so a summary this laptop writes and a summary a session writes come
from the same source bytes and carry the same rule hash.

  .venv/bin/python experiments/exp-045-harvest/code/rederive_local.py 2026-01-gate ...

Each name is a fetch kernel's suffix, such as `2026-01-gate`; the caller supplies the account.
Refuses to start with less than MIN_FREE_GB free, and deletes the points it pulled
whether it succeeds or not.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/exp-045-harvest"
ART = ROOT / "artifacts/exp-045"
KAGGLE = str(Path.home() / ".local/bin/kaggle")
MIN_FREE_GB = 0.3          # the volume this laptop shares runs at 100 %; one day is 6 MB


def free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def run_kernel_file(kernel: Path, inp: Path, out: Path) -> None:
    """Execute the assembled kernel with its two directories redirected."""
    src = kernel.read_text()
    ns: dict = {"__name__": "__main__", "__file__": str(kernel)}
    code = src.replace('CFG = {"out": "/kaggle/working", "input": "/kaggle/input"}',
                       f'CFG = {{"out": {str(out)!r}, "input": {str(inp)!r}}}', 1)
    if code == src:
        raise SystemExit(f"{kernel}: CFG line not found, refusing to guess")
    exec(compile(code, str(kernel), "exec"), ns)


def points_days(src: str, into: Path) -> list[str]:
    """The days a fetch kernel actually delivered.  The CLI cannot list an output, so
    this pulls the per-day run records, which are a few KB, and reads the days off
    them: what the kernel wrote, not what it was asked to write."""
    subprocess.run([KAGGLE, "kernels", "output", f"{kaggle_user()}/{src}", "-p", str(into),
                    "-q", "--page-size", "200", "--file-pattern", "^run_"],
                   check=True, stdout=subprocess.DEVNULL)
    days = sorted(p.name[4:14] for p in into.glob("run_2*.json"))
    for p in into.glob("run_2*.json"):
        p.unlink()
    return days


def rederive(name: str) -> dict:
    month = name
    src = f"prc-harvest-{name}"
    t0 = time.time()
    subprocess.run([sys.executable, str(EXP / "code/assemble_rederive.py"),
                    str(ROOT), month, src], check=True)
    kernel = EXP / f"kernels/rederive-{month}/prc-rederive.py"

    (EXP / "work").mkdir(parents=True, exist_ok=True)
    if free_gb(EXP) < MIN_FREE_GB:
        raise SystemExit(f"{free_gb(EXP):.1f} GB free, under the {MIN_FREE_GB} GB floor")
    tmp = Path(tempfile.mkdtemp(dir=EXP / "work", prefix="rdl-"))
    try:
        inp = tmp / "input" / src
        out = tmp / "out"
        inp.mkdir(parents=True)
        out.mkdir()
        days = points_days(src, tmp)
        if not days:
            raise SystemExit(f"{name}: no points in {kaggle_user()}/{src}")
        print(f"{name}: {len(days)} days of points, {free_gb(EXP):.2f} GB free",
              flush=True)
        mb = 0.0
        for day in days:
            if free_gb(EXP) < MIN_FREE_GB:
                raise SystemExit(f"{free_gb(EXP):.2f} GB free, under the "
                                 f"{MIN_FREE_GB} GB floor, stopped at {day}")
            subprocess.run([KAGGLE, "kernels", "output", f"{kaggle_user()}/{src}",
                            "-p", str(inp), "-q", "--page-size", "200",
                            "--file-pattern", f"^points_{day}"], check=True,
                           stdout=subprocess.DEVNULL)
            got = sorted(inp.glob("points_*.parquet"))
            if not got:
                raise SystemExit(f"{name}: {day} did not arrive")
            mb += sum(p.stat().st_size for p in got) / 1e6
            run_kernel_file(kernel, tmp / "input", out)
            for p in got:                       # one day on disk at a time
                p.unlink()
        print(f"{name}: {mb:.0f} MB pulled in {len(days)} pulls", flush=True)
        subprocess.run([sys.executable, str(EXP / "code/land_rederive.py"),
                        str(out), str(ART), str(EXP),
                        f"local from {kaggle_user()}/{src}"], check=True)
        total = {"days": 0, "segments": 0, "departures": 0}
        for rj in sorted(out.glob("rederive_2*.json")):
            rec = json.loads(rj.read_text())
            total["days"] += 1
            total["segments"] += rec["outputs"]["segments"]["rows"]
            total["departures"] += rec["departures"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total["name"] = name
    total["seconds"] = round(time.time() - t0, 1)
    return total


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        print(json.dumps(rederive(nm)), flush=True)
