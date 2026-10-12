"""exp-045-harvest: does the laptop compute the same summaries as a Kaggle session?

Every month's summaries are now written here rather than in a session (SPEC.md
decision 3), so "the same bytes" has to be a measurement: pandas and numpy are not
the same versions on the two machines.

The test compares like with like.  It does NOT run today's sources against a kernel
that ran yesterday's -- that measures the edit, not the machine.  It takes the rule
block **out of the assembled kernel file that actually ran**, executes that block
here, and compares its output against the `segments_<day>.parquet` the session wrote
from the same points.  Any difference is then the machine and nothing else.

  .venv/bin/python experiments/exp-045-harvest/code/check_machine.py 2026-01-gate

Writes analysis/machine_check.json.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/exp-045-harvest"
KAGGLE = str(Path.home() / ".local/bin/kaggle")
KEYS = ["airport", "icao", "seg_start"]
HEAD = '"""exp-043-pilot:'          # the first line of the inlined rule
TAIL = "RULE_SHA256 = "             # the assembler writes this straight after it


def rule_block(kernel: Path) -> tuple[dict, str]:
    """The rule exactly as that kernel ran it, executed in its own namespace."""
    src = kernel.read_text()
    i, j = src.index(HEAD), src.index(TAIL)
    block = src[i:j]
    ns: dict = {"__name__": "kernel_rule"}
    exec(compile(block, f"{kernel}:rule", "exec"), ns)
    sha = src[j:].split('"')[1]
    return ns, sha


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "2026-01-gate"
    src = f"prc-harvest-{name}"
    kernel = EXP / f"kernels/surface-{name}/prc-harvest.py"
    ns, sha = rule_block(kernel)
    all_segments = ns["all_segments"]
    req = ns.get("REQUIRE_ALT", False)   # the shipped rule had no such constant
    print(f"{kernel.name}: rule sha256 {sha[:16]}, REQUIRE_ALT={req}", flush=True)

    (EXP / "work").mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=EXP / "work", prefix="mach-"))
    try:
        subprocess.run([KAGGLE, "kernels", "output", f"{kaggle_user()}/{src}", "-p", str(tmp),
                        "-q", "--page-size", "200",
                        "--file-pattern", "^(points|segments)_"],
                       check=True, stdout=subprocess.DEVNULL)
        report = {"kernel": f"{kaggle_user()}/{src}", "rule_sha256": sha,
                  "require_alt": bool(req),
                  "reads": "the rule block taken out of the assembled kernel that ran",
                  "days": {}}
        for pp in sorted(tmp.glob("points_*.parquet")):
            day = pp.name[7:17]
            kag = pd.read_parquet(tmp / f"segments_{day}.parquet")
            pts = pd.read_parquet(pp)
            here = all_segments(pts)
            # the points that do not order themselves: same (airport, icao, t), and the
            # session sorted them with pandas' default, which is not stable
            dup = pts.duplicated(subset=["airport", "icao", "t"], keep=False)
            tied = set(map(tuple, pts.loc[dup, ["airport", "icao"]].astype(str).values))
            cols = [c for c in kag.columns if c in here.columns]
            a = kag[cols].sort_values(KEYS).reset_index(drop=True)
            b = here[cols].sort_values(KEYS).reset_index(drop=True)
            same = a.shape == b.shape and a.equals(b)
            diff = ([c for c in cols if not a[c].equals(b[c])]
                    if not same and a.shape == b.shape else [])
            # every row the two machines do not share, and how many of them belong to an
            # aircraft whose points do not order themselves
            ka = set(map(tuple, kag[KEYS].astype(str).values))
            ha = set(map(tuple, here[KEYS].astype(str).values))
            only = (ka - ha) | (ha - ka)
            from_tie = sum(1 for r in only if (r[0], r[1]) in tied)
            settled = a.drop(index=[i for i, r in a[KEYS].astype(str).iterrows()
                                    if (r.airport, r.icao) in tied], errors="ignore")
            report["days"][day] = {"kaggle_rows": int(len(kag)), "laptop_rows": int(len(here)),
                                   "columns_compared": len(cols), "identical": bool(same),
                                   "columns_that_differ": diff,
                                   "rows_not_shared": len(only),
                                   "rows_not_shared_from_a_tied_timestamp": from_tie,
                                   "points": int(len(pts)), "tied_points": int(dup.sum()),
                                   "aircraft_with_a_tied_timestamp": len(tied),
                                   "settled_rows_compared": int(len(settled))}
            print(f"{day}: kaggle {len(kag)} rows, laptop {len(here)} rows; "
                  f"{len(only)} rows not shared, {from_tie} of them on an aircraft whose "
                  f"points share a timestamp ({int(dup.sum())} of {len(pts)})", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    report["every_difference_is_a_tied_timestamp"] = all(
        d["rows_not_shared"] == d["rows_not_shared_from_a_tied_timestamp"]
        for d in report["days"].values())
    report["identical_everywhere"] = all(d["identical"] for d in report["days"].values())
    out = EXP / "analysis/machine_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {os.path.relpath(out, ROOT)}: "
          f"every_difference_is_a_tied_timestamp="
          f"{report['every_difference_is_a_tied_timestamp']}")


if __name__ == "__main__":
    main()
