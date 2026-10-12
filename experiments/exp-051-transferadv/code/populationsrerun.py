"""exp-051, Sol round 2 answer: a fresh write-once rerun of populations.py under new names.  ZERO FIT.

populations.py's own log was deleted after its run, and logs/populations.log was then restored at
the same path from the terminal capture.  That broke the SPEC's "every output is write-once"
(review round 2).  Nothing can bring the deleted log back.  This script gives the restored log and
the committed outputs an independent receipt.  It runs populations.py's main() UNCHANGED, with its
two output locations pointed at new names (analysis/populations-rerun.json, and the OOF files under
artifacts/exp-051/rerun/oof/, git-ignored), and then checks:

  1. every value of the rerun JSON equals the committed analysis/populations.json, except the
     elapsed seconds;
  2. each rerun OOF file has the committed file's sha256, or else the same content;
  3. the rerun's printed lines equal the restored log's lines, except the last line's output path
     and elapsed time.

It writes the check to analysis/populations-rerun-check.json.  Its printed output is
logs/populations-rerun.log, written by a no-clobber shell redirect.

    setopt noclobber; .venv/bin/python experiments/exp-051-transferadv/code/populationsrerun.py \
        > experiments/exp-051-transferadv/logs/populations-rerun.log 2>&1
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time

import numpy as np
import pandas as pd

import common as C
from common import NO
import populations as PO

RERUN_JSON = C.ANALYSIS / "populations-rerun.json"
RERUN_OOF = C.ROOT / "artifacts/exp-051/rerun/oof"
CHECK = C.ANALYSIS / "populations-rerun-check.json"
LOG = C.EXP / "logs/populations.log"
COMMITTED_OOF = C.EXP / "oof"


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


def diff(a, b, path="") -> list:
    """Every path where two JSON trees differ (exact equality, floats included)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: missing on one side" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    PO.OUT, PO.OOF = RERUN_JSON, RERUN_OOF
    NO.refuse_existing([CHECK, RERUN_JSON] + PO.oof_paths(), overwrite=a.overwrite)
    sys.argv = [sys.argv[0]]                       # populations.main() parses no flag: write-once
    buf = Tee()
    with contextlib.redirect_stdout(buf):
        PO.main()
    printed = buf.getvalue().splitlines()

    committed = json.loads((C.ANALYSIS / "populations.json").read_text())
    rerun = json.loads(RERUN_JSON.read_text())
    json_diffs = [d for d in diff(committed, rerun) if not d.startswith("/seconds")]
    oof = {}
    for name, ent in committed["oof"].items():
        new = RERUN_OOF / name
        same_sha = C.sha256(new) == ent["sha256"]
        same_content = same_sha or pd.read_parquet(new).equals(pd.read_parquet(COMMITTED_OOF / name))
        oof[name] = {"committed_sha256": ent["sha256"], "rerun_sha256": C.sha256(new),
                     "same_sha256": same_sha, "same_content": bool(same_content)}
    json_diffs = [d for d in json_diffs if not (d.startswith("/oof/") and d.endswith("sha256")
                                                and oof[d.split("/")[2]]["same_content"])]

    restored = [ln for ln in LOG.read_text().splitlines() if not ln.startswith("#")]
    head_ok = printed[:-1] == restored[:-1]
    key = "; written to"
    last_ok = (len(printed) == len(restored)
               and printed[-1].split(key)[0] == restored[-1].split(key)[0])
    out = {"what": __doc__.strip().splitlines()[0],
           "rerun_json": str(RERUN_JSON.relative_to(C.ROOT)),
           "rerun_oof_dir": str(RERUN_OOF.relative_to(C.ROOT)),
           "json_values_equal_except_seconds": not json_diffs, "json_diffs": json_diffs[:20],
           "oof": oof, "every_oof_same_content": all(v["same_content"] for v in oof.values()),
           "restored_log": str(LOG.relative_to(C.ROOT)), "restored_log_sha256": C.sha256(LOG),
           "printed_lines": len(printed), "restored_lines": len(restored),
           "printed_equals_restored_except_last": bool(head_ok),
           "last_line_equal_before_output_path": bool(last_ok),
           "seconds": round(time.time() - t0, 1)}
    out["passes"] = bool(out["json_values_equal_except_seconds"] and out["every_oof_same_content"]
                         and head_ok and last_ok)
    C.write_json(CHECK, out, overwrite=a.overwrite)
    print(f"rerun JSON equals the committed one except seconds: {not json_diffs}; OOF same sha256: "
          f"{sum(v['same_sha256'] for v in oof.values())}/6, same content: "
          f"{sum(v['same_content'] for v in oof.values())}/6; printed lines equal the restored log: "
          f"{head_ok and last_ok} ({len(printed)} lines); PASSES {out['passes']}")
    print(f"written to {CHECK.relative_to(C.ROOT)} ({time.time()-t0:.0f}s)")
    if not out["passes"]:
        raise SystemExit("the rerun does not reproduce the committed record")


if __name__ == "__main__":
    main()
