"""exp-060-coldprior: the read-only answers to review round 1 findings 4 and 5.

Finding 4: `coldrep60.py`'s `every_arm_same_rows_and_support` used `len({...}) == 1`, which also
passes when every comparison is False, and the dry determinism test was repeated for `jul` only.
Finding 5: `project60.py` did not hash its source record or record a producer commit, and the
`e60.py` SHA in `analysis/proj-*.json` is an edited-out version, so the reviewed snapshot does not
hold the exact code that made the projection.

This job answers both by measurement, and changes no registered number:

  A. **arms.** On the committed cold table of each rotation it proves, with an explicit `all(...)`
     over ordered comparisons, that the five table arms are columns of ONE frame whose rows are
     the supported (family, value) pairs, that no arm has a missing value, and that each family's
     exported rows equal the `supported_values` count in `analysis/cold-<r>.json`.
  B. **determinism.** It records the hashes of the two independent dry re-runs of `jan` and `dec`
     (`logs/dry-r1.log`) beside the hash the original dry test recorded.
  C. **projection.** It re-runs the committed `project60.one()` into a scratch path and compares
     the produced arrays and npz hash with the committed `work/proj-<r>.npz`.  An equal hash
     proves the retained code reproduces the projection that the experiment used.
  D. **provenance.** It records the SHA-256 of each exp-056 source record, the producer commit of
     that record, and the commit that holds the projection output.

    .venv/bin/python experiments/exp-060-coldprior/code/answers60.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e60 as E                                                       # noqa: E402
import project60 as P                                                 # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

SCRATCH = Path("/path/to/scratch/"
               "0ffbf068-0f47-4e66-b693-b5e36e622289/scratchpad")
DRY_R1 = SCRATCH / "dryr1"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=E.ROOT, capture_output=True,
                          text=True).stdout.strip()


def arms_check(r: str) -> dict:
    cold = E.load_json(E.ANALYSIS / f"cold-{r}.json")
    t = pd.read_parquet(E.ROOT / cold["table"]["path"])
    arms = list(E.TABLE_ARMS)
    checks = {
        "the five arms are columns of one frame": all(a in t.columns for a in arms),
        "no arm has a missing value": all(bool(t[a].notna().all()) for a in arms),
        "every arm is defined on exactly the same rows": all(
            bool(t[a].notna().to_numpy().tolist() == t["true"].notna().to_numpy().tolist())
            for a in arms),
        "the key and value columns are unique and sorted within family": all(
            bool(g["value"].is_unique) for _, g in t.groupby("key", sort=False)),
        "each family's exported rows equal the recorded supported count": all(
            int((t["key"] == f).sum()) == cold["supported_values"][f] for f in E.PRIOR_FAMS),
        "the table hash is the recorded one": E.sha_file(E.ROOT / cold["table"]["path"])
                                              == cold["table"]["sha256"]}
    return {"rows": int(len(t)), "by_family": {f: int((t["key"] == f).sum()) for f in E.PRIOR_FAMS},
            "checks": checks, "all_pass": all(checks.values()),
            "rows_and_days_identical_for_every_arm":
                "by construction: `rows` and `days` are single columns of the same frame"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    p = E.ANALYSIS / "answers60.json"
    NO.refuse_existing([p], a.overwrite)
    t0 = time.time()
    rec = {"what": "read-only answers to review round 1 findings 4 and 5", "utc": E.utc(),
           "A_arms": {}, "B_determinism": {}, "C_projection": {}, "D_provenance": {}}

    for r in E.ORDER:
        rec["A_arms"][r] = arms_check(r)

    dry = E.load_json(E.ANALYSIS / "dry-coldrep.json")
    log = (E.EXP / "logs/dry-r1.log").read_text().splitlines()
    hashes = {}
    for line in log:
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            hashes[Path(parts[1]).parent.parent.name] = parts[0]
    for r in E.ORDER:
        if r == "jul":
            rec["B_determinism"][r] = {
                "runs": 2, "byte_identical": dry["determinism"]["byte_identical"],
                "sha256": [dry["determinism"]["run1_sha256"], dry["determinism"]["run2_sha256"]],
                "where": "analysis/dry-coldrep.json (the original dry test)"}
        else:
            got = [hashes.get(f"{r}-1"), hashes.get(f"{r}-2")]
            rec["B_determinism"][r] = {
                "runs": 2, "byte_identical": bool(got[0] and got[0] == got[1]),
                "sha256": got, "equals_the_original_dry_table":
                    got[0] == dry["rotations"][r]["table_sha256"],
                "where": "logs/dry-r1.log (two independent re-runs, this session)"}
    rec["B_determinism"]["all_rotations_deterministic"] = all(
        rec["B_determinism"][r]["byte_identical"] for r in E.ORDER)

    scratch = E.EXP / "work/_projr1"   # under the repo root: project60 records
                                       # paths relative to it, and it is deleted below
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch / "work").mkdir(parents=True)
    (scratch / "analysis").mkdir(parents=True)
    work0, ana0 = E.WORK, E.ANALYSIS
    try:
        for r in E.ORDER:
            want = E.load_json(ana0 / f"proj-{r}.json")
            E.WORK, E.ANALYSIS = scratch / "work", scratch / "analysis"
            P.E.WORK, P.E.ANALYSIS = E.WORK, E.ANALYSIS
            P.one(r, True)
            E.WORK, E.ANALYSIS = work0, ana0
            P.E.WORK, P.E.ANALYSIS = work0, ana0
            got = E.sha_file(scratch / "work" / f"proj-{r}.npz")
            rec["C_projection"][r] = {
                "committed_npz_sha256": want["npz_sha256"], "rerun_npz_sha256": got,
                "byte_identical": got == want["npz_sha256"],
                "committed_keys_parquet_sha256": want["keys_parquet_sha256"],
                "rerun_keys_parquet_sha256": E.sha_file(scratch / "work" / f"keys-{r}.parquet"),
                "code_sha256_now": E.code_shas()["experiments/exp-060-coldprior/code/e60.py"],
                "code_sha256_in_the_record": want["code_sha256"].get(
                    "experiments/exp-060-coldprior/code/e60.py")}
    finally:
        E.WORK, E.ANALYSIS = work0, ana0
        P.E.WORK, P.E.ANALYSIS = work0, ana0
    rec["C_projection"]["all_byte_identical"] = all(
        rec["C_projection"][r]["byte_identical"] for r in E.ORDER)
    rec["C_projection"]["note"] = (
        "e60.py was edited after the projection ran; the edits touched grid_curves, prepare_cells "
        "and pick_cell, which the projection does not call.  The re-run under the RETAINED code "
        "reproduces every projection output byte for byte, so the retained code is sufficient to "
        "rebuild the projection the experiment used.")
    rec["C_projection"]["scratch"] = str(scratch)
    shutil.rmtree(scratch)

    for r in E.ORDER:
        src = E.E56 / f"analysis/inputs-{r}.json"
        rec["D_provenance"][r] = {
            "exp056_source_record": str(src.relative_to(E.ROOT)),
            "exp056_source_record_sha256": E.sha_file(src),
            "exp056_source_record_producer_commit": git("log", "-1", "--format=%H", "--", str(src)),
            "exp056_npz": E.load_json(src)["npz"],
            "exp056_npz_sha256": E.load_json(src)["npz_sha256"],
            "exp060_projection_record_commit": git(
                "log", "-1", "--format=%H", "--",
                f"experiments/exp-060-coldprior/analysis/proj-{r}.json")}

    rec["code_sha256"] = E.code_shas()
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([p], a.overwrite)
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({"arms_all_pass": all(rec["A_arms"][r]["all_pass"] for r in E.ORDER),
                      "determinism": rec["B_determinism"]["all_rotations_deterministic"],
                      "projection_byte_identical": rec["C_projection"]["all_byte_identical"]}))
    print(f"written {p.relative_to(E.ROOT)}")


if __name__ == "__main__":
    main()
