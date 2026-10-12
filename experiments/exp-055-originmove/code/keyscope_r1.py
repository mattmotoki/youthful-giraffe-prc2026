"""exp-055, Sol round 1 finding 1: which fields of exp-051's loss records iteration 2 reads.

`lossfals.py` parses the whole of exp-051's `carrier.json`, `boardlike.json` and
`populations.json` with `json.loads`, although SPEC §10.2 says that only the listed keys are
parsed.  This script runs `lossfals.py` unchanged, as a trial run into a scratch directory, with
`json.loads` replaced for those three files only by a loader that records every key read and
every other operation on each parsed object (enumeration, `get`, membership, length, repr).
It compares the recorded leaf reads with the keys that SPEC §10.2 and §10.6 list, and it
compares the trial record with the committed `analysis/lossfals.json` (every field except the
run time and the output paths).  Any read outside the lists, any other operation or any record
difference is fatal.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-055-originmove/code/keyscope_r1.py --scratch DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import common as C
import lossfals as LF

OUT = C.ANALYSIS / "keyscope-r1.json"
COMMITTED = C.ANALYSIS / "lossfals.json"
FILES = ("carrier.json", "boardlike.json", "populations.json")
OPS = ("keys", "values", "items", "get", "__iter__", "__contains__", "__len__", "__repr__", "__str__",
       "copy", "pop", "setdefault", "update")


def registered() -> set[tuple]:
    """The leaf keys that SPEC §10.2 (gains) and §10.6 (populations) list."""
    want = set()
    for k in LF.KS:
        s = ("carrier.json", "steps", C.STEP[k])
        want |= {s + ("july_s2_per_row", "C3"), s + ("july_s2_per_row", "P3"),
                 s + ("january_s2_per_row_both_arms",), s + ("december_s2_per_row_both_arms",),
                 s + ("sham_K1", "july_s2", "S"), s + ("sham_K1", "july_s2", "P1")}
        for m in C.SEEDS:
            want |= {s + ("members_record_only", str(m), "july_s2", x) for x in ("C", "P")}
        for f in C.FOLDS:
            want |= {("boardlike.json", "steps", C.STEP[k], "by_fold_s2_per_row", f, x) for x in ("BL", "P")}
    for L in range(4):
        for k51 in (f"P{L}", f"BL{L}", f"P3bag_L{L}", f"C3bag_L{L}", f"S1_L{L}"):
            for f in C.FOLDS:
                want |= {("populations.json", "vectors", k51, f, p) for p in ("all", "le3600", "gt3600", "no_nm")}
    return want


def tracked(log: dict):
    class TDict(dict):
        """A parsed JSON object that records each key read and every other operation."""

        def __init__(self, d: dict, path: tuple):
            super().__init__(d)
            self._path = path

        def __getitem__(self, k):
            v = dict.__getitem__(self, k)
            p = self._path + (str(k),)
            if isinstance(v, dict):
                log["nodes"].add(p)
                return TDict(v, p)
            log["leaves"][p] = type(v).__name__
            return v

    def other(op):
        def f(self, *a, **kw):
            log["other_operations"].append({"path": list(self._path), "op": op})
            return getattr(dict, op)(self, *a, **kw)
        return f

    for op in OPS:
        setattr(TDict, op, other(op))
    return TDict


def comparable(r: dict) -> dict:
    r = {k: v for k, v in r.items() if k != "seconds"}
    r["receipts"] = {Path(p).name: h for p, h in r["receipts"].items()}
    r["oof"] = {**r["oof"], "path": Path(r["oof"]["path"]).name}
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True, help="directory for lossfals.py's trial outputs (no repo file)")
    args = ap.parse_args()
    scratch = Path(args.scratch).resolve()
    if C.ROOT in (scratch, *scratch.parents):
        raise SystemExit("the scratch directory must be outside the repository")
    C.NO.refuse_existing([OUT], overwrite=False)
    t0 = time.time()
    texts = {(LF.E51A / f).read_text(): f for f in FILES}
    log = {"nodes": set(), "leaves": {}, "other_operations": []}
    TDict = tracked(log)
    loads = json.loads

    class JsonShim:
        """Stands in for the `json` module inside `lossfals`; wraps only the three loss records."""

        def __getattr__(self, a):
            return getattr(json, a)

        def loads(self, s, *a, **kw):
            obj = loads(s, *a, **kw)
            f = texts.get(s)
            return TDict(obj, (f,)) if f is not None else obj

    LF.json = JsonShim()
    sys.argv = ["lossfals.py", "--scratch", str(scratch)]
    LF.main()

    want = registered()
    got = set(log["leaves"])
    trial = json.loads((scratch / "analysis/lossfals.json").read_text())
    committed = json.loads(COMMITTED.read_text())
    same = json.dumps(comparable(trial), sort_keys=True) == json.dumps(comparable(committed), sort_keys=True)
    rec = {"what": __doc__.strip().splitlines()[0],
           "records": {C.rel(LF.E51A / f): C.sha_file(LF.E51A / f) for f in FILES},
           "lossfals_py_sha256": C.sha_file(Path(LF.__file__)),
           "committed_lossfals_json_sha256": C.sha_file(COMMITTED),
           "leaf_reads": len(got), "registered_leaves": len(want),
           "leaf_reads_by_file": {f: sum(p[0] == f for p in got) for f in FILES},
           "reads_outside_registered_list": sorted(map(list, got - want)),
           "registered_leaves_not_read": sorted(map(list, want - got)),
           "leaf_types": sorted(set(log["leaves"].values())),
           "nodes_read": sorted(map(list, log["nodes"])),
           "other_operations": log["other_operations"],
           "trial_record_equals_committed_except_time_and_paths": same,
           "limits": "records Python-level operations on the parsed objects; the three objects "
                     "were parsed whole, so every field was in memory during the run"}
    rec["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, rec)
    print(f"leaf reads {len(got)} (carrier {rec['leaf_reads_by_file']['carrier.json']}, boardlike "
          f"{rec['leaf_reads_by_file']['boardlike.json']}, populations {rec['leaf_reads_by_file']['populations.json']}); "
          f"registered {len(want)}; outside the list {len(got - want)}; registered not read {len(want - got)}; "
          f"other operations {len(log['other_operations'])}; trial record equals committed: {same}; "
          f"written {C.rel(OUT)}", flush=True)
    if got - want or log["other_operations"] or not same:
        raise SystemExit("the read scope or the record is not as registered")


if __name__ == "__main__":
    main()
