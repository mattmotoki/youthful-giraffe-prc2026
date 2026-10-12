"""exp-054-arrivalrepr: the check of the review-repair re-run (SPEC amendment A1; Sol round 1, finding 1).

Compares every output of the re-run (`rerun/`) with the recorded output of the same name.

- Every parquet (`work/rep-*`, `work/h-*`, `oof/iter1-*`; at --stage all also `work/h2-*`,
  `oof/iter2-*`): the file sha256 and, separately, the frame (`DataFrame.equals`, same columns
  and dtypes).
- Every analysis record (`rep-*`, `fit-*`, `iter1`; at --stage all also `fit2-*`, `iter2`): every
  leaf, exactly (two NaN count as equal).  A leaf may differ only when its key names a clock, a
  load, a duration, a path, or the sha of a record that holds clocks (VOLATILE).  The re-run's
  records may add only the keys in RERUN_ONLY.  Any other difference fails the check (exit 1).
- The order of reads in the re-run, from its own records: per fold, the representation frozen
  before the held month's arrival outcomes are read, and the arm matrices frozen before the first
  label read, for both iterations.

At --stage iter1 this is the gate before `fit2.py`: `fit2.py` and `price2.py` read the
experiment's own iteration-1 representation, fit record and h files, so the re-run's files must
be the same bytes.

    .venv/bin/python experiments/exp-054-arrivalrepr/code/rerun_check.py --stage iter1
    .venv/bin/python experiments/exp-054-arrivalrepr/code/rerun_check.py --stage all
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e54 as E                                                       # noqa: E402

NO = E.NO
RERUN = E.EXP / "rerun"
VOLATILE = {"launch_utc", "launch_load_1m", "seconds", "started_utc", "unix_time", "utc",
            "frozen_unix", "frozen_utc", "label_first_read_unix",
            "held_month_arrival_outcomes_first_read_unix", "representation_frozen_unix",
            "path", "fit_json", "fit2_json_sha256"}
RERUN_ONLY = [("rerun",), ("code_sha256", "e54r.py"), ("code_sha256", "represent_r.py")]


def files(stage: str) -> tuple[list[str], list[str]]:
    pq = [f"work/rep-{f}.parquet" for f in E.FOLDS] + [f"work/h-{f}.parquet" for f in E.FOLDS] \
        + [f"oof/iter1-{f}.parquet" for f in E.FOLDS]
    js = [f"analysis/rep-{f}.json" for f in E.FOLDS] + [f"analysis/fit-{f}.json" for f in E.FOLDS] \
        + ["analysis/iter1.json"]
    if stage == "all":
        pq += [f"work/h2-{f}.parquet" for f in E.FOLDS] + [f"oof/iter2-{f}.parquet" for f in E.FOLDS]
        js += [f"analysis/fit2-{f}.json" for f in E.FOLDS] + ["analysis/iter2.json"]
    return pq, js


def leaves(x, p=()):
    if isinstance(x, dict) and x:
        for k, v in x.items():
            yield from leaves(v, p + (str(k),))
    elif isinstance(x, list) and x:
        for i, v in enumerate(x):
            yield from leaves(v, p + (str(i),))
    else:
        yield p, x


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def compare_json(rel: str) -> dict:
    a = dict(leaves(json.loads((E.EXP / rel).read_text())))
    b = dict(leaves(json.loads((RERUN / rel).read_text())))
    bad, vol, eq = [], [], 0
    for k in a.keys() & b.keys():
        if same(a[k], b[k]):
            eq += 1
        else:
            (vol if k[-1] in VOLATILE else bad).append("/".join(k))
    for k in a.keys() - b.keys():
        bad.append("missing in re-run: " + "/".join(k))
    extra = []
    for k in b.keys() - a.keys():
        (extra if any(k[:len(r)] == r for r in RERUN_ONLY) else bad).append("/".join(k))
    return {"leaves": len(a), "equal": eq,
            "volatile_differences": sorted(vol), "rerun_only_keys": sorted(extra),
            "substantive_differences": sorted(bad), "pass": not bad}


def compare_parquet(rel: str) -> dict:
    fa, fb = E.EXP / rel, RERUN / rel
    sa, sb = E.sha_file(fa), E.sha_file(fb)
    A, B = pd.read_parquet(fa), pd.read_parquet(fb)
    frame = list(A.columns) == list(B.columns) and list(A.dtypes) == list(B.dtypes) and A.equals(B)
    return {"sha256_recorded": sa, "sha256_rerun": sb, "file_equal": sa == sb, "frame_equal": bool(frame),
            "rows": int(len(A)), "pass": bool(frame)}


def order(stage: str) -> dict:
    out = {}
    for f in E.FOLDS:
        rj = json.loads((RERUN / f"analysis/rep-{f}.json").read_text())
        fj = json.loads((RERUN / f"analysis/fit-{f}.json").read_text())
        o = {"rep_frozen_utc": rj["frozen"]["utc"],
             "rep_frozen_before_held_arrival_outcomes": rj["frozen"]["unix_time"]
             < rj["held_month_arrival_outcomes_first_read_unix"],
             "rep_reader": rj["rerun"]["reader"],
             "fit_matrices_frozen_utc": fj["matrices"]["frozen_utc"],
             "fit_matrices_frozen_before_label": fj["matrices"]["frozen_unix"] < fj["label_first_read_unix"],
             "fit_after_every_rep_frozen": all(
                 fj["matrices"]["frozen_unix"] > json.loads((RERUN / f"analysis/rep-{g}.json").read_text())
                 ["frozen"]["unix_time"] for g in E.FOLDS)}
        if stage == "all":
            f2 = json.loads((RERUN / f"analysis/fit2-{f}.json").read_text())
            o["fit2_matrices_frozen_utc"] = f2["matrices"]["frozen_utc"]
            o["fit2_matrices_frozen_before_label"] = f2["matrices"]["frozen_unix"] < f2["label_first_read_unix"]
        out[f] = o
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["iter1", "all"], required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = E.EXP / ("analysis/rerun-check-iter1.json" if args.stage == "iter1" else "analysis/rerun-check.json")
    NO.refuse_existing([OUT], args.overwrite)
    pq, js = files(args.stage)
    rec = {"what": "exp-054 review-repair re-run check (SPEC amendment A1)", "stage": args.stage,
           "code_sha256": {n: E.sha_file(E.HERE / n) for n in ("rerun_check.py", "e54.py")},
           "volatile_keys": sorted(VOLATILE), "rerun_only_keys": ["/".join(r) for r in RERUN_ONLY],
           "parquet": {p: compare_parquet(p) for p in pq}, "records": {j: compare_json(j) for j in js},
           "order": order(args.stage), "checked_utc": E.utc()}
    ok_order = all(v for o in rec["order"].values() for k, v in o.items() if isinstance(v, bool))
    rec["pass"] = (all(v["pass"] for v in rec["parquet"].values())
                   and all(v["pass"] for v in rec["records"].values()) and ok_order)
    for p, v in rec["parquet"].items():
        print(f"  {p:32s} file equal {v['file_equal']}  frame equal {v['frame_equal']}", flush=True)
    for j, v in rec["records"].items():
        print(f"  {j:32s} leaves {v['leaves']:5d}  volatile {len(v['volatile_differences']):4d}  "
              f"substantive {v['substantive_differences'][:5]}", flush=True)
    print(f"  order {rec['order']}", flush=True)
    print(f"PASS {rec['pass']}", flush=True)
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"written {OUT.relative_to(E.ROOT)}")
    if not rec["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
