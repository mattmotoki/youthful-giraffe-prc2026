"""One stage from two or more seed sets of the same fit: the mean over all their seeds.

    $PY code/merge.py --dir out/fits-HB1 --parts HB1 HB1s2 --stage HB1k10 --splits jjrank jjrefit janjul

Each part is a stage that `fit.py` wrote with the same arm, split, frame, parameters and rounds
and with other seeds. The merged stage has the form of one fit with all the seeds: the anchor plus
the float32 mean of the seed vectors in seed order. Checks before a file is written: the parts
pass `pilot.load`; their records agree on every field that defines the fit; the seeds do not
repeat; the ids and the anchors are equal. Write-once.

The merged stage is not bit-equal to one `fit.py` run with all the seeds. `fit.py` builds one
LightGBM dataset per run, and the first seed of the run sets its bins (the bin sample). The seed
vectors of a part that starts at seed 606 are thus not the vectors that seeds 606-610 give in a
run that starts at seed 601 (2026-10-02 check: RMS 34 s between them, two seeds of one run differ
by 37 s; the two ten-seed bags read G +11.8 [-46.6, +65.5] against each other). Use the same
parts for every split of a stage.

A part may be `dir/stage` when it is not in `--dir`. `--mixed-settings` allows parts whose
LightGBM parameters differ (a bag over tree shapes, SPEC-push addendum 6): every other field must
still agree, and the record holds the parameters of each part (`fit.params_by_part`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C
import pilot as P                     # noqa: E402

SAME = ("arm", "target", "learner", "lightgbm", "features_sha256", "train_rows", "train_rows_by_month",
        "train_split", "frame_set", "inputs", "trees_used", "split", "domain", "class", "owner", "rows",
        "source_commit", "ranking_mode")
SAME_FIT = ("params", "rounds", "num_threads", "categorical", "drop_months")
CARRY = ("arm", "target", "learner", "lightgbm", "numpy", "pandas", "features", "features_sha256", "train_rows",
         "frame_set", "train_rows_by_month", "train_split", "class", "inputs", "source_commit", "domain",
         "trees_used", "ranking_mode")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--mixed-settings", action="store_true", help="the parts may differ in their LightGBM parameters")
    a = ap.parse_args()
    where = {Path(p).name: (Path(p).parent if "/" in p else a.dir) for p in a.parts}
    a.parts = [Path(p).name for p in a.parts]
    if len(set(a.parts)) < 2 or len(set(a.parts)) != len(a.parts) or a.stage in a.parts:
        raise SystemExit("two or more parts with different names, and a new stage name")
    same_fit = tuple(k for k in SAME_FIT if not (a.mixed_settings and k == "params"))
    for n in a.splits:
        C.refuse_existing(a.dir / f"{a.stage}-{n}.parquet", a.dir / f"{a.stage}-{n}.json",
                          a.dir / f"{a.stage}-{n}.seeds.npz")
    for n in a.splits:
        recs = [json.loads((where[p] / f"{p}-{n}.json").read_text()) for p in a.parts]
        for p, r in zip(a.parts[1:], recs[1:]):
            for k in SAME:
                if r.get(k) != recs[0].get(k):
                    raise SystemExit(f"{n}: part {p} differs from {a.parts[0]} in {k}")
            for k in same_fit:
                if r["fit"].get(k) != recs[0]["fit"].get(k):
                    raise SystemExit(f"{n}: part {p} differs from {a.parts[0]} in fit.{k}")
        fit_owner = recs[0]["owner"]
        frames = [P.load(where[p] / f"{p}-{n}.parquet", fit_owner) for p in a.parts]
        npz = [np.load(where[p] / f"{p}-{n}.seeds.npz") for p in a.parts]
        seeds = [int(s) for z in npz for s in z["seeds"]]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"{n}: a seed is in two parts: {seeds}")
        for p, r, z in zip(a.parts, recs, npz):
            if [int(s) for s in z["seeds"]] != r["fit"]["seeds"]:
                raise SystemExit(f"{n}: the seed vectors of {p} are not the seeds of its record")
        ids, anchor = npz[0]["ids"], npz[0]["anchor"]
        for p, z, f in zip(a.parts, npz, frames):
            if not (np.array_equal(z["ids"], ids) and np.array_equal(z["anchor"], anchor)
                    and np.array_equal(f[C.ID].to_numpy("int64"), ids)):
                raise SystemExit(f"{n}: part {p} has other rows or another anchor")
            back = anchor + z["z"].mean(0).astype("float64")
            if not np.array_equal(back, f[f"pred_{p}"].to_numpy("float64")):
                raise SystemExit(f"{n}: the seed vectors of {p} do not give its prediction")
        order = np.argsort(seeds, kind="stable")
        zz = np.concatenate([z["z"] for z in npz])[order]
        seeds = [seeds[i] for i in order]
        pred = frames[0][[C.ID, "month", "utc_day", "airport"]].reset_index(drop=True)
        pred[f"pred_{a.stage}"] = anchor + zz.mean(0).astype("float64")
        np.savez_compressed(a.dir / f"{a.stage}-{n}.seeds.npz", ids=ids, anchor=anchor, z=zz, seeds=np.array(seeds))
        rec = {k: recs[0][k] for k in CARRY if k in recs[0]}
        secs = dict(zip([int(s) for z in npz for s in z["seeds"]], [t for r in recs for t in r["fit"]["seconds_per_seed"]]))
        rec["fit"] = dict({k: recs[0]["fit"][k] for k in same_fit}, seeds=seeds,
                          seconds_per_seed=[secs[s] for s in seeds])
        if a.mixed_settings:
            rec["fit"]["params"] = "mixed: see params_by_part"
            rec["fit"]["params_by_part"] = {p: r["fit"]["params"] for p, r in zip(a.parts, recs)}
        rec["merged_from"] = {p: {"output_sha256": r["output_sha256"], "seeds": r["fit"]["seeds"],
                                  "code_sha256": r["code_sha256"]} for p, r in zip(a.parts, recs)}
        rec["code_sha256"] = {"merge.py": C.sha(__file__), "common.py": C.sha(C.__file__)}
        P.write(a.dir, a.stage, P.split(n), fit_owner, pred, rec)
        print(f"wrote {a.stage}-{n}: {len(pred)} rows, seeds {seeds}", flush=True)


if __name__ == "__main__":
    main()
