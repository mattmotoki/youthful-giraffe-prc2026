"""One fixed, label-blocked exp-066 stage controller.

The controller owns all paths and operations. There are no callbacks or caller-provided paths,
digests, groups, schemas, or thread settings. Tests may replace ``STATE_ROOT`` before construction;
the shipped CLI always uses the fixed state directory beside this experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
STATE_ROOT = EXP / "state"
STAGES = ("produce-a", "freeze-features", "freeze-null", "read-labels")
RAW_NAMES = tuple(f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet" for m in range(1, 12)) + (
    "training_2025-12-01_2026-01-01.parquet",)
FORBIDDEN_PARTS = ("exp-056-q12advocate", "exp-060-sparsecold", "exp-061-sparseprior",
                   "exp-063-sparseholdout", "exp-065-sparseholdout-retry")
SHAM_COUNT = 10
SUPPORT_NAMES = ("10", "30", "100", "300", "1000", "inf")
MONTHS = ("S", "I", "V")
DRAW_COUNT = 2000
BLOCK_LENGTHS = (1, 2, 3)
DRAW_SEED = 660505


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream: json.dump(value, stream, sort_keys=True, indent=2); stream.write("\n")
    return sha(path)


class Controller:
    def __init__(self):
        self.root = STATE_ROOT.resolve()
        self.inputs = self.root / "inputs"; self.frozen = self.root / "frozen"

    def _safe(self, relative, *, must_exist=True):
        if Path(relative).is_absolute() or ".." in Path(relative).parts: raise ValueError("traversal path")
        path = self.root / relative
        probe = path
        while probe != self.root.parent:
            if probe.is_symlink(): raise ValueError("link alias")
            if probe == self.root: break
            probe = probe.parent
        resolved = path.resolve(strict=must_exist)
        if self.root != resolved and self.root not in resolved.parents: raise ValueError("path escaped fixed root")
        if any(part in FORBIDDEN_PARTS for part in resolved.parts): raise ValueError("forbidden root")
        return resolved

    def _registration(self):
        path = self._safe("registration.json")
        record = json.loads(path.read_text())
        if set(record) != {"schema", "source_commit", "raw_sha256", "mode"} or record["schema"] != 1:
            raise ValueError("registration schema")
        commit = self._safe("GIT_COMMIT.txt").read_text().strip()
        if record["source_commit"] != commit: raise ValueError("wrong source")
        if set(record["raw_sha256"]) != set(RAW_NAMES): raise ValueError("raw contract incomplete")
        if record["mode"] not in ("synthetic-preflight", "registered"): raise ValueError("mode")
        return record

    def _gate_inputs(self):
        reg = self._registration()
        for path in self.inputs.rglob("*"):
            relative = path.relative_to(self.root)
            self._safe(relative, must_exist=True)
        for name in RAW_NAMES:
            path = self._safe(Path("inputs/raw") / name)
            if sha(path) != reg["raw_sha256"][name]: raise ValueError("raw hash mismatch")
        return reg

    def _require_manifest(self, name):
        path = self._safe(Path("frozen") / name)
        rec = json.loads(path.read_text())
        for relative, digest in rec["artifacts"].items():
            if sha(self._safe(relative)) != digest: raise ValueError("changed frozen artifact")
        return rec

    def produce_a(self):
        reg = self._gate_inputs()
        if reg["mode"] != "synthetic-preflight": raise RuntimeError("exp-066 is unregistered")
        source = self._safe("inputs/a_features.json")
        rows = json.loads(source.read_text())
        if any(set(row) - {"x", "y_synthetic"} for row in rows): raise ValueError("A input schema")
        proof = fit_thread_probes(rows)
        if set(proof) != {"lightgbm", "catboost", "tail_model"} or any(not 1 <= int(v) <= 3 for v in proof.values()):
            raise ValueError("wrong effective fitted thread setting")
        artifact = self._safe("frozen/a-only.json", must_exist=False)
        digest = write_once(artifact, {"schema": 1, "thread_proof": proof, "rows": len(rows)})
        write_once(self._safe("frozen/01-produce-manifest.json", must_exist=False),
                   {"schema": 1, "artifacts": {"frozen/a-only.json": digest}})

    def freeze_features(self):
        self._gate_inputs(); self._require_manifest("01-produce-manifest.json")
        rows = json.loads(self._safe("inputs/feature_rows.json").read_text())
        required = {"MVT_ID_mvt", "utc_date", "group", "key", *MONTHS}
        if any(set(row) != required for row in rows): raise ValueError("feature-only schema")
        ids = np.asarray([str(r["MVT_ID_mvt"]) for r in rows], dtype="U")
        order = np.argsort(ids, kind="stable"); ids=ids[order]
        dates = np.asarray([date.fromisoformat(rows[i]["utc_date"]).isoformat() for i in order], dtype="U")
        groups = np.asarray([str(rows[i]["group"]) for i in order], dtype="U")
        keys = np.asarray([str(rows[i]["key"]) for i in order], dtype="U")
        contributions = np.asarray([[float(rows[i][m]) for m in MONTHS] for i in order])
        unique_dates = np.asarray(sorted(set(dates)), dtype="U")
        parsed = [date.fromisoformat(x) for x in unique_dates]
        rng=np.random.default_rng(DRAW_SEED); arrays={"schema":np.asarray([1]),"ids":ids,"dates":dates,
            "groups":groups,"keys":keys,"contributions":contributions,"unique_dates":unique_dates}
        dimensions={}
        for length in BLOCK_LENGTHS:
            starts=[i for i in range(len(parsed)-length+1)
                    if all((parsed[i+j]-parsed[i]).days == j for j in range(length))]
            if not starts: raise ValueError(f"no non-wrapping calendar block length {length}")
            blocks=math.ceil(len(parsed)/length)
            chosen=rng.choice(starts,size=(DRAW_COUNT,blocks),replace=True)
            draw=np.stack([chosen+j for j in range(length)],axis=2)
            arrays[f"draw_l{length}"]=draw.astype("int32"); dimensions[f"draw_l{length}"]=list(draw.shape)
        artifact=self._safe("frozen/features-draws.npz",must_exist=False); artifact.parent.mkdir(parents=True,exist_ok=True)
        with artifact.open("xb") as stream: np.savez(stream,**arrays)
        digest=sha(artifact)
        manifest={"schema":1,"draw_count":DRAW_COUNT,"block_lengths":list(BLOCK_LENGTHS),
          "dimensions":dimensions,"ordered_ids":ids.tolist(),"ordered_dates":dates.tolist(),
          "required_groups":sorted(set(groups)),"support_names":list(SUPPORT_NAMES),
          "months":list(MONTHS),"artifacts":{"frozen/features-draws.npz":digest}}
        write_once(self._safe("frozen/02-freeze-manifest.json",must_exist=False),manifest)

    def _load_features(self):
        manifest=self._require_manifest("02-freeze-manifest.json")
        z=np.load(self._safe("frozen/features-draws.npz"))
        if set(z.files)!={"schema","ids","dates","groups","keys","contributions","unique_dates","draw_l1","draw_l2","draw_l3"}: raise ValueError("freeze schema")
        if z["ids"].tolist()!=manifest["ordered_ids"] or z["dates"].tolist()!=manifest["ordered_dates"]: raise ValueError("freeze identity")
        parsed=[date.fromisoformat(str(x)) for x in z["unique_dates"]]
        for length in BLOCK_LENGTHS:
            draw=z[f"draw_l{length}"]
            if list(draw.shape)!=manifest["dimensions"][f"draw_l{length}"]: raise ValueError("draw dimensions")
            for row in draw:
                for block in row:
                    if any(int(x)<0 or int(x)>=len(parsed) for x in block): raise ValueError("draw element")
                    if any((parsed[int(block[j])]-parsed[int(block[j-1])]).days!=1 for j in range(1,length)): raise ValueError("nonconsecutive block")
        return manifest,z

    def freeze_null(self):
        self._gate_inputs(); self._require_manifest("01-produce-manifest.json"); manifest,z=self._load_features()
        groups=z["groups"].astype(str); keys=z["keys"].astype(str); values=z["contributions"]
        if sorted(set(groups))!=manifest["required_groups"]: raise ValueError("required group absent")
        attempts=[]; selected=[]
        for sham in range(SHAM_COUNT):
            chosen=None
            for seed in range(1,100001):
                perm=np.arange(len(groups)); rng=np.random.default_rng([seed,sham])
                for group in manifest["required_groups"]:
                    idx=np.flatnonzero(groups==group); perm[idx]=rng.permutation(idx)
                moved=keys[perm]!=keys; affected={}; reasons=[]
                for group in manifest["required_groups"]:
                    idx=groups==group; moved_keys=int(moved[idx].sum())
                    if not moved_keys: reasons.append(f"unchanged:{group}")
                    for month_i,month in enumerate(MONTHS):
                        for support in SUPPORT_NAMES:
                            count=int(np.count_nonzero(values[perm[idx],month_i]!=values[idx,month_i]))
                            affected[f"{group}:{month}:{support}"]=count
                            if count==0: reasons.append(f"stratum:{group}:{month}:{support}")
                attempt={"sham":sham,"seed":seed,"accepted":not reasons,
                         "moved_keys":int(moved.sum()),"affected_rows":affected,"reasons":reasons}
                attempts.append(attempt)
                if not reasons: chosen={"sham":sham,"seed":seed,"permutation":perm.tolist()};break
            if chosen is None: raise RuntimeError("matched null incomplete; open (before label read)")
            selected.append(chosen)
        artifact=self._safe("frozen/matched-null.json",must_exist=False)
        digest=write_once(artifact,{"schema":1,"sham_count":SHAM_COUNT,"required_groups":manifest["required_groups"],
          "support_names":list(SUPPORT_NAMES),"months":list(MONTHS),"seed_first":1,"seed_last":100000,
          "attempts":attempts,"selected":selected})
        write_once(self._safe("frozen/03-null-manifest.json",must_exist=False),
          {"schema":1,"feature_manifest_sha256":sha(self._safe("frozen/02-freeze-manifest.json")),
           "selected_sha256":digest,"artifacts":{"frozen/matched-null.json":digest}})

    def read_labels(self):
        reg=self._gate_inputs(); self._require_manifest("01-produce-manifest.json"); self._load_features()
        null_manifest=self._require_manifest("03-null-manifest.json")
        if null_manifest["feature_manifest_sha256"]!=sha(self._safe("frozen/02-freeze-manifest.json")): raise ValueError("wrong freeze source")
        null=json.loads(self._safe("frozen/matched-null.json").read_text())
        if null["sham_count"]!=SHAM_COUNT or len(null["selected"])!=SHAM_COUNT: raise ValueError("incomplete null")
        if reg["mode"]!="synthetic-preflight": raise RuntimeError("exp-066 label route remains blocked until registration authorization")
        labels=json.loads(self._safe("inputs/synthetic_labels.json").read_text())
        if set(labels)!={"synthetic_labels"}: raise ValueError("label schema")
        return len(labels["synthetic_labels"])

    def run(self, stage):
        return {"produce-a":self.produce_a,"freeze-features":self.freeze_features,
                "freeze-null":self.freeze_null,"read-labels":self.read_labels}[stage]()


def fit_thread_probes(rows):
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    x=np.asarray([[float(r["x"])] for r in rows]); y=np.asarray([float(r["y_synthetic"]) for r in rows])
    lgb=LGBMRegressor(n_estimators=2,num_threads=3,verbosity=-1).fit(x,y)
    cat=CatBoostRegressor(
        iterations=2,
        thread_count=3,
        task_type="CPU",
        verbose=False,
        allow_writing_files=False,
    ).fit(x,y)
    tail=LGBMRegressor(n_estimators=2,n_jobs=3,verbosity=-1).fit(x,y)
    proof={"lightgbm":int(lgb.get_params()["num_threads"]),
           "catboost":int(cat.get_params()["thread_count"]),"tail_model":int(tail.get_params()["n_jobs"])}
    if any(not 1<=v<=3 for v in proof.values()): raise ValueError("wrong effective fitted thread setting")
    return proof


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=STAGES);args=parser.parse_args()
    Controller().run(args.stage)
