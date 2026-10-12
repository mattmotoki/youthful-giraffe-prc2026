"""exp-056-q12advocate: Stage 0 lineage, revision 1 -- the replay nodes completed (review round 1, finding 7).

lineage.py's 43 historical nodes stand as built (`analysis/dag.json`, referenced by hash). Its
replay nodes lacked producer commits, feature months, schemas, source and output hashes, had no
census, and marked every member fit `excludes_S_and_V: true`, the exposure fits included. This
job rebuilds the replay part:

- one node per member fit (45: 3 members x 3 rotations x v, s, i, x, plus the matched-size s_m or
  x_m), with the kernel slug and its pushed script (hash), the code commit (the code dataset is a
  `git archive` of it), the producer entry files and their repo-local import closure with each
  file's SHA-256 at that commit, the command template, train and predicted months, the feature
  rule, the label loads inside the kernel, output files (SHA-256, rows, parquet schema from the
  footer only), the recorded cost, and S, I and V exclusion flags computed from the train months;
- the historical recipe constants each member keeps (copied from its historical node) and the
  statement that they were chosen on folds that include S or V, so isolation item 1 is not met;
- nodes for the freeze, the bases, the correction fits of both iterations, pricing (one process
  for all three V months), the post-hoc jobs and the prior tables, each with its code file hash
  and last commit, label months, inputs and outputs (hash, size), and every edge.

Outputs `analysis/dag-r1.json` and `analysis/artifacts-r1.json` (artifacts.json's paths plus
every replay path). No label is read: files are hashed by bytes and parquet files are opened for
their footer only.

    .venv/bin/python experiments/exp-056-q12advocate/code/lineage_r1.py
"""
from __future__ import annotations

import os

import ast
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

EXP = "experiments/exp-056-q12advocate"
DAG0, ARTS0 = E.ANALYSIS / "dag.json", E.ANALYSIS / "artifacts.json"
DAG, ARTS = E.ANALYSIS / "dag-r1.json", E.ANALYSIS / "artifacts-r1.json"
HIST = {"F1-fresh": "member:F1-fresh-s1", "T-tailv4": "member:T-tailv4", "L2-hub8725": "member:L2-hub8725"}
ENTRY = {"F1-fresh": ["experiments/exp-018-fresh/code/run.py"],
         "T-tailv4": ["experiments/exp-009-noac2/deliver.py"],
         "L2-hub8725": ["experiments/exp-008-diversity/run.py"]}
THIN = {"jul": ("iso_jul_xm", "iso_jul_x"), "jan": ("iso_jan_sm", "iso_jan_s"), "dec": ("iso_dec_sm", "iso_dec_s")}
FEATURES = "each month's features are built from that month's own departures after the valid-month mask " \
           "(prc/data.py masks BLOCK_TIME_UTC_mvt, TAXITIME_SEC_mvt and descendants before features)"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=E.ROOT, capture_output=True, text=True, check=True).stdout


def git_bytes(commit: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=E.ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def last_commit(path: str) -> str | None:
    out = git("log", "-1", "--format=%H", "--", path).strip()
    return out or None


def closure(commit: str, entries: list[str]) -> dict:
    """Repo-local import closure at `commit`: modules found next to the importing file, or under
    the repo root (`prc.x` -> prc/x.py), followed recursively. SHA-256 of each file's bytes."""
    seen, todo = {}, list(entries)
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        b = git_bytes(commit, p)
        if b is None:
            seen[p] = None
            continue
        seen[p] = hashlib.sha256(b).hexdigest()
        try:
            tree = ast.parse(b)
        except SyntaxError:
            continue
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module)
                names |= {f"{node.module}.{a.name}" for a in node.names}
        here = str(Path(p).parent)
        for n in names:
            for cand in (f"{here}/{n.replace('.', '/')}.py", f"{n.replace('.', '/')}.py",
                         f"{n.replace('.', '/')}/__init__.py"):
                if git_bytes(commit, cand) is not None:
                    todo.append(cand)
                    break
    return {k: v for k, v in sorted(seen.items()) if v is not None}


def file_row(rel: str) -> dict:
    p = E.ROOT / rel
    if not p.exists():
        return {"path": rel, "exists_on_mac": False}
    row = {"path": rel, "exists_on_mac": True, "bytes": p.stat().st_size, "sha256": E.sha_file(p),
           "last_commit": last_commit(rel)}
    if rel.endswith(".parquet"):
        md = pq.read_metadata(p)
        row["rows"] = md.num_rows
        row["schema"] = [f"{f.name}:{f.physical_type}" for f in md.schema]
    return row


def main() -> None:
    NO.refuse_existing([DAG, ARTS], False)
    t0 = time.time()
    dag0, arts0 = E.load_json(DAG0), E.load_json(ARTS0)
    hist = {n["id"]: n for n in dag0["nodes"]}
    rec = E.load_json(E.ANALYSIS / "kernel-receipts.json")["kernels"]
    arts = dict(arts0["paths"])
    nodes, edges = [], []

    def add(node: dict, files: list[str]) -> None:
        node["census"] = [file_row(f) for f in files]
        for row in node["census"]:
            arts[row["path"]] = row
        nodes.append(node)
        edges.extend({"from": i, "to": node["id"]} for i in node.get("inputs", []))

    code_git = sorted({k["git"] for k in rec.values()})
    assert len(code_git) == 1, code_git
    commit = code_git[0]
    src = {m: closure(commit, ENTRY[m] + ["prc/data.py"]) for m in E.MEMBERS}
    wrapper_tpl = {}
    for w in ("prc-iso.py", "prc-iso-thin.py"):
        s = (E.ROOT / f"{EXP}/kernels/iso/{w}").read_text()
        tree = ast.parse(s)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "MEMBERS":
                wrapper_tpl[w] = {k.value: ast.get_source_segment(s, v) for k, v in zip(node.value.keys, node.value.values)}
    for r in E.ORDER:
        V, S, I, M = E.ROT[r]
        thin_alias, thin_base = THIN[r]
        for m in E.MEMBERS:
            h = hist[HIST[m]]
            for group in (r, "thin"):
                kname = f"{m}-{group}"
                k = rec[kname]
                wrapper = "prc-iso-thin.py" if group == "thin" else "prc-iso.py"
                kdir = f"{EXP}/kernels/iso-{m}-{group}"
                folds = [thin_alias] if group == "thin" else [f"iso_{r}_{w}" for w in "vsix"]
                for fold in folds:
                    base_fold = thin_base if fold == thin_alias else fold
                    fo = E.D.FOLDS[base_fold]
                    train = list(fo.train_months)
                    valid = fo.valid_month
                    step = next(s_ for s_ in k["steps"] if s_["label"] == f"{m} {fold}")
                    out = k["folds"][fold]
                    node = {
                        "id": f"replay:{m}:{fold}", "kind": "member fit (label-isolated refit)", "member": m,
                        "rotation": r, "fold": fold, "thinned_from": base_fold if fold == thin_alias else None,
                        "kernel": {"slug": k["slug"], "script": f"{kdir}/{wrapper}", "gpu": k["gpu"],
                                   "wall_h": k["wall_h"], "step_wall_s": step["wall_s"],
                                   "peak_host_used_gb": k["peak_host_used_gb"], "peak_gpu_mib": k["peak_gpu_mib"],
                                   "data_sha256_match_mac": k["data_sha256_match_mac"],
                                   "producer_stdout_sealed": {"bytes": step["stdout_bytes"],
                                                              "sha256": step["stdout_sha256"]}},
                        "code_commit": commit,
                        "code_dataset": f"{kaggle_user()}/experiment-code (git archive of the code commit)",
                        "data_dataset": f"{kaggle_user()}/{os.environ['PRC_KAGGLE_DATA_SLUG']}",
                        "producer_command_template": wrapper_tpl[wrapper][m],
                        "producer_sources_sha256_at_commit": src[m],
                        "train_months": train, "predicts": valid,
                        "excludes": {"S": S not in train, "I": I not in train, "V": V not in train},
                        "excludes_S_and_V": bool(S not in train and V not in train),
                        "feature_months": FEATURES + "; " + str(h.get("feature_months")),
                        "label_loads_in_kernel": {
                            "train": train,
                            "predicted_month": valid,
                            "use_of_predicted_month_labels": "the producer's own scoring (RMSE printed to the sealed "
                                                             "stdout) and its y_true column (dropped before write); no "
                                                             "value left Kaggle; review/round1-answers.md traces the use",
                            "before_registration": True},
                        "recipe_constants": {
                            "historical_selection": h.get("early_stop_or_selection"),
                            "label_months_of_historical_choices": h.get("label_months"),
                            "chosen_on_folds_that_include_S_or_V": True,
                            "isolation_item_1_met": False},
                        "outputs": {"file": out["file"], "rows": out["rows"], "file_sha256": out["file_sha256"],
                                    "vector_sha256_ids_then_pred": out["vector_sha256_ids_then_pred"]},
                    }
                    if fold == thin_alias:
                        node["thin"] = k["thin"][fold]
                        node["thin_note"] = ("features from whole months; L2's guard range and T-tailv4's band head "
                                             "and phi see the unthinned training months (finding 4)")
                    add(node, [out["file"], f"{kdir}/{wrapper}", f"{kdir}/kernel-metadata.json"])
        inp = E.load_json(E.ANALYSIS / f"inputs-{r}.json")
        fits = [f"replay:{m}:iso_{r}_{w}" for m in E.MEMBERS for w in "vsix"] + \
               [f"replay:{m}:{thin_alias}" for m in E.MEMBERS]
        add({"id": f"replay:inputs:{r}", "kind": "freeze (label-free)", "code": f"{EXP}/code/freeze.py",
             "description": "ordered V, S, I departures, masks, keys, priors, thinning sets and the bases "
                            "max(60, mean of the three members): main (A fits), exposed on S (x or x_m), "
                            "matched isolated on S (s_m, or s for jul)",
             "inputs": fits + [f"prior:exp-054-rep-{E.REP_FOLD[r]}"], "label_reads": inp["label_reads"],
             "freeze_utc": inp["freeze_utc"], "npz_sha256": inp["npz_sha256"]},
            [f"{EXP}/code/freeze.py", f"{EXP}/analysis/inputs-{r}.json", inp["npz"]])
        add({"id": f"prior:exp-054-rep-{E.REP_FOLD[r]}", "kind": "nuisance fit (arrival factors)",
             "code": "experiments/exp-054-arrivalrepr/code/represent_r.py",
             "label_months": {"arrival_taxi_in": [x for x in E.D.MONTHS if x != V]},
             "uses_S_arrivals": True, "uses_I_arrivals": True, "excludes_V": True, "departure_labels": "none"},
            ["experiments/exp-054-arrivalrepr/code/represent_r.py",
             f"experiments/exp-054-arrivalrepr/work/rep-{E.REP_FOLD[r]}.parquet"])
        for it, rec_name in ((1, f"fit-{r}.json"), (2, f"it2-fit-{r}.json")):
            fr = E.load_json(E.ANALYSIS / rec_name)
            add({"id": f"replay:correction:{r}:it{it}", "kind": "correction fit and inner selection",
                 "code": f"{EXP}/code/fit.py", "command": f".venv/bin/python {EXP}/code/fit.py {r}"
                 + (" --iteration 2" if it == 2 else ""), "inputs": [f"replay:inputs:{r}"],
                 "label_months": {"fit": [S], "selection": [S, I], "time_origin": [S, I]},
                 "label_reads": fr["label_reads"], "excludes_V": True, "pred_npz_sha256": fr["pred_npz_sha256"]},
                [f"{EXP}/code/fit.py", f"{EXP}/code/e56.py", f"{EXP}/analysis/{rec_name}", fr["pred_npz"]])
    for it in (1, 2):
        pr = E.load_json(E.ANALYSIS / f"iter{it}.json")
        add({"id": f"replay:pricing:it{it}", "kind": "scoring on V (all three rotations in one process)",
             "code": f"{EXP}/code/price.py", "inputs": [f"replay:correction:{r}:it{it}" for r in E.ORDER],
             "label_months": {"score": [E.ROT[r][0] for r in E.ORDER]}, "label_reads": pr["label_reads"]},
            [f"{EXP}/code/price.py", f"{EXP}/analysis/iter{it}.json", pr["artifacts"]["rows"]["path"],
             pr["artifacts"]["draws"]["path"]] + sorted(pr["artifacts"]["oof"]))
    post = {"q12oracle": ("oracle-{r}.json", "descriptive regression of zero's V residual on raw prior columns; "
                                              "not a bound (finding 11)", ["V"]),
            "gridoracle": ("gridoracle-{r}.json", "bound on V over the registered correction family", ["S", "V"]),
            "refineoracle": ("refineoracle-{r}.json", "bound on V over the per-airport refinement", ["S", "V"]),
            "origin_r1": ("origin-r1-{r}.json", "time-origin control with equal fit rows", ["S", "I", "V"])}
    for job, (pat, what, months) in post.items():
        for r in E.ORDER:
            pr = E.load_json(E.ANALYSIS / pat.format(r=r))
            rot = dict(zip("VSIM", E.ROT[r]))
            files = [f"{EXP}/code/{job}.py", f"{EXP}/analysis/{pat.format(r=r)}"]
            if job == "origin_r1":
                files.append(pr["loss_npz"])
            add({"id": f"posthoc:{job}:{r}", "kind": "post-hoc (after both iterations; nothing chosen from it)",
                 "code": f"{EXP}/code/{job}.py", "what": what, "inputs": [f"replay:correction:{r}:it1"],
                 "label_months": [rot[x] for x in months], "label_reads": pr["label_reads"]}, files)
    for job, outs in (("datesums", ["datesums-iter1.json", "datesums-iter2.json"]),
                      ("datesums_sparse", ["datesums-sparse-iter1.json", "datesums-sparse-iter2.json"]),
                      ("caxis_check", ["caxis-check.json"])):
        add({"id": f"posthoc:{job}", "kind": "record check (no label read)", "code": f"{EXP}/code/{job}.py",
             "inputs": ["replay:pricing:it1", "replay:pricing:it2"]},
            [f"{EXP}/code/{job}.py"] + [f"{EXP}/analysis/{o}" for o in outs])
    fits = [n for n in nodes if n["kind"].startswith("member fit")]
    res = {"what": __doc__.strip().splitlines()[0], "built_utc": E.utc(), "code_sha256": E.code_shas(),
           "historical_nodes": {"file": str(DAG0.relative_to(E.ROOT)), "sha256": E.sha_file(DAG0),
                                "count": len(dag0["nodes"]), "note": "unchanged; their S/V flags are false"},
           "supersedes": "dag.json's replay_nodes",
           "isolation_items": {
               "1": "not met: the members keep historical recipe constants chosen on folds that include S or V "
                    "(recipe_constants on every member node); the training rows exclude S and V (A fits)",
               "2": "met for the correction and its controls (S and I labels only; V at pricing); the factor "
                    "tables use S's and I's arrivals (recorded)",
               "3": "met: masks before features; inputs frozen before the first S label read on this Mac; "
                    "but the Kaggle producers loaded V, S and I labels for their own sealed scoring (finding 1)",
               "4": "not met in order: the producers computed sealed replay losses before SPEC.md was committed"},
           "member_fits": len(fits),
           "member_fits_excluding_S_and_V": sum(n["excludes_S_and_V"] for n in fits),
           "nodes": nodes, "edges": edges, "seconds": round(time.time() - t0, 1)}
    missing = sorted(p for p, v in arts.items() if not v.get("exists_on_mac"))
    art = {"what": "every repo path of dag.json and dag-r1.json, hashed by bytes on this Mac",
           "built_utc": E.utc(), "paths": dict(sorted(arts.items())), "count": len(arts),
           "exp056_paths": sum(1 for p in arts if "exp-056" in p), "missing_on_mac": missing}
    NO.refuse_existing([DAG, ARTS], False)
    DAG.write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    ARTS.write_text(json.dumps(art, indent=1, sort_keys=True))
    print(f"replay nodes {len(nodes)} (member fits {len(fits)}, of which {res['member_fits_excluding_S_and_V']} "
          f"exclude S and V), edges {len(edges)}; artifacts {len(arts)} paths, {art['exp056_paths']} exp-056, "
          f"{len(missing)} missing ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
