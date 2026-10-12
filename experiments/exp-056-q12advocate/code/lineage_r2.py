"""exp-056-q12advocate: Stage 0 lineage, revision 2 -- every job a node, edges from the files each script reads (review round 2, finding 2).

Revision 1 (`analysis/dag-r1.json`) left the freeze, correction and pricing nodes without feature
months, output schemas and complete S/V flags; it had no common-base or gate node; it gave every
post-hoc job the iteration-1 correction as its only input; and its census left out the combined
records, the checks and 102 files of analysis/. This revision rebuilds the replay part:

- the 45 member-fit nodes and the 3 factor-table nodes of revision 1 are carried over (their census
  files hashed again and checked equal); each member fit gains its kernel's port record and log,
  and the factor-table nodes gain feature months and output schemas;
- per rotation, a common-base node (max(60, mean of the three members) for V, S, I and the two
  sides of the exposure pair, with the member fold behind each array), a freeze node, and one
  correction node per iteration; per iteration, a pricing node and a gate node; iteration 2's
  corrections take iteration 1's gate as a decision input;
- every other process that wrote a record is a node: the post-hoc jobs per rotation and their
  combines, the dry runs (one node per record, linked by the input record and prediction hashes
  the records carry), the record checks, the reports, the receipts and the lineage jobs;
- each node's inputs are the files its script reads, declared here; for every script, each file
  name in its source (string constants and f-strings ending in .json, .npz, .parquet or .md) must
  match an output of the script's own nodes or of their inputs, or the job stops;
- each output has its schema: JSON top-level keys, npz array headers, parquet footers;
- flags per node: the months whose departure labels its own process reads (from its record's
  label-read log, or the kernel's loads), the months whose labels enter its output (its own fit,
  selection or score, and every ancestor's, carried along the data and decision edges), and for
  its rotation's S, I and V whether they are excluded; arrival labels and historical recipe
  constants are carried as separate flags;
- a census of every file of this experiment (tracked or git-ignored, except __pycache__ and pid
  files), of every file under artifacts/exp-056/, and of revision 1's paths, with two checks:
  every file under analysis/ is an output of a node, and every log is attached to a node.

Outputs `analysis/dag-r2.json` and `analysis/artifacts-r2.json`. It reads no label file
(e56.dep_labels is never called). It parses the JSON records for their keys, label-read logs,
paths and hashes, and opens parquet and npz files for their headers only.

    .venv/bin/python experiments/exp-056-q12advocate/code/lineage_r2.py
"""
from __future__ import annotations

import ast
import fnmatch
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from numpy.lib import format as npfmt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402
import lineage_r1 as L1                                               # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

EXP = L1.EXP
R1 = E.ANALYSIS / "dag-r1.json"
DAG, ARTS = E.ANALYSIS / "dag-r2.json", E.ANALYSIS / "artifacts-r2.json"
SELF = "lineage_r2.py"
# scripts whose file names are not checked against their nodes: the lineage jobs name the
# historical producers' files, and dryprobe.py re-runs other scripts into work/probe (deleted)
NO_SOURCE_CHECK = {"lineage.py": "names historical producers' files (dag.json nodes)",
                   "lineage_r1.py": "names historical producers' files and every replay record",
                   SELF: "this job", "dryprobe.py": "wrapper that re-ran freeze/fit/price into work/probe",
                   "e56.py": "library, not a job"}
FEATURES_DEP = ("e56.dep_frame: the month's departures after prc/data.py's valid-month mask (keys, UTC "
                "date, population, regime); no label column")


def a_(name: str) -> str:
    return f"{EXP}/analysis/{name}"


def c_(name: str) -> str:
    return f"{EXP}/code/{name}"


def log_(name: str) -> str:
    return f"{EXP}/logs/{name}"


def npz_schema(p: Path) -> dict:
    """Array name -> dtype and shape, from each member's .npy header only."""
    out = {}
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            with z.open(n) as f:
                v = npfmt.read_magic(f)
                shape, _, dtype = (npfmt.read_array_header_1_0 if v == (1, 0) else npfmt.read_array_header_2_0)(f)
            out[n.removesuffix(".npy")] = f"{dtype}{list(shape)}"
    return out


def file_row(rel: str, recorded_sha: str | None = None) -> dict:
    p = E.ROOT / rel
    if not p.exists():
        return {"path": rel, "exists_on_mac": False, "recorded_sha256": recorded_sha}
    if p.is_dir():
        return {"path": rel, "exists_on_mac": True, "directory": True,
                "files": sum(1 for x in p.rglob("*") if x.is_file())}
    row = {"path": rel, "exists_on_mac": True, "bytes": p.stat().st_size, "sha256": E.sha_file(p),
           "last_commit": L1.last_commit(rel)}
    if recorded_sha is not None:
        row["recorded_sha256"] = recorded_sha
        row["equals_record"] = row["sha256"] == recorded_sha
    if rel.endswith(".parquet"):
        md = pq.read_metadata(p)
        row["rows"] = md.num_rows
        row["schema"] = [f"{f.name}:{f.physical_type}" for f in md.schema]
    elif rel.endswith(".npz"):
        row["schema"] = npz_schema(p)
    elif rel.endswith(".json"):
        d = json.loads(p.read_text())
        row["schema"] = sorted(d) if isinstance(d, dict) else type(d).__name__
    return row


def source_patterns(path: Path) -> set[str]:
    """File names a script's source spells: string constants and f-strings (a formatted part -> *)."""
    tree = ast.parse(path.read_text())
    inner = {id(v) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) for v in n.values}
    pats = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.JoinedStr):
            s = "".join(v.value if isinstance(v, ast.Constant) else "*" for v in n.values)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in inner:
            s = n.value
        else:
            continue
        if re.fullmatch(r"[\w*.{}/-]*[\w*}]\.(json|npz|parquet|md)", s):
            pats.add(s.rsplit("/", 1)[-1])
    return pats


def months_read(rec: dict) -> list[str]:
    return sorted({x["month"] for x in rec.get("label_reads", [])})


def main() -> None:
    NO.refuse_existing([DAG, ARTS], False)
    t0 = time.time()
    r1 = E.load_json(R1)
    arts0 = E.load_json(E.ANALYSIS / "artifacts-r1.json")["paths"]
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add(node: dict, outputs: list, logs: list[str] = (), extra: list[str] = (),
            order: list[str] = (), decision: list[str] = ()) -> None:
        """outputs: paths, or (path, recorded sha) pairs; logs and extra (code, launch scripts) go to
        the census too. Data edges from node['inputs']; order and decision edges as given."""
        if node["id"] in nodes:
            raise SystemExit(f"duplicate node {node['id']}")
        outs = [o if isinstance(o, tuple) else (o, None) for o in outputs]
        node["outputs"] = [file_row(p, s) for p, s in outs]
        node["logs"] = [file_row(p) for p in logs]
        node["code_files"] = [file_row(p) for p in extra]
        node.setdefault("inputs", [])
        node["order_after"] = list(order)
        nodes[node["id"]] = node
        edges.extend({"from": i, "to": node["id"], "kind": "data"} for i in node["inputs"])
        edges.extend({"from": i, "to": node["id"], "kind": "order"} for i in order)
        edges.extend({"from": i, "to": node["id"], "kind": "decision"} for i in decision)

    # raw competition files (the Kaggle data dataset carries the same bytes, kernel receipts)
    for m in E.D.MONTHS:
        add({"id": f"data:{m}", "kind": "raw data (competition month file)", "month": m,
             "label_reads": [], "labels_enter_output": []}, [str(E.D.month_file(m).relative_to(E.ROOT))])

    # carried over from revision 1: 45 member fits and 3 factor tables
    rec = E.load_json(E.ANALYSIS / "kernel-receipts.json")["kernels"]
    for n in r1["nodes"]:
        if not (n["kind"].startswith("member fit") or n["id"].startswith("prior:")):
            continue
        for row in n["census"]:
            now = file_row(row["path"])
            if now.get("sha256") != row.get("sha256"):
                raise SystemExit(f"{row['path']} changed since revision 1")
        x = {k: v for k, v in n.items() if k not in ("census", "outputs")}
        if n["kind"].startswith("member fit"):
            group = "thin" if n["thinned_from"] else n["rotation"]
            kname = f"{n['member']}-{group}"
            x["inputs"] = [f"data:{m}" for m in sorted(set(n["train_months"]) | {n["predicts"]})]
            x["label_reads"] = sorted(set(n["train_months"]) | {n["predicts"]})
            x["labels_enter_output"] = list(n["train_months"])
            x["recipe_constants_include_S_or_V"] = True
            x["feature_months"] = {"train": n["train_months"], "predicted": n["predicts"], "rule": n["feature_months"]}
            kag = "artifacts/exp-056/kaggle"
            add(x, [(n["outputs"]["file"], n["outputs"]["file_sha256"]), f"{kag}/port-{kname}.json"],
                logs=[f"{kag}/kernel-{kname}.log"], extra=[r["path"] for r in n["census"][1:]])
            if rec[kname]["folds"][n["fold"]]["file_sha256"] != n["outputs"]["file_sha256"]:
                raise SystemExit(f"{n['id']}: kernel receipt and revision 1 disagree")
        else:
            fold = n["id"].removeprefix("prior:exp-054-rep-")
            V = next(E.ROT[r][0] for r in E.ORDER if E.REP_FOLD[r] == fold)
            x["inputs"] = []
            x["label_reads"] = []
            x["labels_enter_output"] = []
            x["arrival_labels_enter_output"] = [m for m in E.D.MONTHS if m != V]
            x["feature_months"] = {"arrivals": [m for m in E.D.MONTHS if m != V],
                                   "rule": "exp-054 represent_r.py: arrival keys and taxi-in outcomes of the "
                                           "eleven months other than V"}
            add(x, [(f"experiments/exp-054-arrivalrepr/work/rep-{fold}.parquet", E.REP_SHA[fold])],
                extra=["experiments/exp-054-arrivalrepr/code/represent_r.py"])

    # per rotation: common base, freeze, corrections
    fitrec = {}
    for r in E.ORDER:
        V, S, I, M = E.ROT[r]
        inp = E.load_json(E.ANALYSIS / f"inputs-{r}.json")
        ex, ref = E.EXPOSURE_PAIR[r]
        main = {"p_V": f"iso_{r}_v", "p_S": f"iso_{r}_s", "p_I": f"iso_{r}_i"}
        if ref != "s":
            main["p_Sref"] = f"iso_{r}_{ref}"
        sides = {"base": (main, "the main bases (A fits) on V, S and I" + ("" if ref == "s" else
                                                                           ", and the matched isolated base on S")),
                 "base-expo": ({"p_Sexpo": f"iso_{r}_{ex}"}, "the exposed base on S (A - M + S fit), one side of "
                                                             "the training-size pair")}
        for nid, (arrays, what) in sides.items():
            member_of = {a: [f"replay:{m}:{fold}" for m in E.MEMBERS] for a, fold in arrays.items()}
            add({"id": f"replay:{nid}:{r}", "kind": "common base (arithmetic; no fit)", "rotation": r, "what": what,
                 "code": c_("freeze.py"), "function": "member_preds (freeze.py:37-72): max(60, mean of the three members)",
                 "inputs": sorted({i for v in member_of.values() for i in v}), "arrays_from_member_fits": member_of,
                 "arrays_in": inp["npz"], "label_reads": [], "labels_enter_output": [],
                 "base_vector_sha256": {a: inp["months_detail"][a[2]]["base"]["base_vector_sha256_ids_then_pred"]
                                        for a in arrays if a in ("p_V", "p_S", "p_I")},
                 "feature_months": {"from_member_fits": "see each member-fit node"}},
                [(inp["npz"], inp["npz_sha256"])])
        array_sources = {a: f"replay:base:{r}" for a in main} | {"p_Sexpo": f"replay:base-expo:{r}"}
        if ref == "s":
            array_sources["p_Sref"] = f"replay:base:{r} (p_Sref is p_S: the unthinned isolated side of the pair)"
        vocab = str((E.WORK / f"vocab-{r}.json").relative_to(E.ROOT))
        add({"id": f"replay:inputs:{r}", "kind": "freeze (label-free)", "rotation": r, "code": c_("freeze.py"),
             "command": f".venv/bin/python {EXP}/code/freeze.py {r}",
             "inputs": [f"replay:base:{r}", f"replay:base-expo:{r}", f"prior:exp-054-rep-{E.REP_FOLD[r]}",
                        f"data:{V}", f"data:{S}", f"data:{I}"], "base_arrays_from": array_sources,
             "unused_member_fits": [f"replay:{m}:iso_jul_x" for m in E.MEMBERS] if r == "jul" else [],
             "feature_months": {"V": V, "S": S, "I": I, "rule": FEATURES_DEP,
                                "factor_table": f"exp-054 rep-{E.REP_FOLD[r]} (arrivals of the months other than V)"},
             "label_reads": months_read(inp), "labels_enter_output": [], "freeze_utc": inp["freeze_utc"]},
            [(inp["npz"], inp["npz_sha256"]), a_(f"inputs-{r}.json"), (vocab, inp.get("vocab_sha256"))],
            logs=[log_(f"freeze-{r}.log")], extra=[c_("freeze.py"), c_("e56.py")])
        for it, name in ((1, f"fit-{r}.json"), (2, f"it2-fit-{r}.json")):
            fr = E.load_json(E.ANALYSIS / name)
            fitrec[(r, it)] = fr
            add({"id": f"replay:correction:{r}:it{it}", "kind": "correction fit and inner selection", "rotation": r,
                 "iteration": it, "code": c_("fit.py"),
                 "command": f".venv/bin/python {EXP}/code/fit.py {r}" + (" --iteration 2" if it == 2 else ""),
                 "inputs": [f"replay:inputs:{r}", f"data:{S}", f"data:{I}"],
                 "feature_months": {"fit": S, "selection": [S, I], "time_origin_fit": I, "predicted": [V, S, I],
                                    "rule": "one-hot stand, runway and operator keys of each month's departures "
                                            "(inputs npz); the base vectors of the freeze"},
                 "label_reads": months_read(fr), "labels_enter_output": sorted({S, I}),
                 "label_uses": {S: "correction fit; inner selection S -> I", I: "inner selection I -> S; time-origin fit"}},
                [(fr["pred_npz"], fr["pred_npz_sha256"]), a_(name)],
                logs=[log_(name.replace(".json", ".log"))], extra=[c_("fit.py")],
                decision=["replay:gate:it1"] if it == 2 else [])

    # pricing and gates
    for it in (1, 2):
        pr = E.load_json(E.ANALYSIS / f"iter{it}.json")
        for r in E.ORDER:
            if pr["inputs"][r]["fit_pred_npz_sha256"] != fitrec[(r, it)]["pred_npz_sha256"]:
                raise SystemExit(f"iter{it}.json did not price fit {r} it{it}")
        ins = [f"replay:correction:{r}:it{it}" for r in E.ORDER] + [f"replay:inputs:{r}" for r in E.ORDER] + \
              ([f"replay:correction:{r}:it1" for r in E.ORDER] if it == 2 else []) + \
              [f"data:{E.ROT[r][0]}" for r in E.ORDER]
        arts = pr["artifacts"]
        add({"id": f"replay:pricing:it{it}", "kind": "scoring on V (all three rotations in one process)",
             "iteration": it, "code": c_("price.py"),
             "command": f".venv/bin/python {EXP}/code/price.py" + (" --iteration 2" if it == 2 else ""),
             "inputs": ins, "feature_months": {"scored": [E.ROT[r][0] for r in E.ORDER]},
             "label_reads": months_read(pr), "labels_enter_output": [E.ROT[r][0] for r in E.ORDER]},
            [a_(f"iter{it}.json"), (arts["rows"]["path"], arts["rows"]["sha256"]),
             (arts["draws"]["path"], arts["draws"]["sha256"])] + sorted(arts["oof"].items()),
            logs=[log_(f"price-iter{it}.log")], extra=[c_("price.py")])
        add({"id": f"replay:gate:it{it}", "kind": "registered gate (a decision; no new read)", "iteration": it,
             "code": c_("price.py"), "registration": "SPEC.md section 3.8 (61e3050)" if it == 1 else
             "SPEC.md section 4 (965cf63)", "inputs": [f"replay:pricing:it{it}"], "record_field": f"iter{it}.json: gate",
             "outcome": "fail: iteration 2 registered and run" if it == 1 else "fail: no advance",
             "label_reads": [], "labels_enter_output": []}, [])
        add({"id": f"report:iter{it}", "kind": "report (tables from the record)", "code": c_("report.py"),
             "command": f".venv/bin/python {EXP}/code/report.py analysis/iter{it}.json",
             "inputs": [f"replay:pricing:it{it}"], "label_reads": [], "labels_enter_output": []},
            [a_(f"report-iter{it}.md")], extra=[c_("report.py")])

    # post-hoc jobs (after both iterations; nothing chosen from them)
    pricings = ["replay:pricing:it1", "replay:pricing:it2"]
    post = {"q12oracle": ("oracle", lambda r: [f"replay:inputs:{r}", f"replay:correction:{r}:it1"], True),
            "gridoracle": ("gridoracle", lambda r: [f"replay:inputs:{r}", f"replay:correction:{r}:it1",
                                                    f"replay:correction:{r}:it2"], True),
            "refineoracle": ("refineoracle", lambda r: [f"replay:inputs:{r}", f"replay:correction:{r}:it1",
                                                        f"posthoc:gridoracle:{r}"], False),
            "origin_r1": ("origin-r1", lambda r: [f"replay:inputs:{r}", f"replay:correction:{r}:it1",
                                                  f"replay:correction:{r}:it2"], False)}
    for job, (stem, ins, checks_order) in post.items():
        for r in E.ORDER:
            d = E.load_json(E.ANALYSIS / f"{stem}-{r}.json")
            outs = [a_(f"{stem}-{r}.json")]
            if job == "origin_r1":
                outs.append((d["loss_npz"], d["loss_npz_sha256"]))
            add({"id": f"posthoc:{job}:{r}", "kind": "post-hoc (after both iterations; nothing chosen from it)",
                 "rotation": r, "code": c_(f"{job}.py"), "command": f".venv/bin/python {EXP}/code/{job}.py {r}",
                 "inputs": ins(r) + [f"data:{m}" for m in months_read(d)],
                 "label_reads": months_read(d), "labels_enter_output": months_read(d)},
                outs, logs=[log_(f"{stem}-{r}.log")], extra=[c_(f"{job}.py")],
                order=pricings if checks_order else [])
        cins = [f"posthoc:{job}:{r}" for r in E.ORDER]
        if job == "origin_r1":
            cins += [f"replay:inputs:{r}" for r in E.ORDER] + pricings
        add({"id": f"posthoc:{job}:combine", "kind": "post-hoc combine (board recombination; no label read)",
             "code": c_(f"{job}.py"), "command": f".venv/bin/python {EXP}/code/{job}.py --combine",
             "inputs": cins, "label_reads": [], "labels_enter_output": []},
            [a_(f"{stem}.json")], logs=[log_(f"{stem}-combine.log")], extra=[c_(f"{job}.py")])

    # dry runs: one node per record; synthetic labels, no label read. Pass 1 reads the records;
    # pass 2 links each dry pricing record to the dry fits and inputs whose hashes it carries.
    specs, fit_by_sha, in_by_sha = [], {}, {}
    for name in sorted(p.name for p in E.ANALYSIS.glob("dry*.json") if re.match(r"dry\d?-", p.name)):
        d = E.load_json(E.ANALYSIS / name)
        if months_read(d):
            raise SystemExit(f"{name} logs a label read")
        stem = name.removesuffix(".json")
        m_in = re.fullmatch(r"(dry\d?)-inputs-(jul|jan|dec)", stem)
        m_fit = re.fullmatch(r"(dry\d?)-(\w+?)(-it2)?-fit-(jul|jan|dec)", stem)
        m_pr = re.fullmatch(r"(dry\d?)-(\w+?)(-p[23])?-iter([12])", stem)
        m_go = re.fullmatch(r"dry-(null|planted)-(gridoracle|refineoracle)(-(jul|jan|dec))?", stem)
        node = {"id": f"dry:{stem}", "kind": "dry run (synthetic labels; label-free)", "label_reads": [],
                "labels_enter_output": [], "record_code_sha256": d.get("code_sha256"), "inputs": []}
        outs, logs, order = [a_(name)], [], []
        if m_in:
            r = m_in[2]
            node.update(code=c_("freeze.py"), rotation=r,
                        inputs=[f"data:{m}" for m in E.ROT[r][:3]] + [f"prior:exp-054-rep-{E.REP_FOLD[r]}"],
                        feature_months={"V": E.ROT[r][0], "S": E.ROT[r][1], "I": E.ROT[r][2],
                                        "rule": FEATURES_DEP + "; stand-in bases (freeze.py dry_base)"})
            outs.append((d["npz"], d["npz_sha256"]))
            in_by_sha[d["npz_sha256"]] = node["id"]
            logs = [log_(f"{m_in[1]}-freeze-{r}.log")]
        elif m_fit:
            node.update(code=c_("fit.py"), rotation=m_fit[4], scenario=m_fit[2], iteration=2 if m_fit[3] else 1,
                        inputs=["dry:" + Path(d["inputs"]["record"]).name.removesuffix(".json")])
            outs.append((d["pred_npz"], d["pred_npz_sha256"]))
            key = (m_fit[1], d["pred_npz_sha256"])
            if key in fit_by_sha:
                raise SystemExit(f"two dry fits with one prediction hash: {fit_by_sha[key]}, {node['id']}")
            fit_by_sha[key] = node["id"]
            logs = [log_(f"{stem}.log")]
        elif m_pr:
            v, sc, p_, it = m_pr[1], m_pr[2], m_pr[3], int(m_pr[4])
            lg = f"{v}-{sc}{'-it2' if it == 2 else ''}-price.log" if v == "dry3" else \
                 f"{v}-{sc}-price{'-' + p_[1:] if p_ else ''}.log"
            logs = [log_(lg)] + [log_(x.name) for x in (E.EXP / "logs").glob(lg.replace(".log", "-r*.log"))]
            node.update(code=c_("price.py"), scenario=sc, iteration=it,
                        link={r: (d["inputs"][r]["fit_pred_npz_sha256"], d["inputs"][r]["record_npz_sha256"])
                              for r in E.ORDER}, version=v)
        elif m_go:
            job, sc, r = m_go[2], m_go[1], m_go[4]
            node.update(code=c_(f"{job}.py"), scenario=sc, rotation=r)
            if r:
                node["inputs"] = [f"replay:inputs:{r}", f"replay:correction:{r}:it1"] + (
                    [f"replay:correction:{r}:it2"] if job == "gridoracle" else [f"dry:dry-{sc}-gridoracle-{r}"])
                order = pricings if job == "gridoracle" else []
                logs = [log_(f"{stem}.log")]
            else:
                node["inputs"] = [f"dry:dry-{sc}-{job}-{x}" for x in E.ORDER]
                logs = [log_(f"dry-{sc}-{job}-combine.log")]
        else:
            raise SystemExit(f"unknown dry record {name}")
        specs.append((node, outs, logs, order))
    for node, outs, logs, order in specs:
        if "link" in node:
            link, v = node.pop("link"), node.pop("version")
            fits = [fit_by_sha[(v, link[r][0])] for r in E.ORDER]
            if node["iteration"] == 2:                                  # price.py also reads iteration 1's fits
                fits += [f.replace("-it2-fit-", "-fit-") for f in fits]
            node["inputs"] = fits + [in_by_sha[link[r][1]] for r in E.ORDER]
        add(node, outs, logs=logs, extra=[node["code"]], order=order)

    # record checks, reports and audits
    per_r = lambda f: [f(r) for r in E.ORDER]                                   # noqa: E731
    checks = [
        ("check:caxis_check", "caxis_check.py", per_r(lambda r: f"replay:inputs:{r}")
         + per_r(lambda r: f"replay:correction:{r}:it1") + per_r(lambda r: f"replay:correction:{r}:it2"),
         [a_("caxis-check.json")], ["caxis-check.log"]),
        ("check:datesums", "datesums.py", pricings + per_r(lambda r: f"replay:inputs:{r}"),
         [a_("datesums-iter1.json"), a_("datesums-iter2.json")], ["datesums.log"]),
        ("check:datesums_sparse", "datesums_sparse.py", pricings + per_r(lambda r: f"replay:inputs:{r}"),
         [a_("datesums-sparse-iter1.json"), a_("datesums-sparse-iter2.json")], ["datesums-sparse.log"]),
        ("check:common_tables", "common_tables.py", pricings + ["posthoc:gridoracle:combine"],
         [a_("common-tables.md")], ["common-tables.log"]),
        ("check:common_tables_r1", "common_tables_r1.py", pricings, [a_("common-tables-r1.md")],
         ["common-tables-r1.log"]),
        ("check:prior_p_corr", "prior_p_corr.py", per_r(lambda r: f"replay:inputs:{r}"),
         [a_("prior-p-corr.json")], ["prior-p-corr.log"]),
        ("check:fullcontract_ids", "fullcontract_ids.py", [f"data:{m}" for m in E.D.MONTHS],
         [a_("fullcontract-ids.json")], ["fullcontract-ids.log"]),
        ("check:drynull_why", "drynull_why.py", per_r(lambda r: f"replay:inputs:{r}")
         + per_r(lambda r: f"dry:dry-null-gridoracle-{r}"), [a_("drynull-why.json")], ["drynull-why.log"]),
        ("report:dry3-support-p3-iter2", "report.py", ["dry:dry3-support-p3-iter2"],
         [a_("report-dry3-support-p3-iter2.md")], []),
        ("audit:receipts", "receipts.py", sorted(i for i in nodes if nodes[i].get("kind", "").startswith("member")),
         [a_("kernel-receipts.json"), a_("kernel-receipts.md")], []),
        ("audit:thin_smoke", "thin_smoke.py", [], [a_("thin-smoke.json")], ["thin-smoke.log", "thin-smoke-r1.log"]),
        ("audit:lineage", "lineage.py", ["audit:lineage-source"], [a_("dag.json"), a_("artifacts.json")],
         ["lineage.log"]),
        ("audit:lineage_r1", "lineage_r1.py", ["audit:lineage", "audit:receipts"]
         + per_r(lambda r: f"replay:inputs:{r}") + [f"replay:correction:{r}:it{k}" for r in E.ORDER for k in (1, 2)]
         + pricings + [f"posthoc:{j}:{r}" for j in post for r in E.ORDER] + ["check:datesums", "check:datesums_sparse",
                                                                              "check:caxis_check"],
         [a_("dag-r1.json"), a_("artifacts-r1.json")], ["lineage-r1.log"]),
    ]
    add({"id": "audit:lineage-source", "kind": "external audit (a read-only agent wrote the historical node list)",
         "inputs": [], "label_reads": [], "labels_enter_output": []}, [a_("lineage-source.json")])
    for nid, script, ins, outs, logs in checks:
        label_free = {"label_reads": [], "labels_enter_output": []}
        add({"id": nid, "kind": "record check, report or audit (no label read)", "code": c_(script),
             "inputs": ins, **label_free}, outs, logs=[log_(x) for x in logs], extra=[c_(script)])
    nodes["audit:thin_smoke"]["scratch_files"] = {
        "patterns": ["res-*-*.json", "keep-*.npy", "thin-*.json"],
        "note": "written and read inside the process in a temporary directory (thin_smoke.py TMP); not kept"}

    # launcher logs and scripts (no record of their own)
    attached = {row["path"] for n in nodes.values() for row in n["logs"] + n["code_files"]}
    for p in sorted((E.EXP / "logs").iterdir()):
        rel = str(p.relative_to(E.ROOT))
        if p.suffix == ".pid" or rel in attached:
            continue
        add({"id": f"launcher:{p.name}", "kind": "launcher or watcher (runs other nodes' commands; no record)",
             "inputs": [], "label_reads": [], "labels_enter_output": []}, [], logs=[rel])

    # flags: labels entering each output over data edges; labels behind the decision to run, over
    # decision edges (iteration 2 runs because iteration 1's gate failed on V)
    parents: dict[str, dict[str, list[str]]] = {i: {"data": [], "decision": []} for i in nodes}
    for e in edges:
        if e["from"] not in nodes:
            raise SystemExit(f"edge from unknown node {e['from']} to {e['to']}")
        if e["kind"] in ("data", "decision"):
            parents[e["to"]][e["kind"]].append(e["from"])
    memo: dict[str, tuple] = {}

    def upstream(i: str, stack: tuple = ()) -> tuple[frozenset, frozenset, bool, frozenset]:
        if i in stack:
            raise SystemExit(f"cycle at {i}")
        if i not in memo:
            n = nodes[i]
            dep, arr, rc = set(n["labels_enter_output"]), set(n.get("arrival_labels_enter_output", [])), \
                bool(n.get("recipe_constants_include_S_or_V"))
            dec = set()
            for p in parents[i]["data"]:
                d_, a2, r2, c2 = upstream(p, stack + (i,))
                dep |= d_
                arr |= a2
                rc |= r2
                dec |= c2
            for p in parents[i]["decision"]:
                d_, _, _, c2 = upstream(p, stack + (i,))
                dec |= d_ | c2
            memo[i] = (frozenset(dep), frozenset(arr), rc, frozenset(dec))
        return memo[i]
    for i, n in nodes.items():
        dep, arr, rc, dec = upstream(i)
        n["flags"] = {"reads_departure_labels_of": sorted(n["label_reads"]),
                      "departure_labels_entering_output": sorted(dep),
                      "arrival_labels_entering_output": sorted(arr),
                      "historical_recipe_constants_upstream": rc,
                      "run_decided_after_departure_labels_of": sorted(dec)}
        rots = [n["rotation"]] if n.get("rotation") in E.ORDER else list(E.ORDER) if dep else []
        n["flags"]["by_rotation"] = {r: {k: (E.ROT[r][j] not in dep) for j, k in enumerate(("excludes_V", "excludes_S",
                                                                                          "excludes_I"))}
                                     for r in rots}

    # every script's named files are covered by its nodes' outputs or their inputs' outputs
    by_code: dict[str, list[str]] = {}
    for i, n in nodes.items():
        if str(n.get("code", "")).startswith(f"{EXP}/code/"):                # this experiment's scripts only
            by_code.setdefault(Path(n["code"]).name, []).append(i)
    source_check = {}
    for script, ids in sorted(by_code.items()):
        if script in NO_SOURCE_CHECK:
            continue
        files = {Path(row["path"]).name for i in ids for j in [i] + nodes[i]["inputs"] + nodes[i]["order_after"]
                 for row in nodes[j]["outputs"]}
        files |= {x for i in ids for x in nodes[i].get("scratch_files", {}).get("patterns", [])}
        pats = source_patterns(E.EXP / "code" / script)
        unmatched = sorted(p for p in pats if not any(fnmatch.fnmatch(f, p) for f in files))
        source_check[script] = {"patterns": sorted(pats), "unmatched": unmatched}
    bad = {s: v["unmatched"] for s, v in source_check.items() if v["unmatched"]}

    # census and its two checks
    census, drift = {}, []
    for pth, row in arts0.items():
        census[pth] = file_row(pth)
        if row.get("sha256") and census[pth].get("sha256") != row["sha256"]:
            drift.append(pth)
    for n in nodes.values():
        for row in n["outputs"] + n["logs"] + n["code_files"]:
            census[row["path"]] = row
    for base in (E.EXP, E.ART):
        for p in sorted(base.rglob("*")):
            rel = str(p.relative_to(E.ROOT))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pid" and rel not in census:
                census[rel] = file_row(rel)
    census[a_("dag-r2.json")] = {"path": a_("dag-r2.json"), "note": "this job's output"}
    census[a_("artifacts-r2.json")] = {"path": a_("artifacts-r2.json"), "note": "this job's output"}
    out_paths = {row["path"] for n in nodes.values() for row in n["outputs"]} | {a_("dag-r2.json"), a_("artifacts-r2.json")}
    orphans = sorted(str(p.relative_to(E.ROOT)) for p in E.ANALYSIS.iterdir()
                     if p.is_file() and str(p.relative_to(E.ROOT)) not in out_paths)
    log_paths = {row["path"] for n in nodes.values() for row in n["logs"]}
    unattached_logs = sorted(str(p.relative_to(E.ROOT)) for p in (E.EXP / "logs").iterdir()
                             if p.suffix != ".pid" and str(p.relative_to(E.ROOT)) not in log_paths)
    mismatched = sorted(r["path"] for r in census.values() if r.get("equals_record") is False)
    missing = sorted(p for p, r in census.items() if r.get("exists_on_mac") is False)
    print(f"nodes {len(nodes)}, edges {len(edges)}; census {len(census)} paths; analysis files not an output: "
          f"{orphans}; logs not attached: {unattached_logs}; files unequal to their record: {mismatched}; "
          f"script names not covered: {bad}", flush=True)
    if orphans or unattached_logs or mismatched or bad:
        raise SystemExit("a check failed; nothing written")

    kinds: dict[str, int] = {}
    for n in nodes.values():
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    fits = [n for n in nodes.values() if n["kind"].startswith("member fit")]
    res = {"what": __doc__.strip().splitlines()[0], "built_utc": E.utc(), "code_sha256": E.code_shas(),
           "supersedes": {"file": str(R1.relative_to(E.ROOT)), "sha256": E.sha_file(R1),
                          "note": "revision 1's replay part; dag.json's 43 historical nodes stand as built"},
           "historical_nodes": r1["historical_nodes"], "isolation_items": r1["isolation_items"],
           "node_kinds": kinds, "member_fits": len(fits),
           "member_fits_excluding_S_and_V": sum(all(v["excludes_S"] and v["excludes_V"]
                                                    for v in n["flags"]["by_rotation"].values()) for n in fits),
           "edge_kinds": {k: sum(e["kind"] == k for e in edges) for k in ("data", "order", "decision")},
           "edge_rule": "data: the node's process reads an output of the source (declared per node; each script's "
                        "file names checked against them, `source_check`); order: the node checks that the source's "
                        "output exists before it runs and reads none of its values; decision: the node runs because "
                        "of the source's outcome",
           "source_check": source_check, "source_check_skipped": NO_SOURCE_CHECK,
           "nodes": list(nodes.values()), "edges": edges, "seconds": round(time.time() - t0, 1)}
    art = {"what": "every file of exp-056 (tracked or git-ignored, except __pycache__ and pid files), every file "
                   "under artifacts/exp-056/, and every path of revision 1's census, hashed by bytes on this Mac",
           "built_utc": E.utc(), "paths": dict(sorted(census.items())), "count": len(census),
           "exp056_paths": sum(1 for p in census if "exp-056" in p), "missing_on_mac": missing,
           "checks": {"analysis_files_not_an_output": orphans, "logs_not_attached": unattached_logs,
                      "files_unequal_to_their_record": mismatched},
           "changed_since_revision_1": drift}
    NO.refuse_existing([DAG, ARTS], False)
    DAG.write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    ARTS.write_text(json.dumps(art, indent=1, sort_keys=True))
    print(f"member fits {len(fits)}, of which {res['member_fits_excluding_S_and_V']} exclude S and V; kinds {kinds}; "
          f"edges {res['edge_kinds']}; census {len(census)} paths, {art['exp056_paths']} exp-056, "
          f"{len(missing)} missing ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
