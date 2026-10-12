"""exp-056-q12advocate: Stage 0 lineage DAG and artifact census (SPEC section 2).

Input: `analysis/lineage-source.json`, the node list a read-only lineage audit wrote on
2026-09-11 (43 nodes, 142 edges; copied verbatim, sha in the output). This job checks it against
the machine: every repo path a node names as producer or output is expanded (braces, fold and
month placeholders, globs) and hashed by bytes (SHA-256; no file is parsed, so no label vector
is formed), with its size and the last commit that touched it. It adds the S/V exclusion status
per rotation, the machine facts the assignment asks to re-check, and the replay nodes of this
experiment. Outputs `analysis/dag.json` and `analysis/artifacts.json`.

    .venv/bin/python experiments/exp-056-q12advocate/code/lineage.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


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

sys.path.insert(0, str(E.ROOT / "release"))
from lineage_profile import profile_is_configured                    # noqa: E402

SRC = E.ANALYSIS / "lineage-source.json"
DAG = E.ANALYSIS / "dag.json"
ARTS = E.ANALYSIS / "artifacts.json"
FOLD_NAMES = ["july_loo", "december", "january"]
MAX_HASH = 2 * 1024 ** 3


def expand(token: str) -> list[str]:
    t = token.rstrip(",;").strip("`")
    t = t.replace("{fold}", "{" + ",".join(FOLD_NAMES) + "}").replace("<month>", "*")
    t = re.sub(r"\{9$", "*", t)
    if "{" in t and "}" in t:
        m = re.search(r"\{([^{}]*)\}", t)
        out = []
        for alt in m.group(1).split(","):
            out += expand(t[:m.start()] + alt + t[m.end():])
        return out
    return [t]


def git_last(path: str) -> str | None:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%H", "--", path], cwd=E.ROOT,
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return out or None
    except Exception:                                               # noqa: BLE001
        return None


def census(token: str) -> list[dict]:
    rows = []
    for pat in expand(token):
        if not re.match(r"^(experiments|artifacts|prc|data|docs|submissions|scripts|oof)", pat):
            continue
        hits = sorted(glob.glob(str(E.ROOT / pat))) if any(c in pat for c in "*?[") else \
            ([str(E.ROOT / pat)] if (E.ROOT / pat).exists() else [])
        if not hits:
            rows.append({"path": pat, "exists_on_mac": False})
            continue
        for h in hits[:60]:
            p = Path(h)
            rel = str(p.relative_to(E.ROOT))
            if p.is_dir():
                n = sum(1 for _ in p.rglob("*"))
                rows.append({"path": rel + "/", "exists_on_mac": True, "dir_entries": n})
                continue
            sz = p.stat().st_size
            rows.append({"path": rel, "exists_on_mac": True, "bytes": sz,
                         "sha256": E.sha_file(p) if sz <= MAX_HASH else None,
                         "last_commit": git_last(rel)})
        if len(hits) > 60:
            rows.append({"path": pat, "note": f"{len(hits)} matches, first 60 hashed"})
    return rows


def machine_facts() -> dict:
    home = Path.home()
    f = {"checked_utc": E.utc(), "host": os.uname().nodename,
         "data_raw": sorted(p.name for p in (E.ROOT / "data/raw").glob("*.parquet")),
         "data_raw_sha256": {p.name: E.sha_file(p) for p in sorted((E.ROOT / "data/raw").glob("*.parquet"))},
         "venv_python": (E.ROOT / ".venv/bin/python").exists(),
         "configured_aws_profile_present": profile_is_configured(os.environ.get("AWS_PROFILE"), (
             (home / ".aws/credentials").read_text()
             if (home / ".aws/credentials").exists() else "")),
         "data_external": (E.ROOT / "data/external").exists(),
         "copernicus_cdsapirc": (home / ".cdsapirc").exists(),
         "era5_hourly_artifact": (E.ROOT / "artifacts/weather/era5_hourly.parquet").exists(),
         "v22_upload": (E.ROOT / "submissions/uploaded/youthful-giraffe_v22.parquet").exists(),
         "v22_candidate": (E.ROOT / "submissions/candidates/exp-046-surface-v11.parquet").exists(),
         "kaggle_cli": (home / ".local/bin/kaggle").exists(),
         "kaggle_datasets": [f"{kaggle_user()}/{os.environ['PRC_KAGGLE_DATA_SLUG']} (raw months + live ranking, 2026-09-07)",
                             f"{kaggle_user()}/experiment-code (git archive of 3a3bf9b, this experiment)"]}
    try:
        import prc.weather_features  # noqa: F401
        f["import_prc_weather_features"] = "ok"
    except Exception as e:                                          # noqa: BLE001
        f["import_prc_weather_features"] = f"{type(e).__name__}: {e}"
    return f


def replay_nodes() -> list[dict]:
    """The nodes this experiment adds: its own label-isolated replay (reduced library)."""
    out = []
    for r in E.ORDER:
        V, S, I, M = E.ROT[r]
        A = [m for m in E.D.MONTHS if m not in (V, S, I)]
        for m in E.MEMBERS:
            out.append({"id": f"replay:{m}:{r}", "kind": "member (label-isolated refit)",
                        "producer_paths": ["experiments/exp-056-q12advocate/kernels/iso/prc-iso.py"],
                        "producer_command": f"bash experiments/exp-056-q12advocate/code/push_iso.sh {m} {r} "
                                            f"iso_{r}_v iso_{r}_s iso_{r}_i iso_{r}_x",
                        "label_months": {"fit": A, "exposure_fit": sorted(set(A) - {M} | {S})},
                        "predicts": {"V": V, "S": S, "I": I, "S_exposed": S},
                        "excludes_S_and_V": True,
                        "recipe_constants": "historical (round budgets, feature sets, clip values chosen on "
                                            "earlier reads that included S and V months; SPEC 1, reading c)",
                        "outputs": [f"artifacts/exp-056/kaggle/{m}--iso_{r}_{w}.parquet" for w in "vsix"]})
        out.append({"id": f"replay:base:{r}", "kind": "blend (fixed rule)",
                    "description": "max(60, mean of the three members); no fitted scalar",
                    "inputs": [f"replay:{m}:{r}" for m in E.MEMBERS], "excludes_S_and_V": True})
        out.append({"id": f"replay:correction:{r}", "kind": "correction",
                    "description": "within-airport shrunk key effects fitted on S, (lam, beta) chosen on "
                                   "S<->I; V read only by the pricing job",
                    "inputs": [f"replay:base:{r}", f"prior:exp-054-rep-{E.REP_FOLD[r]}"],
                    "label_months": {"fit": [S], "selection": [S, I], "score": [V]},
                    "excludes_V": True})
        out.append({"id": f"prior:exp-054-rep-{E.REP_FOLD[r]}", "kind": "nuisance fit (arrival factors)",
                    "producer_paths": ["experiments/exp-054-arrivalrepr/code/represent_r.py"],
                    "outputs": [f"experiments/exp-054-arrivalrepr/work/rep-{E.REP_FOLD[r]}.parquet"],
                    "label_months": {"arrival_taxi_in": [m for m in E.D.MONTHS if m != V]},
                    "uses_S_arrivals": True, "excludes_V": True, "departure_labels": "none"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    NO.refuse_existing([DAG, ARTS], a.overwrite)
    t0 = time.time()
    src = json.loads(SRC.read_text())
    arts, nodes = {}, []
    for n in src["nodes"]:
        rows = []
        for key in ("producer_paths", "outputs"):
            for s in n.get(key) or []:
                for row in census(s.split()[0]):
                    row["role"] = key
                    rows.append(row)
                    arts[row["path"]] = {k: v for k, v in row.items() if k != "role"}
        nn = dict(n)
        nn["census"] = rows
        nn["exists_on_mac"] = all(r_.get("exists_on_mac", False) for r_ in rows) if rows else None
        nn["excludes_S_and_V"] = {r: False for r in E.ORDER}
        nn["exclusion_reason"] = ("existing vector: fitted on 11-month folds or on stackers and "
                                  "selections that read all three scored folds (rotation_exclusion_note)")
        nodes.append(nn)
    rep = replay_nodes()
    dag = {"what": __doc__.strip().splitlines()[0], "source": str(SRC.relative_to(E.ROOT)),
           "source_sha256": E.sha_file(SRC), "built_utc": E.utc(), "code_sha256": E.code_shas(),
           "machine": machine_facts(), "roots": src["roots"], "folds": src["folds"],
           "rotations": {r: dict(zip("VSIM", E.ROT[r])) for r in E.ORDER},
           "nodes": nodes, "replay_nodes": rep,
           "edges": src["edges"] + [{"from": i, "to": x["id"]} for x in rep for i in x.get("inputs", [])],
           "rotation_exclusion_note": src["rotation_exclusion_note"],
           "not_refittable": src["not_refittable"], "open_questions": src["open_questions"],
           "seconds": round(time.time() - t0, 1)}
    missing = sorted(p for p, v in arts.items() if not v.get("exists_on_mac"))
    art = {"what": "every repo path the lineage names, hashed by bytes on this Mac",
           "built_utc": E.utc(), "paths": dict(sorted(arts.items())),
           "count": len(arts), "missing_on_mac": missing}
    NO.refuse_existing([DAG, ARTS], a.overwrite)
    DAG.write_text(json.dumps(dag, indent=1, sort_keys=True, default=str))
    ARTS.write_text(json.dumps(art, indent=1, sort_keys=True))
    print(f"dag: {len(nodes)} nodes + {len(rep)} replay nodes; artifacts: {len(arts)} paths, "
          f"{len(missing)} missing ({dag['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
