"""exp-057 originfit: the byte census taken BEFORE the registration commit.  BYTES ONLY.

This script opens every file in the classes below as bytes, writes its full SHA-256 and size,
and compares each with every earlier recorded hash of the same path that it knows.  It does not
parse a prediction value, a label or a loss field: the only JSON it parses are exp-051's fit
records, exp-055's census and the two exp-055 records' receipt and OOF hash fields, and only the
hash fields are taken from them.  It imports nothing from the experiments.

A mismatch with an earlier record is fatal (exit 2) and is written into the record.

    .venv/bin/python experiments/exp-057-originfit/code/ofit_prehash.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
OUT = EXP / "analysis/prehash.json"
E46 = "experiments/exp-046-surfacefit"
E51 = "experiments/exp-051-transferadv"
E55 = "experiments/exp-055-originmove"
RANKING_PREFIX, RANKING_SUFFIX = "9db75e3a", "325f86"

#: (class, glob relative to ROOT).  A superset by directory of every source the fit path and
#: the measurement read: the import closure (every experiment code file), the configurations
#: and records it opens, and the data directories that its loaders name.
CLASSES = [
    ("code", "experiments/*/code/**/*.py"),
    ("code", f"{E46}/analysis/chain-check/*.py"),
    ("code", "prc/*.py"),
    ("config", f"{E46}/analysis/**/*.json"),
    ("config", "experiments/exp-036-mask/analysis/*.json"),
    ("config", "experiments/exp-047-queue/analysis/interface.json"),
    ("config", "experiments/exp-045-harvest/analysis/boxes.json"),
    ("config", "experiments/exp-045-harvest/manifest.json"),
    ("record", f"{E51}/analysis/**/*.json"),
    ("record", f"{E55}/analysis/**/*.json"),
    ("receipt", f"{E55}/analysis/receipts/*.csv"),
    ("record", "experiments/exp-053-transport/analysis/*"),
    ("oof", f"{E51}/oof/*.parquet"),
    ("oof", f"{E55}/oof/*.parquet"),
    ("raw", "data/raw/*.parquet"),
    ("block", "artifacts/exp-047/block/*"),
    ("base", "experiments/exp-047-queue/oof/*"),
    ("work", f"{E46}/work/*"),
    ("summaries", "artifacts/exp-045/summaries/**/*"),
    ("summaries", "artifacts/exp-046/summaries/**/*"),
    ("class", "artifacts/exp-046/class/*"),
    ("vector", "artifacts/exp-051/work/*"),
    ("oof", "experiments/exp-016-stack/oof/*"),
    ("oof", "experiments/exp-026-rebuild/oof/*"),
    ("oof", "experiments/exp-027-step/oof/*"),
    ("oof", "experiments/exp-028-window/oof/*"),
    ("zerosum", "experiments/exp-044-zerosum/**/*"),
    ("base", "submissions/candidates/exp-035-k2climb-v11member.parquet"),
    ("upload", "submissions/uploaded/*.parquet"),
    ("panel", "experiments/exp-050-nmblind/work/panel-v22-*.parquet"),
]
SKIP_PARTS = ("__pycache__",)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def earlier_records() -> dict:
    """path -> list of (record, sha256) from hash fields of earlier committed records."""
    out: dict = {}

    def add(path: str, rec: str, sha: str) -> None:
        out.setdefault(path, []).append({"record": rec, "sha256": sha})

    ph = json.loads((ROOT / E55 / "analysis/prehash.json").read_text())
    for key, v in ph["files"].items():
        add(v["path"], f"{E55}/analysis/prehash.json:{key}", v["sha256"])
    r1 = json.loads((ROOT / E55 / "analysis/prehash-r1.json").read_text())
    for m, v in r1["raw_months"].items():
        add(v["path"], f"{E55}/analysis/prehash-r1.json:raw_months.{m}", v["sha256"])
    for sub in ("fits", "fits2"):
        for js in sorted((ROOT / E51 / "analysis" / sub).glob("*.json")):
            rec = json.loads(js.read_text())
            add(rec["vector"], f"{E51}/analysis/{sub}/{js.name}:vector_sha256", rec["vector_sha256"])
    for js in sorted((ROOT / E51 / "analysis").glob("i2-integrity-*.json")):
        rec = json.loads(js.read_text())
        add(f"artifacts/exp-051/work/i2-{rec['tag']}.npz", f"{E51}/analysis/{js.name}:refit_sha256",
            rec["refit_sha256"])
        add(rec["reference"], f"{E51}/analysis/{js.name}:reference_sha256", rec["reference_sha256"])
    for name in ("screen.json", "lossfals.json"):
        rec = json.loads((ROOT / E55 / "analysis" / name).read_text())
        for path, sha in rec["receipts"].items():
            add(path, f"{E55}/analysis/{name}:receipts", sha)
        if "oof" in rec:
            add(rec["oof"]["path"], f"{E55}/analysis/{name}:oof.sha256", rec["oof"]["sha256"])
    return out


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT.relative_to(ROOT)} exists; the census is write-once")
    t0 = time.time()
    files: dict = {}
    for cls, pat in CLASSES:
        hits = sorted(p for p in ROOT.glob(pat) if p.is_file() and not any(s in p.parts for s in SKIP_PARTS))
        if not hits:
            raise SystemExit(f"class {cls}: {pat} matches no file")
        for p in hits:
            rel = str(p.relative_to(ROOT))
            if rel in files:
                continue
            files[rel] = {"class": cls, "bytes": p.stat().st_size, "sha256": sha_file(p)}
    early = earlier_records()
    mismatches, checked = [], 0
    for path, recs in early.items():
        if path not in files:
            p = ROOT / path
            if not p.exists():
                mismatches.append({"path": path, "why": "an earlier record names it, and it does not exist"})
                continue
            files[path] = {"class": "record-named", "bytes": p.stat().st_size, "sha256": sha_file(p)}
        files[path]["earlier"] = [dict(r, match=bool(r["sha256"] == files[path]["sha256"])) for r in recs]
        checked += 1
        for r in files[path]["earlier"]:
            if not r["match"]:
                mismatches.append({"path": path, "record": r["record"], "recorded": r["sha256"],
                                   "now": files[path]["sha256"]})
    rk = files.get("data/raw/ranking.parquet", {}).get("sha256", "")
    if not (rk.startswith(RANKING_PREFIX) and rk.endswith(RANKING_SUFFIX)):
        mismatches.append({"path": "data/raw/ranking.parquet", "why": f"not 9db75e3a...325f86: {rk}"})
    by_class: dict = {}
    for v in files.values():
        c = by_class.setdefault(v["class"], {"files": 0, "bytes": 0})
        c["files"] += 1
        c["bytes"] += v["bytes"]
    rec = {"what": __doc__.strip().splitlines()[0],
           "classes": [{"class": c, "glob": g} for c, g in CLASSES],
           "n_files": len(files), "by_class": by_class,
           "paths_with_an_earlier_recorded_hash": checked,
           "earlier_records_matching": sum(1 for v in files.values() for r in v.get("earlier", []) if r["match"]),
           "mismatches": mismatches, "files": dict(sorted(files.items())),
           "seconds": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"{len(files)} files hashed ({sum(v['bytes'] for v in files.values()) / 1e9:.2f} GB) in "
          f"{rec['seconds']} s; {checked} paths carry an earlier recorded hash, "
          f"{rec['earlier_records_matching']} earlier records match; mismatches {len(mismatches)}")
    for c, v in sorted(by_class.items()):
        print(f"  {c:12s} {v['files']:5d} files {v['bytes'] / 1e6:9.1f} MB")
    if mismatches:
        print(json.dumps(mismatches, indent=1))
        sys.exit(2)


if __name__ == "__main__":
    main()
