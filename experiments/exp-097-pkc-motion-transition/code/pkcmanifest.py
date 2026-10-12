"""exp-097: the frozen source manifest.

Every byte the experiment may read, with its size and full SHA-256, written before any
real value is read and re-verified by the run.  Parquet metadata (row counts, schema)
is recorded; no row value is loaded here.

    OMP_NUM_THREADS=2 .venv/bin/python \
        experiments/exp-097-pkc-motion-transition/code/pkcmanifest.py --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pkcmotion as M                                                  # noqa: E402

ROOT = M.ROOT
OUT = M.EXP / "source-manifest.json"

#: the days the five target months' blocks may load.  `block.load_summaries` widens a
#: month by its neighbouring days, so the manifest covers the month and one day either
#: side of each end.
SPAN = {
    "2025-01": ("2024-12-31", "2025-02-01"),
    "2025-07": ("2025-06-30", "2025-08-01"),
    "2025-12": ("2025-11-30", "2026-01-01"),
    "2026-01": ("2025-12-31", "2026-02-01"),
    "2026-07": ("2026-06-30", "2026-08-01"),
}

#: Named explicitly so the list is a statement and not only an observation; every
#: further module the run imports from inside the repository is added from
#: `sys.modules` by `loaded_code()` (Sol round 1, blocking 6: the first manifest bound
#: neither `rowfeat.py` nor `imports.py` nor the rule and geometry metadata).
CODE = [
    "experiments/exp-046-surfacefit/code/block.py",
    "experiments/exp-046-surfacefit/code/arms.py",
    "experiments/exp-046-surfacefit/code/base.py",
    "experiments/exp-046-surfacefit/code/imports.py",
    "experiments/exp-046-surfacefit/code/nooverwrite.py",
    "experiments/exp-043-pilot/code/segments.py",
    "experiments/exp-045-harvest/code/allsegments.py",
    "prc/data.py",
    "experiments/exp-097-pkc-motion-transition/code/pkcmotion.py",
    "experiments/exp-097-pkc-motion-transition/code/pkccontrol.py",
    "experiments/exp-097-pkc-motion-transition/code/test_pkcmotion.py",
    "experiments/exp-097-pkc-motion-transition/code/test_pkcpath.py",
    "experiments/exp-097-pkc-motion-transition/code/pkcaccept.py",
    "experiments/exp-097-pkc-motion-transition/code/pkcguard.py",
    "experiments/exp-097-pkc-motion-transition/code/pkcvalidate.py",
    "experiments/exp-097-pkc-motion-transition/code/test_pkcaccept.py",
    "experiments/exp-097-pkc-motion-transition/code/test_pkcboundary.py",
    "experiments/exp-097-pkc-motion-transition/code/pkcmanifest.py",
    "experiments/exp-097-pkc-motion-transition/code/ws_run.sh",
    "experiments/exp-097-pkc-motion-transition/code/ws_watch.sh",
]

#: The metadata the matcher reads at run time: which rule wrote each summary day, the
#: airport boxes the stand geometry is measured against, the local summary rule, and the
#: fold population pin.  A missing optional file is recorded as absent, so its later
#: appearance is a mismatch.
METADATA = [
    "experiments/exp-045-harvest/manifest.json",
    "experiments/exp-045-harvest/analysis/boxes.json",
    "experiments/exp-046-surfacefit/analysis/fold-population.json",
]

# `artifacts/exp-046/summaries/rule.json` is NOT bound.  `block.rule_by_day` keys the
# serving rule by summary ROOT: the harvest manifest speaks for `artifacts/exp-045/
# summaries` and that sidecar speaks only for `artifacts/exp-046/summaries`
# (block.py:293-304).  exp-097 reads no file from the second tree, so the sidecar cannot
# reach any byte of this experiment.  Binding it would also make the manifest unverifiable
# off the machine that happens to hold it: it is untracked, `artifacts/` being gitignored.
# `summary_roots` below states the fact positively and `verify` enforces it.
READ_ROOT = "artifacts/exp-045/summaries"
UNREAD_ROOT = "artifacts/exp-046/summaries"


def loaded_code() -> list[str]:
    """Every module under the repository that the import of the run's code pulled in."""
    out = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            rel = Path(f).resolve().relative_to(ROOT)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in ("experiments", "prc", "modeling", "scripts"):
            out.add(str(rel))
    return sorted(out)


def parquet_meta(p: Path) -> dict:
    import pyarrow.parquet as pq                                       # noqa: PLC0415
    f = pq.ParquetFile(p)
    return {"rows": int(f.metadata.num_rows),
            "columns": list(f.schema_arrow.names),
            "types": [str(t) for t in f.schema_arrow.types]}


def record(rel: str, meta: bool = True, optional: bool = False) -> dict:
    p = ROOT / rel
    if not p.exists():
        if optional:
            return {"path": rel, "absent": True}
        raise SystemExit(f"missing source: {rel}")
    r = {"path": rel, "bytes": p.stat().st_size, "sha256": M.sha256(p)}
    if meta and p.suffix == ".parquet":
        r.update(parquet_meta(p))
    return r


def summary_files() -> list[str]:
    out = []
    root = ROOT / "artifacts/exp-045/summaries"
    for airport in sorted(x.name for x in root.iterdir() if x.is_dir()):
        for lo, hi in SPAN.values():
            for f in sorted((root / airport).glob("*.parquet")):
                day = f.stem
                if lo <= day <= hi:
                    out.append(str(f.relative_to(ROOT)))
    return sorted(set(out))


def build() -> dict:
    D = M.prc_data()
    sys.path.insert(0, str(ROOT / "experiments/exp-046-surfacefit/code"))
    import arms as A                                                   # noqa: PLC0415
    rule, rule_report = A.serving_rule()
    man = {
        "experiment": "exp-097-pkc-motion-transition",
        "matcher_sha256": A.matcher_sha(),
        "serving_rule_sha256": rule,
        "serving_rule_cover": rule_report,
        "panels": [record(M.PANEL[f]) for f in M.FOLDS],
        "class_vectors": [record(M.CLASS_FILE[m]) for m in M.RANK_MONTHS],
        "uploaded_v22": record(M.UPLOADED_V22),
        "ranking": record("data/raw/ranking.parquet"),
        "training_months": [record(str(Path(D.month_file(m)).relative_to(ROOT)))
                            for m in D.MONTHS],
        "code": [record(c, meta=False) for c in CODE],
        "metadata": [record(c, meta=False, optional=True) for c in METADATA],
    }
    # everything the import of pkcmotion + arms actually loaded from this repository
    import pkccontrol                                                  # noqa: PLC0415, F401
    seen = {r["path"] for r in man["code"]}
    man["loaded_code"] = [record(c, meta=False) for c in loaded_code() if c not in seen]
    sums = summary_files()
    man["summaries"] = {"files": len(sums),
                        "records": [record(s, meta=False) for s in sums]}
    man["summary_roots"] = {
        "read": READ_ROOT,
        "not_read": UNREAD_ROOT,
        "files_from_read": sum(s.startswith(READ_ROOT) for s in sums),
        "files_from_not_read": sum(s.startswith(UNREAD_ROOT) for s in sums),
        "note": ("the serving rule is keyed by summary root (block.py:293-304); the "
                 "sidecar of the unread root cannot reach this experiment and is not "
                 "bound"),
    }
    return man


def verify(man: dict) -> dict:
    """Re-hash every record without loading a row value."""
    bad = []
    n = 0
    groups = ["panels", "class_vectors", "training_months", "code", "metadata",
              "loaded_code"]
    items = [man["uploaded_v22"], man["ranking"]]
    for g in groups:
        items.extend(man[g])
    items.extend(man["summaries"]["records"])
    for r in items:
        n += 1
        p = ROOT / r["path"]
        if r.get("absent"):
            if p.exists():
                bad.append({"path": r["path"], "why": "appeared since registration"})
            continue
        if not p.exists():
            bad.append({"path": r["path"], "why": "missing"})
            continue
        if p.stat().st_size != r["bytes"]:
            bad.append({"path": r["path"], "why": "size"})
            continue
        if M.sha256(p) != r["sha256"]:
            bad.append({"path": r["path"], "why": "sha256"})
    # the claim that makes the unread root's serving-rule sidecar irrelevant
    roots = man.get("summary_roots")
    if not roots:
        bad.append({"path": "summary_roots", "why": "missing from the manifest"})
    else:
        n += 1
        from_unread = [r["path"] for r in man["summaries"]["records"]
                       if r["path"].startswith(roots["not_read"])]
        if from_unread or roots.get("files_from_not_read"):
            bad.append({"path": roots["not_read"],
                        "why": f"{len(from_unread)} bound summaries come from the root "
                               "this experiment claims not to read"})
        if roots.get("files_from_read") != len(man["summaries"]["records"]):
            bad.append({"path": roots["read"],
                        "why": "not every bound summary comes from the read root"})
    return {"checked": n, "mismatches": bad, "ok": not bad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        man = json.loads(OUT.read_text())
        v = verify(man)
        print(json.dumps({"checked": v["checked"], "ok": v["ok"],
                          "mismatches": v["mismatches"][:20]}, indent=2))
        return 0 if v["ok"] else 1
    man = build()
    if a.write:
        OUT.write_text(json.dumps(man, indent=2, sort_keys=True))
    print(json.dumps({"summaries": man["summaries"]["files"],
                      "code": len(man["code"]),
                      "loaded_code": len(man["loaded_code"]),
                      "metadata": len(man["metadata"]),
                      "matcher_sha256": man["matcher_sha256"],
                      "serving_rule_sha256": man["serving_rule_sha256"],
                      "wrote": str(OUT) if a.write else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
