"""exp-062 ownvalue-refine: the one POSITIVE manifest (SPEC section 2).  BYTES AND FROZEN RECORDS ONLY.

    .venv/bin/python .../code/vp_ctl.py --kind manifest --log logs/manifest.log -- \
        .venv/bin/python .../code/vp_manifest.py

It reads no label, no prediction value, no loss field and no fitted vector.  It parses earlier
records only for hash fields, path fields and the registered identifiers the manifest carries.

What it writes to `analysis/manifest.json`:

 1. `rules`   full SHA-256 of this assignment, assignments/ownvalue.md, COMMON.md, LAPTOP.md,
              docs/input-rule.md and the competitions-root protocol files, plus the SECTION bytes of
              score-feedback rule, recorded rule, recorded rule and recorded rule taken out of the live internal decision record (the whole file is live:
              other sessions append to it, so the section hash is the pin and the file hash is
              recorded as of this run only).
 2. `reviews` the two exp-059 Sol reviews and its round-2 prompt, and exp-055's, exp-057's and
              exp-058's review files.
 3. `environment`  interpreter, platform, library versions, the LightGBM dylib, requirements.txt,
              the git HEAD of this run and `uptime`.
 4. `new_code`  every `code/vp_*.py` file of this experiment with its full SHA-256 and the digest
              of the sorted `path sha256` lines.  The controller refuses to launch anything whose
              bytes differ from these pins (SPEC 2.4), so no code of this experiment can run
              un-pinned after this record is committed.
 5. `exp059_sources`  every ACTUAL exp-059 producer and source, positively enumerated: the named
              classes as literal path tuples or as selections from exp-059's FROZEN census by its
              own recorded `class` field, and the rest derived from exp-059's frozen plan, fit,
              ladder and reproduction records (their `code.files`, `sources.files`,
              `code_files` and `census_check.files_opened` fields).  No live glob is used as a
              source boundary: the boundary is exp-059's committed census plus these literals.
 6. `exp059_census`  every one of the 5,249 recorded paths verified against the hash recorded FOR
              THAT PATH.  A path that is absent or differs is a refusal, except the four exp-056
              code files exp-059 already recorded as mismatch exceptions; those four are NOT
              sources here and their current bytes are verified against exp-059's recorded values.
 7. `reproduction_targets`  the registered values the later processes must reproduce before any new
              interpretation: the k = 24 block hashes, the 23-row count sequence, the fit-free
              anchors, exp-059's fitted V at k = 4, 8 and 12 with their members and intervals,
              February's V* and the band B, and the day-cell receipts they come from.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
ANALYSIS = EXP / "analysis"
OUT = ANALYSIS / "manifest.json"
E46 = ROOT / "experiments/exp-046-surfacefit"
E51 = ROOT / "experiments/exp-051-transferadv"
E55 = ROOT / "experiments/exp-055-originmove"
E57 = ROOT / "experiments/exp-057-originfit"
E58 = ROOT / "experiments/exp-058-origincontrast"
E59 = ROOT / "experiments/exp-059-ownvalue"
CENSUS_PATH = E59 / "analysis/prehash.json"

sys.path.insert(0, str(E46 / "code"))
import nooverwrite as NO                                              # noqa: E402

#: SPEC 2: exp-059's registration commit and its census totals, as its own record states them
E59_SPEC_COMMIT = "49db4157d8bc43edde6c2b8386e90b6f58929578"
E59_CENSUS_FILES = 5_249
E59_CENSUS_BYTES_GB = 4.34
#: SPEC 2.2: exp-059's four recorded mismatch exceptions (against exp-057's census).  They are NOT
#: sources of this experiment; their current bytes must equal the values exp-059 recorded.
E59_EXCEPTIONS = ("experiments/exp-056-q12advocate/code/e56.py",
                  "experiments/exp-056-q12advocate/code/fit.py",
                  "experiments/exp-056-q12advocate/code/freeze.py",
                  "experiments/exp-056-q12advocate/code/price.py")
#: the rule documents
RULES = ("assignments/ownvalue-refine.md", "assignments/ownvalue.md", "assignments/COMMON.md",
         "assignments/LAPTOP.md", "docs/input-rule.md")
DECISIONS_FILE = "docs/DECISIONS.md"
DECISIONS = ("D11", "D32", "D43", "D46")
PROTOCOL_OUTSIDE = ("/path/to/competitions/AGENTS.md", "/path/to/competitions/COMPETING.md",
                    "/path/to/competitions/MODES.md")
REVIEWS = ("experiments/exp-059-ownvalue/review/round1.txt",
           "experiments/exp-059-ownvalue/review/round2.txt",
           "experiments/exp-059-ownvalue/review/round2-prompt.txt",
           "experiments/exp-058-origincontrast/review/round1.txt",
           "experiments/exp-058-origincontrast/review/round1-prompt.txt",
           "experiments/exp-058-origincontrast/review/round2.txt",
           "experiments/exp-058-origincontrast/review/round2-prompt.txt",
           "experiments/exp-057-originfit/review/round1.txt",
           "experiments/exp-057-originfit/review/round1-prompt.txt",
           "experiments/exp-057-originfit/review/round2.txt",
           "experiments/exp-057-originfit/review/round2-prompt.txt",
           "experiments/exp-055-originmove/review/round1.txt",
           "experiments/exp-055-originmove/review/round1-prompt.txt",
           "experiments/exp-055-originmove/review/round2.txt",
           "experiments/exp-055-originmove/review/round2-prompt.txt")
#: the learner, the fit and the bag rule -- "learner settings" in the assignment's list
LEARNER = ("experiments/exp-046-surfacefit/analysis/predeclared.json",
           "experiments/exp-046-surfacefit/code/arms.py",
           "experiments/exp-046-surfacefit/code/chainfold.py",
           "experiments/exp-046-surfacefit/code/bagarm.py",
           "experiments/exp-046-surfacefit/code/classplacebo.py",
           "experiments/exp-046-surfacefit/code/onemonth.py",
           "experiments/exp-046-surfacefit/code/nooverwrite.py")
#: exp-051's construction, exp-055's loaders and equations
CONSTRUCTION = ("experiments/exp-051-transferadv/code/carrier.py",
                "experiments/exp-051-transferadv/code/common.py",
                "experiments/exp-051-transferadv/code/memberspread.py",
                "experiments/exp-051-transferadv/analysis/carrier.json",
                "experiments/exp-051-transferadv/analysis/i2-plan.json",
                "experiments/exp-055-originmove/code/common.py",
                "experiments/exp-055-originmove/code/originvec.py",
                "experiments/exp-055-originmove/code/screen.py",
                "experiments/exp-055-originmove/code/lossfals.py",
                "experiments/exp-055-originmove/analysis/screen.json",
                "experiments/exp-055-originmove/analysis/lossfals.json",
                "experiments/exp-055-originmove/analysis/pin.json",
                "experiments/exp-055-originmove/analysis/prehash.json",
                "experiments/exp-053-transport/analysis/pin.json",
                "experiments/exp-057-originfit/analysis/measure.json",
                "experiments/exp-058-origincontrast/analysis/measure.json",
                "experiments/exp-058-origincontrast/analysis/plan.json",
                "experiments/exp-058-origincontrast/analysis/codepins-r1.json",
                "experiments/exp-058-origincontrast/analysis/costlaw-r1.json")
#: exp-059's registration and final records
E59_RECORDS = ("experiments/exp-059-ownvalue/SPEC.md", "experiments/exp-059-ownvalue/RESULTS.md",
               "experiments/exp-059-ownvalue/climb.log",
               "experiments/exp-059-ownvalue/analysis/prehash.json",
               "experiments/exp-059-ownvalue/analysis/plan.json",
               "experiments/exp-059-ownvalue/analysis/plan-r1.json",
               "experiments/exp-059-ownvalue/analysis/plan3.json",
               "experiments/exp-059-ownvalue/analysis/ladder.json",
               "experiments/exp-059-ownvalue/analysis/ladder2.json",
               "experiments/exp-059-ownvalue/analysis/repro.json",
               "experiments/exp-059-ownvalue/analysis/repro-r1.json",
               "experiments/exp-059-ownvalue/analysis/r8vii-r1.json",
               "experiments/exp-059-ownvalue/analysis/cost.json",
               "experiments/exp-059-ownvalue/code/ov_prehash.py",
               "experiments/exp-059-ownvalue/code/ov_fit.py",
               "experiments/exp-059-ownvalue/code/ov_ladder.py",
               "experiments/exp-059-ownvalue/code/ov_measure.py",
               "experiments/exp-059-ownvalue/code/ov_r8vii_r1.py",
               "experiments/exp-059-ownvalue/code/ov_tables.py")
#: SPEC 3: the near-origin grid rows this experiment fits, and the rows it reproduces, exactly as
#: exp-059's committed analysis/plan.json registered them
REGISTERED_GRID = {
    1: dict(N=2536, cv=[562, 388, 45, 725, 467, 349],
            cv_sha="185f4306b3d4ca86894f312c713d787d249a515ef054e3e8ac04995c64335731",
            ids_block="6d88d5e5282f0a59a0b8b2f2f7a6fcf2c4abd5d93dd397fa42439df3c7e3e641",
            ids_sorted="d5998ebaef8e325e95ca20442be9377a6801f93e50bbe4b83f88c281128570f5"),
    2: dict(N=5068, cv=[1124, 775, 89, 1449, 933, 698],
            cv_sha="14eb68ec5b6552df1de7c45bbca4df564b681063e478bb6f5a5dc4e3e20dcb6f",
            ids_block="ceef4f08b6ac034427c6a2a2a4df650e8524f08f00300c1f530099a918032b1f",
            ids_sorted="b5ef0e371ed03a1be1f0ce7aa79145a428aa553ce7463f2f4646333889f70562"),
    3: dict(N=7601, cv=[1686, 1162, 134, 2173, 1399, 1047],
            cv_sha="572c72136fdd8373f35fdff1b1aeddf3520b1400216ccf5b5daa27fb730a5dee",
            ids_block="c8a5fb8a7d88bddb6fddb6887f77cdccb936b8849c115d402320c8a3cb4213cd",
            ids_sorted="757a8ad481b46a4eeb920e154c7e4f02d9d5351a92a1ce313a6331e3f78e56d5"),
    4: dict(N=10131, cv=[2247, 1549, 178, 2897, 1865, 1395],
            cv_sha="a00ae8c62e2528974d2130ecaeb3846b82c63f53214f9b4841dd8b5b731d1b57",
            ids_block="a605e8702969ddd6ca468afc770a3e179fa18934c28cf80fea0d204dd9c238af",
            ids_sorted="dcc8d01c1d17115a7c31b1140f0c5066850a8e4aaf4072c994f51158754a0292"),
    8: dict(N=20261, cv=[4494, 3098, 356, 5794, 3729, 2790],
            cv_sha="de41d2cfcdeafbce754dbfe8898bbaeab909208a7dd7f4b86e38f6209f83768a",
            ids_block="ceda0adb5bc264a276a4fca25270fbd2a09cffa50a1ab70e2a472547a87cf2e6",
            ids_sorted="74be0770ae2b4b0e2446df3d785c659a7736e312a717a41bb0546cea5894f155"),
    12: dict(N=30391, cv=[6741, 4647, 533, 8691, 5594, 4185],
             cv_sha="dba2f801a9c416659576e72de5687ed51b9c5220253ae3b68eab66e05a3a3e50",
             ids_block="16554eafbfbebce3f7d5f53c6c05ee442904ed5db78da29ca5c9aa3a7b91cea3",
             ids_sorted="1c1527d75538b9b8c662ab38272595be4929c058a4d47bbdb2e55452801c154f"),
    24: dict(N=60777, cv=[13481, 9293, 1066, 17381, 11187, 8369],
             cv_sha="ee89d40963ea3718e88f73757d285786f4bbaafa017f343547a6b82d6776547a",
             ids_block="bdfa7542b4f48eacca11aa5bfab414a04994e6fe26ef385d52c59afcd5b211b8",
             ids_sorted="c9770096dccbbe84fe9f3ecd9f3eb925ae1e2d42fff2bf69668fdc7062baa927"),
}
GRID_N_SEQUENCE = (2_536, 5_068, 7_601, 10_131, 12_666, 15_198, 17_728, 20_261, 22_794, 25_329,
                   27_859, 30_391, 32_924, 35_454, 37_989, 40_521, 43_055, 45_585, 48_117, 50_652,
                   53_182, 55_715, 58_247)
NA = {"EDDF": 13_481, "EDDM": 9_293, "EGLL": 1_066, "EHAM": 17_381, "LEBL": 11_187, "LSZH": 8_369}
SALT = "origincontrast-5801"
V_STAR = 168.58928050561724
BAND = (151.7303524550555, 185.44820855617897)
V_TOP = 493.2541127299781
#: exp-059's three fitted points, from its committed analysis/ladder2.json
E59_FITTED_V = {4: 211.60409594675028, 8: 313.10532588943437, 12: 356.9898753805197}
E59_FITTED_MEMBERS = {
    4: {"s1": 198.71806534359018, "s2": 262.32319839191985, "s3": 191.48830130444878},
    8: {"s1": 359.8958634585765, "s2": 284.0438862521101, "s3": 317.26113833445896},
    12: {"s1": 343.02580611125575, "s2": 409.0850602454033, "s3": 350.1245111832586}}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(files: dict) -> str:
    return hashlib.sha256("".join(f"{k} {v}\n" for k, v in sorted(files.items())).encode()).hexdigest()


def pin(rels, census: dict, note: str) -> dict:
    """Positively pin literal paths: the current bytes, and exp-059's recorded hash when the path is
    in its census.  A path whose bytes differ from the census is a refusal."""
    out, bad = {}, []
    for rel in rels:
        p = ROOT / rel
        if not p.is_file():
            raise SystemExit(f"{rel} is not a file; the manifest cannot pin it")
        s = sha_file(p)
        e = {"sha256": s, "bytes": p.stat().st_size,
             "in_exp059_census": rel in census}
        if rel in census:
            e["exp059_census_sha256"] = census[rel]["sha256"]
            if census[rel]["sha256"] != s:
                bad.append(rel)
        out[rel] = e
    if bad:
        raise SystemExit(f"these pinned paths differ from exp-059's census: {bad}")
    return {"note": note, "n": len(out), "digest": digest({k: v["sha256"] for k, v in out.items()}),
            "files": out}


def decision_sections(text: str) -> dict:
    """The SECTION bytes of each named decision, out of the live internal decision record."""
    heads = [(m.start(), m.group(1)) for m in re.finditer(r"(?m)^## (D\d+) ", text)]
    spans = {}
    for i, (pos, name) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        spans.setdefault(name, (pos, end))
    out = {}
    for d in DECISIONS:
        if d not in spans:
            raise SystemExit(f"{DECISIONS_FILE} has no section {d}")
        a, b = spans[d]
        body = text[a:b].encode()
        out[d] = {"section_sha256": hashlib.sha256(body).hexdigest(), "section_bytes": len(body),
                  "heading": text[a:text.index("\n", a)].strip()}
    return out


def git(*a) -> str:
    r = subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(a)} failed: {r.stderr.strip()[:200]}")
    return r.stdout


def registration() -> dict:
    """SPEC 2: this experiment's own SPEC.md is committed BEFORE this manifest runs, so every later
    record carries the registration commit and no commit hash is written by hand."""
    rel = str((EXP / "SPEC.md").relative_to(ROOT))
    p = ROOT / rel
    if not p.exists():
        raise SystemExit(f"{rel} does not exist: the registration comes before the manifest")
    if git("status", "--porcelain", "--", rel).strip():
        raise SystemExit(f"{rel} is not committed or carries uncommitted changes; the assignment "
                         f"commits SPEC.md before any fit or new value read")
    commit = git("log", "-1", "--format=%H", "--", rel).strip()
    if not commit:
        raise SystemExit(f"{rel} has no commit; the registration comes first")
    return {"spec": rel, "spec_sha256": sha_file(p), "spec_commit": commit,
            "spec_blob_sha1": git("rev-parse", f"{commit}:{rel}").strip(),
            "spec_committed_utc": git("log", "-1", "--format=%cI", "--", rel).strip(),
            "note": "the registration of this experiment: SPEC.md is committed before this manifest "
                    "runs, and every plan, fit, value and measurement record carries this commit"}


def environment() -> dict:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    dylib = ROOT / ".venv/lib/python3.11/site-packages/lightgbm/lib/lib_lightgbm.dylib"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
                          text=True).stdout.strip()
    up = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    return {"python": sys.version.split()[0], "python_executable": sys.executable,
            "python_executable_sha256": sha_file(Path(sys.executable)),
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "numpy": np.__version__, "pandas": pd.__version__,
            "pyarrow": pa.__version__, "lightgbm": lgb.__version__,
            "lightgbm_dylib": str(dylib.relative_to(ROOT)), "lightgbm_dylib_sha256": sha_file(dylib),
            "requirements_txt_sha256": sha_file(ROOT / "requirements.txt"),
            "git_head_at_manifest": head, "uptime_at_manifest": up,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "tz": time.strftime("%z"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def census_verify(census: dict) -> dict:
    """SPEC 2.2: every one of exp-059's 5,249 recorded paths against the hash recorded for it."""
    t0 = time.time()
    ok, differ, absent, nbytes = 0, [], [], 0
    for rel, rec in census.items():
        p = ROOT / rel
        if not p.is_file():
            absent.append(rel)
            continue
        s = sha_file(p)
        nbytes += rec["bytes"]
        if s == rec["sha256"]:
            ok += 1
        else:
            differ.append({"path": rel, "recorded": rec["sha256"], "now": s,
                           "is_recorded_exception": rel in E59_EXCEPTIONS})
    out = {"record": "experiments/exp-059-ownvalue/analysis/prehash.json",
           "record_sha256": sha_file(CENSUS_PATH), "n_paths": len(census), "verified": ok,
           "differ": differ, "absent": absent, "bytes_read": nbytes,
           "seconds": round(time.time() - t0, 1),
           "rule": "every recorded path must carry the hash recorded FOR THAT PATH; a new path "
                   "outside this old manifest is allowed and is not checked here; the only "
                   "tolerated differences are exp-059's four recorded mismatch exceptions"}
    if len(census) != E59_CENSUS_FILES:
        raise SystemExit(f"exp-059's census holds {len(census)} paths, not {E59_CENSUS_FILES}")
    if absent:
        raise SystemExit(f"{len(absent)} recorded paths are absent: {absent[:10]}")
    bad = [d for d in differ if not d["is_recorded_exception"]]
    if bad:
        raise SystemExit(f"{len(bad)} recorded paths differ and are not recorded exceptions: {bad[:5]}")
    return out


def exception_block(census: dict) -> dict:
    """SPEC 2.2: the four exp-056 code files are NOT sources here; their current bytes are verified
    against the values exp-059 recorded for them."""
    mm = {d["path"]: d for d in json.loads(CENSUS_PATH.read_text())["mismatches"]}
    out = {}
    for rel in E59_EXCEPTIONS:
        now = sha_file(ROOT / rel)
        out[rel] = {"exp059_recorded_current_sha256": census[rel]["sha256"],
                    "exp057_recorded_sha256": mm[rel]["recorded"] if rel in mm else None,
                    "now_sha256": now,
                    "equals_exp059_recorded_value": now == census[rel]["sha256"]}
    return {"statement": "these four live exp-056 code files are NOT sources of this experiment: no "
                         "import closure used here holds an exp-056 module, no rule, arm, threshold, "
                         "grid row or value depends on them, and the fit-input guard refuses any fit "
                         "whose process opens a repository file outside this manifest",
            "files": out,
            "all_equal_exp059_recorded_values": all(v["equals_exp059_recorded_value"]
                                                    for v in out.values())}


def from_frozen(census: dict) -> dict:
    """SPEC 2.3: every ACTUAL exp-059 producer and source, enumerated out of exp-059's own frozen
    records -- never out of a live glob.

    A path inside exp-059's census is pinned at the bytes it has now AND must carry the hash
    recorded for it (census_verify has already proved that for every recorded path).  A path outside
    that census is pinned at the bytes it has now and recorded as such: exp-059's census was taken
    before exp-059's own code and records existed, so its plan, fit and ladder records name its own
    code files, which the census cannot hold.  The assignment allows new paths outside the old
    manifest; this keeps the boundary positive for them by pinning their current bytes here."""
    got: dict = {}
    seen_from: dict = {}
    outside: dict = {}

    def take(rels, where: str) -> None:
        for rel in rels:
            p = ROOT / rel
            if not p.is_file():
                raise SystemExit(f"{where} names {rel}, which is not a file; the manifest cannot "
                                 f"pin it")
            s = sha_file(p)
            if rel in census:
                if census[rel]["sha256"] != s:
                    raise SystemExit(f"{where} names {rel}, whose bytes differ from exp-059's "
                                     f"census")
            else:
                outside[rel] = where
            got[rel] = s
            seen_from.setdefault(rel, []).append(where)

    for name in ("plan", "plan-r1", "plan3"):
        p = E59 / f"analysis/{name}.json"
        if not p.is_file():
            continue
        rec = json.loads(p.read_text())
        take(list(rec["code"]["files"]), f"exp-059 {name}.json code.files")
        take(list(rec["sources"]["files"]), f"exp-059 {name}.json sources.files")
    for p in sorted((E59 / "analysis/fits").glob("*.json")):
        rec = json.loads(p.read_text())
        take(list(rec["code_files"]), f"exp-059 fit record {p.stem} code_files")
        take(list(rec.get("sources_opened_by_this_fit_not_in_plan", {})),
             f"exp-059 fit record {p.stem} extra sources")
    for name in ("ladder", "ladder2", "repro", "repro-r1"):
        p = E59 / f"analysis/{name}.json"
        if not p.is_file():
            continue
        rec = json.loads(p.read_text())
        cc = rec.get("census_check") or {}
        take([r for r in (cc.get("files_opened") or []) if r in census],
             f"exp-059 {name}.json census_check.files_opened")
    return {"n": len(got), "digest": digest(got), "files": got,
            "enumerated_from": sorted({w for v in seen_from.values() for w in v}),
            "outside_exp059_census": {"n": len(outside), "files": dict(sorted(outside.items())),
                                      "note": "pinned at the bytes they have now; exp-059's census "
                                              "predates its own code and records, so it cannot hold "
                                              "them"}}


def by_class(census: dict, cls: str) -> list:
    """A selection out of exp-059's FROZEN census by its own recorded class field.  This reads the
    committed record, not the filesystem, so it is not a live glob."""
    return sorted(k for k, v in census.items() if v["class"] == cls)


def exp058_february_arm(census: dict) -> dict:
    """The February arm exp-058 fitted and this experiment compares against: its 12 fit records,
    its 12 vectors, and the block id hash its plan registered."""
    plan = json.loads((E58 / "analysis/plan.json").read_text())
    sel = plan["match"]["months"]["2025-02"]["selection"]
    recs, vecs = [], []
    for s in (1, 2, 3):
        for L in range(4):
            rp = f"experiments/exp-058-origincontrast/analysis/fits/F-s{s}-L{L}.json"
            recs.append(rp)
            r = json.loads((ROOT / rp).read_text())
            if r["block_rows"] != 60_777 or r["month"] != "2025-02":
                raise SystemExit(f"{rp} is not exp-058's 60,777-row February arm")
            vecs.append(r["vector"])
    return {"records": pin(recs, census, "exp-058's 12 February fit records"),
            "vectors": pin(vecs, census, "exp-058's 12 February vectors"),
            "block": {"rows": sel["rows"] if "rows" in sel else 60_777,
                      "ids_sha256_block_order": sel["ids_sha256_block_order"],
                      "ids_sha256_sorted": sel.get("ids_sha256_sorted")}}


def main() -> None:
    t0, c0 = time.time(), time.process_time()
    NO.refuse_existing([OUT], overwrite=False)
    census = json.loads(CENSUS_PATH.read_text())["files"]
    rec = {"what": "exp-062 ownvalue-refine: the one positive manifest (SPEC 2).  Bytes and frozen "
                   "records only: no label, no prediction value, no loss field, no fitted vector.",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "exp059_registration_commit": E59_SPEC_COMMIT}
    rec["exp059_census"] = census_verify(census)
    rec["exp059_census_exceptions"] = exception_block(census)
    rec["registration"] = registration()
    dtext = (ROOT / DECISIONS_FILE).read_text()
    rec["rules"] = {
        "documents": pin(RULES, census, "this assignment, exp-059's assignment, COMMON, LAPTOP and "
                                        "the prediction-time input rule"),
        "decisions_file": {"path": DECISIONS_FILE, "sha256_at_manifest": sha_file(ROOT / DECISIONS_FILE),
                           "note": "a live file: other sessions append decisions to it, so the pin "
                                   "of each decision is its SECTION hash below, not this file hash"},
        "decisions": decision_sections(dtext),
        "protocol_outside_the_repository": {
            q: {"sha256": sha_file(Path(os.path.expanduser(q))),
                "bytes": Path(os.path.expanduser(q)).stat().st_size} for q in PROTOCOL_OUTSIDE},
    }
    rec["reviews"] = pin(REVIEWS, census, "exp-059's two Sol reviews and its round-2 prompt, with "
                                          "exp-055's, exp-057's and exp-058's review files")
    rec["environment"] = environment()
    code = {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(HERE.glob("vp_*.py"))}
    rec["new_code"] = {"n": len(code), "digest": digest(code), "files": code,
                       "note": "every code file of this experiment, pinned before it runs; "
                               "code/vp_ctl.py refuses to launch anything whose bytes differ. "
                               "vp_manifest.py and vp_ctl.py are the disclosed ordering exception "
                               "of SPEC 2.4: they produce these pins, and this manifest is "
                               "committed with them in the registration commit."}
    rec["exp059_sources"] = {
        "boundary": "exp-059's committed 5,249-entry census (verified above) plus the literal "
                    "tuples and frozen-record selections below.  No live glob is a source boundary.",
        "raw_months": pin(by_class(census, "raw"), census, "every raw parquet exp-059's census holds"),
        "carrier_panels": pin(by_class(census, "panel"), census, "the exp-050 v22 fold panels"),
        "class_masks": pin(by_class(census, "class"), census, "exp-046's class masks"),
        "base_vectors": pin(by_class(census, "base") + by_class(census, "block"), census,
                            "the out-of-fold price bases and the month blocks"),
        "feature_cache": pin(by_class(census, "feature-cache"), census, "exp-018's feature cache"),
        "plain_members_and_vectors": pin(by_class(census, "vector") + by_class(census, "work"), census,
                                         "every plain member, carrier, sham and head vector"),
        "receipts": pin(by_class(census, "receipt"), census,
                        "exp-055's, exp-057's and exp-058's receipts"),
        "learner_settings": pin(LEARNER, census, "the learner, the fit, the bag rule and the floor"),
        "construction_and_records": pin(CONSTRUCTION, census,
                                        "exp-051's construction, exp-055's loaders and equations, "
                                        "and the records this experiment reproduces"),
        "exp059_registration_and_final_records": pin(E59_RECORDS, census,
                                                     "exp-059's SPEC, RESULTS, climb log, code and "
                                                     "every analysis record"),
        "exp059_fit_records": pin(sorted(str(p.relative_to(ROOT))
                                         for p in (E59 / "analysis/fits").glob("*.json")), census,
                                  "exp-059's 36 fit records"),
        "exp058_february_arm": exp058_february_arm(census),
        "actual_producers_from_frozen_records": from_frozen(census),
        "june_row_order": {"salt": SALT, "month": "2025-06",
                           "airports": ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LSZH"],
                           "key": "sha256('origincontrast-5801|2025-06|<airport>|<integer id>') "
                                  "ascending, the integer id as the tie-break",
                           "matched_counts_n_a": NA,
                           "registered_by": "experiments/exp-059-ownvalue/analysis/plan.json"},
        "per_airport_prefix_counts": {str(k): {"N": v["N"], "count_vector": v["cv"],
                                               "count_vector_sha256": v["cv_sha"],
                                               "ids_sha256_block_order": v["ids_block"],
                                               "ids_sha256_sorted": v["ids_sorted"]}
                                      for k, v in REGISTERED_GRID.items()},
        "fitting_frames": {p.stem: json.loads(p.read_text())["frame_sha256"]
                           for p in sorted((E59 / "analysis/fits").glob("*.json"))},
        "draws": {"generator": "np.random.default_rng(5500 + L), L = 1, 2, 3",
                  "exp055_draw_sha256": json.loads((E55 / "analysis/screen.json").read_text())["draw_sha256"],
                  "exp057_k10_member_draw_sha256_unsorted":
                      json.loads((E57 / "analysis/measure.json").read_text())["k10_member_draw_sha256_unsorted"]},
    }
    rec["reproduction_targets"] = {
        "k24_block": REGISTERED_GRID[24],
        "grid_N_sequence_k1_to_k23": list(GRID_N_SEQUENCE),
        "anchors": {"V_at_0": 0.0, "V_at_60777": V_TOP},
        "february": {"V_star": V_STAR, "band_B": list(BAND), "band_rule": "V* +/- 10 %"},
        "exp059_fitted_V": {str(k): v for k, v in E59_FITTED_V.items()},
        "exp059_fitted_members": {str(k): v for k, v in E59_FITTED_MEMBERS.items()},
        "day_cells": ["experiments/exp-058-origincontrast/analysis/receipts/daysums-july_loo.csv",
                      "experiments/exp-058-origincontrast/analysis/receipts/days-july_loo.csv",
                      "experiments/exp-057-originfit/analysis/receipts/daysums-july_loo.csv",
                      "experiments/exp-055-originmove/analysis/receipts/i2-daysums-july_loo.csv",
                      "experiments/exp-055-originmove/analysis/receipts/i1-daysums-rank-2026-07.csv"],
        "populations": {"exp058_leaves": 2343, "exp057_leaves": 1287,
                        "rule": "every leaf of both recorded population blocks is compared and a "
                                "missing path on either side is refused (exp-059 review round 2, "
                                "findings 2 and 3)"},
    }
    #: the union of every pinned repository path: the positive source boundary the later processes use
    allp: dict = {}
    def collect(o) -> None:
        if isinstance(o, dict):
            if "files" in o and isinstance(o["files"], dict) and "note" in o:
                for k, v in o["files"].items():
                    allp[k] = v["sha256"] if isinstance(v, dict) else v
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)
    collect(rec["exp059_sources"])
    collect(rec["rules"])
    collect(rec["reviews"])
    allp |= rec["exp059_sources"]["actual_producers_from_frozen_records"]["files"]
    allp |= code
    rec["source_boundary"] = {"n": len(allp), "digest": digest(allp), "files": allp,
                              "rule": "every repository file a later process of this experiment "
                                      "opens must be in here with these bytes, or under "
                                      "experiments/exp-062-ownvalue-refine/ or artifacts/exp-062/; "
                                      "anything else is recorded as a new unrelated path and "
                                      "refuses a fit"}
    rec["seconds"] = round(time.time() - t0, 1)
    rec["seconds_process_cpu"] = round(time.process_time() - c0, 1)
    NO.refuse_existing([OUT], overwrite=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, sort_keys=True, default=float))
    print(f"exp-059 census: {rec['exp059_census']['verified']} of {rec['exp059_census']['n_paths']} "
          f"paths verified, {len(rec['exp059_census']['differ'])} differ, "
          f"{len(rec['exp059_census']['absent'])} absent, {rec['exp059_census']['bytes_read'] / 1e9:.2f} GB "
          f"in {rec['exp059_census']['seconds']} s", flush=True)
    print(f"four recorded exceptions all equal exp-059's values: "
          f"{rec['exp059_census_exceptions']['all_equal_exp059_recorded_values']}", flush=True)
    print(f"positive manifest: {len(allp)} pinned repository paths, digest "
          f"{rec['source_boundary']['digest'][:16]}; new code {rec['new_code']['n']} files, digest "
          f"{rec['new_code']['digest'][:16]}", flush=True)
    print(f"actual exp-059 producers from frozen records: "
          f"{rec['exp059_sources']['actual_producers_from_frozen_records']['n']} paths", flush=True)
    print(f"written {OUT.relative_to(ROOT)} ({rec['seconds']} s wall, "
          f"{rec['seconds_process_cpu']} s process cpu)", flush=True)


if __name__ == "__main__":
    main()
