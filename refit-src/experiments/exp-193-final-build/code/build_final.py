"""exp-193 final build (SPEC.md): run a producer step from a git snapshot, compare a rebuilt part, assemble a file.

Runs on remote inside a snapshot made by drive.py (the snapshot root holds this file at
experiments/exp-193-final-build/code/build_final.py), under arc_holdguard.sh:

    build_final.py run-step --step g0 --root $PRC_FINAL_ROOT            # produce the parts
    build_final.py run-step --step g0 --root $PRC_FINAL_ROOT/release73/rebuild/<id>   # rebuild
    build_final.py compare  --step g0 --rebuild $PRC_FINAL_ROOT/release73/rebuild/<id>
    build_final.py assemble --target final-g0 --out $PRC_FINAL_ROOT/release73/files/final-g0/<id>
    build_final.py check-file --from-build <files/final/<id>> --name youthful-giraffe_v<N>.parquet --out <dir>

`check-file` is the target `final-check`: the January/July rows of a built V (or Z) file, 344,841 rows in
submitting.parquet order, for one board check upload (the desk uploads). Every target refuses the
published-mean-shift recipe (exp-151; desk 10-08 14:08 UTC).

Parts are accepted only by the SHA-256 of their producer's MANIFEST row (the MANIFEST in this snapshot).
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SNAP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SNAP))  # bind the snapshot's prc package (memory: Mac PYTHONPATH empty entry)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
REG = json.loads((SNAP / "experiments/exp-193-final-build/targets.json").read_text())
ROWS = 670_790
BOARD_ROWS = 344_841
# Build guard (desk, 10-08 14:08 UTC): the final targets refuse the published-mean-shift recipe (exp-151; the
# release registry uses it for v37/v40 only, and no v73 part reads it).
FORBIDDEN = ("published-mean-shift", "published_mean_shift", "exp-151-published-means")


def refuse_forbidden(where: str, text: str) -> None:
    hit = [f for f in FORBIDDEN if f in text]
    if hit:
        raise SystemExit(f"{where}: refused, it names the published-mean-shift recipe {hit}")


# Byte checks run on remote x86 (COMMON, finalplan 14:14 UTC); a check across machines (an x86 part against the
# Mac-made reference table v73-parts.parquet) uses this tolerance and reports the largest difference.
CROSS_MACHINE_TOL_S = 1e-9


def cross_machine_check(new_file: str, col: str, alias: str) -> dict:
    """A deterministic part against its v73-era reference on the board rows, at full precision (parity.py rounds
    its differences to 1e-4). Numbers: |new - ref| <= 1e-9 s and NaN at the same rows; text and bool: equal."""
    sys.path.insert(0, str(SNAP / "experiments/exp-186-final-core/code"))
    from parity import ALIASES
    refs = REG["refs"]
    ref_col = ALIASES.get(alias, alias)
    r = pd.read_parquet(refs["path"], columns=[ID, ref_col])
    n = pd.read_parquet(new_file, columns=[ID, col])
    n[ID] = n[ID].astype("float64")
    r[ID] = r[ID].astype("float64")
    j = r.merge(n, on=ID, how="left", suffixes=("", "__new"), validate="one_to_one")
    assert len(j) == BOARD_ROWS, len(j)
    a = j[col if col != ref_col else col + "__new"]
    b = j[ref_col]
    rec = {"part_col": col, "ref_col": ref_col, "rows": BOARD_ROWS, "tolerance_s": CROSS_MACHINE_TOL_S}
    if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
        x, y = a.to_numpy("float64"), b.to_numpy("float64")
        fx, fy = np.isfinite(x), np.isfinite(y)
        both = fx & fy
        d = np.abs(x[both] - y[both])
        rec |= {"finite_mismatch": int((fx != fy).sum()), "bits_equal": int((x.view("u8") == y.view("u8")).sum()),
                "rows_differ": int((d > 0).sum()), "max_abs_diff_s": float(d.max()) if both.any() else 0.0}
        ok = rec["finite_mismatch"] == 0 and rec["max_abs_diff_s"] <= CROSS_MACHINE_TOL_S
    else:
        na, nb = a.isna().to_numpy(), b.isna().to_numpy()
        eq = (a.astype(object).to_numpy() == b.astype(object).to_numpy()) | (na & nb)
        rec["rows_differ"] = int((~eq).sum())
        ok = rec["rows_differ"] == 0
    rec["verdict"] = "PASS" if ok else "FAIL"
    return rec


def adsb_gap_read(ids: np.ndarray, values: np.ndarray) -> dict:
    """H8 information read (desk, DATA_CARD 48935ed22): on 2026-06-11 only EDDM and EHAM have adsb.lol summaries,
    so departures of that day at the other airports have no harvest points. Rows, finite values, mean and spread
    of the file's values by airport on that day (UTC day of take-off), next to 2026-06-10 and 2026-06-12."""
    from prc import data as D
    fr = D.load_ranking(columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"], mode="final")
    fr = fr[fr.PHASE_mvt.eq("DEP")].set_index(ID)
    meta = fr.loc[ids]
    day = meta.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy()
    airport = meta.ADEP_mvt.to_numpy()
    with_summaries = ("EDDM", "EHAM")
    rec = {"source": "DATA_CARD 48935ed22 (finaldata ad7cce060)", "day_rule": "UTC day of MVT_TIME_UTC_mvt",
           "airports_with_summaries_on_06_11": list(with_summaries), "by_airport": {}}
    gap = (day == "2026-06-11") & ~np.isin(airport, with_summaries)
    rec["gap_rows"] = int(gap.sum())
    rec["gap_rows_nonfinite"] = int((~np.isfinite(values[gap])).sum())
    for a in sorted(set(airport)):
        rec["by_airport"][a] = {}
        for d in ("2026-06-10", "2026-06-11", "2026-06-12"):
            x = values[(airport == a) & (day == d)]
            f = np.isfinite(x)
            rec["by_airport"][a][d] = {"rows": int(len(x)), "nonfinite": int((~f).sum()),
                                       **({"mean": round(float(x[f].mean()), 1), "sd": round(float(x[f].std()), 1),
                                           "q10": round(float(np.quantile(x[f], .1)), 1),
                                           "q50": round(float(np.quantile(x[f], .5)), 1),
                                           "q90": round(float(np.quantile(x[f], .9)), 1)} if f.any() else {})}
    return rec


def guard_snapshot() -> dict:
    """The recipe guard of a run. It refuses: a registry that names the recipe; a snapshot path in an exp-151 folder
    (the recipe's code and parameters); the sibling worktree next to the snapshot from which exploit029.py would read
    the recipe's inputs; a code file (outside exp-193) that names the recipe, unless the registry's `recipe_named_in`
    pins it by SHA-256 (exploit029.py: a development-only branch, build(published=False), that no final step calls;
    10-09 09:57 UTC). Accepted MANIFEST rows and JSON parts get the same text check (accepted_part, assemble).
    Returns the files read and the pinned files found."""
    refuse_forbidden("targets.json", json.dumps(REG))
    sibling = SNAP.parent / "prc-data-challenge-2026-exp151"
    if sibling.exists():
        raise SystemExit(f"{sibling}: refused, the recipe's inputs sit next to the snapshot")
    pinned, seen, named = REG.get("recipe_named_in", {}), [], {}
    for f in sorted(SNAP.rglob("*")):
        rel = f.relative_to(SNAP).as_posix()
        if any(part.startswith("exp-151") for part in rel.split("/")):
            raise SystemExit(f"{rel}: refused, a path of the recipe's experiment (exp-151)")
        if f.is_file() and f.suffix in (".py", ".sh", ".json", ".tsv", ".patch") and not rel.startswith(
                "experiments/exp-193-final-build/"):
            text = f.read_text(errors="replace")
            hit = [x for x in FORBIDDEN if x in text]
            if hit:
                got = sha256(f)
                if pinned.get(rel) != got:
                    raise SystemExit(f"{rel}: refused, it names the published-mean-shift recipe {hit} "
                                     f"(SHA-256 {got[:12]}; not pinned in recipe_named_in)")
                named[rel] = got
            seen.append(rel)
    return {"files_read": seen, "pinned_files_named": named}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def template_ids() -> np.ndarray:
    from prc import data as D
    t = D.load_submitting(mode="final")
    ids = t[ID].to_numpy("float64")
    assert len(ids) == ROWS and pd.Index(ids).is_unique, len(ids)
    return ids


def month_of_template() -> np.ndarray:
    from prc import data as D
    fr = D.load_ranking(columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"], mode="final")
    fr = fr[fr.PHASE_mvt.eq("DEP")].set_index(ID)
    return fr.loc[template_ids(), "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy()


def manifest_rows(path: Path) -> dict:
    """Last MANIFEST row per part name (tab separated, the COMMON header of the part contract)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = {}
    for r in rows:
        # COMMON header: part remote_path sha256 rows columns ...; a part cell may carry a note "<name> (...)",
        # and the tolerance column is `tolerance` or `rebuild_tolerance`. finaldata's MANIFEST (location path
        # sha256 ...) has no part column: its rows are named by their path.
        name = r["part"].split(" (")[0].strip() if "part" in r else r["path"]
        path = r.get("remote_path") or r["path"]
        out[name] = {"part": name, "path": path, "sha256": r["sha256"], "command": r.get("command"),
                     "tolerance": r.get("tolerance") or r.get("rebuild_tolerance"), "row": r}
    return out


def part_ref(name: str) -> tuple[str, str, str]:
    """'producer/part:col' -> (producer, part, col)."""
    head, col = name.split(":")
    producer, part = head.split("/")
    return producer, part, col


def part_file(root: Path, step: dict, part: str) -> Path:
    """Where a step writes a part under a root: the part's own `path` (under the root), <producer>/parts, or the
    step's `part_dir` (a folder the producer's script makes itself, example finalplan's parts/<name>)."""
    if "path" in step["parts"][part]:
        return root / step["parts"][part]["path"]
    return root / step.get("part_dir", f"{step['producer']}/parts") / (part + ".parquet")


def local_sources(step: dict, root: Path, work: Path, linked: dict) -> Path:
    """finalplan's sources JSON (step `sources`, written by exp-192 final_sources.py at the snapshot commit, with its
    -from record) with each part path moved into the root: root/<producer>/parts/<part>.parquet, a link to the
    MANIFEST file after its SHA-256 check. A file already there must have the same SHA-256. Returns the new JSON."""
    src = SNAP / step["sources"]
    sources = json.loads(src.read_text())
    origin = json.loads(src.with_name(src.stem + "-from.json").read_text())["origin"]
    producer_of = {v: k for k, v in REG["manifests"].items()}
    where = {}
    for o in origin.values():
        where.setdefault(o["path"], (producer_of[o["manifest"]], o["part"], o["sha256"]))
        assert where[o["path"]] == (producer_of[o["manifest"]], o["part"], o["sha256"]), o

    def move(entry: dict) -> dict:
        prod, part, sha = where[entry["path"]]
        assert sha == entry["sha256"], entry
        row = accepted_part(prod, part)
        if row["sha256"] != sha:
            raise SystemExit(f"{prod}/{part}: sources JSON names {sha[:12]}, the MANIFEST row {row['sha256'][:12]}")
        dest = root / prod / "parts" / (part + ".parquet")
        if dest.exists():
            if sha256(dest) != sha:
                raise SystemExit(f"{dest} is in the root with another SHA-256")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(row["file"])
        linked[f"{prod}/{part}"] = sha
        return {**entry, "path": str(dest)}

    new = {k: (move(v) if "path" in v else {c: move(e) for c, e in v.items()}) for k, v in sources.items()}
    work.mkdir(parents=True, exist_ok=True)
    out = work / "sources.json"
    if out.exists():
        raise SystemExit(f"refusing to replace {out}")
    out.write_text(json.dumps(new, indent=1) + "\n")
    return out


def step_of(producer: str, part: str) -> dict:
    for s in REG["steps"].values():
        if s["producer"] == producer and part in s["parts"]:
            return s
    raise KeyError(f"no step writes {producer}/{part}")


@functools.lru_cache(maxsize=None)
def accepted_part(producer: str, part: str) -> dict:
    """The MANIFEST row of a part or of an input table (the producer's MANIFEST in `manifests`), after the file
    check (SHA-256)."""
    row = manifest_rows(SNAP / REG["manifests"][producer])[part]
    refuse_forbidden(f"{producer}/{part} MANIFEST row", json.dumps(row["row"]))
    path = Path(row["path"])
    got = sha256(path)
    if got != row["sha256"]:
        raise SystemExit(f"{producer}/{part}: file SHA-256 {got} differs from its MANIFEST row {row['sha256']}")
    return {**row, "file": str(path)}


def read_part(path: Path, col: str, ids: np.ndarray) -> np.ndarray:
    f = pd.read_parquet(path, columns=[ID, col])
    assert len(f) == ROWS, (path, len(f))
    assert np.array_equal(f[ID].to_numpy("float64"), ids), f"{path}: keys differ from the template order"
    assert f[col].dtype == np.float64, (path, col, f[col].dtype)
    return f[col].to_numpy()


def fill(arg: str, root: Path, producer: str, step: str) -> str:
    """Placeholders of a step's command and env (anywhere in the string): {python}, {snap}, {root}, {work} (the
    step's own new work folder under the root), {out:<part>}, {in:<producer>/<part>} (an upstream part, linked
    into the root), {input:<producer>/<name>} and {input_sha:<producer>/<name>} (an input table at its MANIFEST
    path, read in place after its SHA-256 check)."""
    def one(m: re.Match) -> str:
        key, _, val = m.group(1).partition(":")
        if key == "python":
            return REG["python"]
        if key == "snap":
            return str(SNAP)
        if key == "root":
            return str(root)
        if key == "work":
            return str(root / producer / "work" / step)
        if key == "out":
            return str(root / producer / "parts" / (val + ".parquet"))
        if key == "in":
            p, part = val.split("/")
            return str(root / p / "parts" / (part + ".parquet"))
        if key in ("input", "input_sha"):
            row = accepted_part(*val.split("/", 1))
            return row["file"] if key == "input" else row["sha256"]
        raise SystemExit(f"step {step}: unknown placeholder {m.group(0)}")
    return re.sub(r"\{([a-z_]+(?::[^{}]+)?)\}", one, arg)


def run_step(a):
    step = REG["steps"][a.step]
    root = Path(a.root)
    data = SNAP / "data" / "raw"
    if not data.exists():
        data.parent.mkdir(parents=True, exist_ok=True)
        data.symlink_to(REG["data_dir"])
    # Upstream parts: in a rebuild root, link each MANIFEST part after its SHA-256 check.
    linked = {}
    for arg in step["command"] + list(step.get("env", {}).values()):
        for val in re.findall(r"\{in:([^{}]+)\}", arg):
            p, part = val.split("/")
            row = accepted_part(p, part)
            dest = root / p / "parts" / (part + ".parquet")
            if Path(row["file"]) != dest:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.symlink_to(row["file"])
                elif sha256(dest) != row["sha256"]:  # an upstream step of this root wrote it: accepted bytes only
                    raise SystemExit(f"{dest} differs from the MANIFEST part {p}/{part} ({row['sha256'][:12]})")
            linked[f"{p}/{part}"] = row["sha256"]
        for val in re.findall(r"\{input(?:_sha)?:([^{}]+)\}", arg):
            linked[val] = accepted_part(*val.split("/", 1))["sha256"]
    # Input tables at the paths a producer's script expects under its work folder (step "links": path under the
    # root -> placeholder), each a symlink to the MANIFEST file after its SHA-256 check.
    for rel, target in step.get("links", {}).items():
        for val in re.findall(r"\{input(?:_sha)?:([^{}]+)\}", target):
            linked[val] = accepted_part(*val.split("/", 1))["sha256"]
        dest = root / rel
        if dest.exists() or dest.is_symlink():
            raise SystemExit(f"refusing to replace {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(fill(target, root, step["producer"], a.step))
    if "sources" in step:
        local_sources(step, root, Path(fill("{work}", root, step["producer"], a.step)), linked)
    outs = {part: part_file(root, step, part) for part in step["parts"]}
    for part, o in outs.items():
        if o.exists():
            raise SystemExit(f"refusing to overwrite {o}")
        # a part_dir or a part's own path is made by the producer's script (it refuses an existing folder)
        if "part_dir" not in step and "path" not in step["parts"][part]:
            o.parent.mkdir(parents=True, exist_ok=True)
    argv = [fill(x, root, step["producer"], a.step) for x in step["command"]]
    step_env = {k: fill(v, root, step["producer"], a.step) for k, v in step.get("env", {}).items()}
    guard_snapshot()
    refuse_forbidden(f"step {a.step} command", " ".join(argv) + json.dumps(step_env))
    env = dict(os.environ, PRC_FINAL_ROOT=str(root), PRC_DATA_DIR=REG["data_dir"], PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS=str(step["threads"]), OPENBLAS_NUM_THREADS=str(step["threads"]),
               MKL_NUM_THREADS=str(step["threads"]), PYTHONPATH=str(SNAP))
    env.update(step_env)  # a step's own env wins (example: PRC_FINAL_ROOT for a script that reads its inputs there)
    if not step.get("pythonpath", True):  # the producer's run had no PYTHONPATH (its scripts set their own paths)
        env.pop("PYTHONPATH")
    env.pop("PRC_KAGGLE_USER", None)
    t0, started = time.time(), now()
    print(started, "run-step", a.step, "root", root, "argv", argv, flush=True)
    proc = subprocess.run(argv, cwd=SNAP, env=env)
    rec = {"step": a.step, "producer": step["producer"], "root": str(root), "snapshot": str(SNAP),
           "snapshot_record": json.loads((SNAP / "SNAPSHOT.json").read_text()) if (SNAP / "SNAPSHOT.json").exists() else None,
           "argv": argv, "env": step_env, "python3": shutil.which("python3", path=env.get("PATH")),
           "threads": step["threads"], "exit_code": proc.returncode, "inputs": linked,
           "minutes": round((time.time() - t0) / 60, 2), "started_utc": started, "finished_utc": now(), "host": platform.node(), "parts": {}}
    if proc.returncode == 0:
        ids = template_ids()
        for part, o in outs.items():
            if step["parts"][part].get("table"):  # an input table of another row set: SHA-256 and size only
                rec["parts"][part] = {"path": str(o), "sha256": sha256(o), "bytes": o.stat().st_size, "table": True}
                continue
            f = pd.read_parquet(o)
            assert len(f) == ROWS and np.array_equal(f[ID].to_numpy("float64"), ids), f"{o}: rows or key order"
            rec["parts"][part] = {"path": str(o), "sha256": sha256(o), "rows": len(f), "columns": list(f.columns),
                                  "dtypes": {k: str(v) for k, v in f.dtypes.items()}}
    runs = root / "release73" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    out = runs / f"{a.step}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1), flush=True)
    sys.exit(proc.returncode)


def diff_stats(new: np.ndarray, old: np.ndarray, month: np.ndarray) -> dict:
    both = np.isfinite(new) & np.isfinite(old)
    d = new[both] - old[both]
    res = {"rows_both_finite": int(both.sum()), "finite_new_only": int((np.isfinite(new) & ~np.isfinite(old)).sum()),
           "finite_old_only": int((~np.isfinite(new) & np.isfinite(old)).sum()), "equal": int((d == 0).sum()),
           "rms": float(np.sqrt(np.mean(d ** 2))) if len(d) else None, "bias": float(d.mean()) if len(d) else None,
           "max_abs": float(np.abs(d).max()) if len(d) else None, "by_month": {}}
    for m in sorted(set(month)):
        s = both & (month == m)
        dm = new[s] - old[s]
        res["by_month"][m] = {"rows": int(s.sum()), "rms": float(np.sqrt(np.mean(dm ** 2))) if s.any() else None,
                              "max_abs": float(np.abs(dm).max()) if s.any() else None}
    return res


def compare(a):
    step = REG["steps"][a.step]
    ids, month = template_ids(), month_of_template()
    rec = {"step": a.step, "rebuild": a.rebuild, "parts": {}, "verdict": "PASS"}
    for part, spec in step["parts"].items():
        row = accepted_part(step["producer"], part)
        new_path = part_file(Path(a.rebuild), step, part)
        r = {"manifest_sha256": row["sha256"], "rebuild_sha256": sha256(new_path), "check": spec["check"]}
        r["same_sha256"] = r["manifest_sha256"] == r["rebuild_sha256"]
        if spec.get("table"):  # an input table of another row set (example features-r4c): SHA-256 only
            assert spec["check"] == "sha256", (part, spec)
            r["verdict"] = "PASS" if r["same_sha256"] else "FAIL"
            if not r["same_sha256"]:
                rec["verdict"] = "FAIL"
            rec["parts"][part] = r
            continue
        f_new, f_old = pd.read_parquet(new_path), pd.read_parquet(row["file"])
        assert list(f_new.columns) == list(f_old.columns), (list(f_new.columns), list(f_old.columns))
        r["columns"] = {}
        for col in f_new.columns:
            if col == ID:
                assert np.array_equal(f_new[col].to_numpy("float64"), ids)
                continue
            if f_new[col].dtype.kind == "f":
                r["columns"][col] = diff_stats(f_new[col].to_numpy(), f_old[col].to_numpy(), month)
            else:
                r["columns"][col] = {"equal": bool(f_new[col].equals(f_old[col]))}
        if spec["check"] == "sha256":
            ok = r["same_sha256"]
        elif spec["check"] == "read":  # producer tolerance "not measured" (GPU refits): the distance is the result
            r["verdict"] = "READ"
            rec["parts"][part] = r
            continue
        else:
            tol = spec["tolerance"]
            ok = all(c.get("rms") is None or (c["rms"] <= tol["rms"] and c["max_abs"] <= tol["max_abs"])
                     for c in r["columns"].values() if "rms" in c)
            r["tolerance"] = tol
        r["verdict"] = "PASS" if ok else "FAIL"
        if not ok:
            rec["verdict"] = "FAIL"
        rec["parts"][part] = r
    out = Path(a.rebuild) / "release73" / f"compare-{a.step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["verdict"] == "PASS" else 1)


def assemble(a):
    tgt = REG["targets"][a.target]
    if "steps" not in tgt:
        raise SystemExit(f"target {a.target}: {tgt.get('status')}")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to write into the existing folder {out}")
    guard = guard_snapshot()
    out.mkdir(parents=True)
    ids, month = template_ids(), month_of_template()
    parts = {}
    for s in tgt["steps"]:
        step = REG["steps"][s]
        for part in step["parts"]:
            parts[f"{step['producer']}/{part}"] = accepted_part(step["producer"], part)
            if parts[f"{step['producer']}/{part}"]["file"].endswith(".json"):  # a JSON part (the pack): text check
                refuse_forbidden(f"{step['producer']}/{part}", Path(parts[f"{step['producer']}/{part}"]["file"]).read_text())
    # The chain of value columns: the stages record counts the rows each step changes, by month.
    stages, prev = [], None
    for ref in tgt["chain"]:
        p, part, col = part_ref(ref)
        v = read_part(Path(parts[f"{p}/{part}"]["file"]), col, ids)
        changed = np.isfinite(v) if prev is None else ~((v == prev) | (np.isnan(v) & np.isnan(prev)))
        stages.append({"name": ref, "changed_by_month": {m: int((changed & (month == m)).sum()) for m in sorted(set(month))},
                       "against": "no value (first part)" if prev is None else tgt["chain"][len(stages) - 1]})
        prev = v
    for extra in tgt.get("stage_records", []):
        stages += json.loads(Path(extra).read_text())["stages"]
    p, part, col = part_ref(tgt["value"])
    values = read_part(Path(parts[f"{p}/{part}"]["file"]), col, ids)
    frame = pd.DataFrame({ID: ids, Y: values.astype("float64")})
    path = out / "youthful-giraffe_final.parquet"
    frame.to_parquet(path, index=False)
    (out / "stages.json").write_text(json.dumps({"stages": stages}, indent=1) + "\n")
    h8_reads = {"adsb_gap_2026_06_11": adsb_gap_read(ids, values)}
    (out / "h8-reads.json").write_text(json.dumps(h8_reads, indent=1) + "\n")
    refs = REG["refs"]
    assert sha256(Path(refs["path"])) == refs["sha256"], "v73-parts reference table SHA-256"
    jj = SNAP / REG["janjul_ref"]["path"]
    assert sha256(jj) == REG["janjul_ref"]["sha256"], "v73 upload SHA-256"
    runs = [{"part": "v73", "new": str(path), "col": Y, "kind": "refit"}]
    cross, cross_fail = {}, []
    for key, row in parts.items():
        p, part = key.split("/")
        spec = step_of(p, part)["parts"][part]
        if spec.get("parity"):
            # the checker reads every part as a refit read (numbers); a deterministic part also gets the
            # full-precision cross-machine check against the Mac-made reference table, which blocks
            runs.append({"part": spec["parity"], "new": row["file"], "col": spec["value"], "kind": "refit"})
            if spec.get("parity_kind") == "deterministic":
                cross[key] = cross_machine_check(row["file"], spec["value"], spec["parity"])
                if cross[key]["verdict"] != "PASS":
                    cross_fail.append(f"cross-machine:{key}")
    (out / "parity-spec.json").write_text(json.dumps({"refs": refs["path"], "out_dir": str(out), "runs": runs}, indent=1) + "\n")
    cmd = [sys.executable, str(SNAP / REG["checker"]), str(path), str(SNAP), "--janjul-ref", str(jj),
           "--stages", str(out / "stages.json"), "--parity", str(out / "parity-spec.json"), "--out", str(out / "check.json")]
    if tgt.get("setters"):
        cmd += ["--setters", tgt["setters"]]
    env = dict(os.environ, PRC_DATA_DIR=REG["data_dir"], PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    chk = subprocess.run(cmd, env=env, capture_output=True, text=True)
    check = json.loads((out / "check.json").read_text()) if (out / "check.json").exists() else {"error": chk.stderr[-3000:]}
    rec = {"target": a.target, "candidate": tgt.get("candidate", True), "built_utc": now(), "host": platform.node(),
           "snapshot": json.loads((SNAP / "SNAPSHOT.json").read_text()) if (SNAP / "SNAPSHOT.json").exists() else str(SNAP),
           "file": str(path), "file_sha256": sha256(path), "bytes": path.stat().st_size, "rows": len(frame),
           "parts": parts, "stages": stages, "checker_exit": chk.returncode,
           "recipe_guard": {"refused": False, "files_read": len(guard["files_read"]),
                            "pinned_files_named": guard["pinned_files_named"]},
           "checker_verdict": check.get("verdict"), "cross_machine_checks": cross, "h8_reads": h8_reads,
           "blocking_failures": (check.get("blocking_failures") or []) + cross_fail}
    if cross_fail:
        rec["checker_verdict"] = "FAIL"
    (out / "build.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("target", "file", "file_sha256", "checker_exit", "checker_verdict",
                                          "blocking_failures")}, indent=1))
    sys.exit(chk.returncode or (1 if cross_fail else 0))


def check_file(a):
    """`final-check`: the January/July rows of a built file, in submitting.parquet order, with its checks."""
    from prc import data as D
    src = Path(a.from_build)
    b = json.loads((src / "build.json").read_text())
    spec = REG["targets"]["final-check"]
    if b["target"] not in spec["sources"] + spec.get("dev_sources", []):
        raise SystemExit(f"final-check takes a build of {spec['sources']}, not {b['target']}")
    refuse_forbidden("source build record", json.dumps(b, default=str))
    f = Path(b["file"])
    assert sha256(f) == b["file_sha256"], "the built file changed after its build record"
    assert b["checker_verdict"] == "PASS", ("the source file did not pass its checks", b["blocking_failures"])
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to write into the existing folder {out}")
    out.mkdir(parents=True)
    sub = D.load_submitting(mode="ranking")
    full = pd.read_parquet(f)
    sid = sub[ID].to_numpy("float64")
    assert len(sub) == BOARD_ROWS and len(full) == ROWS, (len(sub), len(full))
    # PLAN.md (a): the first 344,841 rows of final_submitting.parquet are submitting.parquet, same IDs, same order
    assert np.array_equal(full[ID].to_numpy("float64")[:BOARD_ROWS], sid), "the January/July rows are not first"
    vals = full[Y].to_numpy("float64")[:BOARD_ROWS]
    path = out / a.name
    file_sha = D.write_submission(pd.Series(vals, index=sid), path, mode="ranking")
    c = pd.read_parquet(path)
    v = c[Y].to_numpy()
    rec = {"target": "final-check", "source_target": b["target"], "source_candidate": b.get("candidate"),
           "source_file": str(f), "source_sha256": b["file_sha256"], "file": str(path), "file_sha256": file_sha,
           "bytes": path.stat().st_size, "built_utc": now(), "host": platform.node(),
           "snapshot": json.loads((SNAP / "SNAPSHOT.json").read_text()) if (SNAP / "SNAPSHOT.json").exists() else str(SNAP)}
    fail = []
    rec["schema"] = {k: str(t) for k, t in c.dtypes.items()}
    if list(c.columns) != [ID, Y] or any(t != np.float64 for t in c.dtypes):
        fail.append("schema")
    rec["ids_same_order_as_submitting"] = bool(len(c) == BOARD_ROWS and c[ID].is_unique
                                               and np.array_equal(c[ID].to_numpy("float64"), sid))
    if not rec["ids_same_order_as_submitting"]:
        fail.append("ids")
    rec["nonfinite"] = int((~np.isfinite(v)).sum())
    rec["below_60"], rec["at_or_above_172800"] = int((v < 60).sum()), int((v >= 172800).sum())
    rec["min"], rec["max"] = float(v.min()), float(v.max())
    fail += [k for k, n in (("finite", rec["nonfinite"]), ("floor", rec["below_60"] + rec["at_or_above_172800"])) if n]
    rec["values_equal_source_rows"] = bool(np.array_equal(v, vals))
    if not rec["values_equal_source_rows"]:
        fail.append("values")
    jj = SNAP / REG["janjul_ref"]["path"]
    assert sha256(jj) == REG["janjul_ref"]["sha256"], "v73 upload SHA-256"
    r = pd.read_parquet(jj)
    assert np.array_equal(r[ID].to_numpy("float64"), sid)
    d = v - r[Y].to_numpy()
    rec["vs_v73"] = {"rows": BOARD_ROWS, "equal": int((d == 0).sum()), "rms_diff_s": float(np.sqrt(np.mean(d ** 2))),
                     "mean_sq_diff_s2": float(np.mean(d ** 2)), "bias_s": float(d.mean()),
                     "max_abs_diff_s": float(np.abs(d).max())}
    refs = REG["refs"]
    assert sha256(Path(refs["path"])) == refs["sha256"], "v73-parts reference table SHA-256"
    po = out / "parity-v73.json"
    env = dict(os.environ, PRC_DATA_DIR=REG["data_dir"], PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    pr = subprocess.run([sys.executable, str(SNAP / "experiments/exp-186-final-core/code/parity.py"), "--refs",
                         refs["path"], "--new", str(path), "--part", "v73", "--col", Y, "--kind", "refit", "--out",
                         str(po)], env=env, capture_output=True, text=True)
    rec["parity_v73"] = {"exit": pr.returncode, "record": str(po), "error": pr.stderr[-2000:] if pr.returncode else None}
    rec["blocking_failures"] = fail
    rec["verdict"] = "PASS" if not fail and pr.returncode == 0 else "FAIL"
    (out / "check-file.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps(rec, indent=1, default=str))
    sys.exit(0 if rec["verdict"] == "PASS" else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run-step")
    s.add_argument("--step", required=True)
    s.add_argument("--root", required=True)
    s = sub.add_parser("compare")
    s.add_argument("--step", required=True)
    s.add_argument("--rebuild", required=True)
    s = sub.add_parser("assemble")
    s.add_argument("--target", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("check-file")
    s.add_argument("--from-build", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run-step": run_step, "compare": compare, "assemble": assemble, "check-file": check_file}[a.cmd](a)


if __name__ == "__main__":
    main()
