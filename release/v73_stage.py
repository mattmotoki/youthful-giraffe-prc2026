"""The v73 stage (target v73): v64 plus the c5 steps, the L2C level 2 and the ORD-S column.

    python scripts/build_entry.py --target v73 --data-root DIR --out-dir NEW --record NEW.json

The code of the steps is in iso/ (EXT, LONG, LIRF, NOAC and ROLL: entry c5, upload v68) and w2/
(L2C on clock rows, upload v69; ORD-S with the s7 sequence member at 3 seeds, upload v73). These
folders are the code that built v73; this stage runs them in the order of w2/rebuild.sh:

1. Check the 14 official files (submitting, ranking and the twelve 2025 training months) by hash.
2. Build the v64 chain reference from saved values: the zabd comparison vector must give the
   e7b04ffd bytes, and the T11b-L stage on it the 9cd421dd bytes. iso/dump_chain.py replays the
   v64 chain and compares it with this reference when the platform does not give the v64 bytes
   (another BLAS changes the last bits of the level-2 weights).
3. Make a run folder under the output folder: copies of the registered iso/ and w2/ files, links
   to the release files (release/ZABD-PACK.json is a link to the pack of the v64 entry,
   release/ZABD-PACK-ranking.json) and to the 14 checked files (input-root/data/raw), and the s7 member values of
   release/saved-vectors/s7c/ written back to w2/inputs/s7c.npz as float32 arrays.
4. Run iso/dump_chain.py, iso/features.py (twice), iso/candidate.py (c5), w2/base.py, w2/l2c.py
   and w2/ords.py with the thread settings of iso/rebuild.sh and w2/rebuild.sh.

The steps refit from the official 2025 labels: the LONG member weights, the NOAC slope, the ROLL
logistic, the L2C LightGBM correction (3 seeds) and the ORD-S coefficients. No member fit runs.
The v73 bytes come from x86-64 Linux; on macOS arm64 the L2C trees differ, so the final bytes
differ there, and the facts give the difference to the saved v73 values.

Final mode (PRC_RANKING_MODE=final, set by scripts/build_entry.py from the entry's mode; target
final): the same steps on the final chain (zabd-final, v64-final) with the three run differences of
the final build (exp-191 REPORT section 7): the two final files are also checked and linked;
chain_ids.py runs after iso/dump_chain.py; ORD-S reads the member files work/w2/s7_{fold,ranking}.parquet
(--names s7), which this stage writes from the saved s7 values of the final build and checks against
that build's member files by SHA-256. The w2 steps get OMP_NUM_THREADS=2, as in the final build.

Entries final-ii and final-iii (the port runs of exp-191 with the 9-seed s7 member) use the same steps with their own
saved member values. final-iii also has candidate A's L2C, which reads the s7 member: the entry registers the patched
l2c.py under `code_sources` (the run folder gets it as w2/l2c.py), and `member.l2c_npz` names the npz that L2C reads.
After chain_ids this stage writes that npz with the float32 cast of the saved member values (the form of
w2/ords.py --save-npz) and checks each array by the SHA-256 of its float32 bytes; L2C then runs with an explicit --s7.

Entry final-iii-r9-ld (LD) is final-iii-r9 with another ORD-S: `code_sources` gives the run folder w2/ords_ld.py as
w2/ords.py, and `member.ords_flags` ["--long-days"] goes to its command line (the apply masks of ORD-S also take the
ordinary member-scope rows of long-taxi days; the fit keeps the other rows). ORD_S_FLAGS lists the flags this stage
accepts.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import assembly_replay as R
import t11b_stage as T

ROOT = R.ROOT
ID, Y = R.ID, R.Y
FLOOR = 60.0
#: (label, script and arguments, thread group), in the order of iso/rebuild.sh and w2/rebuild.sh
STEPS = (
    ("chain", ["iso/dump_chain.py", "--data-root", "{run}/input-root", "--out", "work/chain"], "iso"),
    ("features", ["iso/features.py"], "iso"),
    ("features-months", ["iso/features.py", "months"], "iso"),
    ("c5", ["iso/candidate.py", "--steps", "ext,long,lirf,noac,roll", "--out", "outputs/iso-c5.parquet"], "iso"),
    ("base", ["w2/base.py"], "w2"),
    ("l2c", ["w2/l2c.py", "--out", "outputs/w2-l2c.parquet"], "w2"),
    ("ords", ["w2/ords.py", "--on", "work/w2/l2c", "--npz", "w2/inputs/s7c.npz", "--lirf-jul", "0.5",
              "--out", "outputs/w2-l2c-s7c.parquet"], "w2"),
)
#: thread settings of the two rebuild scripts (w2/rebuild.sh leaves OMP_NUM_THREADS to the caller).
#: NT is the LightGBM thread count of L2C, and it changes the trees: NT=4 gives other bytes.
THREADS = {"iso": {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "NT": "8"},
           "w2": {"OPENBLAS_NUM_THREADS": "2", "NT": "8"}}
OUTPUTS = {"c5": "outputs/iso-c5.parquet", "w2-l2c": "outputs/w2-l2c.parquet",
           "v73": "outputs/w2-l2c-s7c.parquet"}
FINAL = R.RANKING_MODE == "final"
ORD_S_FLAGS = ("--long-days",)


def mode_steps(member: dict | None = None) -> tuple:
    """The steps of the mode: in final mode chain_ids.py after the chain dump and ORD-S on the member files; with
    `member.l2c_npz`, L2C reads the member from that npz; `member.ords_flags` go to ORD-S."""
    if not FINAL:
        return STEPS
    out = []
    for label, args, group in STEPS:
        if label == "ords":
            args = ["w2/ords.py", "--on", "work/w2/l2c", "--names", "s7", "--lirf-jul", "0.5",
                    "--save-npz", "work/w2/s7c-final.npz", "--out", "outputs/w2-l2c-s7c.parquet"]
            flags = (member or {}).get("ords_flags", [])
            if any(f not in ORD_S_FLAGS for f in flags) or len(set(flags)) != len(flags):
                raise SystemExit(f"ORD-S flags not accepted: {flags}")
            args += flags
        if label == "l2c" and member and member.get("l2c_npz"):
            args = ["w2/l2c.py", "--s7", member["l2c_npz"], "--out", "outputs/w2-l2c.parquet"]
        out.append((label, args, group))
        if label == "chain":
            out.append(("chain_ids", ["chain_ids.py", ".", "--official", "fold,ranking"], "iso"))
    return tuple(out)


def threads(group: str) -> dict:
    return {**THREADS[group], "OMP_NUM_THREADS": "2"} if FINAL else THREADS[group]


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def platform_key() -> str:
    return f"{platform.system()}-{platform.machine()}"


def versions() -> dict:
    from importlib.metadata import version
    return {name: version(name) for name in ("lightgbm", "scikit-learn", "scipy")}


def official_files(spec: dict, data_root: Path) -> dict:
    """The 14 official files, each checked by hash."""
    paths = {}
    for name, expected in spec["official_sha256"].items():
        path = Path(data_root) / "data/raw" / name
        if not path.is_file() or R.sha256_file(path) != expected:
            raise SystemExit(f"wrong or missing official input: data/raw/{name}")
        paths[name] = path
    return paths


def chain_reference(zabd_spec: dict, v64_spec: dict, data_root: Path, folder: Path) -> tuple[Path, dict]:
    """(ID, zabd, v64) of the registered entries, from the saved zabd vector and the T11b-L stage."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_v32 as legacy
    zabd = legacy.saved_array(zabd_spec["comparison_vector"], zabd_spec["rows"])
    ids = pd.read_parquet(Path(data_root) / R.OFFICIAL_FILES["submitting"])[ID].to_numpy("float64")
    table = pd.DataFrame({ID: ids, Y: zabd})
    buffer = io.BytesIO()
    table.to_parquet(buffer, index=False)
    if sha256_bytes(buffer.getvalue()) != zabd_spec["sha256"]:
        raise SystemExit("the saved zabd vector does not give the registered zabd bytes")
    folder.mkdir(parents=True, exist_ok=True)
    zabd_path = folder / "zabd-reference.parquet"
    zabd_path.write_bytes(buffer.getvalue())
    v64 = T.compose(v64_spec, zabd_path, data_root)
    if sha256_bytes(v64["body"]) != v64_spec["sha256"]:
        raise SystemExit("the T11b-L stage on the saved zabd vector does not give the registered v64 bytes")
    reference = pd.DataFrame({ID: ids, "zabd": zabd, "v64": v64["table"][Y].to_numpy("float64")})
    path = folder / "chain_ref_ranking.parquet"
    reference.to_parquet(path, index=False)
    return path, {"zabd_sha256": zabd_spec["sha256"], "v64_sha256": v64_spec["sha256"]}


def member_files(member: dict, run: Path) -> dict:
    """Final mode: the member files work/w2/<name>_{fold,ranking}.parquet in the form of w2/seq/export.py (the chain
    IDs of chain_ids.py and the seed mean as float64) from the saved values, each checked by SHA-256 against the
    member file of the final build."""
    facts = {}
    for split in ("fold", "ranking"):
        rec = member[split]
        values = R.decode_array(ROOT / rec["public_vector"], rec["public_vector_sha256"], rec["rows"])
        chain = pd.read_parquet(run / f"work/w2/chain_ids_{split}.parquet")[ID].to_numpy()
        if len(chain) != rec["rows"]:
            raise SystemExit(f"{split} chain IDs: {len(chain)} rows, the saved member values {rec['rows']}")
        path = run / f"work/w2/{member['name']}_{split}.parquet"
        if path.exists():
            raise SystemExit(f"refusing to replace {path}")
        pd.DataFrame({ID: chain, member["name"]: values}).to_parquet(path, index=False)
        if R.sha256_file(path) != rec["file_sha256"]:
            raise SystemExit(f"{split} member file differs from the member file of the final build")
        facts[split] = {"rows": int(len(values)), "finite_rows": int(np.isfinite(values).sum()),
                        "file_sha256": rec["file_sha256"]}
    return facts


def member_cast_npz(member: dict, path: Path) -> dict:
    """Final mode, an L2C that reads the member (`member.l2c_npz`): the float32 cast of the saved member values
    (float64 seed means, in the chain order of the member files) as the arrays fold and ranking, as w2/ords.py
    --save-npz writes them. Each array is checked by the SHA-256 of its float32 bytes (equal to the arrays of the
    final build's npz); the file bytes can differ from that npz (np.savez here, np.savez_compressed there)."""
    if path.exists():
        raise SystemExit(f"refusing to replace {path}")
    arrays, facts = {}, {}
    for split in ("fold", "ranking"):
        rec = member[split]
        narrow = R.decode_array(ROOT / rec["public_vector"], rec["public_vector_sha256"], rec["rows"]).astype("<f4")
        if sha256_bytes(narrow.tobytes()) != rec["float32_sha256"]:
            raise SystemExit(f"the float32 {split} member values differ from the arrays of the final build")
        arrays[split] = narrow
        facts[split] = {"rows": int(len(narrow)), "float32_sha256": rec["float32_sha256"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return facts


def final_member(member: dict, run: Path) -> dict:
    """Final mode, after chain_ids: the member files, and the L2C npz when the entry names one."""
    facts = member_files(member, run)
    if member.get("l2c_npz"):
        facts["l2c_npz"] = member_cast_npz(member, run / member["l2c_npz"])
    return facts


def member_npz(member: dict, path: Path) -> dict:
    """Write the saved s7 member values back as the float32 arrays of w2/inputs/s7c.npz."""
    arrays = {}
    for split in ("fold", "ranking"):
        rec = member[split]
        values = R.decode_array(ROOT / rec["public_vector"], rec["public_vector_sha256"], rec["rows"])
        narrow = values.astype("<f4")
        if not np.array_equal(narrow.astype("<f8"), values, equal_nan=True):
            raise SystemExit(f"saved {split} member values are not float32 values")
        if sha256_bytes(narrow.tobytes()) != rec["float32_sha256"]:
            raise SystemExit(f"saved {split} member values differ from the registered member file")
        arrays[split] = narrow
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {split: {"rows": int(len(a)), "finite_rows": int(np.isfinite(a).sum()),
                    "float32_sha256": member[split]["float32_sha256"]} for split, a in arrays.items()}


def run_folder(spec: dict, files: dict, run: Path, pack: str) -> None:
    """Copies of the registered step code, links to the release files and the checked official files.

    iso/chain.py and iso/dump_chain.py read the assembly pack as release/ZABD-PACK.json, so in the run folder that
    name is a link to the pack of this entry's chain (`pack`, a path under the repo root)."""
    if run.exists():
        raise SystemExit(f"refusing to reuse the run folder {run}")
    sources = spec.get("code_sources", {})  # run path -> release path of a registered variant (final-iii's L2C)
    for rel, expected in spec["code"].items():
        source = ROOT / sources.get(rel, rel)
        if not source.is_file() or R.sha256_file(source) != expected:
            raise SystemExit(f"step code differs from the registered file: {rel}")
        target = run / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    release = run / "release"
    release.mkdir()
    for item in sorted((ROOT / "release").iterdir()):
        if item.name not in ("ZABD-PACK.json", "__pycache__"):
            (release / item.name).symlink_to(item, target_is_directory=item.is_dir())
    (release / "ZABD-PACK.json").symlink_to(ROOT / pack)
    raw = run / "input-root/data/raw"
    raw.mkdir(parents=True)
    for name, path in files.items():
        (raw / name).symlink_to(path.resolve())


def run_steps(run: Path, reference: Path, after: dict | None = None, member: dict | None = None) -> list[dict]:
    """Run the steps of the mode; `after` maps a step label to a function of the run folder called after it."""
    done = []
    (run / "logs").mkdir()
    for label, args, group in mode_steps(member):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CHAIN_REF=str(reference), **threads(group))
        argv = [sys.executable, *[a.format(run=run) for a in args]]
        with (run / "logs" / f"{label}.log").open("w") as log:
            proc = subprocess.run(argv, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
        done.append({"step": label, "argv": [Path(argv[0]).name, *argv[1:]], "exit_code": proc.returncode,
                     "threads": {k: env.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NT")}})
        if proc.returncode != 0:
            raise SystemExit(f"step {label} failed; see {run / 'logs' / (label + '.log')}")
        if after and label in after:
            done[-1]["after"] = after[label](run)
    return done


def chain_facts(run: Path, reference: Path, spec_zabd: dict, spec_v64: dict) -> dict:
    """Did the replayed v64 chain give the registered bytes, and how far is it from them?"""
    chain = pd.read_parquet(run / "work/chain/ranking.parquet", columns=[ID, "zabd", "v64"])
    ref = pd.read_parquet(reference)
    if not np.array_equal(chain[ID].to_numpy("float64"), ref[ID].to_numpy("float64")):
        raise SystemExit("the replayed chain is not in the template order")
    facts = {}
    for name, spec in (("zabd", spec_zabd), ("v64", spec_v64)):
        buffer = io.BytesIO()
        pd.DataFrame({ID: chain[ID].to_numpy("float64"), Y: chain[name].to_numpy("float64")}).to_parquet(buffer, index=False)
        facts[name] = {"bytes_equal_registered": sha256_bytes(buffer.getvalue()) == spec["sha256"],
                       "max_abs_difference_s": float(np.max(np.abs(chain[name].to_numpy(float) - ref[name].to_numpy(float))))}
    return facts


def compose(spec: dict, book: dict, data_root: Path, out_dir: Path) -> dict:
    """Run the v73 steps; return the bytes, the table and the facts."""
    if FINAL != (spec.get("mode") == "final"):
        raise SystemExit(f"entry mode {spec.get('mode', 'ranking')} and PRC_RANKING_MODE={R.RANKING_MODE} differ")
    files = official_files(spec, data_root)
    entries = book["entries"]
    if entries[spec["base"]]["sha256"] != spec["base_sha256"]:
        raise SystemExit("v73 base is not the registered v64 entry")
    base = entries[spec["base"]]
    zabd = entries[base["base"]]  # the chain: zabd (or zabd-final) -> T11b-L -> v64 (or v64-final)
    if R.sha256_file(ROOT / base["pack"]) != base["pack_sha256"]:
        raise SystemExit(f"assembly pack hash mismatch: {base['pack']}")
    run = Path(out_dir) / "v73-run"
    run_folder(spec, files, run, base["pack"])
    reference, ref_facts = chain_reference(zabd, base, data_root, run / "work")
    if FINAL:
        member, after = None, {"chain_ids": lambda r: final_member(spec["member"], r)}
    else:
        member, after = member_npz(spec["member"], run / "w2/inputs/s7c.npz"), None
    steps = run_steps(run, reference, after, spec["member"])
    if FINAL:
        member = next(s["after"] for s in steps if s["step"] == "chain_ids")
    chain = chain_facts(run, reference, zabd, base)
    key = platform_key()
    hashes = {name: R.sha256_file(run / rel) for name, rel in OUTPUTS.items()}
    intermediate = {}
    for name, by_platform in spec["intermediate_sha256"].items():
        expected = by_platform.get(key)
        intermediate[name] = {"sha256": hashes[name], "expected_on_this_platform": expected,
                              "equal": None if expected is None else hashes[name] == expected}
    body = (run / OUTPUTS["v73"]).read_bytes()
    table = pd.read_parquet(io.BytesIO(body))
    template = pd.read_parquet(files["final_submitting.parquet" if FINAL else "submitting.parquet"])
    values = table[Y].to_numpy("float64")
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_v32 as legacy
    saved = legacy.saved_array(spec["comparison_vector"], spec["rows"])
    v64 = pd.read_parquet(run / "work/chain/ranking.parquet", columns=["v64"])["v64"].to_numpy("float64")
    checks = {
        "template_order": bool(np.array_equal(table[ID].to_numpy("float64"), template[ID].to_numpy("float64"))),
        "finite": bool(np.isfinite(values).all()),
        "min_60": bool(values.min() >= FLOOR),
        "rows": len(table) == spec["rows"],
        "chain_within_1e-6_s": all(c["max_abs_difference_s"] < 1e-6 for c in chain.values()),
    }
    if not all(checks.values()):
        raise SystemExit(f"stage checks failed: {sorted(k for k, ok in checks.items() if not ok)}")
    shift = values - v64
    difference = values - saved
    facts = {"platform": key, "reference_platform": spec["reference_platform"],
             "official_files": len(files), "chain_reference": ref_facts, "chain": chain,
             "member": member, "steps": steps, "intermediate": intermediate, "checks": checks,
             "rows_changed_against_v64": int((shift != 0).sum()),
             "rms_change_against_v64_s": float(np.sqrt(np.mean(shift ** 2))),
             "max_abs_difference_to_saved_v73_s": float(np.max(np.abs(difference))),
             "rms_difference_to_saved_v73_s": float(np.sqrt(np.mean(difference ** 2))),
             "fitted": ["LONG member weights", "NOAC slope", "ROLL logistic",
                        f"L2C LightGBM ({spec.get('l2c_seeds', 3)} seeds)", "ORD-S coefficients"],
             "member_fit_ran": False, "versions": versions(),
             "run_folder": str(run)}
    return {"body": body, "table": table, "facts": facts}
