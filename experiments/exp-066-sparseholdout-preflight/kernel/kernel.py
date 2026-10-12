"""Offline Kaggle transport probe: strict reconstruction, then synthetic imports."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "PRC_THREADS"):
    os.environ[name] = "3"

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
CONFIG_SHA256 = "da93b2819428938c56a3916401cbf4bb46004fb5aef505f5162e0e9829b3a31b"
TRUSTED = {
 "archive_sha256": "ed4d97d89926d94f4485b65c702447fbbe49ba9fc201dfcdd3b2eae2642774ac",
 "archive_bytes": 296960,
 "internal_manifest_sha256": "d902d154a3fcabdf5018eca61ea7c881d6be879c28db8abaf6df3b51b05ef7fb",
 "manifest_entries": 27, "archive_members": 28,
 "chunks": [
  {"name":"payload.part000","index":0,"bytes":262144,"sha256":"a836882864aa2b05560798fc286937dbf728ce8fcd0af044788f7b2bafc409cd"},
  {"name":"payload.part001","index":1,"bytes":34816,"sha256":"a9dc0bd2f21e5974c0074e7678ea20d276ccab37f6fd7da9066f58eb1a60d8be"},
 ]}
configs = list(INPUT.rglob("transport.json"))
if len(configs) != 1: raise SystemExit("expected exactly one transport.json")
config_bytes = configs[0].read_bytes()
if hashlib.sha256(config_bytes).hexdigest() != CONFIG_SHA256: raise SystemExit("untrusted transport config")
cfg = json.loads(config_bytes)
for key, value in TRUSTED.items():
    if cfg.get(key) != value: raise SystemExit(f"untrusted transport value: {key}")
root = configs[0].parent
actual = sorted(p.name for p in root.iterdir() if p.name.startswith("payload.part"))
expected = [x["name"] for x in cfg["chunks"]]
if actual != expected: raise SystemExit(f"missing, extra, or reordered chunks: {actual} != {expected}")
blob = bytearray()
for item in cfg["chunks"]:
    data = (root/item["name"]).read_bytes()
    if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise SystemExit("changed chunk")
    blob.extend(data)
if len(blob) != cfg["archive_bytes"] or hashlib.sha256(blob).hexdigest() != cfg["archive_sha256"]:
    raise SystemExit("reconstructed archive mismatch")
archive = Path("/tmp/verified.archive"); archive.write_bytes(blob)
dest = Path("/tmp/verified-source")
if dest.exists(): shutil.rmtree(dest)
dest.mkdir()
with tarfile.open(archive) as tf:
    members = tf.getmembers(); names = [member.name for member in members]
    if len(names) != len(set(names)): raise SystemExit("duplicate archive member")
    if len(names) != cfg["archive_members"] or len(names) != cfg["manifest_entries"] + 1:
        raise SystemExit("archive member count mismatch")
    if any(Path(n).is_absolute() or ".." in Path(n).parts for n in names): raise SystemExit("unsafe member")
    if any(not (member.isfile() or member.isdir()) for member in members):
        raise SystemExit("link, device, or special archive member")
    for member in members:
        target = dest/member.name
        if member.isdir(): target.mkdir(parents=True, exist_ok=True); continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = tf.extractfile(member)
        if source is None: raise SystemExit("regular member has no bytes")
        with target.open("xb") as output: shutil.copyfileobj(source, output)
manifest_bytes = (dest/"SOURCE_MANIFEST.json").read_bytes()
if hashlib.sha256(manifest_bytes).hexdigest() != cfg["internal_manifest_sha256"]: raise SystemExit("manifest mismatch")
manifest = json.loads(manifest_bytes)
if set(names) != set(manifest["members"]) | {"SOURCE_MANIFEST.json"}: raise SystemExit("manifest coverage mismatch")
for name, rec in manifest["members"].items():
    data = (dest/name).read_bytes()
    if len(data) != rec["bytes"] or hashlib.sha256(data).hexdigest() != rec["sha256"]: raise SystemExit("member mismatch")
env = dict(os.environ); env["PYTHONPATH"] = str(dest)
checks = {
 "F1-fresh": "import sys; sys.path.insert(0,'experiments/exp-018-fresh/code'); import run,dataset,model,features; print('PASS')",
 "L2-hub8725": "import sys; sys.path.insert(0,'experiments/exp-008-diversity'); import run,frames,learners; assert 'exp-001-retro-v1/features.py' in frames.build_features.__code__.co_filename; print('PASS')",
 "T-tailv4": "import sys,inspect; sys.path.insert(0,'experiments/exp-009-noac2'); import deliver; from prc.tail_model import PhiModel; assert PhiModel is not None; assert inspect.signature(deliver.TailModel.fit).parameters['use_phi'].default is True; print('PASS')",
 "cold-controls": "import sys; sys.path.insert(0,'experiments/exp-066-sparseholdout-preflight/code'); import cold_controls as C; assert (C.sham_table([1,2,3,4],[0,0,1,1],[1,0,3,2]) == [2,1,4,3]).all(); print('PASS')",
 "fixed-controller": "import sys; sys.path.insert(0,'experiments/exp-066-sparseholdout-preflight/code'); import stage_controller as C; assert C.STAGES == ('produce-a','freeze-features','freeze-null','read-labels'); print('PASS')",
}
facts = {}
for name, code in checks.items():
    run = subprocess.run([sys.executable, "-c", code], cwd=dest, env=env, capture_output=True, text=True)
    facts[name] = {"pass": run.returncode == 0 and run.stdout.strip() == "PASS",
                   "stdout_sha256": hashlib.sha256(run.stdout.encode()).hexdigest(),
                   "stderr_sha256": hashlib.sha256(run.stderr.encode()).hexdigest()}
    if not facts[name]["pass"]: raise SystemExit(f"synthetic import failed: {name}")
facts["synthetic_controls"] = facts["cold-controls"]["pass"]
facts["archive_sha256"] = hashlib.sha256(blob).hexdigest()
facts["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
facts["pass"] = all(x["pass"] for x in (facts[k] for k in checks)) and facts["archive_sha256"] == cfg["archive_sha256"]
(WORK/"transport-result.json").write_text(json.dumps(facts, sort_keys=True, indent=2) + "\n")
print(json.dumps({"archive_sha256": facts["archive_sha256"], "manifest_sha256": facts["manifest_sha256"], "pass": facts["pass"]}, sort_keys=True))
