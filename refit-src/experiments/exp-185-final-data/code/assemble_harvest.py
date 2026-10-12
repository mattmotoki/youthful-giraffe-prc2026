"""exp-185: assemble one exp-045 ten-box harvest kernel into this experiment's folder.

The logic is exp-045 `code/assemble_kernel.py` (the stream from
`kernels/surface/prc-harvest.py`, the two rule files at HEAD); the one change is the
output folder, `experiments/exp-185-final-data/kernels/surface-<name>/`, so that this
worker writes only in its own folder.

  .venv/bin/python experiments/exp-185-final-data/code/assemble_harvest.py <name> <days-json> [limit] [budget]
"""
import hashlib
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
EXP045 = ROOT / "experiments/exp-045-harvest"
sys.path.insert(0, str(EXP045 / "code"))
from rule_source import RULE, EXT, rule_sources, strip, version   # noqa: E402

name, days = sys.argv[1], json.loads(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
budget = int(sys.argv[4]) if len(sys.argv) > 4 else 39600
user = os.environ.get("PRC_KAGGLE_USER", "").strip()
assert user, "set PRC_KAGGLE_USER"
dst = ROOT / f"experiments/exp-185-final-data/kernels/surface-{name}"
dst.mkdir(parents=True, exist_ok=True)

base = (EXP045 / "kernels/surface/prc-harvest.py").read_text()
rule, ext = rule_sources(str(ROOT))
boxes = json.load(open(EXP045 / "analysis/boxes.json"))["boxes"]
boxes = {k: {"lat": v["lat"], "lon": v["lon"]} for k, v in boxes.items()}
rule_s, ext_s = strip(rule, []), strip(ext, ["from segments import"])
sha = hashlib.sha256((rule + ext).encode()).hexdigest()
ver = version(rule, ext)
inl = (f"\n# --- inlined verbatim from {RULE}"
       f" (sha256 {hashlib.sha256(rule.encode()).hexdigest()[:16]})\n{rule_s}"
       f"\n# --- inlined verbatim from {EXT}"
       f" (sha256 {hashlib.sha256(ext.encode()).hexdigest()[:16]})\n{ext_s}"
       f'\nRULE_SHA256 = "{sha}"\nRULE_VERSION = "{ver}"\n')
s, n0 = re.subn(r"^# <<< SEGMENT RULE >>>$", inl.replace("\\", "\\\\"), base, count=1, flags=re.M)
s, n1 = re.subn(r'"days": \[[^\]]*\],', '"days": ' + json.dumps(days) + ",", s, count=1)
s, n2 = re.subn(r'"boxes": \{[^\n]*\},\n', '"boxes": ' + json.dumps(boxes) + ",\n", s, count=1)
s, n3 = re.subn(r'"limit_members": \d+,', f'"limit_members": {limit},', s, count=1)
s, n4 = re.subn(r'"budget_s": \d+,', f'"budget_s": {budget},', s, count=1)
assert (n0, n1, n2, n3, n4) == (1, 1, 1, 1, 1), (n0, n1, n2, n3, n4)
compile(s, "prc-harvest.py", "exec")
(dst / "prc-harvest.py").write_text(s)
(dst / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/prc-harvest-{name}",
    "title": f"prc harvest {name}",
    "code_file": "prc-harvest.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}, indent=2) + "\n")
print(f"assembled {dst}/prc-harvest.py: {len(days)} days, {len(boxes)} boxes, rule {ver} {sha[:16]}")
