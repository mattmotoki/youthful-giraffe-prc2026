#!/usr/bin/env python3
"""Write the three private, one-arm Kaggle kernel packages."""
from __future__ import annotations
import os

import json
from pathlib import Path



def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle/arm-kernels"
INPUT = Path(os.environ.get("PRC_EXP136_OUT", ROOT.parents[1] / "artifacts/rebuild/exp136"))
ARMS = ("control", "proxy", "sham")
def slugs():
    return {arm: f"{kaggle_user()}/prc-exp137-with-clock-{arm}-arm" for arm in ARMS}

def dataset_sources():
    return [
    f"{kaggle_user()}/prc-exp130-clean-input",
    f"{kaggle_user()}/prc-exp136-input",
    f"{kaggle_user()}/prc-exp137-input",
]



def kernel_code(arm: str) -> str:
    return f'''#!/usr/bin/env python3
"""Kaggle entry point for exp-137's private {arm} arm."""
from __future__ import annotations

import hashlib
import os
import runpy
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
matches = sorted(path for path in INPUT.rglob("fit_exp137_arm.py") if path.is_file())
if len(matches) != 1:
    roots = sorted(path.name for path in INPUT.iterdir()) if INPUT.exists() else []
    raise RuntimeError(f"expected one exp-137 runner, found {{len(matches)}}; mounts={{roots}}")
source = matches[0]
sys.path.insert(0, str(source.parent))
kernel_code = Path(__file__).resolve()
os.environ["PRC_KERNEL_SLUG"] = "{slugs()[arm]}"
os.environ["PRC_KERNEL_VERSION"] = "2"
os.environ["PRC_EXP130_DATASET_VERSION"] = "1"
os.environ["PRC_EXP136_DATASET_VERSION"] = "4"
os.environ["PRC_EXP137_DATASET_VERSION"] = "3"
os.environ["PRC_EXP137_ARM"] = "{arm}"
os.environ["PRC_KERNEL_CODE_SHA256"] = hashlib.sha256(kernel_code.read_bytes()).hexdigest()
runpy.run_path(str(source), run_name="__main__")
'''


def main() -> None:
    inventory_path = INPUT / "source_inventory.json"
    if not inventory_path.is_file():
        raise SystemExit(f"missing exp-136 inventory: {inventory_path}")
    sources = json.loads(inventory_path.read_text()).get("point_source_kernels")
    if not sources or len(sources) != 17:
        raise SystemExit("exp-136 inventory does not name all 17 point source kernels")
    for arm in ARMS:
        folder = OUT / arm
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "kernel.py").write_text(kernel_code(arm))
        metadata = {
            "id": slugs()[arm],
            "title": f"PRC exp137 with-clock {arm} arm",
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": dataset_sources(),
            "kernel_sources": sources,
            "competition_sources": [],
            "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"arms": list(ARMS), "slugs": slugs(),
                      "dataset_sources": dataset_sources(),
                      "point_kernel_sources": len(sources)}, indent=2))


if __name__ == "__main__":
    main()
