"""exp-045-harvest: assemble one ten-box fetch kernel and its metadata.

Was a pair of here-documents inside `push_kernel.sh` until the laptop volume filled on
2026-09-08 and bash could not spool one.  A file on disk needs no spool.

  python3 code/assemble_kernel.py <root> <name> <days-json> <limit> <budget>

The kernel is three sources joined at the `# <<< SEGMENT RULE >>>` marker: the stream
and day loop from `kernels/surface/prc-harvest.py`, the pilot's frozen `segments.py`,
and this experiment's `allsegments.py`.  The two rule files come from HEAD, never from
the working copy -- see code/rule_source.py.
"""
import hashlib
import json
import pathlib
import re
import os
import sys


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user


def run_kernel():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from rule_source import RULE, EXT, rule_sources, strip, version   # noqa: E402

    root, name, days, limit, budget = sys.argv[1:6]
    days_l = json.loads(days)
    int(limit), int(budget)
    exp = f"{root}/experiments/exp-045-harvest"
    dst_dir = pathlib.Path(f"{exp}/kernels/surface-{name}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    slug = f"prc-harvest-{name}"

    base = open(f"{exp}/kernels/surface/prc-harvest.py").read()
    rule, ext = rule_sources(root)
    boxes = json.load(open(f"{exp}/analysis/boxes.json"))["boxes"]
    boxes = {k: {"lat": v["lat"], "lon": v["lon"]} for k, v in boxes.items()}

    rule_s, ext_s = strip(rule, []), strip(ext, ["from segments import"])
    sha = hashlib.sha256((rule + ext).encode()).hexdigest()   # the SOURCE files, so the
    # same rule hashes the same whichever kernel inlines it
    ver = version(rule, ext)
    inl = (f"\n# --- inlined verbatim from {RULE}"
           f" (sha256 {hashlib.sha256(rule.encode()).hexdigest()[:16]})\n{rule_s}"
           f"\n# --- inlined verbatim from {EXT}"
           f" (sha256 {hashlib.sha256(ext.encode()).hexdigest()[:16]})\n{ext_s}"
           f'\nRULE_SHA256 = "{sha}"\nRULE_VERSION = "{ver}"\n')

    s, n0 = re.subn(r"^# <<< SEGMENT RULE >>>$", inl.replace("\\", "\\\\"), base, count=1, flags=re.M)
    s, n1 = re.subn(r'"days": \[[^\]]*\],', '"days": ' + json.dumps(days_l) + ",", s, count=1)
    s, n2 = re.subn(r'"boxes": \{[^\n]*\},\n', '"boxes": ' + json.dumps(boxes) + ",\n", s, count=1)
    s, n3 = re.subn(r'"limit_members": \d+,', f'"limit_members": {limit},', s, count=1)
    s, n4 = re.subn(r'"budget_s": \d+,', f'"budget_s": {budget},', s, count=1)
    assert (n0, n1, n2, n3, n4) == (1, 1, 1, 1, 1), (n0, n1, n2, n3, n4)
    compile(s, "prc-harvest.py", "exec")                      # never push a kernel that will not parse
    (dst_dir / "prc-harvest.py").write_text(s)
    (dst_dir / "kernel-metadata.json").write_text(json.dumps({
        # exp-064: the account was hard-coded here, so a reader's generated
        # metadata uses the caller account from PRC_KAGGLE_USER.
        "id": f"{kaggle_user()}/{slug}",
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
    print(f"assembled {dst_dir}/prc-harvest.py: {len(days_l)} days, {len(boxes)} boxes, "
          f"rule {ver} {sha[:16]}")


if __name__ == "__main__":
    run_kernel()
