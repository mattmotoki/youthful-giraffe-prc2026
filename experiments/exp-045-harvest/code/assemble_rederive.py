"""exp-045-harvest: assemble the re-derive kernel and its metadata.

Was a here-document inside `push_rederive.sh` until the laptop volume filled on
2026-09-08 and bash could not spool one.  A file on disk needs no spool, and it
can be read and diffed like the rest of the sources.

  python3 code/assemble_rederive.py <root> <month> <src-kernel>
"""

import os
import hashlib
import json
import pathlib
import sys


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user


def run_kernel():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from rule_source import rule_sources, strip, version   # noqa: E402

    root, month, src = sys.argv[1:4]
    exp = f"{root}/experiments/exp-045-harvest"
    dst_dir = pathlib.Path(f"{exp}/kernels/rederive-{month}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    slug = f"prc-rederive-{month}"

    base = open(f"{exp}/kernels/rederive/prc-rederive.py").read()
    rule, ext = rule_sources(root)

    rule_s, ext_s = strip(rule, []), strip(ext, ["from segments import"])
    sha = hashlib.sha256((rule + ext).encode()).hexdigest()   # the SOURCE files, so the
    # same rule hashes the same whichever kernel inlines it
    ver = version(rule, ext)
    inl = f'\n{rule_s}\n{ext_s}\nRULE_SHA256 = "{sha}"\nRULE_VERSION = "{ver}"\n'
    import re                                                  # noqa: E402
    s, n = re.subn(r"^# <<< SEGMENT RULE >>>$", inl.replace("\\", "\\\\"), base, count=1, flags=re.M)
    assert n == 1, "the rule marker is not in the kernel"
    compile(s, "prc-rederive.py", "exec")                     # never push a kernel that will not parse
    (dst_dir / "prc-rederive.py").write_text(s)
    (dst_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{kaggle_user()}/{slug}",
        "title": f"prc rederive {month}",
        "code_file": "prc-rederive.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [f"{kaggle_user()}/{src}"],
        "competition_sources": [],
        "model_sources": [],
    }, indent=2) + "\n")
    print(f"assembled {dst_dir}/prc-rederive.py: rule {ver} sha256 {sha[:16]}")


if __name__ == "__main__":
    run_kernel()
