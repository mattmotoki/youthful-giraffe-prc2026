"""Write one exp-035 Kaggle kernel directory from the shared source.

Every arm of this experiment is the same kernel with a different list of
`run.py` (or `submit.py`) invocations, so the kernel body is written once here
and the arm supplies only its steps. The committed directory is exactly what
ran: `push_kernel.sh` pushes it unedited.

    python make_kernel.py <arm>          # arms are the keys of ARMS below
"""
from __future__ import annotations

import os

import json
import sys
from pathlib import Path


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SRC = HERE / "kernel_body.py"

#: fold recipe, frozen: exp-027 kernels/f1-window, which is exp-025 f1-real plus
#: PRC_WINDOW=1. Nothing here may drift from it except the named arm variable.
FOLD = dict(rounds=1200, clip_z=6000, exact_rounds=300, ghost_mode="mixbase",
            main_model="both", threads=4)
SHIPPED = ""                       # PRC_WINDOW_SCALES unset == (600, 3600)
WIDE = "300,600,1800,3600"


def fold_step(fold: str, seed_start: int, scales: str, tag: str, extras: str = "") -> dict:
    return dict(kind="fold", fold=fold, tag=tag, seed_start=seed_start, scales=scales,
                extras=extras, **FOLD)


def rank_step(seed_start: int, seeds: int, scales: str, tag: str, extras: str = "") -> dict:
    return dict(kind="rank", tag=tag, seed_start=seed_start, seeds=seeds, scales=scales,
                extras=extras,
                rounds=1200, clip_z=6000, exact_rounds=300, ghost_mode="mixbase",
                main_model="both", threads=4)


ARMS = {
    # --- arm 1, seeds: two more seeds of the shipped K2 fold recipe -----------
    "seed1-july": [fold_step("july_loo", 1, SHIPPED, "k2s1-july_loo")],
    "seed2-july": [fold_step("july_loo", 2, SHIPPED, "k2s2-july_loo")],
    "seed1-dj": [fold_step("december", 1, SHIPPED, "k2s1-december"),
                 fold_step("january", 1, SHIPPED, "k2s1-january")],
    "seed2-dj": [fold_step("december", 2, SHIPPED, "k2s2-december"),
                 fold_step("january", 2, SHIPPED, "k2s2-january")],
    "seed2-rank": [rank_step(2, 1, SHIPPED, "k2s2-rank")],
    # --- arm 2, widths: the shipped block plus a 300 s and an 1800 s scale ----
    "wide-july": [fold_step("july_loo", 0, WIDE, "k2w-july_loo")],
    "wide-dj": [fold_step("december", 0, WIDE, "k2w-december"),
                fold_step("january", 0, WIDE, "k2w-january")],
    "wide-rank": [rank_step(0, 2, WIDE, "k2w-rank")],
    # --- arm 3, exp-018b's module axis on the K2 frame -----------------------
    # Two gates, one round: `phi` alone is the module exp-018b expected to be
    # the largest, and all three together says whether the axis carries
    # anything `phi` does not. `deice` is january-only by its own record, so a
    # july_loo gate cannot see it and it rides along rather than being gated.
    "phi-july": [fold_step("july_loo", 0, SHIPPED, "k2p-july_loo", extras="phi")],
    "mod-july": [fold_step("july_loo", 0, SHIPPED, "k2m-july_loo",
                           extras="phi,surface,deice")],
    "phi-dj": [fold_step("december", 0, SHIPPED, "k2p-december", extras="phi"),
               fold_step("january", 0, SHIPPED, "k2p-january", extras="phi")],
    "phi-rank": [rank_step(0, 2, SHIPPED, "k2p-rank", extras="phi")],
    #: `mod-july` beat `phi-july` by +0.31 s and on the ORDINARY rows, so the
    #: axis carries more than `phi`; these complete it on the other two folds
    #: and on the ranking side.
    "mod-dj": [fold_step("december", 0, SHIPPED, "k2m-december", extras="phi,surface,deice"),
               fold_step("january", 0, SHIPPED, "k2m-january", extras="phi,surface,deice")],
    "mod-rank": [rank_step(0, 2, SHIPPED, "k2m-rank", extras="phi,surface,deice")],
    #: `mod-rank` was killed at 28.99 GB of 31 on its second seed, after the
    #: first had finished. One seed per session is the same two-seed mean K2
    #: ships, at a fold-sized peak; `code/land_rank.py` averages them.
    "mod-rank1": [rank_step(0, 1, SHIPPED, "k2m-rank1", extras="phi,surface,deice")],
    "mod-rank2": [rank_step(1, 1, SHIPPED, "k2m-rank2", extras="phi,surface,deice")],
}

SLUG = {a: f"prc-k2c-{a}" for a in ARMS}


def main() -> None:
    arm = sys.argv[1]
    steps = ARMS[arm]
    d = EXP / "kernels" / arm
    d.mkdir(parents=True, exist_ok=True)
    body = SRC.read_text()
    marker = "STEPS = []  # written by make_kernel.py\n"
    assert body.count(marker) == 1, "kernel_body.py lost its STEPS marker"
    body = body.replace(marker, f"ARM = {arm!r}\nSTEPS = {json.dumps(steps, indent=4)}\n")
    (d / f"{SLUG[arm]}.py").write_text(body)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{kaggle_user()}/{SLUG[arm]}",
        "title": f"prc k2c {arm}",
        "code_file": f"{SLUG[arm]}.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": [],
        "dataset_sources": [f"{kaggle_user()}/source-code", f"{kaggle_user()}/{os.environ['PRC_KAGGLE_DATA_SLUG']}"],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }, indent=1) + "\n")
    print(f"wrote {d.relative_to(EXP.parents[1])}  ({len(steps)} step(s))")


if __name__ == "__main__":
    main()
