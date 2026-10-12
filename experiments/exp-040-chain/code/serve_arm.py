"""The ranking side of one arm: v10's whole recipe with `L2-hub8725` swapped.

    .venv/bin/python experiments/exp-040-chain/code/serve_arm.py --arm published
    .venv/bin/python experiments/exp-040-chain/code/serve_arm.py --arm chain

**Nothing is reimplemented.**  The head is exp-027's `serve.py` run unedited on
the `k2-refit-bulk-eham-out` arm (the mixed anchor, the 600 s `depx` stacker, the
`noac` rows taken from `exp-026-stack-5.parquet`, v7's EHAM mask), followed by
exp-036's eight-airport mask as arithmetic over two files.  This file changes
exactly two strings before that runs -- `frames.OOF["L2-hub8725"]` and
`frames.RANKING["L2-hub8725"]` -- which is the same swap `code/q09.py` makes on
the fold side, and redirects the intermediate file into this experiment so no
arm can be mistaken for a submission candidate.

`--arm published` is the serving-path gate: the shipped member through this exact
path must give back v10's bytes, SHA-256 `ad962763...`.  No arm's head may be read
until it does, because a head built from a path that cannot reproduce v10 is not
a measurement of the member change.

Writes `candidates/head-<arm>.parquet` and `analysis/head-<arm>.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
E27 = ROOT / "experiments/exp-027-step"
sys.path.insert(0, str(E27 / "code"))
sys.path.insert(0, str(ROOT / "experiments/exp-028-window"))
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import frames as F                                            # noqa: E402
import regime_table as RT                                     # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                          # noqa: E402

MEMBER = "L2-hub8725"
#: exp-027's own serve script, imported as a module so its `main()` is the one
#: that runs.  It is a script with `argparse`, so `sys.argv` is how it is called.
sys.path.insert(0, str(E27))
import serve as S27                                           # noqa: E402

#: v10 = `np.where(airport in KEEP, refit-bulk, v7)`, exp-036 `analysis/
#: arm-refitbulk-mask.json`.  Read from that record rather than retyped.
MASK_REC = ROOT / "experiments/exp-036-mask/analysis/arm-refitbulk-mask.json"
V7 = ROOT / "submissions/candidates/exp-030-eham-out.parquet"
V7_SHA = "67b70ee379e62a2a3bc9d91c9d771112baf3233efa3f74f4f95adbd7a3c03ef7"
V10 = ROOT / "submissions/uploaded/youthful-giraffe_v10.parquet"
V10_SHA = "ad962763236e9b40ace45a5dac12945568bbce441ed0547ff8e814d248c72c78"
ARM_OOF = "experiments/exp-040-chain/oof/{arm}--{fold}.parquet"
ARM_RANK = "experiments/exp-040-chain/oof/{arm}--ranking.parquet"
PUBLISHED_OOF, PUBLISHED_RANK = F.OOF[MEMBER], F.RANKING[MEMBER]


def column(path: Path, want_sha: str | None = None) -> np.ndarray:
    """One submission file's prediction column, in the template's order."""
    got = D.sha256(path)
    if want_sha and got != want_sha:
        raise SystemExit(f"{path.name} has changed: {got[:8]} against {want_sha[:8]}")
    d = pd.read_parquet(path)
    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    if not np.array_equal(d["MVT_ID_mvt"].to_numpy("float64"), tmpl):
        raise SystemExit(f"{path.name} is not in the template's order")
    return d[D.TARGET].to_numpy("float64")


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["published", "none", "chain", "derived"])
    a = ap.parse_args()

    if a.arm != "published":
        oof = ARM_OOF.replace("{arm}", a.arm)
        rank = ARM_RANK.replace("{arm}", a.arm)
        for p in (oof.format(fold="july_loo"), rank):
            if not (ROOT / p).exists():
                raise SystemExit(f"{p} is absent; the arm's Kaggle session has not landed")
        F.OOF[MEMBER], F.RANKING[MEMBER] = oof, rank
    print(f"[serve_arm] {MEMBER} <- {F.OOF[MEMBER]} / {F.RANKING[MEMBER]}", flush=True)

    # ---- 1. exp-027's refit-bulk head, unedited, into this experiment -------
    S27.DEST = EXP / "candidates"
    S27.DEST.mkdir(parents=True, exist_ok=True)
    name = f"refitbulk-{a.arm}"
    argv = sys.argv
    sys.argv = ["serve.py", "--arm", "k2-refit-bulk-eham-out", "--name", name,
                "--base", str(ROOT / "submissions/candidates/exp-026-stack-5.parquet")]
    try:
        S27.main()
    finally:
        sys.argv = argv
        F.OOF[MEMBER], F.RANKING[MEMBER] = PUBLISHED_OOF, PUBLISHED_RANK
    rb = EXP / "candidates" / f"{name}.parquet"
    rb_sha = D.sha256(rb)

    # ---- 2. exp-036's eight-airport mask, arithmetic over two files ---------
    keep = json.loads(MASK_REC.read_text())["keep"]
    gates = RT.load() if RT.OUT.exists() else RT.build()
    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    apt = gates.set_index("MVT_ID_mvt")["airport"].astype(str).reindex(tmpl).to_numpy()
    if pd.isna(apt).any():
        raise SystemExit("a ranking row has no airport in the gates table")
    on = np.isin(apt, keep)
    head = np.where(on, column(rb), column(V7, V7_SHA))
    out = EXP / "candidates" / f"head-{a.arm}.parquet"
    sha = D.write_submission(pd.Series(head, index=tmpl), out)

    # ---- 3. the gate ---------------------------------------------------------
    v10 = column(V10, V10_SHA)
    dif = np.abs(head - v10)
    rec = {"arm": a.arm, "member_oof": F.OOF[MEMBER] if a.arm == "published" else oof,
           "member_ranking": PUBLISHED_RANK if a.arm == "published" else rank,
           "refitbulk": str(rb.relative_to(ROOT)), "refitbulk_sha256": rb_sha,
           "mask_keep": keep, "rows_masked_in": int(on.sum()),
           "head": str(out.relative_to(ROOT)), "sha256": sha,
           "rows_moved_vs_v10": int((dif > 0).sum()),
           "max_abs_vs_v10": float(dif.max()),
           "rms_vs_v10": float(np.sqrt((dif ** 2).mean())),
           "min": float(head.min()), "finite": bool(np.isfinite(head).all()),
           "wall_s": round(time.time() - t0, 1)}
    if a.arm == "published":
        rec["gate_v10_sha256"] = V10_SHA
        rec["gate_pass"] = bool(sha == V10_SHA)
        print(f"[gate] published head sha {sha[:8]}… against v10 {V10_SHA[:8]}…  "
              f"max abs {dif.max():.10f}  {'PASS' if rec['gate_pass'] else 'FAIL'}",
              flush=True)
    (EXP / "analysis" / f"head-{a.arm}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2), flush=True)
    if a.arm == "published" and not rec["gate_pass"]:
        raise SystemExit("the serving path does not reproduce v10; no arm head is valid")


if __name__ == "__main__":
    main()
