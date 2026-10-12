"""Serve the climbed K2 candidate. One command.

    .venv/bin/python experiments/exp-035-k2climb/serve.py --arm-oof seeds
    .venv/bin/python experiments/exp-035-k2climb/serve.py --arm-oof seeds --identity

Nothing is reimplemented: this is `experiments/exp-027-step/serve.py`, called
with four module constants repointed at this experiment --

  * `PRC_K2`            the climbed member's fold files, so exp-027's
                        `candidate.py` (which `serve.py` imports) fits the
                        stacker on the same column the rotation measured;
  * `K2_RANK`/`RANK_REC` the climbed member's ranking column and the record
                        `land_rank.py` wrote for it, so the hash pin still runs;
  * `ARMS`              the arm's rotation record under `analysis/` here, so
                        `measured_arm`'s refusal to serve an unmeasured
                        construction still runs;
  * `HERE`              this experiment, so `served-*.json` is filed here.

`--identity` points every one of them back at F1-fresh's committed files, where
the substitution is the identity and the result must reproduce v7.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STEP = ROOT / "experiments/exp-027-step"


def pin_members(rot: Path, pattern: str) -> dict:
    """Refuse a serve whose member FILES are not the ones the rotation measured.

    Round 4 (FALLBACK) finding 3: exp-027's `measured_arm` pins the arm's
    construction -- block, window, member NAMES, seeds, rounds, the mask -- but
    not the member files, so a mismatched `--arm-oof X --rot Y` pair passed on
    the operator's typing alone.  The rotation record already carries their
    hashes (`exp_035.member_files`, written by `rotate_refit_bulk.py`); this
    compares them and exits before anything is fitted or written.  A record with
    no such block (exp-035's own earlier rotations) pins nothing and says so.
    """
    want = (json.loads(rot.read_text()).get("exp_035") or {}).get("member_files") or {}
    if not want:
        print(f"[pin] {rot.name} carries no member hashes; nothing pinned", flush=True)
        return {}
    got = {}
    for fold, spec in sorted(want.items()):
        f = ROOT / pattern.format(fold=fold)
        h = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "<missing>"
        if h != spec["sha256"]:
            raise SystemExit(f"[pin] {f} hashes {h[:8]}, but the rotation record "
                             f"measured {spec['sha256'][:8]} ({spec['path']}); "
                             f"refusing to serve an unmeasured member")
        got[fold] = h
    print("[pin] member files match the rotation record: "
          + " ".join(f"{f}={h[:8]}" for f, h in got.items()), flush=True)
    return got


def main() -> None:
    # `allow_abbrev=False` matters: with it on, argparse matched exp-027's
    # `--arm` as an abbreviation of this parser's `--arm-oof` and swallowed it,
    # so `--arm k2-refit-bulk-eham-out` silently served the DEFAULT arm off a
    # member path built from the arm's name.  Unknown options must fall through.
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--arm-oof", default="seeds",
                    help="which climbed member's fold files the stacker is fitted on")
    ap.add_argument("--rank", default="",
                    help="the climbed ranking column (default: candidates/<arm-oof>-ranking.parquet)")
    ap.add_argument("--rot", default="",
                    help="the rotation record (default: analysis/candidate-k2c-<arm-oof>.json)")
    known, rest = ap.parse_known_args()
    if "--identity" not in rest:
        os.environ["PRC_K2"] = f"experiments/exp-035-k2climb/oof/{known.arm_oof}-{{fold}}.parquet"
    os.environ.setdefault("PRC_TAG", f"serve-{known.arm_oof}")

    sys.path.insert(0, str(STEP))
    import serve as SV  # noqa: E402  (exp-027's serve.py)

    SV.HERE = HERE
    SV.K2_RANK = known.rank or f"experiments/exp-035-k2climb/candidates/{known.arm_oof}-ranking.parquet"
    SV.RANK_REC = HERE / "analysis" / f"{known.arm_oof}-ranking.json"
    rot = known.rot or f"candidate-k2c-{known.arm_oof}.json"
    SV.ARMS = {k: (rot, v[1]) for k, v in SV.ARMS.items()}
    if "PRC_K2" in os.environ:
        pin_members(HERE / "analysis" / rot, os.environ["PRC_K2"])
    print(f"[serve-035] member {os.environ.get('PRC_K2', '<identity>')}\n"
          f"[serve-035] ranking {SV.K2_RANK}\n"
          f"[serve-035] rotation record analysis/{rot}", flush=True)
    sys.argv = [sys.argv[0]] + rest
    SV.main()


if __name__ == "__main__":
    main()
