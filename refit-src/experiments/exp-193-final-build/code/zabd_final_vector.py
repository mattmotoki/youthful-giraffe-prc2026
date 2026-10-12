"""Saved zabd values of the final build for the release's zabd-final entry (exp-193, release step 2c).

    .venv/bin/python zabd_final_vector.py --release <release worktree> --zabd <finalplan parts/final-2/zabd.parquet> \
        --submitting <data/raw/final/final_submitting.parquet>

Input: finalplan's zabd part of build 2 (MANIFEST row zabd, SHA-256 e6503136…; columns ID and zabd in
final_submitting order). Writes release/saved-vectors/final-entries/zabd.json with the release's encoder
(assembly_replay.encode_array, float64), decodes it with build_v32.saved_array (the reader of build_entry.py and
v73_stage.chain_reference), and writes the zabd table in the form of v73_stage.chain_reference (ID of
final_submitting as float64, the decoded values): its SHA-256 must be the pack's candidate_sha256 (30d26797…, which
finalplan's pack check replayed on x86). Prints the comparison_vector spec for release/ENTRIES.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ZABD_PART = "e6503136921882b66dfb822775273e2134fd153801f83082e58d71906fa845d4"
CANDIDATE = "30d26797a964e6d7119adaa713cd865f5b6f25f3877c79cbd3c5096f0b5655f8"
SUBMITTING = "submitting"  # the pack keys the official files by role (final mode: final_submitting.parquet)
ROWS = 670_790
OUT = "release/saved-vectors/final-entries/zabd.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--zabd", type=Path, required=True)
    ap.add_argument("--submitting", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.release / "release"))
    sys.path.insert(0, str(a.release / "scripts"))
    import assembly_replay as R  # noqa: E402
    import build_v32 as legacy  # noqa: E402

    assert sha256(a.zabd) == ZABD_PART, f"{a.zabd} is not finalplan's zabd part of build 2"
    pack = json.loads((a.release / "release/ZABD-PACK-final.json").read_text())
    assert pack["candidate_sha256"] == CANDIDATE
    assert sha256(a.submitting) == pack["official_sha256"][SUBMITTING], "not the final submitting file of the pack"
    part = pd.read_parquet(a.zabd)
    ids = pd.read_parquet(a.submitting)[ID].to_numpy("float64")
    assert len(part) == len(ids) == ROWS and "zabd" in part.columns, (len(part), list(part.columns))
    assert np.array_equal(part[ID].to_numpy("float64"), ids), "zabd part IDs differ from final_submitting order"
    assert np.isfinite(part["zabd"].to_numpy("float64")).all()
    out = a.release / OUT
    if out.exists():
        raise SystemExit(f"refusing to replace {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R.encode_array(part["zabd"].astype("float64"))) + "\n")
    spec = {"public_vector": OUT, "public_vector_sha256": sha256(out)}
    values = legacy.saved_array(spec, ROWS)
    assert np.array_equal(values, part["zabd"].to_numpy("float64"))
    buffer = io.BytesIO()  # the form of v73_stage.chain_reference
    pd.DataFrame({ID: ids, Y: values}).to_parquet(buffer, index=False)
    got = hashlib.sha256(buffer.getvalue()).hexdigest()
    if got != CANDIDATE:
        raise SystemExit(f"the zabd table from the saved values has SHA-256 {got}, the pack's candidate is {CANDIDATE}")
    frame = pd.read_parquet(io.BytesIO(buffer.getvalue()))
    print(json.dumps({"comparison_vector": spec, "rows": ROWS, "bytes": len(buffer.getvalue()), "sha256": got,
                      "value_sha256": legacy.value_sha256(frame)}, indent=1))


if __name__ == "__main__":
    main()
