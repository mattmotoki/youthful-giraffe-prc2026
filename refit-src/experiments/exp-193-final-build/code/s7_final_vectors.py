"""Saved s7 member values of the final build for the release's final entry (exp-193, release step 2b).

    .venv/bin/python s7_final_vectors.py --release <release worktree> --fold <s7_fold.parquet> \
        --ranking <s7_ranking.parquet> --chain-fold <chain_ids_fold.parquet> --chain-ranking <chain_ids_ranking.parquet>

Inputs: finalv's member files work/w2/s7_{fold,ranking}.parquet of run-final-c2657cca1 (w2/seq/export.py: the chain
IDs and the seed mean of the held or full fits, float64) and the chain_ids files of the same run. Each input is
checked by SHA-256 against finalv's record (MANIFEST rows of s7c and s7c-held-fold, 788231cfe; log of the s7 inputs).
Writes release/saved-vectors/s7c/final-{fold,ranking}.json with the release's encoder (assembly_replay.encode_array,
float64), then decodes each file and writes the member file again in the form of release/v73_stage.member_files:
its SHA-256 must equal finalv's file. Prints the member spec for release/ENTRIES.json.
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

ID = "MVT_ID_mvt"
#: finalv's records: s7c MANIFEST row (member file work/w2/s7_ranking.parquet), s7c-held-fold row (a byte copy of
#: work/w2/s7_fold.parquet), and the chain_ids hashes of finalv-s7inputs-final-c2657cca1.log
EXPECTED = {
    "ranking": "8e3956fd5c929b68e0f66155f5e16bad4e13d1c00b2310d04a75723fa8095deb",
    "fold": "390733bc8634b9d898faea76f7786d6158fc8cde48200df9b0331a2b01556471",
    "chain_ranking": "fd895855236c8931283695d631944e628ee1a6176e3f4e2319115496c08d26c3",
    "chain_fold": "47a0128f337c04ff8622577f635ad1d9c37101840a99ac49997606c8cfeb4486",
}
ROWS = {"fold": 344_419, "ranking": 670_790}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    for split in ("fold", "ranking"):
        ap.add_argument(f"--{split}", type=Path, required=True)
        ap.add_argument(f"--chain-{split}", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.release / "release"))
    import assembly_replay as R  # noqa: E402

    spec = {"name": "s7", "seeds_per_mode": 3}
    for split in ("fold", "ranking"):
        src, chain_path = getattr(a, split), getattr(a, f"chain_{split}")
        assert sha256(src) == EXPECTED[split], f"{src} is not finalv's {split} member file"
        assert sha256(chain_path) == EXPECTED[f"chain_{split}"], f"{chain_path} is not finalv's chain_ids file"
        member = pd.read_parquet(src)
        chain = pd.read_parquet(chain_path)[ID].to_numpy()
        assert list(member.columns) == [ID, "s7"] and len(member) == ROWS[split] == len(chain)
        assert np.array_equal(member[ID].to_numpy(), chain), f"{split}: member IDs differ from the chain IDs"
        out = a.release / f"release/saved-vectors/s7c/final-{split}.json"
        if out.exists():
            raise SystemExit(f"refusing to replace {out}")
        out.write_text(json.dumps(R.encode_array(member["s7"])) + "\n")  # the form of exp-149 build_v73_vectors.py
        vsha = sha256(out)
        values = R.decode_array(out, vsha, ROWS[split])
        assert np.array_equal(values, member["s7"].to_numpy(), equal_nan=True)
        buffer = io.BytesIO()  # the form of v73_stage.member_files
        pd.DataFrame({ID: chain, "s7": values}).to_parquet(buffer, index=False)
        again = hashlib.sha256(buffer.getvalue()).hexdigest()
        if again != EXPECTED[split]:
            raise SystemExit(f"{split}: the member file written from the saved values has SHA-256 {again}")
        spec[split] = {"public_vector": f"release/saved-vectors/s7c/final-{split}.json", "public_vector_sha256": vsha,
                       "rows": ROWS[split], "file_sha256": EXPECTED[split],
                       "finite_rows": int(np.isfinite(values).sum())}
    print(json.dumps(spec, indent=1))


if __name__ == "__main__":
    main()
