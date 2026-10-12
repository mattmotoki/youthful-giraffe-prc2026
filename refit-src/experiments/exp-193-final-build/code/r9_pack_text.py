"""Take the private decision label out of the R9 pack's final_build.mode text (exp-193, PLAN-R9.md; gate K of
release/check_release.py on manifest commit 379c21d0e: "released file(s) name private decisions").

    PRC_RANKING_MODE=final OMP_NUM_THREADS=2 .venv/bin/python r9_pack_text.py --release <release worktree> \
        --data-root <dir with data/raw>

The text is gainlane's label of its build wrapper; no release code reads final_build.mode (dump_chain.py reads only
final_build.v64_sha256). The script checks the pack written by r9_entries.py (18d0a13c…), changes that one string,
writes the pack in the same form, sets the new pack SHA-256 in the entries zabd-final-r9 and v64-final-r9, and runs
the T11b-L stage on the saved R9 zabd again: it must give the R9 v64 bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
PACK = "release/ZABD-PACK-final-r9.json"
OLD_SHA = "18d0a13c9da825bf6c298cfc81298db474c8e9cc590b6327f786a60555674132"
MODE_OLD = "final SIDE READ: R9 (exp-194 r9_build.py, D169): mix12 on member-scope rows of the primary frame"
MODE_NEW = "final SIDE READ: R9 (exp-194 r9_build.py): mix12 on member-scope rows of the primary frame"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args()
    assert os.environ.get("PRC_RANKING_MODE") == "final", "run with PRC_RANKING_MODE=final"
    sys.path.insert(0, str(a.release / "release"))
    sys.path.insert(0, str(a.release / "scripts"))
    import build_v32 as legacy  # noqa: E402
    import t11b_stage as T  # noqa: E402

    path = a.release / PACK
    text = path.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == OLD_SHA, "not the pack of r9_entries.py"
    pack = json.loads(text)
    assert json.dumps(pack, indent=1) + "\n" == text
    assert pack["final_build"]["mode"] == MODE_OLD
    pack["final_build"]["mode"] = MODE_NEW
    new_text = json.dumps(pack, indent=1) + "\n"
    assert new_text.count("D169") == 0
    new_sha = hashlib.sha256(new_text.encode()).hexdigest()
    book_path = a.release / "release/ENTRIES.json"
    book_text = book_path.read_text()
    book = json.loads(book_text)
    assert json.dumps(book, indent=2) + "\n" == book_text
    for name in ("zabd-final-r9", "v64-final-r9"):
        e = book["entries"][name]
        assert (e["pack"], e["pack_sha256"]) == (PACK, OLD_SHA), name
        e["pack_sha256"] = new_sha
    path.write_text(new_text)
    book_path.write_text(json.dumps(book, indent=2) + "\n")
    zabd, v64 = book["entries"]["zabd-final-r9"], book["entries"]["v64-final-r9"]
    values = legacy.saved_array(zabd["comparison_vector"], zabd["rows"])
    ids = pd.read_parquet(a.data_root / "data/raw/final_submitting.parquet")[ID].to_numpy("float64")
    buffer = io.BytesIO()
    pd.DataFrame({ID: ids, Y: values}).to_parquet(buffer, index=False)
    assert hashlib.sha256(buffer.getvalue()).hexdigest() == zabd["sha256"]
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "zabd-r9.parquet"
        zpath.write_bytes(buffer.getvalue())
        got = T.compose(v64, zpath, a.data_root)
    got_sha = hashlib.sha256(got["body"]).hexdigest()
    assert got_sha == v64["sha256"], got_sha
    print(json.dumps({"pack": PACK, "old_sha256": OLD_SHA, "new_sha256": new_sha, "mode": MODE_NEW,
                      "t11b_mac_sha256": got_sha, "checks": got["facts"]["checks"]}, indent=1))


if __name__ == "__main__":
    main()
