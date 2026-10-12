"""Copy only the selected fixed time field into the original row-control order."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_time_day_052 import sha

ROW = "e996b3cebe25c732a03ba749c654a194b4c310495b6a5ecd6def5f192cf6ad68"
VIEWS = {"Z": "d052ad0239f6ebca4c05320eac7ba0766dbbfcd158afc06a318fa880ce92d3b1",
         "T": "96a7f802e4945d01cbd93397eed4864767891731a53f85ebd7fd0c6962278b49"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-inputs", type=Path, required=True)
    ap.add_argument("--fixed-inputs", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--selection-review", type=Path, required=True)
    ap.add_argument("--selection-review-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert sha(a.selection_review) == a.selection_review_sha
    assert "PASS" in a.selection_review.read_text() and sha(a.selection) in a.selection_review.read_text()
    selection = json.loads(a.selection.read_text())
    assert selection["status"] == "PASS"
    form = selection["chosen_form"]
    assert selection["chosen_input_manifest_sha256"] == VIEWS[form]
    assert sha(a.row_inputs / "MANIFEST.json") == ROW
    assert sha(a.fixed_inputs / "MANIFEST.json") == VIEWS[form]
    row = json.loads((a.row_inputs / "MANIFEST.json").read_text())
    fixed = json.loads((a.fixed_inputs / "MANIFEST.json").read_text())
    for root, rec in [(a.row_inputs, row), (a.fixed_inputs, fixed)]:
        for name, digest in rec["files"].items():
            assert sha(root / name) == digest, str(root / name)
    assert row["numeric"] == fixed["numeric"] and len(row["numeric"]) == 59
    assert row["numeric"][57] == "seconds_in_day"
    receiver = pd.read_parquet(a.row_inputs / "rows.parquet", columns=["MVT_ID_mvt", "year", "month"])
    source = pd.read_parquet(a.fixed_inputs / "rows.parquet", columns=["MVT_ID_mvt", "year", "month"])
    for frame in [receiver, source]:
        assert frame.MVT_ID_mvt.dtype == np.int64 and frame.MVT_ID_mvt.is_unique
        assert len(frame) == 2429888
    index = pd.Index(source.MVT_ID_mvt).get_indexer(receiver.MVT_ID_mvt)
    assert (index >= 0).all() and len(np.unique(index)) == len(receiver)
    assert np.array_equal(source.year.to_numpy()[index], receiver.year.to_numpy())
    assert np.array_equal(source.month.to_numpy()[index], receiver.month.to_numpy())
    old = np.load(a.row_inputs / "numeric.npy", mmap_mode="r")
    chosen = np.load(a.fixed_inputs / "numeric.npy", mmap_mode="r")
    assert old.shape == chosen.shape == (2429888, 118) and old.dtype == chosen.dtype == np.float32
    a.out.mkdir(parents=True, exist_ok=False)
    numeric = np.lib.format.open_memmap(a.out / "numeric.npy", mode="w+", dtype="float32", shape=old.shape)
    same = [c for c in range(118) if c not in [57, 116]]
    for lo in range(0, len(receiver), 65536):
        hi = min(lo + 65536, len(receiver))
        block = numeric[lo:hi]
        block[:] = old[lo:hi]
        block[:, [57, 116]] = chosen[index[lo:hi]][:, [57, 116]]
        assert np.array_equal(block[:, same], old[lo:hi, same])
        assert np.array_equal(block[:, [57, 116]], chosen[index[lo:hi]][:, [57, 116]])
        assert np.isfinite(block).all()
    numeric.flush()
    del numeric
    for name in ["rows.parquet", "category.npy", "anchor.npy", "labels.npy"]:
        (a.out / name).symlink_to((a.row_inputs / name).resolve())
    center, scale = list(row["center"]), list(row["scale"])
    center[57], scale[57] = fixed["center"][57], fixed["scale"][57]
    files = {name: sha(a.out / name) for name in row["files"]}
    assert all(files[name] == row["files"][name] for name in files if name != "numeric.npy")
    receipt = {**row, "parent_manifest_sha256": ROW, "fixed_manifest_sha256": VIEWS[form],
               "selection_sha256": sha(a.selection), "source_code_sha256": sha(__file__),
               "fixed_time_form": form, "changed_numeric_columns": [57, 116],
               "array_rule": "copy selected57/116 by exact int64 ID; all other row-control fields exact",
               "receiver_order_exact": True, "all2025_departure_IDs_match": True,
               "all2026_departure_IDs_match": True, "legacy_row_gaps55_56_exact": True,
               "center": center, "scale": scale, "files": files}
    (a.out / "MANIFEST.json").write_text(json.dumps(receipt, indent=1) + "\n")
    assert sha(a.row_inputs / "numeric.npy") == row["files"]["numeric.npy"]
    print(json.dumps({"manifest_sha256": sha(a.out / "MANIFEST.json"), "fixed_form": form,
                      "rows": len(receiver), "changed_columns": [57, 116]}), flush=True)


if __name__ == "__main__":
    main()
