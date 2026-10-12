"""Write work/w2/chain_ids_{fold,ranking}.parquet (LINEAGE-ZABD gap 8), the files that w2/seq/export.py and
w2/extra_screen.py read: the MVT_ID_mvt column of work/chain/{split}.parquet, in chain row order.

The order is asserted against the official files: fold = the departures of the January and then the July 2025
training files in file order (release/assembly_replay.official_tables); ranking = the template order
(submitting.parquet, or final_submitting.parquet with PRC_RANKING_MODE=final).

    python chain_ids.py <run>                      # from <run>/work/chain/{split}.parquet
    python chain_ids.py <run> --official ranking   # ranking split from the template only (no chain yet)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
HELD = ("training_2025-01-01_2025-02-01.parquet", "training_2025-07-01_2025-08-01.parquet")


def official_order(run: Path, split: str) -> np.ndarray:
    sys.path.insert(0, str(run / "iso"))
    import finalmode as FM
    raw = run / "input-root/data/raw"
    if split == "ranking":
        ids = pd.read_parquet(FM.submitting_path(raw), columns=[ID])[ID].to_numpy("float64")
        assert len(ids) == FM.n_rows() and len(np.unique(ids)) == len(ids)
    else:
        parts = [pd.read_parquet(raw / f, columns=[ID, "PHASE_mvt"]) for f in HELD]
        ids = np.concatenate([p.loc[p.PHASE_mvt.eq("DEP"), ID].to_numpy("float64") for p in parts])
        assert len(np.unique(ids)) == len(ids)
    assert np.array_equal(ids, np.round(ids)), split
    return ids.astype("int64")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--official", default="", help="comma list of splits written from the official order only")
    a = ap.parse_args()
    run = a.run.resolve()
    out = run / "work/w2"
    out.mkdir(parents=True, exist_ok=True)
    only = {s for s in a.official.split(",") if s}
    for split in ("fold", "ranking"):
        want = official_order(run, split)
        chain = run / f"work/chain/{split}.parquet"
        if split in only:
            ids, src = want, "official order"
        else:
            ids, src = pd.read_parquet(chain, columns=[ID])[ID].to_numpy("int64"), str(chain)
            assert np.array_equal(ids, want), f"{split}: chain order differs from the official order"
        pd.DataFrame({ID: ids}).to_parquet(out / f"chain_ids_{split}.parquet", index=False)
        print(split, len(ids), "rows from", src, "-> equal to the official order", flush=True)


if __name__ == "__main__":
    main()
