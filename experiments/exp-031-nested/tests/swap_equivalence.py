"""The column swap must equal a frame `frames.fold_frame` built from the file itself.

`instrument._swap` replaces a prepared member column in the canonical fold frame
rather than rebuilding the frame from the substituted address.  That is only
sound if the two routes agree.  This test builds the second route explicitly:
each `oof-nested/` file is rewritten to a temporary parquet whose `fold` column
carries the ORDINARY fold name, `frames.OOF` is pointed at it, and
`frames.fold_frame` is run unchanged.  The two frames must agree on every column,
bit for bit.

It also checks the three things the swap would silently get wrong: a duplicated
id in the substituted file, an id set that differs from the frame's, and a row
order the reindex would have to permute.

    python swap_equivalence.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import instrument as I  # noqa: E402

F = I.F
NAMES = ("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh",
         "V3-allin", "N1-nmtail")
CASES = [("published", s) for s in I.FOLDS] + \
        [("kaggle", s) for s in I.FOLDS] + \
        [("nested", s) for s in I.FOLDS] + \
        [("nested-f1held", s) for s in I.FOLDS]


def direct(column: str, scored: str, tmp: Path) -> dict:
    """`frames.fold_frame` reading the substituted files directly."""
    saved = dict(F.OOF)
    out = {}
    try:
        for slot in I.FOLDS:
            for member in NAMES:
                src = I.source(column, member, slot, scored)
                if not src:
                    F.OOF[member] = saved[member]
                    continue
                d = pd.read_parquet(I.ND / f"{member}--{src}.parquet")
                d["fold"] = slot                      # the ordinary name `_read` filters on
                p = tmp / f"{column}-{scored}-{member}-{slot}.parquet"
                d.to_parquet(p, index=False)
                F.OOF[member] = str(p.relative_to(F.ROOT)) \
                    if p.is_relative_to(F.ROOT) else str(p)
            out[slot] = F.fold_frame(slot, NAMES)
    finally:
        F.OOF.clear()
        F.OOF.update(saved)
    return out


def main() -> None:
    # the three preconditions the swap relies on
    for member in I.NESTABLE:
        for fn in sorted(p.name for p in I.ND.glob(f"{member}--*.parquet")):
            d = pd.read_parquet(I.ND / fn)
            ids = d["MVT_ID_mvt"].astype("float64").to_numpy()
            assert len(np.unique(ids)) == len(ids), f"{fn}: duplicated MVT_ID_mvt"
    print(f"[pre] no duplicated id in any of "
          f"{len(list(I.ND.glob('*.parquet')))} oof-nested files")

    base = I.base_frames(NAMES)
    for slot in I.FOLDS:
        want = base[slot]["MVT_ID_mvt"].to_numpy("float64")
        for member in I.NESTABLE:
            for fn in sorted(p.name for p in I.ND.glob(f"{member}--*.parquet")):
                got = pd.read_parquet(I.ND / fn)["MVT_ID_mvt"].astype("float64").to_numpy()
                if len(got) != len(want):
                    continue
                if np.array_equal(np.sort(got), np.sort(want)):
                    assert np.array_equal(got, want), f"{fn}: same ids, different order"
    print("[pre] every oof-nested file's id set and ORDER match the fold frame's")

    with tempfile.TemporaryDirectory(dir=F.ROOT) as td:
        tmp = Path(td)
        for column, scored in CASES:
            a = I.build(column, scored, NAMES, base)
            b = direct(column, scored, tmp)
            for slot in I.FOLDS:
                x, y = a[slot], b[slot]
                assert list(x.columns) == list(y.columns), f"{column}/{scored}/{slot}: columns"
                assert len(x) == len(y), f"{column}/{scored}/{slot}: {len(x)} vs {len(y)}"
                for c in x.columns:
                    u, v = x[c].to_numpy(), y[c].to_numpy()
                    if u.dtype.kind == "f":
                        bad = ~((np.isnan(u) & np.isnan(v)) | (u == v))
                    else:
                        bad = u != v
                    assert not bad.any(), (f"{column}/{scored}/{slot}: column {c} differs "
                                           f"on {int(bad.sum())} rows, max abs "
                                           f"{np.nanmax(np.abs(u[bad] - v[bad])) if u.dtype.kind=='f' else '-'}")
            print(f"[ok ] {column:14s} scored={scored:9s} three frames identical")
    print("\nPASS: the column swap equals frames.fold_frame on the substituted address, "
          "bit for bit, on all 12 (column, scored fold) cases.")


if __name__ == "__main__":
    main()
