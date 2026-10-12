"""§5 boundary: the one module in this experiment that reads a 2025 label.

Everything before it — the frames, the distances, the placement, the two-sample
test — runs on files `build.py` produced without opening a label column
(`build.py` asserts that).  This module re-reads the twelve training months for
the target and the departure block time alone, builds the two placeholder
classes, writes `labels_2025.parquet`, and joins it onto the label-free frame to
give `noac_2025_lab.parquet`, which is what §5 onward reads.

`ph` is the record's own placeholder class, `|mvt_sched - y| < 300 s`
(`prc.tail_model.PLACEHOLDER_TOL`).  `ph_minute` is exp-022's minute-equality
class, kept only so §1's table can show that it is a different class.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
from prc.tail_model import PLACEHOLDER_TOL, placeholder_label  # noqa: E402
from describe import LAB  # noqa: E402


def labels() -> pd.DataFrame:
    """`y_true`, the block time and the two placeholder classes, per noac row."""
    out = []
    for m in D.MONTHS:
        raw = D.load_month(m)
        dep = raw[(raw.PHASE_mvt == "DEP") & raw.AIRCRAFT_TYPE_mvt.isna()]
        idx = dep.index
        out.append(pd.DataFrame({
            "MVT_ID_mvt": dep.MVT_ID_mvt.to_numpy("float64"),
            "y_true": raw[D.TARGET].reindex(idx).to_numpy("float64"),
            "block_time": raw.BLOCK_TIME_UTC_mvt.reindex(idx).to_numpy(),
            "ph_minute": placeholder_label(raw, idx).to_numpy(),
        }))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_d.parquet")
    assert "y_true" not in a.columns and "ph" not in a.columns, "the §1 frame is not label-free"
    lab = labels()
    ms = a.set_index("MVT_ID_mvt").mvt_sched.reindex(lab.MVT_ID_mvt).to_numpy(float)
    assert np.isfinite(ms).all(), "a labelled row is missing from the §1 frame"
    lab["ph"] = (np.abs(ms - lab.y_true.to_numpy(float)) < PLACEHOLDER_TOL).astype(float)
    lab.to_parquet(EXP / "work" / "labels_2025.parquet")

    out = a.merge(lab, on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert out.y_true.notna().all() and len(out) == len(a)
    out.to_parquet(EXP / "work" / "noac_2025_lab.parquet")

    rec = pd.read_csv(EXP.parents[0] / "exp-026-rebuild" / "analysis" / "noac_cells.csv")
    rec = rec.set_index("band").P_ph_2025
    rows = []
    for b in LAB:
        m = out.band == b
        rows.append({"band": b, "n": int(m.sum()),
                     "P_tol300": float(out.ph[m].mean()),
                     "P_minute": float(out.ph_minute[m].astype(float).mean()),
                     "record_P_ph_2025": float(rec.get(b, np.nan))})
    t = pd.DataFrame(rows)
    t["tol300_matches_record"] = np.isclose(t.P_tol300, t.record_P_ph_2025, atol=5e-4)
    t.to_csv(EXP / "analysis" / "placeholder_definition.csv", index=False)
    print(t.round(3).to_string(index=False))
    print("\nevery band reproduces the record with the 300 s tolerance:",
          bool(t.tol300_matches_record.all()))


if __name__ == "__main__":
    main()
