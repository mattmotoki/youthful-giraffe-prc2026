"""Score every committed OOF file on the bulk, so the 'best per fold' claim has a
generator (review round 1, non-blocking 7)."""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-011-bulkfloor/code")
import prc.data as d
import common as C

SRC = ["experiments/exp-008-diversity/oof", "experiments/exp-001-retro-v1/oof",
       "experiments/exp-002-tail/oof", "experiments/exp-006-lirf/oof",
       "experiments/exp-005-weather/oof", "experiments/exp-007-winter/oof"]


def main():
    months = {}
    rows = []
    for src in SRC:
        for p in sorted(glob.glob(str(d.ROOT / src / "*.parquet"))):
            try:
                o = pd.read_parquet(p)
            except Exception:
                continue
            if not {"MVT_ID_mvt", "y_true", "y_pred"} <= set(o.columns):
                continue
            folds = o.fold.unique() if "fold" in o.columns else \
                [os.path.basename(p).rsplit("-", 1)[-1][:-8]]
            for f in folds:
                if f not in d.FOLDS:
                    continue
                s = o[o.fold == f] if "fold" in o.columns else o
                vm = d.FOLDS[f].valid_month
                if vm not in months:
                    m = d.load_month(vm, columns=["MVT_ID_mvt", "PHASE_mvt", "FLIGHT_ID_mvt",
                                                  "AIRCRAFT_TYPE_mvt"])
                    months[vm] = m[m.PHASE_mvt == "DEP"]
                j = s.merge(months[vm], on="MVT_ID_mvt")
                b = j.FLIGHT_ID_mvt.notna() & j.AIRCRAFT_TYPE_mvt.notna() & j.y_true.le(3600)
                if b.sum() < 1000:
                    continue
                rows.append({"src": os.path.basename(src), "file": os.path.basename(p),
                             "name": os.path.basename(p)[:-8], "fold": f,
                             "n_bulk": int(b.sum()),
                             "bulk": d.rmse(j.y_true[b], j.y_pred[b])})
    T = pd.DataFrame(rows)
    T.to_csv(C.ANA / "oof_inventory.csv", index=False)
    print(f"{T.file.nunique()} distinct files, {len(T)} file-fold rows")
    for f in ("july", "july_loo", "december", "january"):
        s = T[T.fold == f].nsmallest(3, "bulk")
        if len(s):
            print(f"--- {f} ---")
            print(s[["name", "src", "n_bulk", "bulk"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
