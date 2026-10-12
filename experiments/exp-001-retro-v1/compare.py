"""One table comparing every saved run on both the required outcome split and
the input-defined split the climb actually steers by."""
from __future__ import annotations
import sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import prc.data as D  # noqa: E402

FOLDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["july", "december", "january"]
names = sys.argv[2].split(",") if len(sys.argv) > 2 else None

meta = {}
for f in FOLDS:
    d = D.load_month(D.FOLDS[f].valid_month, columns=["MVT_ID_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt"])
    meta[f] = d[d.PHASE_mvt == "DEP"].set_index("MVT_ID_mvt")

rows = []
for path in sorted(glob.glob(str(HERE / "oof" / "*.parquet"))):
    m = re.match(r"(.+)-(july|december|january|may)$", Path(path).stem)
    if not m:
        continue
    name, fold = m.groups()
    if fold not in meta or (names and name not in names):
        continue
    o = pd.read_parquet(path).set_index("MVT_ID_mvt")
    md = meta[fold]
    noac = md.AIRCRAFT_TYPE_mvt.isna().reindex(o.index).to_numpy()
    # "no NM join" is the absence of the join key itself, which is the definition
    # run.py uses for rmse_no_nm; AOBT_3_flt is null on four more july rows.
    nm = md.FLIGHT_ID_mvt.isna().reindex(o.index).to_numpy()
    se = ((o.y_true - o.y_pred) ** 2).to_numpy()
    y = o.y_true.to_numpy()
    core = ~noac & ~nm
    onm = ~noac & nm
    rms = lambda m: float(np.sqrt(se[m].mean())) if m.any() else float("nan")
    rows.append(dict(
        run=name, fold=fold, n=len(o), rmse=float(np.sqrt(se.mean())),
        rmse_le3600=rms(y <= 3600), rmse_gt3600=rms(y > 3600), rmse_noNM=rms(nm),
        core=rms(core), core_sse=100 * se[core].sum() / se.sum(),
        noNM_not_noac=rms(onm), noac=rms(noac), noac_sse=100 * se[noac].sum() / se.sum(),
    ))
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
order = t[t.fold == "july"].sort_values("rmse").run.tolist()
order += [r for r in t.run.unique() if r not in order]
piv = t.pivot_table(index="run", columns="fold", values="rmse").reindex(order)
print("== total RMSE by fold ==")
print(piv.round(1).to_string())
print("\n== required split (conditions on the outcome) and input-defined split ==")
cols = ["run", "fold", "rmse", "rmse_le3600", "rmse_gt3600", "rmse_noNM", "core", "core_sse", "noNM_not_noac", "noac", "noac_sse"]
print(t[cols].sort_values(["fold", "rmse"]).round(1).to_string(index=False))
