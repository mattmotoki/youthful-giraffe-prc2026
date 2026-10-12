"""exp-064-release: the v22 composition, checked on bytes rather than described.

Reads the uploaded files and the two class vectors and prints one JSON record
that maps every row of youthful-giraffe_v22.parquet to the lane that owns it.
Read-only; fits nothing.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
U = ROOT / "submissions/uploaded"
C = ROOT / "submissions/candidates"
CLS = ROOT / "artifacts/exp-046/class"
DEFAULT_OUT = ROOT / "experiments/exp-064-release/analysis/lineage.json"

# Write-once guard (desk, 2026-09-10): a run refuses to replace an existing
# measured artifact.  Called twice -- after argument parsing, so a refused run
# costs no compute, and again immediately before the write.
sys.path.insert(0, str(ROOT / "experiments/exp-046-surfacefit/code"))
from nooverwrite import refuse_existing                                # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the JSON record")
_ap.add_argument("--overwrite", action="store_true",
                 help="replace an existing record; a naming defect until shown otherwise")
_args = _ap.parse_args()
OUT = Path(_args.out)
refuse_existing([OUT], _args.overwrite)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def load(p):
    d = pd.read_parquet(p)
    return d

out = {}
files = {
    "v22": U / "youthful-giraffe_v22.parquet",
    "v21": U / "youthful-giraffe_v21.parquet",
    "v18": U / "youthful-giraffe_v18.parquet",
    "v17": U / "youthful-giraffe_v17.parquet",
    "v12": C / "exp-035-k2climb-v11member.parquet",
    "cand_v11": C / "exp-046-surface-v11.parquet",
    "cand_v10": C / "exp-046-surface-v10.parquet",
}
out["sha256"] = {k: sha(v) for k, v in files.items() if v.exists()}
out["bytes"] = {k: v.stat().st_size for k, v in files.items() if v.exists()}

v22 = load(files["v22"]); v18 = load(files["v18"]); v12 = load(files["v12"])
v17 = load(files["v17"]); v21 = load(files["v21"])
tmpl = pd.read_parquet(ROOT / "data/raw/submitting.parquet")

col = "TAXITIME_SEC_mvt"
out["shape"] = {"rows": int(len(v22)), "columns": list(v22.columns)}
out["template"] = {
    "rows": int(len(tmpl)),
    "ids_equal_in_order": bool((v22["MVT_ID_mvt"].values == tmpl["MVT_ID_mvt"].values).all()),
}
p22 = v22[col].to_numpy(); p18 = v18[col].to_numpy(); p12 = v12[col].to_numpy()
p17 = v17[col].to_numpy(); p21 = v21[col].to_numpy()
out["values"] = {
    "dtype": str(v22[col].dtype),
    "finite": bool(np.isfinite(p22).all()),
    "min": float(p22.min()), "max": float(p22.max()),
    "rows_at_the_60s_floor": int((p22 == 60.0).sum()),
    "below_60": int((p22 < 60.0).sum()),
}

# the class vector v25 the serve declares
cls = pd.concat([pd.read_parquet(CLS / "2026-01-v25.parquet"),
                 pd.read_parquet(CLS / "2026-07-v25.parquet")], ignore_index=True)
out["class_vector"] = {
    "files": ["artifacts/exp-046/class/2026-01-v25.parquet",
              "artifacts/exp-046/class/2026-07-v25.parquet"],
    "rows": int(len(cls)),
    "columns": list(cls.columns),
}
cls = cls.set_index("MVT_ID_mvt").reindex(v22["MVT_ID_mvt"].values)
for c in ("obs_has_clock", "served_by_exp_046", "occupied_by_exp_046", "written_by_exp_046",
          "guard_held_by_exp_046", "chain_mode_4", "noac", "head_by_exp_046"):
    if c in cls.columns:
        v = cls[c]
        out["class_vector"][c] = (int(v.fillna(False).astype(bool).sum())
                                  if v.dtype != object else v.value_counts(dropna=False).to_dict())

occ = cls["occupied_by_exp_046"].fillna(False).to_numpy(bool) if "occupied_by_exp_046" in cls else None
wri = cls["written_by_exp_046"].fillna(False).to_numpy(bool) if "written_by_exp_046" in cls else None
grd = cls["guard_held_by_exp_046"].fillna(False).to_numpy(bool) if "guard_held_by_exp_046" in cls else None

d_18 = p22 - p18
out["against_v18"] = {
    "rows_changed": int((p22 != p18).sum()),
    "rows_identical": int((p22 == p18).sum()),
    "rms_change_s": float(np.sqrt(np.mean(d_18 ** 2))),
    "max_abs_change_s": float(np.abs(d_18).max()),
}
if occ is not None:
    out["against_v18"]["changed_inside_the_class"] = int(((p22 != p18) & occ).sum())
    out["against_v18"]["changed_outside_the_class"] = int(((p22 != p18) & ~occ).sum())
    out["outside_the_class"] = {
        "rows": int((~occ).sum()),
        "equal_v18": int((p22[~occ] == p18[~occ]).sum()),
        "equal_v12": int((p22[~occ] == p12[~occ]).sum()),
    }
    out["inside_the_class"] = {
        "rows": int(occ.sum()),
        "equal_v18": int((p22[occ] == p18[occ]).sum()),
        "equal_v12": int((p22[occ] == p12[occ]).sum()),
    }
if grd is not None and occ is not None:
    held = grd
    out["guard_held"] = {
        "rows": int(held.sum()),
        "equal_v18": int((p22[held] == p18[held]).sum()),
        "equal_v12": int((p22[held] == p12[held]).sum()),
        "v18_equals_v12_there": int((p18[held] == p12[held]).sum()),
        "inside_the_class": int((held & occ).sum()),
    }

# the queue lane's rows, as they reach v22 through v18
out["queue_rows_carried_through_v18"] = {
    "v18_vs_v17_changed": int((p18 != p17).sum()),
    "v17_vs_v12_changed": int((p17 != p12).sum()),
    "v22_equal_v17_on_rows_v17_moved_off_v12": int(
        (p22[(p17 != p12)] == p17[(p17 != p12)]).sum()),
    "rows_v17_moved_off_v12": int((p17 != p12).sum()),
}
out["against_v21"] = {
    "rows_changed": int((p22 != p21).sum()),
    "rms_change_s": float(np.sqrt(np.mean((p22 - p21) ** 2))),
}

# per-airport census from the ranking file
rk = pd.read_parquet(ROOT / "data/raw/ranking.parquet",
                     columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt"])
rk = rk.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").reindex(v22["MVT_ID_mvt"].values)
ap = rk["ADEP_mvt"].to_numpy()
mon = pd.to_datetime(rk["MVT_TIME_UTC_mvt"]).dt.strftime("%Y-%m").to_numpy()
noac = rk["AIRCRAFT_TYPE_mvt"].isna().to_numpy()
out["noac"] = {"rows": int(noac.sum()),
               "equal_v18": int((p22[noac] == p18[noac]).sum()),
               "equal_v12": int((p22[noac] == p12[noac]).sum())}
tab = {}
for a in sorted(set(ap.tolist())):
    m = ap == a
    tab[a] = {"rows": int(m.sum()),
              "changed_vs_v18": int((p22[m] != p18[m]).sum()),
              "in_class": int(occ[m].sum()) if occ is not None else None}
out["by_airport"] = tab
out["by_month"] = {m: {"rows": int((mon == m).sum()),
                       "changed_vs_v18": int((p22[mon == m] != p18[mon == m]).sum()),
                       "in_class": int(occ[mon == m].sum()) if occ is not None else None}
                   for m in sorted(set(mon.tolist()))}


# --- decomposition of the 148,585 rows v22 leaves at v18's bytes -------------
if occ is not None:
    out2 = {}
    outside = ~occ
    diff12 = p22 != p12
    m = outside & diff12
    out2["rows_outside_the_class_that_differ_from_v12"] = int(m.sum())
    out2["by_airport"] = {a: int((m & (ap == a)).sum()) for a in sorted(set(ap.tolist())) if (m & (ap == a)).sum()}
    out2["by_month"] = {mm: int((m & (mon == mm)).sum()) for mm in sorted(set(mon.tolist()))}
    # do they equal v17 (the last file the queue lane wrote)?
    out2["equal_v17"] = int((p22[m] == p17[m]).sum())
    out2["equal_v18"] = int((p22[m] == p18[m]).sum())
    qcells = {}
    for a in sorted(set(ap.tolist())):
        for mm in sorted(set(mon.tolist())):
            k = (ap == a) & (mon == mm) & m
            if k.sum():
                qcells[f"{a}/{mm}"] = int(k.sum())
    out2["by_airport_month"] = qcells
    out["rows_v22_leaves_at_v18_and_off_v12"] = out2
    out["ownership"] = {
        "exp_046_with_clock_head": int(occ.sum()),
        "of_which_guard_held_at_the_price_base": int(grd.sum()) if grd is not None else None,
        "exp_047_queue_no_clock_arm": int(m.sum()),
        "base_v12_unchanged": int((~occ & ~diff12).sum()),
        "total": int(occ.sum() + m.sum() + (~occ & ~diff12).sum()),
        "noac_inside_base_v12_unchanged": int(noac.sum()),
    }

refuse_existing([OUT], _args.overwrite)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
try:
    _shown = OUT.relative_to(ROOT)
except ValueError:
    _shown = OUT
print(f"\nwrote {_shown}")
