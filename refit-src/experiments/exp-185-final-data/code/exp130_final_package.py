"""exp-185: assemble the exp-130 input package of the final mode (the input of the exp-134 fit in final mode).

    python3 exp130_final_package.py <saved package dir> <dir with month_2026-*.parquet and rank_meta_2026-*.parquet>
                                    <final_submitting.parquet> <out dir> <commit> <source dir>

The 2025 parts do not change with the final file, so they are copied from the saved package (prc-exp130-clean-input):
month_2025-*.parquet, eval_*.parquet, preflight.json (v22 fold preflight: 2025 folds only), census.csv. The packaged
code (source/prc, source/exp018, fit.py) comes from <source dir>, a `git archive` of <commit>, as prepare.py main
copies it from its checkout: the learner kernels need a prc/data.py with the ranking modes. The 2026 parts are the four months of the final file from code/exp130_months.py: month_<m>.parquet and
rank_meta_<m>.parquet. Written here as prepare.py main writes them: rank_counts.csv (no-clock ordinary rows by month
and airport) and submitting_ids.parquet (row_order, MVT_ID_mvt of final_submitting.parquet). No v22_submission.parquet:
p22 exists for ranking.parquet only (design row D-p22 of the REPORT). input-manifest.json keeps the saved keys and
records the new files, the four months and the final mode.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

saved, new2026, submitting_path, out, commit, src = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
                                                     Path(sys.argv[4]), sys.argv[5], Path(sys.argv[6]))
PACKAGED = {  # prepare.py packaged_sources
    "source/prc/__init__.py": "prc/__init__.py",
    "source/prc/data.py": "prc/data.py",
    "source/prc/lirf_features.py": "prc/lirf_features.py",
    # in the saved package too (census.py lists it): prc/data.py imports it for the clean-score exclusions
    "source/prc/_private_clean_score_rows.py": "prc/_private_clean_score_rows.py",
    "source/exp018/dataset.py": "experiments/exp-018-fresh/code/dataset.py",
    "source/exp018/features.py": "experiments/exp-018-fresh/code/features.py",
    "source/exp018/model.py": "experiments/exp-018-fresh/code/model.py",
    "fit.py": "experiments/exp-130-ranking-exposure/code/fit.py",
}
MONTHS = ["2026-01", "2026-02", "2026-06", "2026-07"]
ID = "MVT_ID_mvt"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


if out.exists() and any(out.iterdir()):
    raise SystemExit(f"{out} is not empty; use a new folder")
out.mkdir(parents=True, exist_ok=True)
copied = []
for p in sorted(saved.iterdir()):
    keep = (p.name.startswith("month_2025-") or p.name.startswith("eval_")
            or p.name in {"preflight.json", "census.csv"})
    if not keep:
        continue
    if p.is_dir():
        shutil.copytree(p, out / p.name)
    else:
        shutil.copy2(p, out / p.name)
    copied.append(p.name)
assert sum(n.startswith("month_2025-") for n in copied) == 12, copied
for rel, origin in PACKAGED.items():
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / origin, out / rel)
metas = []
for m in MONTHS:
    for name in (f"month_{m}.parquet", f"rank_meta_{m}.parquet"):
        shutil.copy2(new2026 / name, out / name)
    month = pd.read_parquet(out / f"month_{m}.parquet", columns=[ID, "month"])
    assert month["month"].eq(m).all() and month[ID].is_unique
    meta = pd.read_parquet(out / f"rank_meta_{m}.parquet")
    assert meta["month"].eq(m).all() and meta[ID].is_unique
    assert set(meta[ID]) == set(month[ID].astype("int64")), m
    metas.append(meta)
rank_all = pd.concat(metas, ignore_index=True)
assert rank_all[ID].is_unique
sub = pd.read_parquet(submitting_path, columns=[ID])
ids = pd.to_numeric(sub[ID], errors="raise").astype("int64")
assert ids.is_unique and len(ids) == len(rank_all) and set(ids) == set(rank_all[ID]), "submitting and rank_meta IDs differ"
pd.DataFrame({"row_order": np.arange(len(ids), dtype="int64"), ID: ids.to_numpy()}).to_parquet(
    out / "submitting_ids.parquet", index=False, compression="zstd")
counts = (rank_all.loc[rank_all["rank_class"].eq("no-clock_ordinary")]
          .groupby(["month", "airport"], observed=True).size().rename("ranking_rows").reset_index())
counts.to_csv(out / "rank_counts.csv", index=False)

man = json.loads((saved / "input-manifest.json").read_text())
files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
man.update({
    "ranking_mode": "final",
    "ranking_months": MONTHS,
    "worker_source_commit": commit,
    "assembled_by": "experiments/exp-185-final-data/code/exp130_final_package.py",
    "derived_from": {"package": str(saved), "input_manifest_sha256": sha(saved / "input-manifest.json"),
                     "copied": copied},
    "packaged_source_commit": commit,
    "v22_submission": "none: p22 exists for ranking.parquet only; exp-185 REPORT design row D-p22",
    "submitting_file": {"path": str(submitting_path), "sha256": sha(submitting_path)},
    # the 2026 build record of code/exp130_months.py: final ranking file, exp-018 cache and class SHA-256 per month
    # (source_sha256 above stays the record of the saved package's own build)
    "final_2026_build": {p.name: json.loads(p.read_text()) for p in sorted(new2026.glob("exp130-months*.json"))},
    "input_sha256": files,
    "input_total_bytes": int(sum((out / k).stat().st_size for k in files)),
    "ranking_no_clock_ordinary_rows": int(rank_all["rank_class"].eq("no-clock_ordinary").sum()),
    "ranking_rows_by_month": {m: int(rank_all["month"].eq(m).sum()) for m in MONTHS},
    "rank_class_by_month": {m: rank_all.loc[rank_all["month"].eq(m), "rank_class"].value_counts().to_dict()
                            for m in MONTHS},
})
assert man.get("ranking_target_read") is False
(out / "input-manifest.json").write_text(json.dumps(man, indent=2) + "\n")
print(json.dumps({"files": len(files), "bytes": man["input_total_bytes"],
                  "rows_by_month": man["ranking_rows_by_month"],
                  "no_clock_ordinary": man["ranking_no_clock_ordinary_rows"]}))
