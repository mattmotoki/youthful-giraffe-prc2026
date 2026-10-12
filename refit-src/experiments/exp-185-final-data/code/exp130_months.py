"""exp-185: the exp-130 month package files for 2026 months, from the exp-130 prepare.py of a snapshot.

    [PRC_RANKING_MODE=final] .venv/bin/python exp130_months.py <snapshot> <out_dir> <exp018_cache_dir> <class_dir> <month> [...]

Run-only wrapper (desk, 2026-10-08): it imports prepare.py of the snapshot (a `git archive` of a main commit,
code/mk_snapshot.sh), points its CACHE at <exp018_cache_dir>, CLASS_DIR at <class_dir> and RANK_MONTHS at the
months given, and calls its own functions:
  month_<m>.parquet      cached_month(m) + with_lirf(base, raw_month(m, ranking), m), as prepare.main writes it,
  rank_meta_<m>.parquet  rank_metadata(ranking, submitting) for the months given.
The ranking and submitting files are the mode's (prc/data.py, by SHA-256). The fold files, the v22 preflight
and the census check of prepare.main belong to the ranking months and are not rebuilt.
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

snap, out, cache, class_dir, months = (Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(),
                                       Path(sys.argv[3]).resolve(), Path(sys.argv[4]).resolve(), sys.argv[5:])
sys.argv = [sys.argv[0], str(out)]          # prepare.py reads OUT from argv[1] at import
sys.path.insert(0, str(snap / "experiments/exp-130-ranking-exposure/code"))
import prepare as P  # noqa: E402

D = P.D
mode = D.ranking_mode() if hasattr(D, "ranking_mode") else "ranking"
assert all(m in D.rank_months() for m in months), (months, D.rank_months())
# exp-018 names a final-mode 2026 table <m>-final.parquet (dataset.final_tag, 5e936513f); prepare.cached_month
# reads <m>.parquet, so the final tables are linked under that name in a temporary folder
tag = "-final" if mode == "final" else ""
link_dir = Path(tempfile.mkdtemp())
for m in months:
    (link_dir / f"{m}.parquet").symlink_to(cache / f"{m}{tag}.parquet")
P.CACHE, P.CLASS_DIR, P.RANK_MONTHS = link_dir, class_dir, tuple(months)
out.mkdir(parents=True, exist_ok=True)
ranking_path, submitting_path = D.ranking_file(), D.submitting_file()
ranking = D.load_ranking(columns=sorted(set(P.SIGNAL_COLS)))
submitting = pd.read_parquet(submitting_path, columns=[P.ID])
sources: list[Path] = []
parts, _ = P.rank_metadata(ranking, submitting, sources)
rec = {"snapshot_commit": (snap / "COMMIT").read_text().strip(), "mode": mode, "months": months,
       "ranking_file": str(ranking_path), "ranking_sha256": P.sha256(ranking_path),
       "submitting_file": str(submitting_path), "submitting_sha256": P.sha256(submitting_path),
       "exp018_cache": str(cache), "class_dir": str(class_dir), "files": {}}
for m in months:
    base = P.cached_month(m, include_target=False)
    raw = P.raw_month(m, ranking)
    enriched = P.with_lirf(base, raw, m)
    for name, frame in ((f"month_{m}.parquet", enriched), (f"rank_meta_{m}.parquet", parts[m])):
        frame.to_parquet(out / name, index=False, compression="zstd")
        rec["files"][name] = {"rows": int(len(frame)), "columns": int(frame.shape[1]), "sha256": P.sha256(out / name)}
    rec["files"][f"rank_meta_{m}.parquet"]["rank_class"] = parts[m]["rank_class"].value_counts().to_dict()
    rec["inputs_" + m] = {"exp018_cache_file": f"{m}{tag}.parquet", "exp018_cache_sha256": P.sha256(cache / f"{m}{tag}.parquet"),
                          "class_sha256": P.sha256(class_dir / f"{m}-v25.parquet")}
    print(m, rec["files"][f"month_{m}.parquet"], rec["files"][f"rank_meta_{m}.parquet"]["rank_class"], flush=True)
(out / "exp130-months.json").write_text(json.dumps(rec, indent=2) + "\n")
