"""exp-185: the 2025 month files of the four-month exp-130 package on the Mac (route A, desk 2026-10-08 15:2x UTC).

    .venv/bin/python exp130_2025.py <snapshot> <out_dir> <exp018_cache_dir> <saved input-manifest.json> [month ...]

Run-only wrapper: for each 2025 month it runs the lines of prepare.py main (snapshot copy) that write
month_<m>.parquet: cached_month(m, include_target=True), raw_month(m), with_lirf. The 2025 parts do not depend on the
ranking file. Check: each file's SHA-256 against input_sha256 of the saved package manifest; the run stops on a
difference.
"""
import json
import sys
from pathlib import Path

snap, out, cache, saved = (Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(),
                           Path(sys.argv[4]).resolve())
months = sys.argv[5:] or [f"2025-{i:02d}" for i in range(1, 13)]
assert all(m.startswith("2025-") for m in months), months
sys.argv = [sys.argv[0], str(out)]          # prepare.py reads OUT from argv[1] at import
sys.path.insert(0, str(snap / "experiments/exp-130-ranking-exposure/code"))
import prepare as P  # noqa: E402

P.CACHE = cache
want = json.loads(saved.read_text())["input_sha256"]
out.mkdir(parents=True, exist_ok=True)
rec = {"snapshot_commit": (snap / "COMMIT").read_text().strip(), "exp018_cache": str(cache),
       "saved_manifest": str(saved), "months": {}}
for m in months:
    base = P.cached_month(m, include_target=True)
    raw = P.raw_month(m, None)
    enriched = P.with_lirf(base, raw, m)
    path = out / f"month_{m}.parquet"
    enriched.to_parquet(path, index=False, compression="zstd")
    got, ref = P.sha256(path), want[f"month_{m}.parquet"]
    rec["months"][m] = {"rows": int(len(enriched)), "columns": int(enriched.shape[1]), "sha256": got,
                        "exp018_cache_sha256": P.sha256(cache / f"{m}.parquet"), "saved_sha256": ref,
                        "equal": got == ref}
    print(m, rec["months"][m], flush=True)
    del base, raw, enriched
(out.parent / "exp130-2025.json").write_text(json.dumps(rec, indent=2) + "\n")
bad = [m for m, r in rec["months"].items() if not r["equal"]]
if bad:
    raise SystemExit(f"2025 month files differ from the saved package: {bad}")
