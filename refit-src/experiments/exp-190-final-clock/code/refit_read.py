"""Refit read of exp-190: a rebuilt HB3u chain against the released exp-159 tables and fits (development read).

    $PY experiments/exp-190-final-clock/code/refit_read.py <work dir> <out.json>

1. Tables: SHA-256 of frame, frame12, track, track-r5, track12 against the released records; in the final mode
   (tables differ by design) the 2025 rows of frame12 and track12 against the released 2025 rows.
2. Seed vectors: each seed of `HU18` and `HU30s2` (jjrank, jjrefit, janjul) against the released seed vector of
   the same seed on the same ids: rms and largest difference.
3. The merged HB3u of each split against the released `out/fits-HB3` file: rms, largest difference, equal bits.
4. The saved boosters: file, size, SHA-256.
5. With `<work>/pack` and `<work>/parts/head.parquet`: the pack's jjrank file against the released pack file
   (SHA-256), and the head part against the pack's values on the same ids (equal bits; rows finite in one only).
Reads no label. Write-once.
"""
from __future__ import annotations

import os
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path(os.path.join(os.environ.get("PRC_COMP_ROOT", "/path/to/competitions"), "prc-exp159-work"))
ID = "MVT_ID_mvt"
TABLES = {"frame": "frame", "frame12": "frame12", "track": "track", "track-r5": "track-r5", "track12": "track12"}
PARTS = {"HU18": OLD / "out/hb2x", "HU30s2": OLD / "out/hb2x", "HB3u": OLD / "out/fits-HB3"}
SPLITS = ("jjrank", "jjrefit", "janjul")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a, "float64") - np.asarray(b, "float64")
    return {"rms": float(np.sqrt(np.mean(d ** 2))), "max_abs": float(np.abs(d).max()), "mean": float(d.mean()),
            "rows_differ": int((d != 0).sum()), "bits_equal": bool(np.array_equal(np.asarray(a), np.asarray(b)))}


def main() -> None:
    work, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f"{out} exists; write-once")
    rec: dict = {"work": str(work), "tables": {}, "seeds": {}, "merged": {}, "boosters": {}}
    for name in TABLES:
        new_p, old_rec = work / f"inputs/{name}.parquet", json.loads((OLD / f"inputs/{name}.json").read_text())
        if new_p.exists():
            rec["tables"][name] = {"sha256": sha(new_p), "released": old_rec["output_sha256"]}
            rec["tables"][name]["equal"] = rec["tables"][name]["sha256"] == old_rec["output_sha256"]
    for name in ("frame12", "track12"):
        new_p = work / f"inputs/{name}.parquet"
        if not new_p.exists() or rec["tables"].get(name, {}).get("equal"):
            continue
        months = pd.read_parquet(OLD / "inputs/frame12.parquet", columns=[ID, "month"])
        keep25 = set(months.loc[months.month.str.startswith("2025"), ID])
        old_t = pd.read_parquet(OLD / f"inputs/{name}.parquet")
        new_t = pd.read_parquet(new_p)
        old_t = old_t[old_t[ID].isin(keep25)].sort_values(ID).reset_index(drop=True)
        new_t = new_t[new_t[ID].isin(keep25)].sort_values(ID).reset_index(drop=True)
        same_cols = list(old_t.columns) == list(new_t.columns)
        rec["tables"][name]["rows_2025"] = {"released": int(len(old_t)), "new": int(len(new_t)),
                                            "columns_equal": same_cols,
                                            "equal": bool(same_cols and len(old_t) == len(new_t) and old_t.equals(new_t))}
        if same_cols and len(old_t) == len(new_t) and not rec["tables"][name]["rows_2025"]["equal"]:
            rec["tables"][name]["rows_2025"]["columns_differ"] = [
                c for c in old_t.columns if not old_t[c].equals(new_t[c])]
    fits = work / "out/fits"
    for part, odir in PARTS.items():
        for n in SPLITS:
            new_p, old_p = fits / f"{part}-{n}.seeds.npz", odir / f"{part}-{n}.seeds.npz"
            if not new_p.exists():
                continue
            a, b = np.load(new_p), np.load(old_p)
            r = {"ids_equal": bool(np.array_equal(a["ids"], b["ids"])),
                 "anchor_equal": bool(np.array_equal(a["anchor"], b["anchor"])),
                 "seeds": [int(s) for s in a["seeds"]]}
            if r["ids_equal"] and list(a["seeds"]) == list(b["seeds"]):
                r["per_seed"] = {int(s): diff(a["z"][k], b["z"][k]) for k, s in enumerate(a["seeds"])}
            elif not r["ids_equal"]:                     # final mode: compare on the shared ids
                ia, ib = pd.Index(a["ids"]), pd.Index(b["ids"])
                both = ia.intersection(ib)
                r["shared_ids"] = int(len(both))
                pa, pb = ia.get_indexer(both), ib.get_indexer(both)
                r["per_seed_shared"] = {int(s): diff(a["z"][k][pa], b["z"][k][pb]) for k, s in enumerate(a["seeds"])}
            rec["seeds"][f"{part}-{n}"] = r
            if part == "HB3u":
                pa_, pb_ = pd.read_parquet(fits / f"{part}-{n}.parquet"), pd.read_parquet(odir / f"{part}-{n}.parquet")
                m = pa_[[ID, f"pred_{part}"]].merge(pb_[[ID, f"pred_{part}"]], on=ID, suffixes=("_new", "_old"))
                rec["merged"][n] = {"rows_new": int(len(pa_)), "rows_released": int(len(pb_)), "shared": int(len(m)),
                                    "parquet_sha256_equal": sha(fits / f"{part}-{n}.parquet") == sha(odir / f"{part}-{n}.parquet"),
                                    **diff(m[f"pred_{part}_new"].to_numpy(), m[f"pred_{part}_old"].to_numpy())}
    pack_p, head_p = work / "pack/HB3u-jjrank.parquet", work / "parts/head.parquet"
    if pack_p.exists() and head_p.exists():
        pk, hd = pd.read_parquet(pack_p), pd.read_parquet(head_p)
        m = hd[[ID, "pred_HB3u"]].merge(pk[[ID, "pred_HB3u"]], on=ID, how="outer", suffixes=("_head", "_pack"),
                                        indicator=True)
        a, b = m["pred_HB3u_head"].to_numpy(), m["pred_HB3u_pack"].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        rec["pack_vs_head"] = {
            "pack_sha256": sha(pack_p), "released_pack_sha256": sha(OLD / "out/pack-HB3u/full/HB3u-jjrank.parquet"),
            "rows_head": int(len(hd)), "rows_pack": int(len(pk)), "rows_by_side": m["_merge"].astype(str).value_counts().to_dict(),
            "finite_both": int(both.sum()), "finite_both_bits_equal": bool(np.array_equal(a[both], b[both])),
            "finite_head_only": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
            "finite_pack_only": int((~np.isfinite(a) & np.isfinite(b)).sum())}
    for mdir in sorted(fits.glob("*.models")):
        rec["boosters"][mdir.name] = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(mdir.iterdir())}
    out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"tables": rec["tables"], "merged": rec["merged"], "pack_vs_head": rec.get("pack_vs_head")}, indent=1))


if __name__ == "__main__":
    main()
