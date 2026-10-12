"""Build the refit tree: the source tree in which the final chain can run again from its inputs.

    python release/make_refit_tree.py <out dir> [--unit NAME ...]

The refit tree is a copy of the released source (every file of release/MANIFEST.tsv outside refit-src/ and
refit-pins/), then the refit-src/ overlay at the same repo paths, then, for each --unit, the files of
refit-pins/<unit>/files/ (the bytes one held step ran; see refit-pins/<unit>/README.md). Each copied file is checked
against its SHA-256 in release/MANIFEST.tsv, and each overlay file also against release/REFIT-SOURCES.json. The out
directory must not exist. The script writes <out dir>/REFIT-TREE.sha256 (sha256sum format, sorted by path) and
<out dir>/REFIT-TREE.json (the units, the overlay paths, and the SHA-256 of the list).

Our final runs used the same form: a copy of the export source with the newer source files laid over it. Paths
outside the tree come from the root variables of release/REFIT-SOURCES.json (`roots`); set each one to your folder
before a step runs. docs/REPRODUCE.md gives the steps and their order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ("refit-src/", "refit-pins/")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_checked(src: Path, dst: Path, want: str) -> None:
    got = sha256(src)
    if got != want:
        raise SystemExit(f"{src.relative_to(ROOT)}: SHA-256 {got} is not the listed {want}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--unit", action="append", default=[], help="a refit-pins unit to lay over the tree")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; the refit tree is written once")
    manifest = {}
    for line in (ROOT / "release/MANIFEST.tsv").read_text().splitlines()[1:]:
        path, digest, _size, _role = line.split("\t")
        manifest[path] = digest
    refit = json.loads((ROOT / "release/REFIT-SOURCES.json").read_text())
    shipped = {f["published"]: f for f in refit["files"]}
    units = {u["unit"] for u in refit["units"]}
    unknown = sorted(set(a.unit) - units)
    if unknown:
        raise SystemExit(f"unknown unit {unknown}; units: {sorted(units)}")

    base = sorted(p for p in manifest if not p.startswith(OVERLAY))
    for p in base:
        copy_checked(ROOT / p, a.out / p, manifest[p])
    layers = [("refit-src", None)] + [(f"refit-pins/{u}/files", u) for u in a.unit]
    laid = {}
    for prefix, unit in layers:
        for pub, f in sorted(shipped.items()):
            if f["unit"] != unit or f["path"] is None or not pub.startswith(prefix + "/"):
                continue
            if manifest.get(pub) != f["sha256_shipped"]:
                raise SystemExit(f"{pub}: MANIFEST.tsv and REFIT-SOURCES.json disagree")
            copy_checked(ROOT / pub, a.out / f["path"], f["sha256_shipped"])
            laid[f["path"]] = pub
    rows = sorted((p.relative_to(a.out).as_posix(), sha256(p)) for p in a.out.rglob("*") if p.is_file())
    listing = "".join(f"{d}  {p}\n" for p, d in rows)
    (a.out / "REFIT-TREE.sha256").write_text(listing)
    record = {"units": a.unit, "files": len(rows), "base_files": len(base), "overlay": laid,
              "list_sha256": hashlib.sha256(listing.encode()).hexdigest()}
    (a.out / "REFIT-TREE.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("units", "files", "base_files", "list_sha256")}))
    print(f"{len(laid)} overlay files; set the roots of release/REFIT-SOURCES.json before a step runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
