"""exp-186: a final EXP148_KINPUT root for exp-164 mlp008/prepare.py, in the board layout, from links only.

Called by make_kinput_final.sh. The final prepare reads the root through these lookups:
  fit_exp134.py   exp-153 learner_wp.py:91 setup(): first rglob match; its folder goes on sys.path and gives
                  proxy_features.py (import, :93)
  fit.py          fit_exp134.py:92-93 base_fit_module() -> proxy_features.py:48-52 exact_file (exactly one)
  dataset.py, data.py, model.py   exp-130 fit.py:64-77 setup_modules() -> locate_file :49-53 (exactly one each);
                  dataset.py's folder goes on sys.path (dataset, features and model are imported from it);
                  prc.data binds the snapshot's prc, which learner_wp.py:46-47 imported first
  month_<m>.parquet, 2025-01..12 and the four final 2026 months
                  fit.py:79-86 month_frame (prepare.py:233, :256), locate_file (prepare.py:235, :257)
  proxy_departures.parquet, proxy_arrivals.parquet, points_expected.json, stand_geometry.csv
                  learner_wp.py:122-124 raw_months_cached (cache key, prepare.py:260), proxy_features.py:600-604 build_all,
                  prepare.py:262
  points_*.parquet  proxy_features.py:283-300 _day_point_files (build_all, on a cache miss): every rglob match,
                  one per day, the day set equal to points_expected.json, bytes and SHA-256 equal to each entry
Path.rglob does not enter a symlinked folder (default; tested on Python 3.9.6 and 3.11.12), so every folder is real
and every file of the root is a symlink. Sources:
  prc-exp130-clean-input/<rel>  <- <final>/exp-130/package-final/<rel>       (MANIFEST rows)
  prc-exp134-input/<name>       <- <final>/exp-134/input-final/[code/]<name> (MANIFEST rows; code/ flattened as
                                   on the board)
  prc-harvest-2026-02/, -06/    <- <final>/kinput/prc-harvest-2026-0x/points_*.parquet (MANIFEST rows)
  prc-harvest-<x>/<rel>         <- <board>/prc-harvest-<x>/<rel>, the point file of each other day of the final
                                   points_expected.json, with the bytes and SHA-256 of its entry
Each final file must equal its MANIFEST row (bytes, SHA-256). After the links the script repeats every lookup on
the root and stops unless each resolves once, to its checked source. Writes <root>/KINPUT-ROOT.json (links,
lookups, board files with no final counterpart in the root). Stdlib only.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PACKAGES = {"prc-exp130-clean-input": "exp-130/package-final", "prc-exp134-input": "exp-134/input-final"}
HARVESTS = ("prc-harvest-2026-02", "prc-harvest-2026-06")
MONTHS = tuple(f"2025-{m:02d}" for m in range(1, 13)) + ("2026-01", "2026-02", "2026-06", "2026-07")
PROXY = ("proxy_departures.parquet", "proxy_arrivals.parquet", "points_expected.json", "stand_geometry.csv")
NAMED = ("fit_exp134.py", "proxy_features.py", "fit.py", "dataset.py", "data.py", "model.py", "features.py",
         *PROXY, *(f"month_{m}.parquet" for m in MONTHS))


def stop(msg: str) -> None:
    raise SystemExit(f"make_kinput_final: {msg}; stop")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--manifest", type=Path, required=True, help="exp-185 MANIFEST.tsv")
    ap.add_argument("--final", type=Path, required=True, help="finaldata root, absolute as in the MANIFEST")
    ap.add_argument("--board", type=Path, required=True, help="board kinput root, absolute")
    a = ap.parse_args()
    root = a.root
    if not (a.final.is_absolute() and a.board.is_absolute() and root.is_absolute()):
        stop("root, --final and --board must be absolute paths")
    if root.exists() or root.is_symlink():
        stop(f"{root} exists")
    t0 = time.time()
    rows = {}
    with open(a.manifest, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["path"]] = r
    links, checked = {}, {}            # root rel path -> source; source -> SHA-256 it was checked against

    def link(src: Path, dest: str, digest: str) -> None:
        if dest in links:
            stop(f"two sources for {dest}")
        links[dest] = src
        checked[str(src)] = digest

    def take(src: Path, dest: str) -> None:
        r = rows.get(str(src))
        if r is None:
            stop(f"{src} has no MANIFEST row")
        if not src.is_file():
            stop(f"{src} is missing")
        if src.stat().st_size != int(r["bytes"]) or sha(src) != r["sha256"]:
            stop(f"{src} differs from its MANIFEST row")
        link(src, dest, r["sha256"])

    for top, sub in PACKAGES.items():
        base = a.final / sub
        names = sorted(p for p in rows if p.startswith(str(base) + "/"))
        if not names:
            stop(f"no MANIFEST rows under {base}")
        for p in names:
            rel = Path(p).relative_to(base)
            if top == "prc-exp134-input" and rel.parts[0] == "code":
                rel = Path(*rel.parts[1:])
            take(Path(p), f"{top}/{rel}")
    final_days = {}
    for top in HARVESTS:
        base = a.final / "kinput" / top
        names = sorted(p for p in rows if p.startswith(str(base) + "/points_") and p.endswith(".parquet"))
        if not names:
            stop(f"no point files of {base} in the MANIFEST")
        for p in names:
            day = Path(p).name[7:17]
            if day in final_days:
                stop(f"two final point files of {day}")
            take(Path(p), f"{top}/{Path(p).name}")
            final_days[day] = Path(p)

    # the days of the final points_expected.json: the 16 months, every calendar day, one record each
    expected = json.loads(links["prc-exp134-input/points_expected.json"].read_text())
    if tuple(sorted(expected["months"])) != MONTHS:
        stop(f"points_expected.json months {sorted(expected['months'])} differ from {MONTHS}")
    want = {}
    for month, recs in expected["months"].items():
        if len(recs) != calendar.monthrange(int(month[:4]), int(month[5:]))[1]:
            stop(f"points_expected.json {month}: {len(recs)} days")
        for r in recs:
            if r["day"] in want or r["file"] != f"points_{r['day']}.parquet" or r["day"][:7] != month:
                stop(f"points_expected.json record {r['day']} {r['file']}")
            want[r["day"]] = r
    for day, p in final_days.items():
        if day not in want:
            stop(f"{p}: day {day} is not in points_expected.json")
        if p.stat().st_size != int(want[day]["bytes"]) or checked[str(p)] != want[day]["sha256"]:
            stop(f"{p}: differs from its points_expected.json entry")
    cands = {}
    for folder in sorted(a.board.glob("prc-harvest-*")):
        if folder.is_dir():
            for p in sorted(folder.rglob("points_*.parquet")):
                cands.setdefault(p.name[7:17], []).append(p)
    board_days, same_sha, off_slug = {}, {}, []
    for day in sorted(set(want) - set(final_days)):
        r = want[day]
        slug = r.get("kernel_slug", "").rsplit("/", 1)[-1]
        hits = [p for p in cands.get(day, []) if p.stat().st_size == int(r["bytes"]) and sha(p) == r["sha256"]]
        if not hits:
            stop(f"no board point file of {day} with the bytes and SHA-256 of points_expected.json")
        hits.sort(key=lambda p: (p.relative_to(a.board).parts[0] != slug, str(p)))   # the kernel's folder first
        if len(hits) > 1:
            same_sha[day] = [str(p) for p in hits]
        if hits[0].relative_to(a.board).parts[0] != slug:
            off_slug.append(day)
        board_days[day] = hits[0]
        link(hits[0], str(hits[0].relative_to(a.board)), r["sha256"])

    # build: real folders, file symlinks
    root.mkdir(parents=True)
    for dest, src in sorted(links.items()):
        d = root / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, d)

    # repeat every lookup on the root
    for d in [root, *root.rglob("*")]:
        if d.is_dir() and d.is_symlink():
            stop(f"{d} is a folder link")
    lookups = {}
    for name in NAMED:
        hits = sorted(p for p in root.rglob(name) if p.is_file())
        if len(hits) != 1:
            stop(f"lookup {name} gives {len(hits)} files")
        src = os.readlink(hits[0])
        if src not in checked:
            stop(f"lookup {name} resolves to {src}, which no MANIFEST row checked")
        lookups[name] = {"path": str(hits[0].relative_to(root)), "source": src, "sha256": checked[src]}
    first = next(root.rglob("fit_exp134.py"))
    if first.parent != root / "prc-exp134-input" or not (first.parent / "proxy_features.py").is_file():
        stop("the fit_exp134.py folder does not hold proxy_features.py")
    by_day = {}
    for p in root.rglob("points_*.parquet"):
        day = p.name[7:17]
        if day in by_day:
            stop(f"two point files of {day} in the root")
        if checked.get(os.readlink(p)) != want.get(day, {}).get("sha256"):
            stop(f"{p} does not resolve to the checked file of {day}")
        by_day[day] = p
    if set(by_day) != set(want):
        stop(f"point days differ from points_expected.json: {sorted(set(want) ^ set(by_day))[:4]}")

    # board files with no final counterpart in the root; the run reads the root only
    board_only = []
    for p in sorted(a.board.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and str(p.relative_to(a.board)) not in links:
            board_only.append({"path": str(p.relative_to(a.board)), "name_looked_up_by_final_prepare":
                               p.name in NAMED or (p.name.startswith("points_") and p.name.endswith(".parquet"))})
    rec = {"what": "exp-186 final EXP148_KINPUT root (links only; folders real, files symlinks)",
           "root": str(root), "manifest": {"path": str(a.manifest), "sha256": sha(a.manifest)},
           "final": str(a.final), "board": str(a.board),
           "links": {d: str(s) for d, s in sorted(links.items())}, "link_count": len(links),
           "final_files_checked_against_manifest": len(links) - len(board_days),
           "points": {"days": len(want), "final_days": len(final_days), "board_days": len(board_days),
                      "board_days_outside_their_kernel_folder": off_slug, "board_days_with_two_equal_files": same_sha},
           "lookups": lookups, "board_files_not_in_root": board_only,
           "code_sha256": sha(Path(__file__)), "seconds": round(time.time() - t0, 1)}
    (root / "KINPUT-ROOT.json").write_text(json.dumps(rec, indent=1) + "\n")
    looked = [b["path"] for b in board_only if b["name_looked_up_by_final_prepare"]]
    print(json.dumps({"event": "kinput-ready", "root": str(root), "links": len(links),
                      "final_files_checked_against_manifest": rec["final_files_checked_against_manifest"],
                      "point_days": len(want), "final_point_days": len(final_days),
                      "board_point_days": len(board_days), "board_days_outside_their_kernel_folder": len(off_slug),
                      "lookups": {k: v["path"] for k, v in lookups.items()},
                      "board_files_not_in_root": len(board_only),
                      "board_files_not_in_root_by_top": {t: sum(b["path"].split("/")[0] == t for b in board_only)
                                                         for t in sorted({b["path"].split("/")[0] for b in board_only})},
                      "board_files_not_in_root_with_a_looked_up_name": looked[:10],
                      "seconds": rec["seconds"]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
