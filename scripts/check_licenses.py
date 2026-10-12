"""What may ship in the public GPLv3 release, and what may not.

    .venv/bin/python scripts/check_licenses.py [--list]

The prize requires a public GPLv3 repository the organisers can reproduce the
result from. That makes redistribution a real question rather than a formality:
the repository must carry only material we own or may relicense, and it must not
republish anyone else's data.

The release set is `release/release_set.py`'s, which is also what
`release/MANIFEST.tsv` is built from and what `release/check_release.py`
verifies, so the licence gate and the file gate cannot drift apart. Data under
`data/` and outputs under `artifacts/` and `submissions/` are already gitignored
and were never candidates.

Checks, each of which fails loudly rather than warning:

1. `LICENSE` is present and is verbatim GPLv3.
2. Every third-party-licensed file in the release set has a license that may be
   combined with GPLv3, and its notice is preserved.
3. No file in the release set embeds someone else's DATA. This is the check that
   actually bites: notebook cell outputs render dataframes, and a rendered
   dataframe of a third party's rows is redistribution of those rows.
4. The attribution each external source's licence requires appears in a shipped
   document: Copernicus ERA5 under CC BY 4.0, and adsb.lol under ODbL-1.0.
5. No released module IMPORTS a path that is withheld. A docstring that merely
   cites a withheld path is reported as a dangling citation and counted, not
   failed: a reader who follows it finds nothing, which is a documentation
   defect and not a licence one.

*Repaired 2026-09-12 (exp-064). `git ls-files` was split on whitespace, so the
26 tracked paths containing a space became 153 path fragments that do not
exist; the count printed 7,233 for 7,080 files and every fragment was counted
as released. It now splits on NUL. Check 5 also failed on 44 files, all of
which cite `assignments/…` in a docstring and none of which imports it — and
on this file, which names the exclusions it enforces.*
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "release"))
import release_set as RS                                          # noqa: E402
from check_release import GPLV3_SHA256, GPLV3_STRUCTURE            # noqa: E402

#: Licenses that may be combined into a GPLv3 distribution.
COMPATIBLE = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "CC-BY-4.0", "GPL-3.0"}

#: Markers of a third party's data rendered into a file.
DATA_MARKERS = ("gufi", "GUFI", "departure_runway", "arrival_runway", "FUSER")

ERA5_ATTRIBUTION = "Contains modified Copernicus Climate Change Service information"
ADSB_ATTRIBUTION = "Open Database Licence"


def git_tracked() -> list[str]:
    """Every tracked path.  NUL-separated: a path with a space is one path."""
    if (ROOT / '.git').exists():
        return RS.tracked(ROOT)
    return sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file())


def release_set() -> list[str]:
    return sorted(p for p in git_tracked() if RS.is_released(p))


def check_license_text(fail):
    p = ROOT / "LICENSE"
    if not p.exists():
        return fail("LICENSE is missing; the prize requires a public GPLv3 repository")
    blob = p.read_bytes()
    try:
        t = blob.decode("utf-8")
    except UnicodeDecodeError:
        return fail("LICENSE is not UTF-8 canonical GPLv3 text")
    missing = [what for marker, what in GPLV3_STRUCTURE if marker not in t]
    if missing:
        fail("LICENSE is missing GPLv3 structure: " + ", ".join(missing))
    lines = len(t.splitlines())
    if lines != 674:
        fail(f"LICENSE has {lines} lines, not 674")
    digest = hashlib.sha256(blob).hexdigest()
    if digest != GPLV3_SHA256:
        fail(f"LICENSE is not the canonical GPLv3 text "
             f"({digest[:16]}... != {GPLV3_SHA256[:16]}...)")
    if not missing and lines == 674 and digest == GPLV3_SHA256:
        print(f"  LICENSE: verbatim GPLv3, {lines} lines, sha256 {digest}")


def check_third_party(rel, fail):
    """Third-party notices inside the release set."""
    found = []
    for f in rel:
        if Path(f).name.upper().startswith("LICENSE"):
            t = (ROOT / f).read_text(errors="ignore")
            lic = ("MIT" if "Permission is hereby granted, free of charge" in t
                   else "GPL-3.0" if "GNU GENERAL PUBLIC LICENSE" in t else "UNKNOWN")
            found.append((f, lic))
    for f, lic in found:
        ok = lic in COMPATIBLE
        print(f"  {'ok ' if ok else 'BAD'} {f}: {lic}")
        if not ok:
            fail(f"{f} carries an unrecognised license ({lic})")
    return found


def check_embedded_data(rel, fail):
    """A rendered dataframe of someone else's rows is redistribution of those rows."""
    bad = []
    for f in rel:
        if not f.endswith(".ipynb"):
            continue
        try:
            nb = json.loads((ROOT / f).read_text())
        except Exception:
            continue
        for i, c in enumerate(nb.get("cells", [])):
            for o in c.get("outputs", []):
                txt = "".join(o.get("text", []))
                for k, v in (o.get("data") or {}).items():
                    if k.startswith("text"):
                        txt += "".join(v) if isinstance(v, list) else str(v)
                if any(m in txt for m in DATA_MARKERS):
                    bad.append((f, i))
                    break
    if bad:
        for f, i in bad[:10]:
            print(f"  BAD {f} cell {i}: output embeds third-party data")
        fail(f"{len(bad)} notebook cells in the release set embed third-party data")
    else:
        print(f"  no third-party data embedded in {sum(f.endswith('.ipynb') for f in rel)} "
              f"released notebooks")


def check_attribution(rel, fail):
    for label, marker, licence in (
            ("Copernicus ERA5", ERA5_ATTRIBUTION, "CC BY 4.0"),
            ("adsb.lol", ADSB_ATTRIBUTION, "ODbL-1.0"),
            ("EUROCONTROL", "European Organisation for the Safety of Air Navigation (EUROCONTROL)", "source copyright terms"),
    ):
        where = [f for f in rel
                 if f.endswith(".md") and marker in (ROOT / f).read_text(errors="ignore")]
        if where:
            print(f"  {label} {licence} attribution present in {where[0]}"
                  + (f" (and {len(where) - 1} more)" if len(where) > 1 else ""))
        else:
            fail(f"the {label} attribution line required by {licence} "
                 f"is in no released document")


IMPORT_OF_WITHHELD = re.compile(
    r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_.]*)", re.M)


def check_imports(rel, fail):
    """An IMPORT of a withheld tree breaks the release; a citation does not."""
    withheld_tops = {x.rstrip("/").split("/")[0] for x in RS.EXCLUDE_PREFIX}
    imports, citations = [], []
    for f in rel:
        if not f.endswith(".py"):
            continue
        t = (ROOT / f).read_text(errors="ignore")
        for mod in IMPORT_OF_WITHHELD.findall(t):
            if mod.split(".")[0] in withheld_tops:
                imports.append((f, mod))
        if f == "scripts/check_licenses.py":
            continue                     # it names the exclusions it enforces
        for x in RS.EXCLUDE_PREFIX:
            if x in t:
                citations.append((f, x))
    if imports:
        for f, m in imports:
            print(f"  BAD {f} imports withheld module {m}")
        fail(f"{len(imports)} released module(s) import a withheld path")
    else:
        print(f"  no released module imports a withheld path "
              f"({sum(f.endswith('.py') for f in rel)} modules read)")
    if citations:
        print(f"  {len(citations)} released file(s) CITE a withheld path in prose "
              f"(dangling citations, not a licence failure); first five:")
        for f, x in citations[:5]:
            print(f"      {f} -> {x}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the release file set and exit")
    a = ap.parse_args()
    rel = release_set()
    if a.list:
        print("\n".join(rel))
        return 0
    failures = []
    fail = failures.append
    all_tracked = git_tracked()
    print(f"tracked files {len(all_tracked)}, released {len(rel)}, "
          f"withheld {len(all_tracked) - len(rel)}")
    print("\n1. license text");        check_license_text(fail)
    print("\n2. third-party notices"); check_third_party(rel, fail)
    print("\n3. embedded data");       check_embedded_data(rel, fail)
    print("\n4. external-source attribution"); check_attribution(rel, fail)
    print("\n5. withheld imports");     check_imports(rel, fail)
    print("\nwithheld from the release:")
    for x in RS.EXCLUDE_PREFIX:
        n = sum(1 for f in all_tracked if f.startswith(x))
        print(f"  {x}  ({n} files)")
    for seg in sorted(RS.EXCLUDE_DIR_SEGMENT):
        n = sum(1 for f in all_tracked if seg in f.split("/")[:-1])
        if n:
            print(f"  */{seg}/  ({n} files)")
    for suf in RS.EXCLUDE_SUFFIX:
        n = sum(1 for f in all_tracked if f.endswith(suf) and RS.is_released(f) is False)
        if n:
            print(f"  *{suf}  ({n} files)")
    print(f"  named files ({len(RS.EXCLUDE_NAME)} names, "
          f"{sum(1 for f in all_tracked if f.split('/')[-1] in RS.EXCLUDE_NAME)} files)")
    print(f"  exact exclusions ({len(RS.EXCLUDE_EXACT | RS.ROW_LEVEL_RECORDS | RS.PRIVATE_RECORDS)} paths)")
    if failures:
        print(f"\nFAIL — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS — the release set may be published under GPLv3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
