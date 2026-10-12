"""Verify a tree against `release/MANIFEST.tsv`, and gate what may ship.

    .venv/bin/python release/check_release.py            # check this checkout
    .venv/bin/python release/check_release.py --root DIR --contract release/MANIFEST.tsv
    .venv/bin/python release/check_release.py --json     # machine record

Exit status 0 only when every check passes.  The checks, each of which fails
loudly rather than warning:

  A  MANIFEST.tsv parses, is sorted, and has no duplicate path.
  B  Every manifest path exists under the root.                 (missing file)
  C  Every manifest path hashes to its recorded SHA-256.        (changed file)
  D  The root carries no file the manifest does not name.       (unexpected file)
  E  No released file carries a row-level record of one of the organisers'
     movements.  Schema-aware, after round 1 finding 2: a JSON `"MVT_ID_mvt"`
     followed by a number OR by a list, and a CSV or TSV header carrying a
     movement id written as an integer or exact decimal, a target alias (`y`,
     `y_true`, `TAXITIME_SEC_mvt`) or
     flight-level detail (callsign, stand, runway, type) beside an id; and any
     file that names a movement-id field and carries three or more distinct
     nine-digit movement ids, in any format.  Only two synthetic fixtures skip this free-text heuristic; JSON and
     table detectors still run on them.
  F  No released file carries a credential-shaped string.
  G  The manifest names exactly the release set, in both directions: no
     manifest path breaks `release_set.is_released`, and no path the rule
     releases is left out of the manifest.  The one exception is the manifest
     itself, which ships but cannot record its own digest.
  H  LICENSE is present and hashes to the canonical GPLv3 digest.
  I  No released file carries private account, Kaggle object, or transport
     route metadata.
  J  Every released path in this repository matches its committed content --
     the manifest describes a COMMIT, not somebody's working tree.
  K  No public links point to excluded paths; narrative has no private decision citations.
     Top-level inline repo-path citations must name files or directories in the export.
  L  Imported full text stays within the rewrite scope; every listed line equals its reviewed text.

Checks E, F and I stream every released file, including the manifest, through
the same bounded-memory content scanner. File suffix and size do not bypass it.
Unreadable files fail closed. The one binary kind is a PNG figure directly in
docs/writeup/figures/: it is parsed as a PNG (signature, chunk lengths and CRCs,
IHDR first, IEND last with nothing after it, listed chunk types at their
standard lengths, one image stream that inflates to exactly the declared size),
and the bytes of every chunk except the image data go through the same scanners
as any text file, with the decoded image samples (after the PNG row filters are
undone). Gate K reads that same text. Any other binary file still fails closed. Gates G (reverse membership) and J need the source
Git checkout; a built tree uses the separately trusted immutable manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import json
import os
import re
import sys
import tokenize
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from release_set import (ROOT, EXCLUDE_DIR_SEGMENT, EXCLUDE_NAME,        # noqa: E402
                         EXCLUDE_PREFIX, EXCLUDE_SUFFIX, MANIFEST_PATH,
                         is_released, manifest_set)
from content_policy import MovementScanner, MOVEMENT_FREE_TEXT_ALLOWED
from reference_policy import scan_references
from topdoc_paths import missing_path_citations
from rewrite_hygiene import inspect_imported, inspect_rewritten

MANIFEST = HERE / "MANIFEST.tsv"          # replaced by the root's own, below

#: Policy describes route shapes; it contains no real private identity.
#: Values in source, prose, Markdown, YAML and JSON use the same detectors.
PRIVATE_URL = re.compile(
    r"https?://(?:chatgpt\.com/c/|chat\.openai\.com/c/|"
    r"drive\.google\.com/|docs\.google\.com/(?:document|spreadsheets|presentation)/d/|"
    r"(?:www\.)?kaggle\.com/(?:datasets/|code/|kernels/)?)[A-Za-z0-9_/-]+", re.I)
PRIVATE_FLAG = r"\bis_private[\"'` ]{0,8}[:=][\"'` ]{0,8}true\b"
PRIVATE_OBJECT_REF = (r"\b(?:id|ref|dataset_ref|kernel_ref)[\"'` ]{0,8}[:=]"
                      r"[\"'` ]{0,8}[a-z0-9_-]+/[a-z0-9_-]+")
PRIVATE_ROUTE = [
    ("kaggle_private_metadata", re.compile(
        PRIVATE_FLAG + r".{0,300}" + PRIVATE_OBJECT_REF + "|" +
        PRIVATE_OBJECT_REF + r".{0,300}" + PRIVATE_FLAG, re.I | re.S)),
    # Account-specific OpenSky storage names are private in every textual form:
    # bare prose, assignments/defaults, structured values, shell and URLs.
    # The detector intentionally does not require an s3:// prefix and has no
    # exemption for a data-bucket spelling.
    ("private_storage_identifier", re.compile(
        r"(?i)\bprc[-_]20(?:2[6-9]|[3-9]\d)[-_][a-z0-9_-]+\b")),
    ("aws_profile_identifier", re.compile(r"(?i)\bopensky[-_]prc[-_]\d{4}\b")),
    ("kaggle_slug", re.compile(r"(?i)(?<![./\w$])[a-z0-9_-]+/prc[-_][a-z0-9_-]+(?![\w/.-])")),
    ("private_dataset_slug", re.compile(r"(?i)\bprc[-_]\d{4}[-_]data\b")),
    ("kaggle_account_name", re.compile(
        r"(?i)\b(?:username|kaggle[ _-](?:account|user))['\"` ]{0,8}"
        r"(?:[:=]|\|(?= )|is\b)['\"` ]{0,8}[a-z0-9_-]+")),
    ("kaggle_owner", re.compile(r"(?i)\bowner['\"` ]{0,8}[:=][ ]*(?:['\"`][a-z0-9_-]+['\"`]|(?!(?:owner|int)(?![a-z0-9_-]))[a-z0-9_-]+)")),
    ("kaggle_reference", re.compile(
        r"(?i)\b(?:ref|dataset[ _-]?(?:ref|slug|source)|kernel[ _-]?(?:ref|slug)|source_ref)"
        r"['\"` ]{0,8}(?:[:=]|\|(?= )|is\b)['\"` ]{0,8}[a-z0-9_-]+/[a-z0-9_-]+")),
    ("private_object_id", re.compile(
        r"(?i)\b(?:dataset|kernel|object|version)[ _-]?id['\"` ]{0,8}"
        r"(?:[:=]|\|(?= )|is\b)['\"` ]{0,8}(?:[0-9]{3,}|[a-f0-9-]{12,})")),
    ("transport_receipt", re.compile(
        r"(?i)\b(?:transport[ _-](?:status|receipt)|launch[ _-]gate|remote[ _-]custody)"
        r"['\"` ]{0,8}(?:[:=]|\|(?= )|is\b)")),
    ("remote_custody", re.compile(
        r"(?i)(?:\b(?:remote|laptop):/(?:home|mnt|srv|tmp)/|"
        r"/home/[a-z0-9_-]+/(?:repos|prc|competition)[^\s'\"`]*|"
        r"ssh\s+(?:remote|laptop)\b)")),
]
#: The verbatim GNU GPLv3 text as this repository carries it: 674 lines, 35,147
#: bytes, the FSF distribution whose header URL is `http://fsf.org/` (the later
#: one says `https://`, which is the whole 2-byte difference from the 35,149-byte
#: copy).  Round 1 non-blocking 2: two phrases and a line count let altered text
#: of the same length pass, so the whole file is compared.  The digest is not
#: taken on trust -- GPLV3_STRUCTURE below states what was verified in the text
#: itself before it was pinned.
GPLV3_SHA256 = "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"
GPLV3_STRUCTURE = (
    ("GNU GENERAL PUBLIC LICENSE", "the title"),
    ("Version 3, 29 June 2007", "the version line"),
    ("Copyright (C) 2007 Free Software Foundation, Inc.", "the FSF notice"),
    ("TERMS AND CONDITIONS", "the terms"),
    ("END OF TERMS AND CONDITIONS", "the end of the terms"),
    ("How to Apply These Terms to Your New Programs", "the appendix"),
)
SECRET = [
    ("aws_access_key_id", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("aws_secret_access_key",
     re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{30,}")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("kaggle_key", re.compile(r'"key"\s*:\s*"[0-9a-f]{20,}"')),
    ("bearer_token",
     re.compile(r"(?i)(authorization|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9._\-]{24,}")),
    ("inline_secret",
     re.compile(r"(?i)\b(api[_-]?key|secret|token|passwd|password)\b\s*[=:]\s*"
                r"['\"][^'\"\s]{12,}['\"]")),
    ("cdsapirc_key", re.compile(r"(?i)^key\s*:\s*[0-9a-f-]{20,}", re.M)),
]


#: PNG figures: the folder that may hold them, and what a figure may contain.
#: Review 14 (item 3) found four places where a crafted figure could carry text
#: that no scanner read: padding after the IHDR data, a pHYs or a bKGD chunk of
#: any length, and IDAT data after the end of the image stream. Every chunk now
#: has its standard length, each one except IDAT and tEXt occurs at most once,
#: the image stream must inflate to exactly the size the header declares, and
#: the bytes of every chunk except IDAT reach the scanners.
FIGURE_DIR = "docs/writeup/figures/"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_CHUNKS = frozenset({"IHDR", "PLTE", "IDAT", "IEND", "tRNS", "pHYs", "sRGB",
                        "gAMA", "cHRM", "bKGD", "tEXt"})
PNG_REPEATABLE = frozenset({"IDAT", "tEXt"})
PNG_LENGTH = {"IHDR": 13, "IEND": 0, "pHYs": 9, "sRGB": 1, "gAMA": 4, "cHRM": 32}
#: Per colour type: samples per pixel, allowed bit depths, bKGD and tRNS lengths
#: (None: the chunk is not allowed; tRNS of a palette image is checked apart).
PNG_COLOUR = {0: (1, (1, 2, 4, 8, 16), 2, 2), 2: (3, (8, 16), 6, 6),
              3: (1, (1, 2, 4, 8), 1, None), 4: (2, (8, 16), 2, None),
              6: (4, (8, 16), 6, None)}
PNG_TEXT_LIMIT = 1 << 16
PNG_IMAGE_LIMIT = 1 << 26


def is_figure(rel: str) -> bool:
    """A PNG directly in the figure folder: the only binary file that may ship."""
    name = rel[len(FIGURE_DIR):]
    return rel.startswith(FIGURE_DIR) and name.endswith(".png") and "/" not in name


def png_header(data: bytes) -> tuple[int, int, int]:
    """Colour type, palette-free sample bits per row and rows; fails on a bad header."""
    width, height = int.from_bytes(data[:4], "big"), int.from_bytes(data[4:8], "big")
    depth, colour, compression, filtering, interlace = data[8:13]
    if colour not in PNG_COLOUR or depth not in PNG_COLOUR[colour][1]:
        raise ValueError
    if not (0 < width <= 0x7FFFFFFF and 0 < height <= 0x7FFFFFFF) or compression or filtering or interlace:
        raise ValueError
    return colour, height * (1 + (width * depth * PNG_COLOUR[colour][0] + 7) // 8), height


def png_text(path: Path, rel: str, read_size: int = 1 << 20) -> str:
    """The bytes of every chunk except IDAT, as text, after a full structural check; fails closed."""
    texts: list[str] = []
    try:
        with path.open("rb") as src:
            if src.read(8) != PNG_SIGNATURE:
                raise ValueError
            seen: list[str] = []
            size, colour, expected, inflated, palette = 0, None, 0, 0, 0
            inflate = zlib.decompressobj()
            while True:
                head = src.read(8)
                if len(head) != 8:
                    raise ValueError
                length = int.from_bytes(head[:4], "big")
                kind = head[4:].decode("ascii")
                if length > 0x7FFFFFFF or kind not in PNG_CHUNKS or (not seen) != (kind == "IHDR"):
                    raise ValueError
                if kind in seen and (kind not in PNG_REPEATABLE or (kind == "IDAT" and seen[-1] != "IDAT")):
                    raise ValueError
                if "IDAT" in seen and kind not in ("IDAT", "tEXt", "IEND"):
                    raise ValueError
                if length != PNG_LENGTH.get(kind, length):
                    raise ValueError
                if kind == "bKGD" and length != PNG_COLOUR[colour][2]:
                    raise ValueError
                if kind == "PLTE" and (colour in (0, 4) or not 0 < length <= 768 or length % 3):
                    raise ValueError
                if kind == "tRNS" and (length != PNG_COLOUR[colour][3] if colour != 3
                                       else not 0 < length <= palette):
                    raise ValueError
                crc, left, data = zlib.crc32(head[4:]), length, b""
                while left:
                    block = src.read(min(left, read_size))
                    if not block:
                        raise ValueError
                    crc, left = zlib.crc32(block, crc), left - len(block)
                    if kind != "IDAT":
                        size += len(block)
                        if size > PNG_TEXT_LIMIT:
                            raise ValueError
                        data += block
                        continue
                    if inflate.eof:
                        raise ValueError
                    tail = block
                    while True:
                        room = min(read_size, expected - inflated + 1)
                        out = inflate.decompress(tail, room)
                        inflated += len(out)
                        if inflated > expected or inflate.unused_data:
                            raise ValueError
                        tail = inflate.unconsumed_tail
                        if not tail and len(out) < room:
                            break
                if src.read(4) != crc.to_bytes(4, "big"):
                    raise ValueError
                if kind == "IHDR":
                    colour, expected, _ = png_header(data)
                    if expected > PNG_IMAGE_LIMIT:
                        raise ValueError
                if kind == "PLTE":
                    palette = length // 3
                seen.append(kind)
                if data:
                    texts.append(data.decode("latin-1").replace("\0", ": "))
                if kind == "IEND":
                    if src.read(1) or not inflate.eof or inflated != expected or (colour == 3 and not palette):
                        raise ValueError
                    return "\n".join(texts)
    except (OSError, UnicodeError, ValueError, zlib.error):
        raise ValueError(f"cannot scan released file: {rel}")


def png_samples(path: Path, rel: str) -> str:
    """The decoded samples of a figure that passed png_text, as latin-1 text; fails closed."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError
            image.load()
            return image.tobytes().decode("latin-1")
    except Exception:
        raise ValueError(f"cannot scan released file: {rel}")


def figure_text(path: Path, rel: str) -> str:
    """Everything a figure carries, as text: its chunk bytes, then its decoded samples."""
    return png_text(path, rel) + "\n" + png_samples(path, rel)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan_readable(path: Path, rel: str, read_size: int = 1 << 20) -> tuple[list[dict], list[str], list[str]]:
    """Scan any-size text with bounded memory; exceptions affect free text only."""
    if read_size < 1:
        raise ValueError("read_size must be positive")
    scanner = MovementScanner(rel, path.suffix)
    secret_hits: set[str] = set()
    private_hits: set[str] = set()
    overlap = ""
    if is_figure(rel):
        text = figure_text(path, rel)
        if text:
            scanner.feed(text)
        secret_hits.update(name for name, rx in SECRET if rx.search(text))
        private_hits.update(name for name, rx in PRIVATE_ROUTE if rx.search(text))
        if PRIVATE_URL.search(text):
            private_hits.add("account_bound_url")
        return scanner.findings(rel), sorted(secret_hits), sorted(private_hits)
    try:
        with path.open("rb") as probe:
            prefix = probe.read(4)
        encoding = ("utf-32" if prefix in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff") else
                    "utf-16" if prefix[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig")
        with path.open("r", encoding=encoding) as src:
            while True:
                chunk = src.read(read_size)
                if not chunk:
                    break
                scanner.feed(chunk)
                text = overlap + chunk
                secret_hits.update(name for name, rx in SECRET if rx.search(text))
                private_hits.update(name for name, rx in PRIVATE_ROUTE if rx.search(text))
                if PRIVATE_URL.search(text):
                    private_hits.add("account_bound_url")
                overlap = text[-1024:]
    except (OSError, UnicodeError):
        raise ValueError(f"cannot scan released file: {rel}")
    return scanner.findings(rel), sorted(secret_hits), sorted(private_hits)


def read_manifest(path: Path) -> list[tuple[str, str, int, str]]:
    lines = path.read_text().splitlines()
    if not lines or lines[0].split("\t") != ["path", "sha256", "bytes", "role"]:
        raise SystemExit("bad header")
    rows = []
    for n, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 4:
            raise SystemExit(f"manifest line {n}: expected 4 fields, got {len(parts)}")
        rows.append((parts[0], parts[1], int(parts[2]), parts[3]))
    return rows


def walk(root: Path) -> list[str]:
    """Every file under `root`, repo-relative, ignoring .git."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for f in filenames:
            out.append(str(Path(dirpath, f).relative_to(root)))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="tree to check (default: this repository)")
    ap.add_argument("--contract", default=None,
                    help="trusted MANIFEST.tsv for a built tree (required with --root)")
    ap.add_argument("--json", action="store_true", help="print a JSON record")
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else ROOT
    built = a.root is not None          # a built release, not the working repo
    if built and not a.contract:
        raise SystemExit("--root requires --contract from the intended source revision")

    # Round 2 finding 3: `--root` read the manifest beside this script, so a
    # tree could be checked against a contract it does not contain.
    manifest = (root / "release/MANIFEST.tsv") if built else MANIFEST
    if not manifest.is_file():
        raise SystemExit(f"no manifest at {manifest}")
    report_manifest = str(manifest)
    rows = read_manifest(manifest)
    if built:
        contract = Path(a.contract).resolve()
        if not contract.is_file():
            raise SystemExit(f"trusted contract absent: {contract}")
        if manifest.read_bytes() != contract.read_bytes():
            raise SystemExit("built manifest differs from trusted contract")
    paths = [p for p, _, _, _ in rows]
    report: dict[str, object] = {"root": str(root), "manifest": report_manifest,
                                 "manifest_files": len(rows), "built_release": built}
    fail: list[str] = []

    # A  manifest shape
    if paths != sorted(paths):
        fail.append("A: MANIFEST.tsv is not sorted by path")
    if len(set(paths)) != len(paths):
        fail.append("A: MANIFEST.tsv has a duplicate path")

    # B, C  present and unchanged
    missing, changed = [], []
    for p, want, size, _ in rows:
        f = root / p
        if not f.is_file():
            missing.append(p)
            continue
        if f.stat().st_size != size or sha256(f) != want:
            changed.append(p)
    report["missing"] = missing
    report["changed"] = changed
    if missing:
        fail.append(f"B: {len(missing)} manifest file(s) missing under {root}")
    if changed:
        fail.append(f"C: {len(changed)} manifest file(s) changed")

    # D  nothing unexpected.  Only meaningful for a BUILT release: the working
    # repository legitimately carries data, logs and untracked scratch.
    unexpected: list[str] = []
    if built:
        unexpected = sorted(set(walk(root)) - set(paths) - {MANIFEST_PATH})
        report["unexpected"] = unexpected
        if unexpected:
            fail.append(f"D: {len(unexpected)} file(s) under {root} are not in the manifest")

    # E, F, I  content gates
    movement, secrets, private = [], [], []
    for p in paths + [MANIFEST_PATH]:
        f = root / p
        if not f.is_file():
            continue
        try:
            found_movement, hit, private_hit = scan_readable(f, p)
        except ValueError as error:
            fail.append(f"E/F/I: {error}")
            continue
        movement.extend(found_movement)
        if hit:
            secrets.append({"path": p, "patterns": hit})
        if private_hit:
            private.append({"path": p, "patterns": private_hit})
    report["movement_records"] = movement
    report["secret_patterns"] = secrets
    report["private_routes"] = private
    if movement:
        fail.append(f"E: {len(movement)} released file(s) carry movement-level records")
    if secrets:
        fail.append(f"F: {len(secrets)} released file(s) carry a credential-shaped string")
    if private:
        fail.append(f"I: {len(private)} released file(s) carry private route metadata")

    # G  the manifest and the set rule agree, in both directions.  The reverse
    # half needs the git index, so it runs only against this repository.
    rule_violations = [p for p in paths if not is_released(p)]
    report["rule_violations"] = rule_violations
    if rule_violations:
        fail.append(f"G: {len(rule_violations)} manifest path(s) break release_set.is_released")
    unlisted: list[str] = []
    if not built:
        unlisted = sorted(set(manifest_set()) - set(paths))
        report["unlisted"] = unlisted
        if unlisted:
            fail.append(f"G: {len(unlisted)} released path(s) are missing from the manifest")

    # H  licence
    lic = root / "LICENSE"
    if not lic.is_file():
        fail.append("H: LICENSE is missing")
    else:
        got = sha256(lic)
        report["license_sha256"] = got
        t = lic.read_text(errors="ignore")
        for needle, what in GPLV3_STRUCTURE:
            if needle not in t:
                fail.append(f"H: LICENSE is missing {what}")
        if len(t.splitlines()) != 674:
            fail.append(f"H: LICENSE has {len(t.splitlines())} lines, not 674")
        if got != GPLV3_SHA256:
            # Round 1 non-blocking 2: two phrases and a line count let altered
            # text of the same length pass.  The whole file is compared now.
            fail.append(f"H: LICENSE is not the canonical GPLv3 text "
                        f"({got[:16]}... != {GPLV3_SHA256[:16]}...)")

    # J  the manifest describes a commit, not a working tree
    dirty: list[str] = []
    if not built:
        r = subprocess.run(["git", "status", "--porcelain", "-z", "--"] + paths + [MANIFEST_PATH],
                           cwd=ROOT, capture_output=True)
        for entry in r.stdout.decode().split("\0"):
            if len(entry) > 3:
                dirty.append(entry[3:])
        report["dirty"] = dirty
        if dirty:
            fail.append(f"J: {len(dirty)} released path(s) differ from the commit; "
                        f"the manifest would describe nobody's tree")

    # K  public references must not require private process records.
    excluded_links, decisions, missing_citations = [], [], []
    for p in paths + [MANIFEST_PATH]:
        f = root / p
        if not f.is_file():
            continue
        try:
            text = figure_text(f, p) if is_figure(p) else f.read_text(encoding="utf-8-sig")
            references = scan_references(text, p)
        except (UnicodeError, ValueError, SyntaxError, tokenize.TokenError) as error:
            fail.append(f"K: cannot inspect references in {p}: {error}")
            continue
        citations = missing_path_citations(text, p,
                                           paths + [MANIFEST_PATH])
        if citations:
            missing_citations.append({"path": p, "citations": citations})
        if references["excluded_links"]:
            excluded_links.append({"path": p, "targets": references["excluded_links"]})
        if references["decision_references"]:
            decisions.append({"path": p, "references": references["decision_references"]})
    report["missing_path_citations"] = missing_citations
    if missing_citations:
        fail.append(f"K: {len(missing_citations)} top-level file(s) cite paths absent from the export")
    report["excluded_links"] = excluded_links
    report["decision_references"] = decisions
    if excluded_links:
        fail.append(f"K: {len(excluded_links)} released file(s) link to excluded paths")
    if decisions:
        fail.append(f"K: {len(decisions)} released file(s) name private decisions")

    # L  The production rewrite contracts apply in the checker as well as the tests.
    integrity = []
    if {'release/IMPORTED-SOURCES.json', 'release/REWRITE-LINES.json'}.intersection(paths):
        try:
            imported = json.loads((root / 'release/IMPORTED-SOURCES.json').read_text())
            scope = json.loads((root / 'release/REWRITE-LINES.json').read_text())
            integrity = inspect_imported(root, imported, scope) + inspect_rewritten(root, scope)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
            integrity.append(('release/REWRITE-LINES.json', 0, str(error)))
        if integrity:
            fail.append(f"L: {len(integrity)} imported or rewritten text integrity issue(s)")
    report['rewrite_integrity_issues'] = integrity

    report["failures"] = fail
    if a.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        print(f"root {root}")
        print(f"manifest {len(rows)} files"
              f"{'' if built else '  (working repository: check D skipped)'}")
        for k, label in (("missing", "B missing"), ("changed", "C changed"),
                         ("unexpected", "D unexpected"),
                         ("movement_records", "E movement records"),
                         ("secret_patterns", "F credential strings"),
                         ("rule_violations", "G rule violations"),
                         ("unlisted", "G released, unlisted"),
                         ("private_routes", "I private routes"),
                         ("dirty", "J uncommitted"),
                         ("missing_path_citations", "K missing path citations"),
                         ("excluded_links", "K excluded links"),
                         ("decision_references", "K decision references"),
                         ("rewrite_integrity_issues", "L rewrite integrity")):
            v = report.get(k)
            if v is None:
                continue
            print(f"  {label:24s} {len(v)}")
            for item in list(v)[:10]:
                print(f"      {item}")
        if fail:
            print("\nFAIL")
            for f in fail:
                print(f"  - {f}")
        else:
            print("\nPASS — the tree matches the manifest and the release gates hold")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
