"""Ranking or final mode for the v73 steps (finalv, exp-191).

PRC_RANKING_MODE=final: the ranking file is input-root/data/raw/final_ranking.parquet and the template
final_submitting.parquet (both checked by SHA-256, the values of prc/data.py final mode), 670,790 rows.
Unset or "ranking": the released names, no new check, so the released bytes do not move.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

FINAL_SHA256 = {
    "final_ranking.parquet": "4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f",
    "final_submitting.parquet": "b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924",
}
ROWS = {"ranking": 344841, "final": 670790}
_CHECKED: set = set()


def mode() -> str:
    m = os.environ.get("PRC_RANKING_MODE", "ranking") or "ranking"
    if m not in ROWS:
        raise SystemExit(f"unknown PRC_RANKING_MODE {m!r}")
    return m


def is_final() -> bool:
    return mode() == "final"


def n_rows() -> int:
    return ROWS[mode()]


def _checked(path: Path) -> Path:
    if path not in _CHECKED:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != FINAL_SHA256[path.name]:
            raise SystemExit(f"{path} is not the final file (SHA-256 {h.hexdigest()})")
        _CHECKED.add(path)
    return path


def ranking_path(raw: Path) -> Path:
    raw = Path(raw)
    return _checked(raw / "final_ranking.parquet") if is_final() else raw / "ranking.parquet"


def submitting_path(raw: Path) -> Path:
    raw = Path(raw)
    return _checked(raw / "final_submitting.parquet") if is_final() else raw / "submitting.parquet"


#: season map (COMMON, desk 2026-10-08): February takes January's month scope, June takes July's.
#: The ranking file has no February or June rows, so the ranking mode gives the released masks.
JAN_SCOPE = (1, 2)
JUL_SCOPE = (6, 7)
