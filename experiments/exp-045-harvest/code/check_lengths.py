"""exp-045-harvest: verify the days whose publisher never signed them.

adsb.lol's GitHub releases carry a `digest` on each asset from about mid-2025.  Before
that they carry none: the January and February 2025 assets return `digest: null` from
the API, so `manifest.json` reports 0 of 57 digests verified for 2025-01 -- not a
failed check but an absent one, and the assignment's "a day whose digest fails" does
not describe it.  Those days are good data and are kept.

What the publisher does offer for every asset, signed or not, is a byte length.  This
compares it against the bytes the kernel actually consumed, so the record can say what
was verified rather than leaving a zero to be read as a failure.

    .venv/bin/python experiments/exp-045-harvest/code/check_lengths.py [YYYY-MM ...]

Writes analysis/lengths.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def release(repo: str, tag: str) -> dict | None:
    """The release, or None when GitHub could not be asked.  `None` and `{}` are
    different answers: one is "we do not know", the other is "there is nothing there",
    and reporting the first as a mismatch would be a false alarm."""
    r = subprocess.run(["gh", "api", f"repos/{repo}/releases/tags/{tag}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def main(months: list[str]) -> int:
    man = json.loads((EXP / "manifest.json").read_text())
    out: dict = {}
    bad: list[str] = []
    unknown: list[str] = []
    for mo, rec in sorted(man["months"].items()):
        if months and mo not in months:
            continue
        if rec["digests_verified"] == rec["digests_expected"]:
            continue                       # the publisher signed these; nothing to add
        for day, d in sorted(rec["days"].items()):
            rel = release(d["repo"], d["tag"])
            if rel is None:
                unknown.append(day)
                continue
            sizes = {a["name"]: a["size"] for a in rel.get("assets", [])}
            for a in d["archives"]:
                want = sizes.get(a["name"])
                got = a["bytes"]
                ok = want is not None and want == got
                out.setdefault(mo, {}).setdefault(day, []).append(
                    {"name": a["name"], "bytes_read": got, "bytes_published": want,
                     "length_verified": ok, "signed": a.get("sha256_published") is not None})
                if not ok:
                    bad.append(f"{day} {a['name']}: read {got}, publisher says {want}")
    for line in bad:
        print(line)
    n = sum(len(v) for m in out.values() for v in m.values())
    days = sum(len(m) for m in out.values())
    (EXP / "analysis/lengths.json").write_text(json.dumps(out, indent=2))
    tail = f"; {len(unknown)} days GitHub could not be asked about" if unknown else ""
    print(f"checked {n} unsigned assets over {days} days against the publisher's "
          f"declared length; {len(bad)} disagree{tail}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
