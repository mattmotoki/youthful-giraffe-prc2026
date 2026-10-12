"""exp-045-harvest: how many bytes adsb.lol published for each day.

A day can be empty in the receipt table for two quite different reasons: the feeders
did not hear the airport, or the publisher's archive for that day is short.  They look
identical downstream, so the published size is recorded next to the observations.
`v2025.06.11-planes-readsb-prod-0` is one asset and 0.73 GB where its neighbours are two
assets and 3.1 to 3.3 GB, and no amount of reading the traces would have said so.

Writes `analysis/published_bytes.json`: one record per day of the programme, with the
tag that carries it, the number of assets and the total bytes.

    .venv/bin/python experiments/exp-045-harvest/code/published_bytes.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from statistics import median

EXP = Path(__file__).resolve().parents[1]


def releases(repo: str) -> dict[str, dict]:
    """`{tag: {assets, bytes}}` for every release of `repo` the API will page through."""
    out: dict[str, dict] = {}
    for page in range(1, 11):
        r = subprocess.run(
            ["gh", "api", f"repos/{repo}/releases?per_page=100&page={page}",
             "--jq", '.[] | {tag: .tag_name, n: (.assets|length), '
                     'bytes: ([.assets[].size] | add // 0)}'],
            capture_output=True, text=True)
        rows = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
        if not rows:
            break
        for x in rows:
            out[x["tag"]] = {"assets": x["n"], "bytes": x["bytes"]}
    return out


def main() -> None:
    import sys
    sys.path.insert(0, str(EXP / "code"))
    from days import ORDER, days as month_days                     # noqa: PLC0415
    days = sorted(d for m in ORDER for d in month_days(m))
    rel = {}
    for repo in ("adsblol/globe_history_2025", "adsblol/globe_history_2026"):
        rel[repo] = releases(repo)
    rec: dict[str, dict] = {}
    for d in days:
        repo = ("adsblol/globe_history_2026" if d >= "2025-12-31"
                else f"adsblol/globe_history_{d[:4]}")
        base = f"v{d.replace('-', '.')}-planes-readsb-prod-0"
        for tag in (base, base + "tmp"):
            if tag in rel[repo]:
                rec[d] = {"repo": repo, "tag": tag, **rel[repo][tag]}
                break
        else:
            rec[d] = {"repo": repo, "tag": None, "assets": 0, "bytes": 0}
    got = [v["bytes"] for v in rec.values() if v["bytes"]]
    med = median(got) if got else 0
    short = {d: v for d, v in sorted(rec.items()) if v["bytes"] < 0.5 * med}
    for d, v in rec.items():
        v["short"] = d in short
    (EXP / "analysis/published_bytes.json").write_text(
        json.dumps({"median_bytes": med, "days": rec}, indent=1, sort_keys=True))
    print(f"{len(rec)} days; median {med / 1e9:.2f} GB; "
          f"{len(short)} under half of it: "
          + ", ".join(f"{d} {v['bytes'] / 1e9:.2f} GB" for d, v in short.items()))


if __name__ == "__main__":
    main()
