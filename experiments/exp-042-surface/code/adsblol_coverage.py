"""Q16 calendar-coverage check of the adsb.lol open historical archive.

Reads only GitHub release *metadata* (no flight data is downloaded).
Run: gh auth status must pass.  .venv/bin/python experiments/exp-042-surface/code/adsblol_coverage.py
Writes: experiments/exp-042-surface/analysis/adsblol_coverage.json
"""
import json
import re
import subprocess
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "adsblol_coverage.json"
REPOS = {2025: "adsblol/globe_history_2025", 2026: "adsblol/globe_history_2026"}
INSTANCE = "planes-readsb-prod-0"


def releases(repo: str):
    raw = subprocess.run(
        ["gh", "api", "-X", "GET", f"repos/{repo}/releases", "--paginate", "-q",
         '.[] | "\\(.tag_name)\\t\\([.assets[].size]|add)"'],
        capture_output=True, text=True, check=True).stdout
    out = {}
    for line in raw.splitlines():
        tag, size = line.split("\t")
        if INSTANCE not in tag:
            continue
        m = re.match(r"v(\d{4})\.(\d{2})\.(\d{2})", tag)
        if m:
            out[date(*map(int, m.groups()))] = int(size)
    return out


def main() -> None:
    res = {"instance": INSTANCE, "checked_on": str(date.today()), "years": {}}
    for year, repo in REPOS.items():
        days = releases(repo)
        last = max(d for d in days if d.year == year)
        span = [date(year, 1, 1) + timedelta(n)
                for n in range((last - date(year, 1, 1)).days + 1)]
        missing = [d for d in span if d not in days]
        months = {}
        for mo in range(1, 13):
            sel = {d: s for d, s in days.items() if d.year == year and d.month == mo}
            if sel:
                months[f"{year}-{mo:02d}"] = {
                    "days_present": len(sel), "bytes": sum(sel.values())}
        res["years"][str(year)] = {
            "repo": repo,
            "last_day_present": str(last),
            "days_present": len([d for d in days if d.year == year]),
            "missing_days_in_span": [str(d) for d in missing],
            "months": months,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, sort_keys=True))
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
