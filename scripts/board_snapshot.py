#!/usr/bin/env python3
"""Snapshot the PRC 2026 leaderboard API into docs/board/ (JSON per pull, one CSV of best-per-team)."""
import json, urllib.request, urllib.parse, datetime, pathlib, csv
BASE = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"
out = pathlib.Path(__file__).resolve().parents[1] / "docs" / "board"
out.mkdir(exist_ok=True)
items, cursor = [], None
for _ in range(100):
    url = BASE + "?size=100" + (f"&cursor={urllib.parse.quote(cursor)}" if cursor else "")
    d = json.load(urllib.request.urlopen(url, timeout=60))
    items += d["items"]; cursor = d.get("nextCursor")
    if not cursor: break
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M")
(out / f"{now}.json").write_text(json.dumps(items))
best = {}
for it in items:
    t = it["teamName"]
    if t not in best or it["score"] < best[t]["score"]: best[t] = it
rows = sorted(best.values(), key=lambda r: r["score"])
with open(out / "best_per_team.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["rank", "team", "score", "usedPairs", "filename", "processedAt", "submissions"])
    from collections import Counter; c = Counter(i["teamName"] for i in items)
    for i, r in enumerate(rows, 1): w.writerow([i, r["teamName"], r["score"], r["usedPairs"], r["filename"], r["processedAt"], c[r["teamName"]]])
ours = [r for r in rows if r["teamName"] == "youthful-giraffe"]
print(now, "teams", len(rows), "subs", len(items), "top", rows[0]["score"], "10th", rows[9]["score"] if len(rows) > 9 else None, "ours", ours[0]["score"] if ours else None)
