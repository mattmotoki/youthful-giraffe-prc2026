"""exp-045-harvest: the committed address book of the data stage.

Every measurement this stage produces has to be findable and checkable from the repo
alone (`/path/to/competitions/AGENTS.md`, "a measurement needs a durable address").  This
writes `manifest.json`:

  months -> days -> the kernel that produced the day, its seconds, the release archives
  it consumed with the publisher's SHA-256 and whether the bytes actually read matched,
  the four output files with the SHA-256 the kernel computed on what it wrote, and the
  summary parquet pulled to this laptop for each airport with its own SHA-256 and row
  count.

  months -> kernels: the Kaggle slug and version that holds the month's points, which
  never leave Kaggle (SPEC.md, decision 2).  A month's kernel is never re-run after it
  succeeds, because a re-run replaces its output.

  missing: the days that failed twice, with the reason.

    .venv/bin/python experiments/exp-045-harvest/code/manifest.py
"""
from __future__ import annotations

import os

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ART = ROOT / "artifacts/exp-045"
sys.path.insert(0, str(EXP / "code"))
from days import ORDER, days as month_days                     # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kernel_version(slug: str) -> str | None:
    r = subprocess.run([str(Path.home() / ".local/bin/kaggle"), "kernels", "status",
                        f"{kaggle_user()}/{slug}"], capture_output=True, text=True, timeout=180)
    return (r.stdout + r.stderr).strip() or None


def main() -> None:
    state = json.loads((EXP / "analysis/dispatch_state.json").read_text()) if (
        EXP / "analysis/dispatch_state.json").exists() else {"kernels": []}
    day_kernel = {d: k["name"] for k in state["kernels"] for d in k["days"]}
    missing = json.loads((EXP / "analysis/missing.json").read_text()) if (
        EXP / "analysis/missing.json").exists() else {}

    # what the publisher shipped that day, so a consumer can tell a dark airport from a
    # short archive without leaving the manifest.  2025-06-11 is 0.73 GB against 3.30 the
    # day before and reads 5 % receipt where the rest of its month reads 53 to 63:
    # without this flag that day looks like a network-wide outage.
    pubp = EXP / "analysis/published_bytes.json"
    pub = json.loads(pubp.read_text())["days"] if pubp.exists() else {}

    months: dict[str, dict] = {}
    for month in ORDER:
        rec: dict = {"days": {}, "kernels": {}, "delivered": 0, "expected": len(month_days(month))}
        for day in month_days(month):
            rp = EXP / f"analysis/runs/run_{day}.json"
            if not rp.exists():
                continue
            r = json.loads(rp.read_text())
            name = day_kernel.get(day, "unknown")
            slug = f"prc-harvest-{name}"
            d = {
                "kernel": slug,
                "seconds": r["seconds"],
                "repo": r["repo"], "tag": r["tag"],
                "archives": [{"name": a["name"], "bytes": a["bytes"],
                              "sha256_published": a["published"],
                              "sha256_read": a["computed"], "match": a["ok"]}
                             for a in r["digest_match"]],
                "digests": [r["digests_verified"], r["digests_expected"]],
                "bytes_consumed": sum(a["bytes"] for a in r["assets_read"]),
                "outputs": r["outputs"],
                "summaries": {},
            }
            if day in pub:
                d["published_bytes"] = pub[day]["bytes"]
                d["published_short"] = pub[day]["short"]
            for a in sorted(r["by_airport"]):
                p = ART / f"summaries/{a}/{day}.parquet"
                if p.exists():
                    d["summaries"][a] = {
                        "path": str(p.relative_to(ROOT)),
                        "rows": int(pd.read_parquet(p, columns=["kind"]).shape[0]),
                        "bytes": p.stat().st_size, "sha256": sha256(p)}
            rec["days"][day] = d
            ses = EXP / f"analysis/sessions/{name}.json"
            sj = json.loads(ses.read_text()) if ses.exists() else {}
            # The DAY's own run record is the authority on the rule that wrote it.  The
            # session record is a fallback and was the only source until 2026-09-08
            # 16:22, when the duplicate sessions arrived: dispatch_state maps a day to
            # the LAST kernel that claimed it, that was prc-harvest-2025-01-b, which had
            # not landed and so had no session record, and all 31 of January's days
            # reported a null rule while their summaries were perfectly good.
            # sessions written before 2026-09-08 11:40 carry no name: they are all the
            # rule the desk had frozen, before the pilot's reviewer repaired it
            fetch_rule = (r.get("rule_version") or sj.get("rule_version")
                          or ("shipped-2026-09-07" if (r or sj) else None))
            fetch_sha = r.get("rule_sha256") or sj.get("rule_sha256")
            rd = EXP / f"analysis/rederived/{day}.json"
            if rd.exists():
                r2 = json.loads(rd.read_text())
                d["summaries_rule"] = r2.get("rule_version")
                d["summaries_rule_sha256"] = r2.get("rule_sha256")
                d["summaries_rederived_by"] = r2.get("kernel")
            else:
                d["summaries_rule"] = fetch_rule
                d["summaries_rule_sha256"] = fetch_sha
            rec["kernels"].setdefault(slug, {"days": 0, "points_rows": 0, "points_bytes": 0,
                                             "fetch_rule": fetch_rule,
                                             "fetch_rule_sha256": fetch_sha})
            k = rec["kernels"][slug]
            k["days"] += 1
            k["points_rows"] += r["outputs"]["points"]["rows"]
            k["points_bytes"] += r["outputs"]["points"]["bytes"]
        rec["delivered"] = len(rec["days"])
        rec["seconds"] = round(sum(d["seconds"] for d in rec["days"].values()), 1)
        rec["bytes_consumed"] = sum(d["bytes_consumed"] for d in rec["days"].values())
        rec["digests_verified"] = sum(d["digests"][0] for d in rec["days"].values())
        rec["digests_expected"] = sum(d["digests"][1] for d in rec["days"].values())
        rec["missing"] = [d for d in month_days(month) if d in missing]
        rec["summaries_rules"] = sorted({d.get("summaries_rule")
                                         for d in rec["days"].values()} - {None})
        if rec["days"] or rec["missing"]:
            months[month] = rec

    out = {
        "experiment": "exp-045-harvest",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "boxes": "experiments/exp-045-harvest/analysis/boxes.json",
        "rule": "experiments/exp-043-pilot/code/segments.py (frozen) + "
                "experiments/exp-045-harvest/code/allsegments.py; proof in analysis/rule_check.json",
        "points_live_on": "Kaggle kernel output, one kernel per month; never re-run after success",
        "rule_note": "The pilot's reviewer repaired the rule on 2026-09-08 (a first-airborne "
                     "point with a null altitude is not a take-off) after four months had "
                     "been fetched under the shipped rule. `summaries_rule_sha256` per day "
                     "says which rule wrote that day's summaries; `code/rederive_local.py` "
                     "re-derives a month from its stored points, without re-fetching. "
                     "READ `summaries` FOR WHAT IS ON DISK: a day's `outputs.segments` "
                     "records what the FETCH kernel wrote and is not re-stated by a "
                     "re-derive, so on a re-derived day the two row counts differ by "
                     "what the repair removed.",
        "rules": {
            "shipped-2026-09-07": "any first non-ground point after a ground run is a take-off",
            "repair-2026-09-08": "that point must also carry a real altitude, not readsb's null "
                                 "(exp-043-pilot/analysis/divergence.json prices it at 6 of 5,395 "
                                 "departure segments and 0.07 s of held-out RMSE)",
            "repair-2026-09-08+stable-order": "the same, and the points are ordered stably, so "
                                 "two points that share a timestamp and disagree about on_ground "
                                 "keep the order readsb wrote them in.  Without it the ground run "
                                 "splits differently on different machines: 14 of 8,301 segments "
                                 "on 2026-01-01, every one of them on an aircraft whose points "
                                 "share a timestamp (analysis/machine_check.json)"},
        "summaries_rule": None,
        "days_expected": sum(len(month_days(m)) for m in ORDER),
        "days_delivered": sum(m["delivered"] for m in months.values()),
        "missing": missing,
        "months": months,
    }
    seen = sorted({d.get("summaries_rule") for m in months.values()
                   for d in m["days"].values()} - {None})
    out["summaries_rule"] = seen[0] if len(seen) == 1 else seen
    out["summaries_rule_uniform"] = len(seen) <= 1
    (EXP / "manifest.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"manifest: {out['days_delivered']} of {out['days_expected']} days, "
          f"{len(missing)} missing, "
          f"{sum(m['digests_verified'] for m in months.values())} of "
          f"{sum(m['digests_expected'] for m in months.values())} archive digests verified")
    for m, r in months.items():
        print(f"  {m}: {r['delivered']}/{r['expected']} days, "
              f"{r['bytes_consumed'] / 1e9:.0f} GB consumed, {r['seconds'] / 3600:.1f} kernel hours")


if __name__ == "__main__":
    main()
