#!/usr/bin/env python3
"""Emit the RESULTS ablation tables straight from the JSONL records.

Two of the review findings were transcription errors (a ratio range and two
bootstrap interval rows copied from the wrong line). Generating the markdown
removes that class of error: the tables in RESULTS are this script's output.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "exp-005-weather"
ORDER = ["base", "basic", "precip", "runway_wind", "visibility", "change", "aobt",
         "no_aobt", "precip_vis", "all", "winter_core", "winter_core_plus",
         "noaobt_base", "noaobt_winter_core", "noaobt_all"]
WX = {"base": 0, "basic": 6, "precip": 11, "runway_wind": 5, "visibility": 4,
      "change": 5, "aobt": 13, "no_aobt": 31, "precip_vis": 15, "all": 44,
      "winter_core": 7, "winter_core_plus": 12, "noaobt_base": 0,
      "noaobt_winter_core": 7, "noaobt_all": 31}


def load() -> dict:
    rows = {}
    for name in ("results_v5.jsonl", "results_v6.jsonl"):
        for line in (EXP / name).read_text().splitlines():
            r = json.loads(line)
            rows[(r["fold"], r["config"])] = r
    return rows


def main() -> None:
    rows = load()
    for fold in ("january", "december", "july"):
        base = rows[(fold, "base")]
        nbase = rows[(fold, "noaobt_base")]
        print(f"\n**{fold}**\n")
        print("| config | wx cols | overall | Δ | bulk | Δ | ≤3600 | >3600 | no-NM |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cfg in ORDER:
            r = rows.get((fold, cfg))
            if r is None:
                continue
            ref = nbase if cfg.startswith("noaobt_") else base
            d1 = "–" if r is ref else f"{r['rmse'] - ref['rmse']:+.2f}"
            d2 = "–" if r is ref else f"{r['rmse_bulk'] - ref['rmse_bulk']:+.2f}"
            print(f"| {cfg} | {WX[cfg]} | {r['rmse']:.2f} | {d1} | {r['rmse_bulk']:.2f} | {d2} "
                  f"| {r['rmse_le3600']:.2f} | {r['rmse_gt3600']:.0f} | {r['rmse_no_nm']:.0f} |")

    matched(rows)


def read_log(name: str) -> dict:
    """Parse one analyse.py log into {(fold, slice): (delta, lo, hi)}."""
    out = {}
    for line in (EXP / "logs" / name).read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] not in ("january", "december", "july"):
            continue
        fold = parts[0]
        # slice label is everything between the fold and the trailing 6 numbers
        label = " ".join(parts[1:-6])
        try:
            delta, lo, hi = float(parts[-3]), float(parts[-2]), float(parts[-1])
        except ValueError:
            continue
        out[(fold, label)] = (delta, lo, hi)
    return out


def fmt(v: tuple[float, float, float]) -> str:
    d, lo, hi = v
    star = "**" if (lo < 0 and hi < 0) or (lo > 0 and hi > 0) else ""
    return f"{star}{d:+.2f}{star} [{lo:+.2f}, {hi:+.2f}]"


def matched(rows: dict) -> None:
    """Section 6.3: the matched control, point gains and cluster intervals."""
    anchored = read_log("cluster_bootstrap_base_vs_winter_core.log")
    control = read_log("cluster_bootstrap_noaobt_base_vs_noaobt_winter_core.log")

    print("\n\n**6.3 matched control: the same seven WINTER_CORE columns, "
          "anchor present vs absent**\n")
    print("| fold | with the AOBT anchor | without it | ratio of point gains |")
    print("|---|---|---|---:|")
    for fold in ("january", "december", "july"):
        a = rows[(fold, "winter_core")]["rmse_bulk"] - rows[(fold, "base")]["rmse_bulk"]
        b = rows[(fold, "noaobt_winter_core")]["rmse_bulk"] - rows[(fold, "noaobt_base")]["rmse_bulk"]
        key = (fold, "bulk (<=3600 & NM)")
        print(f"| {fold} | {fmt(anchored[key])} | {fmt(control[key])} | {b / a:.1f}x |")

    print("\nconditional slices, same seven columns:\n")
    print("| slice | fold | with anchor | without |")
    print("|---|---|---|---|")
    for label, fold in [("snow/freezing hours", "january"),
                        ("snow/freezing hours", "december"),
                        ("bulk & snow/freezing", "december"),
                        ("temp<=3C", "december")]:
        key = (fold, label)
        if key in anchored and key in control:
            print(f"| {label} | {fold} | {fmt(anchored[key])} | {fmt(control[key])} |")


if __name__ == "__main__":
    main()
