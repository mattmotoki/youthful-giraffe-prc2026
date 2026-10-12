"""Regenerate every table in RESULTS.md from the saved JSON and OOF files.

Sol's round-1 review noted that the checked-in code could not reproduce the
interval tables. It can now: this script is the only producer of them, and it
uses `compare.py`'s airport-day cluster bootstrap throughout.

    python make_tables.py [--weather]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import prc.data as D
from compare import compare, load_oof, weather_mask

FOLDS = ["july", "december", "january"]
FAMILIES = ["queue", "occupancy", "stand", "pressure", "flow", "geometry", "config"]
SINGLES = (["base"] + [f"base+{f}" for f in FAMILIES]
           + ["base+queue2", "base+all", "all_repaired", "all_repaired_q2"])


def _json(fold: str) -> dict:
    p = HERE / f"ablation_{fold}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def oracle_table() -> str:
    """Section 1: the ceiling instrument, on every fold and every segment."""
    d = json.loads((HERE / "oracle_bound.json").read_text())
    keys = [("rmse", "all"), ("rmse_non_noac", "**non-noac**"), ("rmse_bulk", "bulk ≤3600"),
            ("rmse_nm_bulk", "NM bulk"), ("rmse_tail", "tail >3600"),
            ("rmse_nonm", "no-NM"), ("rmse_noac", "noac")]
    folds = [f for f in FOLDS if f in d]
    out = ["| segment | " + " | ".join(f"{f}: base → oracle" for f in folds) + " |",
           "|---|" + "---|" * len(folds)]
    for key, name in keys:
        cells = []
        for f in folds:
            b, o = d[f]["base"], d[f]["oracle"]
            cells.append("%.1f → **%.1f**" % (b[key], o[key]) if key in b else "–")
        out.append("| %s | %s |" % (name, " | ".join(cells)))
    return "\n".join(out)


def seg_rmse(cfg: str, fold: str) -> dict:
    """Every segment RMSE for one config, computed from the saved OOF.

    Read from the OOF rather than the result JSON so that configs measured
    before `metrics` gained the non-noac segment are reported the same way as
    the rest.
    """
    from compare import load_oof, segments
    o = load_oof(cfg, fold)
    e2 = ((o.y_true - o.y_pred) ** 2).to_numpy()
    return {k: float(np.sqrt(e2[m].mean())) for k, m in segments(o).items() if m.sum()}


SEG_KEY = {"rmse": "all", "rmse_bulk": "bulk (y<=3600)", "rmse_tail": "tail (y>3600)",
           "rmse_nonm": "no-NM", "rmse_non_noac": "non-noac", "rmse_noac": "noac",
           "rmse_nm_bulk": "NM bulk"}


def _metric(d: dict, fold: str, cfg: str, metric: str) -> float:
    """Metric for one cell, from the JSON when present and from the OOF when not.

    Configs measured before `metrics` gained the non-noac and NM-bulk segments
    have no such key in their JSON; their OOF carries the segment flags, so the
    value is recomputed rather than left blank.
    """
    m = d[fold].get(cfg, {})
    if metric in m:
        return float(m[metric])
    return seg_rmse(cfg, fold)[SEG_KEY[metric]]


def fold_table(metric: str, label: str) -> str:
    d = {f: _json(f) for f in FOLDS}
    out = [f"| config | july {label} | Δ | december {label} | Δ | january {label} | Δ |",
           "|---|---|---|---|---|---|---|"]
    for k in SINGLES:
        if not all(k in d[f] for f in FOLDS):
            continue
        cells = []
        for f in FOLDS:
            v, bv = _metric(d, f, k, metric), _metric(d, f, "base", metric)
            cells.append("%.1f | %+.1f" % (v, v - bv))
        out.append("| `%s` | %s |" % (k, " | ".join(cells)))
    return "\n".join(out)


def segment_table(cfg: str = "base+all") -> str:
    d = {f: _json(f) for f in FOLDS}
    segs = [("rmse", "all"), ("rmse_bulk", "bulk ≤3600"), ("rmse_non_noac", "**non-noac**"),
            ("rmse_nm_bulk", "NM bulk"), ("rmse_nonm", "no-NM"), ("rmse_tail", "tail >3600"),
            ("rmse_noac", "noac")]
    out = ["| segment | july base → " + cfg + " | december | january |", "|---|---|---|---|"]
    for key, name in segs:
        cells = []
        for f in FOLDS:
            if cfg not in d[f]:
                cells.append("–"); continue
            cells.append("%.1f → %.1f" % (_metric(d, f, "base", key), _metric(d, f, cfg, key)))
        out.append("| %s | %s |" % (name, " | ".join(cells)))
    return "\n".join(out)


def ci_table(segment: str, weather: bool = False) -> str:
    rows = {}
    for f in FOLDS:
        bad = weather_mask(f, load_oof("base", f)) if weather else None
        for fam in FAMILIES:
            cfg = f"base+{fam}"
            if not (HERE / "oof" / f"{cfg}_{f}.parquet").exists():
                continue
            t = compare("base", cfg, f, bad)
            r = t[t.segment == segment]
            if len(r):
                rows.setdefault(fam, {})[f] = "%+.1f %s" % (r.delta.iloc[0], r.ci95.iloc[0])
    out = ["| family | " + " | ".join(FOLDS) + " |", "|---|" + "---|" * len(FOLDS)]
    for fam in FAMILIES:
        if fam not in rows:
            continue
        out.append("| %s | %s |" % (fam, " | ".join(rows[fam].get(f, "–") for f in FOLDS)))
    return "\n".join(out)


def ledger_rows(reviewer: str = "REVIEWER") -> str:
    """The docs/LEDGER.md rows for this experiment."""
    orc = json.loads((HERE / "oracle_bound.json").read_text())
    out = []
    o = {f: orc[f]["oracle"] for f in orc}
    b = {f: orc[f]["base"] for f in orc}
    out.append("| 2026-09-05 | exp-004 oracle | congestion | ceiling instrument: base + TRUE "
               "same-runway/airport take-off counts between the hidden block time and take-off "
               "(illegal at prediction time) | %.1f | %.1f | %.0f | %.0f | "
               "experiments/exp-004-congestion/oracle_bound.json | %s | jan %.1f; non-noac "
               "%.1f/%.1f/%.1f against base %.1f/%.1f/%.1f; loose bound, the count is a "
               "quantised function of the target |"
               % (o["july"]["rmse"], o["december"]["rmse"], o["july"]["rmse_tail"],
                  o["july"]["rmse_nonm"], reviewer, o["january"]["rmse"],
                  o["july"]["rmse_non_noac"], o["december"]["rmse_non_noac"],
                  o["january"]["rmse_non_noac"], b["july"]["rmse_non_noac"],
                  b["december"]["rmse_non_noac"], b["january"]["rmse_non_noac"]))
    for cfg, mech in [
        ("base", "frozen base: own-flight deltas + airport/stand/runway/type/operator/hour, "
                 "CatBoost 3000x0.05 d8 GPU"),
        ("base+all", "base + all seven surface-state families (96 columns), "
                     "prc.surface_features VERSION 2"),
        ("all_repaired", "base + seven families, config categoricals dropped and flow levels "
                         "made month-relative"),
    ]:
        try:
            m = {f: seg_rmse(cfg, f) for f in FOLDS}
        except FileNotFoundError:
            continue
        out.append("| 2026-09-05 | exp-004 | congestion | %s | %.1f | %.1f | %.0f | %.0f | "
                   "experiments/exp-004-congestion/RESULTS.md | %s | jan %.1f; non-noac "
                   "%.1f/%.1f/%.1f; bulk %.1f/%.1f/%.1f; noac %.0f/%.0f/%.0f |"
                   % (mech, m["july"]["all"], m["december"]["all"], m["july"]["tail (y>3600)"],
                      m["july"]["no-NM"], reviewer, m["january"]["all"],
                      m["july"]["non-noac"], m["december"]["non-noac"], m["january"]["non-noac"],
                      m["july"]["bulk (y<=3600)"], m["december"]["bulk (y<=3600)"],
                      m["january"]["bulk (y<=3600)"],
                      m["july"]["noac"], m["december"]["noac"], m["january"]["noac"]))
    return "\n".join(out)


if __name__ == "__main__":
    if (HERE / "oracle_bound.json").exists():
        print("### Oracle ceiling\n"); print(oracle_table()); print()
    print("### RMSE, all rows\n"); print(fold_table("rmse", "")); print()
    print("### RMSE, bulk\n"); print(fold_table("rmse_bulk", "bulk")); print()
    print("### RMSE, non-noac\n"); print(fold_table("rmse_non_noac", "non-noac")); print()
    print("### RMSE, tail (y>3600)\n"); print(fold_table("rmse_tail", "tail")); print()
    print("### RMSE, no-NM\n"); print(fold_table("rmse_nonm", "no-NM")); print()
    print("### base+all by segment\n"); print(segment_table()); print()
    print("### all_repaired by segment\n"); print(segment_table("all_repaired")); print()
    print("### Δ NM bulk, cluster bootstrap\n"); print(ci_table("NM bulk")); print()
    print("### Δ non-noac, cluster bootstrap\n"); print(ci_table("non-noac")); print()
    if "--ledger" in sys.argv:
        print("### Ledger rows\n"); print(ledger_rows()); print()
    if "--weather" in sys.argv:
        print("### Δ bad-weather NM bulk, cluster bootstrap\n")
        print(ci_table("bad weather, NM bulk", weather=True))
