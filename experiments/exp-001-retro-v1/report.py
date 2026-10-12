"""Tables for RESULTS.md from a finished run: fold decomposition, per-airport
RMSE, and the candidate's prediction histogram against the 2025 target."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

EDGES = [-np.inf, 0, 300, 600, 900, 1200, 1800, 2400, 3600, 7200, np.inf]
LABELS = ["<=0", "0-300", "300-600", "600-900", "900-1200", "1200-1800", "1800-2400",
          "2400-3600", "3600-7200", ">7200"]


def fold_table(rec: dict) -> str:
    rows = ["| fold | n | RMSE | RMSE <=3600 | RMSE >3600 | RMSE no-NM | SSE share >3600 | SSE share no-NM | bias | rounds |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for f, r in rec["folds"].items():
        rows.append(
            f"| {f} | {r['n']:,} | {r['rmse']:.1f} | {r['rmse_bulk']:.1f} | {r['rmse_tail']:.1f} | "
            f"{r['rmse_no_nm']:.1f} | {r['sse_share_tail']:.1%} | {r['sse_share_no_nm']:.1%} | "
            f"{r['bias']:+.1f} | {r['refit_rounds']} |"
        )
    return "\n".join(rows)


def airport_table(name: str) -> str:
    parts = []
    for f in ("july", "december"):
        p = HERE / "oof" / f"{name}-{f}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        return "(no oof)"
    o = pd.concat(parts)
    g = o.groupby(["airport", "fold"]).apply(
        lambda d: pd.Series({"n": len(d), "rmse": np.sqrt(((d.y_true - d.y_pred) ** 2).mean())}),
        include_groups=False,
    ).reset_index()
    piv = g.pivot(index="airport", columns="fold", values="rmse").round(1)
    piv["n_july"] = g[g.fold == "july"].set_index("airport")["n"].astype(int)
    return piv.sort_values("july", ascending=False).to_markdown()


def hist_table(sub_path: Path) -> str:
    pred = pd.read_parquet(sub_path)[D.TARGET].to_numpy()
    y = pd.concat([D.load_month(m, columns=["PHASE_mvt", D.TARGET]) for m in D.MONTHS])
    y = y.loc[y.PHASE_mvt == "DEP", D.TARGET].to_numpy(dtype="float64")
    pb = pd.cut(pred, EDGES, labels=LABELS).value_counts().reindex(LABELS)
    yb = pd.cut(y, EDGES, labels=LABELS).value_counts().reindex(LABELS)
    rows = ["| band (s) | 2025 target share | candidate share | 2025 count | candidate count |", "|---|---|---|---|---|"]
    for b in LABELS:
        rows.append(f"| {b} | {yb[b]/len(y):.3%} | {pb[b]/len(pred):.3%} | {yb[b]:,} | {pb[b]:,} |")
    q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
    rows.append("")
    rows.append("| quantile | 2025 target | candidate |")
    rows.append("|---|---|---|")
    for k, a, b in zip(q, np.quantile(y, q), np.quantile(pred, q)):
        rows.append(f"| p{k*100:g} | {a:,.0f} | {b:,.0f} |")
    rows.append(f"| mean | {y.mean():,.1f} | {pred.mean():,.1f} |")
    rows.append(f"| min | {y.min():,.0f} | {pred.min():,.0f} |")
    rows.append(f"| max | {y.max():,.0f} | {pred.max():,.0f} |")
    return "\n".join(rows)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "v1"
    rec = json.loads((HERE / f"result-{name}.json").read_text())
    print("### Folds\n")
    print(fold_table(rec))
    print("\n### Per airport (RMSE, seconds)\n")
    print(airport_table(name))
    if "submission" in rec:
        print("\n### Candidate distribution against the 2025 target\n")
        print(hist_table(ROOT / rec["submission"]["path"]))
        print(f"\nsha256 `{rec['submission']['sha256']}`  rows {rec['submission']['n_rows']:,}")
