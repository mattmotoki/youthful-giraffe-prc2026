"""Question 1: where the january non-LIRF squared error lives.

Run from the repo root:  .venv/bin/python experiments/exp-007-winter/code/q1_decompose.py
Writes experiments/exp-007-winter/analysis/q1_decomposition.md and the per-row
frame the later questions reuse (work/rows_<fold>.parquet).

Every table is an SSE share, not a row share: RMSE is what the board scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import q1c_trimmed as q1c  # noqa: E402

WORK = HERE.parent / "work"
TRIM = "--exclude-rollover" in sys.argv
OUT = HERE.parent / "analysis" / ("q1_decomposition_clean.md" if TRIM else "q1_decomposition.md")


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:,.{nd}f}"


def sse_table(df, by, name, top=None):
    g = df.groupby(by, dropna=False).agg(n=("se", "size"), sse=("se", "sum"), rmse=("err", C.rmse),
                                         ybar=("y", "mean"), pbar=("pred", "mean"),
                                         bias=("err", "mean"))
    g["sse_share"] = g["sse"] / df["se"].sum()
    g["row_share"] = g["n"] / len(df)
    g = g.sort_values("sse", ascending=False)
    if top:
        g = g.head(top)
    lines = [f"**{name}**", "", f"| {by if isinstance(by, str) else '/'.join(by)} | rows | row % | RMSE | SSE % | mean y | mean pred | mean err |",
             "|---|---|---|---|---|---|---|---|"]
    for k, r in g.iterrows():
        lines.append(f"| {k} | {int(r['n']):,} | {100*r['row_share']:.1f} | {fmt(r['rmse'])} | "
                     f"**{100*r['sse_share']:.1f}** | {fmt(r['ybar'],0)} | {fmt(r['pbar'],0)} | {fmt(r['bias'],0)} |")
    lines.append("")
    return "\n".join(lines)


def day_curve(df):
    g = df.groupby(["apt", "day"])["se"].sum().sort_values(ascending=False)
    tot = g.sum()
    cum = g.cumsum() / tot
    n = len(g)
    out = []
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
        k = max(1, int(round(frac * n)))
        out.append((frac, k, float(cum.iloc[k - 1])))
    return n, out


def classify(df):
    """(a) real long taxi, (b) block on LOBT/SCHED not AOBT, (c) rest.

    The label-free `span` is take-off - AOBT; the label-derived `r` is
    block - AOBT, so y = span - r. A row where the airport's block time is the
    NM off-block has r ~ 0 and y ~ span. A row whose block time was written on
    the filed off-block (LOBT) or the schedule instead has |b_lobt| or
    |b_sched| tiny while |r| is large. Tolerances: 60 s for "sits on",
    300 s for "away from".
    """
    r = df["r"]
    on_lobt = df["b_lobt"].abs() <= 60
    on_sched = df["b_sched"].abs() <= 60
    away_aobt = r.abs() > 300
    cls = pd.Series("(c) other", index=df.index, dtype=object)
    # (b) first: it is defined by where the block time was written
    b = (on_lobt | on_sched) & away_aobt & df["AOBT_3_flt"].notna()
    cls[b] = "(b) block on LOBT/SCHED"
    # (a) the block time is the NM off-block and the taxi itself is long
    a = (~b) & (r.abs() <= 300) & (df["y"] > 1800)
    cls[a] = "(a) real long taxi"
    cls[df["no_nm"]] = "(d) no-NM"
    return cls


def underover(df):
    return pd.Series(np.where(df["err"] > 0, "under-predicted", "over-predicted"), index=df.index)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    parts = []
    parts.append("# exp-007 question 1 — where the january non-LIRF error lives\n")
    parts.append(f"Baseline out-of-fold line: `{C.BASELINE}` "
                 f"(`experiments/exp-001-retro-v1/oof/{C.BASELINE}-<fold>.parquet`).\n")
    if TRIM:
        parts.append("**The two date-rollover rows of `analysis/q1_rollover.md` are removed from "
                     "every table below.** No column tried here separates them, and nothing that "
                     "reaches them is a detector — the one rule that does is 0-for-9 out of sample "
                     "and its hedge costs 6 to 8 s (`analysis/q1_detector.md`). So they are "
                     "UNREACHED rather than unreachable, and this is the decomposition of the "
                     "error a model has actually been able to attack.\n")

    frames = {}
    head = ["| fold | rows | RMSE all | RMSE non-LIRF | RMSE LIRF | non-LIRF SSE % | "
            "non-LIRF bulk | non-LIRF tail>3600 | non-LIRF no-NM |", "|---|---|---|---|---|---|---|---|---|"]
    for fold in C.FOLDS:
        if TRIM:
            d = pd.read_parquet(WORK / f"rows_{fold}.parquet")
            d = d[~q1c.rollover_mask(d)]
        else:
            d = C.fold_rows(fold)
            d.to_parquet(WORK / f"rows_{fold}.parquet", index=False)
        frames[fold] = d
        nl = d[d["apt"] != "LIRF"]
        lf = d[d["apt"] == "LIRF"]
        bulk = nl[(~nl["no_nm"]) & (~nl["tail"])]
        tail = nl[(~nl["no_nm"]) & nl["tail"]]
        nonm = nl[nl["no_nm"]]
        head.append(f"| {fold} | {len(d):,} | {fmt(C.rmse(d['err']))} | **{fmt(C.rmse(nl['err']))}** | "
                    f"{fmt(C.rmse(lf['err']))} | {100*nl['se'].sum()/d['se'].sum():.1f} | "
                    f"{fmt(C.rmse(bulk['err']))} | {fmt(C.rmse(tail['err']))} | {fmt(C.rmse(nonm['err']))} |")
    parts.append("## 1.0 The slice, three folds\n")
    parts.append("\n".join(head) + "\n")

    for fold in C.FOLDS:
        d = frames[fold]
        nl = d[d["apt"] != "LIRF"].copy()
        parts.append(f"\n## {fold}: non-LIRF departures ({len(nl):,} rows, RMSE {fmt(C.rmse(nl['err']))})\n")
        parts.append(sse_table(nl, "apt", "by airport"))
        parts.append(sse_table(nl, "regime", "by regime"))
        nl["hour_band"] = pd.cut(nl["hour"], [-1, 4, 7, 10, 13, 16, 19, 23],
                                 labels=["00-04", "05-07", "08-10", "11-13", "14-16", "17-19", "20-23"])
        parts.append(sse_table(nl, "hour_band", "by take-off hour (UTC)"))
        nl["cls"] = classify(nl)
        parts.append(sse_table(nl, "cls", "by what the row's own columns look like"))
        nl["cls_dir"] = nl["cls"] + " / " + underover(nl)
        parts.append(sse_table(nl, "cls_dir", "the same, split by sign of the error"))

        n, curve = day_curve(nl)
        rows = ["**by airport-day (days ranked by their own SSE)**", "",
                f"{n:,} airport-days.", "", "| worst | days | cumulative SSE share |", "|---|---|---|"]
        for frac, k, c in curve:
            rows.append(f"| {100*frac:.0f} % | {k} | **{100*c:.1f} %** |")
        parts.append("\n".join(rows) + "\n")

        wd = C.worst_days(nl, 0.05)
        key = pd.MultiIndex.from_frame(nl[["apt", "day"]])
        inw = key.isin(wd)
        w, rest = nl[inw], nl[~inw]
        parts.append(f"On the worst 5 % of airport-days ({len(w):,} rows, {100*len(w)/len(nl):.1f} % of rows, "
                     f"{100*w['se'].sum()/nl['se'].sum():.1f} % of SSE): RMSE {fmt(C.rmse(w['err']))} "
                     f"against {fmt(C.rmse(rest['err']))} elsewhere; mean error {fmt(w['err'].mean(),0)} s "
                     f"against {fmt(rest['err'].mean(),0)} s.\n")
        parts.append(sse_table(w, "cls", f"{fold}: the worst 5 % of airport-days, by class"))
        parts.append(sse_table(w, "regime", f"{fold}: the worst 5 % of airport-days, by regime"))
        top = nl.groupby(["apt", "day"])["se"].sum().sort_values(ascending=False).head(15)
        rows = [f"**{fold}: the fifteen worst airport-days**", "",
                "| airport | day | rows | SSE % of fold | RMSE | mean y | mean pred | median span | median r |",
                "|---|---|---|---|---|---|---|---|---|"]
        tot = nl["se"].sum()
        for (a, dy), s in top.items():
            sub = nl[(nl["apt"] == a) & (nl["day"] == dy)]
            rows.append(f"| {a} | {dy:%Y-%m-%d} | {len(sub):,} | {100*s/tot:.2f} | {fmt(C.rmse(sub['err']))} | "
                        f"{fmt(sub['y'].mean(),0)} | {fmt(sub['pred'].mean(),0)} | {fmt(sub['span'].median(),0)} | "
                        f"{fmt(sub['r'].median(),0)} |")
        parts.append("\n".join(rows) + "\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
