"""Rebuild the solution figures; descriptive statistics only, no model fits.

Run from the repository root:
    .venv/bin/python docs/writeup/figures/src/build_figures.py          # all figures
    .venv/bin/python docs/writeup/figures/src/build_figures.py 03 07    # some figures

Requires the official twelve 2025 training files in data/raw (figures 4, 5, 6
and 7). Record-based panels use
evidence.json, a small, fixed extract of the measured results. Writes PNGs
beside this script's parent; a full run also writes descriptive-statistics.json.
Figure 10 is a Mermaid diagram: source 10-pipeline.mmd beside this script, render
command in its header.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Ellipse, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle)
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[3]
E = json.loads((HERE / "evidence.json").read_text())

# One colour language for every chart: amber = old / 2025 / before,
# teal = new / 2026 / after, red = a bound or a recording problem.
INK, MUTED, GRID = "#22313f", "#66737d", "#dfe5ea"
TEAL, TEAL_L = "#127c84", "#cfe9e7"
AMBER, AMBER_L = "#d98e04", "#f8e3b5"
RED, RED_L = "#c0392b", "#f6d7d2"
# Diagram roles (figures 7 and 10): shape and colour say what a block is.
ROLE = {
    "data":    {"fill": "#e3ecf4", "edge": "#5d7d99", "text": INK},
    "feature": {"fill": "#eeeeee", "edge": "#8c8c8c", "text": INK},
    "model":   {"fill": TEAL_L,    "edge": TEAL,      "text": INK},
    "combine": {"fill": "#e7e0f2", "edge": "#6c4fa0", "text": INK},
    "rule":    {"fill": RED_L,     "edge": RED,       "text": INK},
    "output":  {"fill": INK,       "edge": INK,       "text": "white"},
}
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 13,
    "axes.titlesize": 19, "axes.titleweight": "bold", "axes.labelsize": 13,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": GRID,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.axisbelow": True,
})
STATS = {}


def save(fig, name):
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def heading(fig, text, sub=None, y=0.955):
    fig.text(0.035, y, text, fontsize=19, fontweight="bold", va="top")
    if sub:
        fig.text(0.035, y - 0.065, sub, fontsize=13, color=MUTED, va="top")


def arrow(ax, a, b, color=MUTED, lw=1.6, style="-|>", ms=16, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms,
                                 color=color, linewidth=lw, **kw))


def node(ax, role, x, y, w, h, text, size=12, bold=False):
    """Draw one diagram block centred at (x, y); the shape encodes its role."""
    s = ROLE[role]
    kw = {"facecolor": s["fill"], "edgecolor": s["edge"], "linewidth": 1.6}
    l, b = x - w / 2, y - h / 2
    if role == "data":  # cylinder
        e = h * 0.13
        ax.add_patch(Rectangle((l, b + e / 2), w, h - e, **kw))
        ax.add_patch(Ellipse((x, b + e / 2), w, e, **kw))
        ax.add_patch(Rectangle((l + 0.8 * kw["linewidth"] / 72, b + e / 2), w, h - e - 0.001,
                               facecolor=s["fill"], edgecolor="none"))
        ax.plot([l, l], [b + e / 2, b + h - e / 2], color=s["edge"], lw=1.6)
        ax.plot([l + w, l + w], [b + e / 2, b + h - e / 2], color=s["edge"], lw=1.6)
        ax.add_patch(Ellipse((x, b + h - e / 2), w, e, **kw))
        y = y - e * 0.15
    elif role == "model":
        ax.add_patch(FancyBboxPatch((l, b), w, h, boxstyle=f"round,pad=0,rounding_size={h*0.22}", **kw))
    elif role == "feature":
        ax.add_patch(Rectangle((l, b), w, h, **kw))
    elif role == "combine":  # funnel, wide side in
        ax.add_patch(Polygon([(l, b), (l + w, b + h * 0.18), (l + w, b + h * 0.82), (l, b + h)], **kw))
    elif role == "rule":  # hexagon
        c = h * 0.32
        ax.add_patch(Polygon([(l, y), (l + c, b), (l + w - c, b), (l + w, y), (l + w - c, b + h), (l + c, b + h)], **kw))
    elif role == "output":  # stadium
        ax.add_patch(FancyBboxPatch((l, b), w, h, boxstyle=f"round,pad=0,rounding_size={h/2}", **kw))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=s["text"],
            fontweight="bold" if bold else "normal", linespacing=1.25)


def role_key(ax, x0, y, roles, w=1.15, h=0.42, gap=0.42, size=11):
    names = {"data": "Data", "feature": "Feature step", "model": "Learned model",
             "combine": "Combination", "rule": "Rule", "output": "Output"}
    x = x0
    for r in roles:
        node(ax, r, x + w / 2, y, w, h, names[r], size=size)
        x += w + gap


# ---------------------------------------------------------------- figure 1
def progress():
    d = pd.DataFrame(E["scores"]).sort_values("upload")
    d["best"] = d.rmse_seconds.cummin()
    best_up = int(d.loc[d.rmse_seconds.idxmin(), "upload"])
    assert best_up == E["snapshot"]["best_upload"]
    fig, ax = plt.subplots(figsize=(10, 6.6))
    fig.subplots_adjust(left=.085, right=.975, top=.84, bottom=.1)
    heading(fig, "Seeing the departure in ADS-B data gave the biggest step",
            "Each dot is one upload; the line is our best score so far")
    score = dict(zip(d.upload, d.rmse_seconds))
    best = dict(zip(d.upload, d.best))
    ax.scatter(d.upload, d.rmse_seconds, s=22, color="#aab4bc", zorder=2, linewidths=0)
    ax.step(d.upload, d.best, where="post", color=TEAL, lw=2.8, zorder=3)
    nb = d[d.rmse_seconds.eq(d.best)]
    ax.scatter(nb.upload, nb.best, s=26, color=TEAL, zorder=4)
    first, last = score[1], score[best_up]
    ax.annotate(f"first {first:.1f} s", (1, first), xytext=(-3, -34), textcoords="offset points",
                fontsize=12, ha="left", color=INK)
    ax.annotate(f"best {last:.1f} s", (best_up, last), xytext=(34, 16),
                textcoords="offset points", fontsize=12, ha="right", color=INK)
    # The steps of the best-so-far line with a gain of 2.5 s or more, in plain words
    # (docs/SUBMISSIONS.md and METHOD section 5): upload 13; the month-held stacker and its half
    # blend in 35-36; the Rome rule in 48; the linear stack in 49-52. A leader joins each label
    # to the dot where its step ends; relpos is the label point where the leader starts.
    notes = [(12, 13, (16.0, 255.0), "left", (0, 0), "added the off-block time seen\nin ADS-B ground tracks"),
             (32, 36, (32.0, 235.8), "right", (1, 1), "combined the models on whole held-out\nmonths, averaged with an earlier entry"),
             (47, 48, (45.0, 231.2), "right", (1, 0.5), "a rule for Rome flights without a flight plan"),
             (48, 52, (49.0, 225.0), "right", (1, 0.5), "added runway-order and neural network models\nand more rules for how times are recorded")]
    for a, b, xy, ha, rp, text in notes:
        gain = best[a] - best[b]
        assert gain >= 2.5
        label = f"−{gain:.1f} s: {text}"
        lead = {"arrowstyle": "-", "color": MUTED, "lw": 1, "shrinkA": 3, "shrinkB": 4, "relpos": rp}
        ax.annotate(label, (b, best[b]), xytext=xy, fontsize=11.5, color=INK, ha=ha, va="center",
                    linespacing=1.3, arrowprops=lead)
        STATS.setdefault("progress_notes", []).append(
            {"text": label.replace("\n", " "), "from_upload": a, "to_upload": b,
             "from": best[a], "to": best[b], "gain_seconds": round(gain, 4)})
    ax.set(xlim=(0, 77), ylim=(220, 268.5))
    ax.set_yticks([230, 240, 250, 260])
    ax.set_xticks([1, 20, 40, 60, 75])
    ax.set_xlabel("Upload")
    ax.set_ylabel("Public RMSE, seconds")
    ax.grid(axis="y", color=GRID, lw=.8)
    STATS["progress"] = {"first": first, "best": last, "best_upload": best_up,
                         "improvement_seconds": round(first - last, 4)}
    save(fig, "01-progress.png")


# ---------------------------------------------------------------- figure 2
def clock():
    d = E["clock_precision"]
    order = np.argsort(d["median_lead_seconds"])[::-1]
    air = [d["airport"][i] for i in order]
    med = np.array([d["median_lead_seconds"][i] for i in order]) / 60
    p10 = np.array([d["p10_seconds"][i] for i in order]) / 60
    p90 = np.array([d["p90_seconds"][i] for i in order]) / 60
    fig = plt.figure(figsize=(10, 7.4))
    heading(fig, "Timing the first ADS-B point back to the stand\nmoved our clock 3 to 7 minutes earlier")
    top = fig.add_axes([.06, .63, .90, .21]); top.axis("off")
    top.set(xlim=(-0.6, 10.4), ylim=(-1.25, 1.35))
    # Schematic of one departure along its taxi path (not to scale).
    top.plot([0, 10.2], [0, 0], color=GRID, lw=9, solid_capstyle="round", zorder=1)
    top.add_patch(Rectangle((-0.35, -0.35), 0.7, 0.7, facecolor=INK, edgecolor="none", zorder=3))
    top.text(0, 0.62, "Stand", ha="center", fontsize=12, fontweight="bold")
    xs = np.array([2.6, 3.5, 4.4, 5.3, 6.2, 7.1, 8.0, 8.9, 9.8])
    top.scatter(xs, np.zeros_like(xs), s=46, color=INK, zorder=4)
    top.text(2.6, 0.45, "First ADS-B ground point\n(distance d, speed v)", ha="center", fontsize=11.5)
    top.plot([6.2, 6.2], [-0.55, 0.55], color=AMBER, lw=2.4, ls=(0, (3, 2)), zorder=2)
    top.text(6.35, 0.45, "Old clock: aircraft\n250 m from the stand", fontsize=11.5, color=AMBER,
             fontweight="bold", va="bottom")
    arrow(top, (2.45, -0.28), (0.25, -0.28), color=TEAL, lw=2.4,
          connectionstyle="arc3,rad=-0.35")
    top.text(1.35, -1.12, "New clock: off-block = first point time − d ÷ v",
             fontsize=11.5, color=TEAL, fontweight="bold", ha="left")
    ax = fig.add_axes([.12, .09, .84, .47])
    y = np.arange(len(air))
    ax.axvline(0, color=TEAL, lw=2.6, zorder=1)
    ax.hlines(y, p10, p90, color=AMBER_L, lw=11, zorder=2)
    ax.scatter(med, y, s=90, color=AMBER, zorder=3)
    for i, m in enumerate(med):
        ax.text(m + .28, i, f"{m:.1f} min" if i == 0 else f"{m:.1f}", va="center", fontsize=11.5,
                color="#8a5a00", fontweight="bold")
    ax.set_yticks(y, air); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=12.5, labelcolor=INK)
    ax.set_xlim(-0.3, 15.2); ax.set_ylim(len(air) - .5, -1.75)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_xlabel("Minutes from the new clock to the old clock")
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.text(0.15, -1.35, "New clock", color=TEAL, fontsize=12, fontweight="bold", va="center")
    ax.annotate("", (p10[0], -.6), (p90[0], -.6), arrowprops={"arrowstyle": "|-|", "color": MUTED,
                "lw": 1.2, "mutation_scale": 4})
    ax.text((p10[0] + p90[0]) / 2, -.95, "8 in 10 departures: old clock this much later; dot = median",
            fontsize=11.5, color=MUTED, va="center", ha="center")
    save(fig, "02-clock.png")


# ---------------------------------------------------------------- figure 3
def coverage():
    rows = pd.DataFrame(E["coverage"]["rows"])
    rows["y25"] = 100 * rows.with_clock_2025 / rows.departures_2025
    rows["y26"] = 100 * rows.with_clock_2026 / rows.departures_2026
    chk = rows.set_index(["airport", "month"])
    for (a, m), v25, v26 in [(("LIRF", 7), 86.3, 0.08), (("EGLL", 7), 6.5, 96.4)]:
        assert round(chk.loc[(a, m), "y25"], 1) == v25 and round(chk.loc[(a, m), "y26"], 2 if v26 < 1 else 1) == v26
    STATS["coverage"] = {f"{a}-{m:02d}": {"2025_pct": round(r.y25, 2), "2026_pct": round(r.y26, 2)}
                         for (a, m), r in chk.iterrows()}
    airports = sorted(rows.airport.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.3), sharey=True)
    fig.subplots_adjust(left=.09, right=.975, top=.76, bottom=.11, wspace=.08)
    heading(fig, "The share of departures with an ADS-B clock\nchanged sharply between the two years")
    for ax, month, name in [(axes[0], 1, "January"), (axes[1], 7, "July")]:
        g = rows[rows.month.eq(month)].set_index("airport").loc[airports]
        y = np.arange(len(airports))
        for i, (a, r) in enumerate(g.iterrows()):
            big = abs(r.y26 - r.y25) >= 25
            if abs(r.y26 - r.y25) > 3:
                arrow(ax, (r.y25, i), (r.y26, i), color=INK if big else "#b9c2c9",
                      lw=2.0 if big else 1.4, ms=14, shrinkA=6, shrinkB=7, zorder=2)
            ax.scatter(r.y25, i, s=80, color=AMBER, zorder=3)
            ax.scatter(r.y26, i, s=80, color=TEAL, zorder=4)
            if big:
                for v, c, dx in [(r.y25, AMBER, -1), (r.y26, TEAL, 1)]:
                    side = dx if r.y26 > r.y25 else -dx
                    txt = "0%" if v == 0 else (f"{v:.0f}%" if v >= 1 else f"{v:.2f}%")
                    if side < 0 and v < 12:
                        ax.text(v, i + .42, txt, ha="center", va="center", fontsize=10.5, color=c, fontweight="bold")
                    else:
                        ax.text(v + 3.2 * side, i + .02, txt, ha="left" if side > 0 else "right",
                                va="center", fontsize=11, color=c, fontweight="bold")
            if a == "LTFM":
                ax.text(4, i, "no clock in either year", va="center", fontsize=10.5, color=MUTED)
        ax.set_title(name, loc="left", fontsize=14, pad=8)
        ax.set_xlim(-12, 112); ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
        ax.grid(axis="x", color=GRID, lw=.8)
        ax.tick_params(axis="y", length=0, labelsize=12.5, labelcolor=INK)
    axes[0].set_yticks(np.arange(len(airports)), airports); axes[0].invert_yaxis()
    fig.text(.535, .035, "Departures with an ADS-B off-block clock", ha="center", fontsize=13)
    fig.text(.40, .835, "●", color=AMBER, fontsize=16, va="center")
    fig.text(.425, .835, "2025 training", fontsize=13, va="center")
    fig.text(.58, .835, "●", color=TEAL, fontsize=16, va="center")
    fig.text(.605, .835, "2026 ranking", fontsize=13, va="center")
    save(fig, "03-coverage.png")


# ------------------------------------------------------- raw data statistics
def raw_statistics():
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt",
            "AIRCRAFT_TYPE_mvt", "STAND_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt"]
    bins = np.arange(-3660, 3661, 60)
    hist = np.zeros(len(bins) - 1, dtype=np.int64)
    airport_counts = {}; joined_n = outside = 0; max_abs = 0.
    raw_hashes = {}; jan = jul = None
    files = sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))
    assert len(files) == 12, "All twelve official 2025 files are required"
    for p in files:
        raw_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        frame = pd.read_parquet(p, columns=cols)
        d = frame[frame.PHASE_mvt.eq("DEP")].copy()
        delta = (d.BLOCK_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds().dropna()
        hist += np.histogram(delta, bins=bins)[0]
        joined_n += len(delta); outside += int(delta.abs().gt(3606).sum())
        max_abs = max(max_abs, float(delta.abs().max()))
        # Delayed or early by the independent NM time: reduces accidental schedule equality.
        off = (d.AOBT_3_flt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().abs().gt(600)
        z = d[off & d.BLOCK_TIME_UTC_mvt.notna() & d.SCHED_TIME_UTC_mvt.notna()].copy()
        z["placeholder"] = (z.BLOCK_TIME_UTC_mvt - z.SCHED_TIME_UTC_mvt).dt.total_seconds().abs().le(6)
        for airport, g in z.groupby("ADEP_mvt"):
            counts = airport_counts.setdefault(airport, [0, 0]); counts[0] += len(g); counts[1] += int(g.placeholder.sum())
        if "2025-01-01" in p.name: jan = frame
        if "2025-07-01" in p.name: jul = frame
    assert joined_n == 2062577 and outside == 0 and hist.sum() == joined_n
    STATS["raw_sources_sha256"] = raw_hashes
    STATS["join_window"] = {"rows": joined_n, "outside_3606_seconds": outside, "maximum_absolute_seconds": max_abs,
                            "bin_edges_seconds": bins.tolist(), "bin_counts": hist.tolist()}
    STATS["placeholder"] = {a: {"eligible_rows": v[0], "placeholder_rows": v[1], "percent": 100 * v[1] / v[0]}
                            for a, v in airport_counts.items()}
    return bins, hist, jan, jul


# ---------------------------------------------------------------- figure 4
def join_window(bins, hist):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=.09, right=.975, top=.80, bottom=.12)
    heading(fig, "A plan is joined only within an hour of the block time",
            "All 2,062,577 joined departures of 2025; none lies outside ±3,606 s")
    mid = (bins[:-1] + bins[1:]) / 2 / 60
    ax.bar(mid, hist / 1000, width=.98, color=TEAL)
    w = 3606 / 60
    for s in (-1, 1):
        ax.axvline(s * w, color=RED, lw=2)
        ax.axvspan(s * w, s * 80, color=RED_L, alpha=.55, lw=0)
    ax.text(-w + 1.2, 112, "Join window\n±3,606 s", color=RED, fontsize=12, fontweight="bold", va="top")
    ax.text(-70.5, 60, "No plan joined?\nThen the block\ntime is out here,\nor the flight\nhad no plan", ha="center",
            fontsize=11.5, color=RED, va="center")
    ax.text(70.5, 60, "…or out here", ha="center", fontsize=11.5, color=RED, va="center")
    ax.set(xlim=(-80, 80), ylim=(0, 120))
    ax.set_xticks([-60, -30, 0, 30, 60], ["−60", "−30", "0", "+30", "+60 min"])
    ax.set_yticks([0, 40, 80, 120], ["0", "40k", "80k", "120k"])
    ax.set_xlabel("Recorded block time − plan off-block time")
    ax.set_ylabel("Departures per minute")
    ax.grid(axis="y", color=GRID, lw=.8)
    save(fig, "04-join-window.png")


# ---------------------------------------------------------------- figure 5
def placeholders():
    rows = sorted(STATS["placeholder"].items(), key=lambda t: t[1]["percent"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5.9))
    fig.subplots_adjust(left=.09, right=.975, top=.79, bottom=.11)
    heading(fig, "At Rome, delayed or early flights far more often keep\nthe scheduled time as their block time", None)
    fig.text(.035, .835, "2025 departures whose NM off-block is more than 10 min from the schedule",
             fontsize=13, color=MUTED, va="top")
    y = np.arange(len(rows)); values = [r[1]["percent"] for r in rows]
    ax.barh(y, values, color=[RED if a == "LIRF" else "#9fb0bb" for a, _ in rows], height=.62)
    ax.set_yticks(y, [a for a, _ in rows]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=12.5, labelcolor=INK)
    for i, (a, r) in enumerate(rows):
        v = r["percent"]
        txt = f"{v:.1f}%" if v >= .1 else f"{v:.2f}%"
        if a == "LIRF":
            txt += f"   ({r['placeholder_rows']:,} of {r['eligible_rows']:,})"
        ax.text(v + .1, i, txt, va="center", fontsize=12, color=RED if a == "LIRF" else INK,
                fontweight="bold" if a == "LIRF" else "normal")
    ax.set_xlim(0, 11); ax.set_xticks([0, 2, 4, 6, 8, 10], ["0", "2", "4", "6", "8", "10%"])
    ax.set_xlabel("Block time equal to the scheduled time, within 6 s")
    ax.grid(axis="x", color=GRID, lw=.8)
    save(fig, "05-placeholders.png")


# ---------------------------------------------------------------- figure 6
def stand(jul):
    record = E["stand_example"]
    # Select the published, deliberately large example by its measured values.
    # Do not put flight identifiers or an exact flight date in public artifacts.
    candidates = jul[jul.PHASE_mvt.eq("DEP") & jul.ADEP_mvt.eq("EGLL")
                     & jul.TAXITIME_SEC_mvt.eq(record["label_seconds"])]
    assert len(candidates) == 1
    dep = candidates.iloc[0]
    arr = jul[jul.PHASE_mvt.eq("ARR") & jul.ADES_mvt.eq(dep.ADEP_mvt)
              & jul.STAND_mvt.eq(dep.STAND_mvt) & jul.BLOCK_TIME_UTC_mvt.lt(dep.MVT_TIME_UTC_mvt)]
    arr = arr.sort_values(["BLOCK_TIME_UTC_mvt", "MVT_ID_mvt"]).iloc[-1]
    assert arr.AIRCRAFT_TYPE_mvt != dep.AIRCRAFT_TYPE_mvt
    bound = (dep.MVT_TIME_UTC_mvt - arr.BLOCK_TIME_UTC_mvt).total_seconds()
    assert bound == record["bound_seconds"] and dep.TAXITIME_SEC_mvt == record["label_seconds"]
    t0 = dep.BLOCK_TIME_UTC_mvt
    new = (arr.BLOCK_TIME_UTC_mvt - t0).total_seconds() / 60
    end = dep.TAXITIME_SEC_mvt / 60
    model = record["base_seconds"] / 60
    STATS["stand_example"] = {"airport": dep.ADEP_mvt, "month": "July 2025",
                              "takeoff_hour_UTC": int(dep.MVT_TIME_UTC_mvt.hour),
                              "arrival_minutes_after_offblock": round(new, 1), "taxi_minutes": round(end, 1),
                              "bound_minutes": round(bound / 60, 1), "model_before_minutes": round(model, 1),
                              "model_implied_offblock_minutes": round(end - model, 1),
                              "bound_holds": bool(end >= bound / 60)}
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=.035, right=.975, top=.78, bottom=.13)
    heading(fig, "The next aircraft at the stand puts a floor under a long taxi",
            "A deliberately large case: Heathrow, July 2025")
    bars = [(2.0, 0, end, INK, f"Recorded taxi time  {end:.1f} min"),
            (1.0, new, end, RED, f"Stand floor  ≥ {bound/60:.1f} min"),
            (0.0, end - model, end, AMBER, f"Earlier model  {model:.1f} min")]
    for yy, a, b, c, label in bars:
        ax.barh(yy, b - a, left=a, height=.42, color=c)
        ax.text(b - 1.2, yy, label, ha="right", va="center", color="white", fontsize=12.5, fontweight="bold")
    ax.axvline(new, color=RED, lw=1.6, ls=(0, (3, 2)), ymin=.02, ymax=.98)
    ax.axvline(end, color=INK, lw=1.6, ls=(0, (3, 2)), ymin=.02, ymax=.98)
    ax.text(new + .9, 2.62, f"{new:.0f} min: another aircraft\nparks at this stand", color=RED,
            fontsize=12, va="center", fontweight="bold")
    ax.text(end - .9, 2.62, "Take-off", color=INK, fontsize=12, va="center", ha="right", fontweight="bold")
    ax.text(end - model - 1.2, 0, "Implies the aircraft left\nafter the stand was taken",
            ha="right", va="center", color=AMBER, fontsize=11.5)
    ax.set(xlim=(-2, end + 3), ylim=(-.55, 3.05)); ax.set_yticks([])
    ax.set_xlabel("Minutes after the recorded off-block")
    ax.grid(axis="x", color=GRID, lw=.8)
    save(fig, "06-stand-bound.png")


# ---------------------------------------------------------------- figure 7
def sequence(jan):
    d = jan[jan.PHASE_mvt.eq("DEP") & jan.ADEP_mvt.eq("EDDF")].copy()
    d["day"] = d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    groups = d.groupby(["day", "RUNWAY_mvt"]).size()
    day, runway = groups.sort_values(ascending=False).index[0]
    z = d[d.day.eq(day) & d.RUNWAY_mvt.eq(runway)].sort_values(["MVT_TIME_UTC_mvt", "MVT_ID_mvt"])
    middle = len(z) // 2; z = z.iloc[middle - 6:middle + 6]
    ref = z.MVT_TIME_UTC_mvt.iloc[4]
    take = ((z.MVT_TIME_UTC_mvt - ref).dt.total_seconds() / 60).round(1).to_numpy()
    nm = ((z.AOBT_3_flt - ref).dt.total_seconds() / 60).round(1).to_numpy()
    STATS["sequence_example"] = {"airport": "EDDF", "month": "January 2025", "runway": runway,
                                 "relative_takeoff_minutes": take.tolist(),
                                 "relative_nm_offblock_minutes": [None if np.isnan(v) else v for v in nm],
                                 "selection": "Busiest Frankfurt runway-day in January; central 12 departures, ordered by time then movement ID"}
    w = STATS["sequence_example"]["weights"] = E["stack_weights"]["ordinary"]
    fig = plt.figure(figsize=(10, 9.4))
    heading(fig, "The runway sequence gives each departure its neighbours,\nand a twin without them sharpens the signal")
    fig.text(.035, .865, "Each bar runs from the NM off-block time to take-off: inputs only, no neighbour's label",
             fontsize=12.5, color=MUTED, va="top")
    ax = fig.add_axes([.15, .52, .81, .31])
    for i in range(12):
        core = 4 <= i < 8
        c = TEAL if core else "#9fb0bb"
        if np.isnan(nm[i]):
            ax.text(take[i] + .6, i, "no NM time", va="center", fontsize=10.5, color=MUTED)
        else:
            ax.barh(i, take[i] - nm[i], left=nm[i], height=.62, color=c)
        ax.scatter(take[i], i, marker="|", s=160, color=INK, lw=2, zorder=3)
    ax.invert_yaxis(); ax.set_yticks([])
    for a_, b_, label, c in [(0, 3, "4 before", MUTED), (4, 7, "4 core\ndepartures", TEAL), (8, 11, "4 after", MUTED)]:
        ax.plot([-46.5, -46.5], [a_ - .3, b_ + .3], color=c, lw=2.5, clip_on=False)
        ax.text(-47.6, (a_ + b_) / 2, label, ha="right", va="center", color=c, fontsize=12,
                fontweight="bold", clip_on=False)
    ax.set_xlim(-46, 20)
    ax.set_xlabel(f"Minutes from the first core take-off; Frankfurt runway {runway}, a busy day in January 2025")
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.annotate("take-off", (take[11], 11), xytext=(take[11] + 1.2, 9.6), fontsize=11.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 1})
    fl = fig.add_axes([.02, .01, .96, .41]); fl.axis("off"); fl.set(xlim=(0, 16), ylim=(0, 6.2))
    role_key(fl, 2.6, 5.85, ["data", "model", "combine", "output"], w=1.9, h=0.5, gap=0.9, size=10.5)
    node(fl, "data", 2.0, 4.35, 3.4, 1.25, "12 departures on one\nrunway, in take-off order")
    node(fl, "data", 2.0, 2.45, 3.4, 1.25, "Each departure\non its own")
    node(fl, "model", 6.1, 4.35, 3.2, 1.15, "Sequence network\nbidirectional GRU")
    node(fl, "model", 6.1, 2.45, 3.2, 1.15, "Row control\nno neighbours", size=12)
    node(fl, "model", 6.1, 0.7, 3.2, 0.95, "Trees + clock head\nthe base estimate", size=11)
    node(fl, "combine", 10.4, 2.45, 2.9, 3.0, "Linear stack\nper row class")
    node(fl, "output", 14.2, 2.45, 3.0, 1.05, "Taxi-out time\nfor each departure", size=12)
    arrow(fl, (3.75, 4.35), (4.45, 4.35)); arrow(fl, (3.75, 2.45), (4.45, 2.45))
    arrow(fl, (7.75, 4.35), (8.95, 3.3), color=TEAL, lw=2.2)
    arrow(fl, (7.75, 2.45), (8.95, 2.45), color=RED, lw=2.2)
    arrow(fl, (7.75, 0.7), (8.95, 1.45), color=MUTED, lw=1.6)
    arrow(fl, (11.9, 2.45), (12.65, 2.45))
    fl.text(8.25, 4.0, f"+{w['sequence_primary']:.2f}", color=TEAL, fontsize=14, fontweight="bold")
    fl.text(8.0, 2.62, f"−{abs(w['row_control']):.2f}", color=RED, fontsize=14, fontweight="bold")
    fl.text(10.4, 0.45, "The negative weight removes what\nboth networks share; the rest is\nwhat the neighbours add",
            ha="center", va="center", fontsize=11, color=MUTED)
    save(fig, "07-runway-sequence.png")


# ---------------------------------------------------------------- figure 8
def stopping():
    d = E["early_stop"]
    fig, ax = plt.subplots(figsize=(10, 4.9))
    fig.subplots_adjust(left=.14, right=.975, top=.70, bottom=.21)
    heading(fig, "Scoring the early stop only on the rows the model serves\nlet the stacker train 4 to 7 times longer")
    for i, (m, a, b) in enumerate(zip(d["fold"], d["before_rounds"], d["after_rounds"])):
        arrow(ax, (a, i), (b, i), color="#b9c2c9", lw=2.2, ms=18, shrinkA=9, shrinkB=10)
        ax.scatter(a, i, s=200, color=AMBER, zorder=3)
        ax.scatter(b, i, s=200, color=TEAL, zorder=3)
        ax.text(a, i + .26, f"{a:,}", ha="center", va="top", fontsize=13, color=AMBER, fontweight="bold")
        ax.text(b, i + .26, f"{b:,}", ha="center", va="top", fontsize=13, color=TEAL, fontweight="bold")
        ax.text((a + b) / 2, i - .13, f"× {b/a:.1f}", ha="center", fontsize=12, color=MUTED)
    ax.text(d["before_rounds"][0], -.62, "Stop scored on all rows,\nincluding pass-through rows",
            ha="center", va="bottom", fontsize=12, color=AMBER, fontweight="bold")
    ax.text(d["after_rounds"][0], -.62, "Stop scored on\nserved rows only", ha="center",
            va="bottom", fontsize=12, color=TEAL, fontweight="bold")
    fig.text(.975, .03, "Result: +89.8 s² per row locally; −0.22 s on the public board",
             ha="right", fontsize=12, color=INK)
    ax.set_yticks([0, 1], [f"Held {m}" for m in d["fold"]])
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=INK)
    ax.set(xlim=(0, 4700), ylim=(1.6, -0.75))
    ax.set_xticks([0, 1000, 2000, 3000, 4000], ["0", "1,000", "2,000", "3,000", "4,000"])
    ax.set_xlabel("LightGBM rounds at the early stop")
    ax.grid(axis="x", color=GRID, lw=.8)
    save(fig, "08-early-stopping.png")


# ---------------------------------------------------------------- figure 9
def alignment():
    s = {r["upload"]: r["rmse_seconds"] for r in E["scores"]}
    A = E["alignment"]
    gain = lambda a, b: s[a] ** 2 - s[b] ** 2
    cum = [(r["upload"], r["local"], gain(A["cumulative_parent"], r["upload"])) for r in A["cumulative"]]
    one = [(r["upload"], r["local"], gain(r["parent"], r["upload"])) for r in A["single_steps"]]
    recorded = {48: 1187.3, 49: 2261.0, 50: 3699.7, 51: 4127.6, 52: 4648.0,
                53: 127.7, 54: 86.1, 55: 100.1, 56: 84.8, 57: 49.5, 58: 273.4, 61: 27.4}
    for u, _, g in cum + one:
        assert abs(g - recorded[u]) < .06, (u, g)
    STATS["alignment"] = {"cumulative_vs_47": [{"upload": u, "local": l, "board": round(g, 1), "ratio": round(g / l, 3)} for u, l, g in cum],
                          "single_steps": [{"upload": u, "local": l, "board": round(g, 1), "ratio": round(g / l, 3)} for u, l, g in one]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    fig.subplots_adjust(left=.09, right=.975, top=.75, bottom=.14, wspace=.25)
    heading(fig, "Added up, local gains reached the public board at about 0.8\nof their size; single steps scatter on both sides")
    for ax, pts, lim, name, color in [(axes[0], cum, 6000, "Cumulative, uploads 48–52 against 47", TEAL),
                                      (axes[1], one, 300, "Single steps, uploads 53–61", AMBER)]:
        ax.plot([0, lim], [0, lim], color="#b9c2c9", lw=1.6)
        ax.plot([0, lim], [0, .83 * lim], color=MUTED, lw=1.4, ls=(0, (4, 3)))
        ax.set(xlim=(0, lim), ylim=(0, lim))
        fig.canvas.draw()
        ax.set_title(name, loc="left", fontsize=13, pad=8, fontweight="bold")
        ax.grid(color=GRID, lw=.8)
        xs = {"left": (.42, .36), "right": (.40, .72)}["left" if lim > 1000 else "right"]
        for (slope, txt, x), above in zip([(1, "board = local", xs[0]), (.83, "0.83 × local", xs[1])], (True, False)):
            ang = ax.transData.transform_angles(np.array([np.degrees(np.arctan(slope))]), np.array([[0, 0]]))[0]
            ax.text(x * lim, x * lim * slope + (1 if above else -1) * lim * .025, txt, rotation=ang,
                    rotation_mode="anchor", fontsize=11, color=MUTED, ha="left", va="bottom" if above else "top")
        for u, l, g in pts:
            ax.scatter(l, g, s=70, color=color, zorder=3)
            if u == 56:
                continue
            below = u == 54  # 54 and 56 overlap; one label under the pair, away from 55
            ax.annotate("54, 56" if below else str(u), (l, g), xytext=(4, -9) if below else (-7, 7),
                        textcoords="offset points", fontsize=11, color=INK, ha="left" if below else "right",
                        va="top" if below else "baseline")
    axes[0].set_ylabel("Public gain, s² per row")
    fig.text(.53, .03, "Local gain, s² per row; numbers are upload numbers", ha="center", fontsize=13)
    save(fig, "09-alignment.png")


FIGURES = {"01": progress, "02": clock, "03": coverage, "08": stopping, "09": alignment}
RAW = {"04", "05", "06", "07"}


def main(selected):
    for k in sorted(selected):
        if k in FIGURES:
            FIGURES[k]()
    if RAW & set(selected):
        bins, hist, jan, jul = raw_statistics()
        if "04" in selected: join_window(bins, hist)
        if "05" in selected: placeholders()
        if "06" in selected: stand(jul)
        if "07" in selected: sequence(jan)
    if set(selected) >= set(FIGURES) | RAW:
        (HERE / "descriptive-statistics.json").write_text(json.dumps(STATS, indent=2) + "\n")
        print(json.dumps({"png_count": len(list(OUT.glob("*.png"))),
                          "join_rows": STATS["join_window"]["rows"],
                          "placeholder_LIRF": STATS["placeholder"]["LIRF"],
                          "progress": STATS["progress"]}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(set(FIGURES) | RAW))
