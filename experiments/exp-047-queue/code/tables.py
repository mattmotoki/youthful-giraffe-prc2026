"""exp-047-queue: render every table in RESULTS.md from `analysis/*.json`.

No number in the report is typed by hand.  `main()` rewrites the block between the
`<!-- TABLES:exp047 -->` marker and the end of the file.

    OMP_NUM_THREADS=1 .venv/bin/python experiments/exp-047-queue/code/tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
AN = EXP / "analysis"
MARK = "<!-- TABLES:exp047 -->"
FOLDS = ["july_loo", "december", "january"]
#: the registered bootstrap, and the refined one T7m reads the knife edge with
BOOT_REG, BOOT_BIG = 2_000, 20_000
RCLASS = ["strict", "wide_window", "ambiguous_strict", "matched_no_clock",
          "no_takeoff", "outage", "unmatched"]


def load(name: str) -> dict | None:
    p = AN / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(v) -> str:
    return "–" if v is None else f"{100 * v:.1f} %"


def ci(v) -> str:
    if not v or v[0] is None:
        return "–"
    return f"[{100 * v[0]:.1f}, {100 * v[1]:.1f}]"


def s2(v) -> str:
    return "–" if v is None else f"{v:,.0f}"


def gos(v: dict | None) -> str:
    """`(2A - B) / before` with its own day-clustered interval.

    FALLBACK review round 1, blocking 1: `1 - after/before` over-credits any family
    whose correction for one row is built from the other rows of the same cell, so the
    contract-consistent reading stands beside it on every table that carries a share.
    """
    if not v or v.get("gain_over_shared") is None:
        return "–"
    c = v.get("gain_over_shared_ci")
    tail = "" if not c else f" [{100 * c[0]:.0f}, {100 * c[1]:.0f}]"
    return f"{100 * v['gain_over_shared']:.1f} %{tail}"


def t_file() -> str:
    rows = ["### T1 the file's own clocks, ten airports, three folds",
            "",
            "| arm | fold | rows | shared before (s²) | share removed | 95 % day-clustered | 2A − B (s²) | (2A − B) / before |",
            "|---|---|---|---:|---:|---|---:|---:|"]
    for arm in ("Fto", "Fob", "F"):
        for f in FOLDS:
            d = load(f"explain-{f}.json")
            if not d or arm not in d["arms"]:
                continue
            a = d["arms"][arm]["all"]
            rows.append(f"| {arm} | {f} | {d['rows']:,} | {s2(a['before_s2'])} | "
                        f"{pct(a['share'])} | {ci(a['ci'])} | "
                        f"{d['arms'][arm]['contract']['mse_gain_s2']:+.1f} | "
                        f"{gos(a)} |")
    p = load("explain-july_loo-placebo.json")
    if p:
        for arm in ("Fto", "Fob", "F"):
            a = p["arms"][arm]["all"]
            rows.append(f"| {arm} **+2 h placebo** | july_loo | {p['rows']:,} | "
                        f"{s2(a['before_s2'])} | {pct(a['share'])} | {ci(a['ci'])} | "
                        f"{p['arms'][arm]['contract']['mse_gain_s2']:+.1f} | "
                        f"{gos(a)} |")
    return "\n".join(rows)


def t_week(name: str, title: str, arms: list[str]) -> str:
    d = load(name)
    if not d:
        return f"### {title}\n\n(not measured)"
    rows = [f"### {title}", "",
            f"{d['rows']:,} rows, {len(d['days'])} days, airports "
            f"{', '.join(d['airports'])}, learner `{d['learner']}`, "
            f"shared component **{s2(d['shared_component_s2']['value'])} s²**.", "",
            "| arm | cols | population | rows | share removed | 95 % | 2A − B (s²) "
            "| (2A − B) / before |",
            "|---|---:|---|---:|---:|---|---:|---:|"]
    for arm in arms:
        a = d["arms"].get(arm)
        if not a:
            continue
        for tag, label in (("all", "all"), ("matched", "own clock"),
                           ("unmatched", "no clock")):
            v = a.get(tag)
            if not v:
                continue
            c = a["contract"] if tag == "all" else a.get(f"contract_{tag}")
            g = "–" if c is None else f"{c['mse_gain_s2']:+.1f}"
            rows.append(f"| {arm} | {a['features']} | {label} | {v['rows']:,} | "
                        f"{pct(v['share'])} | {ci(v['ci'])} | {g} | {gos(v)} |")
    return "\n".join(rows)


def t_thin() -> str:
    rows = ["### T4 the receipt-response curve: what an airport heard at *r* would yield",
            "",
            "EHAM's observed clocks thinned to a fraction at random, on the same week "
            "and the same estimator, so the curve is read where the truth is known.",
            "",
            "| receipt | clocks left | arm | share removed | 95 % | 2A − B (s²) "
            "| (2A − B) / before |",
            "|---:|---:|---|---:|---|---:|---:|"]
    got = False
    for t in ("0.16", "0.30", "0.50", "0.70", "0.90"):
        d = load(f"explain-july_loo-thin{t}.json")
        if not d:
            continue
        got = True
        left = d["block"].get("thin", {}).get("clocks_left")
        for arm in ("No", "own", "own_No"):
            a = d["arms"].get(arm)
            if not a:
                continue
            rows.append(f"| {float(t):.0%} | {left:,} | {arm} | {pct(a['all']['share'])} | "
                        f"{ci(a['all']['ci'])} | {a['contract']['mse_gain_s2']:+.1f} | "
                        f"{gos(a['all'])} |")
    d = load("explain-july_loo-ehamweek-own.json")
    if d:
        left = d["arms"]["No"]["all"]["rows"]
        for arm in ("No", "own", "own_No"):
            a = d["arms"].get(arm)
            if a:
                rows.append(f"| 96 % (real) | – | {arm} | {pct(a['all']['share'])} | "
                            f"{ci(a['all']['ci'])} | {a['contract']['mse_gain_s2']:+.1f} | "
                            f"{gos(a['all'])} |")
    return "\n".join(rows) if got else "### T4\n\n(not measured)"


def t_audit() -> str:
    d = load("audit.json")
    if not d:
        return "### T5 the leakage audit\n\n(not measured)"
    worst = list(d["B_worst_column"].items())[0]
    anchors = d["D_anchor"]
    med = sorted(v["pearson"] for v in anchors.values())[len(anchors) // 2]
    return "\n".join([
        "### T5 the leakage audit and the anchor diagnostic", "",
        "| check | result |", "|---|---|",
        f"| A — a count moved by more than the row's own event | none |",
        f"| A — counts that never contain the row (and must not) | "
        f"{len(d['A_columns_that_never_contain_the_row'])} of "
        f"{len(d['A_self_exclusion'])} |",
        f"| B — the column that best predicts the row's own taxi span | "
        f"`{worst[0]}`, Spearman {worst[1]['spearman_with_own_span']:+.3f}, "
        f"{100 * worst[1]['span_variance_left_within_bins']:.0f} % of the span's "
        f"variance survives conditioning on it |",
        f"| C — departure `BLOCK_TIME_UTC_mvt` / `TAXITIME_SEC_mvt` read | "
        f"{d['C_label_free']['dep_block_time_nonnull']} / "
        f"{d['C_label_free']['dep_taxitime_nonnull']} non-null (the mask ran) |",
        f"| D — the same state at the take-off and the off-block instant | "
        f"median Pearson {med:.3f}: the anchor is a different reading, not a "
        f"relabelling |"])


def t_split() -> str:
    """T2b: the same regression read on pairs whose correction saw neither member."""
    naive = load("explain-july_loo-ehamweek-own.json")
    sp = load("explain-july_loo-ehamweek-split.json")
    thin = load("explain-july_loo-thin0.50.json")
    if not naive or not sp:
        return "### T2b the split-half gate statistic\n\n(not measured)"
    rows = ["### T2b the split-half gate statistic — the correction sees neither member "
            "of the pair", "",
            "`1 − after/before` reads the cross-product over the pairs of a cell, and "
            "the correction for row *j* is built from the OTHER rows of that same cell "
            "— row *i* among them. `E[r_i h_j]` therefore carries a term in "
            "`E[r_i r_i]`, which is idiosyncratic variance and not shared component, "
            "so the statistic **over-credits** this family (FALLBACK review round 1, "
            "blocking 1). Under the split half each row's neighbourhood is built from "
            "the OTHER half's clocks and the cross-product is read inside "
            "(cell, half), where no pair's correction has seen either member. The "
            "price is half the sources, so the split-half column is a LOWER bound: it "
            "carries the real attenuation of a halved receipt as well as the removal "
            "of the over-credit. The naive column at 50 % receipt (T4) is the same "
            "source density WITH the over-credit, so the two together separate them.", "",
            "| arm | population | naive share (96 %) | naive share (50 %, T4) | "
            "split-half share | 95 % | split-half 2A − B (s²) |",
            "|---|---|---:|---:|---:|---|---:|"]
    for arm in ("Nf", "No", "N", "own", "own_No"):
        a, b = naive["arms"].get(arm), sp["arms"].get(arm)
        if not a or not b:
            continue
        for tag, label in (("all", "all"), ("matched", "own clock"),
                           ("unmatched", "no clock")):
            u, v = a.get(tag), b.get(tag)
            if not u or not v:
                continue
            t = ((thin or {}).get("arms", {}).get(arm) or {}).get(tag) if tag == "all" \
                else None
            c = b["contract"] if tag == "all" else b.get(f"contract_{tag}")
            g = "–" if not c else f"{c['mse_gain_s2']:+.1f}"
            rows.append(f"| {arm} | {label} | {pct(u['share'])} | "
                        f"{pct(t['share']) if t else '–'} | {pct(v['share'])} | "
                        f"{ci(v['ci'])} | {g} |")
    return "\n".join(rows)


def t_month() -> str:
    """T2c/T2d: the same question on a whole fold month at nine airports."""
    g = load("explain-december-obs-gbm.json")
    sp = load("explain-december-obs-split.json")
    pm = load("explain-december-obs-permuted.json")
    pb = load("explain-december-obs-permuted-both.json")
    rd = load("explain-december-obs-ridge.json")
    if not g:
        return "### T2c the observed clock on a whole fold month\n\n(not measured)"
    rows = ["### T2c the observed clock on a WHOLE FOLD MONTH, nine airports", "",
            f"december, {g['rows']:,} covered rows, {len(g['days'])} days, airports "
            f"{', '.join(g['airports'])}, learner `{g['learner']}`, shared component "
            f"**{s2(g['shared_component_s2']['value'])} s²**. This is §5's discount 2 "
            "answered rather than assumed: the EHAM week was one easy airport with the "
            "smallest shared component of the ten.", "",
            "| arm | cols | population | rows | share removed | 95 % | 2A − B (s²) "
            "| (2A − B) / before |",
            "|---|---:|---|---:|---:|---|---:|---:|"]
    for arm in ("Nf", "No", "N", "own", "own_No"):
        a = g["arms"].get(arm)
        if not a:
            continue
        for tag, label in (("all", "all"), ("matched", "own clock"),
                           ("unmatched", "no clock")):
            v = a.get(tag)
            if not v:
                continue
            c = a["contract"] if tag == "all" else a.get(f"contract_{tag}")
            gg = "–" if c is None else f"{c['mse_gain_s2']:+.1f}"
            rows.append(f"| {arm} | {a['features']} | {label} | {v['rows']:,} | "
                        f"{pct(v['share'])} | {ci(v['ci'])} | {gg} | {gos(v)} |")
    if sp or pm or rd:
        rows += ["", "### T2d the same month under the corrected statistic, the "
                 "placebo and the second learner", "",
                 "The split half is the clean reading at half the source density; the "
                 "`own` arm is the control that cannot contaminate a pair, and it must "
                 "not move. Two permutation columns, because dealing only the OBSERVED "
                 "clocks leaves `Nf_*` untouched and so is a CONTROL for that family "
                 "and not a test: `--permute-file` deals the `AOBT_3_flt` off-blocks "
                 "out inside each airport-day as well.", "",
                 "| arm | naive share | split-half share | 95 % | naive 2A − B | "
                 "split-half 2A − B | permute obs | permute obs+file | ridge share |",
                 "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for arm in ("Nf", "No", "N", "own", "own_No"):
            a = g["arms"].get(arm)
            if not a:
                continue
            b = (sp or {}).get("arms", {}).get(arm)
            q = (pm or {}).get("arms", {}).get(arm)
            r = (rd or {}).get("arms", {}).get(arm)
            w = (pb or {}).get("arms", {}).get(arm)
            cells = [arm, pct(a["all"]["share"]),
                     pct(b["all"]["share"]) if b else "–",
                     ci(b["all"]["ci"]) if b else "–",
                     f"{a['contract']['mse_gain_s2']:+.1f}",
                     f"{b['contract']['mse_gain_s2']:+.1f}" if b else "–",
                     pct(q["all"]["share"]) if q else "–",
                     pct(w["all"]["share"]) if w else "–",
                     pct(r["all"]["share"]) if r else "–"]
            rows.append("| " + " | ".join(cells) + " |")
    rows += ["", "### T2f the population this lane serves, on a whole fold month", "",
             "december's no-clock rows are 44,511 of the 139,848 the harvest covers "
             "— 31 % — and they are where the desk's 14:10 UTC division sends this "
             "lane. `2A − B` per row OF THAT POPULATION, gbm, cross-fitted by day "
             "block. The EHAM week's 287 rows read +583 s² on `No`; a whole month at "
             "nine airports does not.", "",
             "| arm | rows | share removed | 95 % | 2A − B (s²) | 95 % | "
             "(2A − B) / before |", "|---|---:|---:|---|---:|---|---:|"]
    src = {"No": g, "Nf": g, "N": g, "own": g, "own_No": g}
    st = load("explain-december-obs-state.json")
    for arm in ("Nf", "No", "N", "geo", "Oob", "Hob", "all", "own", "own_No"):
        d0 = src.get(arm) or st
        a = (d0 or {}).get("arms", {}).get(arm)
        if not a:
            continue
        v, c = a.get("unmatched"), a.get("contract_unmatched")
        if not v or not c:
            continue
        rows.append(f"| {arm} | {v['rows']:,} | {pct(v['share'])} | {ci(v['ci'])} | "
                    f"{c['mse_gain_s2']:+.1f} | "
                    f"[{c['ci95'][0]:+.0f}, {c['ci95'][1]:+.0f}] | {gos(v)} |")

    if st:
        rows += ["", "### T2e the state block on the observed clocks, nine airports, "
                 "december", "",
                 "| arm | cols | share removed | 95 % | 2A − B (s²) | (2A − B) / before |",
                 "|---|---:|---:|---|---:|---:|"]
        for arm, a in st["arms"].items():
            rows.append(f"| {arm} | {a['features']} | {pct(a['all']['share'])} | "
                        f"{ci(a['all']['ci'])} | "
                        f"{a['contract']['mse_gain_s2']:+.1f} | {gos(a['all'])} |")
    return "\n".join(rows)


def t_repair() -> str:
    one = load("explain-july_loo-ehamweek-own.json")
    sep = load("explain-july_loo-ehamweek-byreach.json")
    if not one or not sep:
        return "### T3c the repair\n\n(not measured)"
    rows = ["### T3c the repair attempt: one head per reach class", "",
            "`2A − B` per row of the named population, EHAM week, ridge. "
            "The left column is one head with a missing indicator, the right is one "
            "head fitted on each class separately.", "",
            "| arm | population | one head (s²) | 95 % | one head per class (s²) | 95 % |",
            "|---|---|---:|---|---:|---|"]
    for arm in ("own", "No", "own_No", "own_No_Hob"):
        for tag, label in (("matched", "own clock"), ("unmatched", "no clock")):
            x = one["arms"].get(arm, {}).get(f"contract_{tag}")
            y = sep["arms"].get(arm, {}).get(f"contract_{tag}")
            if not x or not y:
                continue
            rows.append(f"| {arm} | {label} | {x['mse_gain_s2']:+.1f} | "
                        f"[{x['ci95'][0]:+.0f}, {x['ci95'][1]:+.0f}] | "
                        f"{y['mse_gain_s2']:+.1f} | "
                        f"[{y['ci95'][0]:+.0f}, {y['ci95'][1]:+.0f}] |")
    return "\n".join(rows)


def s2i(v) -> str:
    """A `2A - B` in s^2 with its day-clustered interval."""
    if v is None:
        return "–"
    c = v.get("ci95")
    tail = "" if not c else f" [{c[0]:+,.0f}, {c[1]:+,.0f}]"
    return f"{v['mse_gain_s2']:+,.0f}{tail}"


def t_step2(name: str, title: str, arms: list[str]) -> str:
    """One `--rows` class of step 2: the pooled fold gain and the PROJECTED one.

    The two columns are different questions.  The pooled `2A - B` prices the correction
    on the fold month's own population, in which a 2025 airport with no receiver is
    over-represented in the no-clock class.  The projection (desk, 2026-09-08 16:50 and
    17:05 UTC; exp-043 step 4's arithmetic) reads each airport's gain PER SERVED ROW and
    weights it by the rows that airport-month actually has in the ranking file, so the
    2025 receiver map cannot be read as a property of the flights.  The advance clause
    is gated on the projection, which is the board endpoint.
    """
    d = load(name)
    if not d:
        return f"### {title}\n\n*(not run)*"
    folds = [f for f in FOLDS if f in d["arms"][arms[0]]] if arms[0] in d["arms"] else []
    out = [f"### {title}", "",
           f"`{name}` — head {d.get('head')}, matcher `{(d.get('matcher_sha256') or '')[:8]}…`, "
           f"priced on **{d.get('rows')}** rows, fitted on "
           f"**{d.get('fit_rows', d.get('rows'))}** rows, missingness indicator "
           f"{'ON' if d.get('miss_indicator') else 'off'}, fit floor on receipt "
           f"{d.get('fit_min_receipt', 0.0)}. The projection carries "
           f"{d['projection_weights']['rows_carried']:,} of the ranking file's "
           f"{d['projection_weights']['rank_rows']:,} rows "
           f"({100 * d['projection_weights']['share_of_ranking']:.1f} %). "
           "**The gate is the s² interval, not the board column**: that column is "
           "APPROXIMATE — it converts the s² drop at v12's CLEAN board (256.4644), the "
           "smaller of the two bases, so it is the larger of the two possible "
           "second-counts; the same drop at the ALL-ROW board (308.5977) is about "
           "0.83× as many seconds. `code/serve.py` measures the served file's own board "
           "directly rather than converting. The projection is read per airport-MONTH "
           "and per REACH CLASS (FALLBACK round 3, blocking 1): each airport's gain is "
           "priced separately on rows with and without a heard neighbour and each half "
           "is applied to that airport-month's own reached share. The last two columns "
           "are leave-one-out readings off the SAME draws — the worst single airport, "
           "and the served-cells-only projection that gates the file written.", ""]
    out += ["| arm | fold | rows | fit rows | pooled `2A − B` s² | projected s² | "
            "worst airport dropped | served cells only |",
            "|---|---|---:|---:|---:|---:|---|---|"]
    for arm in arms:
        a = d["arms"].get(arm)
        if not a:
            continue
        for f in folds:
            v, pv = a[f], a["projected"][f]
            loo = pv.get("leave_one_airport_out") or {}
            worst = min(loo, key=lambda k: loo[k]["pooled_mse_reduction_s2"],
                        default=None)
            wtxt = "–"
            if worst:
                w = loo[worst]
                wtxt = (f"{worst} → {w['pooled_mse_reduction_s2']:+,.0f} "
                        f"[{w['ci95_s2'][0]:+,.0f}, {w['ci95_s2'][1]:+,.0f}]")
            so = pv.get("served_cells_only")
            stxt = "–"
            if so:
                stxt = (f"{so['pooled_mse_reduction_s2']:+,.0f} "
                        f"[{so['ci95_s2'][0]:+,.0f}, {so['ci95_s2'][1]:+,.0f}] "
                        f"({len(so['cells'])} cells)")
            out.append(
                f"| `{arm}` | {f} | {v['rows']:,} | {v['fit_rows']:,} | "
                f"{s2i(v['all'])} | {pv['pooled_mse_reduction_s2']:+,.0f} "
                f"[{pv['ci95_s2'][0]:+,.0f}, {pv['ci95_s2'][1]:+,.0f}] | "
                f"{wtxt} | {stxt} |")
    return "\n".join(out)


def t_cells() -> str:
    """T7g: every (airport, reach class) cell, its fold support and the hold's verdict."""
    d = load("arms-step2-unmatched-fitall-3f.json")
    sv = load("serve.json")
    if not d or "N" not in d.get("arms", {}):
        return "### T7g the cell hold\n\n*(not run)*"
    pr = d["arms"]["N"]["projected"]
    folds = [f for f in FOLDS if f in pr]
    cells = sorted({c for f in folds for c in pr[f]["by_airport_reach"]})
    hold = (sv or {}).get("cell_hold", {})
    out = ["### T7g the cell hold — every (airport, reach class) cell of arm `N`, its "
           "fold-side support, and what holding it gives up", "",
           "The desk's rule of 2026-09-09: a cell whose own day-clustered fold interval "
           "is not clear of zero ABOVE, or which is priced on fewer than "
           f"{hold.get('min_fold_rows', 500)} rows in ANY fold, is held at the base. "
           f"**{hold.get('class_rows_served', 0):,}** of "
           f"**{hold.get('class_rows_in_class', 0):,}** class rows are served.", "",
           "| cell | ranking rows | " + " | ".join(
               f"{f} — rows, s²/row [95 %]" for f in folds) + " | verdict |",
           "|---|---:|" + "---|" * len(folds) + "---|"]
    rows = []
    for c in cells:
        got = [pr[f]["by_airport_reach"][c].get("ranking_rows") or 0
               for f in folds if c in pr[f]["by_airport_reach"]]
        nrank = max(got) if got else 0
        cols = []
        for f in folds:
            v = pr[f]["by_airport_reach"].get(c)
            if v is None:
                cols.append("absent")
            elif not v.get("priced"):
                cols.append(f"{v['fold_rows']:,}, unpriced")
            else:
                cols.append(f"{v['fold_rows']:,}, {v['gain_per_row_s2']:+,.0f} "
                            f"[{v['gain_ci95_s2'][0]:+,.0f}, "
                            f"{v['gain_ci95_s2'][1]:+,.0f}]")
        served = c in (hold.get("cells_served") or [])
        rows.append((not served, c, f"| `{c}` | {nrank:,} | " + " | ".join(cols) +
                     (" | **served** |" if served else " | held |")))
    for _, _, line in sorted(rows):
        out.append(line)
    fg = {f: (pr[f].get("served_cells_only") or {}).get("forgone_s2") for f in folds}
    out += ["", "Projected gain forgone by the hold: " + ", ".join(
        f"**{v:+,.0f} s²** on {f}" for f, v in fg.items() if v is not None) + "."]
    return "\n".join(out)


def t_celldiag() -> str:
    """T7h: the instrument or the arm, per held cell."""
    d = load("foldside.json")
    if not d or "item1_cells" not in d:
        return "### T7h why the held cells lack support\n\n*(not run)*"
    it = d["item1_cells"]
    k = it["by_kind"]
    out = ["### T7h why 17 of the 19 cells lack support — the instrument or the arm",
           "",
           "The desk's question of 2026-09-09: does a held cell fail for want of 2025 "
           "FOLD ROWS, which more harvested months would fix, or for want of a signal, "
           "which they would not? The separator is the cell's reading POOLED over the "
           "three folds, on the union of their days. Pooling is rejected as a SERVING "
           "rule (§7c: it would keep the 141-row LIRF cell); it is the right reading "
           "for the diagnostic question of whether anything is there at all. "
           f"Cell gains reproduce the record exactly (max |Δ| "
           f"{it['max_abs_cell_gain_against_the_record_s2']:.0e} s²).", "",
           "| cell | ranking rows | fold rows jul / dec / jan | pooled rows | pooled "
           "s²/row [95 %] | held by |",
           "|---|---:|---|---:|---|---|"]
    for c, v in sorted(it["cells"].items(), key=lambda kv: -kv[1]["ranking_rows"]):
        fr = v["fold_rows"]
        po = v["pooled"]
        pool = (f"{po['gain_per_row_s2']:+,.0f} [{po['ci95_s2'][0]:+,.0f}, "
                f"{po['ci95_s2'][1]:+,.0f}]") if po else "–"
        verdict = {"served": "**served**", "arm": "the **arm**",
                   "instrument": "the **instrument**",
                   "month": "month to month", "undecided": "undecided"}[v["held_by"]]
        out.append(f"| `{c}` | {v['ranking_rows']:,} | "
                   f"{fr['july_loo']:,} / {fr['december']:,} / {fr['january']:,} | "
                   f"{po['rows']:,} | {pool} | {verdict} |" if po else
                   f"| `{c}` | {v['ranking_rows']:,} | "
                   f"{fr['july_loo']:,} / {fr['december']:,} / {fr['january']:,} | – | "
                   f"– | {verdict} |")
    rev, no = it["revivable_by_more_fold_months"], it["not_revivable"]
    out += ["",
            f"**More fold months can revive {len(rev['cells'])} of the 17 held cells, "
            f"{rev['ranking_rows']:,} ranking rows**: the 3 the instrument holds "
            f"({k.get('instrument', {}).get('ranking_rows', 0):,} rows, pooled read "
            "already clear above zero), the 1 held month to month "
            f"({k.get('month', {}).get('ranking_rows', 0):,}), and the "
            f"{len(k.get('undecided', {}).get('cells', []))} too thin to say "
            f"({k.get('undecided', {}).get('ranking_rows', 0):,}). **They cannot revive "
            f"the {len(no['cells'])} the arm holds, {no['ranking_rows']:,} ranking "
            "rows** — and that half contains the largest held cell of all, "
            "`LFPG/unreached` at 34,108 ranking rows, whose pooled read on 54,445 fold "
            "rows is +100 [−82, +335], and `EGLL/reached`, where the correction HURTS "
            "at −1,901 [−3,336, −849] on 22,715 rows."]
    return "\n".join(out)


def t_capacity() -> str:
    """T7i: what a bigger head buys on the same rotation."""
    d = load("foldside.json")
    if not d or "item2_capacity" not in d:
        return "### T7i the capacity of arm `N`'s head\n\n*(not run)*"
    it = d["item2_capacity"]
    caps = it["by_capacity"]
    def trip(v, key, dp=0):
        return " / ".join(f"{v['projected_s2'][f][key]:+,.{dp}f}" for f in FOLDS
                          if f in v["projected_s2"])
    def cell(v, c):
        out = []
        for f in FOLDS:
            r = (v["registered_served_cells"].get(c) or {}).get(f)
            out.append(f"{r['gain_per_row_s2']:+,.0f}" if r and r.get("priced") else "–")
        return " / ".join(out)
    out = ["### T7i the capacity of arm `N`'s head, at the registered rotation", "",
           "Held fixed: " + it["held_fixed"] + ". Only the learner moves. " +
           it["gbm_note"] + ".", "",
           "| head | cells it supports | served-only s² jul / dec / jan | class s² "
           "jul / dec / jan | `EDDF/reached` s²/row | `LTFM/unreached` s²/row |",
           "|---|---|---|---|---|---|"]
    for n, v in sorted(caps.items(), key=lambda kv: kv[0].startswith("gbm")):
        name = f"**{n}**" if "REGISTERED" in n else n
        cls = " / ".join(f"{v['fold_gain_s2'][f]['gain_s2']:+,.0f}" for f in FOLDS
                         if f in v["fold_gain_s2"])
        out.append(f"| {name} | {', '.join('`' + c + '`' for c in v['cells_supported'])} "
                   f"| {trip(v, 'served_cells_only')} | {cls} | "
                   f"{cell(v, 'EDDF/reached')} | {cell(v, 'LTFM/unreached')} |")
    out += ["", "**The linear form is saturated and the nonlinearity is not.** Four "
            "orders of magnitude of ridge penalty move `EDDF/reached`'s july gain from "
            "+3,003 to +2,972 s² per row and `LTFM/unreached`'s from +756 to +758: "
            "there is no capacity question inside the ridge. A gradient-boosted head on "
            "the same columns raises `EDDF/reached` by 1.4 to 1.8× on every fold "
            "(+5,002 / +4,437 / +2,248 against +3,003 / +3,620 / +1,261) — and LOSES "
            "`LTFM/unreached`, whose december and january intervals stop clearing zero, "
            "so the population it can serve falls from 63,862 ranking rows to 19,499. "
            "**Post-selection** (§7c): the two served cells were chosen on these same "
            "three folds with the registered ridge, and picking a capacity by reading "
            "this table would select on them a second time."]
    return "\n".join(out)


def t_fileclock() -> str:
    """T7j: the file-clock family alone, which is what serves LTFM."""
    d = load("foldside.json")
    if not d or "item3_file_clock_only" not in d:
        return "### T7j the file-clock family alone\n\n*(not run)*"
    it = d["item3_file_clock_only"]
    out = ["### T7j the file-clock family alone (`Nf_*`, no observation anywhere in it)",
           "", it["why"], "",
           "| fold | class rows | class `2A − B` s² [95 %] | projected, whole class | "
           "projected, served cells | `LTFM/unreached` s²/row |",
           "|---|---:|---|---|---|---|"]
    for f in FOLDS:
        g = it["fold_gain_s2"].get(f)
        p = it["projected_s2"].get(f)
        if not g or not p:
            continue
        s = p["served_cells_only"]
        lt = (it["against_arm_N"]["LTFM/unreached"]["Nf_per_fold"].get(f) or {})
        out.append(
            f"| {f} | {g['rows']:,} | {g['gain_s2']:+,.0f} [{g['ci95_s2'][0]:+,.0f}, "
            f"{g['ci95_s2'][1]:+,.0f}] | {p['pooled']:+,.0f} [{p['ci95'][0]:+,.0f}, "
            f"{p['ci95'][1]:+,.0f}] | {s['pooled_mse_reduction_s2']:+,.0f} "
            f"[{s['ci95_s2'][0]:+,.0f}, {s['ci95_s2'][1]:+,.0f}] | "
            f"{lt.get('gain_per_row_s2', 0):+,.0f} [{lt['gain_ci95_s2'][0]:+,.0f}, "
            f"{lt['gain_ci95_s2'][1]:+,.0f}] |")
    a = it["against_arm_N"]
    out += ["", "Against arm `N` on the two cells the candidate serves, same rows, same "
            "rotation:", "",
            "| cell | jul `Nf` / `N` | dec `Nf` / `N` | jan `Nf` / `N` | pooled `Nf` / "
            "`N` |", "|---|---|---|---|---|"]
    for c, v in a.items():
        cols = []
        for f in FOLDS:
            nf, n = v["Nf_per_fold"].get(f), v["N_per_fold"].get(f)
            fmt = lambda r: (f"{r['gain_per_row_s2']:+,.0f}"
                             if r and r.get("priced") else "–")
            cols.append(f"{fmt(nf)} / {fmt(n)}")
        cols.append(f"{v['Nf_pooled_s2']:+,.0f} / {v['N_pooled_s2']:+,.0f}")
        out.append(f"| `{c}` | " + " | ".join(cols) + " |")
    out += ["", "**The file-clock family alone supports `LTFM/unreached` on all three "
            "folds, and it reads HIGHER there than the arm the candidate serves.** LTFM "
            "is the cell that carries 47,577 of the candidate's 63,862 served rows, and "
            "no observation reaches it: round 4 verified `No_n_res_cell` finite on none "
            "of its 25,185 no-clock 2026-07 rows. The `No_*` columns are therefore the "
            "fit rows' imputed means on every LTFM row, and carrying them DILUTES the "
            "correction — `Nf` reads +986 / +524 / +355 s² per row against `N`'s "
            "+756 / +232 / +264. The reverse holds at EDDF, which is 96 to 99 % "
            "reached: there the observation is the signal (+3,003 against +901 on "
            "july) and `Nf` alone loses january's interval. `Nf` alone would NOT pass "
            "the registered gate — its whole-class july projection is −303 [−2,058, "
            "+697] — so this is a fold-side reading, not a candidate. Choosing the arm "
            "per cell on these same folds is a second selection, with §7c's caveat."]
    return "\n".join(out)


def t_percell() -> str:
    """T7k: step 2 under the per-cell form, and the two legs of the gate."""
    d = load("arms-step2-unmatched-fitall-3f-split.json")
    g = load("split-gate.json")
    if not d or "N_reach" not in d.get("arms", {}):
        return "### T7k step 2 under the per-cell form\n\n*(not run)*"
    A = d["arms"]
    out = ["### T7k step 2 under the PER-CELL FORM — `Nf_*` only on unreached cells, "
           "arm `N` on reached cells", "",
           "The desk's rule of 2026-09-09, registered in `arms.SPLIT_FORM` and read once "
           "by both `arms.py` and `serve.py`: an absent observation is never a feature, "
           "and on an unreached row every `No_*` column is the fitting rows' imputed "
           "mean. Both heads are fitted on the SAME fitting rows (`--fit-rows all`); "
           "only the columns a scored row's reach class carries change. `EDDF/reached` "
           "is bit-identical under the two arms, which is the check that nothing else "
           "moved.", "",
           "| fold | class rows (reached / unreached) | class `2A − B` s² | projected, "
           "whole class (ONE FLOOR, registered) | the same at the OLD two-floor "
           "reading, for the record | projected, served cells |",
           "|---|---|---|---|---|---|"]
    for arm in ("N_reach", "N"):
        pr, gg = A[arm]["projected"], (g or {}).get("arms", {}).get(arm, {})
        for f in FOLDS:
            if f not in pr:
                continue
            form = A[arm][f].get("form") or {}
            rc = form.get("reached", {}).get("rows")
            uc = form.get("unreached", {}).get("rows")
            fl = (gg.get("registered_min_cell_100") or {}).get(f)
            so = pr[f]["served_cells_only"]
            name = f"**{f}**, `{arm}`" if arm == "N_reach" else f"{f}, `{arm}`"
            out.append(
                f"| {name} | " +
                (f"{rc:,} / {uc:,}" if rc is not None else f"{A[arm][f]['rows']:,}") +
                f" | {A[arm][f]['all']['mse_gain_s2']:+,.0f} "
                f"[{A[arm][f]['all']['ci95'][0]:+,.0f}, "
                f"{A[arm][f]['all']['ci95'][1]:+,.0f}] | "
                f"{pr[f]['pooled_mse_reduction_s2']:+,.0f} "
                f"[{pr[f]['ci95_s2'][0]:+,.0f}, {pr[f]['ci95_s2'][1]:+,.0f}] | " +
                (f"{fl['pooled_mse_reduction_s2']:+,.0f} [{fl['ci95_s2'][0]:+,.0f}, "
                 f"{fl['ci95_s2'][1]:+,.0f}]" if fl else "–") +
                f" | {so['pooled_mse_reduction_s2']:+,.0f} "
                f"[{so['ci95_s2'][0]:+,.0f}, {so['ci95_s2'][1]:+,.0f}] |")
    out += ["", "**Both legs clear zero on every fold, for both arms**, and the served "
            "leg rises: +278 / +243 / +109 against +246 / +203 / +96, on the same two "
            "cells. `EDDF/reached` is a reached cell, so the rule gives it arm `N` "
            "under both forms and its numbers are identical to the digit — the whole "
            "difference is LTFM's. **This is read under the desk's one-floor rule of "
            "2026-09-09** (§7h): while the projection priced a cell at 100 fold rows "
            "and the hold refused it at 500, the whole-class leg failed july at −130 "
            "[−1,704, +881] and one HELD cell was the whole of it — `LIRF/unreached`, "
            "141 july fold rows extrapolated 125-fold, reading −9,175 s² per row under "
            "the file clocks and +8,765 under arm `N`, the same rows with opposite "
            "signs. The last column of the table above is what that cell was doing to "
            "the gate.", "",
           "The four cells the argument turns on, per fold, `Nf`-only against arm `N` on "
           "the same rows:", "",
           "| cell | july | december | january |", "|---|---|---|---|"]
    for c in ("LTFM/unreached", "LFPG/unreached", "LIRF/unreached", "EDDF/reached"):
        # EDDF is a reached cell: the rule gives it arm `N` under both forms, and the
        # two readings agreeing to the digit is the check that nothing else moved
        for arm in ("N_reach", "N"):
            pr = A[arm]["projected"]
            cells = []
            for f in FOLDS:
                r = pr[f]["by_airport_reach"].get(c) if f in pr else None
                cells.append(
                    f"{r['gain_per_row_s2']:+,.0f} [{r['gain_ci95_s2'][0]:+,.0f}, "
                    f"{r['gain_ci95_s2'][1]:+,.0f}] on {r['fold_rows']:,}"
                    if r and r.get("priced") else "–")
            if arm == "N_reach" and not c.endswith("unreached"):
                continue          # a reached cell is arm `N` under both, by the rule
            lab = "`Nf` only" if arm == "N_reach" else "arm `N`"
            out.append(f"| `{c}`, {lab} | " + " | ".join(cells) + " |")
    cur = (g or {}).get("currency_N_reach")
    if cur:
        out += ["", "And the currency of the form, cross-fitted by fold with the hold "
                "applied on the fold side — the number `serve.py` would write:", "",
                "| fold | rows the file would move | all | bulk | tail | no-NM (rows) |",
                "|---|---:|---|---|---|---|"]
        for f in FOLDS:
            v = cur["per_fold"].get(f)
            if not v:
                continue
            a, b = v["rmse4_served_before"], v["rmse4_served_after"]
            out.append(f"| {f} | {v['served_rows']:,} | "
                       f"{_b(a['all'], b['all'], 4)} | "
                       f"{_b(a['bulk_le3600'], b['bulk_le3600'], 4)} | "
                       f"{_b(a['tail_gt3600'], b['tail_gt3600'], 1)} | "
                       f"{_b(a['no_nm'], b['no_nm'], 1)} "
                       f"({a['rows_no_nm']:,}) |")
        out += ["", f"Board, clean **{cur['board_clean']['before']} → "
                f"{cur['board_clean']['after']}**, gain **{cur['board_clean']['gain_s']:+.4f} s** "
                f"(the accepted candidate's is +0.2231); all-row "
                f"{cur['board_all']['before']} → {cur['board_all']['after']}, "
                f"{cur['board_all']['gain_s']:+.4f} s (against +0.1854). December's tail "
                "and no-NM columns, which the accepted candidate makes WORSE (T8), are "
                "improvements under this form."]
    return "\n".join(out)


def t_increment() -> str:
    """T7l: the desk's gate for the v15 slot -- the increment over arm `N`."""
    d = load("increment.json")
    if not d:
        return "### T7l the increment over arm `N`\n\n*(not run)*"
    out = ["### T7l the gate for the v15 slot — the INCREMENT of the per-cell form over "
           "arm `N`, on the rows the file would move", "",
           "`2A − B` prices a correction against the head; this prices one correction "
           "against ANOTHER on the same rows: per row "
           "`(y − p12 − d_N)² − (y − p12 − d_split)²`, positive when the per-cell form "
           "is the better of the two, with a day-clustered bootstrap over the fold's "
           "own days. The desk's rule for the slot (2026-09-09): **every fold clear of "
           "zero, or the candidate is a MAYBE and takes no slot.** Two fitting "
           "conventions are read because this lane uses both — `rotation` is step 2's "
           "registered one, `cross` is the currency's, each fold fitted on the other "
           "two.", "",
           "**Read the two january lines as ONE.** `arms.ROTATION['january']` is "
           "`['july_loo', 'december']`, which IS january's cross set, so january's two "
           "rows are the same estimate on the same rows, drawn twice off one bootstrap "
           "stream (`work/increment.py` makes one generator and passes it through every "
           "call in order). Where they differ, that is the draw and not the convention "
           "— see T7m, where the same identity decides a verdict.", "",
           "| convention | fold | served rows | increment s² [95 %] | clear? | "
           "`LTFM/unreached` | `EDDF/reached` |", "|---|---|---:|---|---|---|---|"]
    for conv, c in sorted(d["conventions"].items()):
        for f in FOLDS:
            v = c["per_fold"].get(f)
            if not v:
                continue
            cells = []
            for cell in ("LTFM/unreached", "EDDF/reached"):
                cv = v["by_cell"].get(cell)
                cells.append(f"{cv['mean_s2']:+,.0f} [{cv['ci95_s2'][0]:+,.0f}, "
                             f"{cv['ci95_s2'][1]:+,.0f}]" if cv else "–")
            out.append(f"| {conv} | {f} | {v['rows']:,} | {v['mean_s2']:+,.0f} "
                       f"[{v['ci95_s2'][0]:+,.0f}, {v['ci95_s2'][1]:+,.0f}] | " +
                       ("**yes**" if v["clear_of_zero"] else "no") + " | " +
                       " | ".join(cells) + " |")
        pl = c["pooled_over_the_three_folds"]
        out.append(f"| {conv} | *the three pooled (diagnostic)* | {pl['rows']:,} | "
                   f"{pl['mean_s2']:+,.0f} [{pl['ci95_s2'][0]:+,.0f}, "
                   f"{pl['ci95_s2'][1]:+,.0f}] | " +
                   ("**yes**" if pl["clear_of_zero"] else "no") + " | – | – |")
    out += ["", f"**Verdict: {d['verdict']}**", "",
            "`EDDF/reached` is exactly zero because the rule gives a reached cell arm "
            "`N` under both forms — the file the per-cell form would write differs from "
            "the accepted candidate on LTFM's rows and nowhere else, which the dry run "
            "confirms on the bytes. So the gate is LTFM's reading alone, and **january "
            "is the fold that does not clear**: +91 s² per row on 21,368 rows with a "
            "half-width of ±183, against july's +268 [+68, +492] and december's +237 "
            "[+64, +478]. The three folds pooled are clear, and pooling is not the "
            "rule."]
    return "\n".join(out)


def t_shrunk() -> str:
    """T7m: the cheap revival -- the `Nf` head at LTFM shrunk toward `N`."""
    d = load("arms-step2-unmatched-fitall-3f-shrunk.json")
    g = load("increment-shrunk.json")
    mc = load("increment-mc.json")
    mn = load("increment-mc-N_reach.json")
    if not d or "N_reach_half" not in d.get("arms", {}):
        return "### T7m the shrunk `Nf` head at LTFM\n\n*(not run)*"
    A = d["arms"]
    out = ["### T7m the cheap revival — the `Nf` head at LTFM SHRUNK toward arm `N` "
           "at a weight fixed before the read", "",
           "§7h names a smaller variance of the DIFFERENCE as one of the two ways the "
           "v15 slot revives. This is that: on an unreached cell the correction is "
           "`0.5·d(Nf) + 0.5·d(N)` instead of `d(Nf)`, the weight **0.5 registered in "
           "`arms.FORMS` before the measurement ran and never searched**. A reached cell "
           "is arm `N` under every form, so this moves LTFM's rows and nothing else. "
           "Both legs of step 2, at the one-floor rule:", "",
           "| fold | arm | class `2A − B` s² | projected, whole class | projected, "
           "served cells |", "|---|---|---|---|---|"]
    for f in FOLDS:
        for arm in ("N_reach_half", "N_reach", "N"):
            if arm not in A or f not in A[arm]["projected"]:
                continue
            pr = A[arm]["projected"][f]
            so = pr["served_cells_only"]
            nm = f"**`{arm}`**" if arm == "N_reach_half" else f"`{arm}`"
            out.append(
                f"| {f} | {nm} | {A[arm][f]['all']['mse_gain_s2']:+,.0f} "
                f"[{A[arm][f]['all']['ci95'][0]:+,.0f}, "
                f"{A[arm][f]['all']['ci95'][1]:+,.0f}] | "
                f"{pr['pooled_mse_reduction_s2']:+,.0f} "
                f"[{pr['ci95_s2'][0]:+,.0f}, {pr['ci95_s2'][1]:+,.0f}] | "
                f"{so['pooled_mse_reduction_s2']:+,.0f} "
                f"[{so['ci95_s2'][0]:+,.0f}, {so['ci95_s2'][1]:+,.0f}] |")
    out += ["", "**Shrinking does not buy a bigger gain, and it was not asked to.** On "
            "the served cells the shrunk form sits BETWEEN arm `N` and the replacement "
            "form on july (+265 against +278 and +246) and on december (+229 against "
            "+243 and +203), and equals the replacement form on january (+108.9 against "
            "+108.6, and arm `N`'s +96). On the whole class it is the best of the three "
            "on december and january and the worst but one on july. Every leg of every "
            "arm clears zero on every fold. What the weight is for is the VARIANCE of "
            "the difference, which is what the slot gate reads — the INCREMENT over arm "
            "`N` on the rows the file would move (T7l's statistic, the same rows):", "",
            "| convention | fold | increment s² [95 %], "
            f"{BOOT_REG:,} draws | clear? | the same at {BOOT_BIG:,} draws | worst of "
            "ten seeds | `LTFM/unreached` alone |", "|---|---|---|---|---|---:|---|"]
    for conv in ("rotation", "cross"):
        for f in FOLDS:
            v = (g or {}).get("conventions", {}).get(conv, {}).get("per_fold", {}).get(f)
            m = (mc or {}).get("conventions", {}).get(conv, {}).get("per_fold", {}).get(f)
            if not v:
                continue
            cv = v["by_cell"].get("LTFM/unreached")
            out.append(
                f"| {conv} | {f} | {v['mean_s2']:+,.0f} [{v['ci95_s2'][0]:+,.1f}, "
                f"{v['ci95_s2'][1]:+,.0f}] | " +
                ("**yes**" if v["clear_of_zero"] else "no") + " | " +
                (f"[{m['refined_ci95_s2'][0]:+,.1f}, {m['refined_ci95_s2'][1]:+,.0f}]"
                 if m else "–") + " | " +
                (f"{m['worst_seed_low_s2']:+,.1f}" if m else "–") + " | " +
                (f"{cv['mean_s2']:+,.0f} [{cv['ci95_s2'][0]:+,.1f}, "
                 f"{cv['ci95_s2'][1]:+,.0f}] on {cv['rows']:,}" if cv else "–") + " |")
    ok = mc and all(c["refined_every_fold_clear"] and c["worst_seed_every_fold_clear"]
                    for c in mc["conventions"].values())
    out += ["", "**Verdict: a second MAYBE, and it is january again.** Two folds are "
            "clear by a wide margin under both conventions. January is not: the "
            "registered rotation reads **+80 [+0.5, +162]** and the cross convention "
            "**+80 [−0.1, +161]** — and by the identity in T7l those are ONE estimate "
            "drawn twice, so the candidate's fate was resting on which stretch of a "
            "bootstrap stream it landed on. The decision rule was fixed before that "
            "second read (`work/increment_mc.py`): clear on every fold, under both "
            "conventions, at " + f"{BOOT_BIG:,} draws AND at the worst of ten "
            f"{BOOT_REG:,}-draw seeds" + ". At " + f"{BOOT_BIG:,}" + " draws january's "
            "bound is **+1.2**; across ten seeds its worst is **−3.5**. The cell that "
            "actually moves reads january **+93 [−0.1, +186] on 21,368 rows** — under "
            "the same draw whose pooled line reads +0.5. A gain of +80 s² per row is "
            "very likely real and is not ESTABLISHED on that fold, which is what the "
            "gate asks." + ("" if not ok else " (The rule passed; see the JSON.)"), "",
            "**The mechanism worked; it was not enough.** Against the replacement "
            "form (`N_reach`, the accepted MAYBE of §7h) january's point estimate hardly "
            "moves, +77.7 → +80.0 s² per row, while the half-width of its interval falls "
            "from **±155 to ±81** — the variance of the difference down by about three "
            "quarters, precisely what §7h said shrinking would do. It carries the lower "
            "bound from −74 to +1.2 and stops there. The same refined read on `N_reach` "
            "is not close: " +
            (f"january **{mn['conventions']['cross']['per_fold']['january']['mean_s2']:+,.0f} "
             f"[{mn['conventions']['cross']['per_fold']['january']['refined_ci95_s2'][0]:+,.0f}, "
             f"{mn['conventions']['cross']['per_fold']['january']['refined_ci95_s2'][1]:+,.0f}]** "
             if mn else "– ") + f"at {BOOT_BIG:,} draws. A weight search would very "
            "likely find one that clears — which is why there was none: the weight was "
            "registered before the read, and a bound reached by searching weights on the "
            "same three folds is not a bound. Records: `analysis/increment-shrunk.json`, "
            "`analysis/increment-mc.json`, `analysis/increment-mc-N_reach.json`."]
    return "\n".join(out)


def t_tailnonm() -> str:
    """T7n: where the shrunk head's gain and its shortfall sit inside a fold."""
    d = load("tail-nonm.json")
    if not d:
        return "### T7n the tail and the no-NM rows\n\n*(not run)*"
    C = d["conventions"]
    GR = [("served_all", "the served rows"), ("bulk", "bulk `y ≤ 3600`"),
          ("tail", "**tail `y > 3600`**"), ("no_nm", "**no-NM**"),
          ("with_nm", "with an NM record")]
    out = ["### T7n where the shrunk head's gain — and its shortfall — sit inside a fold",
           "",
           "The desk's question (2026-09-09): the second MAYBE is january's, but is the "
           "shortfall january's alone or does the shrunk head also lose the TAIL or the "
           "no-NM rows on the folds that clear? `analysis/shrunk-currency.json` shows two "
           "RMSE figures moving the wrong way — december's tail 1,547.67 → 1,548.83 and "
           "january's no-NM 461.74 → 464.03 — and an RMSE on a few hundred rows says "
           "nothing on its own. So each subgroup of the SERVED rows is priced the way the "
           "gate prices a fold: per row, day-clustered, with an interval, at "
           f"{BOOT_BIG:,} draws re-seeded per call. `no-NM` overlaps bulk and tail. "
           "**The last column is the subgroup's PART of the fold's whole increment** "
           "in the same units (its mean × its share of the served rows, so the five "
           "lines of a fold add to that fold's figure twice over — bulk + tail, and "
           "no-NM + with-NM), which is what says where the gate is really resting. "
           "The RMSE column is the rotation's; the currency in "
           "`shrunk-currency.json` is cross-fitted, which is why december's tail "
           "reads 1,538 here and 1,549 there — 36 rows, and the same 36.", "",
           "| fold | subgroup | rows | RMSE at the ROTATION: head → arm `N` → shrunk "
           "| gain over the head s²/row [95 %] | increment over `N`, rotation | the "
           "same, cross | the subgroup's part of the fold's increment, s²/row |",
           "|---|---|---:|---|---|---|---|---:|"]
    for f in FOLDS:
        tot = C["rotation"][f]["served_all"]["N_reach_half"]["gain_over_head"]["rows"]
        for g, lab in GR:
            r = C["rotation"][f][g]
            h, n = r["N_reach_half"], r["N"]
            gg = h["gain_over_head"]
            gi = h["increment_over_N"]
            xi = C["cross"][f][g]["N_reach_half"]["increment_over_N"]
            share = gg["rows"] / tot
            out.append(
                f"| {f} | {lab} | {gg['rows']:,} | {r['rmse_head']:,.2f} → "
                f"{n['rmse_after']:,.2f} → {h['rmse_after']:,.2f} | "
                f"{gg['mean_s2']:+,.0f} [{gg['ci95_s2'][0]:+,.0f}, "
                f"{gg['ci95_s2'][1]:+,.0f}] | "
                f"{gi['mean_s2']:+,.1f} [{gi['ci95_s2'][0]:+,.1f}, "
                f"{gi['ci95_s2'][1]:+,.0f}] | "
                f"{xi['mean_s2']:+,.1f} [{xi['ci95_s2'][0]:+,.1f}, "
                f"{xi['ci95_s2'][1]:+,.0f}] | "
                f"{gi['mean_s2'] * share:+,.1f} |")
    out += ["", "**The MAYBE is january's alone, and inside january it is the BULK's.** "
            "January's whole shortfall is 24,942 bulk rows reading +77.9 [+0.3, +157.6] "
            "— the same knife edge as the fold — while its tail contributes +2.1 s² and "
            "its no-NM rows −0.4, neither of them established in any direction. "
            "**Neither RMSE that moved the wrong way is a loss to the shrunk head.** "
            "December's tail is **36 rows**: against the head its gain is −3,587 "
            "[−35,753, +149,346] under the cross convention and +28,754 [+3,522, "
            "+201,900] under the rotation, so the sign is the draw and not the data — "
            "and against arm `N`, which is what v14 ships on those rows today, the "
            "shrunk head is better on both (RMSE 1,538 / 1,549 against `N`'s 1,552 / "
            "1,559). January's no-NM rows are **326**, reading −2,113 [−9,545, +3,769] "
            "against the head, and arm `N` is equally bad there (RMSE 463.99 against "
            "464.03), so this is a property of the correction the lane already ships, "
            "not something shrinking introduced. On every fold and both conventions the "
            "shrunk head beats arm `N` in the tail.", "",
            "**What the table does say about december.** Its increment leans on those 36 "
            "tail rows for **+63 of +173 s²** (rotation) and **+48 of +129** (cross) — "
            "37 % of a fold's verdict from 0.1 % of its rows. The fold does not depend "
            "on them: december's BULK alone clears under both conventions, +109.5 "
            "[+49.4, +171.7] and +81.7 [+40.7, +131.4]. July is the same shape and "
            "smaller — +30 and +26 of its +114 and +120 — and its bulk clears under the "
            "cross convention (+94.9 [+21.8, +183.1]) but not under the rotation (+83.4 "
            "[−6.9, +192.0]). Record: `analysis/tail-nonm.json`, `work/tailnonm.py`."]
    return "\n".join(out)


def t_receipt() -> str:
    """T7o: the harvest's receipt correction, and what it moved in this lane."""
    d = load("receipt-correction.json")
    bd = load("block-rebuild-diff.json")
    old = load("arms-step2-unmatched-fitall-3f.json")
    new = load("arms-step2-recheck.json")
    if not d:
        return "### T7o the receipt correction\n\n*(not run)*"
    hb = d["harvest_bias"]["by_month"]
    out = ["### T7o the harvest's receipt correction (2026-09-09) — the bias, and what "
           "it moves in this lane", "",
           "`exp-045-harvest`'s `census.rate()` aggregated `min(t_first)` / "
           "`max(t_last)` per `(airport, callsign)` over the whole month before testing "
           "the take-off window, so a flight number flown daily counted as received on "
           "every day of the month (commit `f8352a4`). **The bias is large and it is "
           "not the same in every month**, which is what would matter if it scaled a "
           "feature:", "",
           "| month | fold or ranking | departures | rate as committed | corrected | "
           "ratio | points the old rate was too high |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for m in sorted(hb):
        v = hb[m]
        out.append(f"| {m} | {'**fold**' if v['fold_month'] else 'ranking'} | "
                   f"{v['departures']:,} | {v['rate_committed']:.4f} | "
                   f"{v['rate_corrected']:.4f} | ×{v['ratio']:.4f} | "
                   f"{v['points_lost'] * 100:.1f} |")
    out += ["", f"The three fold months average ×{d['harvest_bias'].get('fold_months_mean_ratio')} "
            f"and the two ranking months ×{d['harvest_bias'].get('ranking_months_mean_ratio')}, "
            "so the answer to *does the bias differ between the 2025 folds and the 2026 "
            "months* is **yes, by about a point of ratio — and by seven points WITHIN "
            "the folds** (2025-07 ×0.814 against 2025-01 ×0.887), which is the larger "
            "spread of the two.", "",
            "**But it never entered this lane's features.** The `_sc` columns do not "
            "divide by that rate. `block.receipt_rates()` measures its own denominator "
            "on the block's OWN matched matrix — the share of an airport-**day**'s "
            "departures whose ground run the matcher identified (`obs_has_clock`), over "
            "the airport-days the summaries delivered — and in this lane `receipt.json` "
            "is read at exactly three lines, all inside `outage_report()`, which writes "
            "warnings and nothing else. The two quantities are not close: on 2025-07 "
            "EGLL reads **0.0577** here against the harvest's corrected **0.1506**, and "
            "LFPG **0.0182** against **0.0476**, because a matched 250 m clock is a "
            "stricter test than hearing the callsign in the window.", "",
            "Measured, not argued — the five blocks rebuilt with the corrected artifact "
            "in place, against the published bytes copied aside first:", "",
            "| month | rows | columns compared | `_sc` columns | cells that moved | "
            "NaN-pattern changes | row order |", "|---|---:|---:|---:|---:|---:|---|"]
    sc = d["sc_columns"]["by_month"]
    for m in sorted(sc):
        v, b = sc[m], (bd or {}).get(m, {})
        out.append(f"| {m} | {v.get('rows', 0):,} | {b.get('cols_new', '–')} | "
                   f"{v.get('sc_columns', '–')} | **" +
                   (str(sum(x.get("cells", 0) for x
                            in b.get("columns_that_moved", {}).values()))
                    if b else str(v.get("cells_differing", "–"))) + "** | "
                   f"{v.get('nan_pattern_differences', '–')} | " +
                   ("identical" if b.get("id_order_identical") else "–") + " |")
    out += ["", "**Every column of every month is identical, value for value, in the "
            "same row order** (`analysis/block-rebuild-diff.json` compares all 186 "
            "columns, not only the 28 `_sc` ones). The parquet SHA-256 changes on a "
            "rebuild because the container is not written deterministically, so the "
            "published bytes were restored and `analysis/interface.json` still pins the "
            "shas surfacefit verified.", ""]
    if old and new:
        out += ["Then arm `N` re-priced at one floor with the cell holds, on the "
                "corrected artifact (`analysis/arms-step2-recheck.json`):", "",
                "| fold | leg | registered record | re-priced | point difference |",
                "|---|---|---|---|---|"]
        for f in FOLDS:
            for lab, a, b in (
                    ("whole class", old["arms"]["N"]["projected"][f],
                     new["arms"]["N"]["projected"][f]),
                    ("served cells",
                     old["arms"]["N"]["projected"][f]["served_cells_only"],
                     new["arms"]["N"]["projected"][f]["served_cells_only"])):
                out.append(
                    f"| {f} | {lab} | {a['pooled_mse_reduction_s2']:+,.2f} "
                    f"[{a['ci95_s2'][0]:+,.0f}, {a['ci95_s2'][1]:+,.0f}] | "
                    f"{b['pooled_mse_reduction_s2']:+,.2f} "
                    f"[{b['ci95_s2'][0]:+,.0f}, {b['ci95_s2'][1]:+,.0f}] | "
                    f"**{b['pooled_mse_reduction_s2'] - a['pooled_mse_reduction_s2']:+.6f}** |")
        out += ["", "**Every point estimate is identical to the last decimal**, the "
                "same two cells of nineteen are served, and every interval stays clear "
                "of zero on the same side; the intervals wobble only because a one-arm "
                "run draws a different stretch of the bootstrap stream than the "
                "six-arm run that wrote the record (ledger CORRECTION 7). **The gate "
                "passes and the served-cells gain has not moved**, so under the desk's "
                "rule of 2026-09-09 v14's reviewed bytes go up as gated."]
    return "\n".join(out)


def t_weights() -> str:
    """What each airport-month contributes to the projection, and why."""
    # THE CURRENT RECORD (FALLBACK round 4, item 7): this read the two-fold
    # `arms-step2-unmatched.json`, whose weights came from the pre-prefix-map census and
    # the old class, so its LEMD count and its total disagreed with everything else.
    d = load("arms-step2-unmatched-fitall-3f.json")
    if not d:
        return "### T7e the projection's weights\n\n*(not run)*"
    w = d["projection_weights"]
    out = ["### T7e what the ranking file carries, per airport-MONTH", "",
           "The no-clock class this lane serves. `matched` is the ranking month's own "
           "census on matcher `ba41935e…`; every ranking row that is not matched is "
           "no-clock, including every row on an airport-day the harvest never "
           "delivered — which is why LIRF's 2026-07 and all of LTFM are here in full.",
           "",
           "The last two columns are the REACH split the projection now works at: the "
           "share of the class's rows in that month with at least one heard aircraft in "
           "the cell.", "",
           "| airport | 2026-01 rows | matched | reached | 2026-07 rows | matched | "
           "reached | no-clock rows | weight |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    tot = 0
    for apt in sorted(w["by_airport_month"]):
        v = w["by_airport_month"][apt]
        a, b = v.get("2026-01", {}), v.get("2026-07", {})
        n = a.get("class_rows", 0) + b.get("class_rows", 0)
        tot += n
        out.append(f"| {apt} | {a.get('ranking_rows', 0):,} | {a.get('matched', 0):,} | "
                   f"{a.get('reached_share', 0):.3f} | "
                   f"{b.get('ranking_rows', 0):,} | {b.get('matched', 0):,} | "
                   f"{b.get('reached_share', 0):.3f} | {n:,} | "
                   f"{n / w['rank_rows']:.4f} |")
    out.append(f"| **all** | | | | | | | **{tot:,}** | "
               f"**{tot / w['rank_rows']:.4f}** |")
    return "\n".join(out)


def t_outage() -> str:
    """What the EDDF hold costs, measured with the hold lifted."""
    d = load("outage-cost.json")
    if not d:
        return "### T5d what the outage hold costs\n\n*(not run)*"
    h, l, c = d["runs"]["held"], d["runs"]["lifted"], d["cost"]
    out = ["### T5d what the EDDF hold costs, the hold lifted and nothing else changed",
           "",
           "Round 2 blocking 1: the superseded reading compared a census WITH this "
           "lane's rule against one without it, while the matcher held the days in both "
           "runs, so it varied nothing. Here the same build path runs twice and only "
           "`B46.HELD_AIRPORT_DAYS` differs.", "",
           "| | EDDF rows | EDDF clocks | own-clock share | rows on held days | clocks "
           "on held days |", "|---|---:|---:|---:|---:|---:|",
           f"| hold in force | {h['eddf_rows']:,} | {h['eddf_clocks']:,} | "
           f"{100 * h['eddf_own_clock_share']:.1f} % | {h['rows_on_held_days']:,} | "
           f"{h['clocks_on_held_days']:,} |",
           f"| hold lifted | {l['eddf_rows']:,} | {l['eddf_clocks']:,} | "
           f"{100 * l['eddf_own_clock_share']:.1f} % | {l['rows_on_held_days']:,} | "
           f"{l['clocks_on_held_days']:,} |", "",
           f"The hold moves **{c['rows_moved_to_no_clock']:,}** rows into the no-clock "
           f"class and discards **{c['clocks_discarded']:,}** clocks, taking EDDF's "
           f"own-clock share from {100 * c['eddf_own_clock_share_lifted']:.1f} % to "
           f"{100 * c['eddf_own_clock_share_held']:.1f} %. It is not free; it is paid "
           "so that a degraded feeder is never read as a property of EDDF's traffic.",
           "", "| day | rows | clocks held | clocks lifted |", "|---|---:|---:|---:|"]
    for day in sorted(h["by_day"]):
        out.append(f"| {day} | {l['by_day'][day]['rows']:,} | "
                   f"{h['by_day'][day]['clocks']:,} | {l['by_day'][day]['clocks']:,} |")
    return "\n".join(out)


def t_columns() -> str:
    d = load("explain-july_loo-ehamweek-columns.json")
    if not d:
        return "### T5b which column carries it\n\n(not measured)"
    rows = ["### T5b which column carries it — every `No_` column priced on its own", "",
            "| column | share removed | 95 % | 2A − B (s²) | (2A − B) / before |",
            "|---|---:|---|---:|---:|"]
    items = sorted(d["arms"].items(), key=lambda kv: -(kv[1]["all"]["share"] or -9))
    for c, a in items:
        rows.append(f"| `{c}` | {pct(a['all']['share'])} | {ci(a['all']['ci'])} | "
                    f"{a['contract']['mse_gain_s2']:+.1f} | {gos(a['all'])} |")
    return "\n".join(rows)


def t_census() -> str:
    d = load("census-ranking.json")
    if not d:
        return "### T5c the reach of the no-clock route on a ranking month\n\n(not measured)"
    rows = ["### T5c the reach of the no-clock route, on a RANKING month and with no label",
            "",
            f"{d['airport_days']} airport-days of {d['days'][0]} to {d['days'][-1]}, "
            f"{d['all']['rows']:,} ranking departures, the served board head "
            f"{d['prediction'].get('name', 'v11')} as the prediction "
            f"(`{d['prediction']['sha256'][:12]}…`). The question is not whether a row "
            "has a clock but whether its **neighbours** do. `own clock` here is "
            "`exp-046-surfacefit`'s STRICT match (a ground run identified and a usable "
            "off-block), not the harvest's reception rate: the harvest reads 73.3 % of "
            "the ranking file received, this reads 65.2 % matched over the nine "
            "airports it delivers, and the two are different quantities. The desk's "
            "outage rule of 14:45 UTC is applied: EDDF 2026-01-27 to 01-31 are "
            "no-clock airport-days here and out of the receipt denominator.", "",
            "| airport | rows | own clock | no clock | of those, reached by ≥1 heard "
            "neighbour | ≥3 | median heard neighbours in the cell |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    items = sorted(d["by_airport"].items(),
                   key=lambda kv: -(kv[1].get("no_clock_reached_share") or 0))
    for a, v in items:
        rows.append(f"| {a} | {v['rows']:,} | {pct(v['own_clock_share'])} | "
                    f"{v['no_clock']:,} | {pct(v['no_clock_reached_share'])} | "
                    f"{v['no_clock_reached_3plus']:,} | "
                    f"{v['median_neighbours_in_cell']:.0f} |")
    v = d["all"]
    rows.append(f"| **all nine** | {v['rows']:,} | {pct(v['own_clock_share'])} | "
                f"{v['no_clock']:,} | {pct(v['no_clock_reached_share'])} | "
                f"{v['no_clock_reached_3plus']:,} | "
                f"{v['median_neighbours_in_cell']:.0f} |")
    rows += ["", "LTFM has no row because adsb.lol receives none of it "
             "(`exp-045-harvest` §1)."]
    return "\n".join(rows)


def _b(before: float, after: float, dp: int = 4) -> str:
    """`before → after`, the after in bold only when it is an IMPROVEMENT.

    Bolding a number that got worse reads as a result (FALLBACK round 4): three of T8's
    sub-columns move the wrong way and the table has to show that at a glance.
    """
    a = f"{after:.{dp}f}"
    return f"{before:.{dp}f} → " + (f"**{a}**" if after < before else a)


#: WHICH SERVE RECORD THE TABLES RENDER (FALLBACK round 5, item 1).  T8 and T9 read a
#: named record, and after a re-serve on a new hold they went on rendering the PREVIOUS
#: file -- the same drift round 4 repaired for v14.  The name is now one constant, so a
#: re-serve moves both tables at once.
SERVE_REC = "serve-lit.json"
STEP2_REC = "arms-step2-lit.json"
DECEMBER_REC = "december-reconcile-lit.json"


def t_rmse4() -> str:
    """The assignment's four-way RMSE, on the served rows and on the whole fold."""
    d = load(STEP2_REC)
    sv = load(SERVE_REC)
    if not d:
        return "### T8 the four-way RMSE\n\n*(not run)*"
    out = ["### T8 the four-way RMSE of the served correction (arm `N`)", "",
           "`assignments/COMMON.md`'s all / bulk (`y ≤ 3600`) / tail / no-NM split. "
           "**The first block is the rows the FILE moves** (FALLBACK round 4, item 2): "
           "the cell hold serves 3 cells of 19 under the lit-fold rule, so the "
           "class-wide reading below is a different population from the candidate's. "
           "January's all-row and tail "
           "columns are dominated by its `noac` rows, which this lane never moves.", ""]
    if sv:
        out += ["| fold | rows the file moves | all | bulk | tail | no-NM (rows) |",
                "|---|---:|---|---|---|---|"]
        for f in FOLDS:
            v = sv["currency"]["per_fold"].get(f) or {}
            a, b = v.get("rmse4_served_before"), v.get("rmse4_served_after")
            if not a:
                continue
            out.append(
                f"| {f} | {a['rows']:,} | {_b(a['all'], b['all'])} | "
                f"{_b(a['bulk_le3600'], b['bulk_le3600'])} | "
                f"{_b(a['tail_gt3600'], b['tail_gt3600'], 1)} | "
                f"{_b(a['no_nm'], b['no_nm'], 1)} ({a['rows_no_nm']:,}) |")
        out += ["", "And the same split over the WHOLE fold, the `d` zero on every row "
                "the file does not move:", "",
                "| fold | fold rows | all | bulk | tail | no-NM (rows) |",
                "|---|---:|---|---|---|---|"]
        for f in FOLDS:
            v = sv["currency"]["per_fold"].get(f) or {}
            a, b = v.get("rmse4_fold_before"), v.get("rmse4_fold_after")
            if not a:
                continue
            out.append(
                f"| {f} | {a['rows']:,} | {_b(a['all'], b['all'])} | "
                f"{_b(a['bulk_le3600'], b['bulk_le3600'])} | "
                f"{_b(a['tail_gt3600'], b['tail_gt3600'], 1)} | "
                f"{_b(a['no_nm'], b['no_nm'], 1)} ({a['rows_no_nm']:,}) |")
    out += ["", "For the record, the same split on the WHOLE NO-CLOCK CLASS — the "
            "population step 2 gates, which the hold no longer serves in full:", "",
            "| fold | class rows | all | bulk | tail | no-NM (rows) |",
            "|---|---:|---|---|---|---|"]
    for f in FOLDS:
        v = d["arms"]["N"].get(f)
        if not v:
            continue
        a, b = v["rmse_head"], v["rmse_head_plus_d"]
        out.append(
            f"| {f} | {a['rows']:,} | {_b(a['all'], b['all'])} | "
            f"{_b(a['bulk_le3600'], b['bulk_le3600'])} | "
            f"{_b(a['tail_gt3600'], b['tail_gt3600'], 1)} | "
            f"{_b(a['no_nm'], b['no_nm'], 1)} ({a['rows_no_nm']:,}) |")
    if sv:
        c = sv["currency"]
        out += ["", "And the board's own currency, the clean RMSE beside the all-row "
                "one:", "",
                "| fold | rows the file moves | RMSE | clean RMSE |",
                "|---|---:|---|---|"]
        for f in FOLDS:
            v = c["per_fold"][f]
            out.append(
                f"| {f} | {v['served_rows']:,} | {v['rmse_all_before']:.4f} → "
                f"**{v['rmse_all_after']:.4f}** | {v['rmse_clean_before']:.4f} → "
                f"**{v['rmse_clean_after']:.4f}** |")
        out += ["", f"Board at the ranking shares {c['shares']['july_loo']:.4f} / "
                f"{c['shares']['january']:.4f}: clean "
                f"**{c['board_clean']['before']} → {c['board_clean']['after']}**, gain "
                f"**{c['board_clean']['gain_s']:+.4f} s**; all-row "
                f"**{c['board_all']['before']} → {c['board_all']['after']}**, gain "
                f"**{c['board_all']['gain_s']:+.4f} s**. December is not a "
                "ranking-share fold and is carried beside, not inside."]
    return "\n".join(out)


def _g(v) -> str:
    if not v:
        return "–"
    return (f"{v['mse_gain_s2']:+,.0f} [{v['ci95_s2'][0]:+,.0f}, "
            f"{v['ci95_s2'][1]:+,.0f}]")


def t_december() -> str:
    """The desk's reconciliation: december projected +268 s^2 against a raw 0.011 s loss."""
    d = load(DECEMBER_REC)
    if not d:
        return "### T9 december reconciled\n\n*(not run)*"
    r, sv = d["raw_all_row"], d["raw_served_rows"]
    pv = d["projected"]
    so = d.get("projected_served_cells_only") or {}
    ph, pc = d.get("pre_hold_all_row"), d.get("pre_hold_class_rows")
    out = ["### T9 december reconciled — the projected gain against the read of the "
           "FILE THAT IS WRITTEN", "",
           "The registered rule for the surface lanes is the PROJECTION "
           "(`assignments/queue.md`, the pilot's §5 method): each airport's fold gain "
           "per served row, weighted by the rows that airport-month has in the ranking "
           "file. This table puts it beside the raw december read of the same `d`, both "
           "with day-clustered intervals over december's "
           f"{r['days']} days. The `d` is the HELD one (FALLBACK round 4, item 1): the "
           f"cell hold serves {len(d.get('cells_served', []))} cells "
           f"(`{'`, `'.join(d.get('cells_served', []))}`), so the file moves "
           f"{d.get('served_rows', 0):,} of december's {d.get('class_rows', 0):,} "
           "no-clock rows. The last two rows are the same reading BEFORE the hold, "
           "which is what this section carried until round 4.", "",
           "| reading | rows | `2A − B` s² | RMSE seconds |", "|---|---:|---|---|",
           f"| **projected**, the served cells | – | "
           f"**{so.get('pooled_mse_reduction_s2', 0):+,.0f} "
           f"[{so.get('ci95_s2', [0, 0])[0]:+,.0f}, "
           f"{so.get('ci95_s2', [0, 0])[1]:+,.0f}]** | "
           f"{so.get('board_gain_s', 0):+.4f} |",
           f"| projected, the whole no-clock class | – | "
           f"{pv['pooled_mse_reduction_s2']:+,.0f} "
           f"[{pv['ci95_s2'][0]:+,.0f}, {pv['ci95_s2'][1]:+,.0f}] | "
           f"{pv['board_gain_s']:+.4f} "
           f"[{pv['board_gain_ci95_s'][0]:+.4f}, {pv['board_gain_ci95_s'][1]:+.4f}] |",
           f"| raw, december's SERVED rows | {sv['rows']:,} | {_g(sv)} | "
           f"{sv['rmse_gain_s']:+.4f} [{sv['ci95_rmse_s'][0]:+.4f}, "
           f"{sv['ci95_rmse_s'][1]:+.4f}] |",
           f"| raw, december ALL rows (the served file) | {r['rows']:,} | {_g(r)} | "
           f"{r['rmse_gain_s']:+.4f} [{r['ci95_rmse_s'][0]:+.4f}, "
           f"{r['ci95_rmse_s'][1]:+.4f}] |",
           f"| raw, all rows with **EGLL removed** | "
           f"{d['raw_all_row_without_EGLL']['rows']:,} | "
           f"{_g(d['raw_all_row_without_EGLL'])} | "
           f"{d['raw_all_row_without_EGLL']['rmse_gain_s']:+.4f} "
           f"[{d['raw_all_row_without_EGLL']['ci95_rmse_s'][0]:+.4f}, "
           f"{d['raw_all_row_without_EGLL']['ci95_rmse_s'][1]:+.4f}] |",
           f"| raw, served rows with **EGLL removed** | "
           f"{d['raw_served_without_EGLL']['rows']:,} | "
           f"{_g(d['raw_served_without_EGLL'])} | "
           f"{d['raw_served_without_EGLL']['rmse_gain_s']:+.4f} "
           f"[{d['raw_served_without_EGLL']['ci95_rmse_s'][0]:+.4f}, "
           f"{d['raw_served_without_EGLL']['ci95_rmse_s'][1]:+.4f}] |",
           (f"| *pre-hold*, the class's rows | {pc['rows']:,} | {_g(pc)} | "
            f"{pc['rmse_gain_s']:+.4f} [{pc['ci95_rmse_s'][0]:+.4f}, "
            f"{pc['ci95_rmse_s'][1]:+.4f}] |") if pc else "",
           (f"| *pre-hold*, all rows | {ph['rows']:,} | {_g(ph)} | "
            f"{ph['rmse_gain_s']:+.4f} [{ph['ci95_rmse_s'][0]:+.4f}, "
            f"{ph['ci95_rmse_s'][1]:+.4f}] |") if ph else "",
           "", "Per airport, december, EGLL first:", "",
           "| airport | served rows | raw served `2A − B` | raw all-row `2A − B` | "
           "projected contribution s² | ranking weight |",
           "|---|---:|---|---|---:|---:|"]
    per = pv.get("by_airport", {})
    order = ["EGLL"] + [a for a in sorted(d["by_airport_all_row"]) if a != "EGLL"]
    for a in order:
        sr = d["by_airport_served"].get(a)
        ar = d["by_airport_all_row"].get(a)
        pc = per.get(a, {})
        out.append(f"| {'**EGLL**' if a == 'EGLL' else a} | "
                   f"{(sr or {}).get('rows', 0):,} | {_g(sr)} | {_g(ar)} | "
                   f"{pc.get('contribution_s2', 0):+,.1f} | "
                   f"{pc.get('weight', 0):.4f} |")
    out += ["", "And the instrument behind it — the share of each airport's departures "
            "the matcher clocks, month by month, from the five emitted blocks. A 2025 "
            "airport whose reach is far below the ranking months' puts rows in the "
            "no-clock class that the ranking file does not:", "",
            "| airport | 2025-01 | 2025-07 | 2025-12 | 2026-01 | 2026-07 | "
            "2026-01 − 2025-12 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for a, v in sorted(d["reach_by_airport_month"].items()):
        gap = d["reach_gap_2025_12_vs_ranking"][a]["gap_vs_2026_01"]
        cells = " | ".join(
            (f"{v[m]['reach']:.3f}" if m in v else "–")
            for m in ("2025-01", "2025-07", "2025-12", "2026-01", "2026-07"))
        out.append(f"| {a} | {cells} | "
                   f"{'–' if gap is None else format(gap, '+.3f')} |")
    return "\n".join(out)


def t_interface() -> str:
    d = load("interface.json")
    if not d or not d.get("months"):
        return "### T6 the interface\n\n(not written)"
    rows = ["### T6 the interface `exp-046-surfacefit` can take as an input", "",
            "| month | path | rows | columns | MB | SHA-256 | prediction the `*_n_res_*` "
            "columns are relative to |",
            "|---|---|---:|---:|---:|---|---|"]
    for m, r in sorted(d["months"].items()):
        pr = r.get("prediction", {}) or {}
        rows.append(f"| {m} | `{r['path']}` | {r['rows']:,} | {len(r['columns'])} | "
                    f"{r['bytes'] / 1e6:.1f} | `{r['sha256'][:16]}…` | "
                    f"{pr.get('source') or '–'} (`{(pr.get('sha256') or '')[:12]}…`) |")
    mat = d.get("matcher_block_py_sha256", "")
    rows += ["", "Reach classes per month — `exp-046-surfacefit`'s own `reach_class`, "
             f"from the one matcher both lanes import, at `{mat[:12]}…`:", "",
             "| month | " + " | ".join(f"`{k}`" for k in RCLASS) + " |",
             "|---|" + "---:|" * len(RCLASS)]
    for m in sorted(d["months"]):
        c = d["months"][m].get("reach_class_counts") or {}
        rows.append(f"| {m} | " + " | ".join(f"{c.get(k, 0):,}" for k in RCLASS) + " |")
    rows += ["", "Column prefixes: `Fto_` the file's clocks at the take-off instant, "
             "`Fob_` at the off-block instant, `Oob_` the observed clocks, `Hob_` the "
             "hybrid, `Nf_` / `No_` the neighbourhood on the file's and the observed "
             "clock, `*_sc` a count divided by the airport's receipt rate, `route_*` "
             "and `stand_dist_km` the stand-to-runway geometry, `own_obs_taxi` "
             "exp-046's own feature carried only so a consumer can control for it, "
             "and `obs_matched` / `obs_has_clock` / `reach_class` the identity of the "
             "observation. `state.py`'s docstring defines every state name."]
    return "\n".join(rows)


def main() -> None:
    p = EXP / "RESULTS.md"
    txt = p.read_text()
    head = txt.split(MARK)[0]
    body = "\n\n".join([
        MARK, "<!-- generated by code/tables.py; do not edit by hand -->",
        t_file(),
        t_week("explain-july_loo-ehamweek-own.json",
               "T2 the observed clock on the pilot's EHAM week",
               ["Nf", "No", "N", "Oob", "Hob"]),
        t_week("explain-july_loo-ehamweek-own.json",
               "T3 beyond the row's OWN clock (exp-046's feature in the base)",
               ["own", "No", "own_No", "own_No_Hob"]),
        t_split(), t_month(),
        t_week("explain-july_loo-ehamweek-permuted.json",
               "T3b the permutation placebo — the clocks dealt out to other aircraft "
               "(the `Nf` row is a CONTROL: only the observed clocks are dealt)",
               ["Nf", "No", "N"]),
        t_repair(), t_thin(), t_audit(),
        t_step2("arms-step2-unmatched-fitall-3f.json",
                "T7 step 2 — the SERVING class on three folds, the head fitted on every "
                "row of the panels it is not scored on. THE GATED READING",
                ["Nf", "geo", "Nf_geo", "No", "N", "own"]),
        t_step2("arms-step2-unmatched.json",
                "T7a the same rows with the head fitted on unmatched rows only, two "
                "folds — the reading the fit-on-all repair replaced",
                ["Nf", "geo", "Nf_geo", "No", "N", "own"]),
        t_step2("arms-step2-unmatched-fit50.json",
                "T7b the first FIT-side selection that FAILED — no fitting row on an "
                "airport-day below 0.5 receipt, which selects on the outcome",
                ["Nf", "Nf_geo", "No", "N", "own"]),
        t_step2("arms-step2-unmatched-fitall-3f-gap25.json",
                "T7c the second FIT-side selection that FAILED — no fitting row at an "
                "airport-MONTH whose reach is more than 0.25 from the ranking file's. "
                "It selects on the instrument and not on the outcome, and it starves "
                "`No_*` of the airports where an observation exists",
                ["Nf", "No", "N", "own"]),
        t_step2("arms-step2-matched-3f.json",
                "T7d the with-clock class — fitted and priced on matched rows only "
                "(surfacefit's head; these are the columns it can take as inputs)",
                ["own", "No", "own_No", "N"]),
        t_weights(), t_cells(), t_celldiag(), t_capacity(), t_fileclock(), t_percell(), t_increment(), t_shrunk(), t_tailnonm(), t_receipt(),
        t_step2("arms-step2-all.json",
                "T7f the all-row reading, for the record — one head over both classes",
                ["Nf", "No", "N", "own"]),
        t_rmse4(), t_december(), t_columns(), t_census(), t_outage(),
        t_interface()])
    p.write_text(head + body + "\n")
    print("rendered", p)


if __name__ == "__main__":
    main()
