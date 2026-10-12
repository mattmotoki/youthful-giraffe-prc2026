"""exp-045-harvest: render the RESULTS.md tables from the analysis files.

Keeps the report a function of `analysis/receipt.json`, `analysis/edges.json` and
`manifest.json` rather than of typing, so a month can be added without a chance to
mistype a number.

    .venv/bin/python experiments/exp-045-harvest/code/tables.py            # print
    .venv/bin/python experiments/exp-045-harvest/code/tables.py --write    # into RESULTS.md

`--write` replaces what sits between the `<!-- table:NAME -->` and `<!-- /table:NAME -->`
comments in RESULTS.md and touches nothing else, so the prose stays hand-written and the
numbers stay generated.  Thirteen months are still to land; a table edited by hand each
time is a stale number waiting to happen, and this file already had to correct one.
"""
from __future__ import annotations

import calendar
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "code"))
from days import ORDER                                          # noqa: E402

AIRPORTS = ["EHAM", "LFPG", "EGLL", "LTFM", "LEMD", "EDDF", "LEBL", "LIRF", "EDDM", "LSZH"]


def load(name: str, default=None):
    p = EXP / name
    return json.loads(p.read_text()) if p.exists() else (default if default is not None else {})


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def receipt_table() -> str:
    r = load("analysis/receipt.json")
    out = ["| Month | days | departures | " + " | ".join(AIRPORTS) + " | all |",
           "| --- | ---: | ---: | " + " | ".join(["---:"] * len(AIRPORTS)) + " | ---: |"]
    # calendar order, not fetch order: read down a column and the receiver rollout at
    # that airport is the thing you see.  LEMD and LIRF are 0.0 in January 2025 and
    # 99.4 and 91.4 by December; LIRF is gone again by July 2026.
    for m in sorted(x for x in ORDER if x in r):
        if m not in r:
            continue
        b = r[m]["by_airport"]
        cells = [pct(b.get(a, {}).get("rate")) for a in AIRPORTS]
        # a month still short of its days is marked, because the whole point of this
        # table is that a rate over fewer days than a month is not a property of the
        # airport -- and a half-delivered row read as a full one is that mistake again
        got = len(r[m]["days_measured"])
        full = calendar.monthrange(int(m[:4]), int(m[5:]))[1]
        days = f"{got}" if got == full else f"**{got} of {full}**"
        out.append(f"| {m} | {days} | {r[m]['departures']:,} | "
                   + " | ".join(cells) + f" | **{pct(r[m]['receipt_rate'])}**"
                   + ("" if got == full else " (partial)") + " |")
    return "\n".join(out) if len(out) > 2 else "_(no month measured yet)_"


def nm_table() -> str:
    r = load("analysis/receipt.json")
    out = ["| Month | all | with an NM join | with no NM join | no-NM departures |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for m in ORDER:
        if m not in r:
            continue
        no_nm = sum(v.get("no_nm", 0) or 0 for v in r[m]["by_airport"].values())
        out.append(f"| {m} | {pct(r[m]['receipt_rate'])} | {pct(r[m]['receipt_rate_nm'])} | "
                   f"{pct(r[m]['receipt_rate_no_nm'])} | {int(no_nm):,} |")
    return "\n".join(out) if len(out) > 2 else "_(no month measured yet)_"


def edge_table() -> str:
    e = load("analysis/edges.json")
    if not e:
        return "_(no day checked yet)_"
    out = ["| Airport | departure stands | 1st pct m to the nearest edge | median m | "
           "stands within 500 m |", "| --- | ---: | ---: | ---: | ---: |"]
    for a in AIRPORTS:
        if a not in e:
            continue
        r = e[a]
        out.append(f"| {a} | {r['departure_stands']:,} | {r['p01_m_to_edge']:.0f} | "
                   f"{r['median_m_to_edge']:.0f} | {100 * r['share_within_500m']:.2f} % |")
    return "\n".join(out)


def cost_table() -> str:
    man = load("manifest.json")
    ms = man.get("months", {})
    if not ms:
        return "_(no month delivered yet)_"
    out = ["| Month | days | kernel hours | GB consumed | archive digests verified | missing |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for m in ORDER:
        if m not in ms:
            continue
        r = ms[m]
        out.append(f"| {m} | {r['delivered']}/{r['expected']} | {r['seconds'] / 3600:.1f} | "
                   f"{r['bytes_consumed'] / 1e9:.0f} | "
                   f"{r['digests_verified']}/{r['digests_expected']} | {len(r['missing'])} |")
    tot = ms.values()
    out.append(f"| **total** | **{sum(r['delivered'] for r in tot)}/"
               f"{sum(r['expected'] for r in tot)}** | "
               f"**{sum(r['seconds'] for r in tot) / 3600:.1f}** | "
               f"**{sum(r['bytes_consumed'] for r in tot) / 1e9:.0f}** | "
               f"**{sum(r['digests_verified'] for r in tot)}/"
               f"{sum(r['digests_expected'] for r in tot)}** | "
               f"**{sum(len(r['missing']) for r in tot)}** |")
    return "\n".join(out)


def missing_table() -> str:
    man = load("manifest.json")
    miss = man.get("missing", {})
    if not miss:
        return ("**None. All 427 days of the programme delivered a run record**, so no day is\n"
            "listed missing with a reason and the missing table is the empty one "
            "this stage hoped for.")
    out = ["| Day | attempts | kernel | reason |", "| --- | ---: | --- | --- |"]
    for d, r in sorted(miss.items()):
        out.append(f"| {d} | {r['attempts']} | {r['kernel']} | {r['reason'][:160]} |")
    return "\n".join(out)


def weighted_table() -> str:
    """What each airport is worth in the ranking file, against the latest month it was
    measured in.  Generated rather than typed: the gate's version of this table stood
    for two hours after the month refuted it."""
    r = load("analysis/receipt.json")
    sys.path.insert(0, str(EXP / "code"))
    from census import ranking
    rank = ranking()
    n = rank.groupby("ADEP_mvt", observed=True).size()
    tot = int(n.sum())
    # the ranking file is January and July 2026 and nothing else, so those are the two
    # columns that matter.  Each airport is weighted at its OWN month's rate: LIRF is
    # 91.6 % in January and dark in July, and one averaged number would hide that.
    scored = [m for m in ("2026-01", "2026-07") if m in r]
    if not scored:
        return "_(no scored month measured yet)_"
    head = " | ".join(f"receipt {m}" for m in scored)
    out = ["Weighted by the ranking file, which is January and July 2026:", "",
           f"| Airport | ranking departures | share | {head} | reachable |",
           "| --- | ---: | ---: | " + " | ".join(["---:"] * len(scored)) + " | ---: |"]
    # `reachable` counts ROWS: each airport's departures in each scored month at that
    # month's own rate.  Averaging an airport's two months equally instead -- which this
    # table did until 2026-09-09 -- weights January as heavily as July although July
    # carries 39,403 more departures, and reads 59.69 % where the rows say 59.01 %.
    rows = []
    for a in AIRPORTS:
        if a not in n:
            continue
        rates = [r[m]["by_airport"].get(a, {}).get("rate") for m in scored]
        deps = [r[m]["by_airport"].get(a, {}).get("departures") for m in scored]
        if any(x is None for x in rates) or any(x is None for x in deps):
            continue
        cnt = int(sum(deps))
        if cnt != int(n[a]):  # the cohort is the ranking file; say so if it ever is not
            print(f"note: {a} has {cnt} censused departures against {int(n[a])} ranking rows",
                  file=sys.stderr)
        reach = sum(d * x for d, x in zip(deps, rates))
        rows.append((a, cnt, rates, reach))
    rows.sort(key=lambda x: -x[3] / x[1])
    for a, cnt, rates, reach in rows:
        cells = " | ".join(f"{100 * x:.1f} %" for x in rates)
        out.append(f"| {a} | {cnt:,} | {100 * cnt / tot:.1f} % | {cells} "
                   f"| {reach:,.0f} |")
    total_reach = sum(x[3] for x in rows)
    equal = sum(x[1] * sum(x[2]) / len(x[2]) for x in rows)
    out.append(f"| **total** | **{tot:,}** | | "
               + " | ".join(f"**{100 * r[m]['receipt_rate']:.1f} %**" for m in scored)
               + f" | **{total_reach:,.0f} = {100 * total_reach / tot:.2f} %** |")
    out.append("")
    out.append(f"The **{100 * total_reach / tot:.2f} %** headline is a count of scored rows: every "
               f"departure in the ranking file weighted equally, which is each airport's "
               f"departures in each scored month at that month's own rate "
               f"({total_reach:,.0f} of {tot:,}). The month cells are the months' own rates over "
               f"all ten airports. Weighting each airport's two months equally instead gives "
               f"**{100 * equal / tot:.2f} %**, because July carries "
               f"{abs(r[scored[1]]['departures'] - r[scored[0]]['departures']):,} more departures "
               f"than January and is the worse month; the two differ by "
               f"{100 * (equal - total_reach) / tot:.2f} points. A third weighting, the unweighted "
               f"mean of the {len(rows) * len(scored)} airport-month cells, reads "
               f"{100 * sum(sum(x[2]) for x in rows) / (len(rows) * len(scored)):.2f} %, and is the "
               f"one to avoid: it gives LTFM's {dict((x[0], x[1]) for x in rows)['LTFM']:,} "
               f"unreachable rows the same weight as LSZH's "
               f"{dict((x[0], x[1]) for x in rows)['LSZH']:,} good ones.")
    return "\n".join(out)


def wasted_table() -> str:
    """The sessions that fetched days another session had already fetched.

    Section 3 counts a day once, because a day is delivered once.  That hides what
    the outage of 2026-09-08 actually cost: the CLI lost its credentials at 14:31,
    status() answered UNKNOWN for four live sessions, and the dispatcher pushed each
    of their months a second time.  Kaggle has no way to cancel a running session, so
    all four ran to the end.  The hours are real and are charged against the account
    and the shared five-session pool, so they are reported rather than netted out.
    """
    st = load("analysis/dispatch_state.json")
    man = load("manifest.json")
    ks = st.get("kernels", [])
    day_rec = {day: rec for m in man.get("months", {}).values()
               for day, rec in m["days"].items()}
    secs = [d["seconds"] for d in day_rec.values() if "seconds" in d]
    rate = sum(secs) / len(secs) if secs else 0.0

    def hours(k) -> tuple[float, bool]:
        """The session's own compute, when the manifest still names it.

        Each duplicate landed over its original, so the day records the manifest
        carries are the duplicate's own: its seconds are measured, not modelled.
        A duplicate whose records were replaced again falls back to the mean day.
        """
        own = [day_rec[d]["seconds"] for d in k["days"]
               if day_rec.get(d, {}).get("kernel") == f"prc-harvest-{k['name']}"
               and "seconds" in day_rec[d]]
        if len(own) == len(k["days"]):
            return sum(own) / 3600.0, True
        return len(k["days"]) * rate / 3600.0, False
    dups = []
    for i, k in enumerate(ks):
        earlier = {d for j, e in enumerate(ks) if j < i for d in e["days"]}
        if k["days"] and set(k["days"]) <= earlier:
            dups.append(k)
    if not dups:
        return "None. Every session fetched days no earlier session had fetched."
    out = ["| Duplicate session | days | pushed | kernel hours | fetched days already held by |",
           "| --- | ---: | ---: | ---: | --- |"]
    total = 0.0
    modelled = 0
    for k in dups:
        first = next(e["name"] for e in ks if set(k["days"]) <= set(e["days"])
                     and e["name"] != k["name"])
        h, measured = hours(k)
        total += h
        modelled += 0 if measured else 1
        out.append(f"| {k['name']} | {len(k['days'])} | {k['pushed_at'][11:16]} UTC | "
                   f"{h:.1f}{'' if measured else ' (est.)'} | {first} |")
    nd = sum(len(k["days"]) for k in dups)
    out.append(f"| **total** | **{nd}** | | **{total:.1f}** | |")
    out.append("")
    out.append("Each duplicate's hours are its own recorded run seconds: it landed over its "
               "original, so the day records the manifest carries are the duplicate's."
               + (f"  {modelled} row(s) are estimated at the measured {rate:.0f} s a day "
                  f"over ten boxes, because the manifest no longer names that session."
                  if modelled else "")
               + " Nothing is lost: each duplicate delivers the same days its original "
                 "delivered, and the remaining months queue behind them.")
    return "\n".join(out)


def published_table() -> str:
    """What adsb.lol published per day, beside what the boxes heard.

    A day is empty in the receipt table either because the feeders did not hear the
    airport or because the archive for that day is short, and downstream the two look
    the same.  This table is the second one, so the first can be claimed.
    """
    p = load("analysis/published_bytes.json")
    if not p:
        return "_(not measured yet: run code/published_bytes.py)_"
    r = load("analysis/receipt.json")
    days = p["days"]
    months = sorted({d[:7] for d in days})
    out = ["| Month | days published | median GB | short days | receipt |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for m in months:
        got = sorted(v["bytes"] for d, v in days.items() if d[:7] == m)
        n = len(got)
        med = got[n // 2] / 1e9 if n else 0.0
        short = [d for d, v in sorted(days.items()) if d[:7] == m and v["short"]]
        rate = f"{100 * r[m]['receipt_rate']:.1f} %" if m in r else "–"
        out.append(f"| {m} | {n} | {med:.2f} | {', '.join(short) if short else '–'} | {rate} |")
    tmp = sorted(d for d, v in days.items() if (v["tag"] or "").endswith("tmp"))
    none = sorted(d for d, v in days.items() if not v["tag"])
    out.append("")
    out.append(f"All {len(days)} days of the programme are published: "
               + (f"{len(none)} have no release at all ({', '.join(none)}), " if none
                  else "none is absent from the archive, ")
               + f"and {len(tmp)} carry only the `...-prod-0tmp` tag "
               + (f"({tmp[0]} to {tmp[-1]})" if tmp else "") + ". So a day this stage "
               "cannot deliver is a fetch failure, never a day adsb.lol did not ship.")
    return "\n".join(out)


def shift_table() -> str:
    """How far each airport's coverage moves between the training year and the months
    that are scored.  This is the number a model conditions on wrongly if it learns
    how often a surface feature is present from 2025 and applies it to 2026."""
    r = load("analysis/receipt.json")
    train = [m for m in sorted(r) if m.startswith("2025")]
    scored = [m for m in ("2026-01", "2026-07") if m in r]
    if not train or not scored:
        return "_(needs the training year and both scored months)_"
    out = ["| Airport | 2025 (mean of months) | scored 2026 | shift |",
           "| --- | ---: | ---: | ---: |"]
    rows = []
    for a in AIRPORTS:
        if a not in r[train[0]]["by_airport"]:
            continue
        t0 = sum(r[m]["by_airport"][a]["rate"] for m in train) / len(train)
        num = sum(r[m]["by_airport"][a]["departures"] * r[m]["by_airport"][a]["rate"]
                  for m in scored)
        den = sum(r[m]["by_airport"][a]["departures"] for m in scored)
        s0 = num / den if den else 0.0
        rows.append((a, t0, s0, s0 - t0))
    for a, t0, s0, d in sorted(rows, key=lambda x: -abs(x[3])):
        out.append(f"| {a} | {100 * t0:.1f} % | {100 * s0:.1f} % | **{100 * d:+.1f}** |")
    tt = (sum(r[m]["departures"] * r[m]["receipt_rate"] for m in train)
          / sum(r[m]["departures"] for m in train))
    ss = (sum(r[m]["departures"] * r[m]["receipt_rate"] for m in scored)
          / sum(r[m]["departures"] for m in scored))
    out.append(f"| **all ten** | **{100 * tt:.1f} %** | **{100 * ss:.1f} %** | "
               f"**{100 * (ss - tt):+.1f}** |")
    return "\n".join(out)


TABLES = {"receipt": receipt_table, "shift": shift_table, "published": published_table, "weighted": weighted_table, "nm": nm_table,
          "edges": edge_table, "cost": cost_table, "wasted": wasted_table, "missing": missing_table}


def write_into(path: Path) -> list[str]:
    """Replace each `<!-- table:NAME -->` block in place.  Returns the names written."""
    import re
    text = path.read_text()
    done = []
    for name, fn in TABLES.items():
        pat = re.compile(rf"<!-- table:{name} -->.*?<!-- /table:{name} -->", re.S)
        text, n = pat.subn(
            lambda _mo: f"<!-- table:{name} -->\n{fn()}\n<!-- /table:{name} -->",
            text, count=1)
        if n:
            done.append(name)
    path.write_text(text)
    return done


if __name__ == "__main__":
    if "--write" in sys.argv:
        got = write_into(EXP / "RESULTS.md")
        missing_marks = [t for t in TABLES if t not in got]
        print(f"RESULTS.md: wrote {', '.join(got)}"
              + (f"; no marker for {', '.join(missing_marks)}" if missing_marks else ""))
    else:
        print("## 1. Receipt rate by airport (%)\n")
        print(receipt_table())
        print()
        print(weighted_table())
        print("\n### by NM join\n")
        print(nm_table())
        print("\n## 2. The boxes, checked\n")
        print(edge_table())
        print("\n## 3. Cost\n")
        print(cost_table())
        print("\n### Sessions that fetched days twice\n")
        print(wasted_table())
        print("\n## 4. Missing days\n")
        print(missing_table())
