"""exp-046-surfacefit: WHICH of receipt, box or clock fails at an airport.

The assignment's repair clause: "if an airport's observation does not carry (receipt,
box, or clock offset), say which and exclude it by a rule fitted off the scored rows".
The exclusion is already structural -- every arm carries one weight per airport fitted on
the fitting months, so an airport that does not carry gets weight 0 (`code/arms.py`).
This file supplies the "say which", and it does so **without a label**, so it runs on a
ranking month as readily as on a training month and can be re-run as each month lands.

The harvest's first two days put receipt at 97-98 % (EDDM, LEBL, LEMD), 91-95 % (EHAM,
LSZH) and 69 / 68 / 50 / 16 / 0 % (EDDF, LIRF, EGLL, LFPG, LTFM).  A low number there can
mean three different things and they need different repairs:

    receipt   the network does not hear the aircraft on that apron.  No repair inside
              this programme; the airport stays at v11.
    box       the network hears it, but the harvest's box does not contain the stands,
              so the segment is truncated or missing.  Repairable: redraw the box and
              re-run that month's kernel.
    clock     the aircraft is seen and matched, but the observed take-off does not sit
              where it should relative to `MVT_TIME_UTC_mvt`, so the identity is wrong
              or the segment is not the departure.  Repairable in the matcher.

Four label-free readings per airport:

    seen        the share of departures whose callsign appears on a ground segment
    matched     the share the block matches (`code/block.py`)
    clock       median and spread of `obs_wheels_off - MVT_TIME_UTC_mvt` on matched rows;
                the pilot's EHAM reference is a median near -26 s with 94 % inside 60 s
    edge        how close the matched stands sit to the box edge, in km, and the share
                within 1 km of one -- a box that clips an apron shows here

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/reachwhy.py [month ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
import block as B                                            # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

EDGE_KM = 1.0
DIP_PTS = 0.20          # how far below its own month median a day has to fall to be named
CLOCK_REF = {"median_s": -26.0, "share_within_60s": 0.94, "source": "exp-043 EHAM week"}


def verdict(seen: float | None, matched: float, clock_ok: float | None,
            edge_share: float | None) -> str:
    """Which of receipt, box or matcher costs this airport the most.

    The first version answered "clock" whenever the clock was poor and enough rows
    matched, which named the QUALITY of the matched rows and not what limits REACH.  The
    two losses are now compared directly: `1 - seen` is what the network never heard, and
    `seen - matched` is what the matcher then failed to identify.  Whichever is larger is
    the answer, and the clock only ever qualifies the rows that did match.
    """
    if matched >= 0.80 and (clock_ok or 0) >= 0.80:
        return "carries"
    if seen is None:
        return ("partial: carries on some rows only" if matched >= 0.40
                else "identity: few rows match and there is no receipt figure to compare")
    lost_receipt, lost_matcher = 1.0 - seen, max(seen - matched, 0.0)
    if edge_share is not None and edge_share >= 0.20:
        return "box: matched stands crowd the box edge, so the box likely clips an apron"
    if lost_receipt > lost_matcher:
        return (f"receipt: {lost_receipt:.0%} of departures are never heard on the "
                f"ground, against {lost_matcher:.0%} the matcher then loses")
    if clock_ok is not None and clock_ok < 0.80:
        return (f"identity: the matcher loses {lost_matcher:.0%} against {lost_receipt:.0%} "
                "unheard, and the take-off clock on the rows it keeps is poor")
    return (f"identity: the matcher loses {lost_matcher:.0%} against {lost_receipt:.0%} "
            "unheard")


def receipt_dips(receipt: dict, month: str) -> dict:
    """Airport-days that BREAK AWAY from their airport's own month median receipt.

    The distinction the hold rests on (desk, 2026-09-08 14:45): a low STEADY rate is
    reception and its rows are kept, a break in a high series is an outage and its rows
    are held.  A threshold cannot tell those apart on its own -- LFPG sits at 22 % every
    day and is not an outage -- so this NAMES the candidates and holds nothing.  What is
    held is `block.HELD_AIRPORT_DAYS`, which carries the desk's line.
    """
    day = receipt.get(month, {}).get("by_airport_day", {})
    out = {"drop_points_below_the_airport_median": DIP_PTS, "by_airport": {}}
    for a, v in sorted(day.items()):
        if not v:
            continue
        med = float(np.median(list(v.values())))
        dips = {d: round(float(r), 4) for d, r in sorted(v.items()) if r < med - DIP_PTS}
        out["by_airport"][a] = {
            "month_median": round(med, 4),
            "min_day": round(float(min(v.values())), 4),
            "dips": dips,
            "dips_held": {d: r for d, r in dips.items() if (a, d) in B.HELD_AIRPORT_DAYS},
            "dips_not_held": {d: r for d, r in dips.items()
                              if (a, d) not in B.HELD_AIRPORT_DAYS},
            "receipt_month_rate": receipt.get(month, {}).get("by_airport", {})
                                         .get(a, {}).get("rate"),
            "receipt_rate_on_kept_days": round(float(np.mean(
                [r for d, r in v.items() if (a, d) not in B.HELD_AIRPORT_DAYS])), 4)
            if any((a, d) not in B.HELD_AIRPORT_DAYS for d in v) else None,
            "days_kept": int(sum((a, d) not in B.HELD_AIRPORT_DAYS for d in v)),
            "days_held": int(sum((a, d) in B.HELD_AIRPORT_DAYS for d in v))}
    return out


def month_rows(month: str) -> pd.DataFrame:
    """Departure rows of one month with the block, from inputs only."""
    src = (D.load_ranking() if month.startswith("2026")
           else pd.read_parquet(D.month_file(month)))
    dep = src[src.PHASE_mvt == "DEP"].copy()
    dep["month"] = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    dep = dep[dep.month == month]
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in D.MONTHS if m != month], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
    # one rule version for the month being read, resolved and required
    rule, rule_report = B.one_rule(
        [(sorted(set(dep.ADEP_mvt.dropna()) & set(B.AIRPORTS)),
          sorted(dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").unique()))])
    blk = B.build(dep, tr, require_rule=rule)
    out = dep[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]].merge(
        blk, on="MVT_ID_mvt", how="left", validate="one_to_one")
    out["obs_matched"] = out.obs_matched.astype("boolean").fillna(False).astype(bool)
    out["day"] = out.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    out.attrs["days_present"] = blk.attrs.get("days_present", 0)
    out.attrs["days_present_list"] = blk.attrs.get("days_present_list", [])
    out.attrs["days_missing"] = blk.attrs.get("days_missing", [])
    out.attrs["days_held"] = blk.attrs.get("days_held", [])
    out.attrs["days_held_reason"] = blk.attrs.get("days_held_reason")
    out.attrs["rules_seen"] = blk.attrs.get("rules_seen", [])
    out.attrs["rule_cover"] = rule_report
    out.attrs["normalised_by_airborne_alt_ok"] = blk.attrs.get(
        "normalised_by_airborne_alt_ok")
    return out


def edges_km(g: pd.DataFrame, box: dict) -> tuple[float, float]:
    """Distance of each matched stand to the nearest box edge, in km."""
    la, lo = g.stand_lat.to_numpy("float64"), g.stand_lon.to_numpy("float64")
    ok = np.isfinite(la) & np.isfinite(lo)
    if not ok.any():
        return (np.nan, np.nan)
    la, lo = la[ok], lo[ok]
    kmlat = 110.540
    kmlon = 111.320 * np.cos(np.radians(np.nanmean(la)))
    d = np.minimum.reduce([(la - box["lat"][0]) * kmlat, (box["lat"][1] - la) * kmlat,
                           (lo - box["lon"][0]) * kmlon, (box["lon"][1] - lo) * kmlon])
    return float(np.median(d)), float(np.mean(d <= EDGE_KM))


def main(months: list[str]) -> None:
    boxes = json.loads((ROOT / "experiments/exp-045-harvest/analysis/boxes.json").read_text())
    boxes = boxes.get("boxes", boxes)
    receipt_file = ROOT / "experiments/exp-045-harvest/analysis/receipt.json"
    receipt = json.loads(receipt_file.read_text()) if receipt_file.exists() else {}

    rep = {"clock_reference": CLOCK_REF, "edge_km_threshold": EDGE_KM, "months": {},
           "held_airport_days": {f"{a} {d}": r for (a, d), r in
                                 B.HELD_AIRPORT_DAYS.items()},
           "held_reason": B.HELD_REASON}
    for m in months:
        d = month_rows(m)
        days = sorted({x for _, x in d.attrs["days_missing"]})
        rec = {"rows": int(len(d)),
               "summary_days_present": d.attrs["days_present"],
               "summary_days_missing": len(days),
               "rules_seen": d.attrs["rules_seen"],
               "rule_cover": d.attrs["rule_cover"],
               "normalised_by_airborne_alt_ok":
                   d.attrs["normalised_by_airborne_alt_ok"],
               # the held days are OUT of every share below: their rows carry no
               # observation and their segments feed no neighbour, so an airport's
               # reach here is its reach on the days the feeder was up
               "days_held": [f"{a} {dy} receipt {r:.4f}"
                             for a, dy, r in d.attrs.get("days_held", [])],
               "days_held_reason": d.attrs.get("days_held_reason"),
               "receipt_dips": receipt_dips(receipt, m),
               "by_airport": {}}
        seen_m = receipt.get(m, {}).get("by_airport", {})
        dips = rec["receipt_dips"]["by_airport"]
        # a share is only meaningful on the days whose summaries exist: the harvest
        # delivers a month day by day, so `matched` over the whole month would read the
        # harvest's progress rather than the airport's reach
        have = {}
        for a, dy in d.attrs["days_present_list"]:
            have.setdefault(a, set()).add(dy)
        rec["days_with_summaries_by_airport"] = {a: len(v) for a, v in have.items()}
        for a, g_all in d.groupby("ADEP_mvt"):
            g = g_all[g_all.day.isin(have.get(a, set()))]
            if not len(g):
                rec["by_airport"][a] = {"departures_in_month": int(len(g_all)),
                                        "departures_on_covered_days": 0,
                                        "verdict": "no summary day for this airport yet"}
                continue
            mt = g.obs_matched
            strict = g.match_mode.eq(0.0).fillna(False)
            gm = g[strict]                      # the clock is judged on strict rows
            clock_med = float(gm.dt_takeoff.median()) if len(gm) else None
            clock_ok = float((gm.dt_takeoff.abs() <= 60).mean()) if len(gm) else None
            gw = g[mt]
            med_edge, near_edge = edges_km(gw, boxes[a]) if len(gw) and a in boxes \
                else (None, None)
            # THE RECEIPT SCALING (desk, 2026-09-08 14:45): an airport with a held
            # day is read on the days its feeder was up.  The harvest's own month rate
            # is departures-weighted and the kept-day figure is a mean over days, so
            # both are recorded and the substitution is only made where it matters.
            seen = seen_m.get(a, {}).get("rate")
            seen_month = seen
            seen_src = "harvest month rate, departures-weighted"
            dv = dips.get(a, {})
            if dv.get("days_held") and dv.get("receipt_rate_on_kept_days") is not None:
                seen = dv["receipt_rate_on_kept_days"]
                seen_src = (f"mean over the {dv['days_kept']} days the feeder was up "
                            f"({dv['days_held']} held); month rate {seen_month}")
            rec["by_airport"][a] = {
                "departures_in_month": int(len(g_all)),
                "departures_on_covered_days": int(len(g)),
                "covered_days": len(have.get(a, set())),
                "seen_on_the_ground": seen,
                "seen_on_the_ground_month": seen_month,
                "seen_source": seen_src,
                "matched": round(float(strict.mean()), 4),
                "matched_wide": round(float(mt.mean()), 4),
                "wide_window_rows": int(g.match_mode.eq(1.0).sum()),
                "no_takeoff_rows": int(g.match_mode.eq(2.0).sum()),
                "usable_wide": round(float((mt & g.obs_has_clock.astype("boolean")
                                            .fillna(False)).mean()), 4),
                "with_clock": round(float(g.obs_has_clock.astype("boolean")
                                          .fillna(False).mean()), 4),
                "clock_median_s": clock_med,
                "clock_share_within_60s": clock_ok,
                "stand_to_box_edge_km_median": med_edge,
                "stand_within_1km_of_edge_share": near_edge,
                "verdict": verdict(seen, float(strict.mean()), clock_ok, near_edge),
                "verdict_wide": verdict(seen, float(mt.mean()), 1.0, near_edge),
            }
        rep["months"][m] = rec
        print(f"\n{m}: {rec['rows']} departures, {rec['summary_days_present']} "
              f"airport-days of summaries")
        for a, v in sorted(rec["by_airport"].items(),
                           key=lambda kv: -kv[1].get("matched", -1)):
            if "matched" not in v:
                print("  {}  {}".format(a, v["verdict"]))
                continue
            cm = v["clock_median_s"]
            cw = v["clock_share_within_60s"]
            eg = v["stand_to_box_edge_km_median"]
            print("  {}  strict {:6.1%}  wide {:6.1%}  clock {}  within60 {}  edge {}"
                  "  -> {}".format(
                a, v["matched"], v["matched_wide"],
                "     -" if cm is None else "{:+6.0f} s".format(cm),
                "    -" if cw is None else "{:5.1%}".format(cw),
                "     -" if eg is None else "{:4.1f} km".format(eg),
                v["verdict"]))

    (EXP / "analysis/reachwhy.json").write_text(json.dumps(rep, indent=2, sort_keys=True,
                                                            default=float))


if __name__ == "__main__":
    main(sys.argv[1:] or ["2026-01"])
